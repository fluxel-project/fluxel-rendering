//! Indexed, graph-derived recording access coverage.
//!
//! This remains a private graph-derived authority.  It precomputes declaration
//! coverage only; it does not change binding, provenance, or graph ownership.

use fluxel_rhi::api::TextureAspects;
use std::ops::Range;

use super::{
    binding::{BufferSlot, TextureSlot},
    definition::{
        AccessKind, BufferAccessContract, GraphBuilder, GraphPass, TextureAccessContract,
        TextureRange,
    },
};

/// Failure to derive or exercise a pass-local access authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AccessError {
    UnknownPass,
    DuplicatePass,
    MissingBufferContract,
    MissingTextureContract,
    UndeclaredBuffer,
    UndeclaredTexture,
    IncompatibleBufferContract,
    IncompatibleTextureContract,
    BufferRangeOutsideDeclaration,
    TextureRangeOutsideDeclaration,
}

/// Access authorities for one compiled graph schedule.
#[derive(Clone, Debug)]
pub(crate) struct AccessPlan {
    passes: Vec<(GraphPass, PassAccess)>,
}

impl AccessPlan {
    /// Compiles the declared resource operations for the selected live order.
    pub(crate) fn compile(graph: &GraphBuilder, order: &[GraphPass]) -> Result<Self, AccessError> {
        let mut passes = Vec::with_capacity(order.len());
        for &pass in order {
            let node = graph
                .passes
                .get(pass.1)
                .filter(|node| node.id == pass)
                .ok_or(AccessError::UnknownPass)?;
            if passes.iter().any(|(known, _)| *known == pass) {
                return Err(AccessError::DuplicatePass);
            }

            let mut authority = PassAccess {
                buffers: Vec::with_capacity(node.accesses.len()),
                textures: Vec::with_capacity(node.texture_accesses.len()),
            };
            for access in &node.accesses {
                authority.add_buffer(
                    BufferSlot::from(access.buffer),
                    access.range.bytes(),
                    access.kind,
                    access.contract.ok_or(AccessError::MissingBufferContract)?,
                );
            }
            for access in &node.texture_accesses {
                authority.add_texture(
                    TextureSlot::from(access.texture),
                    access.range.clone(),
                    access.kind,
                    access.contract.ok_or(AccessError::MissingTextureContract)?,
                );
            }
            authority.normalize_buffer_ranges();
            passes.push((pass, authority));
        }
        Ok(Self { passes })
    }

    /// Returns the authority belonging to exactly this scheduled pass.
    pub(crate) fn pass(&self, pass: GraphPass) -> Result<&PassAccess, AccessError> {
        self.passes
            .iter()
            .find_map(|(known, authority)| (*known == pass).then_some(authority))
            .ok_or(AccessError::UnknownPass)
    }
}

/// The only logical resource operations a particular pass may record.
#[derive(Clone, Debug)]
pub(crate) struct PassAccess {
    buffers: Vec<BufferSlotDeclarations>,
    textures: Vec<TextureSlotDeclarations>,
}

impl PassAccess {
    fn add_buffer(
        &mut self,
        slot: BufferSlot,
        range: Range<u64>,
        kind: AccessKind,
        contract: BufferAccessContract,
    ) {
        let declarations = match self.buffers.iter().position(|known| known.slot == slot) {
            Some(index) => &mut self.buffers[index],
            None => {
                self.buffers.push(BufferSlotDeclarations {
                    slot,
                    groups: Vec::new(),
                });
                self.buffers
                    .last_mut()
                    .expect("buffer slot was just inserted")
            }
        };
        let group = match declarations
            .groups
            .iter()
            .position(|known| known.kind == kind && known.contract == contract)
        {
            Some(index) => &mut declarations.groups[index],
            None => {
                declarations.groups.push(BufferDeclarationGroup {
                    kind,
                    contract,
                    ranges: Vec::new(),
                });
                declarations
                    .groups
                    .last_mut()
                    .expect("buffer group was just inserted")
            }
        };
        group.ranges.push(range);
    }

    fn add_texture(
        &mut self,
        slot: TextureSlot,
        range: TextureRange,
        kind: AccessKind,
        contract: TextureAccessContract,
    ) {
        let declarations = match self.textures.iter().position(|known| known.slot == slot) {
            Some(index) => &mut self.textures[index],
            None => {
                self.textures.push(TextureSlotDeclarations {
                    slot,
                    declarations: Vec::new(),
                });
                self.textures
                    .last_mut()
                    .expect("texture slot was just inserted")
            }
        };
        if !declarations.declarations.iter().any(|declaration| {
            declaration.range == range
                && declaration.kind == kind
                && declaration.contract == contract
        }) {
            declarations.declarations.push(TextureDeclaration {
                range,
                kind,
                contract,
            });
        }
    }

    fn normalize_buffer_ranges(&mut self) {
        for declarations in &mut self.buffers {
            for group in &mut declarations.groups {
                normalize_ranges(&mut group.ranges);
            }
        }
    }

    /// Requires a fully declared buffer operation before recording it.
    pub(crate) fn require_buffer(
        &self,
        slot: BufferSlot,
        range: Range<u64>,
        kind: AccessKind,
        contract: BufferAccessContract,
    ) -> Result<(), AccessError> {
        let Some(declarations) = self
            .buffers
            .iter()
            .find(|declarations| declarations.slot == slot)
        else {
            return Err(AccessError::UndeclaredBuffer);
        };
        let mut compatible = declarations
            .groups
            .iter()
            .filter(|group| group.kind == kind && buffer_contract_covers(group.contract, contract));
        let Some(first) = compatible.next() else {
            return Err(AccessError::IncompatibleBufferContract);
        };
        if range.start >= range.end {
            return Err(AccessError::BufferRangeOutsideDeclaration);
        }
        let Some(second) = compatible.next() else {
            return first
                .covers(&range)
                .then_some(())
                .ok_or(AccessError::BufferRangeOutsideDeclaration);
        };
        if ranges_cover(
            std::iter::once(first)
                .chain(std::iter::once(second))
                .chain(compatible)
                .flat_map(|group| group.ranges.iter()),
            &range,
        ) {
            Ok(())
        } else {
            Err(AccessError::BufferRangeOutsideDeclaration)
        }
    }

    /// Requires a fully declared texture operation before recording it.
    pub(crate) fn require_texture(
        &self,
        slot: TextureSlot,
        range: TextureRange,
        kind: AccessKind,
        contract: TextureAccessContract,
    ) -> Result<(), AccessError> {
        let Some(declarations) = self
            .textures
            .iter()
            .find(|declarations| declarations.slot == slot)
        else {
            return Err(AccessError::UndeclaredTexture);
        };
        let compatible = declarations.declarations.iter().filter(|declaration| {
            declaration.kind == kind && texture_contract_covers(declaration.contract, contract)
        });
        if compatible.clone().next().is_none() {
            return Err(AccessError::IncompatibleTextureContract);
        }
        if texture_ranges_cover(compatible.map(|declaration| &declaration.range), &range) {
            Ok(())
        } else {
            Err(AccessError::TextureRangeOutsideDeclaration)
        }
    }
}

#[derive(Clone, Debug)]
struct BufferSlotDeclarations {
    slot: BufferSlot,
    groups: Vec<BufferDeclarationGroup>,
}

#[derive(Clone, Debug)]
struct BufferDeclarationGroup {
    kind: AccessKind,
    contract: BufferAccessContract,
    ranges: Vec<Range<u64>>,
}

impl BufferDeclarationGroup {
    fn covers(&self, requested: &Range<u64>) -> bool {
        let index = self
            .ranges
            .partition_point(|range| range.end <= requested.start);
        self.ranges
            .get(index)
            .is_some_and(|range| range.start <= requested.start && range.end >= requested.end)
    }
}

#[derive(Clone, Debug)]
struct TextureSlotDeclarations {
    slot: TextureSlot,
    declarations: Vec<TextureDeclaration>,
}

#[derive(Clone, Debug)]
struct TextureDeclaration {
    range: TextureRange,
    kind: AccessKind,
    contract: TextureAccessContract,
}

fn normalize_ranges(ranges: &mut Vec<Range<u64>>) {
    ranges.sort_unstable_by(|left, right| {
        left.start.cmp(&right.start).then(left.end.cmp(&right.end))
    });
    let mut normalized: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(previous) = normalized.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        normalized.push(range);
    }
    *ranges = normalized;
}

fn buffer_contract_covers(declared: BufferAccessContract, requested: BufferAccessContract) -> bool {
    declared.uses.stages.contains(requested.uses.stages)
        && declared.uses.accesses.contains(requested.uses.accesses)
        && declared
            .descriptor_usage
            .contains(requested.descriptor_usage)
}

fn texture_contract_covers(
    declared: TextureAccessContract,
    requested: TextureAccessContract,
) -> bool {
    declared.layout == requested.layout
        && declared.uses.stages.contains(requested.uses.stages)
        && declared.uses.accesses.contains(requested.uses.accesses)
        && declared
            .descriptor_usage
            .contains(requested.descriptor_usage)
}

fn ranges_cover<'a, T: Copy + Ord + 'a>(
    ranges: impl Iterator<Item = &'a Range<T>>,
    requested: &Range<T>,
) -> bool {
    if requested.start >= requested.end {
        return false;
    }
    let mut ranges = ranges.cloned().collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| range.start);
    let mut covered_end = requested.start;
    for range in ranges {
        if range.end <= covered_end || range.start > covered_end {
            continue;
        }
        covered_end = range.end;
        if covered_end >= requested.end {
            return true;
        }
    }
    false
}

fn texture_ranges_cover<'a>(
    ranges: impl Iterator<Item = &'a TextureRange>,
    requested: &TextureRange,
) -> bool {
    let known_aspects =
        TextureAspects::COLOR.0 | TextureAspects::DEPTH.0 | TextureAspects::STENCIL.0;
    if requested.aspects.0 == 0
        || requested.aspects.0 & !known_aspects != 0
        || requested.mips.start >= requested.mips.end
        || requested.layers.start >= requested.layers.end
    {
        return false;
    }
    let ranges = ranges.collect::<Vec<_>>();
    for aspect in [1_u8, 2, 4] {
        if requested.aspects.0 & aspect == 0 {
            continue;
        }
        let mut boundaries = vec![requested.mips.start, requested.mips.end];
        for range in &ranges {
            if range.aspects.0 & aspect != 0 {
                boundaries.push(range.mips.start.max(requested.mips.start));
                boundaries.push(range.mips.end.min(requested.mips.end));
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        if boundaries.first() != Some(&requested.mips.start)
            || boundaries.last() != Some(&requested.mips.end)
        {
            return false;
        }
        for window in boundaries.windows(2) {
            let mip_span = window[0]..window[1];
            if mip_span.start == mip_span.end
                || !ranges_cover(
                    ranges.iter().filter_map(|range| {
                        (range.aspects.0 & aspect != 0
                            && range.mips.start <= mip_span.start
                            && range.mips.end >= mip_span.end)
                            .then_some(&range.layers)
                    }),
                    &requested.layers,
                )
            {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
#[path = "../tests/graph/access.rs"]
mod tests;
