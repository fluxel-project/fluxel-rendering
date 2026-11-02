//! Private state planning derived solely from declared graph access contracts.

use std::{collections::HashMap, ops::Range};

use fluxel_rhi::api::{BufferUses, ImageLayout, MemoryClass, TextureAspects, TextureUses};

use super::{
    binding::{BufferSlot, TextureSlot},
    definition::{AccessKind, GraphBuilder, GraphPass, TextureRange},
};

/// A state change required before one scheduled pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BufferTransition {
    pub(crate) slot: BufferSlot,
    /// Buffer state is physical-resource-wide even when the logical hazard was
    /// detected from a smaller access range.
    pub(crate) range: Range<u64>,
    pub(crate) from: BufferUses,
    pub(crate) to: BufferUses,
}

/// A state and layout change required before one scheduled pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextureTransition {
    pub(crate) slot: TextureSlot,
    pub(crate) range: TextureRange,
    pub(crate) from: TextureUses,
    pub(crate) to: TextureUses,
    pub(crate) layout: (ImageLayout, ImageLayout),
}

/// State changes to record before a scheduled pass.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PassTransitions {
    pub(crate) buffers: Vec<BufferTransition>,
    pub(crate) textures: Vec<TextureTransition>,
}

/// Structured failure for an executable graph that lacks state facts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StateError {
    UnknownPass,
    DuplicatePass,
    MissingBufferContract { pass: GraphPass, slot: BufferSlot },
    MissingTextureContract { pass: GraphPass, slot: TextureSlot },
    MissingBufferIncomingState { slot: BufferSlot },
    MissingTextureIncomingState { slot: TextureSlot },
    ConflictingTextureUse { pass: GraphPass, slot: TextureSlot },
    IllegalFixedMemoryUse { pass: GraphPass, slot: BufferSlot },
}

/// The private per-frame state plan. It has no backend policy: it is only the
/// RHI state vocabulary already declared by graph authors.
#[derive(Clone, Debug, Default)]
pub(crate) struct StatePlan {
    passes: HashMap<GraphPass, PassTransitions>,
}

#[derive(Clone)]
struct BufferState {
    uses: BufferUses,
    /// Accesses while `uses` is current. The resource state itself is
    /// whole-buffer, but range history avoids losing A/B/A hazards.
    accesses: Vec<RangeAccess>,
}

#[derive(Clone)]
struct RangeAccess {
    range: Range<u64>,
    write: bool,
}

#[derive(Clone, Copy)]
struct TextureState {
    uses: TextureUses,
    layout: ImageLayout,
    wrote: bool,
}

#[derive(Clone)]
struct DesiredBuffer {
    slot: BufferSlot,
    uses: BufferUses,
    accesses: Vec<RangeAccess>,
}

#[derive(Clone)]
struct DesiredTexture {
    slot: TextureSlot,
    range: TextureRange,
    uses: TextureUses,
    layout: ImageLayout,
    write: bool,
}

impl StatePlan {
    pub(crate) fn compile(graph: &GraphBuilder, order: &[GraphPass]) -> Result<Self, StateError> {
        let mut plan = Self::default();
        let mut buffers = HashMap::<BufferSlot, BufferState>::new();
        let mut textures = HashMap::<(TextureSlot, u8, u32, u32), TextureState>::new();

        for &pass in order {
            let node = graph
                .passes
                .get(pass.1)
                .filter(|node| node.id == pass)
                .ok_or(StateError::UnknownPass)?;
            if plan.passes.contains_key(&pass) {
                return Err(StateError::DuplicatePass);
            }

            let desired_buffers = desired_buffers(graph, pass, node)?;
            let desired_textures = desired_textures(graph, pass, node)?;
            let mut transitions = PassTransitions::default();

            for desired in desired_buffers {
                let buffer = &graph.buffers[desired.slot.resource];
                let initial = buffer
                    .initial
                    .ok_or(StateError::MissingBufferIncomingState { slot: desired.slot })?;

                // Upload and readback heaps have fixed DX12 native states.
                // Existing lowerers intentionally omit their source/destination
                // barriers, so this planner must never synthesize one.
                if matches!(buffer.memory, MemoryClass::Upload | MemoryClass::Readback) {
                    validate_fixed_memory_use(pass, &desired, buffer.memory)?;
                    continue;
                }

                let state = buffers.entry(desired.slot).or_insert_with(|| BufferState {
                    uses: initial,
                    accesses: initial_accesses(initial, buffer.size),
                });
                let changed = state.uses != desired.uses;
                let hazard = !changed && accesses_hazard(&state.accesses, &desired.accesses);
                if changed || hazard {
                    transitions.buffers.push(BufferTransition {
                        slot: desired.slot,
                        range: 0..buffer.size,
                        from: state.uses,
                        to: desired.uses,
                    });
                }
                if changed {
                    state.uses = desired.uses;
                    state.accesses.clear();
                }
                // A whole-resource barrier orders the prior accesses against
                // all uses in this pass. Disjoint uses without a barrier stay
                // in history so A/B/A cannot lose the earlier A hazard.
                if hazard {
                    state.accesses.clear();
                }
                state.accesses.extend(desired.accesses);
            }

            for desired in desired_textures {
                let texture = &graph.textures[desired.slot.resource];
                let (initial_uses, initial_layout) = texture
                    .initial
                    .ok_or(StateError::MissingTextureIncomingState { slot: desired.slot })?;
                for aspect in selected_aspects(desired.range.aspects) {
                    for mip in desired.range.mips.clone() {
                        for layer in desired.range.layers.clone() {
                            let key = (desired.slot, aspect, mip, layer);
                            let state = textures.entry(key).or_insert(TextureState {
                                uses: initial_uses,
                                layout: initial_layout,
                                wrote: texture_use_writes(initial_uses),
                            });
                            let changed =
                                state.uses != desired.uses || state.layout != desired.layout;
                            // Unlike a layout transition, an identical state does
                            // not establish ordering by itself. A prior write
                            // therefore still needs a barrier before either a
                            // subsequent read (RAW) or write (WAW); a current
                            // write also orders a prior read (WAR).
                            let hazard = !changed && (state.wrote || desired.write);
                            if changed || hazard {
                                transitions.textures.push(TextureTransition {
                                    slot: desired.slot,
                                    range: TextureRange::new(
                                        TextureAspects(aspect),
                                        mip..mip + 1,
                                        layer..layer + 1,
                                    ),
                                    from: state.uses,
                                    to: desired.uses,
                                    layout: (state.layout, desired.layout),
                                });
                            }
                            if changed {
                                state.uses = desired.uses;
                                state.layout = desired.layout;
                            }
                            // A whole-subresource barrier has consumed the
                            // earlier history. Retaining it would manufacture a
                            // second barrier for following read-only uses.
                            if changed || hazard {
                                state.wrote = desired.write;
                            } else {
                                state.wrote |= desired.write;
                            }
                        }
                    }
                }
            }

            plan.passes.insert(pass, transitions);
        }
        Ok(plan)
    }

    pub(crate) fn pass(&self, pass: GraphPass) -> Result<&PassTransitions, StateError> {
        self.passes.get(&pass).ok_or(StateError::UnknownPass)
    }
}

fn desired_buffers(
    graph: &GraphBuilder,
    pass: GraphPass,
    node: &super::definition::PassNode,
) -> Result<Vec<DesiredBuffer>, StateError> {
    let mut desired = Vec::<DesiredBuffer>::new();
    for access in &node.accesses {
        let slot = BufferSlot::from(access.buffer);
        let contract = access
            .contract
            .ok_or(StateError::MissingBufferContract { pass, slot })?;
        // Logical handles already validate graph provenance. This defensive
        // check makes manually assembled executable plans fail structurally.
        if slot.resource >= graph.buffers.len() {
            return Err(StateError::MissingBufferIncomingState { slot });
        }
        if let Some(existing) = desired.iter_mut().find(|item| item.slot == slot) {
            existing.uses = union_buffer_uses(existing.uses, contract.uses);
            existing.accesses.push(RangeAccess {
                range: access.range.bytes(),
                write: access.kind == AccessKind::Write,
            });
        } else {
            desired.push(DesiredBuffer {
                slot,
                uses: contract.uses,
                accesses: vec![RangeAccess {
                    range: access.range.bytes(),
                    write: access.kind == AccessKind::Write,
                }],
            });
        }
    }
    Ok(desired)
}

fn desired_textures(
    graph: &GraphBuilder,
    pass: GraphPass,
    node: &super::definition::PassNode,
) -> Result<Vec<DesiredTexture>, StateError> {
    let mut desired = Vec::<DesiredTexture>::new();
    for access in &node.texture_accesses {
        let slot = TextureSlot::from(access.texture);
        let contract = access
            .contract
            .ok_or(StateError::MissingTextureContract { pass, slot })?;
        if slot.resource >= graph.textures.len() {
            return Err(StateError::MissingTextureIncomingState { slot });
        }
        for aspect in selected_aspects(access.range.aspects) {
            for mip in access.range.mips.clone() {
                for layer in access.range.layers.clone() {
                    if let Some(existing) = desired.iter_mut().find(|item| {
                        item.slot == slot
                            && item.range.aspects == TextureAspects(aspect)
                            && item.range.mips == (mip..mip + 1)
                            && item.range.layers == (layer..layer + 1)
                    }) {
                        if existing.layout != contract.layout {
                            return Err(StateError::ConflictingTextureUse { pass, slot });
                        }
                        existing.uses = union_texture_uses(existing.uses, contract.uses);
                        existing.write |= access.kind == AccessKind::Write;
                    } else {
                        desired.push(DesiredTexture {
                            slot,
                            range: TextureRange::new(
                                TextureAspects(aspect),
                                mip..mip + 1,
                                layer..layer + 1,
                            ),
                            uses: contract.uses,
                            layout: contract.layout,
                            write: access.kind == AccessKind::Write,
                        });
                    }
                }
            }
        }
    }
    Ok(desired)
}

fn union_buffer_uses(left: BufferUses, right: BufferUses) -> BufferUses {
    BufferUses {
        stages: left.stages | right.stages,
        accesses: left.accesses | right.accesses,
    }
}

fn union_texture_uses(left: TextureUses, right: TextureUses) -> TextureUses {
    TextureUses {
        stages: left.stages | right.stages,
        accesses: left.accesses | right.accesses,
    }
}

fn selected_aspects(aspects: TextureAspects) -> impl Iterator<Item = u8> {
    [
        TextureAspects::COLOR.0,
        TextureAspects::DEPTH.0,
        TextureAspects::STENCIL.0,
    ]
    .into_iter()
    .filter(move |aspect| aspects.0 & aspect != 0)
}

fn ranges_overlap(left: &Range<u64>, right: &Range<u64>) -> bool {
    left.start < right.end && right.start < left.end
}

fn initial_accesses(uses: BufferUses, size: u64) -> Vec<RangeAccess> {
    vec![RangeAccess {
        range: 0..size,
        write: buffer_use_writes(uses),
    }]
}

fn accesses_hazard(prior: &[RangeAccess], current: &[RangeAccess]) -> bool {
    prior.iter().any(|prior| {
        current.iter().any(|current| {
            ranges_overlap(&prior.range, &current.range) && (prior.write || current.write)
        })
    })
}

fn validate_fixed_memory_use(
    pass: GraphPass,
    desired: &DesiredBuffer,
    memory: MemoryClass,
) -> Result<(), StateError> {
    let legal = match memory {
        MemoryClass::Upload => !buffer_use_writes(desired.uses),
        MemoryClass::Readback => {
            desired.uses.stages == fluxel_rhi::api::StageMask::COPY
                && desired.uses.accesses == fluxel_rhi::api::AccessMask::COPY_WRITE
                && desired.accesses.iter().all(|access| access.write)
        }
        MemoryClass::Device => true,
    };
    legal
        .then_some(())
        .ok_or(StateError::IllegalFixedMemoryUse {
            pass,
            slot: desired.slot,
        })
}

fn buffer_use_writes(uses: BufferUses) -> bool {
    const WRITES: u64 = fluxel_rhi::api::AccessMask::HOST_WRITE.0
        | fluxel_rhi::api::AccessMask::COPY_WRITE.0
        | fluxel_rhi::api::AccessMask::SHADER_WRITE.0
        | fluxel_rhi::api::AccessMask::COLOR_WRITE.0
        | fluxel_rhi::api::AccessMask::DEPTH_WRITE.0
        | fluxel_rhi::api::AccessMask::STENCIL_WRITE.0
        | fluxel_rhi::api::AccessMask::ACCEL_WRITE.0
        | fluxel_rhi::api::AccessMask::QUERY_RESOLVE_WRITE.0;
    uses.accesses.0 & WRITES != 0
}

fn texture_use_writes(uses: TextureUses) -> bool {
    buffer_use_writes(BufferUses {
        stages: uses.stages,
        accesses: uses.accesses,
    })
}

#[cfg(test)]
#[path = "../tests/graph/state.rs"]
mod tests;
