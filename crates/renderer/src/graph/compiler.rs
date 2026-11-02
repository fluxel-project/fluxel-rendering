//! Deterministic private graph compilation, culling, and lifetime analysis.

use fluxel_rhi::api::{AccessMask, AccessScope, StageMask};
use std::collections::{BTreeSet, HashMap};

use super::{
    binding::{
        BufferBindingManifest, BufferRequirement, BufferSlot, TextureBindingManifest,
        TextureRequirement, TextureSlot,
    },
    definition::{
        AccessKind, BufferAccessContract, BufferHandle, GraphBuilder, GraphError, GraphPass,
        TextureAccessContract, TextureHandle, TextureRange,
    },
};

/// The first and final live scheduled pass that use one logical version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResourceLifetime {
    pub(crate) buffer: BufferHandle,
    pub(crate) first_pass: usize,
    pub(crate) last_pass: usize,
}

/// The first and final live scheduled pass that use one logical texture version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TextureLifetime {
    pub(crate) texture: TextureHandle,
    pub(crate) first_pass: usize,
    pub(crate) last_pass: usize,
}

/// A compiled, culled, deterministic execution order.
#[derive(Clone, Debug)]
pub(crate) struct CompiledGraph {
    order: Vec<GraphPass>,
    pass_contracts: Vec<PassContract>,
    lifetimes: Vec<ResourceLifetime>,
    texture_lifetimes: Vec<TextureLifetime>,
    buffer_bindings: BufferBindingManifest,
    texture_bindings: TextureBindingManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PassContract {
    dependencies: Vec<usize>,
    buffers: Vec<(usize, u32, AccessKind, Option<BufferAccessContract>)>,
    textures: Vec<(
        usize,
        u32,
        TextureRange,
        AccessKind,
        Option<TextureAccessContract>,
    )>,
}

/// Why a candidate logical graph cannot reuse the cached device plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecompileReason {
    PassTopology,
    Dependencies,
    BufferLayout,
    BufferAccess,
    TextureLayout,
    TextureAccess,
}

impl CompiledGraph {
    pub(crate) fn pass_dependencies(&self, index: usize) -> &[usize] {
        &self.pass_contracts[index].dependencies
    }

    pub(crate) fn pass_scope(&self, index: usize) -> AccessScope {
        let pass = &self.pass_contracts[index];
        let mut scope = AccessScope {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        for (_, _, _, contract) in &pass.buffers {
            if let Some(contract) = contract {
                scope.stages |= contract.uses.stages;
                scope.accesses |= contract.uses.accesses;
            }
        }
        for (_, _, _, _, contract) in &pass.textures {
            if let Some(contract) = contract {
                scope.stages |= contract.uses.stages;
                scope.accesses |= contract.uses.accesses;
            }
        }
        if scope.stages == StageMask::EMPTY {
            scope = AccessScope {
                stages: StageMask::ALL_COMMANDS,
                accesses: AccessMask::ALL,
            };
        }
        scope
    }

    /// Returns the deterministic live pass order.
    pub(crate) fn passes(&self) -> &[GraphPass] {
        &self.order
    }
    /// Returns one lifetime for each live logical version with a declared use.
    #[allow(dead_code)] // The first runtime lowerer validates topology only.
    pub(crate) fn lifetimes(&self) -> &[ResourceLifetime] {
        &self.lifetimes
    }
    /// Returns texture-version lifetimes derived by this same compilation.
    #[allow(dead_code)] // The raster lowerer consumes this after its first fixture.
    pub(crate) fn texture_lifetimes(&self) -> &[TextureLifetime] {
        &self.texture_lifetimes
    }
    /// Returns concrete buffer requirements from the compiled live accesses.
    pub(crate) fn buffer_bindings(&self) -> &BufferBindingManifest {
        &self.buffer_bindings
    }
    /// Returns concrete texture requirements from the compiled live accesses.
    pub(crate) fn texture_bindings(&self) -> &TextureBindingManifest {
        &self.texture_bindings
    }

    /// Compares a newly compiled frame graph with this cached logical topology.
    ///
    /// Buffer byte ranges may change while the pass/resource/use contract stays
    /// fixed. The candidate compiler has already checked definedness and
    /// dependencies; changed dependency edges or texture subresources require
    /// a new device plan.
    #[allow(dead_code)] // Fixed-shape lowerers still use their explicit shape cache keys.
    pub(crate) fn recompile_reason(&self, candidate: &Self) -> Option<RecompileReason> {
        self.compare_contract(candidate, false)
    }

    /// Allows buffer extents to vary only when the frame recorder uses the
    /// candidate graph's ranges and validates its own concrete bindings.
    pub(crate) fn recompile_reason_for_dynamic_buffers(
        &self,
        candidate: &Self,
    ) -> Option<RecompileReason> {
        self.compare_contract(candidate, true)
    }

    fn compare_contract(
        &self,
        candidate: &Self,
        dynamic_buffer_extents: bool,
    ) -> Option<RecompileReason> {
        if self.pass_contracts.len() != candidate.pass_contracts.len() {
            return Some(RecompileReason::PassTopology);
        }
        if self
            .pass_contracts
            .iter()
            .zip(&candidate.pass_contracts)
            .any(|(old, new)| old.dependencies != new.dependencies)
        {
            return Some(RecompileReason::Dependencies);
        }
        for (old, new) in self.pass_contracts.iter().zip(&candidate.pass_contracts) {
            if old.buffers.len() != new.buffers.len()
                || old
                    .buffers
                    .iter()
                    .zip(&new.buffers)
                    .any(|(a, b)| a.0 != b.0 || a.1 != b.1 || a.2 != b.2 || a.3 != b.3)
            {
                return Some(RecompileReason::BufferAccess);
            }
            if old.textures != new.textures {
                return Some(RecompileReason::TextureAccess);
            }
        }
        let old_buffers = self.buffer_bindings.requirements();
        let new_buffers = candidate.buffer_bindings.requirements();
        if old_buffers.len() != new_buffers.len()
            || old_buffers.iter().zip(new_buffers).any(|(old, new)| {
                old.slot.resource != new.slot.resource
                    || (!dynamic_buffer_extents && old.size != new.size)
                    || old.usage != new.usage
                    || old.memory != new.memory
                    || old.initial != new.initial
                    || old.first_pass != new.first_pass
                    || old.last_pass != new.last_pass
                    || old.accesses.len() != new.accesses.len()
            })
        {
            return Some(RecompileReason::BufferLayout);
        }
        let old_textures = self.texture_bindings.requirements();
        let new_textures = candidate.texture_bindings.requirements();
        if old_textures.len() != new_textures.len()
            || old_textures.iter().zip(new_textures).any(|(old, new)| {
                old.slot.resource != new.slot.resource
                    || !same_texture_desc(&old.desc, &new.desc)
                    || old.initial != new.initial
                    || old.first_pass != new.first_pass
                    || old.last_pass != new.last_pass
                    || old.accesses != new.accesses
            })
        {
            return Some(RecompileReason::TextureLayout);
        }
        None
    }
}

fn same_texture_desc(
    left: &fluxel_rhi::api::TextureDesc,
    right: &fluxel_rhi::api::TextureDesc,
) -> bool {
    left.dimension == right.dimension
        && left.extent == right.extent
        && left.array_layers == right.array_layers
        && left.format == right.format
        && left.mip_levels == right.mip_levels
        && left.samples == right.samples
        && left.usage == right.usage
        && left.view_formats == right.view_formats
}

/// Compiles one graph definition; it owns no RHI execution state.
#[derive(Default)]
pub(crate) struct GraphCompiler;

impl GraphCompiler {
    /// Culls dead passes, topologically orders the remainder, and computes lifetimes.
    pub(crate) fn compile(&self, graph: &GraphBuilder) -> Result<CompiledGraph, GraphError> {
        let live = live_passes(graph)?;
        let order = topological_order(graph, &live)?;
        Ok(CompiledGraph {
            pass_contracts: pass_contracts(graph, &order),
            lifetimes: lifetimes(graph, &order),
            texture_lifetimes: texture_lifetimes(graph, &order),
            buffer_bindings: BufferBindingManifest::new(buffer_requirements(graph, &order)),
            texture_bindings: TextureBindingManifest::new(texture_requirements(graph, &order)),
            order,
        })
    }
}

fn pass_contracts(graph: &GraphBuilder, order: &[GraphPass]) -> Vec<PassContract> {
    let schedule = order
        .iter()
        .enumerate()
        .map(|(position, pass)| (pass.1, position))
        .collect::<HashMap<_, _>>();
    order
        .iter()
        .map(|pass| {
            let node = &graph.passes[pass.1];
            let mut dependencies = node
                .dependencies
                .iter()
                .filter_map(|dependency| schedule.get(&dependency.1).copied())
                .collect::<Vec<_>>();
            dependencies.sort_unstable();
            let buffers = node
                .accesses
                .iter()
                .map(|access| {
                    (
                        access.buffer.resource,
                        access.buffer.version,
                        access.kind,
                        access.contract,
                    )
                })
                .collect();
            let textures = node
                .texture_accesses
                .iter()
                .map(|access| {
                    (
                        access.texture.resource,
                        access.texture.version,
                        access.range.clone(),
                        access.kind,
                        access.contract,
                    )
                })
                .collect();
            PassContract {
                dependencies,
                buffers,
                textures,
            }
        })
        .collect()
}

fn texture_requirements(graph: &GraphBuilder, order: &[GraphPass]) -> Vec<TextureRequirement> {
    let mut output = Vec::<TextureRequirement>::new();
    for (position, pass) in order.iter().enumerate() {
        for access in &graph.passes[pass.1].texture_accesses {
            let slot = TextureSlot::from(access.texture);
            if let Some(requirement) = output.iter_mut().find(|item| item.slot == slot) {
                requirement.first_pass = requirement.first_pass.min(position);
                requirement.last_pass = requirement.last_pass.max(position);
                requirement.ranges.push(access.range.clone());
                requirement
                    .accesses
                    .push((access.range.clone(), access.kind, access.contract));
                continue;
            }
            let node = &graph.textures[access.texture.resource];
            output.push(TextureRequirement {
                slot,
                desc: node.desc.clone(),
                initial: node.initial,
                ranges: vec![access.range.clone()],
                accesses: vec![(access.range.clone(), access.kind, access.contract)],
                first_pass: position,
                last_pass: position,
            });
        }
    }
    output
}

fn buffer_requirements(graph: &GraphBuilder, order: &[GraphPass]) -> Vec<BufferRequirement> {
    let mut output = Vec::<BufferRequirement>::new();
    for (position, pass) in order.iter().enumerate() {
        for access in &graph.passes[pass.1].accesses {
            let slot = BufferSlot::from(access.buffer);
            if let Some(requirement) = output.iter_mut().find(|item| item.slot == slot) {
                requirement.first_pass = requirement.first_pass.min(position);
                requirement.last_pass = requirement.last_pass.max(position);
                requirement.ranges.push(access.range.bytes());
                requirement
                    .accesses
                    .push((access.range.bytes(), access.kind, access.contract));
                continue;
            }
            let node = &graph.buffers[access.buffer.resource];
            output.push(BufferRequirement {
                slot,
                size: node.size,
                usage: node.usage,
                memory: node.memory,
                initial: node.initial,
                ranges: vec![access.range.bytes()],
                accesses: vec![(access.range.bytes(), access.kind, access.contract)],
                first_pass: position,
                last_pass: position,
            });
        }
    }
    output
}

fn live_passes(graph: &GraphBuilder) -> Result<BTreeSet<GraphPass>, GraphError> {
    let mut live = BTreeSet::new();
    let mut roots = graph
        .passes
        .iter()
        .filter_map(|pass| pass.side_effect.then_some(pass.id))
        .collect::<Vec<_>>();
    for exported in &graph.exports {
        if let Some(producer) = graph.version(*exported)?.producer {
            roots.push(producer);
        }
    }
    for exported in &graph.texture_exports {
        if let Some(producer) = graph.texture_version(*exported)?.producer {
            roots.push(producer);
        }
    }
    while let Some(pass) = roots.pop() {
        if live.insert(pass) {
            roots.extend(graph.passes[pass.1].dependencies.iter().copied());
        }
    }
    Ok(live)
}

fn topological_order(
    graph: &GraphBuilder,
    live: &BTreeSet<GraphPass>,
) -> Result<Vec<GraphPass>, GraphError> {
    let mut remaining = live.clone();
    let mut order = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let next = remaining.iter().copied().find(|pass| {
            graph.passes[pass.1]
                .dependencies
                .iter()
                .all(|dependency| !remaining.contains(dependency))
        });
        let Some(next) = next else {
            return Err(GraphError::Cycle);
        };
        remaining.remove(&next);
        order.push(next);
    }
    Ok(order)
}

fn lifetimes(graph: &GraphBuilder, order: &[GraphPass]) -> Vec<ResourceLifetime> {
    let mut output = Vec::<ResourceLifetime>::new();
    for (position, pass) in order.iter().enumerate() {
        let _ = &graph.passes[pass.1].name;
        for access in &graph.passes[pass.1].accesses {
            let _ = (&access.range, access.kind);
            if let Some(lifetime) = output.iter_mut().find(|item| item.buffer == access.buffer) {
                lifetime.last_pass = position;
            } else {
                output.push(ResourceLifetime {
                    buffer: access.buffer,
                    first_pass: position,
                    last_pass: position,
                });
            }
        }
    }
    output
}

fn texture_lifetimes(graph: &GraphBuilder, order: &[GraphPass]) -> Vec<TextureLifetime> {
    let mut output = Vec::<TextureLifetime>::new();
    for (position, pass) in order.iter().enumerate() {
        for access in &graph.passes[pass.1].texture_accesses {
            let _ = (&access.range, access.kind);
            if let Some(lifetime) = output
                .iter_mut()
                .find(|item| item.texture == access.texture)
            {
                lifetime.last_pass = position;
            } else {
                output.push(TextureLifetime {
                    texture: access.texture,
                    first_pass: position,
                    last_pass: position,
                });
            }
        }
    }
    output
}

#[cfg(test)]
#[path = "../tests/graph/compiler.rs"]
mod tests;
