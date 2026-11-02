//! Checked pass resources and portable lowering of compiled graph contracts.

use std::{ops::Range, sync::Arc};

use fluxel_rhi::{
    api::{
        Buffer, BufferBarrier, BufferDesc, BufferSlice, BufferUses, CommandEncoder, Device,
        DeviceId, Error, ImageLayout, LaneId, PlanId, ResourceTransition, Result, StateTransition,
        StepDesc, StepId, StepKind, StepWait, SubmissionPlan, Subresources, Texture,
        TextureBarrier, TextureDesc, TextureUses,
    },
    backend::{Backend, CommandBackend},
};

use super::{
    access::{AccessPlan, PassAccess},
    binding::{
        BindingError, BufferBinding, BufferSlot, ResourceInstanceId, TextureBinding, TextureSlot,
    },
    compiler::{CompiledGraph, GraphCompiler},
    definition::{
        AccessKind, BufferAccessContract, GraphBuilder, GraphPass, TextureAccessContract,
        TextureRange,
    },
    state::{PassTransitions, StatePlan},
};

/// Executable resource contracts derived from one logical graph definition.
#[derive(Clone, Debug)]
pub(crate) struct CompiledExecution {
    logical: CompiledGraph,
    access: AccessPlan,
    states: StatePlan,
}

impl CompiledExecution {
    pub(crate) fn compile(graph: &GraphBuilder) -> Result<Self> {
        let logical = GraphCompiler
            .compile(graph)
            .map_err(|_| Error::InvalidState)?;
        u32::try_from(logical.passes().len()).map_err(|_| Error::InvalidRange)?;
        let access =
            AccessPlan::compile(graph, logical.passes()).map_err(|_| Error::InvalidState)?;
        let states =
            StatePlan::compile(graph, logical.passes()).map_err(|_| Error::InvalidState)?;
        Ok(Self {
            logical,
            access,
            states,
        })
    }

    pub(crate) fn logical(&self) -> &CompiledGraph {
        &self.logical
    }

    /// Lowers only the already compiled dependencies on one selected RHI lane.
    ///
    /// Fixture plans may append an RHI-native terminal step such as presentation
    /// before constructing their one validated submission plan.
    pub(crate) fn steps<B: Backend>(
        &self,
        device: &Device<B>,
        id: PlanId,
        lane: LaneId,
    ) -> Result<Vec<StepDesc>> {
        if id.device != device.id() || lane.device != device.id() {
            return Err(Error::WrongDevice);
        }
        Ok((0..self.logical.passes().len())
            .map(|index| {
                let destination = self.logical.pass_scope(index);
                let waits = self
                    .logical
                    .pass_dependencies(index)
                    .iter()
                    .map(|previous| StepWait {
                        previous: StepId {
                            plan: id,
                            index: *previous as u32,
                        },
                        destination,
                    })
                    .collect();
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits,
                }
            })
            .collect())
    }

    /// Creates and validates a submission plan for the graph-only command steps.
    pub(crate) fn submission<B: Backend>(
        &self,
        device: &Device<B>,
        id: PlanId,
        lane: LaneId,
    ) -> Result<SubmissionPlan<B>> {
        let steps = self.steps(device, id, lane)?;
        // SAFETY: the unique compiler established the live DAG and the chosen
        // lane preserves its deterministic schedule. No resource work is issued here.
        let plan = unsafe { SubmissionPlan::from_compiled(id, Arc::from(steps)) };
        plan.validate_for_execution(device)?;
        Ok(plan)
    }
}

pub(crate) fn binding_error(error: BindingError) -> Error {
    match error {
        BindingError::WrongDevice => Error::WrongDevice,
        _ => Error::InvalidState,
    }
}

/// Owner-supplied facts are tied to the exact handle that pass recording resolves.
pub(crate) struct BoundBuffer<'a, B: Backend> {
    facts: BufferBinding,
    resource: &'a Buffer<B>,
}

impl<'a, B: Backend> BoundBuffer<'a, B> {
    pub(crate) fn new(
        slot: BufferSlot,
        instance: ResourceInstanceId,
        resource: &'a Buffer<B>,
        desc: &BufferDesc,
        initial: BufferUses,
    ) -> Self {
        Self {
            facts: BufferBinding::new(slot, instance, resource, desc, initial),
            resource,
        }
    }
}

pub(crate) struct BoundTexture<'a, B: Backend> {
    facts: TextureBinding,
    resource: &'a Texture<B>,
}

impl<'a, B: Backend> BoundTexture<'a, B> {
    pub(crate) fn new(
        slot: TextureSlot,
        instance: ResourceInstanceId,
        resource: &'a Texture<B>,
        desc: &TextureDesc,
        initial: TextureUses,
        layout: ImageLayout,
    ) -> Self {
        Self {
            facts: TextureBinding::new(slot, instance, resource, desc, initial, layout),
            resource,
        }
    }
}

/// Complete validated resources for a frame, borrowed from its retained owner.
pub(crate) struct FrameBindings<'a, B: Backend> {
    graph: &'a CompiledExecution,
    buffers: Vec<BoundBuffer<'a, B>>,
    textures: Vec<BoundTexture<'a, B>>,
}

impl<'a, B: Backend> FrameBindings<'a, B> {
    pub(crate) fn new(
        graph: &'a CompiledExecution,
        device: DeviceId,
        buffers: Vec<BoundBuffer<'a, B>>,
        textures: Vec<BoundTexture<'a, B>>,
    ) -> std::result::Result<Self, BindingError> {
        let buffer_facts = buffers
            .iter()
            .map(|binding| binding.facts.clone())
            .collect::<Vec<_>>();
        let texture_facts = textures
            .iter()
            .map(|binding| binding.facts.clone())
            .collect::<Vec<_>>();
        graph
            .logical
            .buffer_bindings()
            .validate(device, &buffer_facts)?;
        graph
            .logical
            .texture_bindings()
            .validate(device, &texture_facts)?;
        Ok(Self {
            graph,
            buffers,
            textures,
        })
    }

    pub(crate) fn pass(&self, pass: GraphPass) -> Result<CheckedPass<'_, 'a, B>> {
        let access = self
            .graph
            .access
            .pass(pass)
            .map_err(|_| Error::InvalidState)?;
        let transitions = self
            .graph
            .states
            .pass(pass)
            .map_err(|_| Error::InvalidState)?;
        Ok(CheckedPass {
            bindings: self,
            access,
            transitions,
        })
    }
}

/// Pass-local authority resolves only declared resources/ranges/contracts.
pub(crate) struct CheckedPass<'p, 'a, B: Backend> {
    bindings: &'p FrameBindings<'a, B>,
    access: &'p PassAccess,
    transitions: &'p PassTransitions,
}

impl<B: Backend> CheckedPass<'_, '_, B> {
    pub(crate) fn buffer(
        &self,
        slot: BufferSlot,
        range: Range<u64>,
        kind: AccessKind,
        contract: BufferAccessContract,
    ) -> Result<&Buffer<B>> {
        self.access
            .require_buffer(slot, range, kind, contract)
            .map_err(|_| Error::InvalidState)?;
        self.raw_buffer(slot)
    }

    pub(crate) fn texture(
        &self,
        slot: TextureSlot,
        range: TextureRange,
        kind: AccessKind,
        contract: TextureAccessContract,
    ) -> Result<&Texture<B>> {
        self.access
            .require_texture(slot, range, kind, contract)
            .map_err(|_| Error::InvalidState)?;
        self.raw_texture(slot)
    }

    fn raw_buffer(&self, slot: BufferSlot) -> Result<&Buffer<B>> {
        self.bindings
            .buffers
            .iter()
            .find(|binding| binding.facts.slot == slot)
            .map(|binding| binding.resource)
            .ok_or(Error::InvalidState)
    }

    fn raw_texture(&self, slot: TextureSlot) -> Result<&Texture<B>> {
        self.bindings
            .textures
            .iter()
            .find(|binding| binding.facts.slot == slot)
            .map(|binding| binding.resource)
            .ok_or(Error::InvalidState)
    }

    pub(crate) fn transition(&self, encoder: &mut CommandEncoder<B>) -> Result<()>
    where
        B: CommandBackend,
    {
        let mut transitions = Vec::new();
        for change in &self.transitions.buffers {
            transitions.push(ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: self.raw_buffer(change.slot)?,
                    range: change.range.clone(),
                },
                usage: StateTransition {
                    from: change.from,
                    to: change.to,
                },
                ownership: None,
            }));
        }
        for change in &self.transitions.textures {
            transitions.push(ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: self.raw_texture(change.slot)?,
                    aspects: change.range.aspects,
                    mips: change.range.mips.clone(),
                    layers: change.range.layers.clone(),
                },
                usage: StateTransition {
                    from: change.from,
                    to: change.to,
                },
                layout: Some(change.layout),
                ownership: None,
            }));
        }
        if !transitions.is_empty() {
            encoder.transition_resources(&transitions)?;
        }
        Ok(())
    }
}
