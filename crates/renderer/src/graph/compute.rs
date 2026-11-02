//! Fixed storage-buffer compute lowering for the first S1 fixture.

use fluxel_rhi::{
    api::{
        AccessMask, BindGroup, BindGroupDesc, BindGroupEntry, BindGroupLayout, BindingResource,
        BufferDesc, BufferUsage, BufferUses, CommandEncoder, CompletionPoint, ComputePipeline,
        Device, DeviceId, Error, ExecutionStatus, LaneClass, LaneId, MemoryClass, PlanId,
        ReadbackTicket, Result, StageMask, StepId, StorageBufferView, SubmissionPlan,
    },
    backend::{
        Backend, CommandBackend, ExecutionBackend, ExecutionMode, ExecutionOutcome,
        ReadbackBackend, ResourceBackend,
    },
};

use super::{
    binding::{BindingError, BufferSlot},
    definition::{AccessKind, BufferAccessContract, GraphBuilder, GraphPass},
    execution::{BoundBuffer, CompiledExecution, FrameBindings, binding_error},
};
use crate::residency::ComputeBuffers;

type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;

const ELEMENT_BYTES: u64 = 4;
const WORKGROUP_SIZE: u32 = 64;

/// One bind group created from the frame storage binding that Graph preflight
/// accepted for this fixed compute plan.
///
/// The private wrapper prevents recording code from substituting another
/// same-device bind group whose storage access was never declared to Graph.
pub(crate) struct ComputeFixtureBindings<B: Backend> {
    plan: PlanId,
    storage_instance: super::binding::ResourceInstanceId,
    group: BindGroup<B>,
    _layout: BindGroupLayout<B>,
}

/// A fixed upload, in-place compute, and readback submission plan.
pub(crate) struct ComputePlan<B: Backend> {
    device: Device<B>,
    id: PlanId,
    len: u64,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    execution: CompiledExecution,
    upload: GraphPass,
    compute: GraphPass,
    readback: GraphPass,
    source_slot: BufferSlot,
    storage_slot: BufferSlot,
}

impl<B: Backend> ComputePlan<B> {
    /// Compiles the fixed `value * 3 + 7` storage-buffer topology.
    pub(crate) fn compile(device: &Device<B>, id: PlanId, len: u64) -> Result<Self> {
        if device.compute().is_none() {
            return Err(Error::Unsupported("compute"));
        }
        if len == 0 || len % ELEMENT_BYTES != 0 {
            return Err(Error::InvalidRange);
        }
        if id.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let lane = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .map(|lane| lane.id)
            .ok_or(Error::WrongLane)?;

        let mut graph = GraphBuilder::default();
        let source_desc = BufferDesc {
            label: Some("fluxel renderer compute upload".to_owned()),
            size: len,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        };
        let storage_desc = BufferDesc {
            label: Some("fluxel renderer compute storage".to_owned()),
            size: len,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC | BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        };
        let source = graph
            .import_buffer_desc(source_desc, host_write_use())
            .map_err(|_| Error::InvalidRange)?;
        let storage = graph
            .create_buffer_desc(storage_desc)
            .map_err(|_| Error::InvalidRange)?;
        let source_slot = BufferSlot::from(source);
        let storage_slot = BufferSlot::from(storage);
        let (upload, storage) = graph
            .add_pass("compute-upload", |pass| {
                pass.read_with_use(source, 0..len, copy_read_use(), BufferUsage::COPY_SRC)?;
                pass.write_with_use(storage, 0..len, copy_write_use(), BufferUsage::COPY_DST)
            })
            .map_err(|_| Error::InvalidState)?;
        let (compute, storage) = graph
            .add_pass("compute-dispatch", |pass| {
                pass.read_with_use(storage, 0..len, compute_read_use(), BufferUsage::STORAGE)?;
                pass.write_with_use(storage, 0..len, compute_write_use(), BufferUsage::STORAGE)
            })
            .map_err(|_| Error::InvalidState)?;
        let (readback, ()) = graph
            .add_pass("compute-readback", |pass| {
                pass.read_with_use(storage, 0..len, copy_read_use(), BufferUsage::COPY_SRC)?;
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        let execution = CompiledExecution::compile(&graph)?;
        if execution.logical().passes() != [upload, compute, readback] {
            return Err(Error::InvalidState);
        }
        let submission = execution.submission(device, id, lane)?;
        Ok(Self {
            device: device.clone(),
            id,
            len,
            lane,
            submission,
            execution,
            upload,
            compute,
            readback,
            source_slot,
            storage_slot,
        })
    }

    pub(crate) fn id(&self) -> PlanId {
        self.id
    }
    pub(crate) fn count(&self) -> Result<u32> {
        u32::try_from(self.len / ELEMENT_BYTES).map_err(|_| Error::InvalidRange)
    }
    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }
    pub(crate) fn upload_step(&self) -> StepId {
        self.step_for(self.upload)
    }
    pub(crate) fn compute_step(&self) -> StepId {
        self.step_for(self.compute)
    }
    pub(crate) fn readback_step(&self) -> StepId {
        self.step_for(self.readback)
    }
    /// Builds the fixture bind group only after the exact frame resources have
    /// passed the compiled binding manifest.
    pub(crate) fn fixture_bindings(
        &self,
        layout: BindGroupLayout<B>,
        buffers: &ComputeBuffers<B>,
    ) -> Result<ComputeFixtureBindings<B>>
    where
        B: ResourceBackend,
    {
        self.check_device(layout.device())?;
        self.frame_bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let entries = [BindGroupEntry {
            binding: 0,
            resource: BindingResource::StorageBufferView {
                buffer: &buffers.storage,
                range: 0..self.len,
                view: StorageBufferView::Raw,
            },
        }];
        let group = self.device.create_bind_group(&BindGroupDesc {
            label: Some("fluxel renderer compute fixture"),
            layout: &layout,
            entries: &entries,
        })?;
        Ok(ComputeFixtureBindings {
            plan: self.id,
            storage_instance: buffers.storage_instance,
            group,
            _layout: layout,
        })
    }

    fn frame_bindings<'a>(
        &'a self,
        device: DeviceId,
        buffers: &'a ComputeBuffers<B>,
    ) -> std::result::Result<FrameBindings<'a, B>, BindingError>
    where
        B: ResourceBackend,
    {
        FrameBindings::new(
            &self.execution,
            device,
            vec![
                BoundBuffer::new(
                    self.source_slot,
                    buffers.upload_instance,
                    &buffers.upload,
                    &buffers.upload_desc,
                    host_write_use(),
                ),
                BoundBuffer::new(
                    self.storage_slot,
                    buffers.storage_instance,
                    &buffers.storage,
                    &buffers.storage_desc,
                    empty_use(),
                ),
            ],
            Vec::new(),
        )
    }

    pub(crate) fn record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &ComputeBuffers<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let bindings = self
            .frame_bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let pass = bindings.pass(self.upload)?;
        let upload = pass.buffer(
            self.source_slot,
            0..self.len,
            AccessKind::Read,
            copy_read_contract(),
        )?;
        let storage = pass.buffer(
            self.storage_slot,
            0..self.len,
            AccessKind::Write,
            copy_write_contract(),
        )?;
        pass.transition(encoder)?;
        encoder.copy_buffer(upload, 0, storage, 0, self.len)
    }

    pub(crate) fn record_compute(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &ComputeBuffers<B>,
        pipeline: &ComputePipeline<B>,
        fixture_bindings: &ComputeFixtureBindings<B>,
        count: u32,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        self.check_device(pipeline.device())?;
        if fixture_bindings.plan != self.id
            || fixture_bindings.storage_instance != buffers.storage_instance
        {
            return Err(Error::InvalidState);
        }
        if count != self.count()? {
            return Err(Error::InvalidRange);
        }
        let frame_bindings = self
            .frame_bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let pass = frame_bindings.pass(self.compute)?;
        let _read = pass.buffer(
            self.storage_slot,
            0..self.len,
            AccessKind::Read,
            compute_read_contract(),
        )?;
        pass.buffer(
            self.storage_slot,
            0..self.len,
            AccessKind::Write,
            compute_write_contract(),
        )?;
        pass.transition(encoder)?;
        let domain = self.device.compute().ok_or(Error::Unsupported("compute"))?;
        let mut pass = domain.begin_labeled(encoder, "fluxel compute fixture")?;
        pass.set_pipeline(pipeline)?;
        pass.set_bind_group(0, &fixture_bindings.group, &[])?;
        pass.dispatch([dispatch_groups(count)?, 1, 1])?;
        pass.finish()
    }

    pub(crate) fn record_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &ComputeBuffers<B>,
    ) -> Result<ReadbackTicket<B>>
    where
        B: ReadbackBackend + ResourceBackend,
    {
        let bindings = self
            .frame_bindings(self.id.device, buffers)
            .map_err(binding_error)?;
        let pass = bindings.pass(self.readback)?;
        let storage = pass.buffer(
            self.storage_slot,
            0..self.len,
            AccessKind::Read,
            copy_read_contract(),
        )?;
        pass.transition(encoder)?;
        encoder.readback_buffer(storage, 0..self.len)
    }

    fn step_for(&self, pass: GraphPass) -> StepId {
        let index = self
            .execution
            .logical()
            .passes()
            .iter()
            .position(|known| *known == pass)
            .expect("stored fixture pass remains in its compiled logical graph");
        StepId {
            plan: self.id,
            index: u32::try_from(index).expect("compiled pass count is u32-bounded"),
        }
    }

    fn check_device(&self, device: fluxel_rhi::api::DeviceId) -> Result<()> {
        if device == self.id.device {
            Ok(())
        } else {
            Err(Error::WrongDevice)
        }
    }
}

impl<B: ExecutionBackend> ComputePlan<B> {
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>> {
        let facts = outcome.facts();
        let expected = [
            self.upload_step(),
            self.compute_step(),
            self.readback_step(),
        ];
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == self.readback_step())
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }
}

fn dispatch_groups(count: u32) -> Result<u32> {
    let groups = count / WORKGROUP_SIZE;
    groups
        .checked_add(u32::from(count % WORKGROUP_SIZE != 0))
        .ok_or(Error::InvalidRange)
}
fn empty_use() -> BufferUses {
    BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}
fn host_write_use() -> BufferUses {
    BufferUses {
        stages: StageMask::HOST,
        accesses: AccessMask::HOST_WRITE,
    }
}
fn copy_write_use() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}
fn compute_read_use() -> BufferUses {
    BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ,
    }
}
fn compute_write_use() -> BufferUses {
    BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
    }
}
fn copy_read_use() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}
fn copy_read_contract() -> BufferAccessContract {
    BufferAccessContract {
        uses: copy_read_use(),
        descriptor_usage: BufferUsage::COPY_SRC,
    }
}
fn copy_write_contract() -> BufferAccessContract {
    BufferAccessContract {
        uses: copy_write_use(),
        descriptor_usage: BufferUsage::COPY_DST,
    }
}
fn compute_read_contract() -> BufferAccessContract {
    BufferAccessContract {
        uses: compute_read_use(),
        descriptor_usage: BufferUsage::STORAGE,
    }
}
fn compute_write_contract() -> BufferAccessContract {
    BufferAccessContract {
        uses: compute_write_use(),
        descriptor_usage: BufferUsage::STORAGE,
    }
}

#[cfg(test)]
#[path = "../tests/graph/compute.rs"]
mod tests;
