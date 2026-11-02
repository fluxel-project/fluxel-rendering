//! Fixed Copy consumer of the shared graph execution contracts.

use std::ops::Range;

use fluxel_rhi::{
    api::{
        AccessMask, BufferDesc, BufferUsage, BufferUses, CommandEncoder, CompletionPoint, Device,
        DeviceId, Error, ExecutionStatus, LaneClass, LaneId, MemoryClass, PlanId, ReadbackTicket,
        Result, StageMask, StepId, SubmissionPlan,
    },
    backend::{
        Backend, CommandBackend, ExecutionBackend, ExecutionMode, ExecutionOutcome,
        ReadbackBackend, ResourceBackend,
    },
};

use super::{
    binding::{BindingError, BufferSlot},
    compiler::CompiledGraph,
    definition::{AccessKind, BufferAccessContract, GraphBuilder, GraphPass},
    execution::{BoundBuffer, CompiledExecution, FrameBindings, binding_error},
};
use crate::residency::UploadBuffers;

type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;

/// One frame's concrete byte-range declarations and executable contracts.
pub(crate) struct CopyFrame {
    len: u64,
    source_range: Range<u64>,
    source_slot: BufferSlot,
    target_slot: BufferSlot,
    upload: GraphPass,
    readback: GraphPass,
    execution: CompiledExecution,
}

impl CopyFrame {
    pub(crate) fn compile_range(source_bytes: u64, source_range: Range<u64>) -> Result<Self> {
        if source_range.start >= source_range.end || source_range.end > source_bytes {
            return Err(Error::InvalidRange);
        }
        let len = source_range.end - source_range.start;
        let mut graph = GraphBuilder::default();
        let source = graph
            .import_buffer_desc(
                BufferDesc {
                    label: Some("fluxel renderer upload".to_owned()),
                    size: source_bytes,
                    usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
                    memory: MemoryClass::Upload,
                },
                host_write_use(),
            )
            .map_err(|_| Error::InvalidRange)?;
        let target = graph
            .create_buffer_desc(BufferDesc {
                label: Some("fluxel renderer target".to_owned()),
                size: len,
                usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            })
            .map_err(|_| Error::InvalidRange)?;
        let source_slot = BufferSlot::from(source);
        let target_slot = BufferSlot::from(target);
        let (upload, target) = graph
            .add_pass("copy-upload", |pass| {
                pass.read_with_use(
                    source,
                    source_range.clone(),
                    copy_read_scope(),
                    BufferUsage::COPY_SRC,
                )?;
                pass.write_with_use(target, 0..len, copy_write_scope(), BufferUsage::COPY_DST)
            })
            .map_err(|_| Error::InvalidState)?;
        let (readback, ()) = graph
            .add_pass("copy-readback", |pass| {
                pass.read_with_use(target, 0..len, copy_read_scope(), BufferUsage::COPY_SRC)?;
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        let execution = CompiledExecution::compile(&graph)?;
        if execution.logical().passes() != [upload, readback] {
            return Err(Error::InvalidState);
        }
        Ok(Self {
            len,
            source_range,
            source_slot,
            target_slot,
            upload,
            readback,
            execution,
        })
    }

    fn bind<'a, B: ResourceBackend>(
        &'a self,
        device: DeviceId,
        buffers: &'a UploadBuffers<B>,
    ) -> std::result::Result<FrameBindings<'a, B>, BindingError> {
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
                    self.target_slot,
                    buffers.target_instance,
                    &buffers.target,
                    &buffers.target_desc,
                    empty_scope(),
                ),
            ],
            vec![],
        )
    }

    pub(crate) fn validate_bindings<B: ResourceBackend>(
        &self,
        device: DeviceId,
        buffers: &UploadBuffers<B>,
    ) -> std::result::Result<(), BindingError> {
        self.bind(device, buffers).map(|_| ())
    }
}

/// One device execution topology, independent of each frame's concrete ranges.
pub(crate) struct CopyPlan<B: Backend> {
    id: PlanId,
    lane: LaneId,
    submission: SubmissionPlan<B>,
    upload_step: StepId,
    readback_step: StepId,
    logical: CompiledGraph,
}

impl<B: Backend> CopyPlan<B> {
    pub(crate) fn compile(device: &Device<B>, id: PlanId, frame: &CopyFrame) -> Result<Self> {
        if id.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let lane = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .map(|lane| lane.id)
            .ok_or(Error::WrongLane)?;
        let submission = frame.execution.submission(device, id, lane)?;
        Ok(Self {
            id,
            lane,
            submission,
            upload_step: StepId { plan: id, index: 0 },
            readback_step: StepId { plan: id, index: 1 },
            logical: frame.execution.logical().clone(),
        })
    }

    pub(crate) fn validate_frame(&self, frame: &CopyFrame) -> Result<()> {
        if self
            .logical
            .recompile_reason_for_dynamic_buffers(frame.execution.logical())
            .is_some()
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
    pub(crate) fn id(&self) -> PlanId {
        self.id
    }
    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }
    pub(crate) fn upload_step(&self) -> StepId {
        self.upload_step
    }
    pub(crate) fn readback_step(&self) -> StepId {
        self.readback_step
    }

    pub(crate) fn record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &UploadBuffers<B>,
        frame: &CopyFrame,
    ) -> Result<()>
    where
        B: ResourceBackend + CommandBackend,
    {
        self.validate_frame(frame)?;
        let bindings = frame.bind(self.id.device, buffers).map_err(binding_error)?;
        let pass = bindings.pass(frame.upload)?;
        let source = pass.buffer(
            frame.source_slot,
            frame.source_range.clone(),
            AccessKind::Read,
            BufferAccessContract {
                uses: copy_read_scope(),
                descriptor_usage: BufferUsage::COPY_SRC,
            },
        )?;
        let target = pass.buffer(
            frame.target_slot,
            0..frame.len,
            AccessKind::Write,
            BufferAccessContract {
                uses: copy_write_scope(),
                descriptor_usage: BufferUsage::COPY_DST,
            },
        )?;
        pass.transition(encoder)?;
        encoder.copy_buffer(source, frame.source_range.start, target, 0, frame.len)
    }

    pub(crate) fn record_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        buffers: &UploadBuffers<B>,
        frame: &CopyFrame,
    ) -> Result<ReadbackTicket<B>>
    where
        B: ResourceBackend + ReadbackBackend,
    {
        self.validate_frame(frame)?;
        let bindings = frame.bind(self.id.device, buffers).map_err(binding_error)?;
        let pass = bindings.pass(frame.readback)?;
        let target = pass.buffer(
            frame.target_slot,
            0..frame.len,
            AccessKind::Read,
            BufferAccessContract {
                uses: copy_read_scope(),
                descriptor_usage: BufferUsage::COPY_SRC,
            },
        )?;
        pass.transition(encoder)?;
        encoder.readback_buffer(target, 0..frame.len)
    }
}

impl<B: ExecutionBackend> CopyPlan<B> {
    /// Extracts only the exact checkpoint covering accepted fixed-plan readback work.
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>> {
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(&[self.upload_step, self.readback_step])
        {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|checkpoints| {
                checkpoints.iter().find(|checkpoint| {
                    checkpoint.point.device() == self.id.device
                        && checkpoint
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == self.readback_step)
                })
            })
            .map(|checkpoint| checkpoint.point.clone())
            .ok_or(Error::InvalidState)
    }
}

fn empty_scope() -> BufferUses {
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
fn copy_read_scope() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}
fn copy_write_scope() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}

#[cfg(test)]
#[path = "../tests/graph/copy.rs"]
mod tests;
