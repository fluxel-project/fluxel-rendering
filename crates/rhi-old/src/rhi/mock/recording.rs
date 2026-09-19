//! The mock's recorder backend.
//!
//! The recorder façade performs every portable check before it calls a backend,
//! so this backend receives only commands that are already legal in portable
//! terms, and it never re-decides them. What it does do is journal every call
//! and fail on demand, which is the only way a test can reach the branch of the
//! recording contract where an *injected backend failure* — not a portable
//! parameter error — poisons the recorder.
//!
//! The journal is why a test can tell "the recorder refused this before the
//! backend saw it" from "the backend refused it": in the first case the
//! command's entry is absent, in the second it is present.

use std::sync::Arc;

use crate::rhi::command::{
    RecordedCommand, RecordedWorkBackend, RecorderBackend, RasterScopeDescriptor, attachment,
};
use crate::rhi::platform::{DeviceIdentity, Label, ObjectId, RhiResult, next_object_id};
use crate::rhi::resource::{ReadbackRequest, ReadbackTicket, ReadbackTicketBackend};

use super::MockState;
use super::resources::MockReadbackTicket;

/// The backend half of one mock recording.
pub(super) struct MockRecorderBackend {
    device: DeviceIdentity,
    state: Arc<MockState>,
}

impl MockRecorderBackend {
    /// A recorder over `state`, for `device`.
    ///
    /// The descriptor's label is journalled here rather than stored: it is a
    /// diagnostic name for a recording, and a recording that has been created
    /// has nothing further to do with it.
    pub(super) fn new(device: DeviceIdentity, state: Arc<MockState>, label: Label) -> Self {
        state.note(format!("recorder:new:{label:?}"));
        Self { device, state }
    }
}

impl RecorderBackend for MockRecorderBackend {
    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn begin_raster(
        &mut self,
        desc: &RasterScopeDescriptor,
        geometry: attachment::AttachmentGeometry,
    ) -> RhiResult<()> {
        self.state.note("recorder:begin_raster");
        debug_assert_eq!(geometry.device, self.device);
        debug_assert!(desc.active_color_count() > 0, "the façade validated this");
        Ok(())
    }

    fn end_raster(&mut self) -> RhiResult<()> {
        self.state.note("recorder:end_raster");
        match self.state.take_failure(super::Injected::EndRaster, "end_raster") {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn begin_compute(&mut self, _label: &Label) -> RhiResult<()> {
        self.state.note("recorder:begin_compute");
        Ok(())
    }

    fn end_compute(&mut self) -> RhiResult<()> {
        self.state.note("recorder:end_compute");
        match self.state.take_failure(super::Injected::EndCompute, "end_compute") {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn record(&mut self, command: &RecordedCommand) -> RhiResult<()> {
        self.state
            .note(format!("recorder:record:{}", command_name(command)));
        match self.state.take_failure(super::Injected::Record, "record") {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn encode_readback(&mut self, request: &ReadbackRequest) -> RhiResult<ReadbackTicket> {
        self.state.note("recorder:encode_readback");
        match self.state.take_failure(super::Injected::Record, "encode_readback") {
            Some(error) => Err(error),
            None => Ok(ReadbackTicket::new(Arc::new(MockReadbackTicket {
                id: next_object_id(),
                device: self.device,
                request: request.clone(),
            }) as Arc<dyn ReadbackTicketBackend>)),
        }
    }

    fn finish(self: Box<Self>) -> RhiResult<Box<dyn RecordedWorkBackend>> {
        self.state.note("recorder:finish");
        match self.state.take_failure(super::Injected::Finish, "finish") {
            Some(error) => Err(error),
            None => Ok(Box::new(MockRecordedWork {
                id: next_object_id(),
                device: self.device,
            })),
        }
    }
}

/// The backend-owned work one mock recording produced.
///
/// It carries no native object: everything the plan needs — domains, uses,
/// attachments, commands — was derived by the recorder façade, and the mock has
/// no lower representation to add.
pub(super) struct MockRecordedWork {
    id: ObjectId,
    device: DeviceIdentity,
}

impl RecordedWorkBackend for MockRecordedWork {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }
}

/// The journal tag for one recorded command.
fn command_name(command: &RecordedCommand) -> &'static str {
    match command {
        RecordedCommand::BeginRaster(_) => "BeginRaster",
        RecordedCommand::EndRaster => "EndRaster",
        RecordedCommand::BeginCompute(_) => "BeginCompute",
        RecordedCommand::EndCompute => "EndCompute",
        RecordedCommand::SetRasterPipeline(_) => "SetRasterPipeline",
        RecordedCommand::SetComputePipeline(_) => "SetComputePipeline",
        RecordedCommand::SetBindGroup { .. } => "SetBindGroup",
        RecordedCommand::SetVertexBuffer { .. } => "SetVertexBuffer",
        RecordedCommand::SetIndexBuffer { .. } => "SetIndexBuffer",
        RecordedCommand::SetViewport(_) => "SetViewport",
        RecordedCommand::SetScissor(_) => "SetScissor",
        RecordedCommand::SetBlendConstant(_) => "SetBlendConstant",
        RecordedCommand::SetStencilReference(_) => "SetStencilReference",
        RecordedCommand::Draw { .. } => "Draw",
        RecordedCommand::DrawIndexed { .. } => "DrawIndexed",
        RecordedCommand::Dispatch { .. } => "Dispatch",
        RecordedCommand::EncodeUpload(_) => "EncodeUpload",
        RecordedCommand::EncodeReadback(_) => "EncodeReadback",
        RecordedCommand::CopyBuffer(_) => "CopyBuffer",
        RecordedCommand::CopyBufferToTexture(_) => "CopyBufferToTexture",
        RecordedCommand::CopyTextureToBuffer(_) => "CopyTextureToBuffer",
        RecordedCommand::CopyTexture(_) => "CopyTexture",
        RecordedCommand::ResolveTexture(_) => "ResolveTexture",
        RecordedCommand::BlitTexture(_) => "BlitTexture",
        RecordedCommand::PushDebugGroup(_) => "PushDebugGroup",
        RecordedCommand::PopDebugGroup => "PopDebugGroup",
        RecordedCommand::DebugMarker(_) => "DebugMarker",
    }
}
