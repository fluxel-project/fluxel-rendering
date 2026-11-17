//! Compilation and completion evidence for the same-Graph Forward capture path.

use super::*;

/// One executable Graph containing capture and receiver Forward frames.
pub(crate) struct CaptureForwardPlan<B: ResourceBackend> {
    pub(super) id: PlanId,
    pub(super) lane: LaneId,
    pub(super) submission: SubmissionPlan<B>,
    pub(super) execution: CompiledExecution,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_raster: GraphPass,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_raster: GraphPass,
    pub(super) receiver_readback: GraphPass,
    pub(super) capture: CaptureFrameSlots,
    pub(super) receiver: CaptureFrameSlots,
    pub(super) capture_topology: FrameTopology,
    pub(super) receiver_topology: FrameTopology,
    pub(super) alignment: u64,
    pub(super) capture_shape: ForwardShape,
    pub(super) receiver_shape: ForwardShape,
}

impl<B: ResourceBackend> CaptureForwardPlan<B> {
    /// Declares both Forward frames into one Graph transaction.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn declare_with<C, R>(
        device: &Device<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_topology: FrameTopology,
        receiver_topology: FrameTopology,
        capture_contribute: C,
        receiver_contribute: R,
    ) -> std::result::Result<CaptureForwardDeclaration, ForwardPlanBuildError>
    where
        C: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
        R: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
    {
        declare_capture_forward(
            device,
            capture,
            receiver,
            capture_topology,
            receiver_topology,
            capture_contribute,
            receiver_contribute,
        )
    }

    /// Compiles the capture-upload, capture-raster, receiver-upload,
    /// receiver-raster and receiver-readback schedule.
    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: CaptureForwardDeclaration,
    ) -> std::result::Result<Self, ForwardPlanBuildError> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice.into());
        }
        let execution = CompiledExecution::compile(&declaration.graph)?;
        if execution.logical().passes()
            != [
                declaration.capture_upload,
                declaration.capture_raster,
                declaration.receiver_upload,
                declaration.receiver_raster,
                declaration.receiver_readback,
            ]
        {
            return Err(Error::InvalidState.into());
        }
        let submission = execution.submission(device, id, declaration.lane)?;
        Ok(Self {
            id,
            lane: declaration.lane,
            submission,
            execution,
            capture_upload: declaration.capture_upload,
            capture_raster: declaration.capture_raster,
            receiver_upload: declaration.receiver_upload,
            receiver_raster: declaration.receiver_raster,
            receiver_readback: declaration.receiver_readback,
            capture: declaration.capture,
            receiver: declaration.receiver,
            capture_topology: declaration.capture_topology,
            receiver_topology: declaration.receiver_topology,
            alignment: declaration.alignment,
            capture_shape: declaration.capture_shape,
            receiver_shape: declaration.receiver_shape,
        })
    }

    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }
    pub(crate) fn capture_upload_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 0,
        }
    }
    pub(crate) fn capture_raster_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 1,
        }
    }
    pub(crate) fn receiver_upload_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 2,
        }
    }
    pub(crate) fn receiver_raster_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 3,
        }
    }
    pub(crate) fn receiver_readback_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 4,
        }
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let expected = [
            self.capture_upload_step(),
            self.capture_raster_step(),
            self.receiver_upload_step(),
            self.receiver_raster_step(),
            self.receiver_readback_step(),
        ];
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point.covers.lane_frontiers.iter().any(|&(lane, step)| {
                            lane == self.lane && step == self.receiver_readback_step()
                        })
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }
}
