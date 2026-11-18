//! Compilation and completion evidence for the same-Graph Forward capture path.

use super::super::mip_generation::MipGenerationPass;
use super::*;

/// One executable Graph containing capture and receiver Forward frames.
pub(crate) struct CaptureForwardPlan<B: ResourceBackend> {
    pub(super) id: PlanId,
    pub(super) lane: LaneId,
    pub(super) submission: SubmissionPlan<B>,
    pub(super) execution: CompiledExecution,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_mip_generations: Vec<MipGenerationPass>,
    pub(super) capture_raster: GraphPass,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_mip_generations: Vec<MipGenerationPass>,
    pub(super) receiver_background_copy: Option<GraphPass>,
    pub(super) receiver_depth_alias: bool,
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
        preserve_capture_background: bool,
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
            preserve_capture_background,
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
        let expected = declared_passes(&declaration);
        if execution.logical().passes() != expected {
            return Err(Error::InvalidState.into());
        }
        let submission = execution.submission(device, id, declaration.lane)?;
        Ok(Self {
            id,
            lane: declaration.lane,
            submission,
            execution,
            capture_upload: declaration.capture_upload,
            capture_mip_generations: declaration.capture_mip_generations,
            capture_raster: declaration.capture_raster,
            receiver_upload: declaration.receiver_upload,
            receiver_mip_generations: declaration.receiver_mip_generations,
            receiver_background_copy: declaration.receiver_background_copy,
            receiver_depth_alias: declaration.receiver_depth_alias,
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
            index: 1 + u32::try_from(self.capture_mip_generations.len()).unwrap_or(u32::MAX),
        }
    }
    pub(crate) fn capture_mip_generations(&self) -> &[MipGenerationPass] {
        &self.capture_mip_generations
    }
    pub(crate) fn receiver_mip_generations(&self) -> &[MipGenerationPass] {
        &self.receiver_mip_generations
    }
    pub(crate) fn capture_mip_generation_step(&self, index: usize) -> Result<StepId> {
        if index >= self.capture_mip_generations.len() {
            return Err(Error::InvalidRange);
        }
        Ok(StepId {
            plan: self.id,
            index: 1 + u32::try_from(index).map_err(|_| Error::InvalidRange)?,
        })
    }
    pub(crate) fn receiver_upload_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: self.capture_raster_step().index + 1,
        }
    }
    pub(crate) fn receiver_raster_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: self.receiver_upload_step().index
                + 1
                + u32::try_from(self.receiver_mip_generations.len()).unwrap_or(u32::MAX)
                + u32::from(self.receiver_background_copy.is_some()),
        }
    }
    pub(crate) fn receiver_background_copy_step(&self) -> Option<StepId> {
        self.receiver_background_copy.map(|_| StepId {
            plan: self.id,
            index: self.receiver_upload_step().index
                + 1
                + u32::try_from(self.receiver_mip_generations.len()).unwrap_or(u32::MAX),
        })
    }
    pub(crate) fn receiver_mip_generation_step(&self, index: usize) -> Result<StepId> {
        if index >= self.receiver_mip_generations.len() {
            return Err(Error::InvalidRange);
        }
        Ok(StepId {
            plan: self.id,
            index: self.receiver_upload_step().index
                + 1
                + u32::try_from(index).map_err(|_| Error::InvalidRange)?,
        })
    }
    pub(crate) fn receiver_readback_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: self.receiver_raster_step().index + 1,
        }
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let expected = self.accepted_steps();
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(expected.as_slice())
        {
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

impl<B: ResourceBackend> CaptureForwardPlan<B> {
    fn accepted_steps(&self) -> Vec<StepId> {
        let mut steps = Vec::with_capacity(
            5 + self.capture_mip_generations.len()
                + self.receiver_mip_generations.len()
                + usize::from(self.receiver_background_copy.is_some()),
        );
        steps.push(self.capture_upload_step());
        for index in 0..self.capture_mip_generations.len() {
            steps.push(
                self.capture_mip_generation_step(index)
                    .expect("bounded by length"),
            );
        }
        steps.push(self.capture_raster_step());
        steps.push(self.receiver_upload_step());
        for index in 0..self.receiver_mip_generations.len() {
            steps.push(
                self.receiver_mip_generation_step(index)
                    .expect("bounded by length"),
            );
        }
        if let Some(step) = self.receiver_background_copy_step() {
            steps.push(step);
        }
        steps.push(self.receiver_raster_step());
        steps.push(self.receiver_readback_step());
        steps
    }
}

fn declared_passes(declaration: &CaptureForwardDeclaration) -> Vec<GraphPass> {
    let mut passes = Vec::with_capacity(
        5 + declaration.capture_mip_generations.len()
            + declaration.receiver_mip_generations.len()
            + usize::from(declaration.receiver_background_copy.is_some()),
    );
    passes.push(declaration.capture_upload);
    passes.extend(
        declaration
            .capture_mip_generations
            .iter()
            .map(|generation| generation.pass),
    );
    passes.push(declaration.capture_raster);
    passes.push(declaration.receiver_upload);
    passes.extend(
        declaration
            .receiver_mip_generations
            .iter()
            .map(|generation| generation.pass),
    );
    if let Some(copy) = declaration.receiver_background_copy {
        passes.push(copy);
    }
    passes.push(declaration.receiver_raster);
    passes.push(declaration.receiver_readback);
    passes
}
