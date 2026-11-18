//! Compilation and completion evidence for one capture Graph with optional OIT.

use super::*;

pub(crate) struct CaptureOitPlan<B: ResourceBackend> {
    pub(super) id: PlanId,
    pub(super) lane: LaneId,
    pub(super) submission: SubmissionPlan<B>,
    pub(super) execution: CompiledExecution,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) capture_raster: GraphPass,
    pub(super) capture_oit: Option<super::oit_plan::ForwardOitPlan>,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) receiver_raster: GraphPass,
    pub(super) receiver_oit: Option<super::oit_plan::ForwardOitPlan>,
    pub(super) terminal: super::capture_oit_declaration::CaptureOitTerminal,
    pub(super) capture: CaptureFrameSlots,
    pub(super) receiver: CaptureFrameSlots,
    pub(super) receiver_final: TextureSlot,
    pub(super) alignment: u64,
    pub(super) capture_shape: ForwardShape,
    pub(super) receiver_shape: ForwardShape,
}

impl<B: ResourceBackend> CaptureOitPlan<B> {
    /// Declares capture work with producer-specific ordinary and OIT reads.
    ///
    /// Both callbacks are invoked while constructing this one Graph.  The
    /// ordinary callback is used only when the producer selected no peeling
    /// candidates; the OIT callback declares every selected material stage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn declare_with_capture_contribution<N, C>(
        device: &Device<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_oit: Option<&ForwardOitInput<'_, B>>,
        receiver_oit: Option<&ForwardOitInput<'_, B>>,
        capture_ordinary: N,
        capture_contribute: C,
    ) -> std::result::Result<CaptureOitDeclaration, ForwardPlanBuildError>
    where
        N: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
        C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
    {
        super::capture_oit_declaration::declare_capture_oit_with_capture_contribution(
            device,
            capture,
            receiver,
            capture_oit,
            receiver_oit,
            capture_ordinary,
            capture_contribute,
        )
    }

    /// Surface form of [`Self::declare_with_capture_contribution`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn declare_surface_with_capture_contribution<N, C>(
        device: &Device<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_oit: Option<&ForwardOitInput<'_, B>>,
        receiver_oit: Option<&ForwardOitInput<'_, B>>,
        image: &SurfaceImage<B>,
        capture_ordinary: N,
        capture_contribute: C,
    ) -> std::result::Result<CaptureOitDeclaration, ForwardPlanBuildError>
    where
        N: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
        C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
    {
        super::capture_oit_surface_declaration::declare_capture_oit_surface_with_capture_contribution(
            device,
            capture,
            receiver,
            capture_oit,
            receiver_oit,
            image,
            capture_ordinary,
            capture_contribute,
        )
    }

    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: CaptureOitDeclaration,
    ) -> std::result::Result<Self, ForwardPlanBuildError> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice.into());
        }
        let execution = CompiledExecution::compile(&declaration.graph)?;
        let terminal_pass = match declaration.terminal {
            super::capture_oit_declaration::CaptureOitTerminal::Readback { pass }
            | super::capture_oit_declaration::CaptureOitTerminal::Surface { pass, .. } => pass,
        };
        if !execution.logical().passes().contains(&terminal_pass) {
            return Err(Error::InvalidState.into());
        }
        let mut steps = execution.steps(device, id, declaration.lane)?;
        let has_surface = matches!(
            declaration.terminal,
            super::capture_oit_declaration::CaptureOitTerminal::Surface { .. }
        );
        if has_surface {
            let previous = StepId {
                plan: id,
                index: u32::try_from(steps.len().checked_sub(1).ok_or(Error::InvalidState)?)
                    .map_err(|_| Error::InvalidRange)?,
            };
            steps.push(StepDesc {
                lane: declaration.lane,
                kind: StepKind::Present,
                waits: vec![StepWait {
                    previous,
                    destination: AccessScope {
                        stages: StageMask::ALL_COMMANDS,
                        accesses: AccessMask::ALL,
                    },
                }],
            });
        }
        let submission = if has_surface {
            // SAFETY: the exact compiled Graph command steps precede one
            // native presentation that explicitly waits for the final command.
            let submission = unsafe { SubmissionPlan::from_compiled(id, Arc::from(steps)) };
            submission.validate_for_execution(device)?;
            submission
        } else {
            execution.submission(device, id, declaration.lane)?
        };
        Ok(Self {
            id,
            lane: declaration.lane,
            submission,
            execution,
            capture_upload: declaration.capture_upload,
            capture_mips: declaration.capture_mips,
            capture_raster: declaration.capture_raster,
            capture_oit: declaration.capture_oit,
            receiver_upload: declaration.receiver_upload,
            receiver_mips: declaration.receiver_mips,
            receiver_raster: declaration.receiver_raster,
            receiver_oit: declaration.receiver_oit,
            terminal: declaration.terminal,
            capture: declaration.capture,
            receiver: declaration.receiver,
            receiver_final: declaration.receiver_final,
            alignment: declaration.alignment,
            capture_shape: declaration.capture_shape,
            receiver_shape: declaration.receiver_shape,
        })
    }

    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }

    pub(crate) fn steps(&self) -> impl Iterator<Item = (StepId, GraphPass)> + '_ {
        self.execution
            .logical()
            .passes()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, pass)| {
                (
                    StepId {
                        plan: self.id,
                        index: u32::try_from(index).expect("compiled pass index fits u32"),
                    },
                    pass,
                )
            })
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        if !matches!(
            self.terminal,
            super::capture_oit_declaration::CaptureOitTerminal::Readback { .. }
        ) {
            return Err(Error::InvalidState);
        }
        let expected: Vec<_> = self.steps().map(|(step, _)| step).collect();
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(expected.as_slice())
        {
            return Err(Error::InvalidState);
        }
        let terminal = expected.last().copied().ok_or(Error::InvalidState)?;
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == terminal)
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }

    pub(crate) fn present_step(&self) -> Result<StepId> {
        if !matches!(
            self.terminal,
            super::capture_oit_declaration::CaptureOitTerminal::Surface { .. }
        ) {
            return Err(Error::InvalidState);
        }
        Ok(StepId {
            plan: self.id,
            index: u32::try_from(self.execution.logical().passes().len())
                .map_err(|_| Error::InvalidRange)?,
        })
    }

    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let present = self.present_step()?;
        let mut expected: Vec<_> = self.steps().map(|(step, _)| step).collect();
        expected.push(present);
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(expected.as_slice())
            || !facts
                .presents
                .is_some_and(|p| p.len() == 1 && p[0].accepted && p[0].step == Some(present))
        {
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
                            .any(|&(lane, step)| lane == self.lane && step == present)
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }
}
