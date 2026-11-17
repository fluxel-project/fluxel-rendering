//! Same-submission Forward capture ending in an acquired surface lease.
use super::capture_bindings::validate_capture_alias;
use super::*;
use crate::renderer::SurfaceCompositeOwner;
use fluxel_rhi::api::SurfaceImage;

/// Five command scopes and a terminal present for a captured Forward image.
pub(crate) struct CaptureForwardSurfacePlan<B: ResourceBackend> {
    pub(super) id: PlanId,
    pub(super) lane: LaneId,
    pub(super) submission: SubmissionPlan<B>,
    pub(super) execution: CompiledExecution,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_raster: GraphPass,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_raster: GraphPass,
    pub(super) composite: GraphPass,
    pub(super) present: StepId,
    pub(super) capture: CaptureFrameSlots,
    pub(super) receiver: CaptureFrameSlots,
    pub(super) surface_target: TextureSlot,
    pub(super) capture_topology: FrameTopology,
    pub(super) receiver_topology: FrameTopology,
    pub(super) alignment: u64,
    pub(super) capture_shape: ForwardShape,
    pub(super) receiver_shape: ForwardShape,
    pub(super) extent: [u32; 2],
    pub(super) color_format: TextureFormat,
}

impl<B: ResourceBackend> CaptureForwardSurfacePlan<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn declare_with<C, R>(
        device: &Device<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
        capture_topology: FrameTopology,
        receiver_topology: FrameTopology,
        capture_contribute: C,
        receiver_contribute: R,
    ) -> std::result::Result<CaptureForwardSurfaceDeclaration, ForwardPlanBuildError>
    where
        C: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
        R: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
    {
        declare_capture_surface_forward(
            device,
            capture,
            receiver,
            image,
            capture_topology,
            receiver_topology,
            capture_contribute,
            receiver_contribute,
        )
    }

    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: CaptureForwardSurfaceDeclaration,
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
                declaration.composite,
            ]
        {
            return Err(Error::InvalidState.into());
        }
        let present = StepId { plan: id, index: 5 };
        let mut steps = execution.steps(device, id, declaration.lane)?;
        steps.push(StepDesc {
            lane: declaration.lane,
            kind: StepKind::Present,
            waits: vec![StepWait {
                previous: StepId { plan: id, index: 4 },
                destination: AccessScope {
                    stages: StageMask::ALL_COMMANDS,
                    accesses: AccessMask::ALL,
                },
            }],
        });
        let submission = unsafe { SubmissionPlan::from_compiled(id, Arc::from(steps)) };
        submission.validate_for_execution(device)?;
        Ok(Self {
            id,
            lane: declaration.lane,
            submission,
            execution,
            capture_upload: declaration.capture_upload,
            capture_raster: declaration.capture_raster,
            receiver_upload: declaration.receiver_upload,
            receiver_raster: declaration.receiver_raster,
            composite: declaration.composite,
            present,
            capture: declaration.capture,
            receiver: declaration.receiver,
            surface_target: declaration.surface_target,
            capture_topology: declaration.capture_topology,
            receiver_topology: declaration.receiver_topology,
            alignment: declaration.alignment,
            capture_shape: declaration.capture_shape,
            receiver_shape: declaration.receiver_shape,
            extent: declaration.extent,
            color_format: declaration.color_format,
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
    pub(crate) fn composite_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 4,
        }
    }
    pub(crate) fn present_step(&self) -> StepId {
        self.present
    }

    pub(crate) fn validate_frames_with_image(
        &self,
        device: &Device<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        image: &SurfaceImage<B>,
        composite: &SurfaceCompositeOwner<B>,
    ) -> Result<()> {
        if device.id() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != self.extent || image.format() != self.color_format {
            return Err(Error::InvalidRange);
        }
        if composite.extent() != self.extent {
            return Err(Error::InvalidRange);
        }
        composite.validate_for(device, image)?;
        self.validate_frames(capture, receiver)?;
        if composite.target_instance() != receiver.frame.target_instance {
            return Err(Error::InvalidState);
        }
        self.bindings(capture, receiver, image).map(|_| ())
    }

    fn validate_frames(&self, capture: &ForwardFrame<B>, receiver: &ForwardFrame<B>) -> Result<()> {
        for (frame, shape, topology) in [
            (capture, &self.capture_shape, self.capture_topology),
            (receiver, &self.receiver_shape, self.receiver_topology),
        ] {
            validate_resident_texture_devices(&frame.resident, self.id.device)?;
            if let Some(owner) = frame.specular.as_ref() {
                owner.validate_for_sampling(self.id.device)?;
            }
            if frame.resident.device != self.id.device || forward_shape(frame) != *shape {
                return Err(Error::InvalidState);
            }
            validate_forward_topology(topology, frame.draws.len(), frame.draw_batches.len())?;
            validate_uploads(frame)?;
            validate_batches(frame)?;
            if let Some(environment) = frame.environment.as_ref() {
                environment.validate_for_sampling(self.id.device)?;
            }
            for draw in &frame.draws {
                validate_draw(draw, frame.resident.streams.len())?;
                validate_offsets(draw, self.alignment)?;
                draw_accesses(frame, draw)?;
            }
        }
        validate_capture_alias(capture, receiver)
    }

    pub(super) fn bindings<'a>(
        &'a self,
        capture: &'a ForwardFrame<B>,
        receiver: &'a ForwardFrame<B>,
        image: &'a SurfaceImage<B>,
    ) -> Result<FrameBindings<'a, B>> {
        validate_resident_texture_devices(&capture.resident, self.id.device)?;
        validate_resident_texture_devices(&receiver.resident, self.id.device)?;
        validate_capture_alias(capture, receiver)?;
        let (buffers, textures) = super::capture_binding_collection::collect(
            &self.execution,
            self.id.device,
            &self.capture,
            capture,
            &self.receiver,
            receiver,
        )?;
        FrameBindings::new_with_surface_textures(
            &self.execution,
            self.id.device,
            buffers,
            textures,
            vec![BoundSurfaceTexture::new(self.surface_target, image)?],
        )
        .map_err(binding_error)
    }
}

impl<B: ExecutionBackend + ResourceBackend> CaptureForwardSurfacePlan<B> {
    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>> {
        let expected = [
            self.capture_upload_step(),
            self.capture_raster_step(),
            self.receiver_upload_step(),
            self.receiver_raster_step(),
            self.composite_step(),
            self.present_step(),
        ];
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(&expected)
            || !facts.presents.is_some_and(|p| {
                p.len() == 1 && p[0].accepted && p[0].step == Some(self.present_step())
            })
        {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points
                    .iter()
                    .find(|point| {
                        point.point.device() == self.id.device
                            && point.covers.lane_frontiers.iter().any(|&(lane, step)| {
                                lane == self.lane && step == self.present_step()
                            })
                    })
                    .map(|point| point.point.clone())
            })
            .ok_or(Error::InvalidState)
    }
}
