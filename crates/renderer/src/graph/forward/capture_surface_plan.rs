//! Same-submission Forward capture ending in an acquired surface lease.
use super::super::mip_generation::MipGenerationPass;
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
    pub(super) capture_mip_generations: Vec<MipGenerationPass>,
    pub(super) capture_raster: GraphPass,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_mip_generations: Vec<MipGenerationPass>,
    pub(super) receiver_background_copy: Option<GraphPass>,
    pub(super) receiver_depth_alias: bool,
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
        preserve_capture_background: bool,
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
            preserve_capture_background,
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
        let expected = declared_passes(&declaration);
        if execution.logical().passes() != expected {
            return Err(Error::InvalidState.into());
        }
        let present = StepId {
            plan: id,
            index: u32::try_from(expected.len()).map_err(|_| Error::InvalidRange)?,
        };
        let mut steps = execution.steps(device, id, declaration.lane)?;
        steps.push(StepDesc {
            lane: declaration.lane,
            kind: StepKind::Present,
            waits: vec![StepWait {
                previous: StepId {
                    plan: id,
                    index: present.index - 1,
                },
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
            capture_mip_generations: declaration.capture_mip_generations,
            capture_raster: declaration.capture_raster,
            receiver_upload: declaration.receiver_upload,
            receiver_mip_generations: declaration.receiver_mip_generations,
            receiver_background_copy: declaration.receiver_background_copy,
            receiver_depth_alias: declaration.receiver_depth_alias,
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
    pub(crate) fn composite_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: self.receiver_raster_step().index + 1,
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
            self.receiver_depth_alias,
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
        let expected = self.accepted_steps();
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(expected.as_slice())
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

impl<B: ResourceBackend> CaptureForwardSurfacePlan<B> {
    fn accepted_steps(&self) -> Vec<StepId> {
        let mut steps = Vec::with_capacity(
            6 + self.capture_mip_generations.len()
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
        steps.push(self.composite_step());
        steps.push(self.present_step());
        steps
    }
}

fn declared_passes(declaration: &CaptureForwardSurfaceDeclaration) -> Vec<GraphPass> {
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
    passes.push(declaration.composite);
    passes
}
