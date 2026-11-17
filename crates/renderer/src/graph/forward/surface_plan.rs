//! Surface Plan for the bounded Forward product.
use super::*;

impl<B: ResourceBackend> SurfaceForwardPlan<B> {
    pub(crate) fn declare_with<F>(
        device: &Device<B>,
        frame: &SurfaceForwardFrame<B>,
        expected: FrameTopology,
        contribute: F,
    ) -> std::result::Result<SurfaceForwardDeclaration, ForwardPlanBuildError>
    where
        F: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
    {
        declare_surface_with(device, frame, expected, contribute)
    }

    pub(crate) fn validate_declaration(
        &self,
        declaration: &SurfaceForwardDeclaration,
    ) -> Result<()> {
        if !self.declaration.same_declaration(&declaration.graph)
            || self.lane != declaration.lane
            || self.upload_pass.1 != declaration.upload_pass.1
            || self.raster_pass.1 != declaration.raster_pass.1
            || self.resident_upload_slot.resource != declaration.resident_upload_slot.resource
            || self.frame_upload_slot.resource != declaration.frame_upload_slot.resource
            || self.uniform_slot.resource != declaration.uniform_slot.resource
            || self.environment_upload_slot.map(|slot| slot.resource)
                != declaration
                    .environment_upload_slot
                    .map(|slot| slot.resource)
            || self.specular_upload_slot.map(|slot| slot.resource)
                != declaration.specular_upload_slot.map(|slot| slot.resource)
            || self.specular_view_uniform_slot.map(|slot| slot.resource)
                != declaration
                    .specular_view_uniform_slot
                    .map(|slot| slot.resource)
            || !same_buffer_slots(&self.stream_slots, &declaration.stream_slots)
            || !same_texture_slots(&self.texture_slots, &declaration.texture_slots)
            || self.environment_slot.map(|slot| slot.resource)
                != declaration.environment_slot.map(|slot| slot.resource)
            || self.specular_radiance_slot.map(|slot| slot.resource)
                != declaration.specular_radiance_slot.map(|slot| slot.resource)
            || self.specular_brdf_lut_slot.map(|slot| slot.resource)
                != declaration.specular_brdf_lut_slot.map(|slot| slot.resource)
            || self.depth_slot.map(|slot| slot.resource)
                != declaration.depth_slot.map(|slot| slot.resource)
            || self.surface_target_slot.resource != declaration.surface_target_slot.resource
            || self.extent != declaration.extent
            || self.draw_count != declaration.draw_count
            || self.stream_count != declaration.stream_count
            || self.texture_count != declaration.texture_count
            || self.color_format != declaration.color_format
            || self.shape != declaration.shape
            || self.topology != declaration.topology
            || self.uniform_alignment != declaration.uniform_alignment
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }

    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: SurfaceForwardDeclaration,
    ) -> std::result::Result<Self, ForwardPlanBuildError> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice.into());
        }
        let execution = CompiledExecution::compile(&declaration.graph)?;
        if execution.logical().passes() != [declaration.upload_pass, declaration.raster_pass] {
            return Err(Error::InvalidState.into());
        }
        let present_step = StepId { plan: id, index: 2 };
        let mut steps = execution.steps(device, id, declaration.lane)?;
        steps.push(StepDesc {
            lane: declaration.lane,
            kind: StepKind::Present,
            waits: vec![StepWait {
                previous: StepId { plan: id, index: 1 },
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
            declaration: declaration.graph.clone(),
            upload_pass: declaration.upload_pass,
            raster_pass: declaration.raster_pass,
            present_step,
            resident_upload_slot: declaration.resident_upload_slot,
            frame_upload_slot: declaration.frame_upload_slot,
            uniform_slot: declaration.uniform_slot,
            environment_upload_slot: declaration.environment_upload_slot,
            specular_upload_slot: declaration.specular_upload_slot,
            specular_view_uniform_slot: declaration.specular_view_uniform_slot,
            stream_slots: declaration.stream_slots,
            texture_slots: declaration.texture_slots,
            environment_slot: declaration.environment_slot,
            specular_radiance_slot: declaration.specular_radiance_slot,
            specular_brdf_lut_slot: declaration.specular_brdf_lut_slot,
            depth_slot: declaration.depth_slot,
            surface_target_slot: declaration.surface_target_slot,
            extent: declaration.extent,
            draw_count: declaration.draw_count,
            stream_count: declaration.stream_count,
            texture_count: declaration.texture_count,
            color_format: declaration.color_format,
            shape: declaration.shape,
            topology: declaration.topology,
            uniform_alignment: declaration.uniform_alignment,
        })
    }

    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }
    pub(crate) fn id(&self) -> PlanId {
        self.id
    }
    pub(crate) fn upload_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 0,
        }
    }
    pub(crate) fn raster_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 1,
        }
    }
    pub(crate) fn present_step(&self) -> StepId {
        self.present_step
    }
    pub(crate) fn validate_frame(&self, frame: &SurfaceForwardFrame<B>) -> Result<()> {
        if frame.resident.device != self.id.device
            || frame
                .environment
                .as_ref()
                .is_some_and(|owner| owner.device != self.id.device)
            || frame.extent != self.extent
            || frame.color_format != self.color_format
            || frame.draws.len() != self.draw_count
            || frame.resident.streams.len() != self.stream_count
            || frame.resident.textures.len() != self.texture_count
            || self.shape != surface_forward_shape(frame)
        {
            return Err(Error::InvalidState);
        }
        validate_surface_topology(self.topology, frame.draws.len(), frame.draw_batches.len())?;
        validate_surface_uploads(frame)?;
        if let Some(environment) = frame.environment.as_ref() {
            environment.validate_for_sampling(self.id.device)?;
        }
        if let Some(specular) = frame.specular.as_ref() {
            specular.validate_for_sampling(self.id.device)?;
        }
        validate_surface_batches(frame)?;
        for draw in &frame.draws {
            validate_draw(draw, self.stream_count)?;
            validate_offsets(draw, self.uniform_alignment)?;
            surface_draw_accesses(frame, draw)?;
            if draw.texture >= self.texture_count {
                return Err(Error::InvalidRange);
            }
        }
        Ok(())
    }

    pub(crate) fn validate_frame_with_image(
        &self,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()> {
        self.validate_frame(frame)?;
        self.bindings(frame, image).map(|_| ())
    }

    pub(super) fn bindings<'a>(
        &'a self,
        frame: &'a SurfaceForwardFrame<B>,
        image: &'a SurfaceImage<B>,
    ) -> Result<FrameBindings<'a, B>> {
        let mut buffers = vec![
            BoundBuffer::new(
                self.frame_upload_slot,
                frame.frame.upload_instance,
                &frame.frame.upload,
                &frame.frame.upload_desc,
                frame.frame.upload_incoming,
            ),
            BoundBuffer::new(
                self.uniform_slot,
                frame.frame.uniform_instance,
                &frame.frame.uniform,
                &frame.frame.uniform_desc,
                frame.frame.uniform_incoming,
            ),
        ];
        let uses_buffer = |slot| {
            self.execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|requirement| requirement.slot == slot)
        };
        if uses_buffer(self.resident_upload_slot) {
            buffers.push(BoundBuffer::new(
                self.resident_upload_slot,
                frame.resident.upload_instance,
                &frame.resident.upload,
                &frame.resident.upload_desc,
                frame.resident.upload_incoming,
            ));
        }
        if let (Some(slot), Some(owner)) =
            (self.environment_upload_slot, frame.environment.as_ref())
        {
            buffers.push(BoundBuffer::new(
                slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            ));
        }
        if let (Some(upload_slot), Some(uniform_slot), Some(owner)) = (
            self.specular_upload_slot,
            self.specular_view_uniform_slot,
            frame.specular.as_ref(),
        ) {
            buffers.push(BoundBuffer::new(
                upload_slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            ));
            buffers.push(BoundBuffer::new(
                uniform_slot,
                owner.view_uniform_instance,
                &owner.view_uniform,
                &owner.view_uniform_desc,
                owner.view_uniform_incoming,
            ));
        }
        for (slot, stream) in self
            .stream_slots
            .iter()
            .copied()
            .zip(&frame.resident.streams)
        {
            if uses_buffer(slot) {
                buffers.push(BoundBuffer::new(
                    slot,
                    stream.instance,
                    &stream.buffer,
                    &stream.desc,
                    stream.incoming,
                ));
            }
        }
        let lease = BoundSurfaceTexture::new(self.surface_target_slot, image)?;
        FrameBindings::new_with_surface_textures(
            &self.execution,
            self.id.device,
            buffers,
            self.texture_slots
                .iter()
                .copied()
                .zip(&frame.resident.textures)
                .map(|(slot, texture)| {
                    BoundTexture::new(
                        slot,
                        texture.instance,
                        &texture.texture,
                        &texture.desc,
                        texture.incoming,
                        texture.layout,
                    )
                })
                .chain(self.environment_slot.zip(frame.environment.as_ref()).map(
                    |(slot, owner)| {
                        BoundTexture::new(
                            slot,
                            owner.texture_instance,
                            &owner.texture,
                            &owner.desc,
                            owner.incoming,
                            owner.layout,
                        )
                    },
                ))
                .chain(
                    self.specular_radiance_slot
                        .zip(frame.specular.as_ref())
                        .map(|(slot, owner)| {
                            BoundTexture::new(
                                slot,
                                owner.radiance.instance,
                                &owner.radiance.texture,
                                &owner.radiance.desc,
                                owner.radiance.incoming,
                                owner.radiance.layout,
                            )
                        }),
                )
                .chain(
                    self.specular_brdf_lut_slot
                        .zip(frame.specular.as_ref())
                        .map(|(slot, owner)| {
                            BoundTexture::new(
                                slot,
                                owner.brdf_lut.instance,
                                &owner.brdf_lut.texture,
                                &owner.brdf_lut.desc,
                                owner.brdf_lut.incoming,
                                owner.brdf_lut.layout,
                            )
                        }),
                )
                .chain(
                    self.depth_slot
                        .zip(frame.frame.depth.as_ref())
                        .map(|(slot, owner)| {
                            BoundTexture::new(
                                slot,
                                owner.instance,
                                &owner.texture,
                                &owner.desc,
                                owner.incoming,
                                owner.layout,
                            )
                        }),
                )
                .collect(),
            vec![lease],
        )
        .map_err(binding_error)
    }
}

impl<B: ExecutionBackend + ResourceBackend> SurfaceForwardPlan<B> {
    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>> {
        let expected = [self.upload_step(), self.raster_step(), self.present_step];
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(&expected)
            || !facts.presents.is_some_and(|presents| {
                presents.len() == 1
                    && presents[0].accepted
                    && presents[0].step == Some(self.present_step)
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
                            && point
                                .covers
                                .lane_frontiers
                                .iter()
                                .any(|&(lane, step)| lane == self.lane && step == self.present_step)
                    })
                    .map(|point| point.point.clone())
            })
            .ok_or(Error::InvalidState)
    }
}

#[cfg(test)]
#[path = "../../tests/graph/forward_surface_plan.rs"]
mod test_helpers;
