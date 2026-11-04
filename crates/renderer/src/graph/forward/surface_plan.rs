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
            || !same_buffer_slots(&self.stream_slots, &declaration.stream_slots)
            || !same_texture_slots(&self.texture_slots, &declaration.texture_slots)
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
            stream_slots: declaration.stream_slots,
            texture_slots: declaration.texture_slots,
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
        validate_surface_batches(frame)?;
        for draw in &frame.draws {
            validate_draw(draw, self.stream_count)?;
            validate_offsets(draw, self.uniform_alignment)?;
            surface_draw_accesses(frame, draw)?;
            if draw.texture >= self.texture_count {
                return Err(Error::InvalidRange);
            }
        }
        self.bindings(frame).map(|_| ())
    }

    pub(super) fn bindings<'a>(
        &'a self,
        frame: &'a SurfaceForwardFrame<B>,
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
        FrameBindings::new(
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
                .collect(),
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
