//! Offscreen Plan for the bounded Forward product.
use super::*;

impl<B: ResourceBackend> ForwardPlan<B> {
    pub(crate) fn declare_with<F>(
        device: &Device<B>,
        frame: &ForwardFrame<B>,
        expected: FrameTopology,
        contribute: F,
    ) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
    where
        F: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
    {
        declare_offscreen_with(device, frame, expected, contribute)
    }

    pub(crate) fn validate_declaration(
        &self,
        declaration: &OffscreenForwardDeclaration,
    ) -> Result<()> {
        if !self.declaration.same_declaration(&declaration.graph)
            || self.lane != declaration.lane
            || self.upload_pass.1 != declaration.upload_pass.1
            || self.raster_pass.1 != declaration.raster_pass.1
            || self.readback_pass.1 != declaration.readback_pass.1
            || self.resident_upload_slot.resource != declaration.resident_upload_slot.resource
            || self.frame_upload_slot.resource != declaration.frame_upload_slot.resource
            || self.uniform_slot.resource != declaration.uniform_slot.resource
            || !same_buffer_slots(&self.stream_slots, &declaration.stream_slots)
            || !same_texture_slots(&self.texture_slots, &declaration.texture_slots)
            || self.target_slot.resource != declaration.target_slot.resource
            || self.readback_slot.resource != declaration.readback_slot.resource
            || self.draw_count != declaration.draw_count
            || self.stream_count != declaration.stream_count
            || self.texture_count != declaration.texture_count
            || self.extent != declaration.extent
            || self.shape != declaration.shape
            || self.uniform_alignment != declaration.uniform_alignment
            || self.topology != declaration.topology
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }

    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: OffscreenForwardDeclaration,
    ) -> std::result::Result<Self, ForwardPlanBuildError> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice.into());
        }
        let execution = CompiledExecution::compile(&declaration.graph)?;
        if execution.logical().passes()
            != [
                declaration.upload_pass,
                declaration.raster_pass,
                declaration.readback_pass,
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
            declaration: declaration.graph.clone(),
            upload_pass: declaration.upload_pass,
            raster_pass: declaration.raster_pass,
            readback_pass: declaration.readback_pass,
            resident_upload_slot: declaration.resident_upload_slot,
            frame_upload_slot: declaration.frame_upload_slot,
            uniform_slot: declaration.uniform_slot,
            stream_slots: declaration.stream_slots,
            texture_slots: declaration.texture_slots,
            target_slot: declaration.target_slot,
            readback_slot: declaration.readback_slot,
            draw_count: declaration.draw_count,
            stream_count: declaration.stream_count,
            texture_count: declaration.texture_count,
            extent: declaration.extent,
            shape: declaration.shape,
            uniform_alignment: declaration.uniform_alignment,
            topology: declaration.topology,
        })
    }

    pub(crate) fn id(&self) -> PlanId {
        self.id
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
    pub(crate) fn readback_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 2,
        }
    }

    /// Rejects a frame whose structural shape cannot use this plan.
    ///
    /// This bounded initial plan declares exact byte ranges, so callers must
    /// include all declared ranges and layout facts in their cache key. Resource
    /// identity itself is deliberately absent from that key; `FrameBindings`
    /// validates each incoming owner independently.
    pub(crate) fn validate_frame(&self, frame: &ForwardFrame<B>) -> Result<()> {
        if frame.resident.device != self.id.device
            || frame.extent != self.extent
            || frame.draws.len() != self.draw_count
            || frame.resident.streams.len() != self.stream_count
            || frame.resident.textures.len() != self.texture_count
            || self.shape != forward_shape(frame)
            || frame
                .draws
                .iter()
                .any(|draw| validate_draw(draw, self.stream_count).is_err())
        {
            return Err(Error::InvalidState);
        }
        validate_forward_topology(self.topology, frame.draws.len(), frame.draw_batches.len())?;
        validate_uploads(frame)?;
        validate_batches(frame)?;
        for draw in &frame.draws {
            validate_offsets(draw, self.uniform_alignment)?;
            if draw.texture >= self.texture_count {
                return Err(Error::InvalidRange);
            }
            draw_accesses(frame, draw)?;
        }
        self.validate_bindings(frame)
    }

    pub(super) fn bindings<'a>(
        &'a self,
        frame: &'a ForwardFrame<B>,
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
            if !uses_buffer(slot) {
                continue;
            }
            buffers.push(BoundBuffer::new(
                slot,
                stream.instance,
                &stream.buffer,
                &stream.desc,
                stream.incoming,
            ));
        }
        buffers.push(BoundBuffer::new(
            self.readback_slot,
            frame.frame.readback_instance,
            &frame.frame.readback,
            &frame.frame.readback_desc,
            frame.frame.readback_incoming,
        ));
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
                .chain(std::iter::once(BoundTexture::new(
                    self.target_slot,
                    frame.frame.target_instance,
                    &frame.frame.target,
                    &frame.frame.target_desc,
                    frame.frame.target_incoming,
                    frame.frame.target_layout,
                )))
                .collect(),
        )
        .map_err(binding_error)
    }

    pub(crate) fn validate_bindings(&self, frame: &ForwardFrame<B>) -> Result<()> {
        self.bindings(frame).map(|_| ())
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let expected = [self.upload_step(), self.raster_step(), self.readback_step()];
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
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
                                lane == self.lane && step == self.readback_step()
                            })
                    })
                    .map(|point| point.point.clone())
            })
            .ok_or(Error::InvalidState)
    }
}

#[cfg(test)]
#[path = "../../tests/graph/forward_offscreen_plan.rs"]
mod test_helpers;
