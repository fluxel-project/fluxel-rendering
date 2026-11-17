//! Bind both frame namespaces to their actual owners, with one capture target.
use super::*;

impl<B: ResourceBackend> CaptureForwardPlan<B> {
    pub(crate) fn validate_frames(
        &self,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
    ) -> Result<()> {
        for (frame, shape, topology) in [
            (capture, &self.capture_shape, self.capture_topology),
            (receiver, &self.receiver_shape, self.receiver_topology),
        ] {
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
        validate_capture_alias(capture, receiver)?;
        self.bindings(capture, receiver).map(|_| ())
    }

    pub(super) fn bindings<'a>(
        &'a self,
        capture: &'a ForwardFrame<B>,
        receiver: &'a ForwardFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        validate_capture_alias(capture, receiver)?;
        let mut buffers = Vec::new();
        let mut textures = Vec::new();
        for (slots, frame, capture_side) in [
            (&self.capture, capture, true),
            (&self.receiver, receiver, false),
        ] {
            super::capture_specular_bindings::append(
                slots,
                frame.specular.as_ref(),
                self.id.device,
                &mut buffers,
                &mut textures,
            )?;
            let mut push_buffer = |slot, instance, buffer, desc, incoming| {
                if self
                    .execution
                    .logical()
                    .buffer_bindings()
                    .requirements()
                    .iter()
                    .any(|requirement| requirement.slot == slot)
                {
                    buffers.push(BoundBuffer::new(slot, instance, buffer, desc, incoming));
                }
            };
            push_buffer(
                slots.resident_upload,
                frame.resident.upload_instance,
                &frame.resident.upload,
                &frame.resident.upload_desc,
                frame.resident.upload_incoming,
            );
            push_buffer(
                slots.frame_upload,
                frame.frame.upload_instance,
                &frame.frame.upload,
                &frame.frame.upload_desc,
                frame.frame.upload_incoming,
            );
            push_buffer(
                slots.uniform,
                frame.frame.uniform_instance,
                &frame.frame.uniform,
                &frame.frame.uniform_desc,
                frame.frame.uniform_incoming,
            );
            push_buffer(
                slots.readback,
                frame.frame.readback_instance,
                &frame.frame.readback,
                &frame.frame.readback_desc,
                frame.frame.readback_incoming,
            );
            for (slot, stream) in slots.streams.iter().copied().zip(&frame.resident.streams) {
                push_buffer(
                    slot,
                    stream.instance,
                    &stream.buffer,
                    &stream.desc,
                    stream.incoming,
                );
            }
            if let (Some(slot), Some(owner)) =
                (slots.environment_upload, frame.environment.as_ref())
            {
                push_buffer(
                    slot,
                    owner.upload_instance,
                    &owner.upload,
                    &owner.upload_desc,
                    owner.upload_incoming,
                );
            }
            if capture_side {
                for (slot, texture) in slots.textures.iter().copied().zip(&frame.resident.textures)
                {
                    textures.push(BoundTexture::new(
                        slot,
                        texture.instance,
                        &texture.texture,
                        &texture.desc,
                        texture.incoming,
                        texture.layout,
                    ));
                }
            }
            textures.push(BoundTexture::new(
                slots.target,
                frame.frame.target_instance,
                &frame.frame.target,
                &frame.frame.target_desc,
                frame.frame.target_incoming,
                frame.frame.target_layout,
            ));
            if let (Some(slot), Some(owner)) = (slots.environment, frame.environment.as_ref()) {
                textures.push(BoundTexture::new(
                    slot,
                    owner.texture_instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout,
                ));
            }
            if let (Some(slot), Some(owner)) = (slots.depth, frame.frame.depth.as_ref()) {
                textures.push(BoundTexture::new(
                    slot,
                    owner.instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout,
                ));
            }
        }
        FrameBindings::new(&self.execution, self.id.device, buffers, textures)
            .map_err(binding_error)
    }
}

pub(super) fn validate_capture_alias<B: ResourceBackend>(
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
) -> Result<()> {
    let alias = receiver
        .resident
        .textures
        .first()
        .ok_or(Error::InvalidState)?;
    if receiver.resident.textures.len() != 1
        || !receiver.resident_upload.textures.is_empty()
        || alias.instance != capture.frame.target_instance
        || alias.texture.device() != capture.frame.target.device()
        || alias.view.device() != capture.frame.target_view.device()
        || !capture_declaration::same_texture_owner_desc(&alias.desc, &capture.frame.target_desc)
        || alias.incoming != capture.frame.target_incoming
        || alias.layout != capture.frame.target_layout
        || !alias.filterable
        || capture.frame.target_instance == receiver.frame.target_instance
        || capture.extent != receiver.extent
        || !alias.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}
