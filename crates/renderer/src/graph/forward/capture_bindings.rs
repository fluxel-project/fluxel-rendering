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
        validate_capture_alias(capture, receiver)?;
        self.bindings(capture, receiver).map(|_| ())
    }

    pub(super) fn bindings<'a>(
        &'a self,
        capture: &'a ForwardFrame<B>,
        receiver: &'a ForwardFrame<B>,
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
        FrameBindings::new(&self.execution, self.id.device, buffers, textures)
            .map_err(binding_error)
    }
}

pub(super) fn validate_capture_alias<B: ResourceBackend>(
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
) -> Result<()> {
    let expected =
        capture_declaration::capture_texture_slots(&ForwardGraphSpec::from_frame(receiver)?)?;
    if receiver
        .resident_upload
        .textures
        .iter()
        .any(|upload| expected.get(upload.texture) == Some(&true))
        || receiver
            .resident
            .textures
            .iter()
            .zip(expected)
            .any(|(alias, capture_texture)| {
                capture_texture
                    && (alias.instance != capture.frame.target_instance
                        || alias.texture.device() != capture.frame.target.device()
                        || alias.view.device() != capture.frame.target_view.device()
                        || !capture_declaration::same_texture_owner_desc(
                            &alias.desc,
                            &capture.frame.target_desc,
                        )
                        || alias.incoming != capture.frame.target_incoming
                        || alias.layout != capture.frame.target_layout
                        || !alias.desc.usage.contains(TextureUsage::TEXTURE_BINDING))
                    || (!capture_texture && alias.instance == capture.frame.target_instance)
            })
        || capture.frame.target_instance == receiver.frame.target_instance
        || capture.extent != receiver.extent
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}
