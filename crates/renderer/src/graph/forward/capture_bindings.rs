//! Bind both frame namespaces to their actual owners, with one capture target.
use super::*;
use crate::graph::ResourceInstanceId;
use fluxel_rhi::api::{Texture, TextureView};

/// Borrowed actual producer product used by capture alias validation.
///
/// It carries Graph instance/state facts alongside native handles; cloned RHI
/// handles are therefore accepted when they name the same actual product.
pub(super) struct CaptureProductOwner<'a, B: ResourceBackend> {
    pub(super) texture: &'a Texture<B>,
    pub(super) view: &'a TextureView<B>,
    pub(super) desc: &'a TextureDesc,
    pub(super) instance: ResourceInstanceId,
    pub(super) incoming: TextureUses,
    pub(super) layout: ImageLayout,
    pub(super) extent: [u32; 2],
}

impl<'a, B: ResourceBackend> CaptureProductOwner<'a, B> {
    pub(super) fn from_forward_frame(frame: &'a ForwardFrame<B>) -> Self {
        Self {
            texture: &frame.frame.target,
            view: &frame.frame.target_view,
            desc: &frame.frame.target_desc,
            instance: frame.frame.target_instance,
            incoming: frame.frame.target_incoming,
            layout: frame.frame.target_layout,
            extent: frame.extent,
        }
    }
}

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
        validate_capture_alias(capture, receiver, self.receiver_depth_alias)?;
        self.bindings(capture, receiver).map(|_| ())
    }

    pub(super) fn bindings<'a>(
        &'a self,
        capture: &'a ForwardFrame<B>,
        receiver: &'a ForwardFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        validate_resident_texture_devices(&capture.resident, self.id.device)?;
        validate_resident_texture_devices(&receiver.resident, self.id.device)?;
        validate_capture_alias(capture, receiver, self.receiver_depth_alias)?;
        let (buffers, mut textures) = super::capture_binding_collection::collect(
            &self.execution,
            self.id.device,
            &self.capture,
            capture,
            &self.receiver,
            receiver,
            self.receiver_depth_alias,
            self.receiver_depth_alias,
        )?;
        if let Some(terminal) = self.hdr_terminal() {
            terminal.append_binding(
                receiver.frame.output.as_ref().ok_or(Error::InvalidState)?,
                &mut textures,
            );
        }
        FrameBindings::new(&self.execution, self.id.device, buffers, textures)
            .map_err(binding_error)
    }
}

pub(super) fn validate_capture_alias<B: ResourceBackend>(
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    scene_refraction: bool,
) -> Result<()> {
    validate_capture_product_alias(
        &CaptureProductOwner::from_forward_frame(capture),
        receiver,
        scene_refraction,
    )
}

/// Validates receiver aliases against a real producer product (ordinary or
/// OIT final output) without requiring a CPU-uploaded copy.
pub(super) fn validate_capture_product_alias<B: ResourceBackend>(
    product: &CaptureProductOwner<'_, B>,
    receiver: &ForwardFrame<B>,
    scene_refraction: bool,
) -> Result<()> {
    let expected =
        capture_declaration::capture_texture_slots(&ForwardGraphSpec::from_frame(receiver)?)?;
    if receiver.resident.textures.len() != expected.len() {
        return Err(Error::InvalidState);
    }
    let mut refraction_alias = None;
    if receiver
        .resident_upload
        .textures
        .iter()
        .any(|upload| expected.get(upload.texture) == Some(&true))
        || receiver
            .resident
            .textures
            .iter()
            .zip(expected.iter().copied())
            .enumerate()
            .any(|(index, (alias, capture_texture))| {
                if !capture_texture {
                    return alias.instance == product.instance;
                }
                if alias.desc.mip_levels == 1 {
                    return alias.instance != product.instance
                        || alias.texture.device() != product.texture.device()
                        || alias.view.device() != product.view.device()
                        || !capture_declaration::same_texture_owner_desc(&alias.desc, product.desc)
                        || alias.incoming != product.incoming
                        || alias.layout != product.layout
                        || !alias.desc.usage.contains(TextureUsage::TEXTURE_BINDING);
                }
                if !scene_refraction || refraction_alias.replace(index).is_some() {
                    return true;
                }
                let expected = crate::renderer::refraction_capture_desc(product.desc);
                alias.instance == product.instance
                    || alias.texture.device() != product.texture.device()
                    || alias.view.device() != product.view.device()
                    || !capture_declaration::same_texture_owner_desc(&alias.desc, &expected)
                    || alias.incoming
                        != TextureUses {
                            stages: StageMask::EMPTY,
                            accesses: AccessMask::EMPTY,
                        }
                    || alias.layout != ImageLayout::Undefined
                    || !alias.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
            })
        || product.instance == receiver.frame.target_instance
        || product.extent != receiver.extent
    {
        return Err(Error::InvalidState);
    }
    if let Some(index) = refraction_alias {
        let instance = receiver
            .resident
            .textures
            .get(index)
            .ok_or(Error::InvalidState)?
            .instance;
        if receiver
            .resident
            .textures
            .iter()
            .enumerate()
            .any(|(other, texture)| other != index && texture.instance == instance)
        {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}
