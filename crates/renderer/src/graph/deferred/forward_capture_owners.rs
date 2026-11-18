//! Producer product identity for an independently allocated Forward receiver.

use super::*;
use crate::{graph::ForwardOitInput, residency::ForwardFrame};

pub(super) fn validate_owners<B: ResourceBackend>(
    producer: &DeferredFrame<B>,
    overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
    receiver: &ForwardFrame<B>,
    receiver_oit: Option<&ForwardOitInput<'_, B>>,
    scene_refraction: bool,
) -> Result<()> {
    super::capture_owner_validation::validate_overlay_owners(producer, overlay)?;
    let (instance, texture, view, desc, incoming, layout) =
        match overlay.and_then(|o| o.oit.as_ref()) {
            Some(input) => (
                input.output.instance,
                &input.output.texture,
                &input.output.view,
                &input.output.desc,
                input.output.incoming,
                input.output.layout,
            ),
            None => (
                producer.scene_color.instance,
                &producer.scene_color.texture,
                &producer.scene_color.view,
                &producer.scene_color.desc,
                producer.scene_color.incoming,
                producer.scene_color.layout,
            ),
        };
    let device = producer.resident.device;
    if receiver.resident.device != device
        || producer.extent != receiver.extent
        || receiver.frame.target_instance == instance
        || receiver.frame.target.device() != device
        || receiver.frame.target_view.device() != device
        || producer
            .draws
            .iter()
            .any(|draw| draw.capture_textures.iter().any(|&capture| capture))
    {
        return Err(Error::InvalidState);
    }
    if scene_refraction
        && (receiver_oit.is_some() || desc.format != receiver.frame.target_desc.format)
    {
        return Err(Error::InvalidState);
    }
    let depth = receiver.frame.depth.as_ref().ok_or(Error::InvalidState)?;
    if depth.device != device
        || depth.instance == instance
        || depth.texture.device() != device
        || depth.view.device() != device
        || (scene_refraction
            && (depth.instance != producer.gbuffer[2].instance
                || depth.incoming != producer.gbuffer[2].incoming
                || depth.layout != producer.gbuffer[2].layout
                || !crate::graph::forward::same_texture_owner_desc(
                    &depth.desc,
                    &producer.gbuffer[2].desc,
                )))
    {
        return Err(Error::InvalidState);
    }
    let mut roles = vec![None; receiver.resident.textures.len()];
    let mut captures = 0;
    for draw in &receiver.draws {
        if draw.textures.len() != draw.capture_textures.len() {
            return Err(Error::InvalidRange);
        }
        for (&index, &capture) in draw.textures.iter().zip(&draw.capture_textures) {
            let role = roles.get_mut(index).ok_or(Error::InvalidRange)?;
            if role.is_some_and(|previous| previous != capture) {
                return Err(Error::InvalidState);
            }
            *role = Some(capture);
            let owner = receiver
                .resident
                .textures
                .get(index)
                .ok_or(Error::InvalidRange)?;
            if capture {
                if owner.instance != instance
                    || owner.texture.device() != texture.device()
                    || owner.view.device() != view.device()
                    || owner.device != device
                    || !crate::graph::forward::same_texture_owner_desc(&owner.desc, desc)
                    || owner.incoming != incoming
                    || owner.layout != layout
                    || !owner.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
                    || receiver
                        .resident_upload
                        .textures
                        .iter()
                        .any(|upload| upload.texture == index)
                {
                    return Err(Error::InvalidState);
                }
                captures += 1;
            } else if owner.instance == instance {
                return Err(Error::InvalidState);
            }
        }
    }
    if captures == 0 {
        return Err(Error::InvalidState);
    }
    if let Some(input) = receiver_oit {
        input.owners.validate(device)?;
        let output = input.output;
        if output.instance == instance
            || output.device != device
            || output.texture.device() != device
            || output.view.device() != device
            || output.desc.extent != [receiver.extent[0], receiver.extent[1], 1]
            || output.desc.format != receiver.frame.target_desc.format
            || output.desc.mip_levels != 1
            || output.desc.array_layers != 1
            || output.desc.samples != 1
            || output.desc.dimension != fluxel_rhi::api::TextureDimension::D2
            || !output.desc.usage.contains(
                TextureUsage::TEXTURE_BINDING
                    | TextureUsage::RENDER_ATTACHMENT
                    | TextureUsage::COPY_SRC,
            )
            || output.view_desc.format != output.desc.format
            || output.view_desc.dimension != fluxel_rhi::api::TextureViewDimension::D2
            || output.view_desc.aspects != TextureAspects::COLOR
            || output.view_desc.mips != (0..1)
            || output.view_desc.layers != (0..1)
        {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}
