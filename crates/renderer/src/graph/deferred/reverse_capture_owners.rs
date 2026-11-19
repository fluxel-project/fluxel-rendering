//! Actual ownership checks for Forward-to-Deferred capture aliases.

use super::*;
use crate::{graph::ForwardOitInput, residency::ForwardFrame};

pub(super) fn validate_owners<B: ResourceBackend>(
    producer: &ForwardFrame<B>,
    producer_oit: Option<&ForwardOitInput<'_, B>>,
    receiver: &DeferredFrame<B>,
) -> Result<()> {
    let device = producer.resident.device;
    if receiver.resident.device != device
        || producer.extent != receiver.extent
        || receiver.deformation.is_some()
    {
        return Err(Error::InvalidState);
    }
    let (instance, desc, incoming, layout) = match producer_oit {
        Some(input) => (
            input.output.instance,
            &input.output.desc,
            input.output.incoming,
            input.output.layout,
        ),
        None => (
            producer.frame.target_instance,
            &producer.frame.target_desc,
            producer.frame.target_incoming,
            producer.frame.target_layout,
        ),
    };
    if producer.frame.target.device() != device
        || producer.frame.target_view.device() != device
        || receiver.frame.target.device() != device
        || receiver.frame.target_view.device() != device
    {
        return Err(Error::WrongDevice);
    }
    let mut roles = vec![None; receiver.resident.textures.len()];
    let mut captures = 0usize;
    for draw in &receiver.draws {
        if draw.textures.len() != draw.capture_textures.len() {
            return Err(Error::InvalidRange);
        }
        for (&index, &capture) in draw.textures.iter().zip(&draw.capture_textures) {
            let role = roles.get_mut(index).ok_or(Error::InvalidRange)?;
            if role.is_some_and(|old| old != capture) {
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
                    || owner.device != device
                    || owner.texture.device() != device
                    || owner.view.device() != device
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
    if captures == 0
        || receiver.frame.target_instance == instance
        || receiver.scene_color.instance == instance
        || receiver
            .gbuffer
            .iter()
            .any(|owner| owner.instance == instance)
        || receiver
            .effects
            .as_ref()
            .is_some_and(|owner| owner.instance == instance)
        || receiver
            .diffuse_roughness
            .as_ref()
            .is_some_and(|owner| owner.instance == instance)
    {
        return Err(Error::InvalidState);
    }
    if let Some(input) = producer_oit {
        input.owners.validate(device)?;
        if input.opaque_count > producer.draws.len()
            || input.peel_count == 0
            || input
                .opaque_count
                .checked_add(input.peel_count)
                .ok_or(Error::InvalidRange)?
                > producer.draws.len()
        {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}
