//! Actual-owner checks for the one permitted Deferred capture alias.

use super::*;

struct BorrowedTextureOwner<'a, B: ResourceBackend> {
    instance: crate::graph::ResourceInstanceId,
    texture: &'a fluxel_rhi::api::Texture<B>,
    view: &'a fluxel_rhi::api::TextureView<B>,
    desc: &'a fluxel_rhi::api::TextureDesc,
    incoming: TextureUses,
    layout: ImageLayout,
    device: fluxel_rhi::api::DeviceId,
}

pub(super) fn validate_capture_owners<B: ResourceBackend>(
    producer: &DeferredFrame<B>,
    overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
    receiver: &DeferredFrame<B>,
) -> Result<()> {
    for frame in [producer, receiver] {
        let device = frame.resident.device;
        if frame.frame.target.device() != device || frame.frame.target_view.device() != device {
            return Err(Error::WrongDevice);
        }
        for owner in frame
            .gbuffer
            .iter()
            .chain(frame.effects.iter())
            .chain([&frame.scene_color])
        {
            if owner.device != device
                || owner.texture.device() != device
                || owner.view.device() != device
            {
                return Err(Error::WrongDevice);
            }
        }
    }
    if producer
        .draws
        .iter()
        .any(|draw| draw.capture_textures.iter().any(|capture| *capture))
    {
        return Err(Error::InvalidState);
    }
    validate_overlay_owners(producer, overlay)?;
    let product = match overlay.and_then(|overlay| overlay.oit.as_ref()) {
        Some(input) => BorrowedTextureOwner {
            instance: input.output.instance,
            texture: &input.output.texture,
            view: &input.output.view,
            desc: &input.output.desc,
            incoming: input.output.incoming,
            layout: input.output.layout,
            device: input.output.device,
        },
        None => BorrowedTextureOwner {
            instance: producer.scene_color.instance,
            texture: &producer.scene_color.texture,
            view: &producer.scene_color.view,
            desc: &producer.scene_color.desc,
            incoming: producer.scene_color.incoming,
            layout: producer.scene_color.layout,
            device: producer.scene_color.device,
        },
    };
    let mut captures = Vec::new();
    let mut roles = vec![None; receiver.resident.textures.len()];
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
            let texture = receiver
                .resident
                .textures
                .get(index)
                .ok_or(Error::InvalidRange)?;
            if capture {
                if texture.instance != product.instance
                    || texture.device != product.device
                    || texture.texture.device() != product.texture.device()
                    || texture.view.device() != product.view.device()
                    || !crate::graph::forward::same_texture_owner_desc(&texture.desc, &product.desc)
                    || texture.incoming != product.incoming
                    || texture.layout != product.layout
                    || !texture.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
                {
                    return Err(Error::InvalidState);
                }
                captures.push(index);
            } else if texture.instance == product.instance {
                return Err(Error::InvalidState);
            }
        }
    }
    if captures.is_empty()
        || receiver
            .resident_upload
            .textures
            .iter()
            .any(|copy| captures.contains(&copy.texture))
        || producer
            .resident
            .textures
            .iter()
            .any(|texture| texture.instance == product.instance)
        || receiver.frame.target_instance == product.instance
    {
        return Err(Error::InvalidState);
    }
    let receiver_samples = receiver
        .resident
        .textures
        .iter()
        .map(|owner| owner.instance)
        .chain(
            receiver
                .environment
                .iter()
                .map(|owner| owner.texture_instance),
        )
        .chain(
            receiver
                .specular
                .iter()
                .flat_map(|owner| [owner.radiance.instance, owner.brdf_lut.instance]),
        )
        .collect::<Vec<_>>();
    if receiver
        .environment
        .iter()
        .any(|owner| owner.texture_instance == product.instance)
        || receiver.specular.iter().any(|owner| {
            owner.radiance.instance == product.instance
                || owner.brdf_lut.instance == product.instance
        })
    {
        return Err(Error::InvalidState);
    }
    let mut producer_attachments = producer
        .gbuffer
        .iter()
        .map(|owner| owner.instance)
        .chain(producer.effects.iter().map(|owner| owner.instance))
        .chain([
            producer.scene_color.instance,
            product.instance,
            producer.frame.target_instance,
        ]);
    if producer_attachments
        .any(|instance| instance != product.instance && receiver_samples.contains(&instance))
    {
        return Err(Error::InvalidState);
    }
    let mut receiver_attachments = receiver
        .gbuffer
        .iter()
        .map(|owner| owner.instance)
        .chain(receiver.effects.iter().map(|owner| owner.instance))
        .chain([
            receiver.scene_color.instance,
            receiver.frame.target_instance,
        ]);
    if receiver_attachments.any(|instance| instance == product.instance) {
        return Err(Error::InvalidState);
    }
    let producer_samples = producer
        .resident
        .textures
        .iter()
        .map(|owner| owner.instance)
        .chain(
            producer
                .environment
                .iter()
                .map(|owner| owner.texture_instance),
        )
        .chain(
            producer
                .specular
                .iter()
                .flat_map(|owner| [owner.radiance.instance, owner.brdf_lut.instance]),
        )
        .collect::<Vec<_>>();
    let mut receiver_attachments = receiver
        .gbuffer
        .iter()
        .map(|owner| owner.instance)
        .chain(receiver.effects.iter().map(|owner| owner.instance))
        .chain([
            receiver.scene_color.instance,
            receiver.frame.target_instance,
        ]);
    if receiver_attachments.any(|instance| producer_samples.contains(&instance)) {
        return Err(Error::InvalidState);
    }
    Ok(())
}

pub(super) fn validate_overlay_owners<B: ResourceBackend>(
    producer: &DeferredFrame<B>,
    overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
) -> Result<()> {
    let Some(overlay) = overlay else {
        return Ok(());
    };
    let device = producer.resident.device;
    if overlay.frame.resident.device != device || overlay.frame.extent != producer.extent {
        return Err(Error::InvalidState);
    }
    let depth = overlay
        .frame
        .frame
        .depth
        .as_ref()
        .ok_or(Error::InvalidState)?;
    if !same_attachment(
        overlay.frame.frame.target_instance,
        &overlay.frame.frame.target,
        &overlay.frame.frame.target_desc,
        overlay.frame.frame.target_incoming,
        overlay.frame.frame.target_layout,
        producer.scene_color.instance,
        &producer.scene_color.texture,
        &producer.scene_color.desc,
        producer.scene_color.incoming,
        producer.scene_color.layout,
    ) || !same_attachment(
        depth.instance,
        &depth.texture,
        &depth.desc,
        depth.incoming,
        depth.layout,
        producer.gbuffer[2].instance,
        &producer.gbuffer[2].texture,
        &producer.gbuffer[2].desc,
        producer.gbuffer[2].incoming,
        producer.gbuffer[2].layout,
    ) || overlay.frame.frame.target.device() != device
        || overlay.frame.frame.target_view.device() != device
        || depth.device != device
        || depth.texture.device() != device
        || depth.view.device() != device
    {
        return Err(Error::InvalidState);
    }
    if let Some(input) = overlay.oit.as_ref() {
        validate_oit_input(device, producer, overlay.frame, input)?;
    }
    Ok(())
}

fn same_attachment<B: ResourceBackend>(
    left_instance: crate::graph::ResourceInstanceId,
    left_texture: &fluxel_rhi::api::Texture<B>,
    left_desc: &fluxel_rhi::api::TextureDesc,
    left_incoming: TextureUses,
    left_layout: ImageLayout,
    right_instance: crate::graph::ResourceInstanceId,
    right_texture: &fluxel_rhi::api::Texture<B>,
    right_desc: &fluxel_rhi::api::TextureDesc,
    right_incoming: TextureUses,
    right_layout: ImageLayout,
) -> bool {
    left_instance == right_instance
        && left_texture.device() == right_texture.device()
        && crate::graph::forward::same_texture_owner_desc(left_desc, right_desc)
        && left_incoming == right_incoming
        && left_layout == right_layout
}

fn validate_oit_input<B: ResourceBackend>(
    device: fluxel_rhi::api::DeviceId,
    producer: &DeferredFrame<B>,
    overlay: &crate::residency::ForwardFrame<B>,
    input: &crate::graph::ForwardOitInput<'_, B>,
) -> Result<()> {
    if input.opaque_count != 0
        || input.peel_count == 0
        || input.peel_count > overlay.draws.len()
        || input.rounds.get() == 0
        || input.owners.extent != producer.extent
    {
        return Err(Error::InvalidState);
    }
    input.owners.validate(device)?;
    let output = input.output;
    if output.device != device
        || output.texture.device() != device
        || output.view.device() != device
        || output.desc.dimension != fluxel_rhi::api::TextureDimension::D2
        || output.desc.extent != [producer.extent[0], producer.extent[1], 1]
        || output.desc.array_layers != 1
        || output.desc.mip_levels != 1
        || output.desc.samples != 1
        || output.desc.format != producer.scene_color.desc.format
        || !output.desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
        || !output.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        || output.view_desc.format != output.desc.format
        || output.view_desc.dimension != fluxel_rhi::api::TextureViewDimension::D2
        || output.view_desc.aspects != TextureAspects::COLOR
        || output.view_desc.mips != (0..1)
        || output.view_desc.layers != (0..1)
    {
        return Err(Error::InvalidState);
    }
    let mut owners = Vec::with_capacity(8);
    for side in &input.owners.ping_pong {
        owners.extend([
            side.min_max.instance,
            side.front.instance,
            side.back.instance,
        ]);
    }
    owners.push(input.owners.back_accumulator.instance);
    owners.push(output.instance);
    owners.extend([
        producer.scene_color.instance,
        producer.gbuffer[0].instance,
        producer.gbuffer[1].instance,
        producer.gbuffer[2].instance,
    ]);
    owners.extend(
        producer
            .resident
            .textures
            .iter()
            .map(|owner| owner.instance),
    );
    owners.extend(overlay.resident.textures.iter().map(|owner| owner.instance));
    owners.extend(
        producer
            .environment
            .iter()
            .map(|owner| owner.texture_instance),
    );
    owners.extend(
        overlay
            .environment
            .iter()
            .map(|owner| owner.texture_instance),
    );
    owners.extend(
        producer
            .specular
            .iter()
            .flat_map(|owner| [owner.radiance.instance, owner.brdf_lut.instance]),
    );
    owners.extend(
        overlay
            .specular
            .iter()
            .flat_map(|owner| [owner.radiance.instance, owner.brdf_lut.instance]),
    );
    if owners
        .iter()
        .enumerate()
        .any(|(index, instance)| owners[..index].contains(instance))
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}
