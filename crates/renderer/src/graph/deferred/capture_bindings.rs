//! Combined concrete binding collection for Deferred capture namespaces.

use super::*;
use std::collections::HashMap;

type BufferOwner = (
    crate::graph::ResourceInstanceId,
    fluxel_rhi::api::DeviceId,
    fluxel_rhi::api::BufferDesc,
    BufferUses,
);
type TextureOwner = (
    crate::graph::ResourceInstanceId,
    fluxel_rhi::api::DeviceId,
    fluxel_rhi::api::TextureDesc,
    TextureUses,
    ImageLayout,
);

impl<B: ResourceBackend> DeferredCapturePlan<B> {
    pub(super) fn bindings<'a>(
        &'a self,
        producer: &'a DeferredFrame<B>,
        overlay: Option<&'a super::capture_producer::DeferredCaptureOverlay<'a, B>>,
        receiver: &'a DeferredFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        super::capture_owner_validation::validate_capture_owners(producer, overlay, receiver)?;
        let (buffers, textures) = collect_capture_bindings(
            &self.execution,
            self.id.device,
            &self.producer,
            producer,
            self.producer_composition.as_ref(),
            overlay,
            &self.receiver,
            receiver,
        )?;
        FrameBindings::new(&self.execution, self.id.device, buffers, textures)
            .map_err(binding_error)
    }

    pub(super) fn surface_bindings<'a>(
        &'a self,
        producer: &'a DeferredFrame<B>,
        overlay: Option<&'a super::capture_producer::DeferredCaptureOverlay<'a, B>>,
        receiver: &'a DeferredFrame<B>,
        image: &'a fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<FrameBindings<'a, B>> {
        super::capture_owner_validation::validate_capture_owners(producer, overlay, receiver)?;
        let (buffers, textures) = collect_capture_bindings(
            &self.execution,
            self.id.device,
            &self.producer,
            producer,
            self.producer_composition.as_ref(),
            overlay,
            &self.receiver,
            receiver,
        )?;
        let slot = self
            .receiver
            .surface_target_slot
            .ok_or(Error::InvalidState)?;
        FrameBindings::new_with_surface_textures(
            &self.execution,
            self.id.device,
            buffers,
            textures,
            vec![BoundSurfaceTexture::new(slot, image)?],
        )
        .map_err(binding_error)
    }
}

fn collect_capture_bindings<'a, B: ResourceBackend>(
    execution: &CompiledExecution,
    device: fluxel_rhi::api::DeviceId,
    producer_plan: &'a DeferredPlan<B>,
    producer: &'a DeferredFrame<B>,
    producer_composition: Option<&'a super::capture_producer::DeferredCaptureProducerComposition>,
    overlay: Option<&'a super::capture_producer::DeferredCaptureOverlay<'a, B>>,
    receiver_plan: &'a DeferredPlan<B>,
    receiver: &'a DeferredFrame<B>,
) -> Result<(Vec<BoundBuffer<'a, B>>, Vec<BoundTexture<'a, B>>)> {
    super::validate_resident_texture_devices(&producer.resident, device)?;
    super::validate_resident_texture_devices(&receiver.resident, device)?;
    let uses_buffer = |slot| {
        execution
            .logical()
            .buffer_bindings()
            .requirements()
            .iter()
            .any(|requirement| requirement.slot == slot)
    };
    let mut buffers = Vec::new();
    let mut textures = Vec::new();
    let mut buffer_owners = HashMap::<BufferSlot, BufferOwner>::new();
    let mut texture_owners = HashMap::<TextureSlot, TextureOwner>::new();

    macro_rules! push_buffer {
        ($slot:expr, $instance:expr, $buffer:expr, $desc:expr, $incoming:expr) => {{
            let slot = $slot;
            if uses_buffer(slot) {
                let owner = ($instance, $buffer.device(), $desc.clone(), $incoming);
                match buffer_owners.get(&slot) {
                    Some(previous)
                        if previous.0 != owner.0
                            || previous.1 != owner.1
                            || !crate::graph::forward::same_buffer_owner_desc(
                                &previous.2,
                                &owner.2,
                            )
                            || previous.3 != owner.3 =>
                    {
                        return Err(Error::InvalidState);
                    }
                    Some(_) => {}
                    None => {
                        buffer_owners.insert(slot, owner);
                        buffers.push(BoundBuffer::new(slot, $instance, $buffer, $desc, $incoming));
                    }
                }
            }
        }};
    }
    macro_rules! push_texture {
        ($slot:expr, $instance:expr, $texture:expr, $desc:expr, $incoming:expr, $layout:expr) => {{
            let slot = $slot;
            let owner = (
                $instance,
                $texture.device(),
                $desc.clone(),
                $incoming,
                $layout,
            );
            match texture_owners.get(&slot) {
                Some(previous)
                    if previous.0 != owner.0
                        || previous.1 != owner.1
                        || !crate::graph::forward::same_texture_owner_desc(
                            &previous.2,
                            &owner.2,
                        )
                        || previous.3 != owner.3
                        || previous.4 != owner.4 =>
                {
                    return Err(Error::InvalidState);
                }
                Some(_) => {}
                None => {
                    texture_owners.insert(slot, owner);
                    textures.push(BoundTexture::new(
                        slot, $instance, $texture, $desc, $incoming, $layout,
                    ));
                }
            }
        }};
    }

    match (producer_composition, overlay) {
        (None, None) => {}
        (
            Some(super::capture_producer::DeferredCaptureProducerComposition::Ordinary(_)),
            Some(overlay),
        ) if overlay.oit.is_none() => {}
        (
            Some(super::capture_producer::DeferredCaptureProducerComposition::Oit(composition)),
            Some(overlay),
        ) => {
            let input = overlay.oit.as_ref().ok_or(Error::InvalidState)?;
            if input.peel_count != composition.peel_count
                || input.rounds.get() != composition.rounds
                || composition.product.passes.iterations.len() != composition.rounds as usize
            {
                return Err(Error::InvalidState);
            }
            for (slots, owners) in composition
                .slots
                .ping_pong
                .iter()
                .zip(&input.owners.ping_pong)
            {
                push_texture!(
                    slots.min_max,
                    owners.min_max.instance,
                    &owners.min_max.texture,
                    &owners.min_max.desc,
                    owners.min_max.incoming,
                    owners.min_max.layout
                );
                push_texture!(
                    slots.front,
                    owners.front.instance,
                    &owners.front.texture,
                    &owners.front.desc,
                    owners.front.incoming,
                    owners.front.layout
                );
                push_texture!(
                    slots.back,
                    owners.back.instance,
                    &owners.back.texture,
                    &owners.back.desc,
                    owners.back.incoming,
                    owners.back.layout
                );
            }
            push_texture!(
                composition.slots.back_accumulator,
                input.owners.back_accumulator.instance,
                &input.owners.back_accumulator.texture,
                &input.owners.back_accumulator.desc,
                input.owners.back_accumulator.incoming,
                input.owners.back_accumulator.layout
            );
            push_texture!(
                composition.final_slot,
                input.output.instance,
                &input.output.texture,
                &input.output.desc,
                input.output.incoming,
                input.output.layout
            );
        }
        _ => return Err(Error::InvalidState),
    }

    for (plan, frame) in [(producer_plan, producer), (receiver_plan, receiver)] {
        push_buffer!(
            plan.frame_upload_slot,
            frame.frame.upload_instance,
            &frame.frame.upload,
            &frame.frame.upload_desc,
            frame.frame.upload_incoming
        );
        push_buffer!(
            plan.uniform_slot,
            frame.frame.uniform_instance,
            &frame.frame.uniform,
            &frame.frame.uniform_desc,
            frame.frame.uniform_incoming
        );
        if let Some(slot) = plan.readback_slot {
            push_buffer!(
                slot,
                frame.frame.readback_instance,
                &frame.frame.readback,
                &frame.frame.readback_desc,
                frame.frame.readback_incoming
            );
        }
        push_buffer!(
            plan.resident_upload_slot,
            frame.resident.upload_instance,
            &frame.resident.upload,
            &frame.resident.upload_desc,
            frame.resident.upload_incoming
        );
        if let (Some(slot), Some(owner)) =
            (plan.environment_upload_slot, frame.environment.as_ref())
        {
            push_buffer!(
                slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming
            );
        }
        if let Some(owner) = frame.specular.as_ref() {
            if let Some(slot) = plan.specular_upload_slot {
                push_buffer!(
                    slot,
                    owner.upload_instance,
                    &owner.upload,
                    &owner.upload_desc,
                    owner.upload_incoming
                );
            }
            if let Some(slot) = plan.specular_view_uniform_slot {
                push_buffer!(
                    slot,
                    owner.view_uniform_instance,
                    &owner.view_uniform,
                    &owner.view_uniform_desc,
                    owner.view_uniform_incoming
                );
            }
        }
        for (slot, owner) in plan
            .stream_slots
            .iter()
            .copied()
            .zip(&frame.resident.streams)
        {
            push_buffer!(
                slot,
                owner.instance,
                &owner.buffer,
                &owner.desc,
                owner.incoming
            );
        }
        if let (Some(slots), Some(deformation)) =
            (plan.deformation_slots, frame.deformation.as_ref())
        {
            for (slot, owner) in [
                (slots[0], &deformation.upload),
                (slots[1], &deformation.source),
                (slots[2], &deformation.product),
                (slots[3], &deformation.parameters),
            ] {
                push_buffer!(
                    slot,
                    owner.instance,
                    &owner.buffer,
                    &owner.desc,
                    owner.incoming
                );
            }
        }
        for (slot, owner) in plan
            .texture_slots
            .iter()
            .copied()
            .zip(&frame.resident.textures)
        {
            push_texture!(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout
            );
        }
        for (slot, owner) in plan.gbuffer_slots.iter().copied().zip(&frame.gbuffer) {
            push_texture!(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout
            );
        }
        if let (Some(slot), Some(owner)) = (plan.effects_slot, frame.effects.as_ref()) {
            push_texture!(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout
            );
        }
        if let (Some(slot), Some(owner)) = (
            plan.diffuse_roughness_slot,
            frame.diffuse_roughness.as_ref(),
        ) {
            push_texture!(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout
            );
        }
        if let (Some(slot), Some(owner)) = (
            plan.ambient_occlusion_slot,
            frame.ambient_occlusion.as_ref(),
        ) {
            push_texture!(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout
            );
        }
        if let (Some(slots), Some(msaa)) = (plan.msaa_gbuffer_slots, frame.msaa.as_ref()) {
            for (slot, owner) in slots.into_iter().zip(&msaa.gbuffer) {
                push_texture!(
                    slot,
                    owner.instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout
                );
            }
            if let (Some(slot), Some(owner)) = (plan.msaa_effects_slot, msaa.effects.as_ref()) {
                push_texture!(
                    slot,
                    owner.instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout
                );
            }
            if let (Some(slot), Some(owner)) = (
                plan.msaa_diffuse_roughness_slot,
                msaa.diffuse_roughness.as_ref(),
            ) {
                push_texture!(
                    slot,
                    owner.instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout
                );
            }
            if let (Some(slot), Some(owner)) = (
                plan.msaa_ambient_occlusion_slot,
                msaa.ambient_occlusion.as_ref(),
            ) {
                push_texture!(
                    slot,
                    owner.instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout
                );
            }
        }
        if let (Some(slot), Some(owner)) = (plan.environment_slot, frame.environment.as_ref()) {
            push_texture!(
                slot,
                owner.texture_instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout
            );
        }
        if let Some(owner) = frame.specular.as_ref() {
            if let Some(slot) = plan.specular_radiance_slot {
                push_texture!(
                    slot,
                    owner.radiance.instance,
                    &owner.radiance.texture,
                    &owner.radiance.desc,
                    owner.radiance.incoming,
                    owner.radiance.layout
                );
            }
            if let Some(slot) = plan.specular_brdf_lut_slot {
                push_texture!(
                    slot,
                    owner.brdf_lut.instance,
                    &owner.brdf_lut.texture,
                    &owner.brdf_lut.desc,
                    owner.brdf_lut.incoming,
                    owner.brdf_lut.layout
                );
            }
        }
        push_texture!(
            plan.scene_color_slot.ok_or(Error::InvalidState)?,
            frame.scene_color.instance,
            &frame.scene_color.texture,
            &frame.scene_color.desc,
            frame.scene_color.incoming,
            frame.scene_color.layout
        );
        if let Some(slot) = plan.target_slot {
            push_texture!(
                slot,
                frame.frame.target_instance,
                &frame.frame.target,
                &frame.frame.target_desc,
                frame.frame.target_incoming,
                frame.frame.target_layout
            );
        }
    }
    if let (Some(composition), Some(overlay)) = (producer_composition, overlay) {
        match composition {
            super::capture_producer::DeferredCaptureProducerComposition::Ordinary(composition) => {
                composition.overlay.validate_frame(device, overlay.frame)?;
                composition.overlay.append_bindings(
                    execution,
                    overlay.frame,
                    &mut buffers,
                    &mut textures,
                )?;
            }
            super::capture_producer::DeferredCaptureProducerComposition::Oit(composition) => {
                composition.overlay.validate_frame(device, overlay.frame)?;
                composition.overlay.append_bindings(
                    execution,
                    overlay.frame,
                    &mut buffers,
                    &mut textures,
                )?;
            }
        }
    }
    Ok((buffers, textures))
}
