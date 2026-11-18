//! Actual owner binding collection for capture OIT products.

use super::capture_bindings::CaptureProductOwner;
use super::*;
use crate::graph::ResourceInstanceId;
use std::collections::HashSet;

impl<B: ResourceBackend> CaptureOitPlan<B> {
    pub(crate) fn validate_frames(
        &self,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_oit: Option<&ForwardOitInput<'_, B>>,
        receiver_oit: Option<&ForwardOitInput<'_, B>>,
    ) -> Result<()> {
        self.validate_owner_facts(capture, receiver, capture_oit, receiver_oit)?;
        self.bindings(capture, receiver, capture_oit, receiver_oit)
            .map(|_| ())
    }

    fn validate_owner_facts(
        &self,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_oit: Option<&ForwardOitInput<'_, B>>,
        receiver_oit: Option<&ForwardOitInput<'_, B>>,
    ) -> Result<()> {
        for (frame, shape) in [
            (capture, &self.capture_shape),
            (receiver, &self.receiver_shape),
        ] {
            validate_resident_texture_devices(&frame.resident, self.id.device)?;
            if frame.resident.device != self.id.device || forward_shape(frame) != *shape {
                return Err(Error::InvalidState);
            }
            validate_uploads(frame)?;
            validate_batches(frame)?;
            for draw in &frame.draws {
                validate_draw(draw, frame.resident.streams.len())?;
                validate_offsets(draw, self.alignment)?;
                draw_accesses(frame, draw)?;
            }
        }
        self.validate_oit_side(capture, &self.capture_oit, capture_oit)?;
        self.validate_oit_side(receiver, &self.receiver_oit, receiver_oit)?;
        if self.scene_refraction {
            let (Some(capture_depth), Some(receiver_depth)) =
                (&capture.frame.depth, &receiver.frame.depth)
            else {
                return Err(Error::InvalidState);
            };
            if capture_depth.device != self.id.device
                || receiver_depth.device != self.id.device
                || capture_depth.instance != receiver_depth.instance
                || capture_depth.texture.device() != self.id.device
                || receiver_depth.texture.device() != self.id.device
                || capture_depth.view.device() != self.id.device
                || receiver_depth.view.device() != self.id.device
                || capture_depth.incoming != receiver_depth.incoming
                || capture_depth.layout != receiver_depth.layout
                || !super::capture_declaration::same_texture_owner_desc(
                    &capture_depth.desc,
                    &receiver_depth.desc,
                )
            {
                return Err(Error::InvalidState);
            }
        }
        if let (Some(left), Some(right)) = (capture_oit, receiver_oit) {
            let left = oit_instances(left);
            if oit_instances(right)
                .iter()
                .any(|instance| left.contains(instance))
            {
                return Err(Error::InvalidState);
            }
        }
        // Only the producer's final color may alias a receiver capture
        // binding. No OIT intermediate may also be imported as an ordinary
        // texture, attachment or environment in either frame namespace.
        let receiver_captures = super::capture_declaration::capture_texture_slots(
            &ForwardGraphSpec::from_frame(receiver)?,
        )?;
        for (input, producer) in [(capture_oit, true), (receiver_oit, false)] {
            let Some(input) = input else {
                continue;
            };
            let instances = oit_instances(input);
            for (frame, receiver_side) in [(capture, false), (receiver, true)] {
                let attachments =
                    std::iter::once(frame.frame.target_instance)
                        .chain(frame.frame.depth.as_ref().map(|depth| depth.instance))
                        .chain(
                            frame
                                .environment
                                .as_ref()
                                .map(|owner| owner.texture_instance),
                        )
                        .chain(
                            frame.specular.as_ref().into_iter().flat_map(|owner| {
                                [owner.radiance.instance, owner.brdf_lut.instance]
                            }),
                        );
                if attachments
                    .into_iter()
                    .any(|instance| instances.contains(&instance))
                {
                    return Err(Error::InvalidState);
                }
                for (index, texture) in frame.resident.textures.iter().enumerate() {
                    let permitted_alias = producer
                        && receiver_side
                        && receiver_captures.get(index) == Some(&true)
                        && texture.instance == input.output.instance;
                    if instances.contains(&texture.instance) && !permitted_alias {
                        return Err(Error::InvalidState);
                    }
                }
            }
        }
        let product = match capture_oit {
            Some(input) => CaptureProductOwner {
                texture: &input.output.texture,
                view: &input.output.view,
                desc: &input.output.desc,
                instance: input.output.instance,
                incoming: input.output.incoming,
                layout: input.output.layout,
                extent: capture.extent,
            },
            None => CaptureProductOwner::from_forward_frame(capture),
        };
        super::capture_bindings::validate_capture_product_alias(
            &product,
            receiver,
            self.scene_refraction,
        )
    }

    pub(super) fn bindings<'a>(
        &'a self,
        capture: &'a ForwardFrame<B>,
        receiver: &'a ForwardFrame<B>,
        capture_oit: Option<&'a ForwardOitInput<'a, B>>,
        receiver_oit: Option<&'a ForwardOitInput<'a, B>>,
    ) -> Result<FrameBindings<'a, B>> {
        self.bindings_inner(capture, receiver, capture_oit, receiver_oit, None)
    }

    pub(crate) fn validate_frames_with_image(
        &self,
        device: &Device<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_oit: Option<&ForwardOitInput<'_, B>>,
        receiver_oit: Option<&ForwardOitInput<'_, B>>,
        image: &SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()> {
        if device.id() != self.id.device || image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        let (extent, format) = match self.terminal {
            super::capture_oit_declaration::CaptureOitTerminal::Surface {
                extent, format, ..
            } => (extent, format),
            _ => return Err(Error::InvalidState),
        };
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != extent || image.format() != format || composite.extent() != extent {
            return Err(Error::InvalidRange);
        }
        composite.validate_for(device, image)?;
        if composite.target_instance()
            != receiver_oit.map_or(receiver.frame.target_instance, |input| {
                input.output.instance
            })
        {
            return Err(Error::InvalidState);
        }
        self.bindings_inner(capture, receiver, capture_oit, receiver_oit, Some(image))
            .map(|_| ())
    }

    pub(super) fn bindings_with_image<'a>(
        &'a self,
        capture: &'a ForwardFrame<B>,
        receiver: &'a ForwardFrame<B>,
        capture_oit: Option<&'a ForwardOitInput<'a, B>>,
        receiver_oit: Option<&'a ForwardOitInput<'a, B>>,
        image: &'a SurfaceImage<B>,
    ) -> Result<FrameBindings<'a, B>> {
        self.bindings_inner(capture, receiver, capture_oit, receiver_oit, Some(image))
    }

    fn bindings_inner<'a>(
        &'a self,
        capture: &'a ForwardFrame<B>,
        receiver: &'a ForwardFrame<B>,
        capture_oit: Option<&'a ForwardOitInput<'a, B>>,
        receiver_oit: Option<&'a ForwardOitInput<'a, B>>,
        image: Option<&'a SurfaceImage<B>>,
    ) -> Result<FrameBindings<'a, B>> {
        self.validate_owner_facts(capture, receiver, capture_oit, receiver_oit)?;
        let (buffers, mut textures) = super::capture_binding_collection::collect(
            &self.execution,
            self.id.device,
            &self.capture,
            capture,
            &self.receiver,
            receiver,
            self.scene_refraction,
            self.scene_refraction,
        )?;
        self.append_oit(&mut textures, &self.capture_oit, capture_oit)?;
        self.append_oit(&mut textures, &self.receiver_oit, receiver_oit)?;
        // The producer OIT final owner is already bound through its OIT slot.
        // Receiver aliases point to that same Graph version, so adding another
        // binding for every logical capture texture would be a duplicate owner.
        let surface_textures = match (self.terminal, image) {
            (super::capture_oit_declaration::CaptureOitTerminal::Readback { .. }, None) => {
                Vec::new()
            }
            (super::capture_oit_declaration::CaptureOitTerminal::HdrReadback(terminal), None) => {
                terminal.append_binding(
                    receiver.frame.output.as_ref().ok_or(Error::InvalidState)?,
                    &mut textures,
                );
                Vec::new()
            }
            (
                super::capture_oit_declaration::CaptureOitTerminal::Surface { slot, .. },
                Some(image),
            ) => vec![BoundSurfaceTexture::new(slot, image)?],
            _ => return Err(Error::InvalidState),
        };
        FrameBindings::new_with_surface_textures(
            &self.execution,
            self.id.device,
            buffers,
            textures,
            surface_textures,
        )
        .map_err(binding_error)
    }

    fn append_oit<'a>(
        &self,
        textures: &mut Vec<BoundTexture<'a, B>>,
        plan: &Option<super::oit_plan::ForwardOitPlan>,
        input: Option<&'a ForwardOitInput<'a, B>>,
    ) -> Result<()> {
        match (plan, input) {
            (None, None) => Ok(()),
            (Some(plan), Some(input)) => {
                for (slots, owners) in plan.slots.ping_pong.iter().zip(&input.owners.ping_pong) {
                    for (slot, owner) in [
                        (slots.min_max, &owners.min_max),
                        (slots.front, &owners.front),
                        (slots.back, &owners.back),
                    ] {
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
                for (slot, owner) in [
                    (plan.slots.back_accumulator, &input.owners.back_accumulator),
                    (plan.slots.output, input.output),
                ] {
                    textures.push(BoundTexture::new(
                        slot,
                        owner.instance,
                        &owner.texture,
                        &owner.desc,
                        owner.incoming,
                        owner.layout,
                    ));
                }
                Ok(())
            }
            _ => Err(Error::InvalidState),
        }
    }

    fn validate_oit_side(
        &self,
        frame: &ForwardFrame<B>,
        plan: &Option<super::oit_plan::ForwardOitPlan>,
        input: Option<&ForwardOitInput<'_, B>>,
    ) -> Result<()> {
        match (plan, input) {
            (None, None) => Ok(()),
            (Some(plan), Some(input))
                if plan.opaque_count == input.opaque_count
                    && plan.peel_count == input.peel_count
                    && plan.rounds == input.rounds
                    && input.owners.extent == frame.extent
                    && input.output.device == self.id.device
                    && input.output.desc.extent == [frame.extent[0], frame.extent[1], 1] =>
            {
                input.owners.validate(self.id.device)?;
                let output = input.output;
                if output.texture.device() != self.id.device
                    || output.view.device() != self.id.device
                    || output.desc.dimension != fluxel_rhi::api::TextureDimension::D2
                    || output.desc.array_layers != 1
                    || output.desc.mip_levels != 1
                    || output.desc.samples != 1
                    || output.desc.format != frame.frame.target_desc.format
                    || output.view_desc.format != output.desc.format
                    || output.view_desc.dimension != fluxel_rhi::api::TextureViewDimension::D2
                    || output.view_desc.aspects != TextureAspects::COLOR
                    || output.view_desc.mips != (0..1)
                    || output.view_desc.layers != (0..1)
                    || !output.desc.usage.contains(
                        TextureUsage::RENDER_ATTACHMENT
                            | TextureUsage::TEXTURE_BINDING
                            | TextureUsage::COPY_SRC,
                    )
                {
                    return Err(Error::InvalidState);
                }

                let instances = oit_instances(input);
                if instances.len() != 8
                    || instances.contains(&frame.frame.target_instance)
                    || frame
                        .frame
                        .depth
                        .as_ref()
                        .is_some_and(|depth| instances.contains(&depth.instance))
                    || frame
                        .resident
                        .textures
                        .iter()
                        .any(|texture| instances.contains(&texture.instance))
                {
                    return Err(Error::InvalidState);
                }
                Ok(())
            }
            _ => Err(Error::InvalidState),
        }
    }
}

fn oit_instances<B: ResourceBackend>(
    input: &ForwardOitInput<'_, B>,
) -> HashSet<ResourceInstanceId> {
    let mut result = HashSet::with_capacity(8);
    for side in &input.owners.ping_pong {
        result.extend([
            side.min_max.instance,
            side.front.instance,
            side.back.instance,
        ]);
    }
    result.insert(input.owners.back_accumulator.instance);
    result.insert(input.output.instance);
    result
}
