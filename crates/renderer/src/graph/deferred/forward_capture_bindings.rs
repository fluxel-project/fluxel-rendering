//! Actual combined bindings for a Deferred producer and Forward receiver.

use super::*;
use crate::graph::execution::{merge_buffer_bindings, merge_texture_bindings};
use crate::{graph::ForwardOitInput, residency::ForwardFrame};

impl<B: ResourceBackend> DeferredForwardCapturePlan<B> {
    pub(super) fn bindings<'a>(
        &'a self,
        producer: &'a DeferredFrame<B>,
        overlay: Option<&'a super::capture_producer::DeferredCaptureOverlay<'a, B>>,
        receiver: &'a ForwardFrame<B>,
        receiver_oit: Option<&'a ForwardOitInput<'a, B>>,
        image: Option<&'a fluxel_rhi::api::SurfaceImage<B>>,
    ) -> Result<FrameBindings<'a, B>> {
        let (mut buffers, mut textures) = self.producer.collect_bindings(producer)?;
        match (self.producer_composition.as_ref(), overlay) {
            (None, None) => {}
            (Some(composition), Some(overlay)) => match composition {
                super::capture_producer::DeferredCaptureProducerComposition::Ordinary(
                    composition,
                ) => {
                    if overlay.oit.is_some() {
                        return Err(Error::InvalidState);
                    }
                    composition
                        .overlay
                        .validate_frame(self.id.device, overlay.frame)?;
                    composition.overlay.append_bindings(
                        &self.execution,
                        overlay.frame,
                        &mut buffers,
                        &mut textures,
                    )?;
                }
                super::capture_producer::DeferredCaptureProducerComposition::Oit(composition) => {
                    let input = overlay.oit.as_ref().ok_or(Error::InvalidState)?;
                    if input.peel_count != composition.peel_count
                        || input.rounds.get() != composition.rounds
                    {
                        return Err(Error::InvalidState);
                    }
                    composition
                        .overlay
                        .validate_frame(self.id.device, overlay.frame)?;
                    composition.overlay.append_bindings(
                        &self.execution,
                        overlay.frame,
                        &mut buffers,
                        &mut textures,
                    )?;
                    for (slots, owners) in composition
                        .slots
                        .ping_pong
                        .iter()
                        .zip(&input.owners.ping_pong)
                    {
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
                        (
                            composition.slots.back_accumulator,
                            &input.owners.back_accumulator,
                        ),
                        (composition.final_slot, input.output),
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
            },
            _ => return Err(Error::InvalidState),
        }
        self.receiver
            .append_bindings(&self.execution, receiver, &mut buffers, &mut textures)?;
        self.receiver
            .append_oit_bindings(receiver_oit, &mut textures)?;
        let buffers = merge_buffer_bindings(buffers)?;
        let textures = merge_texture_bindings(textures)?;
        match (&self.terminal, image) {
            (crate::graph::forward::ForwardReceiverTerminal::Offscreen { .. }, None) => {
                FrameBindings::new(&self.execution, self.id.device, buffers, textures)
                    .map_err(binding_error)
            }
            (crate::graph::forward::ForwardReceiverTerminal::Surface { slot, .. }, Some(image)) => {
                FrameBindings::new_with_surface_textures(
                    &self.execution,
                    self.id.device,
                    buffers,
                    textures,
                    vec![BoundSurfaceTexture::new(*slot, image)?],
                )
                .map_err(binding_error)
            }
            _ => Err(Error::InvalidState),
        }
    }
}
