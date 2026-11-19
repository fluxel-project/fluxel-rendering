//! Concrete binding collection for a Forward producer and Deferred receiver.

use super::*;
use crate::graph::{
    ForwardOitInput,
    execution::{merge_buffer_bindings, merge_texture_bindings},
};
use crate::residency::ForwardFrame;

impl<B: ResourceBackend> ForwardDeferredCapturePlan<B> {
    pub(super) fn bindings<'a>(
        &'a self,
        producer: &'a ForwardFrame<B>,
        producer_oit: Option<&'a ForwardOitInput<'a, B>>,
        receiver: &'a DeferredFrame<B>,
        image: Option<&'a fluxel_rhi::api::SurfaceImage<B>>,
    ) -> Result<FrameBindings<'a, B>> {
        super::reverse_capture_owners::validate_owners(producer, producer_oit, receiver)?;
        let mut buffers = Vec::new();
        let mut textures = Vec::new();
        // Install the actual product first: aliases in the Deferred namespace
        // must resolve to this owner rather than to a cloned wrapper.
        let final_owner = producer_oit.map(|input| input.output);
        match final_owner {
            Some(owner) => textures.push(BoundTexture::new(
                self.producer.final_slot(),
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            )),
            None => textures.push(BoundTexture::new(
                self.producer.final_slot(),
                producer.frame.target_instance,
                &producer.frame.target,
                &producer.frame.target_desc,
                producer.frame.target_incoming,
                producer.frame.target_layout,
            )),
        }
        self.producer
            .append_bindings(&self.execution, producer, &mut buffers, &mut textures)?;
        self.producer
            .append_oit_bindings(producer_oit, &mut textures)?;
        let (receiver_buffers, receiver_textures) = self.receiver.collect_bindings(receiver)?;
        buffers.extend(receiver_buffers);
        textures.extend(receiver_textures);
        let buffers = merge_buffer_bindings(buffers)?;
        let textures = merge_texture_bindings(textures)?;
        match (self.surface, image) {
            (None, None) => FrameBindings::new(&self.execution, self.id.device, buffers, textures)
                .map_err(binding_error),
            (Some(_), Some(image)) => FrameBindings::new_with_surface_textures(
                &self.execution,
                self.id.device,
                buffers,
                textures,
                vec![BoundSurfaceTexture::new(
                    self.receiver
                        .surface_target_slot
                        .ok_or(Error::InvalidState)?,
                    image,
                )?],
            )
            .map_err(binding_error),
            _ => Err(Error::InvalidState),
        }
    }
}
