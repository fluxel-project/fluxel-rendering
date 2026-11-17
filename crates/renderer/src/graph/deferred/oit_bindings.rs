//! Concrete bindings for Deferred OIT's opaque prefix, overlay, and product.

use super::*;
use crate::residency::ForwardFrame;

impl<B: ResourceBackend> DeferredOitPlan<B> {
    pub(super) fn bindings<'a>(
        &'a self,
        opaque: &'a DeferredFrame<B>,
        overlay: &'a ForwardFrame<B>,
        input: &'a crate::graph::ForwardOitInput<'a, B>,
    ) -> Result<FrameBindings<'a, B>> {
        let (buffers, textures) = self.binding_parts(opaque, overlay, input)?;
        FrameBindings::new(&self.base.execution, self.base.id.device, buffers, textures)
            .map_err(binding_error)
    }

    fn binding_parts<'a>(
        &'a self,
        opaque: &'a DeferredFrame<B>,
        overlay: &'a ForwardFrame<B>,
        input: &'a crate::graph::ForwardOitInput<'a, B>,
    ) -> Result<(Vec<BoundBuffer<'a, B>>, Vec<BoundTexture<'a, B>>)> {
        self.overlay.validate_frame(self.base.id.device, overlay)?;
        let (mut buffers, mut textures) = self.base.collect_bindings(opaque)?;
        self.overlay
            .append_bindings(&self.base.execution, overlay, &mut buffers, &mut textures)?;
        let mut add = |slot: TextureSlot, owner: &'a crate::residency::OitTextureOwner<B>| {
            textures.push(BoundTexture::new(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            ));
        };
        for (side, owners) in self.slots.ping_pong.iter().zip(&input.owners.ping_pong) {
            add(side.min_max, &owners.min_max);
            add(side.front, &owners.front);
            add(side.back, &owners.back);
        }
        add(self.slots.back_accumulator, &input.owners.back_accumulator);
        add(self.final_slot, input.output);
        Ok((buffers, textures))
    }

    pub(super) fn surface_bindings<'a>(
        &'a self,
        opaque: &'a DeferredFrame<B>,
        overlay: &'a ForwardFrame<B>,
        input: &'a crate::graph::ForwardOitInput<'a, B>,
        image: &'a fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<FrameBindings<'a, B>> {
        self.surface.ok_or(Error::InvalidState)?;
        let (buffers, textures) = self.binding_parts(opaque, overlay, input)?;
        let surface = BoundSurfaceTexture::new(
            self.base.surface_target_slot.ok_or(Error::InvalidState)?,
            image,
        )?;
        FrameBindings::new_with_surface_textures(
            &self.base.execution,
            self.base.id.device,
            buffers,
            textures,
            vec![surface],
        )
        .map_err(binding_error)
    }
}
