//! Combined concrete bindings for a Deferred opaque frame and Forward overlay.

use super::*;
use crate::residency::ForwardFrame;

impl<B: ResourceBackend> DeferredTransparentPlan<B> {
    pub(super) fn bindings<'a>(
        &'a self,
        opaque: &'a DeferredFrame<B>,
        transparent: &'a ForwardFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        self.overlay
            .validate_frame(self.base.id.device, transparent)?;
        let (mut buffers, mut textures) = self.base.collect_bindings(opaque)?;
        self.overlay.append_bindings(
            &self.base.execution,
            transparent,
            &mut buffers,
            &mut textures,
        )?;
        FrameBindings::new(&self.base.execution, self.base.id.device, buffers, textures)
            .map_err(binding_error)
    }

    pub(super) fn surface_bindings<'a>(
        &'a self,
        opaque: &'a DeferredFrame<B>,
        transparent: &'a ForwardFrame<B>,
        image: &'a fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<FrameBindings<'a, B>> {
        let (slot, _, _) = self.surface.ok_or(Error::InvalidState)?;
        self.overlay
            .validate_frame(self.base.id.device, transparent)?;
        let (mut buffers, mut textures) = self.base.collect_bindings(opaque)?;
        self.overlay.append_bindings(
            &self.base.execution,
            transparent,
            &mut buffers,
            &mut textures,
        )?;
        FrameBindings::new_with_surface_textures(
            &self.base.execution,
            self.base.id.device,
            buffers,
            textures,
            vec![BoundSurfaceTexture::new(slot, image)?],
        )
        .map_err(binding_error)
    }
}
