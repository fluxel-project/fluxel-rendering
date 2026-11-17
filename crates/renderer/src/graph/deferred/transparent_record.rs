//! Recording dispatch for the single Deferred-plus-transparent Graph.

use super::*;
use crate::residency::ForwardFrame;

impl<B: ResourceBackend> DeferredTransparentPlan<B> {
    pub(crate) fn record_surface_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        opaque: &DeferredFrame<B>,
        transparent: &ForwardFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: CommandBackend + RasterBackend,
    {
        self.validate_frames_with_image(device, opaque, transparent, image, composite)?;
        let bindings = self.surface_bindings(opaque, transparent, image)?;
        if pass == self.overlay_upload {
            return self
                .overlay
                .record_upload(encoder, &bindings, transparent, pass);
        }
        if let Some(generation) = self
            .overlay_mips
            .iter()
            .find(|generation| generation.pass == pass)
        {
            return self
                .overlay
                .record_mip(encoder, &bindings, transparent, generation);
        }
        if pass == self.overlay_raster {
            return self.overlay.record_raster(
                encoder,
                &bindings,
                transparent,
                &opaque.scene_color,
                &opaque.gbuffer[2],
                pass,
            );
        }
        if pass == self.base.upload_pass {
            return self
                .base
                .record_upload_with_bindings(encoder, opaque, &bindings);
        }
        if let Some(generation) = self
            .base
            .mip_generation
            .iter()
            .find(|generation| generation.pass == pass)
        {
            return self
                .base
                .record_mip_with_bindings(encoder, opaque, generation, &bindings);
        }
        if pass == self.base.geometry_pass {
            return self
                .base
                .record_geometry_with_bindings(encoder, opaque, &bindings);
        }
        if pass == self.base.lighting_pass {
            return self
                .base
                .record_lighting_with_bindings(encoder, opaque, &bindings);
        }
        if self.base.composite_pass == Some(pass) {
            return self.record_surface_composite(encoder, opaque, image, composite, &bindings);
        }
        Err(Error::InvalidState)
    }

    fn record_surface_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        opaque: &DeferredFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let (surface_slot, _, _) = self.surface.ok_or(Error::InvalidState)?;
        let pass = bindings.pass(self.base.composite_pass.ok_or(Error::InvalidState)?)?;
        pass.texture(
            self.base.scene_color_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.surface_texture(
            surface_slot,
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred transparent surface composite"),
            colors: &colors,
            depth_stencil: None,
            render_area: render_area(opaque.extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, opaque.extent)?;
        raster.set_pipeline(composite.pipeline())?;
        raster.set_bind_group(0, composite.bind_group(), &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }

    pub(crate) fn record_pass(
        &self,
        encoder: &mut CommandEncoder<B>,
        opaque: &DeferredFrame<B>,
        transparent: &ForwardFrame<B>,
        pass: GraphPass,
    ) -> Result<Option<fluxel_rhi::api::ReadbackTicket<B>>>
    where
        B: CommandBackend + RasterBackend + ReadbackBackend,
    {
        self.validate_frames(opaque, transparent)?;
        let bindings = self.bindings(opaque, transparent)?;
        if pass == self.overlay_upload {
            self.overlay
                .record_upload(encoder, &bindings, transparent, pass)?;
            return Ok(None);
        }
        if let Some(generation) = self
            .overlay_mips
            .iter()
            .find(|generation| generation.pass == pass)
        {
            self.overlay
                .record_mip(encoder, &bindings, transparent, generation)?;
            return Ok(None);
        }
        if pass == self.overlay_raster {
            self.overlay.record_raster(
                encoder,
                &bindings,
                transparent,
                &opaque.scene_color,
                &opaque.gbuffer[2],
                pass,
            )?;
            return Ok(None);
        }
        if pass == self.base.upload_pass {
            self.base
                .record_upload_with_bindings(encoder, opaque, &bindings)?;
            return Ok(None);
        }
        if let Some(generation) = self
            .base
            .mip_generation
            .iter()
            .find(|generation| generation.pass == pass)
        {
            self.base
                .record_mip_with_bindings(encoder, opaque, generation, &bindings)?;
            return Ok(None);
        }
        if self.base.deformation_pass == Some(pass) {
            return Err(Error::Unsupported("deferred transparent deformation"));
        }
        if pass == self.base.geometry_pass {
            self.base
                .record_geometry_with_bindings(encoder, opaque, &bindings)?;
            return Ok(None);
        }
        if pass == self.base.lighting_pass {
            self.base
                .record_lighting_with_bindings(encoder, opaque, &bindings)?;
            return Ok(None);
        }
        if self.base.composite_pass == Some(pass) {
            self.base
                .record_composite_with_bindings(encoder, opaque, &bindings)?;
            return Ok(None);
        }
        if matches!(
            self.base.output,
            DeferredOutputPlan::Offscreen { readback_pass } if readback_pass == pass
        ) {
            return self
                .base
                .record_readback_with_bindings(encoder, opaque, &bindings)
                .map(Some);
        }
        Err(Error::InvalidState)
    }
}
