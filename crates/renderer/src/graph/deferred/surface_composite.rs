//! Deferred presentation compositing for surface output.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn surface_record_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != frame.extent || image.format() != frame.color_format {
            return Err(Error::InvalidRange);
        }
        let bindings = self.surface_bindings(frame, image)?;
        let pass = bindings.pass(self.composite_pass.ok_or(Error::InvalidState)?)?;
        pass.texture(
            self.scene_color_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.surface_texture(
            self.surface_target_slot.ok_or(Error::InvalidState)?,
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
        let area = Rect {
            x: 0,
            y: 0,
            width: frame.extent[0],
            height: frame.extent[1],
        };
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred surface scene-color composite"),
            colors: &colors,
            depth_stencil: None,
            render_area: area,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        raster.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: frame.extent[0] as f32,
            height: frame.extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        raster.scissor(area)?;
        raster.set_pipeline(&frame.composite_pipeline)?;
        raster.set_bind_group(0, &frame.composite_group, &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }
}
