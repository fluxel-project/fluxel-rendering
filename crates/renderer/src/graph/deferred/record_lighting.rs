//! Deferred lighting recording, including the optional surface-effects input.

use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn record_lighting(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let bindings = self.bindings(frame)?;
        self.record_lighting_with_bindings(encoder, frame, &bindings)
    }

    pub(super) fn record_lighting_with_bindings(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let scene_color_slot = self.scene_color_slot.ok_or(Error::InvalidState)?;
        let pass = bindings.pass(self.lighting_pass)?;
        for (slot, range) in [
            (self.gbuffer_slots[0], whole_color()),
            (self.gbuffer_slots[1], whole_color()),
            (self.gbuffer_slots[2], whole_depth()),
        ] {
            pass.texture(slot, range, AccessKind::Read, sampled_read())?;
        }
        if let (Some(slot), Some(_)) = (self.effects_slot, frame.effects.as_ref()) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        if let (Some(slot), Some(_)) = (
            self.diffuse_roughness_slot,
            frame.diffuse_roughness.as_ref(),
        ) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        if let (Some(slot), Some(_)) = (
            self.ambient_occlusion_slot,
            frame.ambient_occlusion.as_ref(),
        ) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        pass.texture(
            scene_color_slot,
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        if let (Some(slot), Some(_)) = (self.environment_slot, frame.environment.as_ref()) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        if let (Some(radiance), Some(lut), Some(view_uniform), Some(owner)) = (
            self.specular_radiance_slot,
            self.specular_brdf_lut_slot,
            self.specular_view_uniform_slot,
            frame.specular.as_ref(),
        ) {
            pass.texture(
                radiance,
                all_color_mips(owner.radiance.desc.mip_levels),
                AccessKind::Read,
                sampled_read(),
            )?;
            pass.texture(lut, one_color_mip(0), AccessKind::Read, sampled_read())?;
            pass.buffer(
                view_uniform,
                0..SPECULAR_VIEW_BYTES,
                AccessKind::Read,
                uniform_read(),
            )?;
        }
        if let Some(range) = &frame.lighting_view {
            pass.buffer(
                self.uniform_slot,
                range.clone(),
                AccessKind::Read,
                lighting_uniform_read(),
            )?;
        }
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&frame.scene_color.view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred lighting"),
            colors: &colors,
            depth_stencil: None,
            render_area: render_area(frame.extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, frame.extent)?;
        raster.set_pipeline(&frame.lighting_pipeline)?;
        raster.set_bind_group(0, &frame.lighting_group, &[])?;
        if let Some(group) = &frame.lighting_environment_group {
            raster.set_bind_group(1, group, &[])?;
        }
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }
}
