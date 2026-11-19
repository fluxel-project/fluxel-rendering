//! Shader depth resolve for the Deferred multisample G-buffer.
//!
//! Native resolves are color-only in Fluxel RHI.  This pass samples all
//! G-buffer inputs at sample zero and writes the resolved depth through
//! `frag_depth`, keeping the resolved `Depth32Float` owner usable by every
//! existing transparent and capture path.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn record_resolve(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let bindings = self.bindings(frame)?;
        self.record_resolve_with_bindings(
            encoder,
            super::record_view::DeferredRecordView::offscreen(frame),
            &bindings,
        )
    }

    pub(super) fn record_resolve_with_bindings(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let Some(resolve_pass) = self.resolve_pass else {
            return Ok(());
        };
        let msaa = frame.msaa.ok_or(Error::InvalidState)?;
        let slots = self.msaa_gbuffer_slots.ok_or(Error::InvalidState)?;
        let pass = bindings.pass(resolve_pass)?;
        for (index, slot) in slots.into_iter().enumerate() {
            pass.texture(
                slot,
                if index == 2 {
                    whole_depth()
                } else {
                    whole_color()
                },
                AccessKind::Read,
                sampled_read(),
            )?;
        }
        if let (Some(slot), Some(_)) = (self.msaa_effects_slot, msaa.effects.as_ref()) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        if let (Some(slot), Some(_)) = (
            self.msaa_diffuse_roughness_slot,
            msaa.diffuse_roughness.as_ref(),
        ) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        if let (Some(slot), Some(_)) = (
            self.msaa_ambient_occlusion_slot,
            msaa.ambient_occlusion.as_ref(),
        ) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        for (index, slot) in self.gbuffer_slots.into_iter().enumerate() {
            pass.texture(
                slot,
                if index == 2 {
                    whole_depth()
                } else {
                    whole_color()
                },
                AccessKind::Write,
                if index == 2 {
                    depth_write()
                } else {
                    color_write()
                },
            )?;
        }
        if let (Some(slot), Some(_)) = (self.effects_slot, frame.effects) {
            pass.texture(slot, whole_color(), AccessKind::Write, color_write())?;
        }
        if let (Some(slot), Some(_)) = (self.diffuse_roughness_slot, frame.diffuse_roughness) {
            pass.texture(slot, whole_color(), AccessKind::Write, color_write())?;
        }
        if let (Some(slot), Some(_)) = (self.ambient_occlusion_slot, frame.ambient_occlusion) {
            pass.texture(slot, whole_color(), AccessKind::Write, color_write())?;
        }
        pass.transition(encoder)?;
        let mut colors = vec![
            Some(ColorAttachment {
                view: AttachmentView::Texture(&frame.gbuffer[0].view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }),
            Some(ColorAttachment {
                view: AttachmentView::Texture(&frame.gbuffer[1].view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }),
        ];
        if let Some(effects) = frame.effects {
            colors.push(Some(ColorAttachment {
                view: AttachmentView::Texture(&effects.view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }));
        }
        if let Some(diffuse_roughness) = frame.diffuse_roughness {
            colors.push(Some(ColorAttachment {
                view: AttachmentView::Texture(&diffuse_roughness.view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }));
        }
        if let Some(ambient_occlusion) = frame.ambient_occlusion {
            colors.push(Some(ColorAttachment {
                view: AttachmentView::Texture(&ambient_occlusion.view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }));
        }
        let depth_stencil = DepthStencilAttachment {
            view: AttachmentView::Texture(&frame.gbuffer[2].view),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: true,
        };
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred msaa resolve"),
            colors: &colors,
            depth_stencil: Some(depth_stencil),
            render_area: render_area(frame.extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, frame.extent)?;
        raster.set_pipeline(&msaa.pipeline)?;
        raster.set_bind_group(0, &msaa.group, &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }
}
