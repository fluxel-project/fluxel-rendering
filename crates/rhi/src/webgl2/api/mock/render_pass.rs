//! Mock framebuffer and raster-command domains.
//!
//! Covers the pass lifetime (attachments, draw buffers, blit) and the
//! pipeline/draw commands issued inside it.

use super::*;

impl GlFramebufferApi for MockGlFamilyApi {
    fn create_framebuffer(
        &mut self,
        d: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        self.ready("create-framebuffer")?;
        for v in d
            .color_attachments
            .iter()
            .copied()
            .chain(d.depth_stencil_attachment)
        {
            self.validate_attachment("create-framebuffer", v)?;
        }
        d.validate(
            self.discovery.limits().max_color_attachments,
            self.discovery.limits().max_draw_buffers,
            self.stamp,
        )
        .map_err(|_| GlError::Validation {
            operation: "create-framebuffer",
            message: "invalid framebuffer descriptor".into(),
        })?;
        let id = FramebufferId::new(self.stamp, self.slot()?, 0);
        self.framebuffers.insert(id, d.clone());
        self.calls.push(MockCall::CreateFramebuffer(id));
        Ok(id)
    }
    fn destroy_framebuffer(&mut self, id: FramebufferId) -> Result<(), GlError> {
        self.ready("destroy-framebuffer")?;
        self.live("destroy-framebuffer", id, |this| {
            this.framebuffers.contains_key(&id)
        })?;
        self.framebuffers.remove(&id);
        self.calls.push(MockCall::DestroyFramebuffer(id));
        Ok(())
    }
    fn begin_render_pass(&mut self, d: &GlRenderPassDescriptor) -> Result<(), GlError> {
        self.ready("begin-render-pass")?;
        self.live("begin-render-pass", d.framebuffer, |this| {
            this.framebuffers.contains_key(&d.framebuffer)
        })?;
        if self.pass_active {
            return self.invalid("begin-render-pass", "render pass already active");
        }
        let framebuffer = match self.framebuffers.get(&d.framebuffer).cloned() {
            Some(framebuffer) => framebuffer,
            None => return self.invalid("begin-render-pass", "framebuffer is not live"),
        };
        d.validate(
            &framebuffer,
            self.discovery.limits().max_color_attachments,
            self.discovery.limits().max_draw_buffers,
            self.stamp,
        )
        .map_err(|_| GlError::Validation {
            operation: "begin-render-pass",
            message: "render pass does not match its framebuffer descriptor".into(),
        })?;
        for a in &d.color_attachments {
            self.validate_attachment("begin-render-pass", a.view)?;
            if let Some(v) = a.resolve_target {
                self.validate_attachment("begin-render-pass", v)?;
            }
        }
        self.pass_active = true;
        self.calls.push(MockCall::BeginRenderPass(d.framebuffer));
        Ok(())
    }
    fn end_render_pass(&mut self) -> Result<(), GlError> {
        self.ready("end-render-pass")?;
        if !self.pass_active {
            return self.invalid("end-render-pass", "no active render pass");
        }
        self.pass_active = false;
        self.calls.push(MockCall::EndRenderPass);
        Ok(())
    }
    fn blit_framebuffer(
        &mut self,
        source: FramebufferId,
        destination: FramebufferId,
        region: GlBlitRegion,
        filter: GlFilterMode,
        masks: GlBlitMask,
    ) -> Result<(), GlError> {
        self.ready("blit-framebuffer")?;
        self.live("blit-framebuffer", source, |this| {
            this.framebuffers.contains_key(&source)
        })?;
        self.live("blit-framebuffer", destination, |this| {
            this.framebuffers.contains_key(&destination)
        })?;
        if source == destination {
            return self.invalid(
                "blit-framebuffer",
                "blit source and destination are identical",
            );
        }
        if masks.is_empty() {
            return self.invalid(
                "blit-framebuffer",
                "blit selects no color/depth/stencil plane",
            );
        }
        region.validate().map_err(|_| GlError::Validation {
            operation: "blit-framebuffer",
            message: "invalid blit region".into(),
        })?;
        let sample_count = |descriptor: &GlFramebufferDescriptor| {
            descriptor
                .color_attachments
                .first()
                .map(|view| view.sample_count)
                .or_else(|| {
                    descriptor
                        .depth_stencil_attachment
                        .as_ref()
                        .map(|view| view.sample_count)
                })
                .unwrap_or(1)
        };
        let shape = |descriptor: &GlFramebufferDescriptor| {
            descriptor
                .color_attachments
                .first()
                .copied()
                .or(descriptor.depth_stencil_attachment)
                .map(|view| (view.width, view.height))
                .unwrap_or((0, 0))
        };
        // Both descriptors were proven live above; sample counts and extents
        // come from the recorded attachment views exactly as a real
        // completeness check would.
        let (source_shape, destination_shape) = {
            let source_shape = self.framebuffers.get(&source).map(shape).unwrap_or((0, 0));
            let destination_shape = self
                .framebuffers
                .get(&destination)
                .map(shape)
                .unwrap_or((0, 0));
            (source_shape, destination_shape)
        };
        let within = |offset: [u32; 2], extent: [u32; 2], shape: (u32, u32)| {
            offset[0]
                .checked_add(extent[0])
                .is_some_and(|end| end <= shape.0)
                && offset[1]
                    .checked_add(extent[1])
                    .is_some_and(|end| end <= shape.1)
        };
        if !within(region.src_offset, region.src_extent, source_shape) {
            return self.invalid("blit-framebuffer", "blit source leaves its framebuffer");
        }
        if !within(region.dst_offset, region.dst_extent, destination_shape) {
            return self.invalid(
                "blit-framebuffer",
                "blit destination leaves its framebuffer",
            );
        }
        if filter != GlFilterMode::Nearest {
            let multisampled = |id: FramebufferId| {
                self.framebuffers
                    .get(&id)
                    .map(sample_count)
                    .map(|count| count > 1)
                    .unwrap_or(false)
            };
            if multisampled(source) || multisampled(destination) {
                return self.invalid(
                    "blit-framebuffer",
                    "multisampled blit targets only accept nearest filtering",
                );
            }
        }
        // Depth/stencil planes never scale and never filter.
        if (masks.depth || masks.stencil)
            && (filter != GlFilterMode::Nearest || region.src_extent != region.dst_extent)
        {
            return self.invalid(
                "blit-framebuffer",
                "depth/stencil blits require nearest filtering and identical extents",
            );
        }
        self.calls.push(MockCall::BlitFramebuffer {
            source,
            destination,
        });
        Ok(())
    }
}
impl GlRasterCommandApi for MockGlFamilyApi {
    fn set_raster_pipeline(&mut self, p: &GlRasterPipeline) -> Result<(), GlError> {
        self.ready("set-raster-pipeline")?;
        if !self.pass_active {
            return self.invalid("set-raster-pipeline", "no active render pass");
        }
        self.live("set-raster-pipeline", p.program, |this| {
            this.programs.contains(&p.program)
        })?;
        self.live("set-raster-pipeline", p.vertex_array, |this| {
            this.vaos.contains(&p.vertex_array)
        })?;
        self.calls.push(MockCall::SetRasterPipeline {
            program: p.program,
            vertex_array: p.vertex_array,
        });
        Ok(())
    }
    fn draw_raster(&mut self, d: GlDrawCommand) -> Result<(), GlError> {
        self.ready("draw-raster")?;
        if !self.pass_active {
            return self.invalid("draw-raster", "no active render pass");
        }
        let zero = match d {
            GlDrawCommand::NonIndexed(x) => x.vertex_count == 0 || x.instance_count == 0,
            GlDrawCommand::Indexed(x) => x.index_count == 0 || x.instance_count == 0,
        };
        if zero {
            return self.invalid("draw-raster", "draw count and instances must be nonzero");
        }
        self.calls.push(MockCall::DrawRaster(d));
        Ok(())
    }
}
