//! Native framebuffer, render-pass, and blit execution.
//!
//! Pass load/clear executes exactly once per pass through the `glClearBuffer*`
//! commands, which honor scissor state but ignore color and depth masks; the
//! pass therefore disables scissor around its clears, and pipeline
//! application restores the scissor group afterwards. Resolve stays an
//! explicit `glBlitFramebuffer` from the multisample source, per the shared
//! framebuffer contract.

use super::super::{
    FramebufferId, GlAttachmentTarget, GlBlitMask, GlBlitRegion, GlDepthStencilAttachment, GlError,
    GlFamilyApi as _, GlFilterMode, GlFramebufferApi, GlFramebufferDescriptor, GlLoadOp,
    GlRenderPassDescriptor, GlStoreOp, GlTextureDimension, GlTextureView,
};
use super::provider::{
    NativeFramebuffer, NativeGlProvider, depth_attachment_point, has_stencil_plane,
};

const fn blit_mask_bits(masks: GlBlitMask) -> u32 {
    let mut bits = 0;
    if masks.color {
        bits |= glow::COLOR_BUFFER_BIT;
    }
    if masks.depth {
        bits |= glow::DEPTH_BUFFER_BIT;
    }
    if masks.stencil {
        bits |= glow::STENCIL_BUFFER_BIT;
    }
    bits
}

const fn blit_filter(filter: GlFilterMode) -> u32 {
    match filter {
        GlFilterMode::Nearest => glow::NEAREST,
        GlFilterMode::Linear => glow::LINEAR,
    }
}

/// The attachment constant of one `glDrawBuffers` selection entry.
const fn draw_buffer_constant(index: u32) -> u32 {
    glow::COLOR_ATTACHMENT0 + index
}

/// `(width, height, sample_count)` facts of a stored descriptor.
fn framebuffer_shape(descriptor: &GlFramebufferDescriptor) -> (u32, u32, u32) {
    descriptor
        .color_attachments
        .first()
        .copied()
        .or(descriptor.depth_stencil_attachment)
        .map(|view| (view.width, view.height, view.sample_count))
        .unwrap_or((0, 0, 1))
}

impl GlFramebufferApi for NativeGlProvider<'_> {
    fn create_framebuffer(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-framebuffer";
        self.assert_ready(OP)?;
        for view in descriptor
            .color_attachments
            .iter()
            .copied()
            .chain(descriptor.depth_stencil_attachment)
        {
            self.validate_attachment(OP, view)?;
        }
        let limits = self.discovery.limits();
        descriptor
            .validate(
                limits.max_color_attachments,
                limits.max_draw_buffers,
                self.context_stamp(),
            )
            .map_err(|_| Self::validation(OP, "invalid framebuffer descriptor"))?;
        // SAFETY: current-context contract; the framebuffer is deleted on
        // every error path before this function returns.
        let raw = unsafe { self.gl.create_framebuffer() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        let attached = self.attach_all(OP, raw, descriptor);
        // Depth-only framebuffers must not keep a draw buffer selected.
        let result = unsafe {
            if descriptor.color_attachments.is_empty() {
                self.gl.draw_buffers(&[glow::NONE]);
            } else if !descriptor.draw_buffers.is_empty() {
                let constants: Vec<u32> = descriptor
                    .draw_buffers
                    .iter()
                    .map(|index| draw_buffer_constant(*index))
                    .collect();
                self.gl.draw_buffers(&constants);
            }
            attached.and_then(|()| self.require_complete(OP))
        };
        let result = result.and_then(|()| self.driver_error(OP));
        // SAFETY: see above; restore the default framebuffer binding so no
        // half-validated target stays selected on any path.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        if let Err(error) = result {
            // SAFETY: current-context contract.
            unsafe { self.gl.delete_framebuffer(raw) };
            return Err(error);
        }
        let slot = self.slot(OP)?;
        let id = FramebufferId::new(self.context_stamp(), slot, 0);
        self.framebuffers.insert(
            id,
            NativeFramebuffer {
                generation: id.generation,
                raw,
                descriptor: descriptor.clone(),
            },
        );
        Ok(id)
    }

    fn destroy_framebuffer(&mut self, framebuffer: FramebufferId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-framebuffer";
        self.assert_ready(OP)?;
        self.framebuffer(OP, framebuffer)?;
        let entry = self
            .framebuffers
            .remove(&framebuffer)
            .ok_or_else(|| Self::validation(OP, "framebuffer disappeared"))?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_framebuffer(entry.raw) };
        self.driver_error(OP)
    }

    fn begin_render_pass(&mut self, descriptor: &GlRenderPassDescriptor) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-render-pass";
        self.assert_ready(OP)?;
        if self.pass.is_some() {
            return Err(Self::validation(OP, "render pass already active"));
        }
        let record = self.framebuffer(OP, descriptor.framebuffer)?;
        descriptor
            .validate(
                &record.descriptor,
                self.discovery.limits().max_color_attachments,
                self.discovery.limits().max_draw_buffers,
                self.context_stamp(),
            )
            .map_err(|_| {
                Self::validation(OP, "render pass does not match its framebuffer descriptor")
            })?;
        for attachment in &descriptor.color_attachments {
            self.validate_attachment(OP, attachment.view)?;
            if let Some(resolve) = attachment.resolve_target {
                // Resolving stays an explicit blit; the pass only proves the
                // resolve target is live and structurally valid.
                self.validate_attachment(OP, resolve)?;
            }
        }
        if let Some(depth_stencil) = &descriptor.depth_stencil_attachment {
            self.validate_attachment(OP, depth_stencil.view)?;
            if depth_stencil.stencil_load == GlLoadOp::Clear
                && !has_stencil_plane(depth_stencil.view.format)
            {
                return Err(Self::validation(
                    OP,
                    "attachment has no stencil plane to clear",
                ));
            }
        }
        let raw = record.raw;
        let shape = framebuffer_shape(&record.descriptor);
        let discard_color: Vec<bool> = descriptor
            .color_attachments
            .iter()
            .map(|attachment| attachment.store == GlStoreOp::Discard)
            .collect();
        let discard_depth_stencil = descriptor
            .depth_stencil_attachment
            .as_ref()
            .map(|attachment| attachment.depth_store == GlStoreOp::Discard);

        // SAFETY: current-context contract; the framebuffer is live and the
        // clears are total (scissor disabled) exactly once per pass begin.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(raw));
            if !record.descriptor.draw_buffers.is_empty() {
                let constants: Vec<u32> = record
                    .descriptor
                    .draw_buffers
                    .iter()
                    .map(|index| draw_buffer_constant(*index))
                    .collect();
                self.gl.draw_buffers(&constants);
            }
            // glClearBuffer* honors scissor only; make pass-load clears total.
            // The installed pipeline re-applies the scissor group after pass
            // begin.
            self.gl.disable(glow::SCISSOR_TEST);
            for (index, attachment) in descriptor.color_attachments.iter().enumerate() {
                if attachment.load == GlLoadOp::Clear {
                    let clear = &attachment.clear;
                    self.gl.clear_buffer_f32_slice(
                        glow::COLOR,
                        index as u32,
                        &[
                            f32::from_bits(clear.red),
                            f32::from_bits(clear.green),
                            f32::from_bits(clear.blue),
                            f32::from_bits(clear.alpha),
                        ],
                    );
                }
            }
            if let Some(attachment) = &descriptor.depth_stencil_attachment {
                let depth_clears = attachment.depth_load == GlLoadOp::Clear;
                let stencil_clears = attachment.stencil_load == GlLoadOp::Clear
                    && has_stencil_plane(attachment.view.format);
                match (depth_clears, stencil_clears) {
                    (true, true) => self.gl.clear_buffer_depth_stencil(
                        glow::DEPTH_STENCIL,
                        0,
                        f32::from_bits(attachment.clear.depth),
                        attachment.clear.stencil as i32,
                    ),
                    (true, false) => self.gl.clear_buffer_f32_slice(
                        glow::DEPTH,
                        0,
                        &[f32::from_bits(attachment.clear.depth)],
                    ),
                    (false, true) => self.gl.clear_buffer_u32_slice(
                        glow::STENCIL,
                        0,
                        &[attachment.clear.stencil],
                    ),
                    (false, false) => {}
                }
            }
        }
        if let Err(error) = self.driver_error(OP) {
            // SAFETY: current-context contract.
            unsafe { self.gl.bind_framebuffer(glow::FRAMEBUFFER, None) };
            return Err(error);
        }
        self.pass = Some(super::provider::ActivePass {
            framebuffer: descriptor.framebuffer,
            width: shape.0,
            height: shape.1,
            samples: shape.2,
            discard_color,
            discard_depth_stencil,
        });
        Ok(())
    }

    fn end_render_pass(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "end-render-pass";
        self.assert_ready(OP)?;
        let Some(pass) = self.pass.take() else {
            return Err(Self::validation(OP, "no active render pass"));
        };
        self.raster = None;
        let raw = self.framebuffer(OP, pass.framebuffer)?.raw;
        let mut invalidate: Vec<u32> = pass
            .discard_color
            .iter()
            .enumerate()
            .filter(|(_, discard)| **discard)
            .map(|(index, _)| draw_buffer_constant(index as u32))
            .collect();
        if pass.discard_depth_stencil == Some(true) {
            invalidate.push(glow::DEPTH_ATTACHMENT);
        }
        // SAFETY: current-context contract; glInvalidateFramebuffer is issued
        // only where the profile's core supplies it (ES 3.x, desktop 4.3+);
        // desktop cores before 4.3 keep store semantics, which is the honest
        // fallback because no invalidate command exists to call.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(raw));
            let result = if invalidate.is_empty()
                || !supports_framebuffer_invalidate(self.discovery.context().profile())
            {
                Ok(())
            } else {
                self.gl
                    .invalidate_framebuffer(glow::FRAMEBUFFER, &invalidate);
                self.driver_error(OP)
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            result
        }
    }

    fn blit_framebuffer(
        &mut self,
        source: FramebufferId,
        destination: FramebufferId,
        region: GlBlitRegion,
        filter: GlFilterMode,
        masks: GlBlitMask,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "blit-framebuffer";
        self.assert_ready(OP)?;
        if source == destination {
            return Err(Self::validation(
                OP,
                "blit source and destination are identical",
            ));
        }
        if masks.is_empty() {
            return Err(Self::validation(
                OP,
                "blit selects no color/depth/stencil plane",
            ));
        }
        region
            .validate()
            .map_err(|_| Self::validation(OP, "invalid blit region"))?;
        let source_shape = self
            .framebuffer(OP, source)
            .map(|record| framebuffer_shape(&record.descriptor))?;
        let destination_shape = self
            .framebuffer(OP, destination)
            .map(|record| framebuffer_shape(&record.descriptor))?;
        let within = |offset: [u32; 2], extent: [u32; 2], shape: (u32, u32, u32)| {
            offset[0]
                .checked_add(extent[0])
                .is_some_and(|end| end <= shape.0)
                && offset[1]
                    .checked_add(extent[1])
                    .is_some_and(|end| end <= shape.1)
        };
        if !within(region.src_offset, region.src_extent, source_shape) {
            return Err(Self::validation(OP, "blit source leaves its framebuffer"));
        }
        if !within(region.dst_offset, region.dst_extent, destination_shape) {
            return Err(Self::validation(
                OP,
                "blit destination leaves its framebuffer",
            ));
        }
        // The shared contract: multisampled targets only accept nearest.
        if filter != GlFilterMode::Nearest && (source_shape.2 > 1 || destination_shape.2 > 1) {
            return Err(Self::validation(
                OP,
                "multisampled blit targets only accept nearest filtering",
            ));
        }
        // Depth/stencil planes never scale and never filter.
        if (masks.depth || masks.stencil)
            && (filter != GlFilterMode::Nearest || region.src_extent != region.dst_extent)
        {
            return Err(Self::validation(
                OP,
                "depth/stencil blits require nearest filtering and identical extents",
            ));
        }
        let source_raw = self.framebuffer(OP, source)?.raw;
        let destination_raw = self.framebuffer(OP, destination)?.raw;
        // SAFETY: current-context contract; both framebuffers are live and
        // every bounds/filter rule was validated before the blit.
        unsafe {
            self.gl
                .bind_framebuffer(glow::READ_FRAMEBUFFER, Some(source_raw));
            self.gl
                .bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(destination_raw));
            self.gl.blit_framebuffer(
                region.src_offset[0] as i32,
                region.src_offset[1] as i32,
                (region.src_offset[0] + region.src_extent[0]) as i32,
                (region.src_offset[1] + region.src_extent[1]) as i32,
                region.dst_offset[0] as i32,
                region.dst_offset[1] as i32,
                (region.dst_offset[0] + region.dst_extent[0]) as i32,
                (region.dst_offset[1] + region.dst_extent[1]) as i32,
                blit_mask_bits(masks),
                blit_filter(filter),
            );
            let result = self.driver_error(OP);
            self.gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            result
        }
    }
}

/// `glInvalidateFramebuffer` is core ES 3.x and desktop 4.3+.
fn supports_framebuffer_invalidate(profile: super::super::GlFamilyProfile) -> bool {
    match profile {
        super::super::GlFamilyProfile::Embedded { .. } => true,
        super::super::GlFamilyProfile::Desktop { major, minor } => major > 4 || minor >= 3,
        super::super::GlFamilyProfile::WebGl2 => true,
    }
}

impl NativeGlProvider<'_> {
    /// Attaches every validated view of a descriptor to one fresh FBO.
    fn attach_all(
        &self,
        op: &'static str,
        raw: glow::NativeFramebuffer,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; every attachment was validated.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(raw));
            for (index, view) in descriptor.color_attachments.iter().enumerate() {
                let texture = self.attachment_raw(op, *view)?;
                self.gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    draw_buffer_constant(index as u32),
                    glow::TEXTURE_2D,
                    Some(texture),
                    view.mip_level as i32,
                );
            }
            if let Some(view) = descriptor.depth_stencil_attachment {
                let point = depth_attachment_point(view.format).ok_or_else(|| {
                    Self::validation(op, "depth attachment format has no attachment point")
                })?;
                let texture = self.attachment_raw(op, view)?;
                self.gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    point,
                    glow::TEXTURE_2D,
                    Some(texture),
                    view.mip_level as i32,
                );
            }
        }
        Ok(())
    }

    fn attachment_raw(
        &self,
        op: &'static str,
        view: GlTextureView,
    ) -> Result<glow::NativeTexture, GlError> {
        let GlAttachmentTarget::Texture(texture) = view.target else {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "surface-image attachments are not part of this framebuffer slice",
            });
        };
        Ok(self.texture(op, texture)?.0)
    }

    /// Validates one attachment view against the live allocation it names.
    fn validate_attachment(&self, op: &'static str, view: GlTextureView) -> Result<(), GlError> {
        let GlAttachmentTarget::Texture(texture) = view.target else {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "surface-image attachments are not part of this framebuffer slice",
            });
        };
        let (_, desc) = self.texture(op, texture)?;
        if desc.format != view.format {
            return Err(Self::validation(
                op,
                "attachment view format does not match the allocation",
            ));
        }
        let Some(mip) = desc.mip_extent(view.mip_level) else {
            return Err(Self::validation(op, "attachment mip level is invalid"));
        };
        if view.width != mip.width || view.height != mip.height {
            return Err(Self::validation(
                op,
                "attachment view extent does not match the mip extent",
            ));
        }
        if view.array_layer != 0 || desc.dimension != GlTextureDimension::D2 {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "layered attachments are not part of this framebuffer slice",
            });
        }
        if desc.sample_count != view.sample_count {
            return Err(Self::validation(
                op,
                "attachment sample count does not match the allocation",
            ));
        }
        let facts = self.discovery.formats().get_for(
            super::super::GlFormatResourceKind::Texture,
            view.format,
            1,
        );
        if facts.is_none_or(|facts| !facts.renderable) {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "attachment format lacks renderable evidence on this context",
            });
        }
        Ok(())
    }
}
