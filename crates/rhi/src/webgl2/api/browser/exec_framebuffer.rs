//! Browser framebuffer, render-pass, and blit execution.
//!
//! Pass load/clear executes exactly once per pass through the `clearBuffer*`
//! commands, which honor scissor state but ignore color and depth masks; the
//! pass therefore disables scissor around its clears, and pipeline
//! application restores the scissor group afterwards. Resolve stays an
//! explicit `blit_framebuffer` from the multisample source, per the shared
//! framebuffer contract.

use js_sys::Array;
use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer};

use super::super::{
    FramebufferId, GlAttachmentTarget, GlBlitMask, GlBlitRegion, GlDepthStencilAttachment, GlError,
    GlFamilyApi as _, GlFilterMode, GlFramebufferApi, GlFramebufferDescriptor, GlLoadOp,
    GlRenderPassDescriptor, GlStoreOp, GlTextureDimension, GlTextureView,
};
use super::discovery::WebGl2BrowserDiscovery;
use super::format_map;
use super::objects::{ActivePass, BrowserFramebuffer};

const fn blit_mask_bits(masks: GlBlitMask) -> u32 {
    let mut bits = 0;
    if masks.color {
        bits |= Gl::COLOR_BUFFER_BIT;
    }
    if masks.depth {
        bits |= Gl::DEPTH_BUFFER_BIT;
    }
    if masks.stencil {
        bits |= Gl::STENCIL_BUFFER_BIT;
    }
    bits
}

const fn blit_filter(filter: GlFilterMode) -> u32 {
    match filter {
        GlFilterMode::Nearest => Gl::NEAREST,
        GlFilterMode::Linear => Gl::LINEAR,
    }
}

/// The attachment constant of one `drawBuffers` selection entry.
const fn draw_buffer_constant(index: u32) -> u32 {
    Gl::COLOR_ATTACHMENT0 + index
}

fn attachment_list(values: &[u32]) -> JsValue {
    Array::from_iter(
        values
            .iter()
            .map(|value| JsValue::from_f64(f64::from(*value))),
    )
    .into()
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

impl GlFramebufferApi for WebGl2BrowserDiscovery {
    fn create_framebuffer(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        const OP: &str = "create-framebuffer";
        self.assert_provider_ready(OP)?;
        for view in descriptor
            .color_attachments
            .iter()
            .copied()
            .chain(descriptor.depth_stencil_attachment)
        {
            self.validate_attachment(OP, view)?;
        }
        let limits = self.discovery().limits();
        descriptor
            .validate(
                limits.max_color_attachments,
                limits.max_draw_buffers,
                self.context_stamp(),
            )
            .map_err(|_| Self::validation(OP, "invalid framebuffer descriptor"))?;
        let raw = self
            .raw
            .create_framebuffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        let attached = self.attach_all(OP, &raw, descriptor);
        // Depth-only framebuffers must not keep a draw buffer selected.
        if descriptor.color_attachments.is_empty() {
            self.raw.draw_buffers(&attachment_list(&[]));
        } else if !descriptor.draw_buffers.is_empty() {
            let constants: Vec<u32> = descriptor
                .draw_buffers
                .iter()
                .map(|index| draw_buffer_constant(*index))
                .collect();
            self.raw.draw_buffers(&attachment_list(&constants));
        }
        let completed = attached.and_then(|()| self.require_complete(OP));
        if let Err(error) = completed {
            self.cleanup_framebuffer(&raw);
            return Err(error);
        }
        let result = self.driver_error(OP);
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        if let Err(error) = result {
            self.cleanup_framebuffer(&raw);
            return Err(error);
        }
        let slot = Self::allocate_slot(&mut self.next_framebuffer_slot, OP)?;
        let id = FramebufferId::new(self.context_stamp(), slot, 0);
        self.framebuffers.insert(
            slot,
            BrowserFramebuffer {
                generation: id.generation,
                raw,
                descriptor: descriptor.clone(),
            },
        );
        Ok(id)
    }

    fn destroy_framebuffer(&mut self, framebuffer: FramebufferId) -> Result<(), GlError> {
        const OP: &str = "destroy-framebuffer";
        self.framebuffer(OP, framebuffer)?;
        let entry = self
            .framebuffers
            .remove(&framebuffer.slot)
            .ok_or_else(|| Self::validation(OP, "framebuffer disappeared"))?;
        self.raw.delete_framebuffer(Some(&entry.raw));
        self.driver_error(OP)
    }

    fn begin_render_pass(&mut self, descriptor: &GlRenderPassDescriptor) -> Result<(), GlError> {
        const OP: &str = "begin-render-pass";
        self.assert_provider_ready(OP)?;
        if self.pass.is_some() {
            return Err(Self::validation(OP, "render pass already active"));
        }
        let record = self.framebuffer(OP, descriptor.framebuffer)?;
        descriptor
            .validate(
                &record.descriptor,
                self.discovery().limits().max_color_attachments,
                self.discovery().limits().max_draw_buffers,
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
                && !format_map::has_stencil_plane(depth_stencil.view.format)
            {
                return Err(Self::validation(
                    OP,
                    "attachment has no stencil plane to clear",
                ));
            }
        }
        let raw = record.raw.clone();
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

        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&raw));
        if !record.descriptor.draw_buffers.is_empty() {
            let constants: Vec<u32> = record
                .descriptor
                .draw_buffers
                .iter()
                .map(|index| draw_buffer_constant(*index))
                .collect();
            self.raw.draw_buffers(&attachment_list(&constants));
        }
        // clearBuffer* honors scissor only; make pass-load clears total. The
        // installed pipeline re-applies the scissor group after pass begin.
        self.raw.disable(Gl::SCISSOR_TEST);
        for (index, attachment) in descriptor.color_attachments.iter().enumerate() {
            if attachment.load == GlLoadOp::Clear {
                let clear = &attachment.clear;
                self.raw.clear_bufferfv_with_f32_array(
                    Gl::COLOR,
                    index as i32,
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
                && format_map::has_stencil_plane(attachment.view.format);
            match (depth_clears, stencil_clears) {
                (true, true) => self.raw.clear_bufferfi(
                    Gl::DEPTH_STENCIL,
                    0,
                    f32::from_bits(attachment.clear.depth),
                    attachment.clear.stencil as i32,
                ),
                (true, false) => self.raw.clear_bufferfv_with_f32_array(
                    Gl::DEPTH,
                    0,
                    &[f32::from_bits(attachment.clear.depth)],
                ),
                (false, true) => self.raw.clear_bufferuiv_with_u32_array(
                    Gl::STENCIL,
                    0,
                    &[attachment.clear.stencil],
                ),
                (false, false) => {}
            }
        }
        if let Err(error) = self.driver_error(OP) {
            self.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
            return Err(error);
        }
        self.pass = Some(ActivePass {
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
        const OP: &str = "end-render-pass";
        self.assert_provider_ready(OP)?;
        let pass = self
            .pass
            .take()
            .ok_or_else(|| Self::validation(OP, "no active render pass"))?;
        self.raster = None;
        let raw = self.framebuffer(OP, pass.framebuffer)?.raw.clone();
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&raw));
        let mut invalidate: Vec<u32> = pass
            .discard_color
            .iter()
            .enumerate()
            .filter(|(_, discard)| **discard)
            .map(|(index, _)| draw_buffer_constant(index as u32))
            .collect();
        if pass.discard_depth_stencil == Some(true) {
            invalidate.push(Gl::DEPTH_ATTACHMENT);
        }
        let mut result = Ok(());
        if !invalidate.is_empty() {
            result = self
                .raw
                .invalidate_framebuffer(Gl::FRAMEBUFFER, &attachment_list(&invalidate))
                .map_err(|value| GlError::Driver {
                    operation: OP,
                    message: format!("browser exception: {value:?}"),
                })
                .and_then(|()| self.driver_error(OP));
        }
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        result
    }

    fn blit_framebuffer(
        &mut self,
        source: FramebufferId,
        destination: FramebufferId,
        region: GlBlitRegion,
        filter: GlFilterMode,
        masks: GlBlitMask,
    ) -> Result<(), GlError> {
        const OP: &str = "blit-framebuffer";
        self.assert_provider_ready(OP)?;
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
        let source_raw = self.framebuffer(OP, source)?.raw.clone();
        let destination_raw = self.framebuffer(OP, destination)?.raw.clone();
        self.raw
            .bind_framebuffer(Gl::READ_FRAMEBUFFER, Some(&source_raw));
        self.raw
            .bind_framebuffer(Gl::DRAW_FRAMEBUFFER, Some(&destination_raw));
        self.raw.blit_framebuffer(
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
        self.raw.bind_framebuffer(Gl::READ_FRAMEBUFFER, None);
        self.raw.bind_framebuffer(Gl::DRAW_FRAMEBUFFER, None);
        result
    }
}

impl WebGl2BrowserDiscovery {
    fn cleanup_framebuffer(&self, raw: &WebGlFramebuffer) {
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        self.raw.delete_framebuffer(Some(raw));
    }

    /// Attaches every validated view of a descriptor to one fresh FBO.
    fn attach_all(
        &mut self,
        op: &'static str,
        raw: &WebGlFramebuffer,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<(), GlError> {
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(raw));
        for (index, view) in descriptor.color_attachments.iter().enumerate() {
            let texture = self.attachment_raw(op, *view)?;
            self.raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                draw_buffer_constant(index as u32),
                Gl::TEXTURE_2D,
                Some(&texture),
                view.mip_level as i32,
            );
        }
        if let Some(view) = descriptor.depth_stencil_attachment {
            let point = format_map::depth_attachment_point(view.format).ok_or_else(|| {
                Self::validation(op, "depth attachment format has no attachment point")
            })?;
            let texture = self.attachment_raw(op, view)?;
            self.raw.framebuffer_texture_2d(
                Gl::FRAMEBUFFER,
                point,
                Gl::TEXTURE_2D,
                Some(&texture),
                view.mip_level as i32,
            );
        }
        Ok(())
    }

    fn attachment_raw(
        &self,
        op: &'static str,
        view: GlTextureView,
    ) -> Result<web_sys::WebGlTexture, GlError> {
        let GlAttachmentTarget::Texture(texture) = view.target else {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "surface-image attachments are not part of this framebuffer slice",
            });
        };
        Ok(self.texture(op, texture)?.raw.clone())
    }

    /// Validates one attachment view against the live allocation it names.
    fn validate_attachment(&self, op: &'static str, view: GlTextureView) -> Result<(), GlError> {
        let GlAttachmentTarget::Texture(texture) = view.target else {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "surface-image attachments are not part of this framebuffer slice",
            });
        };
        let desc = self.texture(op, texture)?.desc;
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
        let facts = self.snapshot.formats().get_for(
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
