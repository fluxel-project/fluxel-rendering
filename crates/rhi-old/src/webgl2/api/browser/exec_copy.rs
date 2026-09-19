//! Browser copy, upload, and readback execution.
//!
//! WebGL2 exposes no direct texture-to-texture copy, so `copy_texture_region`
//! and `read_texture` honestly take the framebuffer route: the source region
//! is attached to a per-operation scratch framebuffer and served with
//! `copyTexSubImage2D` / `readPixels`. This is the backend-provided route for
//! the common copy semantic (plan functional order 3), not a claim that a
//! `glCopyImageSubData`-style command exists. Region coordinates follow the
//! GL bottom-left convention; orientation is a Layer 2/3 concern.

use std::borrow::Cow;
use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer};

use super::super::{
    GlBufferRange, GlCopyDomainApi, GlError, GlFormatResourceKind, GlPixelLayout,
    GlPixelStoreState, GlReadback, GlRepackPolicy, GlTextureDesc, GlTextureDimension,
    GlTextureRegion, GlTextureUsage, validate_texture_copy,
};
use super::discovery::WebGl2BrowserDiscovery;

impl GlCopyDomainApi for WebGl2BrowserDiscovery {
    fn copy_buffer_range(
        &mut self,
        source: GlBufferRange,
        destination: GlBufferRange,
    ) -> Result<(), GlError> {
        const OP: &str = "copy-buffer";
        self.assert_provider_ready(OP)?;
        let (source_raw, source_desc) = {
            let entry = self.buffer(OP, source.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        let (destination_raw, destination_desc) = {
            let entry = self.buffer(OP, destination.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        source
            .validate_for(source_desc)
            .map_err(|_| validation(OP, "invalid source buffer range"))?;
        destination
            .validate_for(destination_desc)
            .map_err(|_| validation(OP, "invalid destination buffer range"))?;
        if source.size != destination.size {
            return Err(validation(OP, "copy ranges have different sizes"));
        }
        // The copy targets are Layer 1-private scratch: bind immediately before
        // use, no restore on return (`GlCopyDomainApi` documents why).
        self.raw
            .bind_buffer(Gl::COPY_READ_BUFFER, Some(&source_raw));
        self.raw
            .bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&destination_raw));
        self.raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
            Gl::COPY_READ_BUFFER,
            Gl::COPY_WRITE_BUFFER,
            source.offset as f64,
            destination.offset as f64,
            source.size as f64,
        );
        self.driver_error(OP)
    }

    /// Copies one 2D color rectangle through a scratch framebuffer.
    ///
    /// Preconditions are validated before any state changes: identical
    /// formats, single samples, color aspect, live copy facts, and distinct
    /// mips. Everything else rejects with a structured error.
    fn copy_texture_region(
        &mut self,
        source: GlTextureRegion,
        destination: GlTextureRegion,
    ) -> Result<(), GlError> {
        const OP: &str = "copy-texture";
        self.assert_provider_ready(OP)?;
        let (source_raw, source_desc) = {
            let entry = self.texture(OP, source.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        let (destination_raw, destination_desc) = {
            let entry = self.texture(OP, destination.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        validate_texture_copy(source, source_desc, destination, destination_desc)
            .map_err(|_| validation(OP, "invalid texture copy"))?;
        if source.subresource.texture == destination.subresource.texture
            && source.subresource.mip_level == destination.subresource.mip_level
        {
            // Reading and writing one mip is a driver-dependent feedback loop;
            // reject deterministically instead of trusting the driver check.
            return Err(validation(
                OP,
                "copy source and destination name the same mip",
            ));
        }
        if !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !destination_desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(validation(
                OP,
                "copy source or destination usage is missing",
            ));
        }
        let facts = self.snapshot.formats();
        let source_copy = facts
            .get_for(GlFormatResourceKind::Texture, source_desc.format, 1)
            .map(|fact| fact.copy_source)
            .unwrap_or(false);
        let destination_copy = facts
            .get_for(GlFormatResourceKind::Texture, destination_desc.format, 1)
            .map(|fact| fact.copy_destination)
            .unwrap_or(false);
        if !source_copy || !destination_copy {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks a proved copy fact on this context",
            });
        }
        ensure_single_2d_rect(OP, source)?;
        ensure_single_2d_rect(OP, destination)?;
        let (level, dst_x, dst_y, width, height) = rect_coords(OP, destination)?;
        let src_x = i32::try_from(source.origin[0])
            .map_err(|_| validation(OP, "source x exceeds GLint"))?;
        let src_y = i32::try_from(source.origin[1])
            .map_err(|_| validation(OP, "source y exceeds GLint"))?;
        let src_level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| validation(OP, "source mip level exceeds GLint"))?;

        // Scratch framebuffer lifecycle: created, completed, used, and deleted
        // on every path so no half-initialized object survives a failure.
        let scratch = self
            .raw
            .create_framebuffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&scratch));
        self.raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::TEXTURE_2D,
            Some(&source_raw),
            src_level,
        );
        if let Err(error) = self.require_complete(OP) {
            self.cleanup_scratch(&scratch);
            return Err(error);
        }
        self.raw
            .bind_texture(Gl::TEXTURE_2D, Some(&destination_raw));
        self.raw.copy_tex_sub_image_2d(
            Gl::TEXTURE_2D,
            level,
            dst_x,
            dst_y,
            src_x,
            src_y,
            width,
            height,
        );
        let result = self.driver_error(OP);
        self.cleanup_scratch(&scratch);
        result
    }

    fn upload_buffer(&mut self, destination: GlBufferRange, bytes: &[u8]) -> Result<(), GlError> {
        const OP: &str = "upload-buffer";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.buffer(OP, destination.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        destination
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid destination buffer range"))?;
        if u64::try_from(bytes.len()).ok() != Some(destination.size) {
            return Err(validation(OP, "byte length differs from the range"));
        }
        // COPY_WRITE_BUFFER keeps ARRAY_BUFFER and ELEMENT_ARRAY_BUFFER
        // vertex-state bindings untouched by transfer work, and the two copy
        // targets are the only ones an index buffer can still reach: creation
        // allocates it through ELEMENT_ARRAY_BUFFER, and this bind neither
        // revokes that association nor disturbs any bound vertex array.
        // Measured on the real adapter (AMD Radeon 780M through ANGLE/D3D11;
        // see the 0.15 series plan's 2026-09-17 browser entry) -- which is why
        // an index buffer needs no upload path of its own.
        self.raw.bind_buffer(Gl::COPY_WRITE_BUFFER, Some(&raw));
        self.raw.buffer_sub_data_with_f64_and_u8_array(
            Gl::COPY_WRITE_BUFFER,
            destination.offset as f64,
            bytes,
        );
        self.driver_error(OP)
    }

    fn read_buffer(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError> {
        const OP: &str = "read-buffer";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.buffer(OP, source.buffer)?;
            (entry.raw.clone(), entry.desc)
        };
        source
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid source buffer range"))?;
        let mut out = vec![
            0u8;
            usize::try_from(source.size)
                .map_err(|_| GlError::OutOfMemory { operation: OP })?
        ];
        self.raw.bind_buffer(Gl::COPY_READ_BUFFER, Some(&raw));
        self.raw.get_buffer_sub_data_with_f64_and_u8_array(
            Gl::COPY_READ_BUFFER,
            source.offset as f64,
            &mut out,
        );
        self.driver_error(OP)?;
        Ok(out)
    }

    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        const OP: &str = "upload-texture";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.texture(OP, destination.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        destination
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid texture region"))?;
        if desc.format.compressed_info().is_some() {
            return self.upload_compressed_mip(OP, &raw, desc, destination, bytes);
        }
        // Depth storage is filled through clears and FBO work in GL, never
        // from client pixels; every other unmapped format fails closed.
        let Some((gl_format, gl_type)) = super::format_map::upload_encoding(desc.format) else {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format accepts no CPU pixel upload in WebGL2",
            });
        };
        super::format_map::transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no WebGL2 transfer route",
        })?;
        ensure_single_2d_rect(OP, destination)?;
        let needed = layout
            .required_bytes(destination)
            .map_err(|_| validation(OP, "invalid pixel layout"))?;
        if u64::try_from(bytes.len()).ok() != Some(needed) {
            return Err(validation(OP, "upload source length differs from layout"));
        }
        let (level, x, y, width, height) = rect_coords(OP, destination)?;
        let payload = staged_unpack(OP, layout, destination, bytes)?;

        let saved = self.pixel_store;
        self.apply_unpack(layout);
        self.raw.bind_texture(Gl::TEXTURE_2D, Some(&raw));
        let result = self
            .raw
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                Gl::TEXTURE_2D,
                level,
                x,
                y,
                width,
                height,
                gl_format,
                gl_type,
                Some(payload.as_ref()),
            )
            .map_err(|value| js_failure(OP, value))
            .and_then(|()| self.driver_error(OP));
        self.restore_pixel_store(&saved, PixelDirection::Unpack);
        result
    }

    fn read_texture(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        const OP: &str = "read-texture";
        self.assert_provider_ready(OP)?;
        let (raw, desc) = {
            let entry = self.texture(OP, source.subresource.texture)?;
            (entry.raw.clone(), entry.desc)
        };
        source
            .validate_for(desc)
            .map_err(|_| validation(OP, "invalid read region"))?;
        let (gl_format, gl_type) =
            super::format_map::readback_encoding(desc.format).ok_or(GlError::Unsupported {
                operation: OP,
                reason: "format has no WebGL2 readback encoding",
            })?;
        super::format_map::transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no WebGL2 readback route",
        })?;
        ensure_single_2d_rect(OP, source)?;
        let needed = layout
            .required_bytes(source)
            .map_err(|_| validation(OP, "invalid pixel layout"))?;
        // GL fills exactly the offset-trimmed body; the layout offset is a
        // client-side placement applied by this module afterwards.
        let body = needed
            .checked_sub(layout.offset)
            .filter(|body| *body > 0)
            .ok_or_else(|| validation(OP, "layout offset leaves no readback body"))?;
        let x = i32::try_from(source.origin[0]).map_err(|_| validation(OP, "x exceeds GLint"))?;
        let y = i32::try_from(source.origin[1]).map_err(|_| validation(OP, "y exceeds GLint"))?;
        let width = i32::try_from(source.extent.width)
            .map_err(|_| validation(OP, "width exceeds GLint"))?;
        let height = i32::try_from(source.extent.height)
            .map_err(|_| validation(OP, "height exceeds GLint"))?;

        let scratch = self
            .raw
            .create_framebuffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&scratch));
        self.raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::COLOR_ATTACHMENT0,
            Gl::TEXTURE_2D,
            Some(&raw),
            source.subresource.mip_level as i32,
        );
        if let Err(error) = self.require_complete(OP) {
            self.cleanup_scratch(&scratch);
            return Err(error);
        }
        // readPixels consumes READ_FRAMEBUFFER; the FRAMEBUFFER binding above
        // selects this scratch framebuffer on both ends.
        let saved = self.pixel_store;
        let mut body_bytes =
            vec![0u8; usize::try_from(body).map_err(|_| GlError::OutOfMemory { operation: OP })?];
        self.apply_pack(layout);
        let read = self
            .raw
            .read_pixels_with_opt_u8_array(
                x,
                y,
                width,
                height,
                gl_format,
                gl_type,
                Some(&mut body_bytes),
            )
            .map_err(|value| js_failure(OP, value));
        self.restore_pixel_store(&saved, PixelDirection::Pack);
        let result = read.and_then(|()| self.driver_error(OP));
        self.cleanup_scratch(&scratch);
        result?;

        let mut bytes =
            vec![0u8; usize::try_from(needed).map_err(|_| GlError::OutOfMemory { operation: OP })?];
        place_rows(layout, source, &body_bytes, &mut bytes);
        Ok(GlReadback { layout, bytes })
    }

    fn pixel_store(&self) -> GlPixelStoreState {
        self.pixel_store
    }
}

/// Scoped pixel-store half to restore.
#[derive(Clone, Copy)]
enum PixelDirection {
    Unpack,
    Pack,
}

impl WebGl2BrowserDiscovery {
    /// Unbinds and deletes a per-operation scratch framebuffer.
    fn cleanup_scratch(&self, scratch: &WebGlFramebuffer) {
        self.raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        self.raw.delete_framebuffer(Some(scratch));
    }

    /// Applies the unpack half of one transfer layout before `texSubImage2D`.
    ///
    /// The client offset is applied by slicing the payload, so GL skip
    /// parameters stay zero and the tracked snapshot resumes exactly.
    fn apply_unpack(&mut self, layout: GlPixelLayout) {
        // The same pair the staging branches on, so WebGL2 is told the pitch of
        // the payload it is actually given rather than of the caller's buffer.
        // Reading the caller's pitch over tight staging is an out-of-bounds
        // read on the native provider and a structured error here, which is how
        // the two providers came to disagree about the same input.
        let (alignment, row_length) = layout.staged_row_parameters();
        let mappable = layout.stride_is_mappable();
        self.raw
            .pixel_storei(Gl::UNPACK_ALIGNMENT, alignment as i32);
        self.raw
            .pixel_storei(Gl::UNPACK_ROW_LENGTH, row_length as i32);
        self.raw.pixel_storei(
            Gl::UNPACK_IMAGE_HEIGHT,
            if mappable {
                layout.rows_per_image as i32
            } else {
                0
            },
        );
        self.raw.pixel_storei(Gl::UNPACK_SKIP_PIXELS, 0);
        self.raw.pixel_storei(Gl::UNPACK_SKIP_ROWS, 0);
        self.raw.pixel_storei(Gl::UNPACK_SKIP_IMAGES, 0);
    }

    /// Applies the pack half of one transfer layout before `readPixels`.
    fn apply_pack(&mut self, layout: GlPixelLayout) {
        let (alignment, row_length) = layout.staged_row_parameters();
        self.raw.pixel_storei(Gl::PACK_ALIGNMENT, alignment as i32);
        self.raw
            .pixel_storei(Gl::PACK_ROW_LENGTH, row_length as i32);
    }

    /// Restores the exact snapshot so tracked pixel-store state survives
    /// every return path, including driver and browser failures.
    fn restore_pixel_store(&mut self, saved: &GlPixelStoreState, direction: PixelDirection) {
        match direction {
            PixelDirection::Unpack => {
                self.raw
                    .pixel_storei(Gl::UNPACK_ALIGNMENT, i32::from(saved.unpack_alignment));
                self.raw
                    .pixel_storei(Gl::UNPACK_ROW_LENGTH, saved.unpack_row_length as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_IMAGE_HEIGHT, saved.unpack_image_height as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_SKIP_PIXELS, saved.unpack_skip_pixels as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_SKIP_ROWS, saved.unpack_skip_rows as i32);
                self.raw
                    .pixel_storei(Gl::UNPACK_SKIP_IMAGES, saved.unpack_skip_images as i32);
            }
            PixelDirection::Pack => {
                self.raw
                    .pixel_storei(Gl::PACK_ALIGNMENT, i32::from(saved.pack_alignment));
                self.raw
                    .pixel_storei(Gl::PACK_ROW_LENGTH, saved.pack_row_length as i32);
                self.raw
                    .pixel_storei(Gl::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32);
                self.raw
                    .pixel_storei(Gl::PACK_SKIP_ROWS, saved.pack_skip_rows as i32);
            }
        }
        self.pixel_store = *saved;
    }

    /// Compressed mips stay on the existing route: storage is undefined until
    /// the first `compressedTexImage2D` defines it (audit P2-7).
    fn upload_compressed_mip(
        &mut self,
        op: &'static str,
        raw: &web_sys::WebGlTexture,
        desc: GlTextureDesc,
        destination: GlTextureRegion,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        let Some(info) = desc.format.compressed_info() else {
            return Err(validation(op, "compressed info disappeared"));
        };
        if desc.dimension != GlTextureDimension::D2
            || destination.subresource.base_layer != 0
            || destination.subresource.layer_count != 1
            || destination.origin != [0; 3]
            || destination.extent.depth_or_layers != 1
            || desc.mip_extent(destination.subresource.mip_level) != Some(destination.extent)
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "compressed upload must define one complete 2D mip",
            });
        }
        let exact = info
            .checked_encoded_size(destination.extent.width, destination.extent.height)
            .map_err(|_| validation(op, "compressed encoded size overflow"))?;
        if u64::try_from(bytes.len()).ok() != Some(exact) {
            return Err(validation(
                op,
                "compressed bytes do not match exact block layout",
            ));
        }
        let internal =
            super::format_map::internal_format(desc.format).ok_or(GlError::Unsupported {
                operation: op,
                reason: "compressed format has no WebGL2 internal format",
            })?;
        let level = i32::try_from(destination.subresource.mip_level)
            .map_err(|_| validation(op, "mip level exceeds GLint"))?;
        let width = i32::try_from(destination.extent.width)
            .map_err(|_| validation(op, "width exceeds GLint"))?;
        let height = i32::try_from(destination.extent.height)
            .map_err(|_| validation(op, "height exceeds GLint"))?;
        self.raw.bind_texture(Gl::TEXTURE_2D, Some(raw));
        self.raw.compressed_tex_image_2d_with_u8_array(
            Gl::TEXTURE_2D,
            level,
            internal,
            width,
            height,
            0,
            bytes,
        );
        self.driver_error(op)
    }
}

fn js_failure(operation: &'static str, value: JsValue) -> GlError {
    GlError::Driver {
        operation,
        message: format!("browser exception: {value:?}"),
    }
}

fn validation(operation: &'static str, message: &'static str) -> GlError {
    GlError::Validation {
        operation,
        message: message.into(),
    }
}

/// Rejects regions that are not one axis-aligned rectangle of one slice.
fn ensure_single_2d_rect(operation: &'static str, region: GlTextureRegion) -> Result<(), GlError> {
    if region.subresource.base_layer != 0
        || region.subresource.layer_count != 1
        || region.origin[2] != 0
        || region.extent.depth_or_layers != 1
    {
        return Err(GlError::Unsupported {
            operation,
            reason: "this copy slice transfers one 2D rectangle only",
        });
    }
    Ok(())
}

/// Converts one 2D rectangle's geometry into GLint command coordinates.
fn rect_coords(
    operation: &'static str,
    region: GlTextureRegion,
) -> Result<(i32, i32, i32, i32, i32), GlError> {
    let level = i32::try_from(region.subresource.mip_level)
        .map_err(|_| validation(operation, "mip level exceeds GLint"))?;
    let x =
        i32::try_from(region.origin[0]).map_err(|_| validation(operation, "x exceeds GLint"))?;
    let y =
        i32::try_from(region.origin[1]).map_err(|_| validation(operation, "y exceeds GLint"))?;
    let width = i32::try_from(region.extent.width)
        .map_err(|_| validation(operation, "width exceeds GLint"))?;
    let height = i32::try_from(region.extent.height)
        .map_err(|_| validation(operation, "height exceeds GLint"))?;
    Ok((level, x, y, width, height))
}

/// Produces the exact client bytes one `texSubImage2D` call reads.
///
/// GL expresses any layout whose row pitch is a whole number of pixels wide
/// and divisible by the requested alignment; otherwise a staging copy is
/// packed tightly, exactly as [`GlRepackPolicy::Bounded`] permits. The
/// returned bytes always start at the layout offset so GL skip parameters
/// remain zero.
fn staged_unpack<'a>(
    operation: &'static str,
    layout: GlPixelLayout,
    region: GlTextureRegion,
    bytes: &'a [u8],
) -> Result<Cow<'a, [u8]>, GlError> {
    let bpp = layout.format.bytes_per_pixel();
    let mappable = layout.stride_is_mappable();
    let offset = usize::try_from(layout.offset)
        .map_err(|_| validation(operation, "layout offset exceeds addressable range"))?;
    if offset > bytes.len() {
        return Err(validation(operation, "layout offset leaves the source"));
    }
    if mappable {
        return Ok(Cow::Borrowed(&bytes[offset..]));
    }
    match layout.repack {
        GlRepackPolicy::Disallow => Err(validation(
            operation,
            "row stride is not expressible and repack is disallowed",
        )),
        GlRepackPolicy::Bounded { max_bytes } => {
            let total = layout
                .required_bytes(region)
                .map_err(|_| validation(operation, "invalid pixel layout"))?;
            if total > max_bytes {
                return Err(validation(
                    operation,
                    "repack staging exceeds the declared bound",
                ));
            }
            let width_bytes = (region.extent.width as usize) * (bpp as usize);
            let row = layout.bytes_per_row as usize;
            let mut staged = Vec::with_capacity((region.extent.height as usize) * width_bytes);
            for image in 0..region.extent.depth_or_layers as usize {
                let image_base = offset + image * row * (layout.rows_per_image as usize);
                for row_index in 0..region.extent.height as usize {
                    let start = image_base + row_index * row;
                    staged.extend_from_slice(
                        bytes
                            .get(start..start + width_bytes)
                            .ok_or_else(|| validation(operation, "source ends inside a row"))?,
                    );
                }
            }
            Ok(Cow::Owned(staged))
        }
    }
}

/// Places readback rows into the caller-declared output layout.
///
/// GL rows arrive with the pitch `apply_pack` configured: mappable layouts
/// are already in place and only need the offset placement, while repacked
/// layouts arrive tightly packed and are scattered row by row.
fn place_rows(layout: GlPixelLayout, region: GlTextureRegion, body: &[u8], bytes: &mut [u8]) {
    let bpp = layout.format.bytes_per_pixel() as usize;
    let width_bytes = (region.extent.width as usize) * bpp;
    let row = layout.bytes_per_row as usize;
    let mappable = layout.stride_is_mappable();
    let offset = layout.offset as usize;
    if mappable {
        if let Some(dst) = bytes.get_mut(offset..offset + body.len()) {
            dst.copy_from_slice(body);
        }
        return;
    }
    for image in 0..region.extent.depth_or_layers as usize {
        let image_base = offset + image * row * (layout.rows_per_image as usize);
        let read_base = image * (region.extent.height as usize) * width_bytes;
        for row_index in 0..region.extent.height as usize {
            let dst = image_base + row_index * row;
            let src = read_base + row_index * width_bytes;
            if let (Some(dst), Some(src)) = (
                bytes.get_mut(dst..dst + width_bytes),
                body.get(src..src + width_bytes),
            ) {
                dst.copy_from_slice(src);
            }
        }
    }
}
