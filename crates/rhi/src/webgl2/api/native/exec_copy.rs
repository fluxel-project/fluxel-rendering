//! Native copy, upload, and readback execution.
//!
//! Texture copy has two honest routes: `glCopyImageSubData` when the profile's
//! core version supplies it (every ES 3.x and desktop GL 4.3+ context), and
//! otherwise the framebuffer route shared with the browser backend (a scratch
//! framebuffer served by `glCopyTexSubImage2D` / `glReadPixels`). The route
//! decision is a pure function of the discovered profile fact, recorded in
//! [`texture_copy_route`]. Region coordinates follow the GL bottom-left
//! convention; orientation is a Layer 2/3 concern.

use std::borrow::Cow;

use super::super::{
    GlBufferRange, GlCopyDomainApi, GlError, GlFamilyApi as _, GlFormatResourceKind, GlPixelFormat,
    GlPixelLayout, GlPixelStoreState, GlReadback, GlRepackPolicy, GlTextureDesc,
    GlTextureDimension, GlTextureRegion, GlTextureUsage, validate_texture_copy,
};
use super::provider::NativeGlProvider;

impl GlCopyDomainApi for NativeGlProvider<'_> {
    fn copy_buffer_range(
        &mut self,
        source: GlBufferRange,
        destination: GlBufferRange,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "copy-buffer";
        self.assert_ready(OP)?;
        let (source_name, source_desc) = self.buffer(OP, source.buffer)?;
        let (destination_name, destination_desc) = self.buffer(OP, destination.buffer)?;
        source
            .validate_for(source_desc)
            .and_then(|_| destination.validate_for(destination_desc))
            .map_err(|_| Self::validation(OP, "invalid buffer range"))?;
        if source.size != destination.size {
            return Err(Self::validation(OP, "copy sizes differ"));
        }
        let read_offset = i32::try_from(source.offset)
            .map_err(|_| Self::validation(OP, "source offset exceeds GLintptr"))?;
        let write_offset = i32::try_from(destination.offset)
            .map_err(|_| Self::validation(OP, "destination offset exceeds GLintptr"))?;
        let size = i32::try_from(source.size)
            .map_err(|_| Self::validation(OP, "copy size exceeds GLsizeiptr"))?;
        // The copy targets are Layer 1-private scratch: bind immediately before
        // use, no restore on return (`GlCopyDomainApi` documents why).
        // SAFETY: current-context contract; both live resources and all ranges
        // were validated before bindings or the copy command are changed.
        unsafe {
            self.gl
                .bind_buffer(glow::COPY_READ_BUFFER, Some(source_name));
            self.gl
                .bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination_name));
            self.gl.copy_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                glow::COPY_WRITE_BUFFER,
                read_offset,
                write_offset,
                size,
            );
        }
        self.driver_error(OP)
    }

    fn copy_texture_region(
        &mut self,
        source: GlTextureRegion,
        destination: GlTextureRegion,
    ) -> Result<(), GlError> {
        const OP: &str = "copy-texture";
        self.assert_ready(OP)?;
        let (source_name, source_desc) = self.texture(OP, source.subresource.texture)?;
        let (destination_name, destination_desc) =
            self.texture(OP, destination.subresource.texture)?;
        validate_texture_copy(source, source_desc, destination, destination_desc)
            .map_err(|_| Self::validation(OP, "invalid texture copy"))?;
        if source.subresource.texture == destination.subresource.texture
            && source.subresource.mip_level == destination.subresource.mip_level
        {
            // Reading and writing one mip is a driver-dependent feedback loop;
            // reject deterministically instead of trusting the driver check.
            return Err(Self::validation(
                OP,
                "copy source and destination name the same mip",
            ));
        }
        if source_desc.sample_count != 1 || destination_desc.sample_count != 1 {
            // The shared copy semantic is single-sample; multisample transfer
            // is a resolve word (`blit_framebuffer`), not a texture copy.
            return Err(Self::validation(
                OP,
                "copy operates on single-sample textures",
            ));
        }
        if !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !destination_desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
        {
            return Err(Self::validation(
                OP,
                "copy source or destination usage is missing",
            ));
        }
        let facts = self.discovery.formats();
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
        match texture_copy_route(self.discovery.context().profile()) {
            CopyRoute::DirectImage => {
                self.copy_texture_direct(OP, source, destination, source_name, destination_name)
            }
            CopyRoute::Framebuffer => self.copy_texture_framebuffer(
                OP,
                source,
                destination,
                source_name,
                destination_name,
            ),
        }
    }

    fn upload_buffer(&mut self, destination: GlBufferRange, bytes: &[u8]) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "upload-buffer";
        self.assert_ready(OP)?;
        let (name, desc) = self.buffer(OP, destination.buffer)?;
        destination
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid destination buffer range"))?;
        if u64::try_from(bytes.len()).ok() != Some(destination.size) {
            return Err(Self::validation(OP, "byte length differs from the range"));
        }
        let offset = i32::try_from(destination.offset)
            .map_err(|_| Self::validation(OP, "offset exceeds GLintptr"))?;
        // COPY_WRITE_BUFFER keeps ARRAY_BUFFER and ELEMENT_ARRAY_BUFFER
        // vertex-state bindings untouched by transfer work.  The copy targets
        // themselves are Layer 1-private scratch: bound immediately before use,
        // not restored on return (`GlCopyDomainApi` documents why).
        // SAFETY: current-context contract; the exact byte slice was validated.
        unsafe {
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
            self.gl
                .buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset, bytes);
        }
        self.driver_error(OP)
    }

    fn read_buffer(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError> {
        use glow::HasContext as _;
        const OP: &str = "read-buffer";
        self.assert_ready(OP)?;
        let (name, desc) = self.buffer(OP, source.buffer)?;
        source
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid source buffer range"))?;
        let offset = i32::try_from(source.offset)
            .map_err(|_| Self::validation(OP, "offset exceeds GLintptr"))?;
        let mut out = vec![
            0u8;
            usize::try_from(source.size)
                .map_err(|_| GlError::OutOfMemory { operation: OP })?
        ];
        // SAFETY: current-context contract; the exact byte slice was validated.
        unsafe {
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(name));
            self.gl
                .get_buffer_sub_data(glow::COPY_READ_BUFFER, offset, &mut out);
        }
        self.driver_error(OP)?;
        Ok(out)
    }

    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "upload-texture";
        self.assert_ready(OP)?;
        let (name, desc) = self.texture(OP, destination.subresource.texture)?;
        destination
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid texture region"))?;
        if desc.format.compressed_info().is_some() {
            return self.upload_compressed_mip(OP, name, desc, destination, bytes);
        }
        // Depth storage is filled through clears and FBO work in GL, never
        // from client pixels in this shared semantic; every other unmapped
        // format fails closed.
        let Some((gl_format, gl_type)) = upload_encoding(desc.format) else {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format accepts no CPU pixel upload in this shared semantic",
            });
        };
        transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no native transfer route",
        })?;
        ensure_single_2d_rect(OP, destination)?;
        let needed = layout
            .required_bytes(destination)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        if u64::try_from(bytes.len()).ok() != Some(needed) {
            return Err(Self::validation(
                OP,
                "upload source length differs from layout",
            ));
        }
        let (level, x, y, width, height) = rect_coords(OP, destination)?;
        let payload = staged_unpack(OP, layout, destination, bytes)?;

        let saved = self.pixel_store;
        // SAFETY: current-context contract; the layout was validated and the
        // exact tracked pixel-store state is restored on every return path.
        let result = unsafe {
            self.apply_unpack(layout);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(name));
            self.gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                level,
                x,
                y,
                width,
                height,
                gl_format,
                gl_type,
                glow::PixelUnpackData::Slice(Some(payload.as_ref())),
            );
            self.driver_error(OP)
        };
        // SAFETY: current-context contract; exact snapshot restoration.
        unsafe { self.restore_pixel_store(UnpackDirection::Unpack, saved) };
        result
    }

    fn read_texture(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        use glow::HasContext as _;
        const OP: &str = "read-texture";
        self.assert_ready(OP)?;
        let (name, desc) = self.texture(OP, source.subresource.texture)?;
        source
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid read region"))?;
        let (gl_format, gl_type) = readback_encoding(desc.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no native readback encoding",
        })?;
        transfer_format(layout.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "pixel encoding has no native readback route",
        })?;
        ensure_single_2d_rect(OP, source)?;
        let needed = layout
            .required_bytes(source)
            .map_err(|_| Self::validation(OP, "invalid pixel layout"))?;
        // GL fills exactly the offset-trimmed body; the layout offset is a
        // client-side placement applied by this module afterwards.
        let body = needed
            .checked_sub(layout.offset)
            .filter(|body| *body > 0)
            .ok_or_else(|| Self::validation(OP, "layout offset leaves no readback body"))?;
        let (level, x, y, width, height) = rect_coords(OP, source)?;
        let body_len = usize::try_from(body).map_err(|_| GlError::OutOfMemory { operation: OP })?;
        let needed_len =
            usize::try_from(needed).map_err(|_| GlError::OutOfMemory { operation: OP })?;
        // Scratch framebuffer lifecycle: created, completed, used, and deleted
        // on every path so no half-initialized object survives a failure.
        // SAFETY: current-context contract; the scratch framebuffer exists
        // only inside this block and is unbound/deleted before every return.
        unsafe {
            let Ok(scratch) = self.gl.create_framebuffer() else {
                return Err(GlError::OutOfMemory { operation: OP });
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scratch));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(name),
                level,
            );
            let mut outcome = match self.require_complete(OP) {
                Ok(()) => None,
                Err(error) => Some(Err(error)),
            };
            if outcome.is_none() {
                let saved = self.pixel_store;
                self.apply_pack(layout);
                let mut body_bytes = vec![0u8; body_len];
                self.gl.read_pixels(
                    x,
                    y,
                    width,
                    height,
                    gl_format,
                    gl_type,
                    glow::PixelPackData::Slice(Some(&mut body_bytes)),
                );
                let read = self.driver_error(OP);
                self.restore_pixel_store(UnpackDirection::Pack, saved);
                outcome = match read {
                    Ok(()) => {
                        let mut bytes = vec![0u8; needed_len];
                        place_rows(layout, source, &body_bytes, &mut bytes);
                        Some(Ok(GlReadback { layout, bytes }))
                    }
                    Err(error) => Some(Err(error)),
                };
            }
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_framebuffer(scratch);
            // The teardown must not leak its own error into the caller's next
            // observation.
            let _ = self.gl.get_error();
            outcome.expect("readback outcome was computed on every path")
        }
    }

    fn pixel_store(&self) -> GlPixelStoreState {
        self.pixel_store
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnpackDirection {
    Unpack,
    Pack,
}

/// The texture-copy route the discovered profile fact selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CopyRoute {
    /// `glCopyImageSubData` is core: ES 3.x always, desktop from 4.3.
    DirectImage,
    /// Older desktop cores use the scratch-framebuffer route.
    Framebuffer,
}

fn texture_copy_route(profile: super::super::GlFamilyProfile) -> CopyRoute {
    match profile {
        super::super::GlFamilyProfile::Embedded { .. } => CopyRoute::DirectImage,
        super::super::GlFamilyProfile::Desktop { major, minor } => {
            if major > 4 || (major == 4 && minor >= 3) {
                CopyRoute::DirectImage
            } else {
                CopyRoute::Framebuffer
            }
        }
        super::super::GlFamilyProfile::WebGl2 => CopyRoute::Framebuffer,
    }
}

impl NativeGlProvider<'_> {
    fn copy_texture_direct(
        &mut self,
        op: &'static str,
        source: GlTextureRegion,
        destination: GlTextureRegion,
        source_name: glow::NativeTexture,
        destination_name: glow::NativeTexture,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let src_level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let dst_level = i32::try_from(destination.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let (_, dx, dy, width, height) = rect_coords(op, destination)?;
        let sx =
            i32::try_from(source.origin[0]).map_err(|_| Self::validation(op, "x exceeds GLint"))?;
        let sy =
            i32::try_from(source.origin[1]).map_err(|_| Self::validation(op, "y exceeds GLint"))?;
        // SAFETY: current-context contract; both live textures and the exact
        // rectangle were validated, and identical-format copies need no
        // format-conversion precondition.
        unsafe {
            self.gl.copy_image_sub_data(
                source_name,
                glow::TEXTURE_2D,
                src_level,
                sx,
                sy,
                0,
                destination_name,
                glow::TEXTURE_2D,
                dst_level,
                dx,
                dy,
                0,
                width,
                height,
                1,
            );
        }
        self.driver_error(op)
    }

    /// Framebuffer route for desktop cores before 4.3, mirroring the browser
    /// backend: a per-operation scratch framebuffer feeds `glCopyTexSubImage2D`.
    fn copy_texture_framebuffer(
        &mut self,
        op: &'static str,
        source: GlTextureRegion,
        destination: GlTextureRegion,
        source_name: glow::NativeTexture,
        destination_name: glow::NativeTexture,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let src_level = i32::try_from(source.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let (level, dx, dy, width, height) = rect_coords(op, destination)?;
        let sx =
            i32::try_from(source.origin[0]).map_err(|_| Self::validation(op, "x exceeds GLint"))?;
        let sy =
            i32::try_from(source.origin[1]).map_err(|_| Self::validation(op, "y exceeds GLint"))?;
        // SAFETY: current-context contract; the scratch framebuffer exists only
        // inside this block and is unbound/deleted before every return.
        unsafe {
            let Ok(scratch) = self.gl.create_framebuffer() else {
                return Err(GlError::OutOfMemory { operation: op });
            };
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scratch));
            self.gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(source_name),
                src_level,
            );
            let mut result = match self.require_complete(op) {
                Ok(()) => None,
                Err(error) => Some(Err(error)),
            };
            if result.is_none() {
                self.gl
                    .bind_texture(glow::TEXTURE_2D, Some(destination_name));
                self.gl.copy_tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    level,
                    dx,
                    dy,
                    sx,
                    sy,
                    width,
                    height,
                );
                result = Some(self.driver_error(op));
            }
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_framebuffer(scratch);
            let _ = self.gl.get_error();
            result.expect("copy outcome was computed on every path")
        }
    }

    /// Compressed mips stay on the whole-mip route: storage is undefined until
    /// the first `compressedTexImage2D` defines it (audit P2-7).
    fn upload_compressed_mip(
        &mut self,
        op: &'static str,
        name: glow::NativeTexture,
        desc: GlTextureDesc,
        destination: GlTextureRegion,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        let Some(info) = desc.format.compressed_info() else {
            return Err(Self::validation(op, "compressed info disappeared"));
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
            .map_err(|_| Self::validation(op, "compressed encoded size overflow"))?;
        if u64::try_from(bytes.len()).ok() != Some(exact) {
            return Err(Self::validation(
                op,
                "compressed bytes do not match exact block layout",
            ));
        }
        let internal = super::provider::native_texture_format(desc.format)
            .expect("proven compressed format is mapped");
        let level = i32::try_from(destination.subresource.mip_level)
            .map_err(|_| Self::validation(op, "mip level exceeds GLint"))?;
        let width = i32::try_from(destination.extent.width)
            .map_err(|_| Self::validation(op, "width exceeds GLsizei"))?;
        let height = i32::try_from(destination.extent.height)
            .map_err(|_| Self::validation(op, "height exceeds GLsizei"))?;
        let size = i32::try_from(exact)
            .map_err(|_| Self::validation(op, "compressed upload exceeds GLsizei"))?;
        // SAFETY: current-context contract; complete mip and exact block bytes were validated.
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(name));
            self.gl.compressed_tex_image_2d(
                glow::TEXTURE_2D,
                level,
                internal as i32,
                width,
                height,
                0,
                size,
                bytes,
            );
        }
        self.driver_error(op)
    }

    /// Applies the unpack half of one transfer layout before `texSubImage2D`.
    ///
    /// The client offset is applied by slicing the payload, so GL skip
    /// parameters stay zero and the tracked snapshot resumes exactly.
    ///
    /// # Safety
    ///
    /// Current-context contract.
    unsafe fn apply_unpack(&self, layout: GlPixelLayout) {
        use glow::HasContext as _;
        let bpp = layout.format.bytes_per_pixel();
        unsafe {
            self.gl
                .pixel_store_i32(glow::UNPACK_ALIGNMENT, i32::from(layout.alignment));
            self.gl
                .pixel_store_i32(glow::UNPACK_ROW_LENGTH, (layout.bytes_per_row / bpp) as i32);
            self.gl
                .pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, layout.rows_per_image as i32);
            self.gl.pixel_store_i32(glow::UNPACK_SKIP_PIXELS, 0);
            self.gl.pixel_store_i32(glow::UNPACK_SKIP_ROWS, 0);
            self.gl.pixel_store_i32(glow::UNPACK_SKIP_IMAGES, 0);
        }
    }

    /// Applies the pack half of one transfer layout before `readPixels`.
    ///
    /// # Safety
    ///
    /// Current-context contract.
    unsafe fn apply_pack(&self, layout: GlPixelLayout) {
        use glow::HasContext as _;
        let bpp = layout.format.bytes_per_pixel();
        let mappable = layout.bytes_per_row.is_multiple_of(bpp)
            && layout
                .bytes_per_row
                .is_multiple_of(u32::from(layout.alignment));
        unsafe {
            if mappable {
                self.gl
                    .pixel_store_i32(glow::PACK_ALIGNMENT, i32::from(layout.alignment));
                self.gl
                    .pixel_store_i32(glow::PACK_ROW_LENGTH, (layout.bytes_per_row / bpp) as i32);
            } else {
                // A bounded repack reads tightly and is placed client-side.
                self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                self.gl.pixel_store_i32(glow::PACK_ROW_LENGTH, 0);
            }
        }
    }

    /// Restores the exact snapshot so tracked pixel-store state survives
    /// every return path, including driver failures.
    ///
    /// # Safety
    ///
    /// Current-context contract.
    unsafe fn restore_pixel_store(&mut self, direction: UnpackDirection, saved: GlPixelStoreState) {
        use glow::HasContext as _;
        unsafe {
            match direction {
                UnpackDirection::Unpack => {
                    self.gl
                        .pixel_store_i32(glow::UNPACK_ALIGNMENT, i32::from(saved.unpack_alignment));
                    self.gl
                        .pixel_store_i32(glow::UNPACK_ROW_LENGTH, saved.unpack_row_length as i32);
                    self.gl.pixel_store_i32(
                        glow::UNPACK_IMAGE_HEIGHT,
                        saved.unpack_image_height as i32,
                    );
                    self.gl
                        .pixel_store_i32(glow::UNPACK_SKIP_PIXELS, saved.unpack_skip_pixels as i32);
                    self.gl
                        .pixel_store_i32(glow::UNPACK_SKIP_ROWS, saved.unpack_skip_rows as i32);
                    self.gl
                        .pixel_store_i32(glow::UNPACK_SKIP_IMAGES, saved.unpack_skip_images as i32);
                }
                UnpackDirection::Pack => {
                    self.gl
                        .pixel_store_i32(glow::PACK_ALIGNMENT, i32::from(saved.pack_alignment));
                    self.gl
                        .pixel_store_i32(glow::PACK_ROW_LENGTH, saved.pack_row_length as i32);
                    self.gl
                        .pixel_store_i32(glow::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32);
                    self.gl
                        .pixel_store_i32(glow::PACK_SKIP_ROWS, saved.pack_skip_rows as i32);
                }
            }
            self.pixel_store = saved;
        }
    }
}

/// The `(format, type)` pair a native upload accepts for one texture format.
///
/// ES 3.0 and desktop GL 4.x accept exactly one `(format, type)` pair per
/// sized internal format, so unsupported pairs return `None` instead of a
/// driver-dependent guess.
fn upload_encoding(format: super::super::GlFormat) -> Option<(u32, u32)> {
    match format {
        super::super::GlFormat::Rgba8Unorm | super::super::GlFormat::Rgba8Srgb => {
            Some((glow::RGBA, glow::UNSIGNED_BYTE))
        }
        _ => None,
    }
}

/// The readback encoding this backend returns for a discovered texture format.
///
/// `RGBA`/`UNSIGNED_BYTE` is the only combination every accepted profile must
/// support through the framebuffer route.
fn readback_encoding(format: super::super::GlFormat) -> Option<(u32, u32)> {
    upload_encoding(format)
}

/// The client pixel format a transfer must use, if this backend accepts it.
fn transfer_format(format: GlPixelFormat) -> Option<()> {
    match format {
        GlPixelFormat::Rgba8 => Some(()),
        // Native GLES has no BGRA upload and the discovered RGBA8 textures
        // accept no RGB/RED encoding; every one of those rejects before
        // touching pixel state.
        _ => None,
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

fn validation(operation: &'static str, message: &'static str) -> GlError {
    GlError::Validation {
        operation,
        message: message.into(),
    }
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
    let mappable = layout.bytes_per_row.is_multiple_of(bpp)
        && layout
            .bytes_per_row
            .is_multiple_of(u32::from(layout.alignment));
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
    let mappable = layout.bytes_per_row.is_multiple_of(bpp as u32)
        && layout
            .bytes_per_row
            .is_multiple_of(u32::from(layout.alignment));
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
