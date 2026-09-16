//! Mock transfer domain: buffer copies, uploads and readbacks, plus the
//! texture upload/readback paths.

use super::*;

impl GlCopyDomainApi for MockGlFamilyApi {
    fn copy_buffer_range(&mut self, s: GlBufferRange, d: GlBufferRange) -> Result<(), GlError> {
        self.ready("copy-buffer")?;
        let sd = self.buffer("copy-buffer", s.buffer)?;
        let dd = self.buffer("copy-buffer", d.buffer)?;
        s.validate_for(sd)
            .and_then(|_| d.validate_for(dd))
            .map_err(|_| GlError::Validation {
                operation: "copy-buffer",
                message: "invalid buffer range".into(),
            })?;
        if s.size != d.size {
            return self.invalid("copy-buffer", "copy sizes differ");
        }
        if !sd.usage.contains(GlBufferUsage::COPY_SOURCE)
            || !dd.usage.contains(GlBufferUsage::COPY_DESTINATION)
        {
            return self.invalid("copy-buffer", "copy source or destination usage is missing");
        }
        self.calls.push(MockCall::CopyBuffer {
            source: s.buffer,
            destination: d.buffer,
            size: s.size,
        });
        Ok(())
    }
    fn upload_buffer(&mut self, d: GlBufferRange, bytes: &[u8]) -> Result<(), GlError> {
        self.ready("upload-buffer")?;
        let desc = self.buffer("upload-buffer", d.buffer)?;
        d.validate_for(desc).map_err(|_| GlError::Validation {
            operation: "upload-buffer",
            message: "invalid buffer range".into(),
        })?;
        if u64::try_from(bytes.len()).ok() != Some(d.size) {
            return self.invalid("upload-buffer", "byte length mismatch");
        }
        self.calls.push(MockCall::UploadBuffer {
            buffer: d.buffer,
            offset: d.offset,
            size: d.size,
        });
        Ok(())
    }
    fn read_buffer(&mut self, s: GlBufferRange) -> Result<Vec<u8>, GlError> {
        self.ready("read-buffer")?;
        let desc = self.buffer("read-buffer", s.buffer)?;
        s.validate_for(desc).map_err(|_| GlError::Validation {
            operation: "read-buffer",
            message: "invalid buffer range".into(),
        })?;
        let bytes = vec![
            0;
            usize::try_from(s.size).map_err(|_| GlError::OutOfMemory {
                operation: "read-buffer"
            })?
        ];
        self.calls.push(MockCall::ReadBuffer {
            buffer: s.buffer,
            offset: s.offset,
            size: s.size,
        });
        Ok(bytes)
    }
    fn copy_texture_region(
        &mut self,
        s: GlTextureRegion,
        d: GlTextureRegion,
    ) -> Result<(), GlError> {
        self.ready("copy-texture")?;
        let sd = self.texture("copy-texture", s.subresource.texture)?;
        let dd = self.texture("copy-texture", d.subresource.texture)?;
        validate_texture_copy(s, sd, d, dd).map_err(|_| GlError::Validation {
            operation: "copy-texture",
            message: "invalid texture copy".into(),
        })?;
        if s.subresource.texture == d.subresource.texture
            && s.subresource.mip_level == d.subresource.mip_level
        {
            // Same rule as the executable backends: reading and writing one
            // mip is a driver-dependent feedback loop.
            return self.invalid(
                "copy-texture",
                "copy source and destination name the same mip",
            );
        }
        if sd.sample_count != 1 || dd.sample_count != 1 {
            // Multisample transfer belongs to the resolve word.
            return self.invalid("copy-texture", "copy operates on single-sample textures");
        }
        // The executable backends resolve copy facts at sample count one.
        let facts = self.discovery.formats();
        let source_copy = facts
            .get_for(GlFormatResourceKind::Texture, sd.format, 1)
            .map(|fact| fact.copy_source)
            .unwrap_or(false);
        let destination_copy = facts
            .get_for(GlFormatResourceKind::Texture, dd.format, 1)
            .map(|fact| fact.copy_destination)
            .unwrap_or(false);
        if !sd.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !dd.usage.contains(GlTextureUsage::COPY_DESTINATION)
            || !source_copy
            || !destination_copy
        {
            return self.invalid(
                "copy-texture",
                "copy source or destination usage/exact format fact is missing",
            );
        }
        self.calls.push(MockCall::CopyTexture {
            source: s.subresource.texture,
            destination: d.subresource.texture,
        });
        Ok(())
    }
    fn upload_texture(
        &mut self,
        d: GlTextureRegion,
        l: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        self.ready("upload-texture")?;
        let desc = self.texture("upload-texture", d.subresource.texture)?;
        d.validate_for(desc).map_err(|_| GlError::Validation {
            operation: "upload-texture",
            message: "invalid texture upload".into(),
        })?;
        // Same encoding rules as the executable backends: depth storage and
        // unmapped formats accept no CPU pixels, and only the RGBA8 client
        // encoding transfers.
        if desc.format.compressed_info().is_none()
            && !matches!(desc.format, GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb)
        {
            return self.error_result(GlError::Unsupported {
                operation: "upload-texture",
                reason: "format accepts no CPU pixel upload in this shared semantic",
            });
        }
        if !matches!(l.format, GlPixelFormat::Rgba8) {
            return self.error_result(GlError::Unsupported {
                operation: "upload-texture",
                reason: "pixel encoding has no transfer route",
            });
        }
        if d.subresource.base_layer != 0
            || d.subresource.layer_count != 1
            || d.origin[2] != 0
            || d.extent.depth_or_layers != 1
        {
            return self.error_result(GlError::Unsupported {
                operation: "upload-texture",
                reason: "this copy slice transfers one 2D rectangle only",
            });
        }
        let n = l.required_bytes(d).map_err(|_| GlError::Validation {
            operation: "upload-texture",
            message: "invalid texture upload".into(),
        })?;
        if bytes.len() != usize::try_from(n).unwrap_or(usize::MAX) {
            return self.invalid("upload-texture", "byte length mismatch");
        }
        self.calls
            .push(MockCall::UploadTexture(d.subresource.texture));
        Ok(())
    }
    fn read_texture(
        &mut self,
        s: GlTextureRegion,
        l: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        self.ready("read-texture")?;
        let desc = self.texture("read-texture", s.subresource.texture)?;
        s.validate_for(desc).map_err(|_| GlError::Validation {
            operation: "read-texture",
            message: "invalid read region".into(),
        })?;
        if !matches!(desc.format, GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb) {
            return self.error_result(GlError::Unsupported {
                operation: "read-texture",
                reason: "format has no readback encoding",
            });
        }
        if !matches!(l.format, GlPixelFormat::Rgba8) {
            return self.error_result(GlError::Unsupported {
                operation: "read-texture",
                reason: "pixel encoding has no readback route",
            });
        }
        if s.subresource.base_layer != 0
            || s.subresource.layer_count != 1
            || s.origin[2] != 0
            || s.extent.depth_or_layers != 1
        {
            return self.error_result(GlError::Unsupported {
                operation: "read-texture",
                reason: "this copy slice transfers one 2D rectangle only",
            });
        }
        let n = l.required_bytes(s).map_err(|_| GlError::Validation {
            operation: "read-texture",
            message: "invalid layout".into(),
        })?;
        // The layout offset is a client-side placement; it must leave a body.
        if l.offset >= n {
            return self.invalid("read-texture", "layout offset leaves no readback body");
        }
        let bytes = vec![
            0;
            usize::try_from(n).map_err(|_| GlError::OutOfMemory {
                operation: "read-texture"
            })?
        ];
        self.calls
            .push(MockCall::ReadTexture(s.subresource.texture));
        Ok(GlReadback { layout: l, bytes })
    }
    fn pixel_store(&self) -> GlPixelStoreState {
        self.pixel_store
    }
}
