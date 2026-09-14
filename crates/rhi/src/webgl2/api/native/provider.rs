//! Minimal native executable owner for the resource, sampler, and buffer-copy
//! slices. The Host owns the platform context; this type only borrows its
//! already-current `glow` dispatch table.

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
use super::super::GlFamilyApi as _;
use super::super::{ContextStamp, GlDiscoverySnapshot};
use super::discovery::{NativeDiscoveryError, discover_current_glow};
use std::collections::BTreeMap;

/// Minimal native executable owner for the resource, sampler, and buffer-copy
/// slices. The Host owns the platform context; this type only borrows its
/// already-current `glow` dispatch table.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) struct NativeGlProvider<'a> {
    gl: &'a glow::Context,
    discovery: GlDiscoverySnapshot,
    lifecycle: super::super::GlContextLifecycle,
    owner: super::super::OwnerThreadIdentity,
    next_slot: u32,
    buffers: BTreeMap<super::super::BufferId, (glow::NativeBuffer, super::super::GlBufferDesc)>,
    textures: BTreeMap<super::super::TextureId, (glow::NativeTexture, super::super::GlTextureDesc)>,
    samplers: BTreeMap<super::super::SamplerId, glow::NativeSampler>,
    pixel_store: super::super::GlPixelStoreState,
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl<'a> NativeGlProvider<'a> {
    /// # Safety
    ///
    /// Same as [`discover_current_glow`]: the caller keeps `gl` current and
    /// exclusively owned by this thread for the provider's entire lifetime.
    pub(crate) unsafe fn from_current(
        gl: &'a glow::Context,
        stamp: ContextStamp,
    ) -> Result<Self, NativeDiscoveryError> {
        // SAFETY: forwarded from this constructor's current-context contract.
        let discovery = unsafe { discover_current_glow(gl, stamp) }?;
        Ok(Self {
            gl,
            discovery,
            lifecycle: super::super::GlContextLifecycle::Active,
            owner: super::super::OwnerThreadIdentity::current(),
            next_slot: 0,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            pixel_store: super::super::GlPixelStoreState::DEFAULT,
        })
    }

    fn slot(&mut self, operation: &'static str) -> Result<u32, super::super::GlError> {
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .ok_or(super::super::GlError::OutOfMemory { operation })?;
        Ok(slot)
    }

    fn driver_error(&self, operation: &'static str) -> Result<(), super::super::GlError> {
        use glow::HasContext as _;
        // SAFETY: upheld by NativeGlProvider::from_current.
        let error = unsafe { self.gl.get_error() };
        (error == glow::NO_ERROR)
            .then_some(())
            .ok_or_else(|| super::super::GlError::Driver {
                operation,
                message: format!("GL error 0x{error:04x}"),
            })
    }

    fn buffer(
        &self,
        operation: &'static str,
        id: super::super::BufferId,
    ) -> Result<(glow::NativeBuffer, super::super::GlBufferDesc), super::super::GlError> {
        self.validate_object_context(operation, id.context)?;
        self.buffers
            .get(&id)
            .copied()
            .ok_or_else(|| super::super::GlError::Validation {
                operation,
                message: "buffer is not live".into(),
            })
    }

    fn texture(
        &self,
        operation: &'static str,
        id: super::super::TextureId,
    ) -> Result<(glow::NativeTexture, super::super::GlTextureDesc), super::super::GlError> {
        self.validate_object_context(operation, id.context)?;
        self.textures
            .get(&id)
            .copied()
            .ok_or_else(|| super::super::GlError::Validation {
                operation,
                message: "texture is not live".into(),
            })
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::super::GlFamilyApi for NativeGlProvider<'_> {
    fn lifecycle(&self) -> super::super::GlContextLifecycle {
        self.lifecycle
    }
    fn owner_thread(&self) -> super::super::OwnerThreadIdentity {
        self.owner
    }
    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), super::super::GlError> {
        let actual = super::super::OwnerThreadIdentity::current();
        (actual == self.owner)
            .then_some(())
            .ok_or(super::super::GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.discovery
    }
    fn context_lost(&mut self) -> Result<(), super::super::GlError> {
        self.assert_ready("context-lost")?;
        self.lifecycle = super::super::GlContextLifecycle::Lost;
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        Ok(())
    }
    fn context_restored(&mut self) -> Result<ContextStamp, super::super::GlError> {
        Err(super::super::GlError::Unsupported {
            operation: "context-restored",
            reason: "Host must supply a newly current context and rediscover",
        })
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::super::GlResourceApi for NativeGlProvider<'_> {
    fn create_buffer_resource(
        &mut self,
        desc: super::super::GlBufferDesc,
    ) -> Result<super::super::BufferId, super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-buffer")?;
        desc.validate()
            .map_err(|_| super::super::GlError::Validation {
                operation: "create-buffer",
                message: "invalid buffer descriptor".into(),
            })?;
        let size = i32::try_from(desc.size).map_err(|_| super::super::GlError::Validation {
            operation: "create-buffer",
            message: "buffer exceeds GLsizei".into(),
        })?;
        // SAFETY: current-context contract; all validation completed before GL mutation.
        let name = unsafe { self.gl.create_buffer() }.map_err(|message| {
            super::super::GlError::Driver {
                operation: "create-buffer",
                message,
            }
        })?;
        // SAFETY: see above.
        unsafe {
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
            self.gl
                .buffer_data_size(glow::COPY_WRITE_BUFFER, size, glow::STATIC_DRAW);
        }
        if let Err(error) = self.driver_error("create-buffer") {
            unsafe { self.gl.delete_buffer(name) };
            return Err(error);
        }
        let id = super::super::BufferId::new(self.context_stamp(), self.slot("create-buffer")?, 0);
        self.buffers.insert(id, (name, desc));
        Ok(id)
    }
    fn create_texture_resource(
        &mut self,
        desc: super::super::GlTextureDesc,
    ) -> Result<super::super::TextureId, super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-texture")?;
        desc.validate()
            .map_err(|_| super::super::GlError::Validation {
                operation: "create-texture",
                message: "invalid texture descriptor".into(),
            })?;
        let internal =
            native_texture_format(desc.format).ok_or(super::super::GlError::Unsupported {
                operation: "create-texture",
                reason: "format is not in the native texture slice",
            })?;
        if desc.dimension != super::super::GlTextureDimension::D2 || desc.sample_count != 1 {
            return Err(super::super::GlError::Unsupported {
                operation: "create-texture",
                reason: "only single-sample 2D textures are in the native slice",
            });
        }
        if self
            .discovery
            .formats()
            .get_for(super::super::GlFormatResourceKind::Texture, desc.format, 1)
            .is_none()
        {
            return Err(super::super::GlError::Unsupported {
                operation: "create-texture",
                reason: "format lacks discovery evidence",
            });
        }
        let width =
            i32::try_from(desc.extent.width).map_err(|_| super::super::GlError::Validation {
                operation: "create-texture",
                message: "width exceeds GLsizei".into(),
            })?;
        let height =
            i32::try_from(desc.extent.height).map_err(|_| super::super::GlError::Validation {
                operation: "create-texture",
                message: "height exceeds GLsizei".into(),
            })?;
        let levels =
            i32::try_from(desc.mip_level_count).map_err(|_| super::super::GlError::Validation {
                operation: "create-texture",
                message: "mip count exceeds GLsizei".into(),
            })?;
        // SAFETY: current-context contract; all profile/format/size validation preceded mutation.
        let name = unsafe { self.gl.create_texture() }.map_err(|message| {
            super::super::GlError::Driver {
                operation: "create-texture",
                message,
            }
        })?;
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(name));
            if desc.format.compressed_info().is_none() {
                self.gl
                    .tex_storage_2d(glow::TEXTURE_2D, levels, internal, width, height);
            }
        }
        if let Err(error) = self.driver_error("create-texture") {
            unsafe { self.gl.delete_texture(name) };
            return Err(error);
        }
        let id =
            super::super::TextureId::new(self.context_stamp(), self.slot("create-texture")?, 0);
        self.textures.insert(id, (name, desc));
        Ok(id)
    }
    fn destroy_buffer_resource(
        &mut self,
        id: super::super::BufferId,
    ) -> Result<(), super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-buffer")?;
        let (name, _) = self.buffer("destroy-buffer", id)?;
        // SAFETY: current-context contract; liveness was checked before GL mutation.
        unsafe { self.gl.delete_buffer(name) };
        self.driver_error("destroy-buffer")?;
        self.buffers.remove(&id);
        Ok(())
    }
    fn destroy_texture_resource(
        &mut self,
        id: super::super::TextureId,
    ) -> Result<(), super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-texture")?;
        let (name, _) = self.texture("destroy-texture", id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_texture(name) };
        self.driver_error("destroy-texture")?;
        self.textures.remove(&id);
        Ok(())
    }
    fn create_render_buffer(
        &mut self,
        _: super::super::GlRenderBufferDesc,
    ) -> Result<super::super::RenderbufferId, super::super::GlError> {
        Err(super::super::GlError::Unsupported {
            operation: "create-render-buffer",
            reason: "renderbuffer executor not yet profile-lowered",
        })
    }
    fn destroy_render_buffer(
        &mut self,
        _: super::super::RenderbufferId,
    ) -> Result<(), super::super::GlError> {
        Err(super::super::GlError::Unsupported {
            operation: "destroy-render-buffer",
            reason: "renderbuffer executor not yet profile-lowered",
        })
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::super::GlSamplerApi for NativeGlProvider<'_> {
    fn create_sampler(
        &mut self,
        desc: super::super::GlSamplerDesc,
    ) -> Result<super::super::SamplerId, super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-sampler")?;
        desc.validate_for(&self.discovery)
            .map_err(|_| super::super::GlError::Validation {
                operation: "create-sampler",
                message: "invalid sampler descriptor".into(),
            })?;
        // SAFETY: current-context contract; descriptor was fully preflighted.
        let name = unsafe { self.gl.create_sampler() }.map_err(|message| {
            super::super::GlError::Driver {
                operation: "create-sampler",
                message,
            }
        })?;
        // SAFETY: see above. Every parameter comes from a validated closed enum/value.
        unsafe {
            self.gl
                .sampler_parameter_i32(name, 0x2802, native_wrap(desc.address_mode_u));
            self.gl
                .sampler_parameter_i32(name, 0x2803, native_wrap(desc.address_mode_v));
            self.gl
                .sampler_parameter_i32(name, 0x8072, native_wrap(desc.address_mode_w));
            self.gl
                .sampler_parameter_i32(name, 0x2800, native_mag(desc.mag_filter));
            self.gl.sampler_parameter_i32(
                name,
                0x2801,
                native_min(desc.min_filter, desc.mipmap_filter),
            );
            self.gl
                .sampler_parameter_f32(name, 0x813A, f32::from_bits(desc.lod_min_bits));
            self.gl
                .sampler_parameter_f32(name, 0x813B, f32::from_bits(desc.lod_max_bits));
            if let Some(compare) = desc.compare {
                self.gl.sampler_parameter_i32(name, 0x884C, 0x884E);
                self.gl
                    .sampler_parameter_i32(name, 0x884D, native_compare(compare));
            }
            if let Some(anisotropy) = desc.max_anisotropy_bits {
                self.gl
                    .sampler_parameter_f32(name, 0x84FE, f32::from_bits(anisotropy));
            }
        }
        if let Err(error) = self.driver_error("create-sampler") {
            unsafe { self.gl.delete_sampler(name) };
            return Err(error);
        }
        let id =
            super::super::SamplerId::new(self.context_stamp(), self.slot("create-sampler")?, 0);
        self.samplers.insert(id, name);
        Ok(id)
    }
    fn destroy_sampler(
        &mut self,
        id: super::super::SamplerId,
    ) -> Result<(), super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-sampler")?;
        self.validate_object_context("destroy-sampler", id.context)?;
        let name =
            self.samplers
                .get(&id)
                .copied()
                .ok_or_else(|| super::super::GlError::Validation {
                    operation: "destroy-sampler",
                    message: "sampler is not live".into(),
                })?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_sampler(name) };
        self.driver_error("destroy-sampler")?;
        self.samplers.remove(&id);
        Ok(())
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::super::GlCopyDomainApi for NativeGlProvider<'_> {
    fn copy_buffer_range(
        &mut self,
        source: super::super::GlBufferRange,
        destination: super::super::GlBufferRange,
    ) -> Result<(), super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("copy-buffer")?;
        let (source_name, source_desc) = self.buffer("copy-buffer", source.buffer)?;
        let (destination_name, destination_desc) =
            self.buffer("copy-buffer", destination.buffer)?;
        source
            .validate_for(source_desc)
            .and_then(|_| destination.validate_for(destination_desc))
            .map_err(|_| super::super::GlError::Validation {
                operation: "copy-buffer",
                message: "invalid buffer range".into(),
            })?;
        if source.size != destination.size {
            return Err(super::super::GlError::Validation {
                operation: "copy-buffer",
                message: "copy sizes differ".into(),
            });
        }
        let read_offset =
            i32::try_from(source.offset).map_err(|_| super::super::GlError::Validation {
                operation: "copy-buffer",
                message: "source offset exceeds GLintptr".into(),
            })?;
        let write_offset =
            i32::try_from(destination.offset).map_err(|_| super::super::GlError::Validation {
                operation: "copy-buffer",
                message: "destination offset exceeds GLintptr".into(),
            })?;
        let size = i32::try_from(source.size).map_err(|_| super::super::GlError::Validation {
            operation: "copy-buffer",
            message: "copy size exceeds GLsizeiptr".into(),
        })?;
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
        self.driver_error("copy-buffer")
    }
    fn copy_texture_region(
        &mut self,
        _: super::super::GlTextureRegion,
        _: super::super::GlTextureRegion,
    ) -> Result<(), super::super::GlError> {
        Err(super::super::GlError::Unsupported {
            operation: "copy-texture",
            reason: "texture executor not yet profile-lowered",
        })
    }
    fn upload_buffer(
        &mut self,
        _: super::super::GlBufferRange,
        _: &[u8],
    ) -> Result<(), super::super::GlError> {
        Err(super::super::GlError::Unsupported {
            operation: "upload-buffer",
            reason: "buffer upload executor not yet profile-lowered",
        })
    }
    fn read_buffer(
        &mut self,
        _: super::super::GlBufferRange,
    ) -> Result<Vec<u8>, super::super::GlError> {
        Err(super::super::GlError::Unsupported {
            operation: "read-buffer",
            reason: "buffer readback executor not yet profile-lowered",
        })
    }
    fn upload_texture(
        &mut self,
        destination: super::super::GlTextureRegion,
        layout: super::super::GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), super::super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("upload-texture")?;
        let (name, desc) = self.texture("upload-texture", destination.subresource.texture)?;
        destination
            .validate_for(desc)
            .map_err(|_| super::super::GlError::Validation {
                operation: "upload-texture",
                message: "invalid texture region".into(),
            })?;
        if let Some(info) = desc.format.compressed_info() {
            if desc.dimension != super::super::GlTextureDimension::D2
                || destination.subresource.base_layer != 0
                || destination.subresource.layer_count != 1
                || destination.extent.depth_or_layers != 1
                || destination.origin != [0; 3]
                || destination.extent
                    != desc.mip_extent(destination.subresource.mip_level).ok_or(
                        super::super::GlError::Validation {
                            operation: "upload-texture",
                            message: "invalid compressed mip".into(),
                        },
                    )?
            {
                return Err(super::super::GlError::Unsupported {
                    operation: "upload-texture",
                    reason: "compressed upload must define one complete 2D mip",
                });
            }
            let exact = info
                .checked_encoded_size(destination.extent.width, destination.extent.height)
                .map_err(|_| super::super::GlError::Validation {
                    operation: "upload-texture",
                    message: "compressed encoded size overflow".into(),
                })?;
            if u64::try_from(bytes.len()).ok() != Some(exact) {
                return Err(super::super::GlError::Validation {
                    operation: "upload-texture",
                    message: "compressed bytes do not match exact block layout".into(),
                });
            }
            let level = i32::try_from(destination.subresource.mip_level).map_err(|_| {
                super::super::GlError::Validation {
                    operation: "upload-texture",
                    message: "mip level exceeds GLint".into(),
                }
            })?;
            let width = i32::try_from(destination.extent.width).map_err(|_| {
                super::super::GlError::Validation {
                    operation: "upload-texture",
                    message: "width exceeds GLsizei".into(),
                }
            })?;
            let height = i32::try_from(destination.extent.height).map_err(|_| {
                super::super::GlError::Validation {
                    operation: "upload-texture",
                    message: "height exceeds GLsizei".into(),
                }
            })?;
            let size = i32::try_from(exact).map_err(|_| super::super::GlError::Validation {
                operation: "upload-texture",
                message: "compressed upload exceeds GLsizei".into(),
            })?;
            let internal =
                native_texture_format(desc.format).expect("proven compressed format is mapped");
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
            return self.driver_error("upload-texture");
        }
        let needed =
            layout
                .required_bytes(destination)
                .map_err(|_| super::super::GlError::Validation {
                    operation: "upload-texture",
                    message: "invalid pixel layout".into(),
                })?;
        if u64::try_from(bytes.len()).ok() != Some(needed) {
            return Err(super::super::GlError::Validation {
                operation: "upload-texture",
                message: "upload source length differs from layout".into(),
            });
        }
        if desc.dimension != super::super::GlTextureDimension::D2
            || destination.subresource.base_layer != 0
            || destination.subresource.layer_count != 1
            || destination.extent.depth_or_layers != 1
            || layout.format != super::super::GlPixelFormat::Rgba8
            || layout.offset != 0
            || layout.bytes_per_row != destination.extent.width.saturating_mul(4)
            || layout.rows_per_image != destination.extent.height
        {
            return Err(super::super::GlError::Unsupported {
                operation: "upload-texture",
                reason: "only tightly packed RGBA8 2D upload is in the native slice",
            });
        }
        let level = i32::try_from(destination.subresource.mip_level).map_err(|_| {
            super::super::GlError::Validation {
                operation: "upload-texture",
                message: "mip level exceeds GLint".into(),
            }
        })?;
        let x = i32::try_from(destination.origin[0]).map_err(|_| {
            super::super::GlError::Validation {
                operation: "upload-texture",
                message: "x exceeds GLint".into(),
            }
        })?;
        let y = i32::try_from(destination.origin[1]).map_err(|_| {
            super::super::GlError::Validation {
                operation: "upload-texture",
                message: "y exceeds GLint".into(),
            }
        })?;
        let width = i32::try_from(destination.extent.width).map_err(|_| {
            super::super::GlError::Validation {
                operation: "upload-texture",
                message: "width exceeds GLsizei".into(),
            }
        })?;
        let height = i32::try_from(destination.extent.height).map_err(|_| {
            super::super::GlError::Validation {
                operation: "upload-texture",
                message: "height exceeds GLsizei".into(),
            }
        })?;
        let saved = self.pixel_store;
        // SAFETY: current-context contract; every checked input is now representable.
        unsafe {
            self.gl
                .pixel_store_i32(glow::UNPACK_ALIGNMENT, i32::from(layout.alignment));
            self.gl.bind_texture(glow::TEXTURE_2D, Some(name));
            self.gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                level,
                x,
                y,
                width,
                height,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );
        }
        let result = self.driver_error("upload-texture");
        // SAFETY: exact tracked state restoration happens on both success and error paths.
        unsafe {
            self.gl
                .pixel_store_i32(glow::UNPACK_ALIGNMENT, i32::from(saved.unpack_alignment));
        }
        self.pixel_store = saved;
        result
    }
    fn read_texture(
        &mut self,
        _: super::super::GlTextureRegion,
        _: super::super::GlPixelLayout,
    ) -> Result<super::super::GlReadback, super::super::GlError> {
        Err(super::super::GlError::Unsupported {
            operation: "read-texture",
            reason: "texture executor not yet profile-lowered",
        })
    }
    fn pixel_store(&self) -> super::super::GlPixelStoreState {
        self.pixel_store
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_wrap(mode: super::super::GlAddressMode) -> i32 {
    match mode {
        super::super::GlAddressMode::ClampToEdge => 0x812F,
        super::super::GlAddressMode::Repeat => 0x2901,
        super::super::GlAddressMode::MirroredRepeat => 0x8370,
    }
}
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_mag(mode: super::super::GlFilterMode) -> i32 {
    match mode {
        super::super::GlFilterMode::Nearest => 0x2600,
        super::super::GlFilterMode::Linear => 0x2601,
    }
}
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_min(min: super::super::GlFilterMode, mip: super::super::GlMipmapFilterMode) -> i32 {
    match (min, mip) {
        (super::super::GlFilterMode::Nearest, super::super::GlMipmapFilterMode::Nearest) => 0x2700,
        (super::super::GlFilterMode::Linear, super::super::GlMipmapFilterMode::Nearest) => 0x2701,
        (super::super::GlFilterMode::Nearest, super::super::GlMipmapFilterMode::Linear) => 0x2702,
        (super::super::GlFilterMode::Linear, super::super::GlMipmapFilterMode::Linear) => 0x2703,
    }
}
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_compare(compare: super::super::GlCompareFunction) -> i32 {
    match compare {
        super::super::GlCompareFunction::Never => 0x0200,
        super::super::GlCompareFunction::Less => 0x0201,
        super::super::GlCompareFunction::Equal => 0x0202,
        super::super::GlCompareFunction::LessEqual => 0x0203,
        super::super::GlCompareFunction::Greater => 0x0204,
        super::super::GlCompareFunction::NotEqual => 0x0205,
        super::super::GlCompareFunction::GreaterEqual => 0x0206,
        super::super::GlCompareFunction::Always => 0x0207,
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_texture_format(format: super::super::GlFormat) -> Option<u32> {
    match format {
        super::super::GlFormat::Rgba8Unorm => Some(0x8058),
        super::super::GlFormat::Rgba8Srgb => Some(0x8C43),
        super::super::GlFormat::Etc2Rgb8Unorm => Some(0x9274),
        super::super::GlFormat::Etc2Rgb8Srgb => Some(0x9275),
        super::super::GlFormat::Etc2Rgb8A1Unorm => Some(0x9276),
        super::super::GlFormat::Etc2Rgb8A1Srgb => Some(0x9277),
        super::super::GlFormat::Etc2Rgba8Unorm => Some(0x9278),
        super::super::GlFormat::Etc2Rgba8Srgb => Some(0x9279),
        super::super::GlFormat::EacR11Unorm => Some(0x9270),
        super::super::GlFormat::EacR11Snorm => Some(0x9271),
        super::super::GlFormat::EacRg11Unorm => Some(0x9272),
        super::super::GlFormat::EacRg11Snorm => Some(0x9273),
        _ => None,
    }
}
