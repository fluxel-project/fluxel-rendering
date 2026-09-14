//! Native GL/GLES discovery over a context made current by the Host.
//!
//! This module never creates a window, display, surface, or context.  Its one
//! unsafe boundary is the `glow` adapter: callers must keep the supplied
//! context current on its owning thread for the entire call.  The resulting
//! snapshot is data only and remains bound to the caller supplied stamp.

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
use super::GlFamilyApi as _;
use super::{
    ContextStamp, CoreOrExtension, GlCapability, GlContextFlags, GlContextInfo, GlDiscoveryBuilder,
    GlDiscoveryError, GlDiscoverySnapshot, GlExtensionSet, GlFamilyProfile, GlFiniteF32, GlFormat,
    GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable, GlKnownExtension,
    GlLimits, GlOperationProbe, GlVersion,
};
use std::collections::BTreeMap;

/// Failure to obtain a complete native discovery record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeDiscoveryError {
    /// The current context already carried an error before discovery began.
    ///
    /// Discovery never clears an unknown driver error and continues, because
    /// that would make later observations impossible to attribute.
    PreExistingGlError,
    /// A required driver string was absent or malformed.
    InvalidContextString(&'static str),
    /// The driver reported a profile outside Fluxel's native GL-family scope.
    UnsupportedProfile(String),
    /// A required numeric observation failed.
    QueryFailed(&'static str),
    /// The common snapshot contract rejected otherwise collected facts.
    Snapshot(GlDiscoveryError),
}

/// Small mockable subset of native GL used by discovery.
///
/// Implementations must return `None` for a failed query, including a GL
/// error.  This is deliberately not a general command interface.
trait NativeGlQuery {
    /// Consumes exactly one pending driver error, returning true when it was
    /// not `GL_NO_ERROR`.
    fn take_error(&self) -> bool;
    fn string(&self, name: u32) -> Option<String>;
    fn integer(&self, name: u32) -> Option<i64>;
    fn integer_pair(&self, name: u32) -> Option<[i64; 2]>;
    fn indexed_integer(&self, name: u32, index: u32) -> Option<i64>;
    fn float(&self, name: u32) -> Option<f32>;
    fn indexed_string(&self, name: u32, index: u32) -> Option<String>;
}

/// Discovers native facts using an already-current `glow` context.
///
/// # Safety contract
///
/// The Host/RHI provider must have made `context` current on its owner thread,
/// must keep it current throughout this call, and must serialize access to the
/// context.  `glow` forwards to the current native context; violating that
/// contract is outside Rust's type system and can call an unrelated driver.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) unsafe fn discover_current_glow(
    context: &glow::Context,
    stamp: ContextStamp,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    use glow::HasContext as _;

    struct GlowQuery<'a>(&'a glow::Context);
    impl NativeGlQuery for GlowQuery<'_> {
        fn take_error(&self) -> bool {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            unsafe { self.0.get_error() != glow::NO_ERROR }
        }
        fn string(&self, name: u32) -> Option<String> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_string(name) };
            (!self.take_error() && !value.is_empty()).then_some(value)
        }
        fn integer(&self, name: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_i32(name) };
            // SAFETY: get_error reads the same current context and makes failed
            // optional queries fail closed instead of turning them into support.
            (!self.take_error()).then_some(i64::from(value))
        }
        fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
            let mut values = [0_i32; 2];
            // SAFETY: upheld by discover_current_glow's current-context contract.
            unsafe { self.0.get_parameter_i32_slice(name, &mut values) };
            // SAFETY: see integer.
            (!self.take_error()).then_some([i64::from(values[0]), i64::from(values[1])])
        }
        fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_indexed_i32(name, index) };
            // SAFETY: see integer.
            (!self.take_error()).then_some(i64::from(value))
        }
        fn float(&self, name: u32) -> Option<f32> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_f32(name) };
            // SAFETY: see integer.
            (!self.take_error()).then_some(value)
        }
        fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
            // SAFETY: upheld by discover_current_glow's current-context contract.
            let value = unsafe { self.0.get_parameter_indexed_string(name, index) };
            (!self.take_error() && !value.is_empty()).then_some(value)
        }
    }

    // SAFETY: forwarded from this function's documented caller contract.
    discover_with_query(&GlowQuery(context), stamp)
}

/// Minimal native executable owner for the resource, sampler, and buffer-copy
/// slices. The Host owns the platform context; this type only borrows its
/// already-current `glow` dispatch table.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) struct NativeGlProvider<'a> {
    gl: &'a glow::Context,
    discovery: GlDiscoverySnapshot,
    lifecycle: super::GlContextLifecycle,
    owner: super::OwnerThreadIdentity,
    next_slot: u32,
    buffers: BTreeMap<super::BufferId, (glow::NativeBuffer, super::GlBufferDesc)>,
    textures: BTreeMap<super::TextureId, (glow::NativeTexture, super::GlTextureDesc)>,
    samplers: BTreeMap<super::SamplerId, glow::NativeSampler>,
    pixel_store: super::GlPixelStoreState,
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
            lifecycle: super::GlContextLifecycle::Active,
            owner: super::OwnerThreadIdentity::current(),
            next_slot: 0,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            pixel_store: super::GlPixelStoreState::DEFAULT,
        })
    }

    fn slot(&mut self, operation: &'static str) -> Result<u32, super::GlError> {
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .ok_or(super::GlError::OutOfMemory { operation })?;
        Ok(slot)
    }

    fn driver_error(&self, operation: &'static str) -> Result<(), super::GlError> {
        use glow::HasContext as _;
        // SAFETY: upheld by NativeGlProvider::from_current.
        let error = unsafe { self.gl.get_error() };
        (error == glow::NO_ERROR)
            .then_some(())
            .ok_or_else(|| super::GlError::Driver {
                operation,
                message: format!("GL error 0x{error:04x}"),
            })
    }

    fn buffer(
        &self,
        operation: &'static str,
        id: super::BufferId,
    ) -> Result<(glow::NativeBuffer, super::GlBufferDesc), super::GlError> {
        self.validate_object_context(operation, id.context)?;
        self.buffers
            .get(&id)
            .copied()
            .ok_or_else(|| super::GlError::Validation {
                operation,
                message: "buffer is not live".into(),
            })
    }

    fn texture(
        &self,
        operation: &'static str,
        id: super::TextureId,
    ) -> Result<(glow::NativeTexture, super::GlTextureDesc), super::GlError> {
        self.validate_object_context(operation, id.context)?;
        self.textures
            .get(&id)
            .copied()
            .ok_or_else(|| super::GlError::Validation {
                operation,
                message: "texture is not live".into(),
            })
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::GlFamilyApi for NativeGlProvider<'_> {
    fn lifecycle(&self) -> super::GlContextLifecycle {
        self.lifecycle
    }
    fn owner_thread(&self) -> super::OwnerThreadIdentity {
        self.owner
    }
    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), super::GlError> {
        let actual = super::OwnerThreadIdentity::current();
        (actual == self.owner)
            .then_some(())
            .ok_or(super::GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.discovery
    }
    fn context_lost(&mut self) -> Result<(), super::GlError> {
        self.assert_ready("context-lost")?;
        self.lifecycle = super::GlContextLifecycle::Lost;
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        Ok(())
    }
    fn context_restored(&mut self) -> Result<ContextStamp, super::GlError> {
        Err(super::GlError::Unsupported {
            operation: "context-restored",
            reason: "Host must supply a newly current context and rediscover",
        })
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::GlResourceApi for NativeGlProvider<'_> {
    fn create_buffer_resource(
        &mut self,
        desc: super::GlBufferDesc,
    ) -> Result<super::BufferId, super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-buffer")?;
        desc.validate().map_err(|_| super::GlError::Validation {
            operation: "create-buffer",
            message: "invalid buffer descriptor".into(),
        })?;
        let size = i32::try_from(desc.size).map_err(|_| super::GlError::Validation {
            operation: "create-buffer",
            message: "buffer exceeds GLsizei".into(),
        })?;
        // SAFETY: current-context contract; all validation completed before GL mutation.
        let name =
            unsafe { self.gl.create_buffer() }.map_err(|message| super::GlError::Driver {
                operation: "create-buffer",
                message,
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
        let id = super::BufferId::new(self.context_stamp(), self.slot("create-buffer")?, 0);
        self.buffers.insert(id, (name, desc));
        Ok(id)
    }
    fn create_texture_resource(
        &mut self,
        desc: super::GlTextureDesc,
    ) -> Result<super::TextureId, super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-texture")?;
        desc.validate().map_err(|_| super::GlError::Validation {
            operation: "create-texture",
            message: "invalid texture descriptor".into(),
        })?;
        let internal = native_texture_format(desc.format).ok_or(super::GlError::Unsupported {
            operation: "create-texture",
            reason: "format is not in the native texture slice",
        })?;
        if desc.dimension != super::GlTextureDimension::D2 || desc.sample_count != 1 {
            return Err(super::GlError::Unsupported {
                operation: "create-texture",
                reason: "only single-sample 2D textures are in the native slice",
            });
        }
        if self
            .discovery
            .formats()
            .get_for(super::GlFormatResourceKind::Texture, desc.format, 1)
            .is_none()
        {
            return Err(super::GlError::Unsupported {
                operation: "create-texture",
                reason: "format lacks discovery evidence",
            });
        }
        let width = i32::try_from(desc.extent.width).map_err(|_| super::GlError::Validation {
            operation: "create-texture",
            message: "width exceeds GLsizei".into(),
        })?;
        let height = i32::try_from(desc.extent.height).map_err(|_| super::GlError::Validation {
            operation: "create-texture",
            message: "height exceeds GLsizei".into(),
        })?;
        let levels =
            i32::try_from(desc.mip_level_count).map_err(|_| super::GlError::Validation {
                operation: "create-texture",
                message: "mip count exceeds GLsizei".into(),
            })?;
        // SAFETY: current-context contract; all profile/format/size validation preceded mutation.
        let name =
            unsafe { self.gl.create_texture() }.map_err(|message| super::GlError::Driver {
                operation: "create-texture",
                message,
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
        let id = super::TextureId::new(self.context_stamp(), self.slot("create-texture")?, 0);
        self.textures.insert(id, (name, desc));
        Ok(id)
    }
    fn destroy_buffer_resource(&mut self, id: super::BufferId) -> Result<(), super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-buffer")?;
        let (name, _) = self.buffer("destroy-buffer", id)?;
        // SAFETY: current-context contract; liveness was checked before GL mutation.
        unsafe { self.gl.delete_buffer(name) };
        self.driver_error("destroy-buffer")?;
        self.buffers.remove(&id);
        Ok(())
    }
    fn destroy_texture_resource(&mut self, id: super::TextureId) -> Result<(), super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-texture")?;
        let (name, _) = self.texture("destroy-texture", id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_texture(name) };
        self.driver_error("destroy-texture")?;
        self.textures.remove(&id);
        Ok(())
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::GlSamplerApi for NativeGlProvider<'_> {
    fn create_sampler(
        &mut self,
        desc: super::GlSamplerDesc,
    ) -> Result<super::SamplerId, super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-sampler")?;
        desc.validate_for(&self.discovery)
            .map_err(|_| super::GlError::Validation {
                operation: "create-sampler",
                message: "invalid sampler descriptor".into(),
            })?;
        // SAFETY: current-context contract; descriptor was fully preflighted.
        let name =
            unsafe { self.gl.create_sampler() }.map_err(|message| super::GlError::Driver {
                operation: "create-sampler",
                message,
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
        let id = super::SamplerId::new(self.context_stamp(), self.slot("create-sampler")?, 0);
        self.samplers.insert(id, name);
        Ok(id)
    }
    fn destroy_sampler(&mut self, id: super::SamplerId) -> Result<(), super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-sampler")?;
        self.validate_object_context("destroy-sampler", id.context)?;
        let name = self
            .samplers
            .get(&id)
            .copied()
            .ok_or_else(|| super::GlError::Validation {
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
impl super::GlCopyDomainApi for NativeGlProvider<'_> {
    fn copy_buffer_range(
        &mut self,
        source: super::GlBufferRange,
        destination: super::GlBufferRange,
    ) -> Result<(), super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("copy-buffer")?;
        let (source_name, source_desc) = self.buffer("copy-buffer", source.buffer)?;
        let (destination_name, destination_desc) =
            self.buffer("copy-buffer", destination.buffer)?;
        source
            .validate_for(source_desc)
            .and_then(|_| destination.validate_for(destination_desc))
            .map_err(|_| super::GlError::Validation {
                operation: "copy-buffer",
                message: "invalid buffer range".into(),
            })?;
        if source.size != destination.size {
            return Err(super::GlError::Validation {
                operation: "copy-buffer",
                message: "copy sizes differ".into(),
            });
        }
        let read_offset = i32::try_from(source.offset).map_err(|_| super::GlError::Validation {
            operation: "copy-buffer",
            message: "source offset exceeds GLintptr".into(),
        })?;
        let write_offset =
            i32::try_from(destination.offset).map_err(|_| super::GlError::Validation {
                operation: "copy-buffer",
                message: "destination offset exceeds GLintptr".into(),
            })?;
        let size = i32::try_from(source.size).map_err(|_| super::GlError::Validation {
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
        _: super::GlTextureRegion,
        _: super::GlTextureRegion,
    ) -> Result<(), super::GlError> {
        Err(super::GlError::Unsupported {
            operation: "copy-texture",
            reason: "texture executor not yet profile-lowered",
        })
    }
    fn upload_texture(
        &mut self,
        destination: super::GlTextureRegion,
        layout: super::GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), super::GlError> {
        use glow::HasContext as _;
        self.assert_ready("upload-texture")?;
        let (name, desc) = self.texture("upload-texture", destination.subresource.texture)?;
        destination
            .validate_for(desc)
            .map_err(|_| super::GlError::Validation {
                operation: "upload-texture",
                message: "invalid texture region".into(),
            })?;
        if let Some(info) = desc.format.compressed_info() {
            if desc.dimension != super::GlTextureDimension::D2
                || destination.subresource.base_layer != 0
                || destination.subresource.layer_count != 1
                || destination.extent.depth_or_layers != 1
                || destination.origin != [0; 3]
                || destination.extent
                    != desc.mip_extent(destination.subresource.mip_level).ok_or(
                        super::GlError::Validation {
                            operation: "upload-texture",
                            message: "invalid compressed mip".into(),
                        },
                    )?
            {
                return Err(super::GlError::Unsupported {
                    operation: "upload-texture",
                    reason: "compressed upload must define one complete 2D mip",
                });
            }
            let exact = info
                .checked_encoded_size(destination.extent.width, destination.extent.height)
                .map_err(|_| super::GlError::Validation {
                    operation: "upload-texture",
                    message: "compressed encoded size overflow".into(),
                })?;
            if u64::try_from(bytes.len()).ok() != Some(exact) {
                return Err(super::GlError::Validation {
                    operation: "upload-texture",
                    message: "compressed bytes do not match exact block layout".into(),
                });
            }
            let level = i32::try_from(destination.subresource.mip_level).map_err(|_| {
                super::GlError::Validation {
                    operation: "upload-texture",
                    message: "mip level exceeds GLint".into(),
                }
            })?;
            let width = i32::try_from(destination.extent.width).map_err(|_| {
                super::GlError::Validation {
                    operation: "upload-texture",
                    message: "width exceeds GLsizei".into(),
                }
            })?;
            let height = i32::try_from(destination.extent.height).map_err(|_| {
                super::GlError::Validation {
                    operation: "upload-texture",
                    message: "height exceeds GLsizei".into(),
                }
            })?;
            let size = i32::try_from(exact).map_err(|_| super::GlError::Validation {
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
                .map_err(|_| super::GlError::Validation {
                    operation: "upload-texture",
                    message: "invalid pixel layout".into(),
                })?;
        if u64::try_from(bytes.len()).ok() != Some(needed) {
            return Err(super::GlError::Validation {
                operation: "upload-texture",
                message: "upload source length differs from layout".into(),
            });
        }
        if desc.dimension != super::GlTextureDimension::D2
            || destination.subresource.base_layer != 0
            || destination.subresource.layer_count != 1
            || destination.extent.depth_or_layers != 1
            || layout.format != super::GlPixelFormat::Rgba8
            || layout.offset != 0
            || layout.bytes_per_row != destination.extent.width.saturating_mul(4)
            || layout.rows_per_image != destination.extent.height
        {
            return Err(super::GlError::Unsupported {
                operation: "upload-texture",
                reason: "only tightly packed RGBA8 2D upload is in the native slice",
            });
        }
        let level = i32::try_from(destination.subresource.mip_level).map_err(|_| {
            super::GlError::Validation {
                operation: "upload-texture",
                message: "mip level exceeds GLint".into(),
            }
        })?;
        let x = i32::try_from(destination.origin[0]).map_err(|_| super::GlError::Validation {
            operation: "upload-texture",
            message: "x exceeds GLint".into(),
        })?;
        let y = i32::try_from(destination.origin[1]).map_err(|_| super::GlError::Validation {
            operation: "upload-texture",
            message: "y exceeds GLint".into(),
        })?;
        let width =
            i32::try_from(destination.extent.width).map_err(|_| super::GlError::Validation {
                operation: "upload-texture",
                message: "width exceeds GLsizei".into(),
            })?;
        let height =
            i32::try_from(destination.extent.height).map_err(|_| super::GlError::Validation {
                operation: "upload-texture",
                message: "height exceeds GLsizei".into(),
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
        _: super::GlTextureRegion,
        _: super::GlPixelLayout,
    ) -> Result<super::GlReadback, super::GlError> {
        Err(super::GlError::Unsupported {
            operation: "read-texture",
            reason: "texture executor not yet profile-lowered",
        })
    }
    fn pixel_store(&self) -> super::GlPixelStoreState {
        self.pixel_store
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_wrap(mode: super::GlAddressMode) -> i32 {
    match mode {
        super::GlAddressMode::ClampToEdge => 0x812F,
        super::GlAddressMode::Repeat => 0x2901,
        super::GlAddressMode::MirroredRepeat => 0x8370,
    }
}
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_mag(mode: super::GlFilterMode) -> i32 {
    match mode {
        super::GlFilterMode::Nearest => 0x2600,
        super::GlFilterMode::Linear => 0x2601,
    }
}
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_min(min: super::GlFilterMode, mip: super::GlMipmapFilterMode) -> i32 {
    match (min, mip) {
        (super::GlFilterMode::Nearest, super::GlMipmapFilterMode::Nearest) => 0x2700,
        (super::GlFilterMode::Linear, super::GlMipmapFilterMode::Nearest) => 0x2701,
        (super::GlFilterMode::Nearest, super::GlMipmapFilterMode::Linear) => 0x2702,
        (super::GlFilterMode::Linear, super::GlMipmapFilterMode::Linear) => 0x2703,
    }
}
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_compare(compare: super::GlCompareFunction) -> i32 {
    match compare {
        super::GlCompareFunction::Never => 0x0200,
        super::GlCompareFunction::Less => 0x0201,
        super::GlCompareFunction::Equal => 0x0202,
        super::GlCompareFunction::LessEqual => 0x0203,
        super::GlCompareFunction::Greater => 0x0204,
        super::GlCompareFunction::NotEqual => 0x0205,
        super::GlCompareFunction::GreaterEqual => 0x0206,
        super::GlCompareFunction::Always => 0x0207,
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
const fn native_texture_format(format: super::GlFormat) -> Option<u32> {
    match format {
        super::GlFormat::Rgba8Unorm => Some(0x8058),
        super::GlFormat::Rgba8Srgb => Some(0x8C43),
        super::GlFormat::Etc2Rgb8Unorm => Some(0x9274),
        super::GlFormat::Etc2Rgb8Srgb => Some(0x9275),
        super::GlFormat::Etc2Rgb8A1Unorm => Some(0x9276),
        super::GlFormat::Etc2Rgb8A1Srgb => Some(0x9277),
        super::GlFormat::Etc2Rgba8Unorm => Some(0x9278),
        super::GlFormat::Etc2Rgba8Srgb => Some(0x9279),
        super::GlFormat::EacR11Unorm => Some(0x9270),
        super::GlFormat::EacR11Snorm => Some(0x9271),
        super::GlFormat::EacRg11Unorm => Some(0x9272),
        super::GlFormat::EacRg11Snorm => Some(0x9273),
        _ => None,
    }
}

fn discover_with_query(
    query: &impl NativeGlQuery,
    stamp: ContextStamp,
) -> Result<GlDiscoverySnapshot, NativeDiscoveryError> {
    if query.take_error() {
        return Err(NativeDiscoveryError::PreExistingGlError);
    }
    let version = required_string(query, glow_const::VERSION, "GL_VERSION")?;
    let profile = parse_native_profile(&version)
        .ok_or_else(|| NativeDiscoveryError::UnsupportedProfile(version.clone()))?;
    let glsl = required_string(
        query,
        glow_const::SHADING_LANGUAGE_VERSION,
        "GL_SHADING_LANGUAGE_VERSION",
    )?;
    let vendor = required_string(query, glow_const::VENDOR, "GL_VENDOR")?;
    let renderer = required_string(query, glow_const::RENDERER, "GL_RENDERER")?;
    let extensions = extensions(query, profile)?;
    let limits = limits(query, profile, &extensions)?;
    let formats = baseline_formats(profile)?;
    let mut builder = GlDiscoveryBuilder::new(
        stamp,
        GlContextInfo::new(
            profile,
            &version,
            glsl,
            vendor,
            renderer,
            version.clone(),
            GlContextFlags::default(),
        ),
        extensions,
        limits,
        formats,
    )
    .map_err(NativeDiscoveryError::Snapshot)?;

    // Version and limit reads are evidence only, not executable operation
    // probes.  Keep every optional command domain disabled until the provider
    // performs a separately recorded compile/link/bind/dispatch probe.
    builder.resolve(
        GlCapability::Compute,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbComputeShader),
            extension_requires_probe: true,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::StorageBuffer,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderStorageBufferObject),
            extension_requires_probe: true,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::StorageImage,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 2)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderImageLoadStore),
            extension_requires_probe: true,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::IndirectDraw,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 0)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::IndirectDispatch,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: None,
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::MultiDrawIndirect,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: None,
            extension: None,
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRun,
    );
    builder.resolve(
        GlCapability::TimerQuery,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(3, 3)),
            embedded_core: Some(GlVersion::new(3, 0)),
            extension: None,
            extension_requires_probe: false,
        },
        GlOperationProbe::NotRequired,
    );
    Ok(builder.build())
}

fn required_string(
    query: &impl NativeGlQuery,
    token: u32,
    name: &'static str,
) -> Result<String, NativeDiscoveryError> {
    query
        .string(token)
        .filter(|value| !value.trim().is_empty())
        .ok_or(NativeDiscoveryError::InvalidContextString(name))
}

fn extensions(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
) -> Result<GlExtensionSet, NativeDiscoveryError> {
    let count = nonnegative(
        query.integer(glow_const::NUM_EXTENSIONS),
        "GL_NUM_EXTENSIONS",
    )?;
    let mut result = GlExtensionSet::default();
    for index in 0..count {
        let name = query.indexed_string(glow_const::EXTENSIONS, index).ok_or(
            NativeDiscoveryError::QueryFailed("glGetStringi(GL_EXTENSIONS)"),
        )?;
        result.report_raw(name);
    }
    // Native GL entry points are loaded by the provider before it constructs
    // glow.  Record that acquisition only for typed, legal names; command
    // probes are intentionally left absent and therefore cannot enable an
    // extension-only capability.
    for known in [
        GlKnownExtension::ArbComputeShader,
        GlKnownExtension::ArbShaderStorageBufferObject,
        GlKnownExtension::ArbShaderImageLoadStore,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::KhrRobustness,
        GlKnownExtension::KhrDebug,
    ] {
        if known.is_legal_for(profile) && result.provenance(known).is_some() {
            let _ = result.acquire(known);
        }
    }
    Ok(result)
}

fn limits(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
) -> Result<GlLimits, NativeDiscoveryError> {
    let u = |token, name| nonnegative(query.integer(token), name);
    let pair = query
        .integer_pair(glow_const::MAX_VIEWPORT_DIMS)
        .ok_or(NativeDiscoveryError::QueryFailed("GL_MAX_VIEWPORT_DIMS"))?;
    let compute = profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
        || extensions.is_acquired(GlKnownExtension::ArbComputeShader);
    let storage = profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
        || extensions.is_acquired(GlKnownExtension::ArbShaderStorageBufferObject);
    let image = profile.meets(Some(GlVersion::new(4, 2)), Some(GlVersion::new(3, 1)))
        || extensions.is_acquired(GlKnownExtension::ArbShaderImageLoadStore);
    let texture_multisample = matches!(profile, GlFamilyProfile::Desktop { .. })
        || profile.meets(None, Some(GlVersion::new(3, 1)));
    let optional = |enabled, token, name| if enabled { u(token, name) } else { Ok(0) };
    let indexed = |enabled, token, index, name| {
        if enabled {
            nonnegative(query.indexed_integer(token, index), name)
        } else {
            Ok(0)
        }
    };
    let anisotropy = extensions
        .is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic)
        .then(|| query.float(glow_const::MAX_TEXTURE_MAX_ANISOTROPY_EXT))
        .flatten()
        .and_then(GlFiniteF32::new);
    Ok(GlLimits {
        max_texture_size: u(glow_const::MAX_TEXTURE_SIZE, "GL_MAX_TEXTURE_SIZE")?,
        max_3d_texture_size: u(glow_const::MAX_3D_TEXTURE_SIZE, "GL_MAX_3D_TEXTURE_SIZE")?,
        max_array_texture_layers: u(
            glow_const::MAX_ARRAY_TEXTURE_LAYERS,
            "GL_MAX_ARRAY_TEXTURE_LAYERS",
        )?,
        max_cube_map_texture_size: u(
            glow_const::MAX_CUBE_MAP_TEXTURE_SIZE,
            "GL_MAX_CUBE_MAP_TEXTURE_SIZE",
        )?,
        max_renderbuffer_size: u(
            glow_const::MAX_RENDERBUFFER_SIZE,
            "GL_MAX_RENDERBUFFER_SIZE",
        )?,
        max_color_attachments: u(
            glow_const::MAX_COLOR_ATTACHMENTS,
            "GL_MAX_COLOR_ATTACHMENTS",
        )?,
        max_draw_buffers: u(glow_const::MAX_DRAW_BUFFERS, "GL_MAX_DRAW_BUFFERS")?,
        max_vertex_attributes: u(glow_const::MAX_VERTEX_ATTRIBS, "GL_MAX_VERTEX_ATTRIBS")?,
        max_viewport_dimensions: [
            to_u32(pair[0], "GL_MAX_VIEWPORT_DIMS")?,
            to_u32(pair[1], "GL_MAX_VIEWPORT_DIMS")?,
        ],
        max_viewports: optional(
            matches!(profile, GlFamilyProfile::Desktop { major: 4, minor } if minor >= 1),
            glow_const::MAX_VIEWPORTS,
            "GL_MAX_VIEWPORTS",
        )?,
        max_vertex_texture_image_units: u(
            glow_const::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
            "GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS",
        )?,
        max_fragment_texture_image_units: u(
            glow_const::MAX_TEXTURE_IMAGE_UNITS,
            "GL_MAX_TEXTURE_IMAGE_UNITS",
        )?,
        max_combined_texture_image_units: u(
            glow_const::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS",
        )?,
        max_uniform_buffer_bindings: u(
            glow_const::MAX_UNIFORM_BUFFER_BINDINGS,
            "GL_MAX_UNIFORM_BUFFER_BINDINGS",
        )?,
        max_uniform_block_size: u64::from(u(
            glow_const::MAX_UNIFORM_BLOCK_SIZE,
            "GL_MAX_UNIFORM_BLOCK_SIZE",
        )?),
        uniform_buffer_offset_alignment: u64::from(u(
            glow_const::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
            "GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT",
        )?),
        max_vertex_uniform_blocks: u(
            glow_const::MAX_VERTEX_UNIFORM_BLOCKS,
            "GL_MAX_VERTEX_UNIFORM_BLOCKS",
        )?,
        max_fragment_uniform_blocks: u(
            glow_const::MAX_FRAGMENT_UNIFORM_BLOCKS,
            "GL_MAX_FRAGMENT_UNIFORM_BLOCKS",
        )?,
        max_compute_uniform_blocks: optional(
            compute,
            glow_const::MAX_COMPUTE_UNIFORM_BLOCKS,
            "GL_MAX_COMPUTE_UNIFORM_BLOCKS",
        )?,
        max_combined_uniform_blocks: u(
            glow_const::MAX_COMBINED_UNIFORM_BLOCKS,
            "GL_MAX_COMBINED_UNIFORM_BLOCKS",
        )?,
        max_storage_buffer_bindings: optional(
            storage,
            glow_const::MAX_SHADER_STORAGE_BUFFER_BINDINGS,
            "GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS",
        )?,
        max_storage_block_size: u64::from(optional(
            storage,
            glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE,
            "GL_MAX_SHADER_STORAGE_BLOCK_SIZE",
        )?),
        storage_buffer_offset_alignment: u64::from(optional(
            storage,
            glow_const::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT,
            "GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT",
        )?),
        max_vertex_storage_blocks: optional(
            storage,
            glow_const::MAX_VERTEX_SHADER_STORAGE_BLOCKS,
            "GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS",
        )?,
        max_fragment_storage_blocks: optional(
            storage,
            glow_const::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
            "GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS",
        )?,
        max_compute_storage_blocks: optional(
            storage,
            glow_const::MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS",
        )?,
        max_combined_storage_blocks: optional(
            storage,
            glow_const::MAX_COMBINED_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS",
        )?,
        max_image_units: optional(image, glow_const::MAX_IMAGE_UNITS, "GL_MAX_IMAGE_UNITS")?,
        max_combined_image_units: optional(
            image,
            glow_const::MAX_COMBINED_IMAGE_UNIFORMS,
            "GL_MAX_COMBINED_IMAGE_UNIFORMS",
        )?,
        max_samples: u(glow_const::MAX_SAMPLES, "GL_MAX_SAMPLES")?,
        max_color_texture_samples: optional(
            texture_multisample,
            glow_const::MAX_COLOR_TEXTURE_SAMPLES,
            "GL_MAX_COLOR_TEXTURE_SAMPLES",
        )?
        .max(1),
        max_depth_texture_samples: optional(
            texture_multisample,
            glow_const::MAX_DEPTH_TEXTURE_SAMPLES,
            "GL_MAX_DEPTH_TEXTURE_SAMPLES",
        )?
        .max(1),
        max_integer_samples: optional(
            texture_multisample,
            glow_const::MAX_INTEGER_SAMPLES,
            "GL_MAX_INTEGER_SAMPLES",
        )?
        .max(1),
        max_compute_work_group_count: [
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[0]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[1]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[2]",
            )?,
        ],
        max_compute_work_group_size: [
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[0]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[1]",
            )?,
            indexed(
                compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[2]",
            )?,
        ],
        max_compute_work_group_invocations: optional(
            compute,
            glow_const::MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
            "GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS",
        )?,
        max_multi_draw_indirect_count: None,
        // GL_QUERY_COUNTER_BITS is queried with glGetQueryiv(target, pname),
        // not glGetIntegerv. The small discovery trait intentionally has no
        // query-object API, so preserve it as unavailable instead of issuing
        // an invalid query or inferring timer support from a version string.
        query_counter_bits: 0,
        max_texture_anisotropy: anisotropy,
    })
}

fn baseline_formats(profile: GlFamilyProfile) -> Result<GlFormatTable, NativeDiscoveryError> {
    let mut table = GlFormatTable::default();
    for format in [
        GlFormat::Rgba8Unorm,
        GlFormat::Rgba8Srgb,
        GlFormat::Depth32Float,
    ] {
        table
            .record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                // These are the unconditional profile baseline, not driver
                // operation probes.  Optional formats stay absent until a
                // provider records a real operation probe.
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: format != GlFormat::Depth32Float,
                renderable: true,
                blendable: format != GlFormat::Depth32Float,
                storage_read: false,
                storage_write: false,
                copy_source: format != GlFormat::Depth32Float,
                copy_destination: format != GlFormat::Depth32Float,
            })
            .map_err(|error| {
                NativeDiscoveryError::Snapshot(GlDiscoveryError::InvalidFormats(error))
            })?;
    }
    for format in [
        GlFormat::Etc2Rgb8Unorm,
        GlFormat::Etc2Rgb8Srgb,
        GlFormat::Etc2Rgba8Unorm,
        GlFormat::Etc2Rgba8Srgb,
        GlFormat::Etc2Rgb8A1Unorm,
        GlFormat::Etc2Rgb8A1Srgb,
        GlFormat::EacR11Unorm,
        GlFormat::EacRg11Unorm,
        GlFormat::EacR11Snorm,
        GlFormat::EacRg11Snorm,
    ] {
        if format.is_core_compressed_for(profile) {
            table
                .record(GlFormatCapabilities {
                    format,
                    resource_kind: GlFormatResourceKind::Texture,
                    sample_count: 1,
                    evidence: GlFormatEvidence::CoreGuaranteed,
                    sampled: true,
                    filterable: false,
                    renderable: false,
                    blendable: false,
                    storage_read: false,
                    storage_write: false,
                    copy_source: false,
                    copy_destination: false,
                })
                .map_err(|error| {
                    NativeDiscoveryError::Snapshot(GlDiscoveryError::InvalidFormats(error))
                })?;
        }
    }
    Ok(table)
}

fn nonnegative(value: Option<i64>, name: &'static str) -> Result<u32, NativeDiscoveryError> {
    value
        .ok_or(NativeDiscoveryError::QueryFailed(name))
        .and_then(|value| to_u32(value, name))
}
fn to_u32(value: i64, name: &'static str) -> Result<u32, NativeDiscoveryError> {
    u32::try_from(value).map_err(|_| NativeDiscoveryError::QueryFailed(name))
}

/// Parses only Fluxel's supported native profiles: GL 4.x and GLES 3.x.
pub(crate) fn parse_native_profile(version: &str) -> Option<GlFamilyProfile> {
    let embedded = version.strip_prefix("OpenGL ES ");
    let text = embedded.unwrap_or(version);
    let mut digits = text
        .split(|character: char| !character.is_ascii_digit() && character != '.')
        .find(|part| part.contains('.'))?
        .split('.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    match (embedded.is_some(), major) {
        (false, 4) => Some(GlFamilyProfile::Desktop { major, minor }),
        (true, 3) => Some(GlFamilyProfile::Embedded { major, minor }),
        _ => None,
    }
}

mod glow_const {
    pub const VENDOR: u32 = 0x1F00;
    pub const RENDERER: u32 = 0x1F01;
    pub const VERSION: u32 = 0x1F02;
    pub const EXTENSIONS: u32 = 0x1F03;
    pub const SHADING_LANGUAGE_VERSION: u32 = 0x8B8C;
    pub const NUM_EXTENSIONS: u32 = 0x821D;
    pub const MAX_TEXTURE_SIZE: u32 = 0x0D33;
    pub const MAX_3D_TEXTURE_SIZE: u32 = 0x8073;
    pub const MAX_ARRAY_TEXTURE_LAYERS: u32 = 0x88FF;
    pub const MAX_CUBE_MAP_TEXTURE_SIZE: u32 = 0x851C;
    pub const MAX_RENDERBUFFER_SIZE: u32 = 0x84E8;
    pub const MAX_COLOR_ATTACHMENTS: u32 = 0x8CDF;
    pub const MAX_DRAW_BUFFERS: u32 = 0x8824;
    pub const MAX_VERTEX_ATTRIBS: u32 = 0x8869;
    pub const MAX_VIEWPORT_DIMS: u32 = 0x0D3A;
    pub const MAX_VIEWPORTS: u32 = 0x825B;
    pub const MAX_VERTEX_TEXTURE_IMAGE_UNITS: u32 = 0x8B4C;
    pub const MAX_TEXTURE_IMAGE_UNITS: u32 = 0x8872;
    pub const MAX_COMBINED_TEXTURE_IMAGE_UNITS: u32 = 0x8B4D;
    pub const MAX_UNIFORM_BUFFER_BINDINGS: u32 = 0x8A2F;
    pub const MAX_UNIFORM_BLOCK_SIZE: u32 = 0x8A30;
    pub const UNIFORM_BUFFER_OFFSET_ALIGNMENT: u32 = 0x8A34;
    pub const MAX_VERTEX_UNIFORM_BLOCKS: u32 = 0x8A2B;
    pub const MAX_FRAGMENT_UNIFORM_BLOCKS: u32 = 0x8A2D;
    pub const MAX_COMPUTE_UNIFORM_BLOCKS: u32 = 0x91BB;
    pub const MAX_COMBINED_UNIFORM_BLOCKS: u32 = 0x8A2E;
    pub const MAX_SHADER_STORAGE_BUFFER_BINDINGS: u32 = 0x90DD;
    pub const MAX_SHADER_STORAGE_BLOCK_SIZE: u32 = 0x90DE;
    pub const SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: u32 = 0x90DF;
    pub const MAX_VERTEX_SHADER_STORAGE_BLOCKS: u32 = 0x90D6;
    pub const MAX_FRAGMENT_SHADER_STORAGE_BLOCKS: u32 = 0x90DA;
    pub const MAX_COMPUTE_SHADER_STORAGE_BLOCKS: u32 = 0x90DB;
    pub const MAX_COMBINED_SHADER_STORAGE_BLOCKS: u32 = 0x90DC;
    pub const MAX_IMAGE_UNITS: u32 = 0x8F38;
    pub const MAX_COMBINED_IMAGE_UNIFORMS: u32 = 0x90CF;
    pub const MAX_SAMPLES: u32 = 0x8D57;
    pub const MAX_COLOR_TEXTURE_SAMPLES: u32 = 0x910E;
    pub const MAX_DEPTH_TEXTURE_SAMPLES: u32 = 0x910F;
    pub const MAX_INTEGER_SAMPLES: u32 = 0x9110;
    pub const MAX_COMPUTE_WORK_GROUP_COUNT: u32 = 0x91BE;
    pub const MAX_COMPUTE_WORK_GROUP_SIZE: u32 = 0x91BF;
    pub const MAX_COMPUTE_WORK_GROUP_INVOCATIONS: u32 = 0x90EB;
    pub const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
}

#[cfg(test)]
mod tests {
    use super::super::{ContextEpoch, DeviceIdentity};
    use super::{
        ContextStamp, GlCapability, GlFamilyProfile, GlFormat, GlFormatEvidence,
        NativeDiscoveryError, NativeGlQuery, discover_with_query, glow_const, parse_native_profile,
        required_string,
    };

    struct MissingStringQuery;
    impl NativeGlQuery for MissingStringQuery {
        fn take_error(&self) -> bool {
            false
        }
        fn string(&self, _: u32) -> Option<String> {
            None
        }
        fn integer(&self, _: u32) -> Option<i64> {
            None
        }
        fn integer_pair(&self, _: u32) -> Option<[i64; 2]> {
            None
        }
        fn indexed_integer(&self, _: u32, _: u32) -> Option<i64> {
            None
        }
        fn float(&self, _: u32) -> Option<f32> {
            None
        }
        fn indexed_string(&self, _: u32, _: u32) -> Option<String> {
            None
        }
    }

    struct CompleteQuery {
        preexisting_error: bool,
    }
    impl NativeGlQuery for CompleteQuery {
        fn take_error(&self) -> bool {
            self.preexisting_error
        }
        fn string(&self, name: u32) -> Option<String> {
            match name {
                glow_const::VERSION => Some("4.6 test".into()),
                glow_const::SHADING_LANGUAGE_VERSION => Some("4.60 test".into()),
                glow_const::VENDOR => Some("test-vendor".into()),
                glow_const::RENDERER => Some("test-renderer".into()),
                _ => None,
            }
        }
        fn integer(&self, name: u32) -> Option<i64> {
            Some(if name == glow_const::NUM_EXTENSIONS {
                0
            } else {
                16_384
            })
        }
        fn integer_pair(&self, _: u32) -> Option<[i64; 2]> {
            Some([16_384; 2])
        }
        fn indexed_integer(&self, _: u32, _: u32) -> Option<i64> {
            Some(16_384)
        }
        fn float(&self, _: u32) -> Option<f32> {
            Some(16.0)
        }
        fn indexed_string(&self, _: u32, _: u32) -> Option<String> {
            None
        }
    }

    fn stamp() -> ContextStamp {
        ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
    }

    #[test]
    fn profile_parser_is_strict() {
        assert_eq!(
            parse_native_profile("4.6.0 AMD"),
            Some(GlFamilyProfile::Desktop { major: 4, minor: 6 })
        );
        assert_eq!(
            parse_native_profile("OpenGL ES 3.2 Mesa"),
            Some(GlFamilyProfile::Embedded { major: 3, minor: 2 })
        );
        assert_eq!(parse_native_profile("OpenGL ES 2.0"), None);
        assert_eq!(parse_native_profile("OpenGL 3.3"), None);
    }

    #[test]
    fn mockable_required_queries_fail_closed() {
        assert_eq!(
            required_string(&MissingStringQuery, glow_const::VERSION, "GL_VERSION"),
            Err(NativeDiscoveryError::InvalidContextString("GL_VERSION"))
        );
    }

    #[test]
    fn preexisting_error_is_not_cleared_and_reused_for_discovery() {
        assert_eq!(
            discover_with_query(
                &CompleteQuery {
                    preexisting_error: true
                },
                stamp()
            ),
            Err(NativeDiscoveryError::PreExistingGlError)
        );
    }

    #[test]
    fn not_run_disables_every_optional_command_capability() {
        let snapshot = discover_with_query(
            &CompleteQuery {
                preexisting_error: false,
            },
            stamp(),
        )
        .expect("complete mock discovery");
        for capability in [
            GlCapability::Compute,
            GlCapability::StorageBuffer,
            GlCapability::StorageImage,
            GlCapability::IndirectDraw,
            GlCapability::IndirectDispatch,
            GlCapability::MultiDrawIndirect,
            GlCapability::TimerQuery,
        ] {
            assert!(
                !snapshot.capabilities().supports(capability),
                "{capability:?}"
            );
        }
    }

    #[test]
    fn static_native_baseline_does_not_claim_optional_format_support() {
        let formats = super::baseline_formats(GlFamilyProfile::Desktop { major: 4, minor: 6 })
            .expect("profile guarantees are well-formed");
        assert!(formats.get(GlFormat::Rgba16Float, 1).is_none());
        assert!(formats.get(GlFormat::Rgba32Float, 1).is_none());
        let depth = formats
            .get(GlFormat::Depth32Float, 1)
            .expect("required depth fact");
        assert_eq!(depth.evidence, GlFormatEvidence::CoreGuaranteed);
        assert!(depth.renderable);
        assert!(!depth.filterable && !depth.blendable);
        assert!(!depth.copy_source && !depth.copy_destination);
    }

    #[test]
    fn gles3_records_only_exact_core_etc2_eac_facts() {
        let formats = super::baseline_formats(GlFamilyProfile::Embedded { major: 3, minor: 0 })
            .expect("GLES3 core compressed facts");
        assert_eq!(
            formats.get(GlFormat::Etc2Rgba8Unorm, 1).unwrap().evidence,
            GlFormatEvidence::CoreGuaranteed
        );
        assert_eq!(
            formats.get(GlFormat::EacRg11Snorm, 1).unwrap().evidence,
            GlFormatEvidence::CoreGuaranteed
        );
        assert!(formats.get(GlFormat::Bc1RgbUnorm, 1).is_none());
    }
}
