//! Browser execution domain over the RHI-owned WebGL2 context record.

use web_sys::{WebGl2RenderingContext, WebGlSampler};

use super::super::{
    BufferId, ContextStamp, GlAddressMode, GlBufferDesc, GlBufferRange, GlCompareFunction,
    GlContextLifecycle, GlCopyDomainApi, GlDiscoverySnapshot, GlError, GlFamilyApi, GlFilterMode,
    GlFormat, GlMipmapFilterMode, GlPixelLayout, GlPixelStoreState, GlReadback, GlRenderBufferDesc,
    GlResourceApi, GlSamplerApi, GlSamplerDesc, GlTextureDesc, GlTextureDimension, GlTextureRegion,
    OwnerThreadIdentity, RenderbufferId, SamplerId, TextureId,
};
use super::discovery::{BrowserBuffer, BrowserSampler, BrowserTexture, WebGl2BrowserDiscovery};

impl GlFamilyApi for WebGl2BrowserDiscovery {
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle.get()
    }

    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner_thread
    }

    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        WebGl2BrowserDiscovery::assert_owner_thread(self, operation)
    }

    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.snapshot
    }

    fn context_lost(&mut self) -> Result<(), GlError> {
        self.assert_owner_thread("context-lost")?;
        self.lifecycle.set(GlContextLifecycle::Lost);
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        Ok(())
    }

    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        self.assert_owner_thread("context-restored")?;
        if self.lifecycle.get() != GlContextLifecycle::Lost {
            return Err(Self::validation("context-restored", "context is not lost"));
        }
        let epoch = self
            .snapshot
            .context_stamp()
            .epoch
            .checked_next()
            .ok_or_else(|| GlError::Driver {
                operation: "context-restored",
                message: "context epoch exhausted".into(),
            })?;
        self.lifecycle.set(GlContextLifecycle::Restoring);
        let stamp = ContextStamp::new(self.snapshot.context_stamp().device, epoch);
        let replacement = Self::open(stamp, self.canvas.clone())?;
        self.raw = replacement.raw;
        self.glow = replacement.glow;
        self.snapshot = replacement.snapshot;
        self.owner_thread = replacement.owner_thread;
        self.lifecycle = replacement.lifecycle;
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        self.next_buffer_slot = 0;
        self.next_texture_slot = 0;
        self.next_sampler_slot = 0;
        self.pixel_store = GlPixelStoreState::DEFAULT;
        Ok(stamp)
    }
}

impl GlResourceApi for WebGl2BrowserDiscovery {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        const OP: &str = "create-buffer";
        self.assert_provider_ready(OP)?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid buffer descriptor"))?;
        if desc.size > 9_007_199_254_740_991 {
            return Err(Self::validation(
                OP,
                "buffer size exceeds exact browser integer range",
            ));
        }
        let raw = self
            .raw
            .create_buffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        self.raw
            .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&raw));
        self.raw.buffer_data_with_f64(
            WebGl2RenderingContext::ARRAY_BUFFER,
            desc.size as f64,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
        if let Err(error) = self.driver_error(OP) {
            self.raw.delete_buffer(Some(&raw));
            return Err(error);
        }
        let slot = Self::allocate_slot(&mut self.next_buffer_slot, OP)?;
        let id = BufferId::new(self.context_stamp(), slot, 0);
        self.buffers.insert(
            slot,
            BrowserBuffer {
                generation: id.generation,
                raw,
                desc,
            },
        );
        Ok(id)
    }

    fn create_texture_resource(&mut self, desc: GlTextureDesc) -> Result<TextureId, GlError> {
        const OP: &str = "create-texture";
        self.assert_provider_ready(OP)?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid texture descriptor"))?;
        if desc.dimension != GlTextureDimension::D2 || desc.sample_count != 1 {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "WebGL2 texture allocation slice currently supports single-sample 2D textures",
            });
        }
        let internal_format = webgl2_internal_format(desc.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no proven WebGL2 texture storage mapping",
        })?;
        let compressed = desc.format.compressed_info().is_some();
        if compressed && self.snapshot.formats().get(desc.format, 1).is_none() {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "compressed format was not acquired for this WebGL2 context",
            });
        }
        let raw = self
            .raw
            .create_texture()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        self.raw
            .bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&raw));
        if !compressed {
            self.raw.tex_storage_2d(
                WebGl2RenderingContext::TEXTURE_2D,
                i32::try_from(desc.mip_level_count)
                    .map_err(|_| Self::validation(OP, "mip count exceeds i32"))?,
                internal_format,
                i32::try_from(desc.extent.width)
                    .map_err(|_| Self::validation(OP, "width exceeds i32"))?,
                i32::try_from(desc.extent.height)
                    .map_err(|_| Self::validation(OP, "height exceeds i32"))?,
            );
        }
        if let Err(error) = self.driver_error(OP) {
            self.raw.delete_texture(Some(&raw));
            return Err(error);
        }
        let slot = Self::allocate_slot(&mut self.next_texture_slot, OP)?;
        let id = TextureId::new(self.context_stamp(), slot, 0);
        self.textures.insert(
            slot,
            BrowserTexture {
                generation: id.generation,
                raw,
                desc,
            },
        );
        Ok(id)
    }

    fn destroy_buffer_resource(&mut self, id: BufferId) -> Result<(), GlError> {
        const OP: &str = "destroy-buffer";
        self.buffer(OP, id)?;
        let entry = self
            .buffers
            .remove(&id.slot)
            .ok_or_else(|| Self::validation(OP, "buffer allocation disappeared"))?;
        self.raw.delete_buffer(Some(&entry.raw));
        self.driver_error(OP)
    }

    fn destroy_texture_resource(&mut self, id: TextureId) -> Result<(), GlError> {
        const OP: &str = "destroy-texture";
        self.texture(OP, id)?;
        let entry = self
            .textures
            .remove(&id.slot)
            .ok_or_else(|| Self::validation(OP, "texture allocation disappeared"))?;
        self.raw.delete_texture(Some(&entry.raw));
        self.driver_error(OP)
    }

    fn create_render_buffer(
        &mut self,
        _desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, GlError> {
        Err(GlError::Unsupported {
            operation: "create-render-buffer",
            reason: "WebGL2 renderbuffer allocation slice is not installed",
        })
    }

    fn destroy_render_buffer(&mut self, _id: RenderbufferId) -> Result<(), GlError> {
        Err(GlError::Unsupported {
            operation: "destroy-render-buffer",
            reason: "WebGL2 renderbuffer allocation slice is not installed",
        })
    }
}

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
            .map_err(|_| Self::validation(OP, "invalid source buffer range"))?;
        destination
            .validate_for(destination_desc)
            .map_err(|_| Self::validation(OP, "invalid destination buffer range"))?;
        if source.size != destination.size {
            return Err(Self::validation(OP, "copy ranges have different sizes"));
        }
        self.raw
            .bind_buffer(WebGl2RenderingContext::COPY_READ_BUFFER, Some(&source_raw));
        self.raw.bind_buffer(
            WebGl2RenderingContext::COPY_WRITE_BUFFER,
            Some(&destination_raw),
        );
        self.raw.copy_buffer_sub_data_with_f64_and_f64_and_f64(
            WebGl2RenderingContext::COPY_READ_BUFFER,
            WebGl2RenderingContext::COPY_WRITE_BUFFER,
            source.offset as f64,
            destination.offset as f64,
            source.size as f64,
        );
        self.driver_error(OP)
    }

    fn copy_texture_region(
        &mut self,
        _source: GlTextureRegion,
        _destination: GlTextureRegion,
    ) -> Result<(), GlError> {
        self.assert_provider_ready("copy-texture")?;
        Err(GlError::Unsupported {
            operation: "copy-texture",
            reason: "WebGL2 framebuffer copy slice is not installed",
        })
    }

    fn upload_buffer(&mut self, _destination: GlBufferRange, _bytes: &[u8]) -> Result<(), GlError> {
        Err(GlError::Unsupported {
            operation: "upload-buffer",
            reason: "WebGL2 buffer upload slice is not installed",
        })
    }

    fn read_buffer(&mut self, _source: GlBufferRange) -> Result<Vec<u8>, GlError> {
        Err(GlError::Unsupported {
            operation: "read-buffer",
            reason: "WebGL2 buffer readback slice is not installed",
        })
    }

    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        _layout: GlPixelLayout,
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
            .map_err(|_| Self::validation(OP, "invalid texture region"))?;
        let Some(info) = desc.format.compressed_info() else {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "uncompressed pixel upload slice is not installed",
            });
        };
        let complete = desc.mip_extent(destination.subresource.mip_level);
        if desc.dimension != GlTextureDimension::D2
            || destination.subresource.base_layer != 0
            || destination.subresource.layer_count != 1
            || destination.origin != [0; 3]
            || destination.extent.depth_or_layers != 1
            || complete != Some(destination.extent)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "compressed upload must define one complete 2D mip",
            });
        }
        let exact = info
            .checked_encoded_size(destination.extent.width, destination.extent.height)
            .map_err(|_| Self::validation(OP, "compressed encoded size overflow"))?;
        if u64::try_from(bytes.len()).ok() != Some(exact) {
            return Err(Self::validation(
                OP,
                "compressed bytes do not match exact block layout",
            ));
        }
        let internal = webgl2_internal_format(desc.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "compressed format has no WebGL2 internal format",
        })?;
        self.raw
            .bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&raw));
        self.raw.compressed_tex_image_2d_with_u8_array(
            WebGl2RenderingContext::TEXTURE_2D,
            i32::try_from(destination.subresource.mip_level)
                .map_err(|_| Self::validation(OP, "mip level exceeds GLint"))?,
            internal,
            i32::try_from(destination.extent.width)
                .map_err(|_| Self::validation(OP, "width exceeds GLsizei"))?,
            i32::try_from(destination.extent.height)
                .map_err(|_| Self::validation(OP, "height exceeds GLsizei"))?,
            0,
            bytes,
        );
        self.driver_error(OP)
    }

    fn read_texture(
        &mut self,
        _source: GlTextureRegion,
        _layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError> {
        self.assert_provider_ready("read-texture")?;
        Err(GlError::Unsupported {
            operation: "read-texture",
            reason: "WebGL2 framebuffer readback slice is not installed",
        })
    }

    fn pixel_store(&self) -> GlPixelStoreState {
        self.pixel_store
    }
}

impl GlSamplerApi for WebGl2BrowserDiscovery {
    fn create_sampler(&mut self, desc: GlSamplerDesc) -> Result<SamplerId, GlError> {
        const OP: &str = "create-sampler";
        self.assert_provider_ready(OP)?;
        desc.validate_for(&self.snapshot)
            .map_err(|_| Self::validation(OP, "invalid sampler descriptor"))?;
        let raw = self
            .raw
            .create_sampler()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        configure_sampler(&self.raw, &raw, desc);
        if let Err(error) = self.driver_error(OP) {
            self.raw.delete_sampler(Some(&raw));
            return Err(error);
        }
        let slot = Self::allocate_slot(&mut self.next_sampler_slot, OP)?;
        let id = SamplerId::new(self.context_stamp(), slot, 0);
        self.samplers.insert(
            slot,
            BrowserSampler {
                generation: id.generation,
                raw,
            },
        );
        Ok(id)
    }

    fn destroy_sampler(&mut self, id: SamplerId) -> Result<(), GlError> {
        const OP: &str = "destroy-sampler";
        self.validate_object_context(OP, id.context)?;
        let entry = self
            .samplers
            .get(&id.slot)
            .ok_or_else(|| Self::validation(OP, "sampler allocation is not live"))?;
        if entry.generation != id.generation {
            return Err(Self::validation(
                OP,
                "sampler allocation generation is stale",
            ));
        }
        let entry = self
            .samplers
            .remove(&id.slot)
            .ok_or_else(|| Self::validation(OP, "sampler allocation disappeared"))?;
        self.raw.delete_sampler(Some(&entry.raw));
        self.driver_error(OP)
    }
}

fn webgl2_internal_format(format: GlFormat) -> Option<u32> {
    match format {
        GlFormat::Rgba8Unorm => Some(WebGl2RenderingContext::RGBA8),
        GlFormat::Rgba8Srgb => Some(WebGl2RenderingContext::SRGB8_ALPHA8),
        GlFormat::Depth32Float => Some(WebGl2RenderingContext::DEPTH_COMPONENT32F),
        GlFormat::Bc1RgbUnorm => Some(0x83F0),
        GlFormat::Bc1RgbaUnorm => Some(0x83F1),
        GlFormat::Bc2RgbaUnorm => Some(0x83F2),
        GlFormat::Bc3RgbaUnorm => Some(0x83F3),
        GlFormat::Bc1RgbSrgb => Some(0x8C4C),
        GlFormat::Bc1RgbaSrgb => Some(0x8C4D),
        GlFormat::Bc2RgbaSrgb => Some(0x8C4E),
        GlFormat::Bc3RgbaSrgb => Some(0x8C4F),
        GlFormat::Bc4RUnorm => Some(0x8DBB),
        GlFormat::Bc4RSnorm => Some(0x8DBC),
        GlFormat::Bc5RgUnorm => Some(0x8DBD),
        GlFormat::Bc5RgSnorm => Some(0x8DBE),
        GlFormat::Bc6hRgbUfloat => Some(0x8E8F),
        GlFormat::Bc6hRgbSfloat => Some(0x8E8E),
        GlFormat::Bc7RgbaUnorm => Some(0x8E8C),
        GlFormat::Bc7RgbaSrgb => Some(0x8E8D),
        GlFormat::Etc2Rgb8Unorm => Some(0x9274),
        GlFormat::Etc2Rgb8Srgb => Some(0x9275),
        GlFormat::Etc2Rgb8A1Unorm => Some(0x9276),
        GlFormat::Etc2Rgb8A1Srgb => Some(0x9277),
        GlFormat::Etc2Rgba8Unorm => Some(0x9278),
        GlFormat::Etc2Rgba8Srgb => Some(0x9279),
        GlFormat::EacR11Unorm => Some(0x9270),
        GlFormat::EacR11Snorm => Some(0x9271),
        GlFormat::EacRg11Unorm => Some(0x9272),
        GlFormat::EacRg11Snorm => Some(0x9273),
        GlFormat::Astc { block, color_space } => {
            use super::super::{GlAstcBlock as B, GlCompressedColorSpace as C};
            let index = match block {
                B::B4x4 => 0,
                B::B5x4 => 1,
                B::B5x5 => 2,
                B::B6x5 => 3,
                B::B6x6 => 4,
                B::B8x5 => 5,
                B::B8x6 => 6,
                B::B8x8 => 7,
                B::B10x5 => 8,
                B::B10x6 => 9,
                B::B10x8 => 10,
                B::B10x10 => 11,
                B::B12x10 => 12,
                B::B12x12 => 13,
            };
            Some(match color_space {
                C::Linear => 0x93B0 + index,
                C::Srgb => 0x93D0 + index,
            })
        }
        _ => None,
    }
}

fn configure_sampler(raw: &WebGl2RenderingContext, sampler: &WebGlSampler, desc: GlSamplerDesc) {
    raw.sampler_parameteri(
        sampler,
        WebGl2RenderingContext::TEXTURE_WRAP_S,
        address_mode(desc.address_mode_u),
    );
    raw.sampler_parameteri(
        sampler,
        WebGl2RenderingContext::TEXTURE_WRAP_T,
        address_mode(desc.address_mode_v),
    );
    raw.sampler_parameteri(
        sampler,
        WebGl2RenderingContext::TEXTURE_WRAP_R,
        address_mode(desc.address_mode_w),
    );
    raw.sampler_parameteri(
        sampler,
        WebGl2RenderingContext::TEXTURE_MAG_FILTER,
        filter_mode(desc.mag_filter),
    );
    raw.sampler_parameteri(
        sampler,
        WebGl2RenderingContext::TEXTURE_MIN_FILTER,
        min_filter(desc),
    );
    raw.sampler_parameterf(
        sampler,
        WebGl2RenderingContext::TEXTURE_MIN_LOD,
        f32::from_bits(desc.lod_min_bits),
    );
    raw.sampler_parameterf(
        sampler,
        WebGl2RenderingContext::TEXTURE_MAX_LOD,
        f32::from_bits(desc.lod_max_bits),
    );
    if let Some(compare) = desc.compare {
        raw.sampler_parameteri(
            sampler,
            WebGl2RenderingContext::TEXTURE_COMPARE_MODE,
            WebGl2RenderingContext::COMPARE_REF_TO_TEXTURE as i32,
        );
        raw.sampler_parameteri(
            sampler,
            WebGl2RenderingContext::TEXTURE_COMPARE_FUNC,
            compare_function(compare),
        );
    }
    if let Some(anisotropy) = desc.max_anisotropy_bits {
        raw.sampler_parameterf(sampler, 0x84FE, f32::from_bits(anisotropy));
    }
}

const fn address_mode(mode: GlAddressMode) -> i32 {
    match mode {
        GlAddressMode::ClampToEdge => WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        GlAddressMode::Repeat => WebGl2RenderingContext::REPEAT as i32,
        GlAddressMode::MirroredRepeat => WebGl2RenderingContext::MIRRORED_REPEAT as i32,
    }
}

const fn filter_mode(mode: GlFilterMode) -> i32 {
    match mode {
        GlFilterMode::Nearest => WebGl2RenderingContext::NEAREST as i32,
        GlFilterMode::Linear => WebGl2RenderingContext::LINEAR as i32,
    }
}

const fn min_filter(desc: GlSamplerDesc) -> i32 {
    match (desc.min_filter, desc.mipmap_filter) {
        (GlFilterMode::Nearest, GlMipmapFilterMode::Nearest) => {
            WebGl2RenderingContext::NEAREST_MIPMAP_NEAREST as i32
        }
        (GlFilterMode::Nearest, GlMipmapFilterMode::Linear) => {
            WebGl2RenderingContext::NEAREST_MIPMAP_LINEAR as i32
        }
        (GlFilterMode::Linear, GlMipmapFilterMode::Nearest) => {
            WebGl2RenderingContext::LINEAR_MIPMAP_NEAREST as i32
        }
        (GlFilterMode::Linear, GlMipmapFilterMode::Linear) => {
            WebGl2RenderingContext::LINEAR_MIPMAP_LINEAR as i32
        }
    }
}

const fn compare_function(compare: GlCompareFunction) -> i32 {
    match compare {
        GlCompareFunction::Never => WebGl2RenderingContext::NEVER as i32,
        GlCompareFunction::Less => WebGl2RenderingContext::LESS as i32,
        GlCompareFunction::Equal => WebGl2RenderingContext::EQUAL as i32,
        GlCompareFunction::LessEqual => WebGl2RenderingContext::LEQUAL as i32,
        GlCompareFunction::Greater => WebGl2RenderingContext::GREATER as i32,
        GlCompareFunction::NotEqual => WebGl2RenderingContext::NOTEQUAL as i32,
        GlCompareFunction::GreaterEqual => WebGl2RenderingContext::GEQUAL as i32,
        GlCompareFunction::Always => WebGl2RenderingContext::ALWAYS as i32,
    }
}
