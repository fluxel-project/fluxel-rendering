//! Browser-owned WebGL2 discovery.
//!
//! The Host/JS bridge owns canvas creation, DOM events, RAF, and context-loss
//! listeners. RHI creates and owns the WebGL2 context associated with that
//! Host-provided canvas, then gathers immutable evidence for its `ContextStamp`.

#![cfg(target_arch = "wasm32")]

use core::cell::Cell;
use js_sys::{Array, Object, Reflect};
use std::collections::BTreeMap;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlSampler, WebGlTexture};

use super::{
    BufferId, ContextStamp, GlAddressMode, GlBufferDesc, GlBufferRange, GlCapability,
    GlCompareFunction, GlContextFlags, GlContextInfo, GlContextLifecycle, GlCopyDomainApi,
    GlDiscoveryBuilder, GlDiscoverySnapshot, GlError, GlExtensionSet, GlFamilyApi, GlFamilyProfile,
    GlFilterMode, GlFiniteF32, GlFormat, GlFormatCapabilities, GlFormatEvidence,
    GlFormatResourceKind, GlFormatTable, GlKnownExtension, GlLimits, GlMipmapFilterMode,
    GlOperationProbe, GlPixelLayout, GlPixelStoreState, GlReadback, GlResourceApi, GlSamplerApi,
    GlSamplerDesc, GlTextureDesc, GlTextureDimension, GlTextureRegion, OwnerThreadIdentity,
    SamplerId, TextureId,
};

/// Immutable WebGL2 discovery evidence plus the callable `glow` facade.
///
/// RHI retains the Host-provided canvas and the context it created. `glow`
/// receives a cloned JS context handle, while `raw` remains the RHI-owned
/// browser API handle for discovery and later provider work.
/// It is called only after the fallible WebGL2 version check below succeeds.
/// The remaining assumptions made by glow are WebGL binding invariants (a real
/// WebGL2 context returns a string `VERSION` and an array-or-null extension
/// list); violations are browser defects rather than recoverable application
/// input.  All application-observable queries in this module remain fallible.
pub(crate) struct WebGl2BrowserDiscovery {
    canvas: HtmlCanvasElement,
    raw: WebGl2RenderingContext,
    glow: glow::Context,
    snapshot: GlDiscoverySnapshot,
    owner_thread: OwnerThreadIdentity,
    lifecycle: Cell<GlContextLifecycle>,
    buffers: BTreeMap<u32, BrowserBuffer>,
    textures: BTreeMap<u32, BrowserTexture>,
    samplers: BTreeMap<u32, BrowserSampler>,
    next_buffer_slot: u32,
    next_texture_slot: u32,
    next_sampler_slot: u32,
    pixel_store: GlPixelStoreState,
}

struct BrowserBuffer {
    generation: u32,
    raw: WebGlBuffer,
    desc: GlBufferDesc,
}
struct BrowserTexture {
    generation: u32,
    raw: WebGlTexture,
    desc: GlTextureDesc,
}
struct BrowserSampler {
    generation: u32,
    raw: WebGlSampler,
}

impl WebGl2BrowserDiscovery {
    /// Creates and owns a WebGL2 context for a Host-provided canvas, then
    /// discovers its current generation without issuing a rendering command.
    ///
    /// This deliberately preserves the 0.14 readback/residency-oracle setting:
    /// the default framebuffer remains available after presentation.
    pub(crate) fn open(stamp: ContextStamp, canvas: HtmlCanvasElement) -> Result<Self, GlError> {
        // Context creation is itself a browser side effect, so ownership is
        // captured before configuring or touching the canvas.
        let owner_thread = OwnerThreadIdentity::current();
        let options = Object::new();
        Reflect::set(
            &options,
            &JsValue::from_str("preserveDrawingBuffer"),
            &JsValue::TRUE,
        )
        .map_err(|value| js_error("configure WebGL2 context", value))?;
        let value = canvas
            .get_context_with_context_options("webgl2", &options)
            .map_err(|value| js_error("create WebGL2 context", value))?
            .ok_or_else(|| driver("create WebGL2 context", "browser did not provide WebGL2"))?;
        let raw = value.dyn_into::<WebGl2RenderingContext>().map_err(|_| {
            driver(
                "create WebGL2 context",
                "browser returned a non-WebGL2 context",
            )
        })?;
        Self::from_owned_context(stamp, canvas, raw, owner_thread)
    }

    /// Internal ownership-transfer seam for RHI context restoration only.
    ///
    /// It is intentionally private: Host never creates or owns the WebGL2
    /// context in the Fluxel boundary model.
    fn from_owned_context(
        stamp: ContextStamp,
        canvas: HtmlCanvasElement,
        raw: WebGl2RenderingContext,
        owner_thread: OwnerThreadIdentity,
    ) -> Result<Self, GlError> {
        ensure_context_live(&raw, "discover WebGL2 context")?;
        let version = string_parameter(&raw, WebGl2RenderingContext::VERSION, "VERSION")?;
        require_webgl2_version(&version)?;

        // Do this after validating the only glow constructor precondition that
        // comes from application-visible browser state.  We deliberately do
        // not use glow for discovery: its browser constructor panics on JS
        // binding invariant violations instead of returning `Result`.
        let glow = glow::Context::from_webgl2_context(raw.clone());
        let extensions = discover_extensions(&raw)?;
        let context = GlContextInfo::new(
            GlFamilyProfile::WebGl2,
            version,
            string_parameter(
                &raw,
                WebGl2RenderingContext::SHADING_LANGUAGE_VERSION,
                "SHADING_LANGUAGE_VERSION",
            )?,
            string_parameter(&raw, WebGl2RenderingContext::VENDOR, "VENDOR")?,
            string_parameter(&raw, WebGl2RenderingContext::RENDERER, "RENDERER")?,
            browser_identity()?,
            GlContextFlags::default(),
        );
        let limits = discover_limits(&raw, &extensions)?;
        let formats = webgl2_baseline_formats(&extensions)?;
        let mut builder = GlDiscoveryBuilder::new(stamp, context, extensions, limits, formats)
            .map_err(discovery_error)?;

        // WebGL2 has no general compute, storage, or indirect mapping.  Timer
        // queries are the sole currently normalized browser extension domain;
        // no command is issued while discovering it.
        builder.resolve(
            GlCapability::TimerQuery,
            super::CoreOrExtension {
                desktop_core: None,
                embedded_core: None,
                extension: Some(GlKnownExtension::ExtDisjointTimerQueryWebgl2),
                extension_requires_probe: false,
            },
            GlOperationProbe::NotRequired,
        );
        // A context may be lost between any two browser calls. Do not publish
        // a discovery snapshot across that boundary.
        ensure_context_live(&raw, "finish WebGL2 discovery")?;
        Ok(Self {
            canvas,
            raw,
            glow,
            snapshot: builder.build(),
            owner_thread,
            lifecycle: Cell::new(GlContextLifecycle::Active),
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
            next_buffer_slot: 0,
            next_texture_slot: 0,
            next_sampler_slot: 0,
            pixel_store: GlPixelStoreState::DEFAULT,
        })
    }

    /// Returns the discovery evidence bound to the supplied stamp.
    pub(crate) fn snapshot(&self) -> &GlDiscoverySnapshot {
        &self.snapshot
    }

    /// Preflight every future browser provider call inside this module.
    ///
    /// Raw WebGL and glow handles deliberately have no crate-visible borrowing
    /// method, so an executable seam cannot bypass this owner/loss check.
    fn assert_provider_ready(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_owner_thread(operation)?;
        let lifecycle = self.lifecycle.get();
        if lifecycle != GlContextLifecycle::Active {
            return Err(GlError::InvalidLifecycle {
                operation,
                lifecycle,
            });
        }
        if self.raw.is_context_lost() {
            // Host owns loss events, while RHI owns the execution boundary.
            // The first provider preflight that observes loss durably records
            // it before returning, preventing another browser API call.
            self.lifecycle.set(GlContextLifecycle::Lost);
            Err(GlError::ContextLost { operation })
        } else {
            Ok(())
        }
    }

    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner_thread {
            Ok(())
        } else {
            Err(GlError::WrongThread {
                operation,
                expected: self.owner_thread,
                actual,
            })
        }
    }

    fn allocate_slot(next: &mut u32, operation: &'static str) -> Result<u32, GlError> {
        let slot = *next;
        *next = next.checked_add(1).ok_or_else(|| GlError::Driver {
            operation,
            message: "object allocation slots exhausted".into(),
        })?;
        Ok(slot)
    }

    fn validation(operation: &'static str, message: &'static str) -> GlError {
        GlError::Validation {
            operation,
            message: message.into(),
        }
    }

    fn driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        let error = self.raw.get_error();
        if error == WebGl2RenderingContext::NO_ERROR {
            Ok(())
        } else {
            Err(GlError::Driver {
                operation,
                message: format!("WebGL error 0x{error:04x}"),
            })
        }
    }

    fn buffer(&self, operation: &'static str, id: BufferId) -> Result<&BrowserBuffer, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.buffers.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(operation, "buffer allocation is not live")),
        }
    }

    fn texture(&self, operation: &'static str, id: TextureId) -> Result<&BrowserTexture, GlError> {
        self.validate_object_context(operation, id.context)?;
        match self.textures.get(&id.slot) {
            Some(entry) if entry.generation == id.generation => Ok(entry),
            _ => Err(Self::validation(
                operation,
                "texture allocation is not live",
            )),
        }
    }
}

fn ensure_context_live(
    raw: &WebGl2RenderingContext,
    operation: &'static str,
) -> Result<(), GlError> {
    (!raw.is_context_lost())
        .then_some(())
        .ok_or(GlError::ContextLost { operation })
}

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
            use super::{GlAstcBlock as B, GlCompressedColorSpace as C};
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

fn discover_extensions(raw: &WebGl2RenderingContext) -> Result<GlExtensionSet, GlError> {
    let listed = raw.get_supported_extensions().ok_or_else(|| {
        driver(
            "getSupportedExtensions",
            "browser returned null for a live WebGL2 context",
        )
    })?;
    let mut extensions = GlExtensionSet::default();
    for value in listed.iter() {
        let name = value.as_string().ok_or_else(|| {
            driver(
                "getSupportedExtensions",
                "browser returned a non-string extension name",
            )
        })?;
        extensions.report_raw(name);
    }

    for known in [
        GlKnownExtension::ExtDisjointTimerQueryWebgl2,
        GlKnownExtension::ExtColorBufferFloat,
        GlKnownExtension::ExtFloatBlend,
        GlKnownExtension::OesTextureFloatLinear,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::WebglMultiDraw,
        GlKnownExtension::OvrMultiview2,
        GlKnownExtension::KhrParallelShaderCompile,
        GlKnownExtension::CompressedTextureS3tc,
        GlKnownExtension::CompressedTextureS3tcSrgb,
        GlKnownExtension::CompressedTextureBptc,
        GlKnownExtension::CompressedTextureRgtc,
        GlKnownExtension::CompressedTextureAstc,
        GlKnownExtension::CompressedTextureEtc,
    ] {
        if extensions.provenance(known).is_some() {
            match raw.get_extension(acquisition_name(known, &extensions)) {
                Ok(Some(_)) => {
                    extensions.acquire(known);
                }
                Ok(None) | Err(_) => {
                    // A reported name without a usable extension object is
                    // explicitly failed and can never enable a capability.
                    extensions.fail(known);
                }
            }
        }
    }
    Ok(extensions)
}

fn acquisition_name(known: GlKnownExtension, extensions: &GlExtensionSet) -> &'static str {
    if known == GlKnownExtension::ExtTextureFilterAnisotropic {
        for alias in [
            "EXT_texture_filter_anisotropic",
            "WEBKIT_EXT_texture_filter_anisotropic",
            "MOZ_EXT_texture_filter_anisotropic",
        ] {
            if extensions
                .raw_reported_names()
                .any(|reported| reported == alias)
            {
                return alias;
            }
        }
    }
    known.raw_name()
}

fn discover_limits(
    raw: &WebGl2RenderingContext,
    extensions: &GlExtensionSet,
) -> Result<GlLimits, GlError> {
    let max_samples = u32_parameter(raw, WebGl2RenderingContext::MAX_SAMPLES, "MAX_SAMPLES")?;
    Ok(GlLimits {
        max_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_TEXTURE_SIZE,
            "MAX_TEXTURE_SIZE",
        )?,
        max_3d_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_3D_TEXTURE_SIZE,
            "MAX_3D_TEXTURE_SIZE",
        )?,
        max_array_texture_layers: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_ARRAY_TEXTURE_LAYERS,
            "MAX_ARRAY_TEXTURE_LAYERS",
        )?,
        max_cube_map_texture_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_CUBE_MAP_TEXTURE_SIZE,
            "MAX_CUBE_MAP_TEXTURE_SIZE",
        )?,
        max_renderbuffer_size: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_RENDERBUFFER_SIZE,
            "MAX_RENDERBUFFER_SIZE",
        )?,
        max_color_attachments: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COLOR_ATTACHMENTS,
            "MAX_COLOR_ATTACHMENTS",
        )?,
        max_draw_buffers: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_DRAW_BUFFERS,
            "MAX_DRAW_BUFFERS",
        )?,
        max_vertex_attributes: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_ATTRIBS,
            "MAX_VERTEX_ATTRIBS",
        )?,
        max_viewport_dimensions: viewport_dimensions(raw)?,
        max_viewports: 0,
        max_vertex_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
            "MAX_VERTEX_TEXTURE_IMAGE_UNITS",
        )?,
        max_fragment_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_TEXTURE_IMAGE_UNITS,
            "MAX_TEXTURE_IMAGE_UNITS",
        )?,
        max_combined_texture_image_units: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            "MAX_COMBINED_TEXTURE_IMAGE_UNITS",
        )?,
        max_uniform_buffer_bindings: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_UNIFORM_BUFFER_BINDINGS,
            "MAX_UNIFORM_BUFFER_BINDINGS",
        )?,
        max_uniform_block_size: u64_parameter(
            raw,
            WebGl2RenderingContext::MAX_UNIFORM_BLOCK_SIZE,
            "MAX_UNIFORM_BLOCK_SIZE",
        )?,
        uniform_buffer_offset_alignment: u64_parameter(
            raw,
            WebGl2RenderingContext::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
            "UNIFORM_BUFFER_OFFSET_ALIGNMENT",
        )?,
        max_vertex_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_VERTEX_UNIFORM_BLOCKS,
            "MAX_VERTEX_UNIFORM_BLOCKS",
        )?,
        max_fragment_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_FRAGMENT_UNIFORM_BLOCKS,
            "MAX_FRAGMENT_UNIFORM_BLOCKS",
        )?,
        max_compute_uniform_blocks: 0,
        max_combined_uniform_blocks: u32_parameter(
            raw,
            WebGl2RenderingContext::MAX_COMBINED_UNIFORM_BLOCKS,
            "MAX_COMBINED_UNIFORM_BLOCKS",
        )?,
        max_storage_buffer_bindings: 0,
        max_storage_block_size: 0,
        storage_buffer_offset_alignment: 0,
        max_vertex_storage_blocks: 0,
        max_fragment_storage_blocks: 0,
        max_compute_storage_blocks: 0,
        max_combined_storage_blocks: 0,
        max_image_units: 0,
        max_combined_image_units: 0,
        max_samples,
        // WebGL2 has renderbuffer multisampling but no multisample textures.
        // Do not turn MAX_SAMPLES into a texture capability.
        max_color_texture_samples: 0,
        max_depth_texture_samples: 0,
        max_integer_samples: 0,
        max_compute_work_group_count: [0; 3],
        max_compute_work_group_size: [0; 3],
        max_compute_work_group_invocations: 0,
        max_multi_draw_indirect_count: None,
        query_counter_bits: 0,
        max_texture_anisotropy: anisotropy_limit(raw, extensions)?,
    })
}

fn anisotropy_limit(
    raw: &WebGl2RenderingContext,
    extensions: &GlExtensionSet,
) -> Result<Option<GlFiniteF32>, GlError> {
    // `MAX_TEXTURE_MAX_ANISOTROPY_EXT` has this registry value for all three
    // spellings. It is invalid until an extension object has been acquired.
    const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
    if !extensions.is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic) {
        return Ok(None);
    }
    let value = number_parameter(
        raw,
        MAX_TEXTURE_MAX_ANISOTROPY_EXT,
        "MAX_TEXTURE_MAX_ANISOTROPY_EXT",
    )? as f32;
    GlFiniteF32::new(value)
        .ok_or_else(|| {
            driver(
                "getParameter",
                "MAX_TEXTURE_MAX_ANISOTROPY_EXT was not finite f32",
            )
        })
        .map(Some)
}

fn webgl2_baseline_formats(extensions: &GlExtensionSet) -> Result<GlFormatTable, GlError> {
    let mut formats = GlFormatTable::default();
    for facts in [
        GlFormatCapabilities {
            format: GlFormat::Rgba8Unorm,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: true,
            storage_read: false,
            storage_write: false,
            copy_source: true,
            copy_destination: true,
        },
        GlFormatCapabilities {
            format: GlFormat::Rgba8Srgb,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: true,
            storage_read: false,
            storage_write: false,
            copy_source: true,
            copy_destination: true,
        },
        GlFormatCapabilities {
            format: GlFormat::Depth32Float,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: true,
            filterable: false,
            renderable: true,
            blendable: false,
            storage_read: false,
            storage_write: false,
            // Do not claim a concrete copy operation from a static baseline.
            copy_source: false,
            copy_destination: false,
        },
    ] {
        formats
            .record(facts)
            .map_err(|error| driver("record WebGL2 baseline format", &format!("{error:?}")))?;
    }
    for (extension, exact_formats) in compressed_extension_formats() {
        if !extensions.is_acquired(extension) {
            continue;
        }
        for format in exact_formats {
            formats
                .record(GlFormatCapabilities {
                    format,
                    resource_kind: GlFormatResourceKind::Texture,
                    sample_count: 1,
                    evidence: GlFormatEvidence::ExtensionAcquired(extension),
                    sampled: true,
                    filterable: true,
                    renderable: false,
                    blendable: false,
                    storage_read: false,
                    storage_write: false,
                    copy_source: false,
                    copy_destination: false,
                })
                .map_err(|error| {
                    driver(
                        "record acquired compressed WebGL2 format",
                        &format!("{error:?}"),
                    )
                })?;
        }
    }
    Ok(formats)
}

fn compressed_extension_formats() -> Vec<(GlKnownExtension, Vec<GlFormat>)> {
    use super::{GlAstcBlock as B, GlCompressedColorSpace as C};
    vec![
        (
            GlKnownExtension::CompressedTextureS3tc,
            vec![
                GlFormat::Bc1RgbUnorm,
                GlFormat::Bc1RgbaUnorm,
                GlFormat::Bc2RgbaUnorm,
                GlFormat::Bc3RgbaUnorm,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureS3tcSrgb,
            vec![
                GlFormat::Bc1RgbSrgb,
                GlFormat::Bc1RgbaSrgb,
                GlFormat::Bc2RgbaSrgb,
                GlFormat::Bc3RgbaSrgb,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureRgtc,
            vec![
                GlFormat::Bc4RUnorm,
                GlFormat::Bc4RSnorm,
                GlFormat::Bc5RgUnorm,
                GlFormat::Bc5RgSnorm,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureBptc,
            vec![
                GlFormat::Bc6hRgbUfloat,
                GlFormat::Bc6hRgbSfloat,
                GlFormat::Bc7RgbaUnorm,
                GlFormat::Bc7RgbaSrgb,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureEtc,
            vec![
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
            ],
        ),
        (
            GlKnownExtension::CompressedTextureAstc,
            [
                B::B4x4,
                B::B5x4,
                B::B5x5,
                B::B6x5,
                B::B6x6,
                B::B8x5,
                B::B8x6,
                B::B8x8,
                B::B10x5,
                B::B10x6,
                B::B10x8,
                B::B10x10,
                B::B12x10,
                B::B12x12,
            ]
            .into_iter()
            .flat_map(|block| {
                [
                    GlFormat::Astc {
                        block,
                        color_space: C::Linear,
                    },
                    GlFormat::Astc {
                        block,
                        color_space: C::Srgb,
                    },
                ]
            })
            .collect(),
        ),
    ]
}

fn string_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<String, GlError> {
    raw.get_parameter(pname)
        .map_err(|value| js_error("getParameter", value))?
        .as_string()
        .ok_or_else(|| driver("getParameter", &format!("{name} was not a string")))
}

fn u32_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<u32, GlError> {
    let value = number_parameter(raw, pname, name)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
        return Err(driver("getParameter", &format!("{name} was not a u32")));
    }
    Ok(value as u32)
}

fn u64_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<u64, GlError> {
    let value = number_parameter(raw, pname, name)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > 9_007_199_254_740_991.0
    {
        return Err(driver(
            "getParameter",
            &format!("{name} was not an exact JavaScript integer"),
        ));
    }
    Ok(value as u64)
}

fn number_parameter(
    raw: &WebGl2RenderingContext,
    pname: u32,
    name: &'static str,
) -> Result<f64, GlError> {
    raw.get_parameter(pname)
        .map_err(|value| js_error("getParameter", value))?
        .as_f64()
        .ok_or_else(|| driver("getParameter", &format!("{name} was not a number")))
}

fn viewport_dimensions(raw: &WebGl2RenderingContext) -> Result<[u32; 2], GlError> {
    let value = raw
        .get_parameter(WebGl2RenderingContext::MAX_VIEWPORT_DIMS)
        .map_err(|value| js_error("getParameter", value))?;
    let values = Array::from(&value);
    if values.length() != 2 {
        return Err(driver(
            "getParameter",
            "MAX_VIEWPORT_DIMS did not contain two values",
        ));
    }
    let width = values.get(0).as_f64();
    let height = values.get(1).as_f64();
    match (width, height) {
        (Some(width), Some(height))
            if width.is_finite()
                && height.is_finite()
                && width.fract() == 0.0
                && height.fract() == 0.0
                && width >= 0.0
                && height >= 0.0
                && width <= f64::from(u32::MAX)
                && height <= f64::from(u32::MAX) =>
        {
            Ok([width as u32, height as u32])
        }
        _ => Err(driver(
            "getParameter",
            "MAX_VIEWPORT_DIMS was not a u32 pair",
        )),
    }
}

fn browser_identity() -> Result<String, GlError> {
    let global = js_sys::global();
    let navigator = Reflect::get(&global, &JsValue::from_str("navigator"))
        .map_err(|value| js_error("navigator", value))?;
    let user_agent = Reflect::get(&navigator, &JsValue::from_str("userAgent"))
        .map_err(|value| js_error("navigator.userAgent", value))?;
    user_agent.as_string().ok_or_else(|| {
        driver(
            "navigator.userAgent",
            "browser returned a non-string identity",
        )
    })
}

fn require_webgl2_version(version: &str) -> Result<(), GlError> {
    version
        .strip_prefix("WebGL ")
        .is_some_and(|remainder| remainder.starts_with('2'))
        .then_some(())
        .ok_or_else(|| driver("getParameter(VERSION)", "context did not report WebGL 2"))
}

fn discovery_error(error: super::GlDiscoveryError) -> GlError {
    driver("build WebGL2 discovery", &format!("{error:?}"))
}

fn js_error(operation: &'static str, value: JsValue) -> GlError {
    driver(operation, &format!("browser exception: {value:?}"))
}

fn driver(operation: &'static str, message: &str) -> GlError {
    GlError::Driver {
        operation,
        message: message.to_owned(),
    }
}
