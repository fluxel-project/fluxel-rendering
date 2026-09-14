//! Browser execution domain over the RHI-owned WebGL2 context record.

use web_sys::WebGl2RenderingContext;

use super::super::{
    BufferId, ContextStamp, GlBufferDesc, GlContextLifecycle, GlError, GlFamilyApi,
    GlPixelStoreState, GlRenderBufferDesc, GlResourceApi, GlSamplerApi, GlSamplerDesc,
    GlTextureDesc, GlTextureDimension, OwnerThreadIdentity, RenderbufferId, SamplerId, TextureId,
};
use super::discovery::{BrowserBuffer, BrowserSampler, BrowserTexture, WebGl2BrowserDiscovery};

/// Allocation usage applied to every WebGL2 buffer (audit P2-11).
///
/// WebGL2 offers no persistent mapping and no explicit usage negotiation, so
/// the policy is one explicit, auditable constant: `DYNAMIC_DRAW` matches the
/// common RHI's re-upload-per-frame residency model and keeps driver placement
/// friendly to `bufferSubData` streaming. A `STATIC_DRAW` fast path for
/// never-resident data may only be introduced after profiling attributes a
/// benefit (plan "Private: upload-ring/orphaning strategy").
pub(super) const BUFFER_ALLOCATION_USAGE: u32 = WebGl2RenderingContext::DYNAMIC_DRAW;

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

    fn discovery(&self) -> &super::super::GlDiscoverySnapshot {
        &self.snapshot
    }

    fn context_lost(&mut self) -> Result<(), GlError> {
        self.assert_owner_thread("context-lost")?;
        self.lifecycle.set(GlContextLifecycle::Lost);
        // Loss invalidates every browser handle: drop all records without
        // calling methods on the dead JS objects.
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        self.clear_executable_state();
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
        self.clear_executable_state();
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
            BUFFER_ALLOCATION_USAGE,
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
        let internal_format =
            super::format_map::internal_format(desc.format).ok_or(GlError::Unsupported {
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
        super::format_map::configure_sampler(&self.raw, &raw, desc);
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
