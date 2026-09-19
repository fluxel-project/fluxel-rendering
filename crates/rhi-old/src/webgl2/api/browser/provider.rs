//! Browser execution domain over the RHI-owned WebGL2 context record.

use web_sys::WebGl2RenderingContext;

use super::super::{
    BufferId, ContextStamp, GlBufferDesc, GlBufferUsage, GlContextLifecycle, GlError, GlFamilyApi,
    GlPixelStoreState, GlRenderBufferDesc, GlResourceApi, GlSamplerApi, GlSamplerDesc,
    GlTextureDesc, GlTextureDimension, OwnerThreadIdentity, RenderbufferId, SamplerId, TextureId,
};
use super::discovery::{BrowserBuffer, BrowserSampler, BrowserTexture, WebGl2BrowserDiscovery};
use super::objects::BrowserRenderbuffer;

/// Allocation usage applied to every WebGL2 buffer (audit P2-11).
///
/// WebGL2 offers no persistent mapping and no explicit usage negotiation, so
/// the policy is one explicit, auditable constant: `DYNAMIC_DRAW` matches the
/// common RHI's re-upload-per-frame residency model and keeps driver placement
/// friendly to `bufferSubData` streaming. A `STATIC_DRAW` fast path for
/// never-resident data may only be introduced after profiling attributes a
/// benefit (plan "Private: upload-ring/orphaning strategy").
pub(super) const BUFFER_ALLOCATION_USAGE: u32 = WebGl2RenderingContext::DYNAMIC_DRAW;

/// Whether a usage set asks one buffer to be both an index buffer and a buffer
/// whose binding point an index buffer can never reach.
///
/// A WebGL2 buffer's target is fixed by its *first* bind and cannot be changed
/// afterwards: binding the buffer to a different target is refused, and
/// releasing the first binding does not undo it.  Measured on the real adapter
/// (AMD Radeon 780M through ANGLE/D3D11; the table is in the 0.15 series plan's
/// 2026-09-17 browser entry):
///
/// - A buffer first bound to `ELEMENT_ARRAY_BUFFER` is refused `ARRAY_BUFFER`
///   and `UNIFORM_BUFFER`, through `bindBuffer` and `bindBufferBase` alike.
/// - That same buffer still reaches `COPY_READ_BUFFER` and `COPY_WRITE_BUFFER`,
///   which is what lets the corrected allocation keep using the existing
///   transfer path instead of needing one of its own.
/// - A buffer first bound anywhere else reaches every target except
///   `ELEMENT_ARRAY_BUFFER`, so only the index role has to choose its target.
///
/// So index combines with the copy roles and with nothing else, and the roles
/// it cannot combine with are refused before a driver object exists rather than
/// at whichever bind happens to come second.
const fn index_role_cannot_share_its_buffer(usage: GlBufferUsage) -> bool {
    usage.contains(GlBufferUsage::INDEX)
        && (usage.contains(GlBufferUsage::VERTEX)
            || usage.contains(GlBufferUsage::UNIFORM)
            || usage.contains(GlBufferUsage::STORAGE)
            || usage.contains(GlBufferUsage::INDIRECT))
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
        self.renderbuffers.clear();
        // The retained batch and timer commands belong to the dead context.
        self.multi_draw = None;
        self.timer = None;
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
        // The recreated context re-acquired its own extension objects, so the
        // retained batch and timer commands are the replacement's, never the
        // old ones, and the replacement may have acquired a different set.
        self.multi_draw = replacement.multi_draw;
        self.timer = replacement.timer;
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        self.renderbuffers.clear();
        self.clear_executable_state();
        self.pixel_store = GlPixelStoreState::DEFAULT;
        Ok(stamp)
    }
}

impl GlResourceApi for WebGl2BrowserDiscovery {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        const OP: &str = "create-buffer";
        self.assert_provider_ready(OP)?;
        desc.validate().map_err(|error| GlError::Validation {
            operation: OP,
            message: error.message(),
        })?;
        if desc.size > 9_007_199_254_740_991 {
            return Err(Self::validation(
                OP,
                "buffer size exceeds exact browser integer range",
            ));
        }
        if index_role_cannot_share_its_buffer(desc.usage) {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "an index buffer cannot also serve as a vertex, uniform, storage, or \
                         indirect buffer: WebGL2 fixes a buffer's binding target at its first \
                         bind, and a buffer bound as an index buffer can never reach those \
                         binding points. Create a second buffer for the other role.",
            });
        }
        let raw = self
            .raw
            .create_buffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        // The target is fixed by this first bind and can never change, so the
        // role picks it -- once, here (`index_role_cannot_share_its_buffer`
        // documents which combinations that leaves representable).
        let is_index = desc.usage.contains(GlBufferUsage::INDEX);
        let target = if is_index {
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER
        } else {
            // ARRAY_BUFFER is Layer 1-private scratch for every other role:
            // bound immediately before the allocation, not restored on return
            // (`GlCopyDomainApi` documents why).
            WebGl2RenderingContext::ARRAY_BUFFER
        };
        // ELEMENT_ARRAY_BUFFER is vertex-array state, and the bind is what the
        // bound array stores: allocating an index buffer under a live vertex
        // array would overwrite that array's index binding, and rebinding the
        // array afterwards would not put it back.  Allocating against the
        // default array leaves every live array untouched.
        let rebound = if is_index {
            self.bound_vertex_array
        } else {
            None
        };
        if rebound.is_some() {
            self.raw.bind_vertex_array(None);
        }
        self.raw.bind_buffer(target, Some(&raw));
        self.raw
            .buffer_data_with_f64(target, desc.size as f64, BUFFER_ALLOCATION_USAGE);
        let allocation = self.driver_error(OP);
        if let Some(vertex_array) = rebound {
            // The driver holds the default array now, so putting the previous
            // one back is a plain rebind, and the mirror already says it is
            // bound -- it is only corrected if the array left the table.
            let live = self
                .vertex_arrays
                .get(&vertex_array.slot)
                .filter(|entry| entry.generation == vertex_array.generation)
                .map(|entry| entry.raw.clone());
            match live {
                Some(array) => self.raw.bind_vertex_array(Some(&array)),
                None => self.bound_vertex_array = None,
            }
        }
        if let Err(error) = allocation {
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
        desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, GlError> {
        const OP: &str = "create-render-buffer";
        self.assert_provider_ready(OP)?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid renderbuffer descriptor"))?;
        // Admission reads recorded facts only, so an allocation this context has
        // no evidence for is rejected here instead of being attempted and hoped
        // for; the same rule is what the recorded sample-count ceiling came from.
        let internal = super::renderbuffer_facts::admit(
            &self.snapshot.limits(),
            self.snapshot.formats(),
            desc,
        )
        .map_err(|rejection| rejection.error(OP))?;
        let width = i32::try_from(desc.width)
            .map_err(|_| Self::validation(OP, "width exceeds the browser integer range"))?;
        let height = i32::try_from(desc.height)
            .map_err(|_| Self::validation(OP, "height exceeds the browser integer range"))?;
        let raw = self
            .raw
            .create_renderbuffer()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        // RENDERBUFFER is Layer 1-private scratch: bound immediately before the
        // allocation it describes, exactly like the buffer and texture paths.
        self.raw
            .bind_renderbuffer(WebGl2RenderingContext::RENDERBUFFER, Some(&raw));
        if desc.samples > 1 {
            self.raw.renderbuffer_storage_multisample(
                WebGl2RenderingContext::RENDERBUFFER,
                i32::try_from(desc.samples).map_err(|_| {
                    Self::validation(OP, "sample count exceeds the browser integer range")
                })?,
                internal,
                width,
                height,
            );
        } else {
            self.raw.renderbuffer_storage(
                WebGl2RenderingContext::RENDERBUFFER,
                internal,
                width,
                height,
            );
        }
        if let Err(error) = self.driver_error(OP) {
            self.raw.delete_renderbuffer(Some(&raw));
            return Err(error);
        }
        let slot = Self::allocate_slot(&mut self.next_renderbuffer_slot, OP)?;
        let id = RenderbufferId::new(self.context_stamp(), slot, 0);
        self.renderbuffers.insert(
            slot,
            BrowserRenderbuffer {
                generation: id.generation,
                raw,
                desc,
            },
        );
        Ok(id)
    }

    fn destroy_render_buffer(&mut self, id: RenderbufferId) -> Result<(), GlError> {
        const OP: &str = "destroy-render-buffer";
        self.renderbuffer(OP, id)?;
        let entry = self
            .renderbuffers
            .remove(&id.slot)
            .ok_or_else(|| Self::validation(OP, "renderbuffer allocation disappeared"))?;
        self.raw.delete_renderbuffer(Some(&entry.raw));
        self.driver_error(OP)
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
