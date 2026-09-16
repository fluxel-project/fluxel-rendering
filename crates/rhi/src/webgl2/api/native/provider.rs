//! Native executable owner for every GL-family command domain.
//!
//! The Host owns the platform context; this type borrows its already-current
//! `glow` dispatch table and owns only Fluxel object tables, pass/raster
//! records, and the pixel-store snapshot. Raw GL names live inside these
//! records and never participate in identity comparisons.

use std::collections::BTreeMap;

use super::super::GlFamilyApi as _;
use super::super::{
    BufferId, ContextStamp, FramebufferId, GlBufferDesc, GlContextLifecycle, GlDiscoverySnapshot,
    GlError, GlFenceLeaseBook, GlIndexBinding, GlPixelStoreState, GlPrimitiveTopology,
    GlProgramDescriptor, GlRenderBufferDesc, GlSurfaceLeaseBook, GlTextureDesc, GlVertexLayout,
    OwnerThreadIdentity, ProgramId, QueryId, RenderbufferId, SamplerId, ShaderId, SyncId,
    TextureId, VertexArrayId,
};
use super::discovery::{NativeDiscoveryError, discover_current_glow};

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) struct NativeGlProvider<'a> {
    pub(super) gl: &'a glow::Context,
    pub(super) discovery: GlDiscoverySnapshot,
    pub(super) lifecycle: GlContextLifecycle,
    pub(super) owner: OwnerThreadIdentity,
    pub(super) next_slot: u32,
    pub(super) buffers: BTreeMap<BufferId, (glow::NativeBuffer, GlBufferDesc)>,
    pub(super) textures: BTreeMap<TextureId, (glow::NativeTexture, GlTextureDesc)>,
    pub(super) renderbuffers:
        BTreeMap<RenderbufferId, (glow::NativeRenderbuffer, GlRenderBufferDesc)>,
    pub(super) samplers: BTreeMap<SamplerId, glow::NativeSampler>,
    pub(super) shaders: BTreeMap<ShaderId, glow::NativeShader>,
    pub(super) programs: BTreeMap<ProgramId, NativeProgram>,
    pub(super) vertex_arrays: BTreeMap<VertexArrayId, NativeVertexArray>,
    pub(super) framebuffers: BTreeMap<FramebufferId, NativeFramebuffer>,
    pub(super) queries: BTreeMap<QueryId, NativeQuery>,
    pub(super) syncs: BTreeMap<SyncId, glow::NativeFence>,
    pub(super) fences: GlFenceLeaseBook,
    pub(super) surface: GlSurfaceLeaseBook,
    pub(super) surface_suspended: bool,
    pub(super) pass: Option<ActivePass>,
    pub(super) raster: Option<ActiveRaster>,
    /// The query currently recording a measurement, if any.
    pub(super) active_query: Option<QueryId>,
    /// The compute program installed for dispatch work, if any.
    pub(super) active_compute_program: Option<ProgramId>,
    pub(super) pixel_store: GlPixelStoreState,
}

/// A linked raster or compute program record with its validated descriptor.
pub(super) struct NativeProgram {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeProgram,
    pub(super) descriptor: GlProgramDescriptor,
}

/// A created VAO with its structural layout and last recorded index binding.
pub(super) struct NativeVertexArray {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeVertexArray,
    pub(super) layout: GlVertexLayout,
    pub(super) index: Option<GlIndexBinding>,
}

/// A created framebuffer with its validated descriptor.
pub(super) struct NativeFramebuffer {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeFramebuffer,
    pub(super) descriptor: super::super::GlFramebufferDescriptor,
}

/// The target a query object last recorded a measurement for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum QueryTarget {
    SamplesPassed,
    TimeElapsed,
    Timestamp,
}

pub(super) struct NativeQuery {
    pub(super) generation: u32,
    pub(super) raw: glow::NativeQuery,
    pub(super) target: Option<QueryTarget>,
}

/// Facts of the render pass currently recording on this context.
pub(super) struct ActivePass {
    pub(super) framebuffer: FramebufferId,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: u32,
    /// Per color attachment; `true` when `end_render_pass` must invalidate.
    pub(super) discard_color: Vec<bool>,
    pub(super) discard_depth_stencil: Option<bool>,
}

/// The raster pipeline installed for the active pass.
pub(super) struct ActiveRaster {
    pub(super) vertex_array: VertexArrayId,
    pub(super) topology: GlPrimitiveTopology,
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
        Ok(Self::assemble(gl, discovery))
    }

    /// Assembles the provider over an already-collected discovery snapshot.
    ///
    /// The caller must guarantee `gl` is the exact context that produced
    /// `discovery` and that it is current on the calling thread.
    ///
    /// # Safety
    ///
    /// Same as [`Self::from_current`].
    pub(crate) unsafe fn from_discovered(
        gl: &'a glow::Context,
        discovery: GlDiscoverySnapshot,
    ) -> Self {
        Self::assemble(gl, discovery)
    }

    fn assemble(gl: &'a glow::Context, discovery: GlDiscoverySnapshot) -> Self {
        Self {
            gl,
            discovery,
            lifecycle: GlContextLifecycle::Active,
            owner: OwnerThreadIdentity::current(),
            next_slot: 0,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            renderbuffers: BTreeMap::new(),
            samplers: BTreeMap::new(),
            shaders: BTreeMap::new(),
            programs: BTreeMap::new(),
            vertex_arrays: BTreeMap::new(),
            framebuffers: BTreeMap::new(),
            queries: BTreeMap::new(),
            syncs: BTreeMap::new(),
            fences: GlFenceLeaseBook::default(),
            surface: GlSurfaceLeaseBook::new(),
            surface_suspended: false,
            pass: None,
            raster: None,
            active_query: None,
            active_compute_program: None,
            pixel_store: GlPixelStoreState::DEFAULT,
        }
    }

    /// Allocates one monotonically increasing slot.
    ///
    /// Slots are never reused within one context generation, so the `0`
    /// object generation stays sound; any future slot-reuse design must
    /// introduce per-slot generations first.
    pub(super) fn slot(&mut self, operation: &'static str) -> Result<u32, GlError> {
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .ok_or(GlError::OutOfMemory { operation })?;
        Ok(slot)
    }

    pub(super) fn validation(operation: &'static str, message: &'static str) -> GlError {
        GlError::Validation {
            operation,
            message: message.into(),
        }
    }

    pub(super) fn driver_error(&self, operation: &'static str) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: upheld by NativeGlProvider::from_current.
        let error = unsafe { self.gl.get_error() };
        (error == glow::NO_ERROR)
            .then_some(())
            .ok_or_else(|| GlError::Driver {
                operation,
                message: format!("GL error 0x{error:04x}"),
            })
    }

    /// Shared framebuffer-completeness observation for copy and pass work.
    pub(super) fn require_complete(&self, operation: &'static str) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract.
        let status = unsafe { self.gl.check_framebuffer_status(glow::FRAMEBUFFER) };
        if status == glow::FRAMEBUFFER_COMPLETE {
            Ok(())
        } else {
            Err(GlError::IncompleteFramebuffer { operation, status })
        }
    }

    pub(super) fn buffer(
        &self,
        operation: &'static str,
        id: BufferId,
    ) -> Result<(glow::NativeBuffer, GlBufferDesc), GlError> {
        self.validate_object_context(operation, id.context)?;
        self.buffers
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "buffer is not live"))
    }

    pub(super) fn texture(
        &self,
        operation: &'static str,
        id: TextureId,
    ) -> Result<(glow::NativeTexture, GlTextureDesc), GlError> {
        self.validate_object_context(operation, id.context)?;
        self.textures
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "texture is not live"))
    }

    pub(super) fn renderbuffer(
        &self,
        operation: &'static str,
        id: RenderbufferId,
    ) -> Result<(glow::NativeRenderbuffer, GlRenderBufferDesc), GlError> {
        self.validate_object_context(operation, id.context)?;
        self.renderbuffers
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "renderbuffer is not live"))
    }

    pub(super) fn sampler(
        &self,
        operation: &'static str,
        id: SamplerId,
    ) -> Result<glow::NativeSampler, GlError> {
        self.validate_object_context(operation, id.context)?;
        // Map keys are full identities, so a hit implies the same generation.
        self.samplers
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "sampler is not live"))
    }

    pub(super) fn shader(
        &self,
        operation: &'static str,
        id: ShaderId,
    ) -> Result<glow::NativeShader, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.shaders
            .get(&id)
            .copied()
            .ok_or_else(|| Self::validation(operation, "shader is not live"))
    }

    pub(super) fn program(
        &self,
        operation: &'static str,
        id: ProgramId,
    ) -> Result<&NativeProgram, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.programs
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "program is not live"))
    }

    pub(super) fn vertex_array(
        &self,
        operation: &'static str,
        id: VertexArrayId,
    ) -> Result<&NativeVertexArray, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.vertex_arrays
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "vertex array is not live"))
    }

    pub(super) fn framebuffer(
        &self,
        operation: &'static str,
        id: FramebufferId,
    ) -> Result<&NativeFramebuffer, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.framebuffers
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "framebuffer is not live"))
    }

    pub(super) fn query(
        &self,
        operation: &'static str,
        id: QueryId,
    ) -> Result<&NativeQuery, GlError> {
        self.validate_object_context(operation, id.context)?;
        self.queries
            .get(&id)
            .filter(|entry| entry.generation == id.generation)
            .ok_or_else(|| Self::validation(operation, "query is not live"))
    }

    /// Clears every executable table. Context loss makes every borrowed GL
    /// name invalid; only Fluxel-owned state is reset, and no driver call is
    /// made against a possibly-dead object.
    pub(super) fn reset_executable_state(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.renderbuffers.clear();
        self.samplers.clear();
        self.shaders.clear();
        self.programs.clear();
        self.vertex_arrays.clear();
        self.framebuffers.clear();
        self.queries.clear();
        self.syncs.clear();
        self.fences.revoke_all();
        self.pass = None;
        self.raster = None;
        self.active_query = None;
        self.active_compute_program = None;
        let _ = self.surface.invalidate_generation();
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::super::GlFamilyApi for NativeGlProvider<'_> {
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }
    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }
    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        (actual == self.owner)
            .then_some(())
            .ok_or(GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.discovery
    }
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.assert_ready("context-lost")?;
        self.lifecycle = GlContextLifecycle::Lost;
        self.reset_executable_state();
        Ok(())
    }
    /// Completes restoration over the replacement context.
    ///
    /// The Host must have created a new native context and made it current on
    /// the owner thread before calling. WGL and EGL entry points dispatch to
    /// the *current* context, so the borrowed `glow` table remains valid for
    /// the replacement context and rediscovery observes the new generation
    /// (audit P1-9). Epoch strictly increases and every object table, lease
    /// book, and derived record is invalidated before `Active` is restored.
    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        self.assert_owner_thread("context-restored")?;
        if self.lifecycle != GlContextLifecycle::Lost {
            return Err(Self::validation("context-restored", "context is not lost"));
        }
        let stamp = self.discovery.context_stamp();
        let epoch = stamp.epoch.checked_next().ok_or_else(|| GlError::Driver {
            operation: "context-restored",
            message: "context epoch exhausted".into(),
        })?;
        let new_stamp = ContextStamp::new(stamp.device, epoch);
        // SAFETY: the caller contract guarantees the replacement context is
        // current on this thread for the whole rediscovery call.
        let discovery = unsafe { discover_current_glow(self.gl, new_stamp) }.map_err(|error| {
            GlError::Driver {
                operation: "context-restored",
                message: format!("native GL rediscovery failed: {error:?}"),
            }
        })?;
        self.discovery = discovery;
        self.lifecycle = GlContextLifecycle::Active;
        self.reset_executable_state();
        self.pixel_store = GlPixelStoreState::DEFAULT;
        Ok(new_stamp)
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl<'a> NativeGlProvider<'a> {
    /// Restores over a freshly created `glow` table after context loss.
    ///
    /// Use this when the Host rebuilt the context through a new loader
    /// instance; the borrowed reference type stays `'a`, so the Host must
    /// keep the new `glow` context alive for the provider's lifetime.
    ///
    /// # Safety
    ///
    /// `gl` must be current on the owner thread and must be the context that
    /// will serve every later provider call.
    pub(crate) unsafe fn restore_with_current(&mut self, gl: &'a glow::Context) {
        self.gl = gl;
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::super::GlResourceApi for NativeGlProvider<'_> {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-buffer")?;
        desc.validate()
            .map_err(|_| Self::validation("create-buffer", "invalid buffer descriptor"))?;
        let size = i32::try_from(desc.size)
            .map_err(|_| Self::validation("create-buffer", "buffer exceeds GLsizei"))?;
        // SAFETY: current-context contract; all validation completed before GL mutation.
        let name = unsafe { self.gl.create_buffer() }.map_err(|message| GlError::Driver {
            operation: "create-buffer",
            message,
        })?;
        // SAFETY: see above.
        unsafe {
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(name));
            self.gl
                .buffer_data_size(glow::COPY_WRITE_BUFFER, size, BUFFER_ALLOCATION_USAGE);
        }
        if let Err(error) = self.driver_error("create-buffer") {
            unsafe { self.gl.delete_buffer(name) };
            return Err(error);
        }
        let id = BufferId::new(self.context_stamp(), self.slot("create-buffer")?, 0);
        self.buffers.insert(id, (name, desc));
        Ok(id)
    }
    fn create_texture_resource(&mut self, desc: GlTextureDesc) -> Result<TextureId, GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-texture")?;
        desc.validate()
            .map_err(|_| Self::validation("create-texture", "invalid texture descriptor"))?;
        let internal = native_texture_format(desc.format).ok_or(GlError::Unsupported {
            operation: "create-texture",
            reason: "format has no proven native texture storage mapping",
        })?;
        if desc.dimension != super::super::GlTextureDimension::D2 || desc.sample_count != 1 {
            return Err(GlError::Unsupported {
                operation: "create-texture",
                reason: "multisample texture allocation stays with renderbuffer storage",
            });
        }
        if self
            .discovery
            .formats()
            .get_for(super::super::GlFormatResourceKind::Texture, desc.format, 1)
            .is_none()
        {
            return Err(GlError::Unsupported {
                operation: "create-texture",
                reason: "format lacks discovery evidence",
            });
        }
        let width = i32::try_from(desc.extent.width)
            .map_err(|_| Self::validation("create-texture", "width exceeds GLsizei"))?;
        let height = i32::try_from(desc.extent.height)
            .map_err(|_| Self::validation("create-texture", "height exceeds GLsizei"))?;
        let levels = i32::try_from(desc.mip_level_count)
            .map_err(|_| Self::validation("create-texture", "mip count exceeds GLsizei"))?;
        // SAFETY: current-context contract; all profile/format/size validation preceded mutation.
        let name = unsafe { self.gl.create_texture() }.map_err(|message| GlError::Driver {
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
        let id = TextureId::new(self.context_stamp(), self.slot("create-texture")?, 0);
        self.textures.insert(id, (name, desc));
        Ok(id)
    }
    fn create_render_buffer(
        &mut self,
        desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-render-buffer";
        self.assert_ready(OP)?;
        desc.validate()
            .map_err(|_| Self::validation(OP, "invalid renderbuffer descriptor"))?;
        let limits = self.discovery.limits();
        if desc.width > limits.max_renderbuffer_size || desc.height > limits.max_renderbuffer_size {
            return Err(Self::validation(
                OP,
                "renderbuffer extent exceeds the discovered limit",
            ));
        }
        if desc.samples > limits.max_samples {
            return Err(Self::validation(
                OP,
                "renderbuffer sample count exceeds the discovered limit",
            ));
        }
        let facts = self
            .discovery
            .formats()
            .get_for(
                super::super::GlFormatResourceKind::Renderbuffer,
                desc.format,
                desc.samples,
            )
            .ok_or(GlError::Unsupported {
                operation: OP,
                reason: "no exact renderbuffer format fact for this context",
            })?;
        if !facts.renderable {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "format lacks renderable evidence at this sample count",
            });
        }
        let internal = native_texture_format(desc.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no proven native renderbuffer mapping",
        })?;
        let width =
            i32::try_from(desc.width).map_err(|_| Self::validation(OP, "width exceeds GLsizei"))?;
        let height = i32::try_from(desc.height)
            .map_err(|_| Self::validation(OP, "height exceeds GLsizei"))?;
        // SAFETY: current-context contract; limits and facts were checked first.
        let name = unsafe { self.gl.create_renderbuffer() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        unsafe {
            self.gl.bind_renderbuffer(glow::RENDERBUFFER, Some(name));
            if desc.samples > 1 {
                self.gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    desc.samples as i32,
                    internal,
                    width,
                    height,
                );
            } else {
                self.gl
                    .renderbuffer_storage(glow::RENDERBUFFER, internal, width, height);
            }
        }
        if let Err(error) = self.driver_error(OP) {
            unsafe { self.gl.delete_renderbuffer(name) };
            return Err(error);
        }
        let id = RenderbufferId::new(self.context_stamp(), self.slot(OP)?, 0);
        self.renderbuffers.insert(id, (name, desc));
        Ok(id)
    }
    fn destroy_buffer_resource(&mut self, id: BufferId) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-buffer")?;
        let (name, _) = self.buffer("destroy-buffer", id)?;
        // SAFETY: current-context contract; liveness was checked before GL mutation.
        unsafe { self.gl.delete_buffer(name) };
        self.driver_error("destroy-buffer")?;
        self.buffers.remove(&id);
        Ok(())
    }
    fn destroy_texture_resource(&mut self, id: TextureId) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-texture")?;
        let (name, _) = self.texture("destroy-texture", id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_texture(name) };
        self.driver_error("destroy-texture")?;
        self.textures.remove(&id);
        Ok(())
    }
    fn destroy_render_buffer(&mut self, id: RenderbufferId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-render-buffer";
        self.assert_ready(OP)?;
        let (name, _) = self.renderbuffer(OP, id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_renderbuffer(name) };
        self.driver_error(OP)?;
        self.renderbuffers.remove(&id);
        Ok(())
    }
}

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl super::super::GlSamplerApi for NativeGlProvider<'_> {
    fn create_sampler(&mut self, desc: super::super::GlSamplerDesc) -> Result<SamplerId, GlError> {
        use glow::HasContext as _;
        self.assert_ready("create-sampler")?;
        desc.validate_for(&self.discovery)
            .map_err(|_| Self::validation("create-sampler", "invalid sampler descriptor"))?;
        // SAFETY: current-context contract; descriptor was fully preflighted.
        let name = unsafe { self.gl.create_sampler() }.map_err(|message| GlError::Driver {
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
        let id = SamplerId::new(self.context_stamp(), self.slot("create-sampler")?, 0);
        self.samplers.insert(id, name);
        Ok(id)
    }
    fn destroy_sampler(&mut self, id: SamplerId) -> Result<(), GlError> {
        use glow::HasContext as _;
        self.assert_ready("destroy-sampler")?;
        let name = self.sampler("destroy-sampler", id)?;
        // SAFETY: current-context contract; liveness was checked before mutation.
        unsafe { self.gl.delete_sampler(name) };
        self.driver_error("destroy-sampler")?;
        self.samplers.remove(&id);
        Ok(())
    }
}

/// Allocation usage applied to every native buffer (audit P2-11).
///
/// The policy is one explicit, auditable constant per family: `STATIC_DRAW`
/// matches the native desktop residency model where the 0.14 cache re-uploads
/// through `bufferSubData` while the driver is free to place the store in
/// device-local memory. A `DYNAMIC_DRAW` fast path may only be introduced
/// after profiling attributes a benefit (plan "Private: upload-ring/orphaning
/// strategy").
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(super) const BUFFER_ALLOCATION_USAGE: u32 = glow::STATIC_DRAW;

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

/// The native internal/storage format constant of one discovered `GlFormat`.
///
/// Only formats with a settled mapping are listed; every other format fails
/// closed at its domain's evidence gate even if it maps here.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(super) const fn native_texture_format(format: super::super::GlFormat) -> Option<u32> {
    match format {
        super::super::GlFormat::Rgba8Unorm => Some(glow::RGBA8),
        super::super::GlFormat::Rgba8Srgb => Some(glow::SRGB8_ALPHA8),
        super::super::GlFormat::Rgba16Float => Some(glow::RGBA16F),
        super::super::GlFormat::Rgba32Float => Some(glow::RGBA32F),
        super::super::GlFormat::Depth32Float => Some(glow::DEPTH_COMPONENT32F),
        super::super::GlFormat::Depth16Unorm => Some(glow::DEPTH_COMPONENT16),
        super::super::GlFormat::Depth24PlusStencil8 => Some(glow::DEPTH24_STENCIL8),
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

/// The framebuffer attachment point for a depth/stencil view format.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(super) const fn depth_attachment_point(format: super::super::GlFormat) -> Option<u32> {
    match format {
        super::super::GlFormat::Depth16Unorm | super::super::GlFormat::Depth32Float => {
            Some(glow::DEPTH_ATTACHMENT)
        }
        super::super::GlFormat::Depth24PlusStencil8 => Some(glow::DEPTH_STENCIL_ATTACHMENT),
        _ => None,
    }
}

/// Whether a depth/stencil format carries a stencil plane.
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(super) const fn has_stencil_plane(format: super::super::GlFormat) -> bool {
    matches!(format, super::super::GlFormat::Depth24PlusStencil8)
}
