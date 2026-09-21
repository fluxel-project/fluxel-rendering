//! Deterministic Layer 1 recorder.  It models Fluxel-owned identities, never GL names.
//!
//! This file holds the recorder state, its shared validation helpers and the
//! `GlFamilyApi` lifetime domain.  Each remaining domain lives in its own
//! submodule so a change to one cannot silently invalidate another: objects
//! in `resource`, programs and their inputs in `program`, the pass in
//! `render_pass`, indirect command buffers in `indirect`, copies in
//! `transfer`, and observation in `lifecycle`.  The optional compute/storage
//! wrapper lives in `compute_storage`.
//!
//! Every domain here mirrors the executable providers rather than an
//! idealized contract, so a differential test can trust the trace: it checks
//! what a provider checks, at the point the provider checks it, and it never
//! rejects something a provider accepts.  Where a provider's check needs state
//! the recorder deliberately does not model -- the installed raster pipeline,
//! a VAO's index binding, GL's signed scalar widths -- the recorder checks the
//! strongest condition its own state supports and says so at the call site
//! rather than inventing a rule.

mod compute_storage;
mod indirect;
mod lifecycle;
mod program;
mod render_pass;
mod resource;
// The old cross-module test fixture suite depended on the retired WebGL2
// render-graph harness.  Keep this recorder itself as the reusable oracle;
// current backend conformance tests own fixtures beside the new lowering.
mod transfer;

use std::collections::{BTreeMap, BTreeSet};

use super::*;

/// Exact observable mock operations, deliberately expressed in domain vocabulary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MockCall {
    CreateBuffer(BufferId),
    DestroyBuffer(BufferId),
    CreateTexture(TextureId),
    DestroyTexture(TextureId),
    CreateRenderBuffer(RenderbufferId),
    DestroyRenderBuffer(RenderbufferId),
    CreateSampler(SamplerId),
    DestroySampler(SamplerId),
    CreateShader(ShaderId),
    DestroyShader(ShaderId),
    CreateProgram(ProgramId),
    DestroyProgram(ProgramId),
    CreateVertexArray(VertexArrayId),
    DestroyVertexArray(VertexArrayId),
    BindVertexArray(VertexArrayId),
    CreateFramebuffer(FramebufferId),
    DestroyFramebuffer(FramebufferId),
    BeginRenderPass(FramebufferId),
    EndRenderPass,
    SetRasterPipeline {
        program: ProgramId,
        vertex_array: VertexArrayId,
    },
    DrawRaster(GlDrawCommand),
    /// One draw carrying the optional base-vertex/base-instance offsets.
    DrawAdvancedRaster(GlAdvancedDrawCommand),
    /// One record read from a command buffer, drawn as one command.
    DrawIndirect(GlIndirectCommandRange),
    /// One work-group triple read from a command buffer.
    DispatchIndirect(GlDispatchIndirectCommand),
    /// A batch issued as one combined command, not as its single draws.
    ///
    /// The recorder only produces this variant on the route where a provider
    /// would issue a combined command.  Decomposing a batch is observable as
    /// the `DrawRaster` calls it really is, so the trace always says which
    /// submission shape happened instead of only which batch was requested.
    MultiDraw(GlMultiDraw),
    MultiDrawIndirect(GlIndirectCommandRange),
    MultiDrawIndirectCount {
        commands: GlIndirectCommandRange,
        count: GlIndirectCountRange,
    },
    CopyBuffer {
        source: BufferId,
        destination: BufferId,
        size: u64,
    },
    CopyTexture {
        source: TextureId,
        destination: TextureId,
    },
    UploadBuffer {
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    ReadBuffer {
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    UploadTexture(TextureId),
    ReadTexture(TextureId),
    ActiveTexture(u32),
    BindTexture {
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    },
    BindSampler {
        unit: u32,
        sampler: Option<SamplerId>,
    },
    BindUniformBuffer {
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    },
    BlitFramebuffer {
        source: FramebufferId,
        destination: FramebufferId,
    },
    CreateFence(GlFenceLease),
    DestroyFence(GlFenceLease),
    PollFence(GlFenceLease),
    WaitFence(GlFenceLease),
    Flush,
    AcquireSurface(GlSurfaceLease),
    ResizeSurface(GlSurfaceSize),
    SuspendSurface,
    ResumeSurface,
    PresentSurface(GlSurfaceLease),
    /// One frame published into the drawable from a texture.
    ///
    /// Recorded with both the lease and the source because the recorder has no
    /// drawable to blit into: what it can witness is that a caller paired this
    /// acquisition with this texture, which is the whole of what a publish
    /// claims before any driver is involved.
    PublishSurface {
        lease: GlSurfaceLease,
        source: TextureId,
    },
    CreateQuery(QueryId),
    DestroyQuery(QueryId),
    QueryResult(QueryId),
    BeginOcclusion(QueryId),
    EndOcclusion,
    BeginElapsed(QueryId),
    EndElapsed,
    QueryTimestamp(QueryId),
    SetComputeProgram(ProgramId),
    /// One selection of the driver's single current program.
    ///
    /// Recorded separately from the verb that asked for it because the two are
    /// not the same event: a pipeline install, a compute install, and the
    /// re-assertion a draw or dispatch performs when the slot was taken since
    /// all write this one piece of driver state.  A trace that named only the
    /// verbs could not show the re-assertion, which is the call a provider has
    /// to make for a dispatch to run the program the caller installed.
    SelectProgram(ProgramId),
    Dispatch(GlDispatchGroups),
    BindStorageBuffer {
        binding: u32,
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    BindStorageImage {
        binding: u32,
        texture: TextureId,
    },
    ContextLost,
    ContextRestored(ContextStamp),
    Error(GlError),
}

/// Common-profile mock. It intentionally does not implement compute or storage traits.
#[derive(Debug)]
pub struct MockGlFamilyApi {
    discovery: GlDiscoverySnapshot,
    stamp: ContextStamp,
    lifecycle: GlContextLifecycle,
    owner: OwnerThreadIdentity,
    next_slot: u32,
    buffers: BTreeMap<BufferId, GlBufferDesc>,
    textures: BTreeMap<TextureId, GlTextureDesc>,
    render_buffers: BTreeMap<RenderbufferId, GlRenderBufferDesc>,
    samplers: BTreeSet<SamplerId>,
    shaders: BTreeSet<ShaderId>,
    programs: BTreeSet<ProgramId>,
    vaos: BTreeSet<VertexArrayId>,
    framebuffers: BTreeMap<FramebufferId, GlFramebufferDescriptor>,
    queries: BTreeSet<QueryId>,
    fences: GlFenceLeaseBook,
    syncs: BTreeSet<SyncId>,
    surface: GlSurfaceLeaseBook,
    surface_size: GlSurfaceSize,
    surface_suspended: bool,
    pass_active: bool,
    /// The framebuffer the active pass renders into.
    ///
    /// Kept so an end can look it up, which is what both executable providers
    /// do after they have already consumed the pass: a mock that never checked
    /// would accept an end over a framebuffer the caller destroyed mid-pass.
    pass_framebuffer: Option<FramebufferId>,
    /// The compute program installed for dispatch work, if any.
    installed_compute_program: Option<ProgramId>,
    /// The program the modelled driver holds, or `None` when none is selected.
    ///
    /// GL has one current program and the recorder has to model that, not the
    /// intent of each verb: a pipeline install and a compute install write the
    /// same slot, and a link clears it because its reflection scope ends with no
    /// program selected.  A draw or dispatch that needs a program selects it
    /// again, which is why the trace can show a `SelectProgram` the caller never
    /// asked for -- that is the driver call the real provider makes.
    current_program: Option<ProgramId>,
    /// The raster program the active pass installed, if any.
    ///
    /// The recorder models the pipeline install as a program plus the vertex
    /// array it bound, and only the program half is needed to answer whether a
    /// draw has anything to draw with.  A pass end clears it, as the executable
    /// backends do.
    installed_raster_program: Option<ProgramId>,
    /// The vertex array the modelled driver holds, or `None` when it holds none.
    ///
    /// Separate from the program because the two go stale for different reasons:
    /// nothing but a compute install or a link touches the program slot, while
    /// the vertex-array slot is written by an input reconcile as well -- and an
    /// uncached reconcile destroys and replaces the array on *every* request.  So
    /// a recorder that kept the array the pipeline install named would refuse the
    /// array the driver is actually holding, which is precisely the defect the
    /// real provider had.  Both writers of the slot record it here: the pipeline
    /// install and `bind_vertex_array`.
    bound_vertex_array: Option<VertexArrayId>,
    /// The optional draw offsets this recorder treats as proved.
    ///
    /// A provider learns these from its own context queries, which the
    /// recorder has no equivalent of, so the test names them explicitly for the
    /// same reason fence completion and program reflection are injected.  The
    /// default proves nothing: an unconfigured recorder rejects a nonzero
    /// offset instead of accepting one no provider ever proved.
    advanced_raster: GlAdvancedRasterCapabilities,
    pixel_store: GlPixelStoreState,
    calls: Vec<MockCall>,
    next_error: Option<GlError>,
    /// Deterministic oracle answers for query observations.
    query_results: BTreeMap<QueryId, GlQueryResult>,
    /// Deterministic oracle answers for fence observations, keyed by the fence
    /// the observation names.  A fence absent from this map is reported
    /// `Pending`, never `Complete`: the recorder must not invent completion it
    /// was not told about, or every completion-safe release test would pass
    /// against a mock that never modelled the wait at all.
    fence_statuses: BTreeMap<SyncId, GlFenceStatus>,
    /// Deterministic reflection for the next `create_program` call.
    next_reflection: Option<GlProgramReflection>,
}
impl MockGlFamilyApi {
    pub fn from_discovery(discovery: GlDiscoverySnapshot) -> Self {
        Self {
            stamp: discovery.context_stamp(),
            discovery,
            lifecycle: GlContextLifecycle::Active,
            owner: OwnerThreadIdentity::current(),
            next_slot: 0,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            render_buffers: BTreeMap::new(),
            samplers: BTreeSet::new(),
            shaders: BTreeSet::new(),
            programs: BTreeSet::new(),
            vaos: BTreeSet::new(),
            framebuffers: BTreeMap::new(),
            queries: BTreeSet::new(),
            fences: GlFenceLeaseBook::default(),
            syncs: BTreeSet::new(),
            surface: GlSurfaceLeaseBook::new(),
            surface_size: GlSurfaceSize {
                width: 1,
                height: 1,
            },
            surface_suspended: false,
            pass_active: false,
            pass_framebuffer: None,
            installed_compute_program: None,
            current_program: None,
            installed_raster_program: None,
            bound_vertex_array: None,
            advanced_raster: GlAdvancedRasterCapabilities {
                base_vertex: false,
                first_instance: false,
            },
            pixel_store: GlPixelStoreState::DEFAULT,
            calls: vec![],
            next_error: None,
            query_results: BTreeMap::new(),
            fence_statuses: BTreeMap::new(),
            next_reflection: None,
        }
    }
    pub fn try_with_compute_storage(self) -> Result<MockComputeStorageApi, GlError> {
        MockComputeStorageApi::new(self)
    }
    pub fn calls(&self) -> &[MockCall] {
        &self.calls
    }
    pub fn clear_calls(&mut self) {
        self.calls.clear()
    }
    pub fn fail_next(&mut self, error: GlError) {
        self.next_error = Some(error)
    }
    /// Injects the deterministic answer one query observation returns.
    pub fn inject_query_result(&mut self, query: QueryId, result: GlQueryResult) {
        self.query_results.insert(query, result);
    }
    /// Injects the deterministic answer one fence observation returns.
    ///
    /// Completion is injected rather than simulated because the recorder has no
    /// submission queue: nothing it records can actually finish, so a
    /// completion-safe release path can only be exercised if the test names the
    /// fence that has completed.  Keying by `SyncId` rather than by lease means
    /// an injected completion survives re-issuing a lease for the same fence,
    /// which is what a driver-side signal does.  The map is kept to cover
    /// exactly the live fence set: `destroy_fence` removes the entry.
    pub fn inject_fence_status(&mut self, fence: SyncId, status: GlFenceStatus) {
        self.fence_statuses.insert(fence, status);
    }
    /// Injects the reflection the next `create_program` call returns.
    pub fn set_next_program_reflection(&mut self, reflection: GlProgramReflection) {
        self.next_reflection = Some(reflection);
    }
    /// Records which optional draw offsets this context proved.
    ///
    /// The seam exists because the recorder owns no context query for these
    /// facts, exactly like the injected fence completion and program
    /// reflection.  Without it a caller could only ever observe the
    /// fail-closed default, and the gate itself would stay untested.
    pub fn set_advanced_raster_capabilities(&mut self, capabilities: GlAdvancedRasterCapabilities) {
        self.advanced_raster = capabilities;
    }
    /// The injected answer for one fence, defaulting to `Pending`.
    ///
    /// `Pending` is the only honest default: the recorder has no submission
    /// queue, so nothing it records can complete on its own.  A default of
    /// `Complete` would make every completion-safe release test pass without
    /// the wait ever being modelled.
    fn fence_status(&self, fence: SyncId) -> GlFenceStatus {
        self.fence_statuses
            .get(&fence)
            .copied()
            .unwrap_or(GlFenceStatus::Pending)
    }
    fn owner(&self, op: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        (actual == self.owner)
            .then_some(())
            .ok_or(GlError::WrongThread {
                operation: op,
                expected: self.owner,
                actual,
            })
    }
    /// Makes `program` the modelled driver's current program, if it is not.
    ///
    /// One writer for the one slot, so the recorder cannot hold two beliefs about
    /// which program is current.  It records nothing when the slot already holds
    /// this program, which is what makes a redundant re-assertion invisible in
    /// the trace exactly as the real provider's comparison makes it free.
    fn select_program(&mut self, program: ProgramId) {
        if self.current_program != Some(program) {
            self.current_program = Some(program);
            self.calls.push(MockCall::SelectProgram(program));
        }
    }
    fn ready(&mut self, op: &'static str) -> Result<(), GlError> {
        self.owner(op)?;
        if let Some(e) = self.next_error.take() {
            self.error(e.clone());
            return Err(e);
        }
        match self.lifecycle {
            GlContextLifecycle::Active => Ok(()),
            GlContextLifecycle::Lost | GlContextLifecycle::Restoring => {
                self.error_result(GlError::ContextLost { operation: op })
            }
            GlContextLifecycle::Disposed => self.error_result(GlError::Disposed { operation: op }),
            GlContextLifecycle::Poisoned => self.error_result(GlError::Poisoned { operation: op }),
            _ => self.invalid(op, "context is not active"),
        }
    }
    fn error_result<T>(&mut self, e: GlError) -> Result<T, GlError> {
        self.error(e.clone());
        Err(e)
    }
    fn invalid<T>(&mut self, op: &'static str, message: &'static str) -> Result<T, GlError> {
        self.error_result(GlError::Validation {
            operation: op,
            message: message.into(),
        })
    }
    fn error(&mut self, e: GlError) {
        if matches!(e, GlError::ContextLost { .. }) {
            self.lifecycle = GlContextLifecycle::Lost;
        }
        self.calls.push(MockCall::Error(e));
    }
    fn slot(&mut self) -> Result<u32, GlError> {
        let slot = self.next_slot;
        self.next_slot = self.next_slot.checked_add(1).ok_or(GlError::OutOfMemory {
            operation: "mock-slot",
        })?;
        Ok(slot)
    }
    fn stamp(&mut self, op: &'static str, actual: ContextStamp) -> Result<(), GlError> {
        if actual.device != self.stamp.device {
            self.error_result(GlError::WrongContext {
                operation: op,
                object: actual,
                current: self.stamp,
            })
        } else if actual != self.stamp {
            self.error_result(GlError::StaleObject {
                operation: op,
                object: actual,
                current: self.stamp,
            })
        } else {
            Ok(())
        }
    }
    fn buffer(&mut self, op: &'static str, id: BufferId) -> Result<GlBufferDesc, GlError> {
        self.stamp(op, id.context)?;
        match self.buffers.get(&id).copied() {
            Some(desc) => Ok(desc),
            None => self.invalid(op, "buffer is not live"),
        }
    }
    fn texture(&mut self, op: &'static str, id: TextureId) -> Result<GlTextureDesc, GlError> {
        self.stamp(op, id.context)?;
        match self.textures.get(&id).copied() {
            Some(desc) => Ok(desc),
            None => self.invalid(op, "texture is not live"),
        }
    }
    fn render_buffer(
        &mut self,
        op: &'static str,
        id: RenderbufferId,
    ) -> Result<GlRenderBufferDesc, GlError> {
        self.stamp(op, id.context)?;
        match self.render_buffers.get(&id).copied() {
            Some(desc) => Ok(desc),
            None => self.invalid(op, "renderbuffer is not live"),
        }
    }
    fn binding_limits(&self) -> GlBindingLimits {
        let limits = self.discovery.limits();
        GlBindingLimits {
            max_texture_units: limits.max_combined_texture_image_units,
            max_uniform_buffer_bindings: limits.max_uniform_buffer_bindings,
            uniform_buffer_offset_alignment: limits.uniform_buffer_offset_alignment,
        }
    }
    fn invalid_binding(&mut self, op: &'static str, error: GlBindingValidationError) -> GlError {
        let error = GlError::Validation {
            operation: op,
            message: error.message().into(),
        };
        self.error(error.clone());
        error
    }
    fn live<K: GlObjectKind, F: FnOnce(&Self) -> bool>(
        &mut self,
        op: &'static str,
        id: ObjectIdentity<K>,
        exists: F,
    ) -> Result<(), GlError> {
        self.stamp(op, id.context)?;
        if exists(self) {
            Ok(())
        } else {
            self.invalid(op, "object is not live")
        }
    }
    fn reset_objects(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.render_buffers.clear();
        self.samplers.clear();
        self.shaders.clear();
        self.programs.clear();
        self.vaos.clear();
        self.framebuffers.clear();
        self.queries.clear();
        self.syncs.clear();
        self.fences.revoke_all();
        self.pass_active = false;
        self.pass_framebuffer = None;
        self.installed_compute_program = None;
        self.current_program = None;
        self.installed_raster_program = None;
        self.bound_vertex_array = None;
        // The new epoch has to requery every optional fact, and the recorder
        // cannot requery anything: keeping the previous context's answers would
        // let a restored context accept an offset it never proved.
        self.advanced_raster = GlAdvancedRasterCapabilities {
            base_vertex: false,
            first_instance: false,
        };
        self.query_results.clear();
        // Every fence is revoked above, so no surviving lease can name an
        // injected status; clearing keeps the oracle describing exactly the
        // live fence set rather than a previous context's observations.
        self.fence_statuses.clear();
        let _ = self.surface.invalidate_generation();
    }
    /// Validates one attachment view exactly as the executable backends do:
    /// the named allocation must be live, the view's format, extent, mip
    /// level, layer selection, and sample count must match it, and the format
    /// must carry renderable evidence for the storage class the view names
    /// (Phase C oracle parity).
    ///
    /// Both storage classes run the same rule sequence, so a descriptor is
    /// rejected for the same reason whichever one backs it. A texture addresses
    /// a mip level and may span layers; a renderbuffer has neither, so its only
    /// addressable view is level 0, layer 0, of exactly one layer. The sequence
    /// and every message are the providers' own, so a differential test compares
    /// one rejection rather than two spellings of it.
    fn validate_attachment(
        &mut self,
        op: &'static str,
        view: GlTextureView,
    ) -> Result<(), GlError> {
        match view.target {
            GlAttachmentTarget::Texture(texture) => {
                let desc = self.texture(op, texture)?;
                if desc.format != view.format {
                    return self
                        .invalid(op, "attachment view format does not match the allocation");
                }
                let Some(mip) = desc.mip_extent(view.mip_level) else {
                    return self.invalid(op, "attachment mip level is invalid");
                };
                if view.width != mip.width || view.height != mip.height {
                    return self.invalid(
                        op,
                        "attachment view extent does not match the allocation extent",
                    );
                }
                if view.array_layer != 0 || desc.dimension != GlTextureDimension::D2 {
                    return self.error_result(GlError::Unsupported {
                        operation: op,
                        reason: "layered attachments are not part of this framebuffer slice",
                    });
                }
                if desc.sample_count != view.sample_count {
                    return self
                        .invalid(op, "attachment sample count does not match the allocation");
                }
                // Keyed at the view's own sample count, not at 1: the evidence
                // table is keyed by (kind, format, sample_count), and the check
                // above has already required this view's count to equal the
                // allocation's. A recorder that looked up sample count 1 here
                // would accept a multisample attachment the executable backends
                // reject, which is the one thing the shared oracle must not do.
                self.require_renderable_attachment(
                    op,
                    GlFormatResourceKind::Texture,
                    view.format,
                    view.sample_count,
                )
            }
            GlAttachmentTarget::Renderbuffer(renderbuffer) => {
                let desc = self.render_buffer(op, renderbuffer)?;
                if desc.format != view.format {
                    return self
                        .invalid(op, "attachment view format does not match the allocation");
                }
                if view.mip_level != 0 {
                    return self.invalid(op, "attachment mip level is invalid");
                }
                if view.width != desc.width || view.height != desc.height {
                    return self.invalid(
                        op,
                        "attachment view extent does not match the allocation extent",
                    );
                }
                if view.array_layer != 0 || view.layer_count != 1 {
                    return self.error_result(GlError::Unsupported {
                        operation: op,
                        reason: "layered attachments are not part of this framebuffer slice",
                    });
                }
                if desc.samples != view.sample_count {
                    return self
                        .invalid(op, "attachment sample count does not match the allocation");
                }
                self.require_renderable_attachment(
                    op,
                    GlFormatResourceKind::Renderbuffer,
                    view.format,
                    view.sample_count,
                )
            }
        }
    }

    /// The last rule every attachment runs: this context has renderable evidence.
    fn require_renderable_attachment(
        &mut self,
        op: &'static str,
        kind: GlFormatResourceKind,
        format: GlFormat,
        sample_count: u32,
    ) -> Result<(), GlError> {
        let facts = self.discovery.formats().get_for(kind, format, sample_count);
        if facts.is_none_or(|facts| !facts.renderable) {
            return self.error_result(GlError::Unsupported {
                operation: op,
                reason: "attachment format lacks renderable evidence on this context",
            });
        }
        Ok(())
    }
    fn format_facts(
        &mut self,
        op: &'static str,
        desc: GlTextureDesc,
    ) -> Result<GlFormatCapabilities, GlError> {
        self.discovery
            .formats()
            .get_for(
                GlFormatResourceKind::Texture,
                desc.format,
                desc.sample_count,
            )
            .ok_or_else(|| GlError::Validation {
                operation: op,
                message: "no exact discovered format fact for texture allocation".into(),
            })
            .inspect_err(|error| self.error(error.clone()))
    }
    fn validate_buffer_allocation(&mut self, desc: GlBufferDesc) -> Result<(), GlError> {
        let capabilities = self.discovery.capabilities();
        if desc.usage.contains(GlBufferUsage::STORAGE)
            && !capabilities.supports(GlCapability::StorageBuffer)
        {
            return self.invalid(
                "create-buffer",
                "storage buffer usage lacks proved storage-buffer capability",
            );
        }
        if desc.usage.contains(GlBufferUsage::INDIRECT)
            && !capabilities.supports(GlCapability::IndirectDraw)
            && !capabilities.supports(GlCapability::IndirectDispatch)
        {
            return self.invalid(
                "create-buffer",
                "indirect buffer usage lacks proved indirect capability",
            );
        }
        Ok(())
    }
    fn validate_texture_allocation(&mut self, desc: GlTextureDesc) -> Result<(), GlError> {
        let facts = self.format_facts("create-texture", desc)?;
        if desc.usage.contains(GlTextureUsage::SAMPLED) && !facts.sampled {
            return self.invalid(
                "create-texture",
                "format is not sampled for this sample count",
            );
        }
        if desc.usage.contains(GlTextureUsage::RENDER_ATTACHMENT) && !facts.renderable {
            return self.invalid(
                "create-texture",
                "format is not renderable for this sample count",
            );
        }
        if desc.usage.contains(GlTextureUsage::COPY_SOURCE) && !facts.copy_source {
            return self.invalid("create-texture", "format is not a copy source");
        }
        if desc.usage.contains(GlTextureUsage::COPY_DESTINATION) && !facts.copy_destination {
            return self.invalid("create-texture", "format is not a copy destination");
        }
        if desc.usage.contains(GlTextureUsage::STORAGE_BINDING)
            && (!self
                .discovery
                .capabilities()
                .supports(GlCapability::StorageImage)
                || (!facts.storage_read && !facts.storage_write))
        {
            return self.invalid(
                "create-texture",
                "storage texture usage lacks proved image capability or exact format access",
            );
        }
        Ok(())
    }
    fn validate_storage_image_binding(
        &mut self,
        binding: u32,
        image: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::StorageImage)
        {
            return self.invalid(
                "bind-storage-image",
                "discovery did not prove storage-image capability",
            );
        }
        let desc = self.texture("bind-storage-image", image.texture)?;
        image.validate(
            binding,
            GlStorageImageLimits {
                max_image_units: self.discovery.limits().max_image_units,
            },
            self.discovery.formats(),
        )?;
        if !desc.usage.contains(GlTextureUsage::STORAGE_BINDING)
            || desc.format != image.format
            || desc.sample_count != image.sample_count
            || image.level >= desc.mip_level_count
        {
            return self.invalid(
                "bind-storage-image",
                "texture usage, format, sample count, or mip level is invalid for storage",
            );
        }
        let Some(mip_extent) = desc.mip_extent(image.level) else {
            return self.invalid("bind-storage-image", "storage image mip level is invalid");
        };
        let supports_layered = matches!(
            desc.dimension,
            GlTextureDimension::D3 | GlTextureDimension::D2Array | GlTextureDimension::Cube
        );
        if (image.layered && !supports_layered)
            || (!image.layered
                && image
                    .layer
                    .is_none_or(|layer| layer >= mip_extent.depth_or_layers))
        {
            return self.invalid(
                "bind-storage-image",
                "storage image layer selection is invalid for the texture shape",
            );
        }
        Ok(())
    }
}
impl GlFamilyApi for MockGlFamilyApi {
    fn profile(&self) -> GlFamilyProfile {
        self.discovery.context().profile()
    }
    fn context_stamp(&self) -> ContextStamp {
        self.stamp
    }
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }
    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }
    fn assert_owner_thread(&self, op: &'static str) -> Result<(), GlError> {
        self.owner(op)
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.discovery
    }
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.ready("context-lost")?;
        self.lifecycle = GlContextLifecycle::Lost;
        self.reset_objects();
        self.calls.push(MockCall::ContextLost);
        Ok(())
    }
    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        self.owner("context-restored")?;
        if self.lifecycle != GlContextLifecycle::Lost {
            return self.invalid("context-restored", "context is not lost");
        }
        if let Some(e) = self.next_error.take() {
            self.error(e.clone());
            return Err(e);
        }
        let Some(epoch) = self.stamp.epoch.checked_next() else {
            return self.error_result(GlError::Driver {
                operation: "context-restored",
                message: "context epoch exhausted".into(),
            });
        };
        self.lifecycle = GlContextLifecycle::Restoring;
        self.stamp = ContextStamp::new(self.stamp.device, epoch);
        self.discovery = self.discovery.rebind_for_test(self.stamp);
        self.next_slot = 0;
        self.lifecycle = GlContextLifecycle::Active;
        self.calls.push(MockCall::ContextRestored(self.stamp));
        Ok(self.stamp)
    }
}

pub(crate) use compute_storage::MockComputeStorageApi;
