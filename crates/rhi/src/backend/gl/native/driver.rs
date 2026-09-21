//! Safe bridge from the `Send + Sync` device façade to a current native GL context.
//!
//! WGL/EGL contexts are thread-affine, while the public RHI device is not.  It
//! is therefore unsound for `NativeGlDriver` to contain `glow::Context` or a
//! [`super::NativeContext`] directly: that would either fail `Send + Sync` or
//! tempt an `unsafe impl Send` which lets a public handle call GL from an
//! arbitrary thread.  Instead the platform integration supplies a
//! `NativeGlOwner`.  Its methods synchronously marshal to the thread on which
//! its WGL/EGL context is current, perform real lowering there, and return only
//! opaque GL names across the seam.
//!
//! This is a deliberately narrow internal adapter, not a public context/session
//! model.  A host without a synchronous owner-thread dispatch route cannot
//! honestly create a native GL RHI device: it must leave the driver uninstalled
//! and receive `Unsupported`, rather than report success and defer work forever.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::DeviceLossInfo;
use crate::api::presentation::PresentationTarget;
use crate::api::presentation::{
    AcquireError, AcquiredFrameId, PresentMode, PresentReceiptId, PresentState,
    PresentationConfiguration, PresentationExtentControl, PresentationTargetCapabilities,
};
use crate::api::query::QuerySetDescriptor;
use crate::api::resource::backend::MappingRequestBackend;
use crate::api::resource::buffer::BufferDescriptor;
use crate::api::resource::sampler::SamplerDescriptor;
use crate::api::resource::texture::TextureDescriptor;
use crate::api::resource::view::TextureViewDescriptor;
use crate::api::resource::{BufferRange, MapMode};
use crate::api::shader::ShaderArtifact;
use crate::api::submission::backend::SubmissionOutcome;
use crate::api::submission::{CompletionFailure, CompletionState};
use crate::backend::gl::api::{
    BufferId, GlFamilyApi, GlQueryObjectsApi, GlResourceApi, GlSamplerApi, GlShaderApi, GlSyncApi,
    GlVertexApi, ProgramId, QueryId, SamplerId, ShaderId, TextureId, VertexArrayId,
};
use crate::backend::gl::api::{GlSurfaceAcquire, GlSurfacePresentationApi};
use crate::backend::gl::platform::{
    GlBindGroupPacket, GlBufferRef, GlComputePipelinePacket, GlExecutionDriver, GlLossSink,
    GlObjectKind, GlObjectName, GlRasterPipelinePacket, GlSubmissionPlan, GlTextureRef,
};
use crate::backend::gl::state::{
    BoundGroupPacket, CanonicalBlockId, ContextState, ExecutionMode, PassPacket,
    RasterPipelineBlockInterner, RasterPipelinePacket as StateRasterPipelinePacket, StateDomain,
    StateEvent,
};

use super::{NativeGlProvider, NativeOwnerWorker, WorkerStartupError};

/// Owner-thread implementation behind [`NativeGlDriver`].
///
/// Each method is synchronous by contract.  Returning `Ok` means that the
/// native command has reached the current context and its creation/lowering
/// operation completed; a queued-but-not-run host task is not an acceptable
/// implementation.  `destroy` is intentionally best effort because wrapper
/// `Drop` cannot return an error; owners must record terminal context loss and
/// ignore native deletion only after that terminal transition.
pub(crate) trait NativeGlOwner: Send + Sync + 'static {
    /// Runs a lightweight operation on the actual current owner context.
    fn dispatch(&self, operation: &'static str) -> RhiResult<()>;

    fn create_buffer(&self, _: &BufferDescriptor) -> RhiResult<GlObjectName> {
        unsupported("create_buffer")
    }
    fn create_texture(&self, _: &TextureDescriptor) -> RhiResult<GlObjectName> {
        unsupported("create_texture")
    }
    fn create_texture_view(
        &self,
        _: GlTextureRef,
        _: &TextureViewDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_texture_view")
    }
    fn create_sampler(&self, _: &SamplerDescriptor) -> RhiResult<GlObjectName> {
        unsupported("create_sampler")
    }
    fn create_query_set(&self, _: &QuerySetDescriptor) -> RhiResult<GlObjectName> {
        unsupported("create_query_set")
    }
    fn create_shader(&self, _: &ShaderArtifact) -> RhiResult<GlObjectName> {
        unsupported("create_shader")
    }
    fn create_bind_group(&self, _: &GlBindGroupPacket) -> RhiResult<GlObjectName> {
        unsupported("create_bind_group")
    }
    fn create_compute_pipeline(&self, _: GlComputePipelinePacket<'_>) -> RhiResult<GlObjectName> {
        unsupported("create_compute_pipeline")
    }
    fn create_raster_pipeline(&self, _: GlRasterPipelinePacket<'_>) -> RhiResult<GlObjectName> {
        unsupported("create_raster_pipeline")
    }
    fn map_buffer(
        &self,
        _: GlBufferRef,
        _: MapMode,
        _: BufferRange,
    ) -> RhiResult<Box<dyn MappingRequestBackend>> {
        unsupported("map_buffer")
    }
    fn submit(&self, _: GlSubmissionPlan<'_>) -> RhiResult<SubmissionOutcome> {
        unsupported("submit")
    }
    fn completion(&self, serial: u64) -> CompletionState {
        CompletionState::Failed(CompletionFailure::new(format!(
            "native GL completion serial {serial} was never accepted"
        )))
    }
    fn completion_or_register_waker(&self, serial: u64, _: &Waker) -> CompletionState {
        self.completion(serial)
    }

    fn destroy(&self, _: GlObjectKind, _: GlObjectName) {}
    fn poll(&self) -> RhiResult<()> {
        self.dispatch("NativeGlOwner::poll")
    }
    fn wait_idle(&self) -> RhiResult<()> {
        unsupported("wait_idle")
    }
    fn supports_presentation(&self, _: &PresentationTarget) -> RhiResult<bool> {
        // Native GL presentation belongs to the WGL/EGL owner.  `false` is an
        // honest capability answer (unlike a successful present); an owner
        // that supports a target overrides this after it has verified its
        // drawable/swap route.
        Ok(false)
    }
    fn presentation_capabilities(&self, _: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        unsupported("presentation_capabilities")
    }
    fn configure_presentation(
        &self,
        _: ObjectId,
        _: &PresentationConfiguration,
    ) -> RhiResult<crate::backend::gl::platform::GlPresentationLease> {
        unsupported("configure_presentation")
    }
    fn lease_capabilities(
        &self,
        _: crate::backend::gl::platform::GlPresentationLease,
    ) -> RhiResult<PresentationTargetCapabilities> {
        unsupported("lease_capabilities")
    }
    fn reconfigure_or_register_waker(
        &self,
        _: crate::backend::gl::platform::GlPresentationLease,
        _: &PresentationConfiguration,
        _: &Waker,
    ) -> std::task::Poll<RhiResult<()>> {
        std::task::Poll::Ready(unsupported("reconfigure_presentation"))
    }
    fn try_acquire(
        &self,
        _: crate::backend::gl::platform::GlPresentationLease,
    ) -> Result<Option<crate::backend::gl::platform::GlAcquiredFramebuffer>, AcquireError> {
        Err(AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "native GL presentation is unavailable",
        ))
    }
    fn acquire_or_register_waker(
        &self,
        _: crate::backend::gl::platform::GlPresentationLease,
        _: &Waker,
    ) -> std::task::Poll<Result<crate::backend::gl::platform::GlAcquiredFramebuffer, AcquireError>>
    {
        std::task::Poll::Ready(Err(AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "native GL presentation is unavailable",
        )))
    }
    fn abandon(
        &self,
        _: crate::backend::gl::platform::GlPresentationLease,
        _: AcquiredFrameId,
    ) -> RhiResult<()> {
        unsupported("abandon_frame")
    }
    fn abandon_no_throw(
        &self,
        _: crate::backend::gl::platform::GlPresentationLease,
        _: AcquiredFrameId,
    ) {
    }
    fn release_presentation(&self, _: crate::backend::gl::platform::GlPresentationLease) {}
    fn present(&self, _: crate::backend::gl::platform::GlAcquiredFramebuffer, _: PresentReceiptId) {
    }
    fn terminate_present(
        &self,
        _: crate::backend::gl::platform::GlAcquiredFramebuffer,
        _: PresentReceiptId,
        _: PresentState,
    ) {
    }
    fn present_state(&self, _: PresentReceiptId) -> RhiResult<PresentState> {
        unsupported("present_state")
    }
    fn present_state_or_register_waker(
        &self,
        _: PresentReceiptId,
        _: &Waker,
    ) -> RhiResult<PresentState> {
        unsupported("present_state_or_register_waker")
    }
    fn device_lost(&self, _: &DeviceLossInfo) {}
}

/// `GlExecutionDriver` implementation backed by a synchronous native owner.
///
/// All native object names originate in the owner, whose actual GL object
/// tables remain private to the owner thread.  This forwarding layer never
/// allocates a synthetic name and never turns an unavailable route into `Ok`.
pub(crate) struct NativeGlDriver {
    owner: Arc<dyn NativeGlOwner>,
    /// Installed by the common GL Device after adoption. This is only a
    /// backend-private liveness callback, not a context/session handle.
    loss_sink: Arc<Mutex<Option<Arc<dyn GlLossSink>>>>,
}

/// The owned-provider half of a native owner-thread route.
///
/// `NativeGlProvider` now owns its `glow` dispatch table, so this worker stores
/// a complete executable object table on the thread which created and made the
/// WGL/EGL context current.  It intentionally exposes only the lightweight
/// dispatch proof at this stage: resource/pipeline/submission forwarding must
/// first install the corresponding generation-safe name carriers, and returning
/// success before those carriers exist would violate the v13 backend contract.
///
/// Platform factories construct the worker *after* currentness has been
/// established and move the provider into it. No borrowed dispatch table, raw
/// context handle, or self-reference crosses the thread boundary.
/// Platform-currentness operation needed before using an owned dispatch table.
/// Implemented only by native WGL/EGL context owners; browser contexts never
/// participate in this route.
pub(crate) trait NativePlatformContext: 'static {
    fn make_current(&mut self, operation: &'static str) -> RhiResult<()>;

    /// Whether this owned drawable can be swapped to a Host display. Pbuckets
    /// deliberately retain the default `false` answer.
    fn supports_presentation(&self) -> bool {
        false
    }
    /// Queries the currently drawable default-framebuffer extent. A missing or
    /// zero extent makes acquire wait rather than inventing a frame size.
    fn drawable_extent(&mut self) -> RhiResult<Option<crate::api::presentation::Extent2d>> {
        Ok(None)
    }
    /// Performs the platform flip after the GL executor flushed the accepted
    /// frame. This is separate from GPU completion by v13 contract.
    fn present(&mut self) -> RhiResult<()> {
        unsupported("native GL platform presentation")
    }
}

/// One worker-confined platform context plus its independently owned provider.
/// Field order matters: the provider drops first, deleting GL objects while the
/// platform context still exists; then the WGL/EGL owner tears down its handles.
pub(crate) struct NativeOwnedProvider<C: NativePlatformContext> {
    provider: NativeGlProvider,
    context: C,
    texture_formats: BTreeMap<u32, crate::api::format::TextureFormat>,
    /// v13 object carriers are table slots, never raw GL names.  The typed IDs
    /// retain the context epoch, making a carrier from a lost context fail
    /// lookup rather than accidentally target a replacement context.
    views: BTreeMap<u32, NativeTextureView>,
    bind_groups: BTreeMap<u32, GlBindGroupPacket>,
    // A portable query set owns an independently addressable native query for
    // every slot.  GL's object model has only individual queries, so encoding
    // just slot zero would make valid v13 indices silently alias.
    query_sets: BTreeMap<u32, NativeQuerySet>,
    raster_pipelines: BTreeMap<u32, NativeRasterPipeline>,
    compute_pipelines: BTreeMap<u32, ProgramId>,
    next_virtual: u32,
    next_canonical: u64,
    pipeline_blocks: RasterPipelineBlockInterner,
    geometry_blocks: HashMap<NativeGeometryKey, CanonicalBlockId>,
    context_state: ContextState,
    active_raster_framebuffer: Option<crate::backend::gl::api::FramebufferId>,
    next_completion: u64,
    completions: BTreeMap<u64, NativeCompletion>,
    presentation: NativePresentationState,
}

struct NativePresentationState {
    next_lease: u64,
    next_frame: u64,
    target: Option<ObjectId>,
    /// The public acquired-frame serial and native lease advance together.
    /// Retaining both makes an old FrameAttachment fail even after the same
    /// configured presentation lease acquired a newer frame.
    leases: BTreeMap<u64, Option<(u64, crate::backend::gl::api::GlSurfaceLease)>>,
    presents: HashMap<PresentReceiptId, PresentState>,
}

impl Default for NativePresentationState {
    fn default() -> Self {
        Self {
            next_lease: 1,
            next_frame: 1,
            target: None,
            leases: BTreeMap::new(),
            presents: HashMap::new(),
        }
    }
}

enum NativeCompletion {
    Pending {
        fence: crate::backend::gl::api::GlFenceLease,
        readbacks: Vec<PendingReadback>,
        /// Futures may be parked after the submission call has returned.  The
        /// native owner owns the only context that can observe their fence, so
        /// their wakers stay with this entry rather than relying on callers to
        /// repeatedly call `Device::poll`.
        waiters: Vec<Waker>,
    },
    Complete,
    /// A context-loss observation after the plan crossed the Phase-B commit
    /// boundary.  This must remain distinct from `Failed`: consumers use it
    /// to stop waiting and to rebuild the whole Device identity.
    DeviceLost(DeviceLossInfo),
    Failed(CompletionFailure),
}
#[derive(Clone, Copy)]
struct NativeTextureView {
    texture: TextureId,
    target: crate::backend::gl::api::GlTextureTarget,
    format: crate::backend::gl::api::GlFormat,
    mip_level: u32,
    base_layer: u32,
    layer_count: u32,
}
struct PendingReadback {
    ticket: crate::api::resource::transfer::ReadbackTicket,
    bytes: Vec<u8>,
    layout: Option<crate::api::resource::transfer::ReadbackTexelLayout>,
}

#[derive(Clone)]
struct NativeRasterPipeline {
    pipeline: crate::backend::gl::api::GlRasterPipeline,
    packet: StateRasterPipelinePacket,
}

/// Effective GL vertex-input state.  `vertex_array` owns the immutable
/// attribute layout; the generation-safe buffer identities, offsets and index
/// format own the per-draw VAO bindings.  Do not replace this with a draw
/// serial: equal geometry must be eligible for the ContextState fast path.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct NativeGeometryKey {
    vertex_array: VertexArrayId,
    vertices: Vec<crate::backend::gl::api::GlVertexBufferBinding>,
    index: Option<crate::backend::gl::api::GlIndexBinding>,
}

#[derive(Clone)]
struct NativeBoundGroup {
    packet: BoundGroupPacket,
}

#[derive(Clone)]
struct NativeQuerySet {
    ty: crate::api::query::QueryType,
    queries: Vec<QueryId>,
}

struct NativeRasterBeginAction {
    /// `None` represents the acquired default framebuffer. GL name zero is a
    /// binding convention, not an owned allocation, so it bypasses the FBO
    /// create/destroy table entirely.
    framebuffer: Option<crate::backend::gl::api::GlFramebufferDescriptor>,
    descriptor: crate::backend::gl::api::GlRenderPassDescriptor,
    pass: PassPacket,
}

struct NativeRasterDrawAction {
    pipeline: crate::backend::gl::api::GlRasterPipeline,
    packet: StateRasterPipelinePacket,
    /// Dynamic draw values mutate GL leaves without creating another public
    /// pipeline object.  The stored packet describes creation-time state, so
    /// it cannot be used for an exact identity hit after one of these values
    /// differs.  We currently invalidate the raster domain conservatively;
    /// later dynamic-leaf packets may narrow this to individual leaves.
    has_dynamic_state: bool,
    draw: crate::backend::gl::api::GlDrawCommand,
    geometry: Vec<crate::backend::gl::api::GlVertexBufferBinding>,
    index: Option<crate::backend::gl::api::GlIndexBinding>,
    geometry_key: CanonicalBlockId,
    bind_groups: Vec<NativeBoundGroup>,
}

/// A single GL draw-indirect record.  Count and multi-draw variants are kept
/// out of this packet deliberately: GL 4.0 / the ES extension route only
/// proves the one-record command, and accepting a portable count packet here
/// would turn an unproved extension into a fake capability.
struct NativeRasterIndirectAction {
    draw: NativeRasterDrawAction,
    command: crate::backend::gl::api::GlIndirectCommandRange,
}

/// Fully owned Phase-B work.  It contains only generation-safe provider IDs
/// and scalars; public recording handles never cross the owner-thread seam.
enum NativePhaseBAction {
    CopyBuffer {
        source: crate::backend::gl::api::GlBufferRange,
        destination: crate::backend::gl::api::GlBufferRange,
    },
    CopyTexture {
        source: crate::backend::gl::api::GlTextureRegion,
        destination: crate::backend::gl::api::GlTextureRegion,
    },
    UploadBuffer {
        destination: crate::backend::gl::api::GlBufferRange,
        bytes: Vec<u8>,
    },
    UploadTexture {
        destination: crate::backend::gl::api::GlTextureRegion,
        layout: crate::backend::gl::api::GlPixelLayout,
        bytes: Vec<u8>,
    },
    ReadBuffer {
        source: crate::backend::gl::api::GlBufferRange,
        ticket: crate::api::resource::transfer::ReadbackTicket,
    },
    ReadTexture {
        source: crate::backend::gl::api::GlTextureRegion,
        layout: crate::backend::gl::api::GlPixelLayout,
        ticket: crate::api::resource::transfer::ReadbackTicket,
    },
    RasterBegin(NativeRasterBeginAction),
    RasterDraw(NativeRasterDrawAction),
    RasterIndirect(NativeRasterIndirectAction),
    RasterEnd,
    /// Native GL4 compute is admitted only after the complete program and
    /// binding packet has been resolved during phase A.
    ComputeDispatch {
        program: ProgramId,
        program_key: CanonicalBlockId,
        groups: crate::backend::gl::api::GlDispatchGroups,
        bind_groups: Vec<NativeBoundGroup>,
    },
    QueryBegin {
        query: QueryId,
        ty: crate::api::query::QueryType,
        state_key: CanonicalBlockId,
    },
    QueryEnd {
        ty: crate::api::query::QueryType,
    },
    Timestamp(QueryId),
}

impl<C: NativePlatformContext> NativeOwnedProvider<C> {
    pub(crate) fn new(context: C, provider: NativeGlProvider) -> Self {
        Self {
            provider,
            context,
            texture_formats: BTreeMap::new(),
            views: BTreeMap::new(),
            bind_groups: BTreeMap::new(),
            query_sets: BTreeMap::new(),
            raster_pipelines: BTreeMap::new(),
            compute_pipelines: BTreeMap::new(),
            next_virtual: 1,
            next_canonical: 1,
            pipeline_blocks: RasterPipelineBlockInterner::new(),
            geometry_blocks: HashMap::new(),
            context_state: ContextState::new(ExecutionMode::Optimized),
            active_raster_framebuffer: None,
            next_completion: 1,
            completions: BTreeMap::new(),
            presentation: NativePresentationState::default(),
        }
    }

    fn canonical(&mut self, operation: &'static str) -> RhiResult<CanonicalBlockId> {
        let value = self.next_canonical;
        self.next_canonical = self.next_canonical.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "native GL canonical state namespace exhausted",
            )
            .at(operation)
        })?;
        Ok(CanonicalBlockId::new(value))
    }

    fn geometry_canonical(
        &mut self,
        key: NativeGeometryKey,
        operation: &'static str,
    ) -> RhiResult<CanonicalBlockId> {
        if let Some(canonical) = self.geometry_blocks.get(&key) {
            return Ok(*canonical);
        }
        let canonical = self.canonical(operation)?;
        self.geometry_blocks.insert(key, canonical);
        Ok(canonical)
    }

    fn ready(&mut self, operation: &'static str) -> RhiResult<()> {
        self.context.make_current(operation)?;
        crate::backend::gl::api::GlFamilyApi::assert_ready(&self.provider, operation).map_err(
            |error| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    format!("{operation}: {error:?}"),
                )
            },
        )
    }

    fn virtual_name(&mut self, operation: &'static str) -> RhiResult<GlObjectName> {
        let name = self.next_virtual;
        self.next_virtual = self.next_virtual.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "native GL virtual carrier namespace exhausted",
            )
            .at(operation)
        })?;
        GlObjectName::new(name, operation)
    }

    fn name(slot: u32, operation: &'static str) -> RhiResult<GlObjectName> {
        GlObjectName::new(
            slot.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "native GL object slot cannot be encoded",
                )
                .at(operation)
            })?,
            operation,
        )
    }

    fn slot(name: GlObjectName, operation: &'static str) -> RhiResult<u32> {
        name.raw().checked_sub(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "native GL object carrier is invalid",
            )
            .at(operation)
        })
    }

    fn buffer_id(&self, name: GlObjectName, operation: &'static str) -> RhiResult<BufferId> {
        let id = BufferId::new(
            self.provider.context_stamp(),
            Self::slot(name, operation)?,
            0,
        );
        self.provider
            .buffer(operation, id)
            .map_err(|e| map_gl(e, operation))?;
        Ok(id)
    }
    fn texture_id(&self, name: GlObjectName, operation: &'static str) -> RhiResult<TextureId> {
        let id = TextureId::new(
            self.provider.context_stamp(),
            Self::slot(name, operation)?,
            0,
        );
        self.provider
            .texture(operation, id)
            .map_err(|e| map_gl(e, operation))?;
        Ok(id)
    }
    fn shader_id(&self, name: GlObjectName, operation: &'static str) -> RhiResult<ShaderId> {
        let id = ShaderId::new(
            self.provider.context_stamp(),
            Self::slot(name, operation)?,
            0,
        );
        self.provider
            .shader(operation, id)
            .map_err(|e| map_gl(e, operation))?;
        Ok(id)
    }

    fn query_id(
        &self,
        set: GlObjectName,
        index: u32,
        expected: Option<crate::api::query::QueryType>,
        operation: &'static str,
    ) -> RhiResult<(QueryId, crate::api::query::QueryType)> {
        let record = self.query_sets.get(&set.raw()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "native GL query-set backing is not live",
            )
            .at(operation)
        })?;
        if let Some(expected) = expected {
            if record.ty != expected {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "native GL query operation does not match query-set type",
                )
                .at(operation));
            }
        }
        let query = record.queries.get(index as usize).copied().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "native GL query index is outside its set",
            )
            .at(operation)
        })?;
        self.provider
            .query(operation, query)
            .map_err(|e| map_gl(e, operation))?;
        Ok((query, record.ty))
    }

    fn reserve_completion_serial(&mut self) -> RhiResult<u64> {
        const OP: &str = "NativeProviderOwner::submit phase B";
        let serial = self.next_completion;
        self.next_completion = self.next_completion.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "native GL completion serial space exhausted",
            )
            .at(OP)
        })?;
        Ok(serial)
    }

    fn accept_submission(
        &mut self,
        serial: u64,
        readbacks: Vec<PendingReadback>,
    ) -> RhiResult<SubmissionOutcome> {
        const OP: &str = "NativeProviderOwner::submit phase B";
        let fence = self.provider.create_fence().map_err(|e| map_gl(e, OP))?;
        self.provider.flush().map_err(|e| map_gl(e, OP))?;
        self.completions.insert(
            serial,
            NativeCompletion::Pending {
                fence,
                readbacks,
                waiters: Vec::new(),
            },
        );
        Ok(SubmissionOutcome {
            completion: serial,
            points: Vec::new(),
        })
    }

    /// Executes the already-resolved Phase-B packet.  Once this method is
    /// entered GL may have accepted work, therefore its caller converts every
    /// error into a terminal completion rather than returning `submit(Err)`.
    fn execute_submission_after_commit(
        &mut self,
        actions: Vec<NativePhaseBAction>,
        points: Vec<crate::api::submission::PlanPoint>,
        serial: u64,
    ) -> RhiResult<SubmissionOutcome> {
        use crate::backend::gl::api::GlCopyDomainApi;
        let mut readbacks = Vec::new();
        for action in actions {
            match action {
                NativePhaseBAction::CopyBuffer {
                    source,
                    destination,
                } => self
                    .provider
                    .copy_buffer_range(source, destination)
                    .map_err(|error| map_gl(error, "NativeProviderOwner::submit copy-buffer"))?,
                NativePhaseBAction::CopyTexture {
                    source,
                    destination,
                } => self
                    .provider
                    .copy_texture_region(source, destination)
                    .map_err(|error| map_gl(error, "NativeProviderOwner::submit copy-texture"))?,
                NativePhaseBAction::UploadBuffer { destination, bytes } => self
                    .provider
                    .upload_buffer(destination, &bytes)
                    .map_err(|error| map_gl(error, "NativeProviderOwner::submit upload-buffer"))?,
                NativePhaseBAction::UploadTexture {
                    destination,
                    layout,
                    bytes,
                } => self
                    .provider
                    .upload_texture(destination, layout, &bytes)
                    .map_err(|error| map_gl(error, "NativeProviderOwner::submit upload-texture"))?,
                NativePhaseBAction::ReadBuffer { source, ticket } => {
                    let bytes = self.provider.read_buffer(source).map_err(|error| {
                        map_gl(error, "NativeProviderOwner::submit read-buffer")
                    })?;
                    readbacks.push(PendingReadback {
                        ticket,
                        bytes,
                        layout: None,
                    });
                }
                NativePhaseBAction::ReadTexture {
                    source,
                    layout,
                    ticket,
                } => {
                    let result = self
                        .provider
                        .read_texture(source, layout)
                        .map_err(|error| {
                            map_gl(error, "NativeProviderOwner::submit read-texture")
                        })?;
                    readbacks.push(PendingReadback {
                        ticket,
                        layout: Some(crate::api::resource::transfer::ReadbackTexelLayout {
                            bytes_per_row: result.layout.bytes_per_row,
                            rows_per_image: result.layout.rows_per_image,
                            total_size: result.bytes.len() as u64,
                        }),
                        bytes: result.bytes,
                    });
                }
                NativePhaseBAction::RasterBegin(action) => self.execute_raster_begin(action)?,
                NativePhaseBAction::RasterDraw(action) => self.execute_raster_draw(action)?,
                NativePhaseBAction::RasterIndirect(action) => {
                    self.execute_raster_indirect(action)?
                }
                NativePhaseBAction::RasterEnd => self.execute_raster_end()?,
                NativePhaseBAction::ComputeDispatch {
                    program,
                    program_key,
                    groups,
                    bind_groups,
                } => self.execute_compute_dispatch(program, program_key, groups, bind_groups)?,
                NativePhaseBAction::QueryBegin {
                    query,
                    ty,
                    state_key,
                } => self.execute_query_begin(query, ty, state_key)?,
                NativePhaseBAction::QueryEnd { ty } => self.execute_query_end(ty)?,
                NativePhaseBAction::Timestamp(query) => self.execute_timestamp(query)?,
            }
        }
        let mut outcome = self.accept_submission(serial, readbacks)?;
        outcome.points = points
            .into_iter()
            .map(|point| (point, outcome.completion))
            .collect();
        Ok(outcome)
    }

    fn fail_submission_after_commit(
        &mut self,
        serial: u64,
        points: Vec<crate::api::submission::PlanPoint>,
        readback_tickets: Vec<crate::api::resource::transfer::ReadbackTicket>,
        error: RhiError,
    ) -> SubmissionOutcome {
        let is_lost = error.kind() == RhiErrorKind::DeviceLost;
        for ticket in readback_tickets {
            ticket.set_status(if is_lost {
                crate::api::resource::transfer::ReadbackStatus::DeviceLost
            } else {
                crate::api::resource::transfer::ReadbackStatus::Failed
            });
        }
        let completion = if is_lost {
            NativeCompletion::DeviceLost(DeviceLossInfo::new(error.to_string()))
        } else {
            NativeCompletion::Failed(CompletionFailure::new(error.to_string()))
        };
        self.completions.insert(serial, completion);
        SubmissionOutcome {
            completion: serial,
            points: points.into_iter().map(|point| (point, serial)).collect(),
        }
    }

    fn execute_raster_begin(&mut self, mut action: NativeRasterBeginAction) -> RhiResult<()> {
        use crate::backend::gl::api::GlFramebufferApi as _;
        const OP: &str = "NativeProviderOwner::submit raster-begin";
        let framebuffer = if let Some(framebuffer_descriptor) = action.framebuffer.take() {
            let framebuffer = self
                .provider
                .create_framebuffer(&framebuffer_descriptor)
                .map_err(|e| map_gl(e, OP))?;
            action.descriptor.target =
                crate::backend::gl::api::GlRenderTarget::Offscreen(framebuffer);
            for (attachment, view) in action
                .descriptor
                .color_attachments
                .iter_mut()
                .zip(&framebuffer_descriptor.color_attachments)
            {
                attachment.view = crate::backend::gl::api::GlPassAttachmentView::Allocated(*view);
            }
            if let Some(attachment) = &mut action.descriptor.depth_stencil_attachment {
                attachment.view =
                    framebuffer_descriptor
                        .depth_stencil_attachment
                        .ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::BackendFailure,
                                "native raster depth attachment disappeared",
                            )
                            .at(OP)
                        })?;
            }
            Some(framebuffer)
        } else {
            // The typed GlDefaultFramebufferTarget in the descriptor is the
            // only authority for FBO 0; it is never an owned FramebufferId.
            None
        };
        let _ = self.context_state.prepare_pass(action.pass);
        if let Err(error) = self.provider.begin_render_pass(&action.descriptor) {
            self.context_state.pass_failed();
            if let Some(framebuffer) = framebuffer {
                let _ = self.provider.destroy_framebuffer(framebuffer);
            }
            return Err(map_gl(error, OP));
        }
        self.context_state.commit_pass(action.pass);
        self.active_raster_framebuffer = framebuffer;
        Ok(())
    }

    fn execute_raster_end(&mut self) -> RhiResult<()> {
        use crate::backend::gl::api::GlFramebufferApi as _;
        const OP: &str = "NativeProviderOwner::submit raster-end";
        self.provider.end_render_pass().map_err(|e| {
            self.context_state.pass_failed();
            map_gl(e, OP)
        })?;
        if let Some(framebuffer) = self.active_raster_framebuffer.take() {
            self.provider
                .destroy_framebuffer(framebuffer)
                .map_err(|e| map_gl(e, OP))?;
        }
        self.context_state.end_pass();
        Ok(())
    }

    /// Installs exactly the state shared by direct and one-record indirect
    /// raster draws.  Keeping this common path is important: an indirect draw
    /// must not bypass the state-machine's program/binding/VAO comparisons.
    fn prepare_raster_draw(&mut self, action: &NativeRasterDrawAction) -> RhiResult<()> {
        use crate::backend::gl::api::GlVertexApi as _;
        const OP: &str = "NativeProviderOwner::submit raster-draw";
        if action.has_dynamic_state {
            self.context_state
                .event(StateEvent::DomainFailed(StateDomain::RasterPipeline));
        }
        let diff = self
            .context_state
            .prepare_pipeline(action.packet)
            .map_err(|e| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    format!("native raster pipeline state was not registered: {e:?}"),
                )
                .at(OP)
            })?;
        if !diff.is_empty() {
            self.provider
                .apply_raster_pipeline_diff(&action.pipeline, diff)
                .map_err(|e| {
                    self.context_state.pipeline_failed();
                    map_gl(e, OP)
                })?;
            self.context_state.commit_pipeline(action.packet);
        }
        for group in &action.bind_groups {
            self.context_state.stage_bind_group(group.packet.clone());
        }
        let flush = self.context_state.binding_flush();
        if !flush.is_empty() {
            self.flush_bindings(&flush).map_err(|e| {
                self.context_state
                    .event(StateEvent::DomainFailed(StateDomain::Bindings));
                e
            })?;
            self.context_state.acknowledge_bindings(&flush);
        }
        if self.context_state.prepare_geometry(action.geometry_key) {
            self.provider
                .bind_vertex_array(action.pipeline.vertex_array, &action.geometry, action.index)
                .map_err(|e| {
                    self.context_state.geometry_failed();
                    map_gl(e, OP)
                })?;
            self.context_state.commit_geometry(action.geometry_key);
        }
        Ok(())
    }

    fn execute_raster_draw(&mut self, action: NativeRasterDrawAction) -> RhiResult<()> {
        use crate::backend::gl::api::GlRasterCommandApi as _;
        const OP: &str = "NativeProviderOwner::submit raster-draw";
        self.prepare_raster_draw(&action)?;
        self.provider.draw_raster(action.draw).map_err(|e| {
            self.context_state.pipeline_failed();
            map_gl(e, OP)
        })
    }

    fn execute_raster_indirect(&mut self, action: NativeRasterIndirectAction) -> RhiResult<()> {
        use crate::backend::gl::api::GlDrawIndirectApi as _;
        const OP: &str = "NativeProviderOwner::submit raster-indirect";
        self.prepare_raster_draw(&action.draw)?;
        self.provider
            .draw_indirect(action.command)
            .map_err(|error| {
                self.context_state.pipeline_failed();
                map_gl(error, OP)
            })
    }

    fn execute_compute_dispatch(
        &mut self,
        program: ProgramId,
        program_key: CanonicalBlockId,
        groups: crate::backend::gl::api::GlDispatchGroups,
        bind_groups: Vec<NativeBoundGroup>,
    ) -> RhiResult<()> {
        use crate::backend::gl::api::GlComputeDispatchApi as _;
        const OP: &str = "NativeProviderOwner::submit compute-dispatch";
        if self.context_state.prepare_compute_program(program_key) {
            self.provider
                .set_compute_program(program)
                .map_err(|error| {
                    self.context_state
                        .event(StateEvent::DomainFailed(StateDomain::Compute));
                    map_gl(error, OP)
                })?;
            self.context_state.commit_compute_program(program_key);
        }
        for group in bind_groups {
            self.context_state.stage_bind_group(group.packet);
        }
        let flush = self.context_state.binding_flush();
        if !flush.is_empty() {
            self.flush_bindings(&flush).map_err(|error| {
                self.context_state
                    .event(StateEvent::DomainFailed(StateDomain::Bindings));
                error
            })?;
            self.context_state.acknowledge_bindings(&flush);
        }
        self.provider.dispatch(groups).map_err(|error| {
            self.context_state
                .event(StateEvent::DomainFailed(StateDomain::Compute));
            map_gl(error, OP)
        })
    }

    fn execute_query_begin(
        &mut self,
        query: QueryId,
        ty: crate::api::query::QueryType,
        state_key: CanonicalBlockId,
    ) -> RhiResult<()> {
        use crate::backend::gl::api::{GlElapsedQueryApi as _, GlOcclusionQueryApi as _};
        const OP: &str = "NativeProviderOwner::submit query-begin";
        let result = match ty {
            crate::api::query::QueryType::Occlusion => self.provider.begin_occlusion_query(query),
            // Fluxel pipeline-statistics has no GL4.0 equivalent.  Timer
            // elapsed measurements are not a substitute: their result unit
            // and lifecycle differ, so no silent fallback is allowed.
            crate::api::query::QueryType::PipelineStatistics(_) => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "native GL has no pipeline-statistics query lowering",
                )
                .at(OP));
            }
            crate::api::query::QueryType::Timestamp => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "timestamp queries use write_timestamp rather than begin/end",
                )
                .at(OP));
            }
        };
        result.map_err(|error| {
            self.context_state.query_failed();
            map_gl(error, OP)
        })?;
        self.context_state.commit_query_begin(state_key);
        Ok(())
    }

    fn execute_query_end(&mut self, ty: crate::api::query::QueryType) -> RhiResult<()> {
        use crate::backend::gl::api::{GlElapsedQueryApi as _, GlOcclusionQueryApi as _};
        const OP: &str = "NativeProviderOwner::submit query-end";
        let result = match ty {
            crate::api::query::QueryType::Occlusion => self.provider.end_occlusion_query(),
            crate::api::query::QueryType::PipelineStatistics(_) => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "native GL has no pipeline-statistics query lowering",
                )
                .at(OP));
            }
            crate::api::query::QueryType::Timestamp => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "timestamp queries have no end command",
                )
                .at(OP));
            }
        };
        result.map_err(|error| {
            self.context_state.query_failed();
            map_gl(error, OP)
        })?;
        self.context_state.commit_query_end();
        Ok(())
    }

    fn execute_timestamp(&mut self, query: QueryId) -> RhiResult<()> {
        use crate::backend::gl::api::GlTimestampQueryApi as _;
        const OP: &str = "NativeProviderOwner::submit timestamp";
        self.provider.query_timestamp(query).map_err(|error| {
            self.context_state.query_failed();
            map_gl(error, OP)
        })
    }

    fn flush_bindings(&mut self, flush: &crate::backend::gl::state::BindingFlush) -> RhiResult<()> {
        use crate::backend::gl::api::GlBindingApi as _;
        use crate::backend::gl::platform::GlBindingResource;
        for group in &flush.groups {
            let bindings = self
                .bind_groups
                .get(&group.name.raw())
                .cloned()
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "native GL bind group backing is no longer live",
                    )
                })?;
            let mut offsets = group.dynamic_offsets.iter();
            for entry in &bindings.entries {
                match &entry.resource {
                    GlBindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    } => self.flush_uniform(entry.slot, *buffer, *offset, *size, &mut offsets)?,
                    GlBindingResource::BufferArray(values) => {
                        for (i, (buffer, offset, size)) in values.iter().enumerate() {
                            self.flush_uniform(
                                entry.slot + i as u32,
                                *buffer,
                                *offset,
                                *size,
                                &mut offsets,
                            )?;
                        }
                    }
                    GlBindingResource::Texture(view) => self.flush_texture(entry.slot, *view)?,
                    GlBindingResource::TextureArray(values) => {
                        for (i, view) in values.iter().enumerate() {
                            self.flush_texture(entry.slot + i as u32, *view)?;
                        }
                    }
                    GlBindingResource::Sampler(value) => self.flush_sampler(entry.slot, *value)?,
                    GlBindingResource::SamplerArray(values) => {
                        for (i, value) in values.iter().enumerate() {
                            self.flush_sampler(entry.slot + i as u32, *value)?;
                        }
                    }
                }
            }
            if offsets.next().is_some() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "native GL bind group has unused dynamic offsets",
                ));
            }
        }
        Ok(())
    }

    fn flush_uniform(
        &mut self,
        slot: u32,
        buffer: GlBufferRef,
        offset: u64,
        size: u64,
        offsets: &mut std::slice::Iter<'_, u32>,
    ) -> RhiResult<()> {
        use crate::backend::gl::api::GlBindingApi as _;
        const OP: &str = "NativeProviderOwner::submit bind uniform";
        let offset = offset
            .checked_add(u64::from(*offsets.next().unwrap_or(&0)))
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "native GL dynamic buffer offset overflow",
                )
                .at(OP)
            })?;
        let offset = u32::try_from(offset).map_err(|_| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "native GL uniform offset exceeds u32",
            )
            .at(OP)
        })?;
        let size = u32::try_from(size).map_err(|_| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "native GL uniform range exceeds u32",
            )
            .at(OP)
        })?;
        let id = self.buffer_id(buffer.name, OP)?;
        let key = CanonicalBlockId::uniform_range(id, offset, size);
        if self.context_state.prepare_uniform_slot(slot, key) {
            self.provider
                .bind_uniform_buffer(slot, Some(id), offset, size)
                .map_err(|e| map_gl(e, OP))?;
            self.context_state.commit_uniform_slot(slot, key);
        }
        Ok(())
    }

    fn flush_texture(
        &mut self,
        unit: u32,
        view: crate::backend::gl::platform::GlTextureViewRef,
    ) -> RhiResult<()> {
        use crate::backend::gl::api::GlBindingApi as _;
        const OP: &str = "NativeProviderOwner::submit bind texture";
        let view_ref = view;
        let view = *self.views.get(&view_ref.name.raw()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "native GL texture view backing is not live",
            )
            .at(OP)
        })?;
        // Do not reduce a view to its base texture identity.  Native texture
        // views (when installed) can have distinct target/range/format state.
        let key = CanonicalBlockId::new(u64::from(view_ref.name.raw()));
        let (active, bind) = self.context_state.prepare_texture_slot(unit, key);
        if active {
            self.provider
                .active_texture(unit)
                .map_err(|e| map_gl(e, OP))?;
        }
        if bind {
            self.provider
                .bind_texture(unit, view.target, Some(view.texture))
                .map_err(|e| map_gl(e, OP))?;
        }
        if active || bind {
            self.context_state.commit_texture_slot(unit, key);
        }
        Ok(())
    }

    fn flush_sampler(
        &mut self,
        unit: u32,
        sampler: crate::backend::gl::platform::GlSamplerRef,
    ) -> RhiResult<()> {
        use crate::backend::gl::api::GlBindingApi as _;
        const OP: &str = "NativeProviderOwner::submit bind sampler";
        let id = SamplerId::new(
            self.provider.context_stamp(),
            Self::slot(sampler.name, OP)?,
            0,
        );
        self.provider.sampler(OP, id).map_err(|e| map_gl(e, OP))?;
        let key = CanonicalBlockId::object(id);
        if self.context_state.prepare_sampler_slot(unit, key) {
            self.provider
                .bind_sampler(unit, Some(id))
                .map_err(|e| map_gl(e, OP))?;
            self.context_state.commit_sampler_slot(unit, key);
        }
        Ok(())
    }

    fn completion_state(&mut self, serial: u64) -> CompletionState {
        let mut wake = Vec::new();
        let Some(entry) = self.completions.get_mut(&serial) else {
            return CompletionState::Failed(CompletionFailure::new(format!(
                "native GL completion serial {serial} was never accepted"
            )));
        };
        if let NativeCompletion::Pending {
            fence,
            readbacks,
            waiters,
        } = entry
        {
            match self.provider.poll_fence(*fence) {
                Ok(crate::backend::gl::api::GlFenceStatus::Complete) => {
                    let fence = *fence;
                    for readback in readbacks.drain(..) {
                        readback.ticket.publish(readback.bytes, readback.layout);
                    }
                    wake.append(waiters);
                    *entry = NativeCompletion::Complete;
                    let _ = self.provider.destroy_fence(fence);
                }
                Ok(
                    crate::backend::gl::api::GlFenceStatus::Pending
                    | crate::backend::gl::api::GlFenceStatus::Unknown,
                ) => {}
                Ok(crate::backend::gl::api::GlFenceStatus::Failed) => {
                    for readback in readbacks.drain(..) {
                        readback
                            .ticket
                            .set_status(crate::api::resource::transfer::ReadbackStatus::Failed);
                    }
                    wake.append(waiters);
                    *entry = NativeCompletion::Failed(CompletionFailure::new(
                        "native GL fence reported terminal failure",
                    ));
                }
                Err(error) => {
                    let error = map_gl(error, "poll native GL completion");
                    let lost = error.kind() == RhiErrorKind::DeviceLost;
                    for readback in readbacks.drain(..) {
                        readback.ticket.set_status(if lost {
                            crate::api::resource::transfer::ReadbackStatus::DeviceLost
                        } else {
                            crate::api::resource::transfer::ReadbackStatus::Failed
                        });
                    }
                    wake.append(waiters);
                    *entry = if lost {
                        NativeCompletion::DeviceLost(DeviceLossInfo::new(error.to_string()))
                    } else {
                        NativeCompletion::Failed(CompletionFailure::new(error.to_string()))
                    };
                }
            }
        }
        let state = match entry {
            NativeCompletion::Pending { .. } => CompletionState::Pending,
            NativeCompletion::Complete => CompletionState::Complete,
            NativeCompletion::DeviceLost(info) => CompletionState::DeviceLost(info.clone()),
            NativeCompletion::Failed(failure) => CompletionState::Failed(failure.clone()),
        };
        for waker in wake {
            waker.wake();
        }
        state
    }

    fn completion_or_register_waker(&mut self, serial: u64, waker: &Waker) -> CompletionState {
        let state = self.completion_state(serial);
        if matches!(state, CompletionState::Pending)
            && let Some(NativeCompletion::Pending { waiters, .. }) =
                self.completions.get_mut(&serial)
            && !waiters.iter().any(|registered| registered.will_wake(waker))
        {
            waiters.push(waker.clone());
        }
        state
    }
}

pub(crate) struct NativeProviderOwner<C: NativePlatformContext> {
    worker: Arc<NativeOwnerWorker<NativeOwnedProvider<C>>>,
    completion_monitor_active: Arc<AtomicBool>,
}

impl<C: NativePlatformContext> NativeProviderOwner<C> {
    pub(crate) fn new(worker: NativeOwnerWorker<NativeOwnedProvider<C>>) -> Self {
        Self {
            worker: Arc::new(worker),
            completion_monitor_active: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Starts an owner worker from a factory which creates a fully owned
    /// provider on its own thread.  This is the only constructor native WGL/EGL
    /// adapters should use; constructing a provider on the caller then moving a
    /// borrowed dispatch table would reintroduce the old self-reference.
    pub(crate) fn spawn(
        factory: impl FnOnce() -> Result<NativeOwnedProvider<C>, String> + Send + 'static,
    ) -> Result<Self, WorkerStartupError> {
        NativeOwnerWorker::spawn(factory).map(Self::new)
    }

    /// Constructs the WGL/EGL context and gathers its immutable adoption
    /// information on the owner thread, then returns only that `Send` data to
    /// the factory.  The context/provider itself never crosses the boundary.
    pub(crate) fn spawn_with_info<I: Send + 'static>(
        factory: impl FnOnce() -> Result<(NativeOwnedProvider<C>, I), String> + Send + 'static,
    ) -> Result<(Self, I), WorkerStartupError> {
        NativeOwnerWorker::spawn_with_info(factory).map(|(worker, info)| (Self::new(worker), info))
    }
}

impl<C: NativePlatformContext> NativeGlOwner for NativeProviderOwner<C> {
    fn dispatch(&self, operation: &'static str) -> RhiResult<()> {
        self.worker
            .call(move |owner| owner.ready(operation))
            .map_err(|_| {
                RhiError::new(
                    RhiErrorKind::DeviceLost,
                    "native GL owner thread exited before dispatch completed",
                )
                .at(operation)
            })?
    }

    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<GlObjectName> {
        let descriptor = descriptor.clone();
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_buffer";
                owner.ready(OP)?;
                let id = owner
                    .provider
                    .create_buffer_resource(crate::backend::gl::translate::buffer_descriptor(
                        &descriptor,
                    )?)
                    .map_err(|e| map_gl(e, OP))?;
                NativeOwnedProvider::<C>::name(id.slot, OP)
            })
            .map_err(worker_error)?
    }

    fn create_texture(&self, descriptor: &TextureDescriptor) -> RhiResult<GlObjectName> {
        let descriptor = descriptor.clone();
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_texture";
                owner.ready(OP)?;
                let id = owner
                    .provider
                    .create_texture_resource(crate::backend::gl::translate::texture_descriptor(
                        &descriptor,
                    )?)
                    .map_err(|e| map_gl(e, OP))?;
                owner.texture_formats.insert(id.slot, descriptor.format);
                NativeOwnedProvider::<C>::name(id.slot, OP)
            })
            .map_err(worker_error)?
    }

    fn create_texture_view(
        &self,
        texture: GlTextureRef,
        descriptor: &TextureViewDescriptor,
    ) -> RhiResult<GlObjectName> {
        let descriptor = descriptor.clone();
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_texture_view";
                owner.ready(OP)?;
                let texture = owner.texture_id(texture.name, OP)?;
                // GL core has no independently allocated view in this baseline.
                // Retain the typed backing identity so a destroyed/lost texture can
                // never be resurrected by a virtual view carrier.
                let (_, base_desc) = owner
                    .provider
                    .texture(OP, texture)
                    .map_err(|e| map_gl(e, OP))?;
                let base = *owner.texture_formats.get(&texture.slot).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "native GL texture format carrier is missing",
                    )
                    .at(OP)
                })?;
                let dimension = crate::backend::gl::translate::whole_compatible_virtual_view(
                    &descriptor,
                    base,
                    base_desc,
                )?;
                let name = owner.virtual_name(OP)?;
                owner.views.insert(
                    name.raw(),
                    NativeTextureView {
                        texture,
                        target: binding_target(dimension)?,
                        format: crate::backend::gl::translate::view_format(&descriptor, base)?,
                        mip_level: descriptor.base_mip,
                        base_layer: descriptor.base_layer,
                        layer_count: descriptor.layer_count,
                    },
                );
                Ok(name)
            })
            .map_err(worker_error)?
    }

    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> RhiResult<GlObjectName> {
        let descriptor = descriptor.clone();
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_sampler";
                owner.ready(OP)?;
                let id = owner
                    .provider
                    .create_sampler(crate::backend::gl::translate::sampler_descriptor(
                        &descriptor,
                    )?)
                    .map_err(|e| map_gl(e, OP))?;
                NativeOwnedProvider::<C>::name(id.slot, OP)
            })
            .map_err(worker_error)?
    }

    fn create_query_set(&self, descriptor: &QuerySetDescriptor) -> RhiResult<GlObjectName> {
        let descriptor = descriptor.clone();
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_query_set";
                owner.ready(OP)?;
                // Pipeline-statistics are not emulated with unrelated GL
                // counters.  Reject creation before allocating any native
                // query, keeping the public capability/lowering boundary
                // transactional.
                if matches!(
                    descriptor.ty,
                    crate::api::query::QueryType::PipelineStatistics(_)
                ) {
                    return unsupported(OP);
                }
                let mut queries = Vec::with_capacity(descriptor.count as usize);
                for _ in 0..descriptor.count {
                    match owner.provider.create_query().map_err(|e| map_gl(e, OP)) {
                        Ok(query) => queries.push(query),
                        Err(error) => {
                            for query in queries {
                                let _ = owner.provider.destroy_query(query);
                            }
                            return Err(error);
                        }
                    }
                }
                let name = owner.virtual_name(OP)?;
                owner.query_sets.insert(
                    name.raw(),
                    NativeQuerySet {
                        ty: descriptor.ty,
                        queries,
                    },
                );
                Ok(name)
            })
            .map_err(worker_error)?
    }

    fn create_shader(&self, artifact: &ShaderArtifact) -> RhiResult<GlObjectName> {
        let artifact = artifact.clone();
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_shader";
                owner.ready(OP)?;
                let source = crate::backend::gl::translate::shader_source(&artifact)?;
                let id = owner
                    .provider
                    .create_shader(&source)
                    .map_err(|e| map_gl(e, OP))?;
                NativeOwnedProvider::<C>::name(id.slot, OP)
            })
            .map_err(worker_error)?
    }

    fn create_bind_group(&self, packet: &GlBindGroupPacket) -> RhiResult<GlObjectName> {
        let packet = packet.clone();
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_bind_group";
                owner.ready(OP)?;
                validate_bind_packet(owner, &packet, OP)?;
                let name = owner.virtual_name(OP)?;
                owner.bind_groups.insert(name.raw(), packet);
                Ok(name)
            })
            .map_err(worker_error)?
    }

    fn create_raster_pipeline(
        &self,
        packet: GlRasterPipelinePacket<'_>,
    ) -> RhiResult<GlObjectName> {
        let descriptor = packet.descriptor.clone();
        let vertex = packet.vertex;
        let fragment = packet.fragment;
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_raster_pipeline";
                owner.ready(OP)?;
                owner.shader_id(vertex.name, OP)?;
                if let Some(fragment) = fragment {
                    owner.shader_id(fragment.name, OP)?;
                }
                let default_viewport = crate::backend::gl::api::GlViewport {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    min_depth: 0.0f32.to_bits(),
                    max_depth: 1.0f32.to_bits(),
                };
                let packet = crate::backend::gl::translate_raster::raster_pipeline_packet(
                    &descriptor,
                    default_viewport,
                )?;
                let (program, _) = owner
                    .provider
                    .create_program(&packet.program)
                    .map_err(|e| map_gl(e, OP))?;
                let vertex_array = match owner.provider.create_vertex_array(&packet.vertex_layout) {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = owner.provider.destroy_program(program);
                        return Err(map_gl(error, OP));
                    }
                };
                let name = owner.virtual_name(OP)?;
                let pipeline = crate::backend::gl::api::GlRasterPipeline {
                    program,
                    vertex_array,
                    state: packet.state,
                };
                let state_packet = StateRasterPipelinePacket {
                    identity: name,
                    blocks: owner.pipeline_blocks.intern(&pipeline, OP).map_err(|_| {
                        RhiError::new(
                            RhiErrorKind::BackendFailure,
                            "native GL canonical pipeline namespace exhausted",
                        )
                        .at(OP)
                    })?,
                };
                owner
                    .context_state
                    .register_pipeline(state_packet)
                    .map_err(|error| {
                        RhiError::new(
                            RhiErrorKind::BackendFailure,
                            format!("native GL canonical pipeline registration failed: {error:?}"),
                        )
                        .at(OP)
                    })?;
                owner.raster_pipelines.insert(
                    name.raw(),
                    NativeRasterPipeline {
                        pipeline,
                        packet: state_packet,
                    },
                );
                Ok(name)
            })
            .map_err(worker_error)?
    }

    fn create_compute_pipeline(
        &self,
        packet: GlComputePipelinePacket<'_>,
    ) -> RhiResult<GlObjectName> {
        let descriptor = packet.descriptor.clone();
        let shader = packet.shader;
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::create_compute_pipeline";
                owner.ready(OP)?;
                owner.shader_id(shader.name, OP)?;
                let source =
                    crate::backend::gl::translate::shader_source(descriptor.shader.artifact())?;
                let layout = crate::backend::gl::translate::pipeline_layout_from_artifacts(
                    &descriptor.interface,
                    &[descriptor.shader.artifact()],
                )?;
                let program = crate::backend::gl::api::GlProgramDescriptor {
                    kind: crate::backend::gl::api::GlProgramKind::Compute { shader: source },
                    layout,
                    debug_name: descriptor.label.0.clone(),
                };
                let (program, _) = owner
                    .provider
                    .create_program(&program)
                    .map_err(|e| map_gl(e, OP))?;
                let name = owner.virtual_name(OP)?;
                owner.compute_pipelines.insert(name.raw(), program);
                Ok(name)
            })
            .map_err(worker_error)?
    }

    fn submit(&self, plan: GlSubmissionPlan<'_>) -> RhiResult<SubmissionOutcome> {
        // Phase A resolves each public resource to an owner-table ID before a
        // single GL call.  Its result is owned Phase-B work, so a rejection
        // here proves that no native command was accepted.
        let mut actions = Vec::new();
        let mut points = Vec::with_capacity(plan.batches.len());
        for batch in plan.batches {
            points.push(batch.point);
            for command in batch.commands {
                // ResourceUse is an execution input.  The current native GL
                // binding ABI has no storage-buffer/image packet yet, hence a
                // shader write cannot be lowered honestly even on a context
                // exposing glMemoryBarrier.  Reject before phase B rather than
                // accepting visibility the backend cannot establish.
                if command.uses.iter().any(native_gl_unlowered_shader_write) {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "native GL storage-write visibility requires a complete storage-binding and memory-barrier lowering",
                    ).at("NativeProviderOwner::submit phase A resource visibility"));
                }
                match &command.payload {
                    crate::api::command::record::RecordedPayload::Copy(crate::api::command::record::CopyRecord::Buffer(copy)) => {
                        let source = crate::backend::gl::platform::GlDevice::buffer_ref(&copy.src)?.name;
                        let destination = crate::backend::gl::platform::GlDevice::buffer_ref(&copy.dst)?.name;
                        let size = copy.size;
                        let source_offset = copy.src_offset;
                        let destination_offset = copy.dst_offset;
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            owner.ready("NativeProviderOwner::submit phase A copy-buffer")?;
                            Ok(NativePhaseBAction::CopyBuffer {
                                source: crate::backend::gl::api::GlBufferRange { buffer: owner.buffer_id(source, "NativeProviderOwner::submit phase A copy-buffer")?, offset: source_offset, size },
                                destination: crate::backend::gl::api::GlBufferRange { buffer: owner.buffer_id(destination, "NativeProviderOwner::submit phase A copy-buffer")?, offset: destination_offset, size },
                            })
                        }).map_err(worker_error)??;
                        actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::Copy(crate::api::command::record::CopyRecord::Texture(copy)) => {
                        let source = crate::backend::gl::platform::GlDevice::texture_ref(&copy.src)?.name;
                        let destination = crate::backend::gl::platform::GlDevice::texture_ref(&copy.dst)?.name;
                        let source_layers = copy.src_subresource;
                        let destination_layers = copy.dst_subresource;
                        let source_origin = copy.src_origin;
                        let destination_origin = copy.dst_origin;
                        let extent = copy.extent;
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            const OP: &str = "NativeProviderOwner::submit phase A copy-texture";
                            owner.ready(OP)?;
                            Ok(NativePhaseBAction::CopyTexture {
                                source: texture_region(owner, source, source_layers, source_origin, extent, OP)?,
                                destination: texture_region(owner, destination, destination_layers, destination_origin, extent, OP)?,
                            })
                        }).map_err(worker_error)??;
                        actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::Upload(upload) => match upload.descriptor() {
                        crate::api::resource::transfer::UploadDescriptor::Buffer(upload) => {
                            let name = crate::backend::gl::platform::GlDevice::buffer_ref(&upload.dst)?.name;
                            let offset = upload.dst_offset;
                            let bytes = upload.bytes.to_vec();
                            let size = u64::try_from(bytes.len()).map_err(|_| RhiError::new(RhiErrorKind::OutOfMemory, "native GL upload payload is too large"))?;
                            let action = self.worker.call(move |owner| -> RhiResult<_> {
                                const OP: &str = "NativeProviderOwner::submit phase A upload-buffer";
                                owner.ready(OP)?;
                                Ok(NativePhaseBAction::UploadBuffer {
                                    destination: crate::backend::gl::api::GlBufferRange { buffer: owner.buffer_id(name, OP)?, offset, size },
                                    bytes,
                                })
                            }).map_err(worker_error)??;
                            actions.push(action);
                        }
                        crate::api::resource::transfer::UploadDescriptor::Texture(upload) => {
                            let compressed = crate::backend::gl::translate::is_compressed_texture_format(
                                upload.dst.descriptor().format,
                            );
                            if compressed {
                                crate::backend::gl::translate::validate_compressed_texture_upload(
                                    upload,
                                    "NativeProviderOwner::submit phase A upload-texture",
                                )?;
                            } else if !matches!(upload.dst.descriptor().format, crate::api::format::TextureFormat::Rgba8Unorm | crate::api::format::TextureFormat::Rgba8UnormSrgb)
                                || !matches!(upload.subresource.aspect, crate::api::resource::TextureAspect::Color) {
                                return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL texture upload currently admits RGBA8 color textures").at("NativeProviderOwner::submit phase A"));
                            }
                            let name = crate::backend::gl::platform::GlDevice::texture_ref(&upload.dst)?.name;
                            let layers = upload.subresource;
                            let origin = upload.origin;
                            let extent = upload.extent;
                            let bytes = upload.bytes.to_vec();
                            let bytes_per_row = upload.source_layout.bytes_per_row;
                            let rows_per_image = upload.source_layout.rows_per_image;
                            let alignment = if bytes_per_row.is_multiple_of(8) { 8 } else if bytes_per_row.is_multiple_of(4) { 4 } else if bytes_per_row.is_multiple_of(2) { 2 } else { 1 };
                            let action = self.worker.call(move |owner| -> RhiResult<_> {
                                const OP: &str = "NativeProviderOwner::submit phase A upload-texture";
                                owner.ready(OP)?;
                                // The compressed executor consumes an exact
                                // encoded whole-mip payload and never inspects
                                // this uncompressed pixel-layout carrier.
                                Ok(NativePhaseBAction::UploadTexture {
                                    destination: texture_region(owner, name, layers, origin, extent, OP)?,
                                    layout: crate::backend::gl::api::GlPixelLayout { format: crate::backend::gl::api::GlPixelFormat::Rgba8, bytes_per_row, rows_per_image, offset: 0, alignment, repack: crate::backend::gl::api::GlRepackPolicy::Bounded { max_bytes: bytes.len() as u64 } },
                                    bytes,
                                })
                            }).map_err(worker_error)??;
                            actions.push(action);
                        },
                    },
                    crate::api::command::record::RecordedPayload::Readback(ticket) => match ticket.request() {
                        crate::api::resource::transfer::ReadbackRequest::Buffer { src, range, .. } => {
                            let name = crate::backend::gl::platform::GlDevice::buffer_ref(src)?.name;
                            let range = *range;
                            let ticket = ticket.clone();
                            let action = self.worker.call(move |owner| -> RhiResult<_> {
                                const OP: &str = "NativeProviderOwner::submit phase A read-buffer";
                                owner.ready(OP)?;
                                Ok(NativePhaseBAction::ReadBuffer { source: crate::backend::gl::api::GlBufferRange { buffer: owner.buffer_id(name, OP)?, offset: range.offset, size: range.size }, ticket })
                            }).map_err(worker_error)??;
                            actions.push(action);
                        }
                        crate::api::resource::transfer::ReadbackRequest::Texture { src, subresource, origin, extent, .. } => {
                            if !matches!(src.descriptor().format, crate::api::format::TextureFormat::Rgba8Unorm | crate::api::format::TextureFormat::Rgba8UnormSrgb)
                                || !matches!(subresource.aspect, crate::api::resource::TextureAspect::Color) {
                                return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL texture readback currently admits RGBA8 color textures").at("NativeProviderOwner::submit phase A"));
                            }
                            let name = crate::backend::gl::platform::GlDevice::texture_ref(src)?.name;
                            let layers = *subresource;
                            let origin = *origin;
                            let extent = *extent;
                            let ticket = ticket.clone();
                            let bytes_per_row = extent.width.checked_mul(4).ok_or_else(|| RhiError::new(RhiErrorKind::OutOfMemory, "native GL texture readback row size overflows"))?;
                            let action = self.worker.call(move |owner| -> RhiResult<_> {
                                const OP: &str = "NativeProviderOwner::submit phase A read-texture";
                                owner.ready(OP)?;
                                Ok(NativePhaseBAction::ReadTexture {
                                    source: texture_region(owner, name, layers, origin, extent, OP)?,
                                    layout: crate::backend::gl::api::GlPixelLayout { format: crate::backend::gl::api::GlPixelFormat::Rgba8, bytes_per_row, rows_per_image: extent.height, offset: 0, alignment: 4, repack: crate::backend::gl::api::GlRepackPolicy::Bounded { max_bytes: u64::MAX } },
                                    ticket,
                                })
                            }).map_err(worker_error)??;
                            actions.push(action);
                        },
                    },
                    crate::api::command::record::RecordedPayload::RasterBegin(begin) => {
                        if begin.depth_stencil.is_some() { return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL raster depth/stencil Phase-A lowering is not admitted yet").at("NativeProviderOwner::submit phase A")); }
                        // A presentation attachment remains a typed acquired
                        // framebuffer throughout lowering.  It is never
                        // coerced into a TextureView merely because native GL
                        // happens to bind it through FBO zero.
                        if let [(_, color)] = begin.colors.as_slice() {
                            if let crate::api::command::attachment::ColorAttachmentView::Frame(frame) = &color.view {
                                if color.resolve.is_some() {
                                    return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL default framebuffer resolve lowering is not admitted yet").at("NativeProviderOwner::submit phase A raster-begin"));
                                }
                                let crate::backend::gl::translate_raster::GlFramebufferCarrier::Default { color_locations } = crate::backend::gl::translate_raster::raster_begin_carrier(begin, |view| match view {
                                    crate::api::command::attachment::ColorAttachmentView::Frame(_) => Ok(None),
                                    crate::api::command::attachment::ColorAttachmentView::Texture(_) => Err(RhiError::new(RhiErrorKind::InvalidUsage, "mixed default and texture framebuffer attachments").at("NativeProviderOwner::submit phase A raster-begin")),
                                })? else { unreachable!() };
                                let [location] = color_locations.as_slice() else { unreachable!() };
                                if *location != 0 {
                                    return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL default framebuffer admits color location zero only").at("NativeProviderOwner::submit phase A raster-begin"));
                                }
                                let acquired = crate::backend::gl::platform::framebuffer_ref(frame)?;
                                let extent = frame.extent();
                                if extent.depth != 1 || frame.sample_count() != 1 || extent.width != acquired.extent.width || extent.height != acquired.extent.height {
                                    return Err(RhiError::new(RhiErrorKind::InvalidUsage, "native GL frame facts changed after acquisition").at("NativeProviderOwner::submit phase A raster-begin"));
                                }
                                let format = crate::backend::gl::translate::texture_format(frame.format())?;
                                let load = color.load;
                                let store = color.store;
                                let action = self.worker.call(move |owner| -> RhiResult<_> {
                                    use crate::api::command::geometry::{ColorClearValue, LoadOp, StoreOp};
                                    const OP: &str = "NativeProviderOwner::submit phase A raster-begin-default";
                                    owner.ready(OP)?;
                                    let Some(Some((serial, native_lease))) = owner.presentation.leases.get(&acquired.lease.0).copied() else {
                                        return Err(RhiError::new(RhiErrorKind::InvalidUsage, "native GL frame lease is not currently acquired").at(OP));
                                    };
                                    if serial != acquired.serial
                                        || acquired.framebuffer != 0
                                        || native_lease.size.width != acquired.extent.width
                                        || native_lease.size.height != acquired.extent.height
                                    {
                                        return Err(RhiError::new(RhiErrorKind::InvalidUsage, "native GL acquired frame is stale or names a foreign default framebuffer").at(OP));
                                    }
                                    let current = owner.context.drawable_extent()?;
                                    if current != Some(acquired.extent) {
                                        return Err(RhiError::new(RhiErrorKind::InvalidUsage, "native GL drawable extent changed after acquisition").at(OP));
                                    }
                                    let target = crate::backend::gl::api::GlDefaultFramebufferTarget { frame_serial: acquired.serial, context: owner.provider.context_stamp(), width: extent.width, height: extent.height, sample_count: 1, color_format: format };
                                    let clear = match load { LoadOp::Load => crate::backend::gl::api::GlColorClearValue { red: 0, green: 0, blue: 0, alpha: 0 }, LoadOp::Clear(ColorClearValue::Float(value)) => crate::backend::gl::api::GlColorClearValue { red: value[0].to_bits(), green: value[1].to_bits(), blue: value[2].to_bits(), alpha: value[3].to_bits() }, LoadOp::Clear(ColorClearValue::Sint(value)) => crate::backend::gl::api::GlColorClearValue { red: value[0] as u32, green: value[1] as u32, blue: value[2] as u32, alpha: value[3] as u32 }, LoadOp::Clear(ColorClearValue::Uint(value)) => crate::backend::gl::api::GlColorClearValue { red: value[0], green: value[1], blue: value[2], alpha: value[3] } };
                                    let pass = PassPacket { draw_framebuffer: owner.canonical(OP)?, read_framebuffer: owner.canonical(OP)?, draw_buffers: owner.canonical(OP)? };
                                    Ok(NativePhaseBAction::RasterBegin(NativeRasterBeginAction { framebuffer: None, descriptor: crate::backend::gl::api::GlRenderPassDescriptor { target: crate::backend::gl::api::GlRenderTarget::Default(target), color_attachments: vec![crate::backend::gl::api::GlColorAttachment { view: crate::backend::gl::api::GlPassAttachmentView::DefaultColor(target), resolve_target: None, load: if matches!(load, LoadOp::Clear(_)) { crate::backend::gl::api::GlLoadOp::Clear } else { crate::backend::gl::api::GlLoadOp::Load }, store: if store == StoreOp::Discard { crate::backend::gl::api::GlStoreOp::Discard } else { crate::backend::gl::api::GlStoreOp::Store }, clear }], depth_stencil_attachment: None }, pass }))
                                }).map_err(worker_error)??;
                                actions.push(action);
                                continue;
                            }
                        }
                        let mut colors = Vec::with_capacity(begin.colors.len());
                        for (location, color) in &begin.colors {
                            let crate::api::command::attachment::ColorAttachmentView::Texture(view) = &color.view else { return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL default framebuffer raster passes require presentation lowering").at("NativeProviderOwner::submit phase A")); };
                            if color.resolve.is_some() { return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL raster resolve Phase-A lowering is not admitted yet").at("NativeProviderOwner::submit phase A")); }
                            colors.push((*location, crate::backend::gl::platform::GlDevice::view_ref(view)?.name, color.load, color.store));
                        }
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            use crate::api::command::geometry::{ColorClearValue, LoadOp, StoreOp};
                            const OP: &str = "NativeProviderOwner::submit phase A raster-begin"; owner.ready(OP)?;
                            let mut views = Vec::with_capacity(colors.len()); let mut attachments = Vec::with_capacity(colors.len()); let mut locations = Vec::with_capacity(colors.len());
                            for (location, name, load, store) in colors {
                                let view = attachment_view(owner, name, OP)?; locations.push(location); views.push(view);
                                let clear = match load { LoadOp::Load => crate::backend::gl::api::GlColorClearValue { red: 0, green: 0, blue: 0, alpha: 0 }, LoadOp::Clear(ColorClearValue::Float(value)) => crate::backend::gl::api::GlColorClearValue { red: value[0].to_bits(), green: value[1].to_bits(), blue: value[2].to_bits(), alpha: value[3].to_bits() }, LoadOp::Clear(ColorClearValue::Sint(value)) => crate::backend::gl::api::GlColorClearValue { red: value[0] as u32, green: value[1] as u32, blue: value[2] as u32, alpha: value[3] as u32 }, LoadOp::Clear(ColorClearValue::Uint(value)) => crate::backend::gl::api::GlColorClearValue { red: value[0], green: value[1], blue: value[2], alpha: value[3] } };
                                attachments.push(crate::backend::gl::api::GlColorAttachment { view: crate::backend::gl::api::GlPassAttachmentView::Allocated(view), resolve_target: None, load: if matches!(load, LoadOp::Clear(_)) { crate::backend::gl::api::GlLoadOp::Clear } else { crate::backend::gl::api::GlLoadOp::Load }, store: if store == StoreOp::Discard { crate::backend::gl::api::GlStoreOp::Discard } else { crate::backend::gl::api::GlStoreOp::Store }, clear });
                            }
                            let framebuffer = crate::backend::gl::api::GlFramebufferDescriptor { color_attachments: views, depth_stencil_attachment: None, draw_buffers: locations };
                            let pass = PassPacket { draw_framebuffer: owner.canonical(OP)?, read_framebuffer: owner.canonical(OP)?, draw_buffers: owner.canonical(OP)? };
                            Ok(NativePhaseBAction::RasterBegin(NativeRasterBeginAction { framebuffer: Some(framebuffer), descriptor: crate::backend::gl::api::GlRenderPassDescriptor { target: crate::backend::gl::api::GlRenderTarget::Offscreen(crate::backend::gl::api::FramebufferId::new(owner.provider.context_stamp(), 0, 0)), color_attachments: attachments, depth_stencil_attachment: None }, pass }))
                        }).map_err(worker_error)??; actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::RasterEnd => actions.push(NativePhaseBAction::RasterEnd),
                    crate::api::command::record::RecordedPayload::RasterDraw(draw) => {
                        let scalars = crate::backend::gl::translate_raster::raster_draw_scalars(draw)?;
                        scalars.draw.validate(crate::backend::gl::api::GlAdvancedRasterCapabilities { base_vertex: false, first_instance: false }).map_err(|_| RhiError::new(RhiErrorKind::Unsupported, "native GL baseline raster route has no base-vertex or first-instance lowering").at("NativeProviderOwner::submit phase A"))?;
                        let pipeline = crate::backend::gl::platform::GlDevice::raster_pipeline_ref(&draw.pipeline)?.name;
                        let vertices = draw.vertex_buffers.iter().map(|(slot, binding)| Ok((*slot, crate::backend::gl::platform::GlDevice::buffer_ref(&binding.buffer)?.name, binding.range.offset))).collect::<RhiResult<Vec<_>>>()?;
                        let index = draw.index.as_ref().map(|index| Ok((crate::backend::gl::platform::GlDevice::buffer_ref(&index.binding.buffer)?.name, index.format, index.binding.range.offset))).transpose()?;
                        let groups = draw.groups.iter().map(|group| Ok((group.index.get(), group.group.id(), crate::backend::gl::platform::GlDevice::bind_group_ref(&group.group)?.name, group.dynamic_offsets.clone()))).collect::<RhiResult<Vec<_>>>()?;
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            const OP: &str = "NativeProviderOwner::submit phase A raster-draw"; owner.ready(OP)?;
                            let stored = owner.raster_pipelines.get(&pipeline.raw()).cloned().ok_or_else(|| RhiError::new(RhiErrorKind::WrongDevice, "native GL raster pipeline backing is not live").at(OP))?;
                            let mut pipeline = stored.pipeline;
                            let has_dynamic_state = scalars.viewport.is_some()
                                || pipeline.state.scissor != scalars.scissor
                                || pipeline.state.blend_constant != scalars.blend_constant
                                || pipeline.state.depth_stencil.is_some_and(|value| value.stencil_reference != scalars.stencil_reference);
                            if let Some(viewport) = scalars.viewport { pipeline.state.viewport = viewport; }
                            pipeline.state.scissor = scalars.scissor;
                            pipeline.state.blend_constant = scalars.blend_constant;
                            if let Some(depth) = &mut pipeline.state.depth_stencil { depth.stencil_reference = scalars.stencil_reference; }
                            let geometry = vertices.into_iter().map(|(slot, name, offset)| Ok(crate::backend::gl::api::GlVertexBufferBinding { slot, buffer: owner.buffer_id(name, OP)?, offset })).collect::<RhiResult<Vec<_>>>()?;
                            let index = index.map(|(name, format, offset)| Ok(crate::backend::gl::api::GlIndexBinding { buffer: owner.buffer_id(name, OP)?, format: match format { crate::api::command::IndexFormat::Uint16 => crate::backend::gl::api::GlIndexFormat::Uint16, crate::api::command::IndexFormat::Uint32 => crate::backend::gl::api::GlIndexFormat::Uint32 }, offset })).transpose()?;
                            let mut bound = Vec::with_capacity(groups.len()); for (index, group, name, dynamic_offsets) in groups { if !owner.bind_groups.contains_key(&name.raw()) { return Err(RhiError::new(RhiErrorKind::WrongDevice, "native GL bind group backing is not live").at(OP)); } bound.push(NativeBoundGroup { packet: BoundGroupPacket { group, name, index, dynamic_offsets, program_identity: CanonicalBlockId::object(pipeline.program), dependencies: std::collections::BTreeSet::new() } }); }
                            let geometry_key = owner.geometry_canonical(NativeGeometryKey {
                                vertex_array: pipeline.vertex_array,
                                vertices: geometry.clone(),
                                index,
                            }, OP)?;
                            Ok(NativePhaseBAction::RasterDraw(NativeRasterDrawAction { pipeline, packet: stored.packet, has_dynamic_state, draw: scalars.draw.draw, geometry, index, geometry_key, bind_groups: bound }))
                        }).map_err(worker_error)??; actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::RasterIndirect(indirect) => {
                        // The discovery row intentionally exposes only the
                        // GL 4.0 / GLES extension one-record entry point.
                        // Multi-draw and count-indirect require separate
                        // native functions and must fail before Phase B.
                        if indirect.draw_count != 1 || indirect.count.is_some() {
                            return Err(RhiError::new(
                                RhiErrorKind::Unsupported,
                                "native GL only lowers one-record raster indirect draws",
                            ).at("NativeProviderOwner::submit phase A raster-indirect"));
                        }
                        let synthetic = crate::api::command::record::RasterDraw {
                            pipeline: indirect.pipeline.clone(),
                            groups: indirect.groups.clone(),
                            vertex_buffers: indirect.vertex_buffers.clone(),
                            index: indirect.index.clone(),
                            viewport: indirect.viewport,
                            scissor: indirect.scissor,
                            blend_constant: indirect.blend_constant,
                            stencil_reference: indirect.stencil_reference,
                            // Indirect arguments provide these values.  The
                            // direct command is never emitted, but the shared
                            // scalar translator still needs a well-formed
                            // placeholder to carry dynamic state.
                            range: 0..0,
                            instances: 0..1,
                            base_vertex: 0,
                            immediates: Vec::new(),
                        };
                        let scalars = crate::backend::gl::translate_raster::raster_draw_scalars(&synthetic)?;
                        let pipeline = crate::backend::gl::platform::GlDevice::raster_pipeline_ref(&indirect.pipeline)?.name;
                        let argument_name = crate::backend::gl::platform::GlDevice::buffer_ref(&indirect.arguments)?.name;
                        let argument_size = indirect.arguments.descriptor().size;
                        let vertices = indirect.vertex_buffers.iter().map(|(slot, binding)| Ok((*slot, crate::backend::gl::platform::GlDevice::buffer_ref(&binding.buffer)?.name, binding.range.offset))).collect::<RhiResult<Vec<_>>>()?;
                        let index = indirect.index.as_ref().map(|index| Ok((crate::backend::gl::platform::GlDevice::buffer_ref(&index.binding.buffer)?.name, index.format, index.binding.range.offset))).transpose()?;
                        let groups = indirect.groups.iter().map(|group| Ok((group.index.get(), group.group.id(), crate::backend::gl::platform::GlDevice::bind_group_ref(&group.group)?.name, group.dynamic_offsets.clone()))).collect::<RhiResult<Vec<_>>>()?;
                        let arguments_offset = indirect.arguments_offset;
                        let stride = indirect.stride;
                        let indexed = indirect.index.is_some();
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            const OP: &str = "NativeProviderOwner::submit phase A raster-indirect";
                            owner.ready(OP)?;
                            let stored = owner.raster_pipelines.get(&pipeline.raw()).cloned().ok_or_else(|| RhiError::new(RhiErrorKind::WrongDevice, "native GL raster pipeline backing is not live").at(OP))?;
                            let mut pipeline = stored.pipeline;
                            let has_dynamic_state = scalars.viewport.is_some()
                                || pipeline.state.scissor != scalars.scissor
                                || pipeline.state.blend_constant != scalars.blend_constant
                                || pipeline.state.depth_stencil.is_some_and(|value| value.stencil_reference != scalars.stencil_reference);
                            if let Some(viewport) = scalars.viewport { pipeline.state.viewport = viewport; }
                            pipeline.state.scissor = scalars.scissor;
                            pipeline.state.blend_constant = scalars.blend_constant;
                            if let Some(depth) = &mut pipeline.state.depth_stencil { depth.stencil_reference = scalars.stencil_reference; }
                            let geometry = vertices.into_iter().map(|(slot, name, offset)| Ok(crate::backend::gl::api::GlVertexBufferBinding { slot, buffer: owner.buffer_id(name, OP)?, offset })).collect::<RhiResult<Vec<_>>>()?;
                            let index = index.map(|(name, format, offset)| Ok(crate::backend::gl::api::GlIndexBinding { buffer: owner.buffer_id(name, OP)?, format: match format { crate::api::command::IndexFormat::Uint16 => crate::backend::gl::api::GlIndexFormat::Uint16, crate::api::command::IndexFormat::Uint32 => crate::backend::gl::api::GlIndexFormat::Uint32 }, offset })).transpose()?;
                            let mut bound = Vec::with_capacity(groups.len());
                            for (index, group, name, dynamic_offsets) in groups {
                                if !owner.bind_groups.contains_key(&name.raw()) { return Err(RhiError::new(RhiErrorKind::WrongDevice, "native GL bind group backing is not live").at(OP)); }
                                bound.push(NativeBoundGroup { packet: BoundGroupPacket { group, name, index, dynamic_offsets, program_identity: CanonicalBlockId::object(pipeline.program), dependencies: std::collections::BTreeSet::new() } });
                            }
                            let abi = if indexed { crate::backend::gl::api::GlIndirectAbi::Indexed } else { crate::backend::gl::api::GlIndirectAbi::NonIndexed };
                            let geometry_key = owner.geometry_canonical(NativeGeometryKey {
                                vertex_array: pipeline.vertex_array,
                                vertices: geometry.clone(),
                                index,
                            }, OP)?;
                            Ok(NativePhaseBAction::RasterIndirect(NativeRasterIndirectAction {
                                draw: NativeRasterDrawAction { pipeline, packet: stored.packet, has_dynamic_state, draw: scalars.draw.draw, geometry, index, geometry_key, bind_groups: bound },
                                command: crate::backend::gl::api::GlIndirectCommandRange {
                                    range: crate::backend::gl::api::GlBufferRange { buffer: owner.buffer_id(argument_name, OP)?, offset: 0, size: argument_size },
                                    command_offset: arguments_offset,
                                    draw_count: 1,
                                    stride,
                                    abi,
                                },
                            }))
                        }).map_err(worker_error)??;
                        actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::ComputeBegin(_)
                    | crate::api::command::record::RecordedPayload::ComputeEnd => {
                        // Scope boundaries are validation/diagnostic records.
                        // Native GL has no encoder object to enter/leave; a
                        // dispatch packet owns all executable state.
                    }
                    crate::api::command::record::RecordedPayload::ComputeDispatch(dispatch) => {
                        if !dispatch.immediates.is_empty() {
                            return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL has no verified immediate-data lowering").at("NativeProviderOwner::submit phase A compute-dispatch"));
                        }
                        let pipeline = crate::backend::gl::platform::GlDevice::compute_pipeline_ref(&dispatch.pipeline)?.name;
                        let groups = dispatch.groups.iter().map(|group| Ok((group.index.get(), group.group.id(), crate::backend::gl::platform::GlDevice::bind_group_ref(&group.group)?.name, group.dynamic_offsets.clone()))).collect::<RhiResult<Vec<_>>>()?;
                        let workgroups = dispatch.workgroups;
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            const OP: &str = "NativeProviderOwner::submit phase A compute-dispatch";
                            owner.ready(OP)?;
                            let program = owner.compute_pipelines.get(&pipeline.raw()).copied().ok_or_else(|| RhiError::new(RhiErrorKind::WrongDevice, "native GL compute pipeline backing is not live").at(OP))?;
                            let mut bound = Vec::with_capacity(groups.len());
                            for (index, group, name, dynamic_offsets) in groups {
                                if !owner.bind_groups.contains_key(&name.raw()) {
                                    return Err(RhiError::new(RhiErrorKind::WrongDevice, "native GL bind group backing is not live").at(OP));
                                }
                                bound.push(NativeBoundGroup { packet: BoundGroupPacket { group, name, index, dynamic_offsets, program_identity: CanonicalBlockId::object(program), dependencies: std::collections::BTreeSet::new() } });
                            }
                            Ok(NativePhaseBAction::ComputeDispatch { program, program_key: CanonicalBlockId::object(program), groups: crate::backend::gl::api::GlDispatchGroups([workgroups.0, workgroups.1, workgroups.2]), bind_groups: bound })
                        }).map_err(worker_error)??;
                        actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::QueryBegin { set, index } => {
                        let name = crate::backend::gl::platform::GlDevice::query_set_ref(set)?.name;
                        let index = *index;
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            const OP: &str = "NativeProviderOwner::submit phase A query-begin";
                            owner.ready(OP)?;
                            let (query, ty) = owner.query_id(name, index, None, OP)?;
                            if !matches!(ty, crate::api::query::QueryType::Occlusion) {
                                return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL only lowers occlusion begin/end query sets").at(OP));
                            }
                            Ok(NativePhaseBAction::QueryBegin { query, ty, state_key: owner.canonical(OP)? })
                        }).map_err(worker_error)??;
                        actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::QueryEnd { set, index } => {
                        let name = crate::backend::gl::platform::GlDevice::query_set_ref(set)?.name;
                        let index = *index;
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            const OP: &str = "NativeProviderOwner::submit phase A query-end";
                            owner.ready(OP)?;
                            let (_, ty) = owner.query_id(name, index, None, OP)?;
                            if !matches!(ty, crate::api::query::QueryType::Occlusion) {
                                return Err(RhiError::new(RhiErrorKind::Unsupported, "native GL only lowers occlusion begin/end query sets").at(OP));
                            }
                            Ok(NativePhaseBAction::QueryEnd { ty })
                        }).map_err(worker_error)??;
                        actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::TimestampWrite { set, index } => {
                        let name = crate::backend::gl::platform::GlDevice::query_set_ref(set)?.name;
                        let index = *index;
                        let action = self.worker.call(move |owner| -> RhiResult<_> {
                            const OP: &str = "NativeProviderOwner::submit phase A timestamp";
                            owner.ready(OP)?;
                            let (query, ty) = owner.query_id(name, index, Some(crate::api::query::QueryType::Timestamp), OP)?;
                            Ok(NativePhaseBAction::Timestamp(query))
                        }).map_err(worker_error)??;
                        actions.push(action);
                    }
                    crate::api::command::record::RecordedPayload::QueryResolve(_) => return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "native GL baseline has no query-result-to-buffer route; query-buffer support is not universal across GL4/GLES3",
                    ).at("NativeProviderOwner::submit phase A query-resolve")),
                    crate::api::command::record::RecordedPayload::DebugPush(_)
                    | crate::api::command::record::RecordedPayload::DebugPop
                    | crate::api::command::record::RecordedPayload::DebugMarker(_) => return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "native GL debug-marker commands require a verified KHR_debug entry-point executor",
                    ).at("NativeProviderOwner::submit phase A debug-marker")),
                    _ => return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "native GL v13 Phase A has no complete owned action for this recorded payload",
                    ).at("NativeProviderOwner::submit phase A")),
                }
            }
        }
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::submit phase B";
                owner.ready(OP)?;
                // The serial is reserved before the first native call.  Phase
                // A has already succeeded at this point, so a later GL error
                // may follow accepted work and cannot truthfully become
                // `submit(Err)`.
                let serial = owner.reserve_completion_serial()?;
                let readback_tickets = native_readback_tickets(&actions);
                match owner.execute_submission_after_commit(actions, points.clone(), serial) {
                    Ok(outcome) => Ok(outcome),
                    Err(error) => Ok(owner.fail_submission_after_commit(
                        serial,
                        points,
                        readback_tickets,
                        error,
                    )),
                }
            })
            .map_err(worker_error)?
    }

    fn completion(&self, serial: u64) -> CompletionState {
        self.worker
            .call(move |owner| {
                let _ = owner.ready("NativeProviderOwner::completion");
                owner.completion_state(serial)
            })
            .unwrap_or_else(|error| {
                CompletionState::Failed(CompletionFailure::new(worker_error(error).to_string()))
            })
    }

    fn completion_or_register_waker(&self, serial: u64, waker: &Waker) -> CompletionState {
        let waker = waker.clone();
        let state = self
            .worker
            .call(move |owner| {
                let _ = owner.ready("NativeProviderOwner::completion_or_register_waker");
                owner.completion_or_register_waker(serial, &waker)
            })
            .unwrap_or_else(|error| {
                CompletionState::Failed(CompletionFailure::new(worker_error(error).to_string()))
            });
        if matches!(state, CompletionState::Pending)
            && self
                .completion_monitor_active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            // One owner-thread monitor services all parked futures. It never
            // invokes GL directly from this helper thread.
            let worker = Arc::clone(&self.worker);
            let active = Arc::clone(&self.completion_monitor_active);
            let _ = std::thread::Builder::new()
                .name("fluxel-gl-fence-waker".into())
                .spawn(move || {
                    loop {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        let pending = worker.call(|owner| {
                            let serials: Vec<_> = owner.completions.keys().copied().collect();
                            serials.into_iter().any(|serial| {
                                matches!(owner.completion_state(serial), CompletionState::Pending)
                            })
                        });
                        if !matches!(pending, Ok(true)) {
                            active.store(false, Ordering::Release);
                            break;
                        }
                    }
                });
        }
        state
    }

    fn device_lost(&self, info: &DeviceLossInfo) {
        let info = info.clone();
        let _ = self.worker.call(move |owner| {
            for completion in owner.completions.values_mut() {
                if let NativeCompletion::Pending {
                    readbacks, waiters, ..
                } = completion
                {
                    for readback in readbacks.drain(..) {
                        readback
                            .ticket
                            .set_status(crate::api::resource::transfer::ReadbackStatus::DeviceLost);
                    }
                    let wake = std::mem::take(waiters);
                    *completion = NativeCompletion::DeviceLost(info.clone());
                    for waker in wake {
                        waker.wake();
                    }
                }
            }
            // A present receipt is an independent async domain.  Mark only
            // unobserved answers terminal; an already Accepted hand-off stays
            // historical fact, just as an already Complete completion does.
            for state in owner.presentation.presents.values_mut() {
                if matches!(state, PresentState::Pending) {
                    *state = PresentState::DeviceLost(info.clone());
                }
            }
            // No native consume/present call is valid after loss.  Dropping
            // these leases makes every stale FrameAttachment fail structurally
            // rather than letting it target a replacement drawable generation.
            for lease in owner.presentation.leases.values_mut() {
                *lease = None;
            }
            let _ = owner.provider.context_lost();
        });
    }

    fn destroy(&self, kind: GlObjectKind, name: GlObjectName) {
        // Drop cannot surface an error.  It nevertheless runs on the owner
        // thread and uses the typed table as the authority; a stale carrier is
        // simply already retired, never reinterpreted as a raw GL name.
        let _ = self.worker.call(move |owner| -> RhiResult<()> {
            const OP: &str = "NativeProviderOwner::destroy";
            owner.ready(OP)?;
            match kind {
                GlObjectKind::Buffer => {
                    let id = owner.buffer_id(name, OP)?;
                    // State retirement precedes native deletion: a failed or
                    // delayed delete must never leave a cache hit referring to
                    // this identity.
                    owner.context_state.event(StateEvent::BufferRetired(id));
                    // The key includes full typed identities, so retaining a
                    // stale entry would not create a false hit.  Removing it
                    // nevertheless bounds the private structural interner and
                    // ensures a future allocation does not inherit dead VAO
                    // bookkeeping.
                    owner.geometry_blocks.retain(|key, _| {
                        !key.vertices.iter().any(|binding| binding.buffer == id)
                            && !key.index.is_some_and(|index| index.buffer == id)
                    });
                    // Native bind-group packets predate the dependency index;
                    // until every packet carries resource refs, force their
                    // revalidation rather than retaining a stale applied bind.
                    owner
                        .context_state
                        .event(StateEvent::DomainFailed(StateDomain::Bindings));
                    owner
                        .provider
                        .destroy_buffer_resource(id)
                        .map_err(|e| map_gl(e, OP))?;
                }
                GlObjectKind::Texture => {
                    let id = owner.texture_id(name, OP)?;
                    owner.context_state.event(StateEvent::TextureRetired(id));
                    owner
                        .context_state
                        .event(StateEvent::DomainFailed(StateDomain::Bindings));
                    owner
                        .provider
                        .destroy_texture_resource(id)
                        .map_err(|e| map_gl(e, OP))?;
                    owner.texture_formats.remove(&id.slot);
                    owner.views.retain(|_, view| view.texture != id);
                }
                GlObjectKind::Sampler => {
                    let id = SamplerId::new(
                        owner.provider.context_stamp(),
                        NativeOwnedProvider::<C>::slot(name, OP)?,
                        0,
                    );
                    owner.context_state.event(StateEvent::SamplerRetired(id));
                    owner
                        .context_state
                        .event(StateEvent::DomainFailed(StateDomain::Bindings));
                    owner
                        .provider
                        .destroy_sampler(id)
                        .map_err(|e| map_gl(e, OP))?;
                }
                GlObjectKind::Shader => {
                    let id = owner.shader_id(name, OP)?;
                    owner
                        .provider
                        .destroy_shader(id)
                        .map_err(|e| map_gl(e, OP))?;
                }
                GlObjectKind::QuerySet => {
                    let set = owner.query_sets.remove(&name.raw()).ok_or_else(|| {
                        RhiError::new(RhiErrorKind::WrongDevice, "native GL query set is not live")
                            .at(OP)
                    })?;
                    for query in set.queries {
                        owner.context_state.event(StateEvent::QueryRetired(query));
                        owner
                            .provider
                            .destroy_query(query)
                            .map_err(|e| map_gl(e, OP))?;
                    }
                }
                GlObjectKind::TextureView => {
                    owner.views.remove(&name.raw());
                    // A view has no native object identity, but a cached
                    // bind-group may still resolve it.  Force the next bind
                    // flush to revalidate the private view packet.
                    owner
                        .context_state
                        .event(StateEvent::DomainFailed(StateDomain::Bindings));
                }
                GlObjectKind::BindGroup => {
                    owner.bind_groups.remove(&name.raw());
                    owner.context_state.retire_bind_group(name);
                }
                GlObjectKind::RasterPipeline => {
                    if let Some(pipeline) = owner.raster_pipelines.remove(&name.raw()) {
                        owner
                            .geometry_blocks
                            .retain(|key, _| key.vertex_array != pipeline.pipeline.vertex_array);
                        owner.context_state.event(StateEvent::VertexArrayRetired(
                            pipeline.pipeline.vertex_array,
                        ));
                        owner
                            .context_state
                            .event(StateEvent::ProgramRetired(pipeline.pipeline.program));
                        let _ = owner
                            .provider
                            .destroy_vertex_array(pipeline.pipeline.vertex_array);
                        let _ = owner.provider.destroy_program(pipeline.pipeline.program);
                    }
                }
                GlObjectKind::ComputePipeline => {
                    if let Some(program) = owner.compute_pipelines.remove(&name.raw()) {
                        owner
                            .context_state
                            .event(StateEvent::ProgramRetired(program));
                        let _ = owner.provider.destroy_program(program);
                    }
                }
            }
            Ok(())
        });
    }

    fn poll(&self) -> RhiResult<()> {
        self.dispatch("NativeProviderOwner::poll")
    }

    fn wait_idle(&self) -> RhiResult<()> {
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::wait_idle";
                owner.ready(OP)?;
                use glow::HasContext as _;
                // `finish` is intentionally reserved for explicit idle waits;
                // normal submissions use fences once their phase-B route exists.
                unsafe { owner.provider.gl.finish() };
                owner.provider.driver_error(OP).map_err(|e| map_gl(e, OP))
            })
            .map_err(worker_error)?
    }

    fn supports_presentation(&self, _: &PresentationTarget) -> RhiResult<bool> {
        self.worker
            .call(|owner| Ok(owner.context.supports_presentation()))
            .map_err(worker_error)?
    }

    fn presentation_capabilities(&self, _: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        self.worker
            .call(|owner| native_presentation_capabilities(owner))
            .map_err(worker_error)?
    }

    fn configure_presentation(
        &self,
        target: ObjectId,
        _: &PresentationConfiguration,
    ) -> RhiResult<crate::backend::gl::platform::GlPresentationLease> {
        self.worker
            .call(move |owner| {
                const OP: &str = "NativeProviderOwner::configure_presentation";
                owner.ready(OP)?;
                let _ = native_presentation_capabilities(owner)?;
                match owner.presentation.target {
                    Some(existing) if existing != target => return unsupported(OP),
                    Some(_) => {}
                    None => owner.presentation.target = Some(target),
                }
                if !owner.presentation.leases.is_empty() {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "the native GL drawable already has a configured presentation lease",
                    )
                    .at(OP));
                }
                let id = owner.presentation.next_lease;
                owner.presentation.next_lease = id.checked_add(1).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "native GL presentation lease space exhausted",
                    )
                    .at(OP)
                })?;
                owner.presentation.leases.insert(id, None);
                Ok(crate::backend::gl::platform::GlPresentationLease(id))
            })
            .map_err(worker_error)?
    }

    fn lease_capabilities(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
    ) -> RhiResult<PresentationTargetCapabilities> {
        self.worker
            .call(move |owner| {
                if !owner.presentation.leases.contains_key(&lease.0) {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "native GL presentation lease is no longer live",
                    ));
                }
                native_presentation_capabilities(owner)
            })
            .map_err(worker_error)?
    }

    fn reconfigure_or_register_waker(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        _: &PresentationConfiguration,
        _: &Waker,
    ) -> std::task::Poll<RhiResult<()>> {
        let result = self
            .worker
            .call(move |owner| {
                if !owner.presentation.leases.contains_key(&lease.0) {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "native GL presentation lease is no longer live",
                    ));
                }
                let _ = native_presentation_capabilities(owner)?;
                Ok(())
            })
            .map_err(worker_error)?;
        std::task::Poll::Ready(result)
    }

    fn try_acquire(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
    ) -> Result<Option<crate::backend::gl::platform::GlAcquiredFramebuffer>, AcquireError> {
        self.worker
            .call(move |owner| native_acquire_frame(owner, lease))
            .map_err(|error| {
                AcquireError::new(
                    crate::api::presentation::AcquireErrorKind::DeviceLost,
                    worker_error(error).to_string(),
                )
            })?
    }

    fn acquire_or_register_waker(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        _: &Waker,
    ) -> std::task::Poll<Result<crate::backend::gl::platform::GlAcquiredFramebuffer, AcquireError>>
    {
        match self.try_acquire(lease) {
            Ok(Some(frame)) => std::task::Poll::Ready(Ok(frame)),
            // A native WGL/EGL owner currently exposes no platform event
            // source into which this waker can be registered. Returning
            // Pending without retaining and waking it would strand the
            // future forever, so make the temporary absence explicit.
            Ok(None) => std::task::Poll::Ready(Err(AcquireError::new(
                crate::api::presentation::AcquireErrorKind::NotReady,
                "native GL drawable is temporarily unavailable; no acquire wake source is installed",
            ))),
            Err(error) => std::task::Poll::Ready(Err(error)),
        }
    }

    fn abandon(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        _: AcquiredFrameId,
    ) -> RhiResult<()> {
        self.worker
            // `ConfiguredPresentationBackend` has already authenticated this
            // opaque public id before calling us.  Its serial is intentionally
            // not exposed outside `presentation::frame`, so only the typed
            // present route below can additionally compare a native serial.
            .call(move |owner| native_end_frame(owner, lease, None, false))
            .map_err(worker_error)?
    }

    fn abandon_no_throw(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        _: AcquiredFrameId,
    ) {
        let _ = self
            .worker
            .call(move |owner| native_end_frame(owner, lease, None, false));
    }

    fn release_presentation(&self, lease: crate::backend::gl::platform::GlPresentationLease) {
        let _ = self.worker.call(move |owner| {
            let _ = native_end_frame(owner, lease, None, false);
            owner.presentation.leases.remove(&lease.0);
        });
    }

    fn present(
        &self,
        frame: crate::backend::gl::platform::GlAcquiredFramebuffer,
        receipt: PresentReceiptId,
    ) {
        let _ = self.worker.call(move |owner| {
            let state = match native_end_frame(owner, frame.lease, Some(frame.serial), true) {
                Ok(()) => PresentState::Accepted,
                Err(error) => PresentState::Failed(crate::api::presentation::PresentFailure::new(
                    error.to_string(),
                )),
            };
            owner.presentation.presents.insert(receipt, state);
        });
    }

    fn terminate_present(
        &self,
        frame: crate::backend::gl::platform::GlAcquiredFramebuffer,
        receipt: PresentReceiptId,
        state: PresentState,
    ) {
        let _ = self.worker.call(move |owner| {
            let _ = native_end_frame(owner, frame.lease, Some(frame.serial), false);
            owner.presentation.presents.insert(receipt, state);
        });
    }

    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        self.worker
            .call(move |owner| {
                owner
                    .presentation
                    .presents
                    .get(&receipt)
                    .cloned()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::InvalidUsage,
                            "native GL present receipt was never accepted",
                        )
                    })
            })
            .map_err(worker_error)?
    }

    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        _: &Waker,
    ) -> RhiResult<PresentState> {
        self.present_state(receipt)
    }
}

fn native_presentation_capabilities<C: NativePlatformContext>(
    owner: &mut NativeOwnedProvider<C>,
) -> RhiResult<PresentationTargetCapabilities> {
    const OP: &str = "NativeProviderOwner::presentation_capabilities";
    owner.ready(OP)?;
    if !owner.context.supports_presentation() {
        return unsupported(OP);
    }
    let current = owner.context.drawable_extent()?;
    Ok(PresentationTargetCapabilities::new(
        vec![crate::api::format::TextureFormat::Rgba8Unorm],
        vec![PresentMode::Automatic],
        PresentationExtentControl::HostManaged { current },
    ))
}

fn native_acquire_frame<C: NativePlatformContext>(
    owner: &mut NativeOwnedProvider<C>,
    lease: crate::backend::gl::platform::GlPresentationLease,
) -> Result<Option<crate::backend::gl::platform::GlAcquiredFramebuffer>, AcquireError> {
    const OP: &str = "NativeProviderOwner::acquire_presentation";
    if !owner.presentation.leases.contains_key(&lease.0) {
        return Err(AcquireError::new(
            crate::api::presentation::AcquireErrorKind::Outdated,
            "native GL presentation lease is no longer live",
        ));
    }
    if owner
        .presentation
        .leases
        .get(&lease.0)
        .and_then(|value| *value)
        .is_some()
    {
        return Err(AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "native GL presentation frame is already acquired",
        ));
    }
    owner.ready(OP).map_err(|error| {
        AcquireError::new(
            crate::api::presentation::AcquireErrorKind::DeviceLost,
            error.to_string(),
        )
    })?;
    let Some(extent) = owner.context.drawable_extent().map_err(|error| {
        AcquireError::new(
            crate::api::presentation::AcquireErrorKind::DeviceLost,
            error.to_string(),
        )
    })?
    else {
        return Ok(None);
    };
    let current = crate::backend::gl::api::GlSurfaceSize {
        width: extent.width,
        height: extent.height,
    };
    if owner.provider.surface_extent != Some(current) {
        owner.provider.resize_surface(current).map_err(|error| {
            AcquireError::new(
                crate::api::presentation::AcquireErrorKind::Outdated,
                format!("native GL drawable resize failed: {error:?}"),
            )
        })?;
    }
    let GlSurfaceAcquire::Lease(native_lease) =
        owner.provider.acquire_surface_image().map_err(|error| {
            AcquireError::new(
                crate::api::presentation::AcquireErrorKind::DeviceLost,
                format!("native GL acquire failed: {error:?}"),
            )
        })?
    else {
        return Ok(None);
    };
    let serial = owner.presentation.next_frame;
    owner.presentation.next_frame = serial.checked_add(1).ok_or_else(|| {
        AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "native GL frame serial exhausted",
        )
    })?;
    *owner
        .presentation
        .leases
        .get_mut(&lease.0)
        .expect("validated above") = Some((serial, native_lease));
    Ok(Some(crate::backend::gl::platform::GlAcquiredFramebuffer {
        serial,
        extent,
        suboptimal: false,
        lease,
        framebuffer: 0,
    }))
}

fn native_end_frame<C: NativePlatformContext>(
    owner: &mut NativeOwnedProvider<C>,
    lease: crate::backend::gl::platform::GlPresentationLease,
    expected_serial: Option<u64>,
    present: bool,
) -> RhiResult<()> {
    const OP: &str = "NativeProviderOwner::end_presentation_frame";
    owner.ready(OP)?;
    let (serial, native_lease) = owner
        .presentation
        .leases
        .get_mut(&lease.0)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "native GL presentation lease is no longer live",
            )
            .at(OP)
        })?
        .take()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "native GL frame is not acquired",
            )
            .at(OP)
        })?;
    if let Some(expected_serial) = expected_serial
        && serial != expected_serial
    {
        // Put the lease back: a stale attachment must not be able to consume
        // the newer frame that happens to occupy the same configured lease.
        *owner
            .presentation
            .leases
            .get_mut(&lease.0)
            .expect("lease was validated immediately above") = Some((serial, native_lease));
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "native GL frame serial does not match the currently acquired frame",
        )
        .at(OP));
    }
    if present {
        owner
            .provider
            .present_surface(native_lease)
            .map_err(|error| map_gl(error, OP))?;
        owner.context.present()?;
    } else {
        owner
            .provider
            .surface
            .consume(OP, native_lease)
            .map_err(|error| map_gl(error, OP))?;
    }
    Ok(())
}

impl NativeGlDriver {
    pub(crate) fn new(owner: Arc<dyn NativeGlOwner>) -> Self {
        Self {
            owner,
            loss_sink: Arc::new(Mutex::new(None)),
        }
    }

    fn report_context_loss(&self, info: DeviceLossInfo) {
        let sink = self
            .loss_sink
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if let Some(sink) = sink {
            sink.report_context_loss(info);
        }
    }

    /// Publishes a native owner route through the common v13 provider seam.
    /// Facts are an explicit input from the WGL/EGL discovery adapter: this
    /// bridge never invents capabilities from a context version string.
    pub(crate) fn adopt(
        instance: crate::api::identity::DeviceInstanceId,
        backend: crate::api::platform::BackendKind,
        name: String,
        facts: crate::api::capability::CapabilityFacts,
        owner: Arc<dyn NativeGlOwner>,
    ) -> RhiResult<crate::backend::gl::platform::GlProvider> {
        let driver: Arc<dyn GlExecutionDriver> = Arc::new(Self::new(owner));
        let context =
            crate::backend::gl::platform::GlAdoptedContext::new(backend, name, facts, driver)?;
        Ok(crate::backend::gl::platform::GlProvider::adopt(
            instance, context,
        ))
    }
}

impl GlExecutionDriver for NativeGlDriver {
    fn dispatch(&self, operation: &'static str) -> RhiResult<()> {
        self.owner.dispatch(operation)
    }
    fn install_loss_sink(&self, sink: Arc<dyn GlLossSink>) {
        *self.loss_sink.lock().unwrap_or_else(|p| p.into_inner()) = Some(sink);
    }
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<GlObjectName> {
        self.owner.create_buffer(descriptor)
    }
    fn create_texture(&self, descriptor: &TextureDescriptor) -> RhiResult<GlObjectName> {
        self.owner.create_texture(descriptor)
    }
    fn create_texture_view(
        &self,
        texture: GlTextureRef,
        descriptor: &TextureViewDescriptor,
    ) -> RhiResult<GlObjectName> {
        self.owner.create_texture_view(texture, descriptor)
    }
    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> RhiResult<GlObjectName> {
        self.owner.create_sampler(descriptor)
    }
    fn create_query_set(&self, descriptor: &QuerySetDescriptor) -> RhiResult<GlObjectName> {
        self.owner.create_query_set(descriptor)
    }
    fn create_shader(&self, artifact: &ShaderArtifact) -> RhiResult<GlObjectName> {
        self.owner.create_shader(artifact)
    }
    fn create_bind_group(&self, descriptor: &GlBindGroupPacket) -> RhiResult<GlObjectName> {
        self.owner.create_bind_group(descriptor)
    }
    fn create_compute_pipeline(
        &self,
        descriptor: GlComputePipelinePacket<'_>,
    ) -> RhiResult<GlObjectName> {
        self.owner.create_compute_pipeline(descriptor)
    }
    fn create_raster_pipeline(
        &self,
        descriptor: GlRasterPipelinePacket<'_>,
    ) -> RhiResult<GlObjectName> {
        self.owner.create_raster_pipeline(descriptor)
    }
    fn map_buffer(
        &self,
        buffer: GlBufferRef,
        mode: MapMode,
        range: BufferRange,
    ) -> RhiResult<Box<dyn MappingRequestBackend>> {
        self.owner.map_buffer(buffer, mode, range)
    }
    fn submit(&self, request: GlSubmissionPlan<'_>) -> RhiResult<SubmissionOutcome> {
        // Phase A is entirely CPU-side.  It must finish before forwarding the
        // plan to the owner worker: an `Err` from this method then proves zero
        // native work was accepted.  The owner may perform additional action
        // packet construction, but never emit a GL call before this gate.
        preflight_submission(&request)?;
        self.owner.submit(request)
    }
    fn completion(&self, serial: u64) -> CompletionState {
        let completion = self.owner.completion(serial);
        if let CompletionState::DeviceLost(info) = &completion {
            self.report_context_loss(info.clone());
        }
        completion
    }
    fn completion_or_register_waker(&self, serial: u64, waker: &Waker) -> CompletionState {
        let completion = self.owner.completion_or_register_waker(serial, waker);
        if let CompletionState::DeviceLost(info) = &completion {
            self.report_context_loss(info.clone());
        }
        completion
    }
    fn destroy(&self, kind: GlObjectKind, name: GlObjectName) {
        self.owner.destroy(kind, name)
    }
    fn poll(&self) -> RhiResult<()> {
        self.owner.poll()
    }
    fn wait_idle(&self) -> RhiResult<()> {
        self.owner.wait_idle()
    }
    fn supports_presentation(&self, target: &PresentationTarget) -> RhiResult<bool> {
        self.owner.supports_presentation(target)
    }
    fn presentation_capabilities(
        &self,
        target: ObjectId,
    ) -> RhiResult<PresentationTargetCapabilities> {
        self.owner.presentation_capabilities(target)
    }
    fn configure_presentation(
        &self,
        target: ObjectId,
        config: &PresentationConfiguration,
    ) -> RhiResult<crate::backend::gl::platform::GlPresentationLease> {
        self.owner.configure_presentation(target, config)
    }
    fn lease_capabilities(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
    ) -> RhiResult<PresentationTargetCapabilities> {
        self.owner.lease_capabilities(lease)
    }
    fn reconfigure_or_register_waker(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        config: &PresentationConfiguration,
        waker: &Waker,
    ) -> std::task::Poll<RhiResult<()>> {
        self.owner
            .reconfigure_or_register_waker(lease, config, waker)
    }
    fn try_acquire(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
    ) -> Result<Option<crate::backend::gl::platform::GlAcquiredFramebuffer>, AcquireError> {
        self.owner.try_acquire(lease)
    }
    fn acquire_or_register_waker(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        waker: &Waker,
    ) -> std::task::Poll<Result<crate::backend::gl::platform::GlAcquiredFramebuffer, AcquireError>>
    {
        self.owner.acquire_or_register_waker(lease, waker)
    }
    fn abandon(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        frame: AcquiredFrameId,
    ) -> RhiResult<()> {
        self.owner.abandon(lease, frame)
    }
    fn abandon_no_throw(
        &self,
        lease: crate::backend::gl::platform::GlPresentationLease,
        frame: AcquiredFrameId,
    ) {
        self.owner.abandon_no_throw(lease, frame)
    }
    fn release_presentation(&self, lease: crate::backend::gl::platform::GlPresentationLease) {
        self.owner.release_presentation(lease)
    }
    fn present(
        &self,
        frame: crate::backend::gl::platform::GlAcquiredFramebuffer,
        receipt: PresentReceiptId,
    ) {
        self.owner.present(frame, receipt)
    }
    fn terminate_present(
        &self,
        frame: crate::backend::gl::platform::GlAcquiredFramebuffer,
        receipt: PresentReceiptId,
        state: PresentState,
    ) {
        self.owner.terminate_present(frame, receipt, state)
    }
    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        self.owner.present_state(receipt)
    }
    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        waker: &Waker,
    ) -> RhiResult<PresentState> {
        self.owner.present_state_or_register_waker(receipt, waker)
    }
    fn device_lost(&self, info: &DeviceLossInfo) {
        self.owner.device_lost(info)
    }
}

fn preflight_submission(plan: &GlSubmissionPlan<'_>) -> RhiResult<()> {
    for batch in &plan.batches {
        for command in &batch.commands {
            crate::backend::gl::translate::preflight_recorded_command(command)?;
        }
    }
    Ok(())
}

fn native_gl_unlowered_shader_write(use_: &crate::api::command::ResourceUse) -> bool {
    use crate::api::command::{AccessMask, ResourceUse};
    match use_ {
        ResourceUse::Buffer(value) => value.access.contains(AccessMask::SHADER_WRITE),
        ResourceUse::Texture(value) => value.access.contains(AccessMask::SHADER_WRITE),
        ResourceUse::Frame(_) | ResourceUse::AccelerationStructure(_) => false,
    }
}

/// Extracts every readback sink before Phase B consumes its action vector.
/// If a later native call fails, even a readback action not reached yet belongs
/// to an accepted-but-terminal submission and must not remain indefinitely
/// pending.
fn native_readback_tickets(
    actions: &[NativePhaseBAction],
) -> Vec<crate::api::resource::transfer::ReadbackTicket> {
    actions
        .iter()
        .filter_map(|action| match action {
            NativePhaseBAction::ReadBuffer { ticket, .. }
            | NativePhaseBAction::ReadTexture { ticket, .. } => Some(ticket.clone()),
            _ => None,
        })
        .collect()
}

fn worker_error(error: WorkerStartupError) -> RhiError {
    RhiError::new(
        RhiErrorKind::DeviceLost,
        format!("native GL owner thread is unavailable: {error:?}"),
    )
}

fn map_gl(error: crate::backend::gl::api::GlError, operation: &'static str) -> RhiError {
    use crate::backend::gl::api::{GlContextLifecycle, GlError};
    let kind = match error {
        GlError::ContextLost { .. }
        | GlError::Disposed { .. }
        | GlError::Poisoned { .. }
        | GlError::InvalidLifecycle {
            lifecycle:
                GlContextLifecycle::Lost
                | GlContextLifecycle::Restoring
                | GlContextLifecycle::Poisoned
                | GlContextLifecycle::Disposed,
            ..
        } => RhiErrorKind::DeviceLost,
        GlError::Unsupported { .. } => RhiErrorKind::Unsupported,
        GlError::OutOfMemory { .. } => RhiErrorKind::OutOfMemory,
        GlError::WrongContext { .. } | GlError::StaleObject { .. } => RhiErrorKind::WrongDevice,
        _ => RhiErrorKind::BackendFailure,
    };
    RhiError::new(kind, format!("{operation}: {error:?}"))
}

fn validate_bind_packet<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    packet: &GlBindGroupPacket,
    operation: &'static str,
) -> RhiResult<()> {
    use crate::backend::gl::platform::GlBindingResource;
    for entry in &packet.entries {
        match &entry.resource {
            GlBindingResource::Buffer { buffer, .. } => {
                owner.buffer_id(buffer.name, operation)?;
            }
            GlBindingResource::Texture(view) => {
                let texture = owner.views.get(&view.name.raw()).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "native GL texture view is not live",
                    )
                    .at(operation)
                })?;
                owner
                    .provider
                    .texture(operation, texture.texture)
                    .map_err(|e| map_gl(e, operation))?;
            }
            GlBindingResource::Sampler(sampler) => {
                let id = SamplerId::new(
                    owner.provider.context_stamp(),
                    NativeOwnedProvider::<C>::slot(sampler.name, operation)?,
                    0,
                );
                owner
                    .provider
                    .sampler(operation, id)
                    .map_err(|e| map_gl(e, operation))?;
            }
            GlBindingResource::BufferArray(values) => {
                for (buffer, _, _) in values {
                    owner.buffer_id(buffer.name, operation)?;
                }
            }
            GlBindingResource::TextureArray(values) => {
                for view in values {
                    let texture = owner.views.get(&view.name.raw()).ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "native GL texture view is not live",
                        )
                        .at(operation)
                    })?;
                    owner
                        .provider
                        .texture(operation, texture.texture)
                        .map_err(|e| map_gl(e, operation))?;
                }
            }
            GlBindingResource::SamplerArray(values) => {
                for sampler in values {
                    let id = SamplerId::new(
                        owner.provider.context_stamp(),
                        NativeOwnedProvider::<C>::slot(sampler.name, operation)?,
                        0,
                    );
                    owner
                        .provider
                        .sampler(operation, id)
                        .map_err(|e| map_gl(e, operation))?;
                }
            }
        }
    }
    Ok(())
}

fn texture_region<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    name: GlObjectName,
    layers: crate::api::resource::TextureSubresourceLayers,
    origin: crate::api::resource::Origin3d,
    extent: crate::api::resource::Extent3d,
    operation: &'static str,
) -> RhiResult<crate::backend::gl::api::GlTextureRegion> {
    use crate::api::resource::TextureAspect;
    use crate::backend::gl::api::{
        GlExtent3d, GlTextureAspect, GlTextureRegion, GlTextureSubresource,
    };
    let aspect = match layers.aspect {
        TextureAspect::Color => GlTextureAspect::Color,
        TextureAspect::Depth => GlTextureAspect::DepthOnly,
        TextureAspect::Stencil => GlTextureAspect::StencilOnly,
        TextureAspect::Plane0 | TextureAspect::Plane1 | TextureAspect::Plane2 => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "native GL has no multi-planar texture copy route",
            )
            .at(operation));
        }
    };
    Ok(GlTextureRegion {
        subresource: GlTextureSubresource {
            texture: owner.texture_id(name, operation)?,
            aspect,
            mip_level: layers.mip_level,
            base_layer: layers.base_layer,
            layer_count: layers.layer_count,
        },
        origin: [origin.x, origin.y, origin.z],
        extent: GlExtent3d {
            width: extent.width,
            height: extent.height,
            depth_or_layers: extent.depth,
        },
    })
}

fn attachment_view<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    name: GlObjectName,
    operation: &'static str,
) -> RhiResult<crate::backend::gl::api::GlTextureView> {
    let view = owner.views.get(&name.raw()).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::WrongDevice,
            "native GL texture-view carrier is not live",
        )
        .at(operation)
    })?;
    let (_, desc) = owner
        .provider
        .texture(operation, view.texture)
        .map_err(|e| map_gl(e, operation))?;
    let extent = desc.mip_extent(view.mip_level).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "native GL texture-view mip is outside its allocation",
        )
        .at(operation)
    })?;
    Ok(crate::backend::gl::api::GlTextureView {
        target: crate::backend::gl::api::GlAttachmentTarget::Texture(view.texture),
        format: view.format,
        mip_level: view.mip_level,
        array_layer: view.base_layer,
        layer_count: view.layer_count,
        width: extent.width,
        height: extent.height,
        sample_count: desc.sample_count,
    })
}

/// Maps only bindable GL-family texture targets.  The common vocabulary keeps
/// 1D for validation, but this executable native route has no proved 1D
/// allocation/binding lowering and must refuse it rather than bind it as 2D.
fn binding_target(
    dimension: crate::backend::gl::api::GlTextureDimension,
) -> RhiResult<crate::backend::gl::api::GlTextureTarget> {
    use crate::backend::gl::api::{GlTextureDimension as Dimension, GlTextureTarget as Target};
    match dimension {
        Dimension::D2 => Ok(Target::D2),
        Dimension::D2Array => Ok(Target::D2Array),
        Dimension::Cube => Ok(Target::Cube),
        Dimension::D3 => Ok(Target::D3),
        Dimension::D1 => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "native GL binding route has no verified 1D texture target",
        )
        .at("GL::create_texture_view")),
    }
}

fn unsupported<T>(operation: &'static str) -> RhiResult<T> {
    Err(RhiError::new(
        RhiErrorKind::Unsupported,
        "native WGL/EGL owner has no verified lowering for this operation",
    )
    .at(operation))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::api::error::RhiResult;
    use crate::backend::gl::platform::GlExecutionDriver;

    use super::{NativeGlDriver, NativeGlOwner};

    struct Owner(AtomicUsize);
    impl NativeGlOwner for Owner {
        fn dispatch(&self, _: &'static str) -> RhiResult<()> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn dispatch_is_synchronously_forwarded_to_owner() {
        let owner = Arc::new(Owner(AtomicUsize::new(0)));
        let driver = NativeGlDriver::new(owner.clone());
        driver.dispatch("test-native-dispatch").unwrap();
        assert_eq!(owner.0.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn absent_lowering_is_structured_unsupported_not_success() {
        let driver = NativeGlDriver::new(Arc::new(Owner(AtomicUsize::new(0))));
        assert!(
            matches!(driver.wait_idle(), Err(error) if error.kind() == crate::api::error::RhiErrorKind::Unsupported)
        );
    }
}
