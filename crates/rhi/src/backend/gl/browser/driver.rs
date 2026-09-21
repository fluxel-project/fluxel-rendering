//! Thread-safe v13 driver handle for one thread-affine WebGL2 executor.
//!
//! `WebGl2RenderingContext` and every object it creates are JavaScript values:
//! they are deliberately neither `Send` nor `Sync`.  The platform driver seam
//! is `Send + Sync`, because the portable `Device` is shareable, so this type
//! stores only a Fluxel-owned registration number and an owner-thread identity.
//! The actual WebGL executor stays in a thread-local table on its browser owner
//! thread.  A caller from another thread receives a structured refusal rather
//! than an unsound `unsafe impl Send` for a JS handle.

use core::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};
use std::thread::ThreadId;

use wasm_bindgen::JsCast;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::DeviceLossInfo;
use crate::api::presentation::{
    AcquireError, AcquireErrorKind, AcquiredFrameId, Extent2d, PresentMode, PresentReceiptId,
    PresentState, PresentationConfiguration, PresentationExtentControl,
    PresentationTargetCapabilities,
};
use crate::api::submission::{CompletionFailure, CompletionState};
use crate::backend::gl::api::{
    BufferId, ContextStamp, GlBindingApi, GlContextLifecycle, GlFamilyApi, GlFramebufferApi,
    GlQueryObjectsApi, GlRasterCommandApi, GlResourceApi, GlSamplerApi, GlShaderApi,
    GlSurfaceAcquire, GlSurfaceLease, GlSurfacePresentationApi, GlSyncApi, GlVertexApi, QueryId,
    SamplerId, ShaderId, TextureId,
};
use crate::backend::gl::platform::{
    GlAcquiredFramebuffer, GlBindGroupPacket, GlComputePipelinePacket, GlExecutionDriver,
    GlLossSink, GlObjectKind, GlObjectName, GlPresentationLease, GlRasterPipelinePacket,
    GlTextureRef,
};
use crate::backend::gl::state::{
    BoundGroupPacket, CanonicalBlockId, ContextState, DerivedCacheKey, DerivedCacheKind,
    ExecutionMode, PassPacket, RasterPipelineBlockInterner,
    RasterPipelinePacket as StateRasterPipelinePacket, ResourceRef, StateEvent,
};
use crate::backend::gl::translate;

use super::discovery::WebGl2BrowserDiscovery;
use super::v13_submit::{
    BrowserV13ActionPacket, BrowserV13CopyPacket, BrowserV13ObjectResolver, BrowserV13PhaseBAction,
    BrowserV13QueryPacket, BrowserV13ReadbackPacket, BrowserV13UploadPacket,
};

/// Per-context metadata which must remain on the WebGL owner thread along
/// with the JS executor.  The maps retain typed, generation-safe browser ids;
/// the v13 `GlObjectName` is merely `slot + 1` transport encoding.
pub(super) struct BrowserDriverState {
    executor: WebGl2BrowserDiscovery,
    /// A GL format cannot be reverse-mapped without losing the original
    /// portable color-space fact. Keep it beside the typed texture table for
    /// virtual-view validation.
    texture_formats: BTreeMap<u32, crate::api::format::TextureFormat>,
    /// WebGL has no separately allocated texture-view object.  This table is
    /// nevertheless not optional: it retains the source generation and the
    /// portable base format so a stale/destroyed texture cannot be revived by a
    /// virtual view carrier.
    views: BTreeMap<u32, BrowserTextureView>,
    bind_groups: BTreeMap<u32, crate::backend::gl::platform::GlBindGroupPacket>,
    /// WebGL exposes individual query objects while the portable API owns a
    /// query set.  Keep that fan-out behind one virtual object name so its
    /// backing neither leaks into the public model nor aliases a native slot.
    query_sets: BTreeMap<u32, Vec<QueryId>>,
    /// Stable virtual pipeline identity. The values are immutable creation
    /// packets, not a second state cache: the shared GL state authority owns
    /// desired/applied diffing when a draw later selects one of these records.
    raster_pipelines: BTreeMap<u32, BrowserRasterPipeline>,
    geometry_vaos: HashMap<BrowserGeometryKey, BrowserGeometryVao>,
    /// The sole cache/diff authority for this WebGL context.  Creation tables
    /// above retain immutable backing metadata only; they must never become a
    /// second "current GL state" cache.
    context_state: ContextState,
    /// Namespace for non-pipeline state packets (FBO/query/geometry). Raster
    /// pipeline blocks have their own structural interner below.
    next_canonical_block: u64,
    pipeline_blocks: RasterPipelineBlockInterner,
    next_virtual: u32,
    next_completion: u64,
    completions: BTreeMap<u64, BrowserCompletion>,
    next_readback_sink: Cell<u64>,
    readback_sinks: RefCell<BTreeMap<u64, crate::api::resource::transfer::ReadbackTicket>>,
    pending_readbacks: Vec<BrowserPendingReadback>,
    presentation: BrowserPresentationState,
}

#[derive(Clone, Copy)]
struct BrowserTextureView {
    texture: TextureId,
    target: crate::backend::gl::api::GlTextureTarget,
    base_format: crate::api::format::TextureFormat,
}

#[derive(Clone)]
struct BrowserRasterPipeline {
    program: crate::backend::gl::api::ProgramId,
    vertex_array: crate::backend::gl::api::VertexArrayId,
    state: crate::backend::gl::api::GlRasterState,
    state_packet: StateRasterPipelinePacket,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BrowserGeometryKey {
    pipeline: GlObjectName,
    /// The carrier follows the public pipeline lifetime; this typed identity
    /// adds the context epoch and allocation generation for the VAO layout.
    vertex_array: crate::backend::gl::api::VertexArrayId,
    vertices: Vec<crate::backend::gl::api::GlVertexBufferBinding>,
    index: Option<crate::backend::gl::api::GlIndexBinding>,
}
#[derive(Clone, Copy)]
struct BrowserGeometryVao {
    vertex_array: crate::backend::gl::api::VertexArrayId,
    canonical: CanonicalBlockId,
}
#[derive(Clone)]
struct BrowserBoundGroup {
    packet: BoundGroupPacket,
    bindings: GlBindGroupPacket,
}

/// Fully resolved basic raster work.  It deliberately contains no public RHI
/// handle and no JS value: Phase A copies the immutable browser ids and scalar
/// draw command, while Phase B reaches the owner-thread executor only through
/// `BrowserDriverState`'s single state authority.
struct BrowserRasterDrawAction {
    pipeline: crate::backend::gl::api::GlRasterPipeline,
    packet: StateRasterPipelinePacket,
    /// Dynamic values are applied through the typed pipeline install today.
    /// They must invalidate the otherwise immutable pipeline packet so a
    /// second draw with the same pipeline cannot elide the changed values.
    has_dynamic_state: bool,
    draw: crate::backend::gl::api::GlDrawCommand,
    geometry: Vec<crate::backend::gl::api::GlVertexBufferBinding>,
    index: Option<crate::backend::gl::api::GlIndexBinding>,
    geometry_key: BrowserGeometryKey,
    bind_groups: Vec<BrowserBoundGroup>,
}

struct BrowserRasterBeginAction {
    descriptor: crate::backend::gl::api::GlRenderPassDescriptor,
    /// Offscreen descriptors become real browser framebuffer objects only on
    /// the owner thread.  Keeping the structural descriptor in the action
    /// makes Phase A entirely JS-free while allowing Phase B to cache the
    /// typed executor object by its immutable attachment facts.
    framebuffer: Option<crate::backend::gl::api::GlFramebufferDescriptor>,
    pass: PassPacket,
}
struct BrowserRasterEndAction;

fn bind_group_dependencies(
    packet: &GlBindGroupPacket,
    views: &BTreeMap<u32, BrowserTextureView>,
    executor: &WebGl2BrowserDiscovery,
) -> RhiResult<BTreeSet<ResourceRef>> {
    fn buffer(
        reference: crate::backend::gl::platform::GlBufferRef,
        executor: &WebGl2BrowserDiscovery,
    ) -> RhiResult<ResourceRef> {
        let slot = reference.name.raw().checked_sub(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL buffer carrier was zero",
            )
        })?;
        let entry = executor.buffers.get(&slot).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL buffer backing is not live",
            )
        })?;
        Ok(ResourceRef::Buffer(BufferId::new(
            executor.context_stamp(),
            slot,
            entry.generation,
        )))
    }
    fn sampler(
        reference: crate::backend::gl::platform::GlSamplerRef,
        executor: &WebGl2BrowserDiscovery,
    ) -> RhiResult<ResourceRef> {
        let slot = reference.name.raw().checked_sub(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL sampler carrier was zero",
            )
        })?;
        let entry = executor.samplers.get(&slot).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL sampler backing is not live",
            )
        })?;
        Ok(ResourceRef::Sampler(SamplerId::new(
            executor.context_stamp(),
            slot,
            entry.generation,
        )))
    }
    let mut dependencies = BTreeSet::new();
    for entry in &packet.entries {
        match &entry.resource {
            crate::backend::gl::platform::GlBindingResource::Buffer { buffer: value, .. } => {
                dependencies.insert(buffer(*value, executor)?);
            }
            crate::backend::gl::platform::GlBindingResource::BufferArray(values) => {
                for (value, _, _) in values {
                    dependencies.insert(buffer(*value, executor)?);
                }
            }
            crate::backend::gl::platform::GlBindingResource::Texture(view) => {
                let texture = views
                    .get(&view.name.raw())
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::WrongDevice,
                            "WebGL texture view backing is not live",
                        )
                    })?
                    .texture;
                dependencies.insert(ResourceRef::Texture(texture));
            }
            crate::backend::gl::platform::GlBindingResource::TextureArray(values) => {
                for view in values {
                    let texture = views
                        .get(&view.name.raw())
                        .ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::WrongDevice,
                                "WebGL texture view backing is not live",
                            )
                        })?
                        .texture;
                    dependencies.insert(ResourceRef::Texture(texture));
                }
            }
            crate::backend::gl::platform::GlBindingResource::Sampler(value) => {
                dependencies.insert(sampler(*value, executor)?);
            }
            crate::backend::gl::platform::GlBindingResource::SamplerArray(values) => {
                for value in values {
                    dependencies.insert(sampler(*value, executor)?);
                }
            }
        }
    }
    Ok(dependencies)
}

impl BrowserV13PhaseBAction for BrowserRasterBeginAction {
    fn execute(self: Box<Self>, owner: &mut BrowserDriverState) -> RhiResult<()> {
        let mut descriptor = self.descriptor.clone();
        if let Some(framebuffer) = self.framebuffer.as_ref() {
            let id = owner.framebuffer(framebuffer)?;
            descriptor.target = crate::backend::gl::api::GlRenderTarget::Offscreen(id);
        }
        let _ = owner.context_state.prepare_pass(self.pass);
        if let Err(error) = owner.executor.begin_render_pass(&descriptor) {
            owner.context_state.pass_failed();
            return Err(map_gl_error(error, "begin WebGL2 raster pass"));
        }
        owner.context_state.commit_pass(self.pass);
        Ok(())
    }
}
impl BrowserV13PhaseBAction for BrowserRasterEndAction {
    fn execute(self: Box<Self>, owner: &mut BrowserDriverState) -> RhiResult<()> {
        if let Err(error) = owner.executor.end_render_pass() {
            owner.context_state.pass_failed();
            return Err(map_gl_error(error, "end WebGL2 raster pass"));
        }
        owner.context_state.end_pass();
        Ok(())
    }
}

impl BrowserV13PhaseBAction for BrowserRasterDrawAction {
    fn execute(self: Box<Self>, owner: &mut BrowserDriverState) -> RhiResult<()> {
        if self.has_dynamic_state {
            owner.context_state.event(StateEvent::DomainFailed(
                crate::backend::gl::state::StateDomain::RasterPipeline,
            ));
        }
        let diff = owner
            .context_state
            .prepare_pipeline(self.packet)
            .map_err(|error| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    format!("WebGL raster pipeline state was not registered: {error:?}"),
                )
            })?;
        // The portable pipeline remains one object, but GL state does not: the
        // structural diff is lowered leaf-by-leaf, so switching only blend does
        // not rebind program, viewport/scissor, depth/stencil, or coverage.
        if !diff.is_empty() {
            if let Err(error) = owner
                .executor
                .apply_raster_pipeline_diff(&self.pipeline, diff)
            {
                owner.context_state.pipeline_failed();
                return Err(map_gl_error(error, "install WebGL2 raster pipeline"));
            }
            owner.context_state.commit_pipeline(self.packet);
        }
        owner.stage_bind_groups(&self.bind_groups);
        let flush = owner.context_state.binding_flush();
        if !flush.is_empty() {
            if let Err(error) = owner.flush_bindings(&flush) {
                owner.context_state.event(StateEvent::DomainFailed(
                    crate::backend::gl::state::StateDomain::Bindings,
                ));
                return Err(error);
            }
            owner.context_state.acknowledge_bindings(&flush);
        }
        let geometry = owner.geometry_vao(&self.geometry_key, self.pipeline.vertex_array)?;
        if owner.context_state.prepare_geometry(geometry.canonical) {
            if let Err(error) =
                owner
                    .executor
                    .bind_vertex_array(geometry.vertex_array, &self.geometry, self.index)
            {
                owner.context_state.geometry_failed();
                return Err(map_gl_error(error, "bind WebGL2 raster geometry"));
            }
            owner.context_state.commit_geometry(geometry.canonical);
        }
        if let Err(error) = owner.executor.draw_raster(self.draw) {
            owner.context_state.pipeline_failed();
            return Err(map_gl_error(error, "issue WebGL2 raster draw"));
        }
        Ok(())
    }
}

/// Browser fence state remains entirely on the owner thread. A JS `WebGlSync`
/// is never placed behind an `Arc`; only callers' Rust wakers leave this record.
enum BrowserCompletion {
    Pending {
        fence: crate::backend::gl::api::GlFenceLease,
        wakers: Vec<Waker>,
        readbacks: Vec<BrowserPendingReadback>,
    },
    Complete,
    DeviceLost(DeviceLossInfo),
    Failed(CompletionFailure),
}
struct BrowserPendingReadback {
    ticket: crate::api::resource::transfer::ReadbackTicket,
    bytes: Vec<u8>,
    layout: Option<crate::api::resource::transfer::ReadbackTexelLayout>,
}

struct BrowserPresentationLease {
    target: ObjectId,
    frame: Option<(u64, GlSurfaceLease)>,
    acquire_wakers: Vec<Waker>,
    reconfigure_wakers: Vec<Waker>,
}

struct BrowserPresentationState {
    next_lease: u64,
    next_frame: u64,
    leases: BTreeMap<u64, BrowserPresentationLease>,
    presents: std::collections::HashMap<PresentReceiptId, PresentState>,
    present_wakers: std::collections::HashMap<PresentReceiptId, Vec<Waker>>,
}

impl BrowserPresentationState {
    fn new() -> Self {
        Self {
            next_lease: 1,
            next_frame: 1,
            leases: BTreeMap::new(),
            presents: std::collections::HashMap::new(),
            present_wakers: std::collections::HashMap::new(),
        }
    }
}

impl BrowserDriverState {
    fn framebuffer(
        &mut self,
        descriptor: &crate::backend::gl::api::GlFramebufferDescriptor,
    ) -> RhiResult<crate::backend::gl::api::FramebufferId> {
        // The executor retains generation-safe objects; descriptors are only a
        // cache key, never a second ownership model.
        if let Some((slot, entry)) = self
            .executor
            .framebuffers
            .iter()
            .find(|(_, entry)| entry.descriptor == *descriptor)
        {
            return Ok(crate::backend::gl::api::FramebufferId::new(
                self.executor.context_stamp(),
                *slot,
                entry.generation,
            ));
        }
        self.executor
            .create_framebuffer(descriptor)
            .map_err(|error| map_gl_error(error, "create WebGL2 raster framebuffer"))
    }

    fn attachment_view(
        &self,
        view: &crate::api::resource::view::TextureView,
    ) -> RhiResult<crate::backend::gl::api::GlTextureView> {
        use crate::backend::gl::api::{GlAttachmentTarget, GlTextureView};
        let name = crate::backend::gl::platform::GlDevice::view_ref(view)?.name;
        let virtual_view = self.views.get(&name.raw()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL texture-view backing is not live",
            )
        })?;
        let texture = self
            .executor
            .textures
            .get(&virtual_view.texture.slot)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "WebGL attachment texture backing is not live",
                )
            })?;
        if texture.generation != virtual_view.texture.generation
            || virtual_view.texture.context != self.executor.context_stamp()
        {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL attachment texture belongs to a stale context",
            ));
        }
        let extent = view.extent();
        Ok(GlTextureView {
            target: GlAttachmentTarget::Texture(virtual_view.texture),
            format: translate::texture_format(view.format())?,
            mip_level: view.descriptor().base_mip,
            array_layer: view.descriptor().base_layer,
            layer_count: view.layer_count(),
            width: extent.width,
            height: extent.height,
            sample_count: view.sample_count(),
        })
    }

    fn register_readback_sink(
        &self,
        ticket: &crate::api::resource::transfer::ReadbackTicket,
    ) -> RhiResult<super::v13_submit::BrowserReadbackSinkId> {
        let value = self.next_readback_sink.get();
        self.next_readback_sink
            .set(value.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL readback sink namespace exhausted",
                )
            })?);
        self.readback_sinks
            .borrow_mut()
            .insert(value, ticket.clone());
        Ok(super::v13_submit::BrowserReadbackSinkId(value))
    }
    fn fail_readbacks(&mut self, status: crate::api::resource::transfer::ReadbackStatus) {
        for ticket in std::mem::take(self.readback_sinks.get_mut()).into_values() {
            ticket.set_status(status);
        }
        for pending in self.pending_readbacks.drain(..) {
            pending.ticket.set_status(status);
        }
        for completion in self.completions.values_mut() {
            if let BrowserCompletion::Pending { readbacks, .. } = completion {
                for pending in readbacks.drain(..) {
                    pending.ticket.set_status(status);
                }
            }
        }
    }
    fn fail_current_readbacks(&mut self, status: crate::api::resource::transfer::ReadbackStatus) {
        for ticket in std::mem::take(self.readback_sinks.get_mut()).into_values() {
            ticket.set_status(status);
        }
        for pending in self.pending_readbacks.drain(..) {
            pending.ticket.set_status(status);
        }
    }
    pub(super) fn execute_v13_packet(&mut self, packet: BrowserV13ActionPacket) -> RhiResult<()> {
        use crate::backend::gl::api::{
            GlCopyDomainApi, GlElapsedQueryApi, GlOcclusionQueryApi, GlTimestampQueryApi,
        };
        let is_query = matches!(&packet, BrowserV13ActionPacket::Query(_));
        let result = match packet {
            BrowserV13ActionPacket::Copy(BrowserV13CopyPacket::Buffer {
                source,
                destination,
            }) => self.executor.copy_buffer_range(source, destination),
            BrowserV13ActionPacket::Copy(BrowserV13CopyPacket::Texture {
                source,
                destination,
            }) => self.executor.copy_texture_region(source, destination),
            BrowserV13ActionPacket::Upload(BrowserV13UploadPacket::Buffer {
                destination,
                bytes,
            }) => self.executor.upload_buffer(destination, &bytes),
            BrowserV13ActionPacket::Upload(BrowserV13UploadPacket::Texture {
                destination,
                layout,
                bytes,
            }) => self.executor.upload_texture(destination, layout, &bytes),
            BrowserV13ActionPacket::Readback(BrowserV13ReadbackPacket::Buffer { source, sink }) => {
                let bytes = self
                    .executor
                    .read_buffer(source)
                    .map_err(|error| map_gl_error(error, "read WebGL buffer"))?;
                let ticket = self
                    .readback_sinks
                    .get_mut()
                    .remove(&sink.0)
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::BackendFailure,
                            "WebGL readback sink was not registered",
                        )
                    })?;
                self.pending_readbacks.push(BrowserPendingReadback {
                    ticket,
                    bytes,
                    layout: None,
                });
                return Ok(());
            }
            BrowserV13ActionPacket::Readback(BrowserV13ReadbackPacket::Texture {
                source,
                layout,
                sink,
            }) => {
                let readback = self
                    .executor
                    .read_texture(source, layout)
                    .map_err(|error| map_gl_error(error, "read WebGL texture"))?;
                let ticket = self
                    .readback_sinks
                    .get_mut()
                    .remove(&sink.0)
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::BackendFailure,
                            "WebGL readback sink was not registered",
                        )
                    })?;
                let layout = crate::api::resource::transfer::ReadbackTexelLayout {
                    bytes_per_row: readback.layout.bytes_per_row,
                    rows_per_image: readback.layout.rows_per_image,
                    total_size: readback.bytes.len() as u64,
                };
                self.pending_readbacks.push(BrowserPendingReadback {
                    ticket,
                    bytes: readback.bytes,
                    layout: Some(layout),
                });
                return Ok(());
            }
            BrowserV13ActionPacket::Query(BrowserV13QueryPacket::BeginOcclusion(query)) => {
                self.executor.begin_occlusion_query(query)
            }
            BrowserV13ActionPacket::Query(BrowserV13QueryPacket::EndOcclusion) => {
                self.executor.end_occlusion_query()
            }
            BrowserV13ActionPacket::Query(BrowserV13QueryPacket::BeginElapsed(query)) => {
                self.executor.begin_elapsed_query(query)
            }
            BrowserV13ActionPacket::Query(BrowserV13QueryPacket::EndElapsed) => {
                self.executor.end_elapsed_query()
            }
            BrowserV13ActionPacket::Query(BrowserV13QueryPacket::Timestamp(query)) => {
                self.executor.query_timestamp(query)
            }
        };
        result.map_err(|error| {
            // Transfer/query providers bind internal targets and may have
            // observed an error after a partial state change.  There is no
            // successful packet to commit here, so force the next owner action
            // to re-establish the affected domain.
            self.context_state
                .event(StateEvent::DomainFailed(if is_query {
                    crate::backend::gl::state::StateDomain::Query
                } else {
                    crate::backend::gl::state::StateDomain::PixelTransferCopy
                }));
            map_gl_error(error, "execute WebGL2 v13 packet")
        })
    }
    fn new(executor: WebGl2BrowserDiscovery) -> Self {
        Self {
            executor,
            texture_formats: BTreeMap::new(),
            views: BTreeMap::new(),
            bind_groups: BTreeMap::new(),
            query_sets: BTreeMap::new(),
            raster_pipelines: BTreeMap::new(),
            geometry_vaos: HashMap::new(),
            context_state: ContextState::new(ExecutionMode::Optimized),
            next_canonical_block: 1,
            pipeline_blocks: RasterPipelineBlockInterner::new(),
            next_virtual: 0x8000_0000,
            next_completion: 1,
            completions: BTreeMap::new(),
            next_readback_sink: Cell::new(1),
            readback_sinks: RefCell::new(BTreeMap::new()),
            pending_readbacks: Vec::new(),
            presentation: BrowserPresentationState::new(),
        }
    }

    fn allocate_canonical_block(&mut self) -> RhiResult<CanonicalBlockId> {
        let value = self.next_canonical_block;
        self.next_canonical_block = value.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL canonical state-block namespace exhausted",
            )
        })?;
        Ok(CanonicalBlockId::new(value))
    }
    fn query_set_query(&self, set: &crate::api::query::QuerySet, index: u32) -> RhiResult<QueryId> {
        let name = crate::backend::gl::platform::GlDevice::query_set_ref(set)?.name;
        let queries = self.query_sets.get(&name.raw()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL query-set backing is not live on this context",
            )
        })?;
        let index = usize::try_from(index).map_err(|_| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "WebGL query-set index exceeds addressable range",
            )
        })?;
        let query = *queries.get(index).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "WebGL query-set index is outside its descriptor count",
            )
        })?;
        // The carrier can survive a native retirement only until its logical
        // owner is dropped; reject it rather than letting a reused slot become
        // a different query set's measurement.
        let entry = self.executor.queries.get(&query.slot).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL query backing is not live on this context",
            )
        })?;
        if entry.generation != query.generation || query.context != self.executor.context_stamp() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL query backing belongs to a stale context generation",
            ));
        }
        Ok(query)
    }
    fn geometry_vao(
        &mut self,
        key: &BrowserGeometryKey,
        source: crate::backend::gl::api::VertexArrayId,
    ) -> RhiResult<BrowserGeometryVao> {
        if let Some(value) = self.geometry_vaos.get(key) {
            return Ok(BrowserGeometryVao {
                vertex_array: value.vertex_array,
                canonical: value.canonical,
            });
        }
        let layout = self
            .executor
            .vertex_array("create cached WebGL geometry", source)
            .map_err(|error| map_gl_error(error, "inspect WebGL geometry layout"))?
            .layout
            .clone();
        let vertex_array = self
            .executor
            .create_vertex_array(&layout)
            .map_err(|error| map_gl_error(error, "create cached WebGL geometry"))?;
        let canonical = self.allocate_canonical_block()?;
        let value = BrowserGeometryVao {
            vertex_array,
            canonical,
        };
        let mut dependencies = BTreeSet::new();
        dependencies.insert(ResourceRef::VertexArray(source));
        for binding in &key.vertices {
            dependencies.insert(ResourceRef::Buffer(binding.buffer));
        }
        if let Some(index) = key.index {
            dependencies.insert(ResourceRef::Buffer(index.buffer));
        }
        self.context_state.register_derived(
            DerivedCacheKey {
                kind: DerivedCacheKind::VertexArray,
                canonical,
            },
            dependencies,
        );
        self.geometry_vaos.insert(key.clone(), value);
        Ok(value)
    }

    fn stage_bind_groups(&mut self, groups: &[BrowserBoundGroup]) {
        for group in groups {
            self.context_state.stage_bind_group(group.packet.clone());
        }
    }

    fn flush_bindings(&mut self, flush: &crate::backend::gl::state::BindingFlush) -> RhiResult<()> {
        for group in &flush.groups {
            let packet = self
                .bind_groups
                .get(&group.name.raw())
                .cloned()
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "WebGL bind-group backing is no longer live",
                    )
                })?;
            let mut dynamic_offsets = group.dynamic_offsets.iter();
            for entry in &packet.entries {
                let slot = entry.slot;
                match &entry.resource {
                    crate::backend::gl::platform::GlBindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    } => {
                        let offset = offset
                            .checked_add(u64::from(*dynamic_offsets.next().unwrap_or(&0)))
                            .ok_or_else(|| {
                                RhiError::new(
                                    RhiErrorKind::InvalidUsage,
                                    "WebGL dynamic buffer offset overflow",
                                )
                            })?;
                        let offset = u32::try_from(offset).map_err(|_| {
                            RhiError::new(
                                RhiErrorKind::Unsupported,
                                "WebGL uniform offset exceeds u32",
                            )
                        })?;
                        let size = u32::try_from(*size).map_err(|_| {
                            RhiError::new(
                                RhiErrorKind::Unsupported,
                                "WebGL uniform range exceeds u32",
                            )
                        })?;
                        let id = self.buffer_id(*buffer)?;
                        if self.context_state.prepare_uniform_slot(
                            slot,
                            CanonicalBlockId::uniform_range(id, offset, size),
                        ) {
                            self.executor
                                .bind_uniform_buffer(slot, Some(id), offset, size)
                                .map_err(|error| {
                                    map_gl_error(error, "bind WebGL uniform buffer")
                                })?;
                            self.context_state.commit_uniform_slot(
                                slot,
                                CanonicalBlockId::uniform_range(id, offset, size),
                            );
                        }
                    }
                    crate::backend::gl::platform::GlBindingResource::Texture(view) => {
                        self.flush_texture(slot, *view)?
                    }
                    crate::backend::gl::platform::GlBindingResource::Sampler(sampler) => {
                        self.flush_sampler(slot, *sampler)?
                    }
                    crate::backend::gl::platform::GlBindingResource::BufferArray(values) => {
                        for (i, (buffer, offset, size)) in values.iter().enumerate() {
                            let unit = slot
                                .checked_add(u32::try_from(i).map_err(|_| {
                                    RhiError::new(
                                        RhiErrorKind::BackendFailure,
                                        "WebGL binding array index overflow",
                                    )
                                })?)
                                .ok_or_else(|| {
                                    RhiError::new(
                                        RhiErrorKind::BackendFailure,
                                        "WebGL binding slot overflow",
                                    )
                                })?;
                            let offset = offset
                                .checked_add(u64::from(*dynamic_offsets.next().unwrap_or(&0)))
                                .ok_or_else(|| {
                                    RhiError::new(
                                        RhiErrorKind::InvalidUsage,
                                        "WebGL dynamic buffer offset overflow",
                                    )
                                })?;
                            let offset = u32::try_from(offset).map_err(|_| {
                                RhiError::new(
                                    RhiErrorKind::Unsupported,
                                    "WebGL uniform offset exceeds u32",
                                )
                            })?;
                            let size = u32::try_from(*size).map_err(|_| {
                                RhiError::new(
                                    RhiErrorKind::Unsupported,
                                    "WebGL uniform range exceeds u32",
                                )
                            })?;
                            let id = self.buffer_id(*buffer)?;
                            if self.context_state.prepare_uniform_slot(
                                unit,
                                CanonicalBlockId::uniform_range(id, offset, size),
                            ) {
                                self.executor
                                    .bind_uniform_buffer(unit, Some(id), offset, size)
                                    .map_err(|error| {
                                        map_gl_error(error, "bind WebGL uniform buffer")
                                    })?;
                                self.context_state.commit_uniform_slot(
                                    unit,
                                    CanonicalBlockId::uniform_range(id, offset, size),
                                );
                            }
                        }
                    }
                    crate::backend::gl::platform::GlBindingResource::TextureArray(values) => {
                        for (i, view) in values.iter().enumerate() {
                            self.flush_texture(slot + i as u32, *view)?;
                        }
                    }
                    crate::backend::gl::platform::GlBindingResource::SamplerArray(values) => {
                        for (i, sampler) in values.iter().enumerate() {
                            self.flush_sampler(slot + i as u32, *sampler)?;
                        }
                    }
                }
            }
            if dynamic_offsets.next().is_some() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGL bind group has unused dynamic offsets",
                ));
            }
        }
        Ok(())
    }
    fn buffer_id(&self, buffer: crate::backend::gl::platform::GlBufferRef) -> RhiResult<BufferId> {
        let slot = buffer.name.raw().checked_sub(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL buffer carrier was zero",
            )
        })?;
        let entry = self.executor.buffers.get(&slot).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL buffer backing is not live",
            )
        })?;
        Ok(BufferId::new(
            self.executor.context_stamp(),
            slot,
            entry.generation,
        ))
    }
    fn flush_texture(
        &mut self,
        unit: u32,
        view: crate::backend::gl::platform::GlTextureViewRef,
    ) -> RhiResult<()> {
        let view_ref = view;
        let view = self.views.get(&view_ref.name.raw()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL texture view backing is not live",
            )
        })?;
        let binding = CanonicalBlockId::new(u64::from(view_ref.name.raw()));
        let (activate, bind) = self.context_state.prepare_texture_slot(unit, binding);
        if activate {
            self.executor
                .active_texture(unit)
                .map_err(|error| map_gl_error(error, "select WebGL texture unit"))?;
        }
        if bind {
            self.executor
                .bind_texture(unit, view.target, Some(view.texture))
                .map_err(|error| map_gl_error(error, "bind WebGL texture"))?;
        }
        if activate || bind {
            self.context_state.commit_texture_slot(unit, binding);
        }
        Ok(())
    }
    fn flush_sampler(
        &mut self,
        unit: u32,
        sampler: crate::backend::gl::platform::GlSamplerRef,
    ) -> RhiResult<()> {
        let slot = sampler.name.raw().checked_sub(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL sampler carrier was zero",
            )
        })?;
        let entry = self.executor.samplers.get(&slot).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL sampler backing is not live",
            )
        })?;
        let id = SamplerId::new(self.executor.context_stamp(), slot, entry.generation);
        let key = CanonicalBlockId::object(id);
        if self.context_state.prepare_sampler_slot(unit, key) {
            self.executor
                .bind_sampler(unit, Some(id))
                .map_err(|error| map_gl_error(error, "bind WebGL sampler"))?;
            self.context_state.commit_sampler_slot(unit, key);
        }
        Ok(())
    }
    fn retire_geometry(&mut self, predicate: impl Fn(&BrowserGeometryKey) -> bool) {
        let cached = std::mem::take(&mut self.geometry_vaos);
        let mut retired = Vec::new();
        for (key, value) in cached {
            if predicate(&key) {
                retired.push(value.vertex_array);
            } else {
                self.geometry_vaos.insert(key, value);
            }
        }
        for vertex_array in retired {
            let _ = self.executor.destroy_vertex_array(vertex_array);
        }
    }
    fn clear_geometry(&mut self) {
        let retired = std::mem::take(&mut self.geometry_vaos);
        for (_, value) in retired {
            let _ = self.executor.destroy_vertex_array(value.vertex_array);
        }
    }

    fn wake(wakers: &mut Vec<Waker>) {
        for waker in wakers.drain(..) {
            waker.wake();
        }
    }

    fn fail_pending_for_loss(&mut self, info: &DeviceLossInfo) {
        self.fail_readbacks(crate::api::resource::transfer::ReadbackStatus::DeviceLost);
        for completion in self.completions.values_mut() {
            if let BrowserCompletion::Pending { wakers, .. } = completion {
                Self::wake(wakers);
                *completion = BrowserCompletion::DeviceLost(info.clone());
            }
        }
        for state in self.presentation.presents.values_mut() {
            if matches!(state, PresentState::Pending) {
                *state = PresentState::DeviceLost(info.clone());
            }
        }
        for wakers in self.presentation.present_wakers.values_mut() {
            Self::wake(wakers);
        }
        for lease in self.presentation.leases.values_mut() {
            Self::wake(&mut lease.acquire_wakers);
            Self::wake(&mut lease.reconfigure_wakers);
            lease.frame = None;
        }
    }

    fn completion_state(&mut self, serial: u64, waker: Option<&Waker>) -> CompletionState {
        let Some(entry) = self.completions.get_mut(&serial) else {
            return CompletionState::Failed(CompletionFailure::new(format!(
                "WebGL2 completion serial {serial} was never accepted"
            )));
        };
        if let BrowserCompletion::Pending { fence, wakers, .. } = entry {
            match self.executor.poll_fence(*fence) {
                Ok(crate::backend::gl::api::GlFenceStatus::Complete) => {
                    let fence = *fence;
                    Self::wake(wakers);
                    if let BrowserCompletion::Pending { readbacks, .. } = entry {
                        for pending in readbacks.drain(..) {
                            pending.ticket.publish(pending.bytes, pending.layout);
                        }
                    }
                    *entry = BrowserCompletion::Complete;
                    // A completed sync no longer owns a native lifetime. A
                    // delete failure is not a completion rollback; the work is
                    // already terminal and the executor will be torn down on
                    // loss if the browser refuses the cleanup call.
                    let _ = self.executor.destroy_fence(fence);
                }
                Ok(crate::backend::gl::api::GlFenceStatus::Pending)
                | Ok(crate::backend::gl::api::GlFenceStatus::Unknown) => {
                    if let Some(waker) = waker {
                        wakers.push(waker.clone());
                    }
                }
                Ok(crate::backend::gl::api::GlFenceStatus::Failed) => {
                    Self::wake(wakers);
                    if let BrowserCompletion::Pending { readbacks, .. } = entry {
                        for pending in readbacks.drain(..) {
                            pending
                                .ticket
                                .set_status(crate::api::resource::transfer::ReadbackStatus::Failed);
                        }
                    }
                    *entry = BrowserCompletion::Failed(CompletionFailure::new(
                        "WebGL2 fence reported a terminal failure",
                    ));
                }
                Err(error) => {
                    let mapped = map_gl_error(error, "poll WebGL2 completion");
                    if mapped.kind() == RhiErrorKind::DeviceLost {
                        // The public device loss authority will also call
                        // `device_lost`; publish the terminal state here so a
                        // completion waiter cannot observe a false Pending in
                        // between this operation and that callback.
                        let info = DeviceLossInfo::new(
                            "WebGL2 context was lost while polling completion".into(),
                        );
                        Self::wake(wakers);
                        if let BrowserCompletion::Pending { readbacks, .. } = entry {
                            for pending in readbacks.drain(..) {
                                pending.ticket.set_status(
                                    crate::api::resource::transfer::ReadbackStatus::DeviceLost,
                                );
                            }
                        }
                        *entry = BrowserCompletion::DeviceLost(info);
                    } else {
                        Self::wake(wakers);
                        if let BrowserCompletion::Pending { readbacks, .. } = entry {
                            for pending in readbacks.drain(..) {
                                pending.ticket.set_status(
                                    crate::api::resource::transfer::ReadbackStatus::Failed,
                                );
                            }
                        }
                        *entry =
                            BrowserCompletion::Failed(CompletionFailure::new(mapped.to_string()));
                    }
                }
            }
        }
        match entry {
            BrowserCompletion::Pending { .. } => CompletionState::Pending,
            BrowserCompletion::Complete => CompletionState::Complete,
            BrowserCompletion::DeviceLost(info) => CompletionState::DeviceLost(info.clone()),
            BrowserCompletion::Failed(failure) => CompletionState::Failed(failure.clone()),
        }
    }

    /// Timer-side completion polling is the liveness source for pending RHI
    /// futures.  It is intentionally owner-thread-only: the JS `WebGlSync`
    /// never crosses the portable driver's thread boundary.
    fn poll_scheduled_completion(&mut self, serial: u64) -> bool {
        if self.executor.raw.is_context_lost()
            && self.executor.lifecycle() != GlContextLifecycle::Lost
        {
            let _ = self.executor.context_lost();
            self.geometry_vaos.clear();
            self.query_sets.clear();
            self.context_state.event(StateEvent::ContextLost);
            self.fail_pending_for_loss(&DeviceLossInfo::new(
                "the browser reported WebGL context loss while polling completion".into(),
            ));
            return false;
        }

        match self.completion_state(serial, None) {
            CompletionState::Pending => true,
            CompletionState::DeviceLost(info) => {
                // A fence operation discovering loss is authoritative for this
                // context, not merely for the one serial being sampled.
                self.fail_pending_for_loss(&info);
                false
            }
            CompletionState::Complete | CompletionState::Failed(_) => false,
        }
    }

    fn fail_completion_scheduler(&mut self, serial: u64) {
        let Some(entry) = self.completions.get_mut(&serial) else {
            return;
        };
        if let BrowserCompletion::Pending {
            fence,
            wakers,
            readbacks,
        } = entry
        {
            let fence = *fence;
            Self::wake(wakers);
            for pending in readbacks.drain(..) {
                pending
                    .ticket
                    .set_status(crate::api::resource::transfer::ReadbackStatus::Failed);
            }
            *entry = BrowserCompletion::Failed(CompletionFailure::new(
                "the browser event loop could not schedule a WebGL2 completion poll",
            ));
            let _ = self.executor.destroy_fence(fence);
        }
    }

    fn reserve_completion_serial(&mut self) -> RhiResult<u64> {
        let serial = self.next_completion;
        self.next_completion = self.next_completion.checked_add(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL2 completion serial space exhausted",
            )
        })?;
        Ok(serial)
    }

    /// Inserts a completion only after Phase B has entered the browser stream.
    /// Its serial is reserved before the first fence call, so a create/flush
    /// failure can still become a terminal receipt rather than `submit(Err)`.
    /// It is a Fluxel logical serial, never a WebGL object identity.
    fn accept_fence(&mut self, serial: u64) -> RhiResult<()> {
        let fence = self
            .executor
            .create_fence()
            .map_err(|error| map_gl_error(error, "create WebGL2 completion fence"))?;
        if let Err(error) = self
            .executor
            .flush()
            .map_err(|error| map_gl_error(error, "flush WebGL2 completion fence"))
        {
            // Fence creation has already entered WebGL. Cleanup is best-effort
            // because a loss path may reject deletion too.
            let _ = self.executor.destroy_fence(fence);
            return Err(error);
        }
        self.completions.insert(
            serial,
            BrowserCompletion::Pending {
                fence,
                wakers: Vec::new(),
                readbacks: std::mem::take(&mut self.pending_readbacks),
            },
        );
        Ok(())
    }

    /// Publishes the terminal receipt for a stream which already entered
    /// WebGL but could not complete its Phase-B/fence protocol.  This preserves
    /// the public meaning of `submit Err`: no native work was accepted.
    fn publish_post_commit_failure(&mut self, serial: u64, error: &RhiError) {
        self.fail_current_readbacks(if error.kind() == RhiErrorKind::DeviceLost {
            crate::api::resource::transfer::ReadbackStatus::DeviceLost
        } else {
            crate::api::resource::transfer::ReadbackStatus::Failed
        });
        if error.kind() == RhiErrorKind::DeviceLost {
            let info = DeviceLossInfo::new(error.to_string());
            self.fail_pending_for_loss(&info);
            self.completions
                .insert(serial, BrowserCompletion::DeviceLost(info));
        } else {
            self.completions.insert(
                serial,
                BrowserCompletion::Failed(CompletionFailure::new(error.to_string())),
            );
        }
    }
}

impl BrowserV13ObjectResolver for BrowserDriverState {
    fn phase_a_aborted(&self) {
        // Phase A has not entered WebGL and therefore has not accepted the
        // submission; leave tickets NotSubmitted so the portable layer may
        // report the preflight error or retry a later valid submission.
        self.readback_sinks.borrow_mut().clear();
    }
    fn readback(
        &self,
        ticket: &crate::api::resource::transfer::ReadbackTicket,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        use crate::api::resource::transfer::ReadbackRequest;
        use crate::backend::gl::api::{
            GlBufferRange, GlExtent3d, GlPixelFormat, GlPixelLayout, GlRepackPolicy,
            GlTextureAspect, GlTextureRegion, GlTextureSubresource,
        };
        let sink = self.register_readback_sink(ticket)?;
        let packet = match ticket.request() {
            ReadbackRequest::Buffer { src, range, .. } => {
                let name = crate::backend::gl::platform::GlDevice::buffer_ref(src)?.name;
                let slot = name.raw().checked_sub(1).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "WebGL buffer carrier was zero",
                    )
                })?;
                let entry = self.executor.buffers.get(&slot).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "WebGL readback source is not live on this context",
                    )
                })?;
                BrowserV13ReadbackPacket::Buffer {
                    source: GlBufferRange {
                        buffer: BufferId::new(
                            self.executor.context_stamp(),
                            slot,
                            entry.generation,
                        ),
                        offset: range.offset,
                        size: range.size,
                    },
                    sink,
                }
            }
            ReadbackRequest::Texture {
                src,
                subresource,
                origin,
                extent,
                ..
            } => {
                if !matches!(
                    src.descriptor().format,
                    crate::api::format::TextureFormat::Rgba8Unorm
                        | crate::api::format::TextureFormat::Rgba8UnormSrgb
                ) || !matches!(
                    subresource.aspect,
                    crate::api::resource::TextureAspect::Color
                ) {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "WebGL2 readback packet only admits RGBA8 color textures",
                    )
                    .at("WebGL2::submit phase A"));
                }
                let name = crate::backend::gl::platform::GlDevice::texture_ref(src)?.name;
                let slot = name.raw().checked_sub(1).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "WebGL texture carrier was zero",
                    )
                })?;
                let entry = self.executor.textures.get(&slot).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "WebGL readback source is not live on this context",
                    )
                })?;
                let row = extent.width.checked_mul(4).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "WebGL readback row size overflow",
                    )
                })?;
                BrowserV13ReadbackPacket::Texture {
                    source: GlTextureRegion {
                        subresource: GlTextureSubresource {
                            texture: TextureId::new(
                                self.executor.context_stamp(),
                                slot,
                                entry.generation,
                            ),
                            aspect: GlTextureAspect::Color,
                            mip_level: subresource.mip_level,
                            base_layer: subresource.base_layer,
                            layer_count: subresource.layer_count,
                        },
                        origin: [origin.x, origin.y, origin.z],
                        extent: GlExtent3d {
                            width: extent.width,
                            height: extent.height,
                            depth_or_layers: extent.depth,
                        },
                    },
                    layout: GlPixelLayout {
                        format: GlPixelFormat::Rgba8,
                        bytes_per_row: row,
                        rows_per_image: extent.height,
                        offset: 0,
                        alignment: 4,
                        repack: GlRepackPolicy::Disallow,
                    },
                    sink,
                }
            }
        };
        Ok(Box::new(super::v13_submit::BrowserV13PacketAction(
            BrowserV13ActionPacket::Readback(packet),
        )))
    }
    fn copy(
        &self,
        copy: &crate::api::command::record::CopyRecord,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        use crate::api::command::record::CopyRecord;
        use crate::backend::gl::api::{
            GlBufferRange, GlExtent3d, GlTextureAspect, GlTextureRegion, GlTextureSubresource,
        };

        let buffer = |value: &crate::api::resource::Buffer| -> RhiResult<BufferId> {
            let name = crate::backend::gl::platform::GlDevice::buffer_ref(value)?.name;
            let slot = name.raw().checked_sub(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL buffer carrier was zero",
                )
            })?;
            let entry = self.executor.buffers.get(&slot).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "WebGL buffer backing is not live on this context",
                )
            })?;
            Ok(BufferId::new(
                self.executor.context_stamp(),
                slot,
                entry.generation,
            ))
        };
        let region = |texture: &crate::api::resource::Texture,
                      layers: crate::api::resource::TextureSubresourceLayers,
                      origin: crate::api::resource::Origin3d,
                      extent: crate::api::resource::Extent3d|
         -> RhiResult<GlTextureRegion> {
            let name = crate::backend::gl::platform::GlDevice::texture_ref(texture)?.name;
            let slot = name.raw().checked_sub(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL texture carrier was zero",
                )
            })?;
            let entry = self.executor.textures.get(&slot).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "WebGL texture backing is not live on this context",
                )
            })?;
            let aspect = match layers.aspect {
                crate::api::resource::TextureAspect::Color => GlTextureAspect::Color,
                crate::api::resource::TextureAspect::Depth => GlTextureAspect::DepthOnly,
                crate::api::resource::TextureAspect::Stencil => GlTextureAspect::StencilOnly,
                crate::api::resource::TextureAspect::Plane0
                | crate::api::resource::TextureAspect::Plane1
                | crate::api::resource::TextureAspect::Plane2 => {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "WebGL2 has no multi-planar texture transfer route",
                    )
                    .at("WebGL2::submit phase A"));
                }
            };
            Ok(GlTextureRegion {
                subresource: GlTextureSubresource {
                    texture: TextureId::new(self.executor.context_stamp(), slot, entry.generation),
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
        };
        let packet = match copy {
            CopyRecord::Buffer(value) => BrowserV13CopyPacket::Buffer {
                source: GlBufferRange {
                    buffer: buffer(&value.src)?,
                    offset: value.src_offset,
                    size: value.size,
                },
                destination: GlBufferRange {
                    buffer: buffer(&value.dst)?,
                    offset: value.dst_offset,
                    size: value.size,
                },
            },
            CopyRecord::Texture(value) => BrowserV13CopyPacket::Texture {
                source: region(
                    &value.src,
                    value.src_subresource,
                    value.src_origin,
                    value.extent,
                )?,
                destination: region(
                    &value.dst,
                    value.dst_subresource,
                    value.dst_origin,
                    value.extent,
                )?,
            },
            CopyRecord::BufferToTexture(_)
            | CopyRecord::TextureToBuffer(_)
            | CopyRecord::ClearBuffer { .. }
            | CopyRecord::ClearTexture { .. }
            | CopyRecord::ExternalImage(_)
            | CopyRecord::Resolve(_)
            | CopyRecord::Blit(_) => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "WebGL2 v13 has no Phase-A packet for this copy operation",
                )
                .at("WebGL2::submit phase A"));
            }
        };
        Ok(Box::new(super::v13_submit::BrowserV13PacketAction(
            BrowserV13ActionPacket::Copy(packet),
        )))
    }
    fn upload(
        &self,
        upload: &crate::api::resource::transfer::UploadJob,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        use crate::api::resource::transfer::UploadDescriptor;
        use crate::backend::gl::api::{
            GlBufferRange, GlExtent3d, GlPixelFormat, GlPixelLayout, GlRepackPolicy,
            GlTextureAspect, GlTextureRegion, GlTextureSubresource,
        };
        let packet = match upload.descriptor() {
            UploadDescriptor::Buffer(value) => {
                let name = crate::backend::gl::platform::GlDevice::buffer_ref(&value.dst)?.name;
                let slot = name.raw().checked_sub(1).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "WebGL buffer carrier was zero",
                    )
                })?;
                let entry = self.executor.buffers.get(&slot).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "WebGL upload destination is not live on this context",
                    )
                })?;
                BrowserV13UploadPacket::Buffer {
                    destination: GlBufferRange {
                        buffer: BufferId::new(
                            self.executor.context_stamp(),
                            slot,
                            entry.generation,
                        ),
                        offset: value.dst_offset,
                        size: value.bytes.len() as u64,
                    },
                    bytes: value.bytes.clone(),
                }
            }
            UploadDescriptor::Texture(value) => {
                let compressed = crate::backend::gl::translate::is_compressed_texture_format(
                    value.dst.descriptor().format,
                );
                if compressed {
                    crate::backend::gl::translate::validate_compressed_texture_upload(
                        value,
                        "WebGL2::submit phase A",
                    )?;
                } else if !matches!(
                    value.dst.descriptor().format,
                    crate::api::format::TextureFormat::Rgba8Unorm
                        | crate::api::format::TextureFormat::Rgba8UnormSrgb
                ) || !matches!(
                    value.subresource.aspect,
                    crate::api::resource::TextureAspect::Color
                ) {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "WebGL2 upload packet only admits RGBA8 color textures",
                    )
                    .at("WebGL2::submit phase A"));
                }
                let name = crate::backend::gl::platform::GlDevice::texture_ref(&value.dst)?.name;
                let slot = name.raw().checked_sub(1).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "WebGL texture carrier was zero",
                    )
                })?;
                let entry = self.executor.textures.get(&slot).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "WebGL upload destination is not live on this context",
                    )
                })?;
                let alignment = if value.source_layout.bytes_per_row.is_multiple_of(8) {
                    8
                } else if value.source_layout.bytes_per_row.is_multiple_of(4) {
                    4
                } else if value.source_layout.bytes_per_row.is_multiple_of(2) {
                    2
                } else {
                    1
                };
                BrowserV13UploadPacket::Texture {
                    destination: GlTextureRegion {
                        subresource: GlTextureSubresource {
                            texture: TextureId::new(
                                self.executor.context_stamp(),
                                slot,
                                entry.generation,
                            ),
                            aspect: GlTextureAspect::Color,
                            mip_level: value.subresource.mip_level,
                            base_layer: value.subresource.base_layer,
                            layer_count: value.subresource.layer_count,
                        },
                        origin: [value.origin.x, value.origin.y, value.origin.z],
                        extent: GlExtent3d {
                            width: value.extent.width,
                            height: value.extent.height,
                            depth_or_layers: value.extent.depth,
                        },
                    },
                    // Compressed execution never reads `GlPixelLayout`: its
                    // payload is validated as one exact encoded mip above.
                    // Keep the packet shape shared with uncompressed uploads;
                    // the placeholder pixel format is unreachable there.
                    layout: GlPixelLayout {
                        format: GlPixelFormat::Rgba8,
                        bytes_per_row: value.source_layout.bytes_per_row,
                        rows_per_image: value.source_layout.rows_per_image,
                        offset: 0,
                        alignment,
                        repack: GlRepackPolicy::Bounded {
                            max_bytes: value.bytes.len() as u64,
                        },
                    },
                    bytes: value.bytes.clone(),
                }
            }
        };
        Ok(Box::new(super::v13_submit::BrowserV13PacketAction(
            BrowserV13ActionPacket::Upload(packet),
        )))
    }
    fn query_begin(
        &self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        let id = self.query_set_query(set, index)?;
        let packet = match set.descriptor().ty {
            crate::api::query::QueryType::Occlusion => BrowserV13QueryPacket::BeginOcclusion(id),
            crate::api::query::QueryType::Timestamp => BrowserV13QueryPacket::BeginElapsed(id),
            crate::api::query::QueryType::PipelineStatistics(_) => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "WebGL2 has no pipeline-statistics query route",
                )
                .at("WebGL2::submit phase A"));
            }
        };
        Ok(Box::new(super::v13_submit::BrowserV13PacketAction(
            BrowserV13ActionPacket::Query(packet),
        )))
    }
    fn query_end(
        &self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        let _ = self.query_set_query(set, index)?;
        let packet = match set.descriptor().ty {
            crate::api::query::QueryType::Occlusion => BrowserV13QueryPacket::EndOcclusion,
            crate::api::query::QueryType::Timestamp => BrowserV13QueryPacket::EndElapsed,
            crate::api::query::QueryType::PipelineStatistics(_) => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "WebGL2 has no pipeline-statistics query route",
                )
                .at("WebGL2::submit phase A"));
            }
        };
        Ok(Box::new(super::v13_submit::BrowserV13PacketAction(
            BrowserV13ActionPacket::Query(packet),
        )))
    }
    fn timestamp_write(
        &self,
        set: &crate::api::query::QuerySet,
        index: u32,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        if !matches!(set.descriptor().ty, crate::api::query::QueryType::Timestamp) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "timestamp writes require a timestamp query set",
            )
            .at("WebGL2::submit phase A"));
        }
        let id = self.query_set_query(set, index)?;
        Ok(Box::new(super::v13_submit::BrowserV13PacketAction(
            BrowserV13ActionPacket::Query(BrowserV13QueryPacket::Timestamp(id)),
        )))
    }
    fn raster_begin(
        &self,
        begin: &crate::api::command::record::RasterBegin,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        use crate::api::command::attachment::{
            ColorAttachmentView, DepthAttachmentMode, StencilAttachmentMode,
        };
        use crate::api::command::geometry::{ColorClearValue, LoadOp, StoreOp};
        use crate::backend::gl::api::{
            GlColorAttachment, GlColorClearValue, GlDepthStencilAttachment,
            GlDepthStencilClearValue, GlLoadOp, GlPassAttachmentView, GlRenderPassDescriptor,
            GlRenderTarget, GlStoreOp,
        };
        use crate::backend::gl::translate_raster::GlFramebufferCarrier;

        // The shared carrier builder preserves sparse MRT locations and uses
        // the same framebuffer structural validation as native GL.  A 3D
        // depth slice is not representable by the current typed attachment
        // carrier (which only carries array-layer addressing), so name that
        // genuine seam rather than silently attaching layer zero.
        if begin
            .colors
            .iter()
            .any(|(_, color)| color.depth_slice.is_some())
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "WebGL2 typed raster attachment carrier has no 3D depth-slice route",
            )
            .at("WebGL2::submit phase A"));
        }
        let carrier =
            crate::backend::gl::translate_raster::raster_begin_carrier(begin, |view| match view {
                ColorAttachmentView::Texture(view) => self.attachment_view(view).map(Some),
                ColorAttachmentView::Frame(_) => Ok(None),
            })?;

        // WebGL2's typed executor deliberately leaves resolve as an explicit
        // framebuffer blit.  It does not yet receive an end-pass resolve plan,
        // therefore accepting this would lose the resolve rather than lower it.
        if begin
            .colors
            .iter()
            .any(|(_, color)| color.resolve.is_some())
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "WebGL2 typed executor has no end-pass resolve-blit route",
            )
            .at("WebGL2::submit phase A"));
        }

        let gl_load = |load: &LoadOp<ColorClearValue>| match load {
            LoadOp::Load => GlLoadOp::Load,
            LoadOp::Clear(_) => GlLoadOp::Clear,
        };
        let color_clear = |load: &LoadOp<ColorClearValue>| match load {
            LoadOp::Load => GlColorClearValue {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            },
            LoadOp::Clear(ColorClearValue::Float(value)) => GlColorClearValue {
                red: value[0].to_bits(),
                green: value[1].to_bits(),
                blue: value[2].to_bits(),
                alpha: value[3].to_bits(),
            },
            LoadOp::Clear(ColorClearValue::Sint(value)) => GlColorClearValue {
                red: value[0] as u32,
                green: value[1] as u32,
                blue: value[2] as u32,
                alpha: value[3] as u32,
            },
            LoadOp::Clear(ColorClearValue::Uint(value)) => GlColorClearValue {
                red: value[0],
                green: value[1],
                blue: value[2],
                alpha: value[3],
            },
        };
        let colors = begin
            .colors
            .iter()
            .map(|(_, color)| -> RhiResult<GlColorAttachment> {
                let ColorAttachmentView::Texture(view) = &color.view else {
                    return Err(RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "default framebuffer was lowered as an offscreen attachment",
                    ));
                };
                Ok(GlColorAttachment {
                    view: GlPassAttachmentView::Allocated(self.attachment_view(view)?),
                    resolve_target: None,
                    load: gl_load(&color.load),
                    store: if color.store == StoreOp::Discard {
                        GlStoreOp::Discard
                    } else {
                        GlStoreOp::Store
                    },
                    clear: color_clear(&color.load),
                })
            })
            .collect::<RhiResult<Vec<_>>>();
        let depth_stencil = |attachment: &crate::api::command::attachment::DepthStencilAttachment| -> RhiResult<GlDepthStencilAttachment> {
            let (depth_load, depth_store, depth_clear) = match attachment.depth {
                Some(DepthAttachmentMode::ReadOnly) | None => (GlLoadOp::Load, GlStoreOp::Store, 1.0f32.to_bits()),
                Some(DepthAttachmentMode::ReadWrite { load, store }) => (if matches!(load, LoadOp::Clear(_)) { GlLoadOp::Clear } else { GlLoadOp::Load }, if store == StoreOp::Discard { GlStoreOp::Discard } else { GlStoreOp::Store }, match load { LoadOp::Load => 1.0f32.to_bits(), LoadOp::Clear(value) => value.to_bits() }),
            };
            let (stencil_load, stencil_store, stencil_clear) = match attachment.stencil {
                Some(StencilAttachmentMode::ReadOnly) | None => (GlLoadOp::Load, GlStoreOp::Store, 0),
                Some(StencilAttachmentMode::ReadWrite { load, store }) => (if matches!(load, LoadOp::Clear(_)) { GlLoadOp::Clear } else { GlLoadOp::Load }, if store == StoreOp::Discard { GlStoreOp::Discard } else { GlStoreOp::Store }, match load { LoadOp::Load => 0, LoadOp::Clear(value) => value }),
            };
            Ok(GlDepthStencilAttachment { view: self.attachment_view(&attachment.view)?, depth_load, depth_store, stencil_load, stencil_store, clear: GlDepthStencilClearValue { depth: depth_clear, stencil: stencil_clear } })
        };
        let depth = begin
            .depth_stencil
            .as_ref()
            .map(depth_stencil)
            .transpose()?;
        let (descriptor, framebuffer, pass_seed) = match carrier {
            GlFramebufferCarrier::Offscreen(framebuffer) => {
                if framebuffer
                    .draw_buffers
                    .iter()
                    .enumerate()
                    .any(|(index, location)| *location != index as u32)
                {
                    // The current typed framebuffer descriptor represents a
                    // draw-buffer selection over its dense attachment vector;
                    // it cannot encode WebGL's `NONE` entries for an interior
                    // sparse MRT hole yet.
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "WebGL2 typed framebuffer descriptor has no sparse-MRT draw-buffer route",
                    )
                    .at("WebGL2::submit phase A"));
                }
                let colors = colors?;
                (
                    GlRenderPassDescriptor {
                        target: GlRenderTarget::Offscreen(
                            crate::backend::gl::api::FramebufferId::new(
                                self.executor.context_stamp(),
                                0,
                                0,
                            ),
                        ),
                        color_attachments: colors,
                        depth_stencil_attachment: depth,
                    },
                    Some(framebuffer),
                    self.next_canonical_block,
                )
            }
            GlFramebufferCarrier::Default { color_locations } => {
                let [location] = color_locations.as_slice() else {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "WebGL2 default framebuffer admits exactly one color attachment",
                    )
                    .at("WebGL2::submit phase A"));
                };
                if *location != 0 || begin.depth_stencil.is_some() {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "WebGL2 default framebuffer has no MRT or depth/stencil route",
                    )
                    .at("WebGL2::submit phase A"));
                }
                let [(_, color)] = begin.colors.as_slice() else {
                    unreachable!()
                };
                let ColorAttachmentView::Frame(frame) = &color.view else {
                    unreachable!()
                };
                let acquired = crate::backend::gl::platform::framebuffer_ref(frame)?;
                let Some(lease) = self.presentation.leases.get(&acquired.lease.0) else {
                    return Err(RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "WebGL2 frame lease is not live on this context",
                    )
                    .at("WebGL2::submit phase A"));
                };
                if lease
                    .frame
                    .as_ref()
                    .is_none_or(|(serial, _)| *serial != acquired.serial)
                    || acquired.framebuffer != 0
                {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "WebGL2 acquired frame is stale or not this default framebuffer",
                    )
                    .at("WebGL2::submit phase A"));
                }
                let extent = frame.extent();
                if extent.width != acquired.extent.width
                    || extent.height != acquired.extent.height
                    || frame.sample_count() != 1
                {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "WebGL2 frame facts changed after acquisition",
                    )
                    .at("WebGL2::submit phase A"));
                }
                let target = crate::backend::gl::api::GlDefaultFramebufferTarget {
                    frame_serial: acquired.serial,
                    context: self.executor.context_stamp(),
                    width: extent.width,
                    height: extent.height,
                    sample_count: 1,
                    color_format: translate::texture_format(frame.format())?,
                };
                let clear = match color.load {
                    LoadOp::Load => crate::backend::gl::api::GlColorClearValue {
                        red: 0,
                        green: 0,
                        blue: 0,
                        alpha: 0,
                    },
                    LoadOp::Clear(ColorClearValue::Float(value)) => {
                        crate::backend::gl::api::GlColorClearValue {
                            red: value[0].to_bits(),
                            green: value[1].to_bits(),
                            blue: value[2].to_bits(),
                            alpha: value[3].to_bits(),
                        }
                    }
                    LoadOp::Clear(ColorClearValue::Sint(value)) => {
                        crate::backend::gl::api::GlColorClearValue {
                            red: value[0] as u32,
                            green: value[1] as u32,
                            blue: value[2] as u32,
                            alpha: value[3] as u32,
                        }
                    }
                    LoadOp::Clear(ColorClearValue::Uint(value)) => {
                        crate::backend::gl::api::GlColorClearValue {
                            red: value[0],
                            green: value[1],
                            blue: value[2],
                            alpha: value[3],
                        }
                    }
                };
                let descriptor = GlRenderPassDescriptor {
                    target: GlRenderTarget::Default(target),
                    color_attachments: vec![GlColorAttachment {
                        view: GlPassAttachmentView::DefaultColor(target),
                        resolve_target: None,
                        load: if matches!(color.load, LoadOp::Clear(_)) {
                            GlLoadOp::Clear
                        } else {
                            GlLoadOp::Load
                        },
                        store: if color.store == StoreOp::Discard {
                            GlStoreOp::Discard
                        } else {
                            GlStoreOp::Store
                        },
                        clear,
                    }],
                    depth_stencil_attachment: None,
                };
                let base = acquired.serial.checked_mul(3).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "WebGL frame serial cannot form pass state key",
                    )
                })?;
                (descriptor, None, base)
            }
        };
        let base = pass_seed.checked_mul(3).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL pass key space exhausted",
            )
        })?;
        Ok(Box::new(BrowserRasterBeginAction {
            descriptor,
            framebuffer,
            pass: PassPacket {
                draw_framebuffer: CanonicalBlockId::new(base + 1),
                read_framebuffer: CanonicalBlockId::new(base + 2),
                draw_buffers: CanonicalBlockId::new(base + 3),
            },
        }))
    }
    fn raster_end(&self) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        Ok(Box::new(BrowserRasterEndAction))
    }
    fn raster_draw(
        &self,
        draw: &crate::api::command::record::RasterDraw,
    ) -> RhiResult<Box<dyn BrowserV13PhaseBAction>> {
        let scalars = crate::backend::gl::translate_raster::raster_draw_scalars(draw)?;
        scalars
            .draw
            .validate(crate::backend::gl::api::GlAdvancedRasterCapabilities {
                base_vertex: false,
                first_instance: false,
            })
            .map_err(|_| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "WebGL2 baseline raster route has no base-vertex or first-instance lowering",
                )
                .at("WebGL2::submit phase A")
            })?;
        let name =
            crate::backend::gl::platform::GlDevice::raster_pipeline_ref(&draw.pipeline)?.name;
        let pipeline = self.raster_pipelines.get(&name.raw()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "WebGL2 raster pipeline backing is not live on this context",
            )
        })?;
        let stamp = self.executor.context_stamp();
        let buffer_id = |buffer: &crate::api::resource::Buffer| -> RhiResult<BufferId> {
            let name = crate::backend::gl::platform::GlDevice::buffer_ref(buffer)?.name;
            let slot = name.raw().checked_sub(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL buffer carrier was zero",
                )
            })?;
            let entry = self.executor.buffers.get(&slot).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "WebGL vertex/index buffer backing is not live",
                )
            })?;
            Ok(BufferId::new(stamp, slot, entry.generation))
        };
        let geometry = draw
            .vertex_buffers
            .iter()
            .map(|(slot, binding)| {
                Ok(crate::backend::gl::api::GlVertexBufferBinding {
                    slot: *slot,
                    buffer: buffer_id(&binding.buffer)?,
                    offset: binding.range.offset,
                })
            })
            .collect::<RhiResult<Vec<_>>>()?;
        let index = draw
            .index
            .as_ref()
            .map(|index| {
                Ok(crate::backend::gl::api::GlIndexBinding {
                    buffer: buffer_id(&index.binding.buffer)?,
                    format: match index.format {
                        crate::api::command::IndexFormat::Uint16 => {
                            crate::backend::gl::api::GlIndexFormat::Uint16
                        }
                        crate::api::command::IndexFormat::Uint32 => {
                            crate::backend::gl::api::GlIndexFormat::Uint32
                        }
                    },
                    offset: index.binding.range.offset,
                })
            })
            .transpose()?;
        let mut bind_groups = Vec::with_capacity(draw.groups.len());
        for group in &draw.groups {
            let name = crate::backend::gl::platform::GlDevice::bind_group_ref(&group.group)?.name;
            let bindings = self.bind_groups.get(&name.raw()).cloned().ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "WebGL bind-group backing is not live on this context",
                )
            })?;
            let dependencies = bind_group_dependencies(&bindings, &self.views, &self.executor)?;
            bind_groups.push(BrowserBoundGroup {
                packet: BoundGroupPacket {
                    group: group.group.id(),
                    name,
                    index: group.index.get(),
                    dynamic_offsets: group.dynamic_offsets.clone(),
                    program_identity: CanonicalBlockId::object(pipeline.program),
                    dependencies,
                },
                bindings,
            });
        }
        let geometry_key = BrowserGeometryKey {
            pipeline: name,
            vertex_array: pipeline.vertex_array,
            vertices: geometry.clone(),
            index,
        };
        let mut state = pipeline.state.clone();
        let has_dynamic_state = scalars.viewport.is_some()
            || state.scissor != scalars.scissor
            || state.blend_constant != scalars.blend_constant
            || state
                .depth_stencil
                .is_some_and(|value| value.stencil_reference != scalars.stencil_reference);
        if let Some(viewport) = scalars.viewport {
            state.viewport = viewport;
        }
        state.scissor = scalars.scissor;
        state.blend_constant = scalars.blend_constant;
        if let Some(depth_stencil) = &mut state.depth_stencil {
            depth_stencil.stencil_reference = scalars.stencil_reference;
        }
        Ok(Box::new(BrowserRasterDrawAction {
            pipeline: crate::backend::gl::api::GlRasterPipeline {
                program: pipeline.program,
                vertex_array: pipeline.vertex_array,
                state,
            },
            packet: pipeline.state_packet,
            has_dynamic_state,
            draw: scalars.draw.draw,
            geometry,
            index,
            geometry_key,
            bind_groups,
        }))
    }
}

thread_local! {
    static EXECUTORS: RefCell<BTreeMap<u64, BrowserDriverState>> = const {
        RefCell::new(BTreeMap::new())
    };
    static NEXT_EXECUTOR: Cell<u64> = const { Cell::new(1) };
    // A completion future must have an owner-thread source of polls.  Keeping
    // this registry separate from `BrowserDriverState` makes the scheduled
    // callback a backend-private event-loop detail, rather than a public
    // browser/session object.  The `(registration, serial)` key prevents
    // repeated Future::poll calls from queuing an unbounded timer list.
    static COMPLETION_POLLS: RefCell<BTreeSet<(u64, u64)>> = const {
        RefCell::new(BTreeSet::new())
    };
}

/// Requests one later event-loop poll for an accepted WebGL fence.
///
/// WebGL2 has no completion callback for `WebGLSync`; `client_wait_sync` is
/// therefore sampled from a timer rather than spun synchronously.  A 16 ms
/// delay deliberately gives the browser a rendering turn and avoids a busy
/// loop.  A still-pending fence schedules exactly one successor callback.
fn schedule_completion_poll(registration: u64, serial: u64) -> bool {
    let inserted = COMPLETION_POLLS.with(|polls| polls.borrow_mut().insert((registration, serial)));
    if !inserted {
        return true;
    }

    let Some(window) = web_sys::window() else {
        COMPLETION_POLLS.with(|polls| {
            polls.borrow_mut().remove(&(registration, serial));
        });
        return false;
    };
    let callback = wasm_bindgen::closure::Closure::once_into_js(move || {
        COMPLETION_POLLS.with(|polls| {
            polls.borrow_mut().remove(&(registration, serial));
        });
        let pending = EXECUTORS.with(|executors| {
            let mut executors = executors.borrow_mut();
            let Some(state) = executors.get_mut(&registration) else {
                return false;
            };
            state.poll_scheduled_completion(serial)
        });
        if pending && !schedule_completion_poll(registration, serial) {
            // A browser owner without a Window cannot make forward progress.
            // Turn this into a terminal completion and wake its registered
            // future instead of silently parking it forever.
            EXECUTORS.with(|executors| {
                if let Some(state) = executors.borrow_mut().get_mut(&registration) {
                    state.fail_completion_scheduler(serial);
                }
            });
        }
    });
    let callback: &js_sys::Function = callback.unchecked_ref();
    if window
        .set_timeout_with_callback_and_timeout_and_arguments_0(callback, 16)
        .is_err()
    {
        COMPLETION_POLLS.with(|polls| {
            polls.borrow_mut().remove(&(registration, serial));
        });
        return false;
    }
    true
}

/// An opaque, shareable route to a browser-owner WebGL2 executor.
///
/// This is not a browser session or public token.  It is backend-private
/// dispatch bookkeeping; JavaScript context/object ownership never leaves the
/// thread-local registry above.
#[derive(Clone)]
pub(crate) struct WebGl2ExecutionDriver {
    registration: u64,
    owner: ThreadId,
    /// Private liveness route installed only after the public v13 Device is
    /// created. It never exposes a browser context/session outside the backend.
    loss_sink: Arc<Mutex<Option<Arc<dyn GlLossSink>>>>,
}

impl WebGl2ExecutionDriver {
    /// Adopts one RHI-owned browser context into the common provider seam.
    /// The provider receives immutable facts and this handle's dispatch route;
    /// it never receives a canvas, JS context, browser session, or token.
    pub(crate) fn adopt(
        instance: crate::api::identity::DeviceInstanceId,
        executor: WebGl2BrowserDiscovery,
    ) -> RhiResult<crate::backend::gl::platform::GlProvider> {
        let facts = executor.v13_capability_snapshot().into_facts();
        let name = format!("WebGL2 ({})", executor.snapshot().context().renderer());
        let driver: Arc<dyn GlExecutionDriver> = Arc::new(Self::register(executor)?);
        let context = crate::backend::gl::platform::GlAdoptedContext::new(
            crate::api::platform::BackendKind::WebGl2,
            name,
            facts,
            driver,
        )?;
        Ok(crate::backend::gl::platform::GlProvider::adopt(
            instance, context,
        ))
    }
    fn acquire_frame(
        &self,
        lease: GlPresentationLease,
    ) -> Result<Option<GlAcquiredFramebuffer>, AcquireError> {
        self.with_state("WebGl2ExecutionDriver::try_acquire", |state| {
            let Some(record) = state.presentation.leases.get(&lease.0) else {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGL2 presentation lease is no longer live",
                ));
            };
            if record.frame.is_some() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a WebGL2 presentation frame is already outstanding",
                ));
            }
            match state
                .executor
                .acquire_surface_image()
                .map_err(|error| map_gl_error(error, "acquire WebGL2 surface"))?
            {
                GlSurfaceAcquire::Suspended => Ok(None),
                GlSurfaceAcquire::Lease(surface) => {
                    let serial = state.presentation.next_frame;
                    state.presentation.next_frame = serial.checked_add(1).ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::BackendFailure,
                            "WebGL2 acquired-frame serial space exhausted",
                        )
                    })?;
                    let lease_record =
                        state.presentation.leases.get_mut(&lease.0).ok_or_else(|| {
                            RhiError::new(
                                RhiErrorKind::BackendFailure,
                                "WebGL2 presentation lease disappeared during owner-thread acquire",
                            )
                        })?;
                    lease_record.frame = Some((serial, surface));
                    Ok(Some(GlAcquiredFramebuffer {
                        serial,
                        extent: Extent2d {
                            width: surface.size.width,
                            height: surface.size.height,
                        },
                        suboptimal: false,
                        lease,
                        // WebGL2's drawing buffer is the default framebuffer.
                        framebuffer: 0,
                    }))
                }
            }
        })
        .map_err(acquire_error)
    }

    fn end_frame(
        &self,
        lease: GlPresentationLease,
        frame_serial: u64,
        present: bool,
    ) -> RhiResult<()> {
        self.with_state("WebGl2ExecutionDriver::end_frame", |state| {
            state.finish_frame(lease, frame_serial, present)
        })
    }

    fn with_state<T>(
        &self,
        operation: &'static str,
        action: impl FnOnce(&mut BrowserDriverState) -> RhiResult<T>,
    ) -> RhiResult<T> {
        if std::thread::current().id() != self.owner {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL2 work was requested from a thread other than the context owner",
            )
            .at(operation));
        }
        let mut observed_loss = None;
        let result = EXECUTORS.with(|executors| {
            let mut executors = executors.borrow_mut();
            let state = executors.get_mut(&self.registration).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::DeviceLost,
                    "the WebGL2 executor is no longer registered on its owner thread",
                )
                .at(operation)
            })?;
            if state.executor.raw.is_context_lost()
                && state.executor.lifecycle() != GlContextLifecycle::Lost
            {
                state
                    .executor
                    .context_lost()
                    .map_err(|error| map_gl_error(error, "browser context loss"))?;
                state.geometry_vaos.clear();
                state.query_sets.clear();
                state.context_state.event(StateEvent::ContextLost);
                let info = DeviceLossInfo::new("the browser reported WebGL context loss".into());
                state.fail_pending_for_loss(&info);
                observed_loss = Some(info.clone());
                return Err(
                    RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned())
                        .at(operation),
                );
            }
            action(state)
        });
        if let Some(info) = observed_loss {
            self.report_context_loss(info);
        }
        result
    }
    /// Registers an already opened WebGL2 executor on the calling browser
    /// thread.  Registration exhaustion is terminal rather than wrapping into
    /// a different live executor.
    pub(crate) fn register(executor: WebGl2BrowserDiscovery) -> RhiResult<Self> {
        let registration = NEXT_EXECUTOR.with(|next| {
            let value = next.get();
            let next_value = value.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL2 executor registration space exhausted",
                )
                .at("WebGl2ExecutionDriver::register")
            })?;
            next.set(next_value);
            Ok(value)
        })?;
        EXECUTORS.with(|executors| {
            executors
                .borrow_mut()
                .insert(registration, BrowserDriverState::new(executor));
        });
        Ok(Self {
            registration,
            owner: std::thread::current().id(),
            loss_sink: Arc::new(Mutex::new(None)),
        })
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

    /// Removes the executor from its owner-thread registry.  The browser host
    /// calls this only after it has stopped using the adopted context.
    pub(crate) fn unregister(&self) -> RhiResult<()> {
        self.with_executor("WebGl2ExecutionDriver::unregister", |_| Ok(()))?;
        EXECUTORS.with(|executors| {
            executors.borrow_mut().remove(&self.registration);
        });
        Ok(())
    }

    /// Records a browser `webglcontextlost` notification.  This drops all raw
    /// objects without invoking methods on a dead JS context.
    pub(crate) fn notify_context_lost(&self) -> RhiResult<()> {
        let result = self.with_state("WebGl2ExecutionDriver::notify_context_lost", |state| {
            state
                .executor
                .context_lost()
                .map_err(|error| map_gl_error(error, "context lost"))?;
            state.geometry_vaos.clear();
            state.query_sets.clear();
            state.context_state.event(StateEvent::ContextLost);
            state.fail_pending_for_loss(&DeviceLossInfo::new(
                "the browser reported WebGL context loss".into(),
            ));
            Ok(())
        });
        // The DOM event is the first loss observation in the common case; it
        // must update `Device::status()` even if no subsequent RHI call polls a
        // completion or performs a resource operation.
        self.report_context_loss(DeviceLossInfo::new(
            "the browser reported WebGL context loss".to_string(),
        ));
        result
    }

    /// Reopens discovery after a browser `webglcontextrestored` notification.
    /// The returned stamp has a new epoch; old object records remain invalid.
    pub(crate) fn notify_context_restored(&self) -> RhiResult<ContextStamp> {
        self.with_state("WebGl2ExecutionDriver::notify_context_restored", |state| {
            let stamp = state
                .executor
                .context_restored()
                .map_err(|error| map_gl_error(error, "context restore"))?;
            // Context restoration never revives public v13 handles. The old
            // device stays terminal; these tables are cleared so an embedding
            // that creates a new provider/DeviceIdentity cannot accidentally
            // resolve a carrier from the dead context generation.
            state.texture_formats.clear();
            state.views.clear();
            state.bind_groups.clear();
            state.raster_pipelines.clear();
            // The browser context has already discarded these raw VAOs.  Do
            // not delete them, but never allow a restored context to reuse an
            // identity from the old generation.
            state.geometry_vaos.clear();
            state
                .context_state
                .event(StateEvent::ContextRestored(stamp));
            state.presentation.leases.clear();
            Ok(stamp)
        })
    }

    /// Runs a closure only on the owner browser thread and only while the
    /// registration remains live.  Before issuing work it turns a lazily
    /// observed browser loss into the same terminal state as the event path.
    pub(super) fn with_executor<T>(
        &self,
        operation: &'static str,
        action: impl FnOnce(&mut WebGl2BrowserDiscovery) -> RhiResult<T>,
    ) -> RhiResult<T> {
        self.with_state(operation, |state| action(&mut state.executor))
    }
}

impl BrowserDriverState {
    fn finish_frame(
        &mut self,
        lease: GlPresentationLease,
        expected_serial: u64,
        present: bool,
    ) -> RhiResult<()> {
        let frame = self
            .presentation
            .leases
            .get_mut(&lease.0)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGL2 presentation lease is no longer live",
                )
            })?
            .frame
            .as_ref()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGL2 presentation frame is not outstanding",
                )
            })?;
        if frame.0 != expected_serial {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "WebGL2 presentation frame serial does not match the outstanding lease frame",
            ));
        }
        // Validate before consuming: a stale abandon/present must leave the
        // live newer frame intact for its rightful owner.
        let surface = self
            .presentation
            .leases
            .get_mut(&lease.0)
            .expect("presentation lease was just proven live")
            .frame
            .take()
            .expect("presentation frame was just proven outstanding")
            .1;
        if present {
            self.executor
                .present_surface(surface)
                .map_err(|error| map_gl_error(error, "present WebGL2 surface"))?;
        } else {
            // The old surface book has no explicit abandon verb: consuming the
            // lease ends the acquired-image ownership without promising a
            // compositor handoff. It is intentionally not a GL call.
            self.executor
                .surface
                .consume("abandon-surface-image", surface)
                .map_err(|error| map_gl_error(error, "abandon WebGL2 surface"))?;
        }
        Ok(())
    }
}

fn acquire_error(error: RhiError) -> AcquireError {
    let kind = match error.kind() {
        RhiErrorKind::DeviceLost => AcquireErrorKind::DeviceLost,
        RhiErrorKind::OutOfMemory => AcquireErrorKind::OutOfMemory,
        _ => AcquireErrorKind::NotReady,
    };
    AcquireError::new(kind, error.to_string())
}

impl WebGl2BrowserDiscovery {
    /// Narrows the browser's immutable discovery ledger to the v13 routes this
    /// driver has actually installed.  It deliberately does not promote a
    /// WebGL extension object to a public feature by itself.
    fn v13_capability_snapshot(&self) -> crate::backend::gl::capabilities::GlCapabilitySnapshot {
        use crate::api::format::{TextureFormat, TextureSupportLimits, TextureSupportQuery};
        use crate::api::resource::texture::{Extent3d, TextureDimension, TextureUsage};
        use crate::backend::gl::api::{GlFormatResourceKind, GlKnownExtension};
        use crate::backend::gl::capabilities::{
            GlCapabilitySnapshot, GlFormatEvidence, GlLoweringClosure, GlTextureEvidence,
        };
        use crate::backend::gl::facts::{GlExtension, GlFeatureProbe, GlFunction};

        let discovery = self.snapshot();
        let mut feature_probe = GlFeatureProbe::new(discovery.context().profile());
        // Browser-core sync/query entry points are known callable because this
        // executor acquired the typed WebGL2 context before publishing it.
        for function in [
            GlFunction::FenceSync,
            GlFunction::ClientWaitSync,
            GlFunction::DeleteSync,
            GlFunction::BeginQuery,
            GlFunction::EndQuery,
            GlFunction::GetQueryObject,
            // These WebGL2 core methods are invoked by the whole-mip
            // compressed upload path. Extensions select individual format
            // enums; they do not supply an unobserved command route.
            GlFunction::CompressedTexImage2d,
            GlFunction::CompressedTexSubImage2d,
        ] {
            feature_probe.report_function(function);
        }
        for (known, fact) in [
            (
                GlKnownExtension::ExtDisjointTimerQueryWebgl2,
                GlExtension::ExtDisjointTimerQuery,
            ),
            (
                GlKnownExtension::ExtTextureFilterAnisotropic,
                GlExtension::ExtTextureFilterAnisotropic,
            ),
            (
                GlKnownExtension::CompressedTextureS3tc,
                GlExtension::ExtTextureCompressionS3tc,
            ),
            (
                GlKnownExtension::CompressedTextureRgtc,
                GlExtension::ExtTextureCompressionRgtc,
            ),
            (
                GlKnownExtension::CompressedTextureBptc,
                GlExtension::ArbTextureCompressionBptc,
            ),
            (
                GlKnownExtension::CompressedTextureEtc,
                GlExtension::OesCompressedEtc2Rgb8Texture,
            ),
            (
                GlKnownExtension::CompressedTextureAstc,
                GlExtension::KhrTextureCompressionAstcLdr,
            ),
            (
                GlKnownExtension::WebglMultiDraw,
                GlExtension::WebglMultiDraw,
            ),
            (GlKnownExtension::OvrMultiview2, GlExtension::OvrMultiview2),
            (GlKnownExtension::KhrDebug, GlExtension::KhrDebug),
        ] {
            if discovery.extensions().is_acquired(known) {
                feature_probe.report_extension(fact);
            }
        }
        if self.astc_hdr
            && discovery
                .extensions()
                .is_acquired(GlKnownExtension::CompressedTextureAstc)
        {
            feature_probe.report_extension(GlExtension::KhrTextureCompressionAstcHdr);
        }
        if self.timer.is_some() {
            feature_probe.report_function(GlFunction::QueryCounterExt);
            feature_probe.report_function(GlFunction::GetQueryObjectExt);
        }
        if self.multi_draw.is_some() {
            feature_probe.report_function(GlFunction::WebGlMultiDrawArrays);
            feature_probe.report_function(GlFunction::WebGlMultiDrawElements);
        }
        if discovery
            .extensions()
            .is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic)
        {
            feature_probe.report_function(GlFunction::TexParameterAnisotropy);
        }

        let mut formats = Vec::new();
        let mut textures = Vec::new();
        let limits = discovery.limits();
        let texture_limits = TextureSupportLimits::new(
            Extent3d::d2(limits.max_texture_size, limits.max_texture_size),
            // WebGL2 `texStorage2D` creates a complete immutable mip chain;
            // the portable descriptor validator separately rejects impossible
            // mip counts for the requested extent.
            31,
            1,
        );
        for format in TextureFormat::all() {
            let Ok(gl_format) = translate::texture_format(format) else {
                continue;
            };
            let Some(evidence) =
                discovery
                    .formats()
                    .get_for(GlFormatResourceKind::Texture, gl_format, 1)
            else {
                continue;
            };
            let depth_or_stencil = matches!(
                format,
                TextureFormat::Depth16Unorm
                    | TextureFormat::Depth24Plus
                    | TextureFormat::Depth24PlusStencil8
                    | TextureFormat::Depth32Float
                    | TextureFormat::Depth32FloatStencil8
                    | TextureFormat::Stencil8
            );
            formats.push(GlFormatEvidence {
                format,
                storage_read: false,
                storage_write: false,
                storage_read_write: false,
                color_attachment: evidence.renderable && !depth_or_stencil,
                depth_attachment: evidence.renderable
                    && matches!(
                        format,
                        TextureFormat::Depth16Unorm
                            | TextureFormat::Depth24Plus
                            | TextureFormat::Depth24PlusStencil8
                            | TextureFormat::Depth32Float
                            | TextureFormat::Depth32FloatStencil8
                    ),
                stencil_attachment: evidence.renderable
                    && matches!(
                        format,
                        TextureFormat::Stencil8
                            | TextureFormat::Depth24PlusStencil8
                            | TextureFormat::Depth32FloatStencil8
                    ),
                blendable: evidence.blendable,
                filterable: evidence.filterable,
                storage_atomic: false,
            });
            // The present WebGL allocation route is deliberately narrow: 2D,
            // sample-count one, and no storage image use. Enumerate every
            // remaining usage mask so capability lookup is total rather than a
            // hopeful singleton table.
            for usage in TextureUsage::all() {
                if usage.is_empty() || usage.contains(TextureUsage::STORAGE) {
                    continue;
                }
                let rgba8 = matches!(
                    format,
                    TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
                );
                let compressed =
                    crate::backend::gl::translate::is_compressed_texture_format(format);
                // TextureUsage is a promise about every public operation that
                // consumes that usage.  v13 GL readback is currently defined
                // only for RGBA8, so native texture-copy support alone must not
                // publish COPY_SRC for other formats.  COPY_DST additionally
                // admits the exact compressed whole-mip upload route.
                if usage.contains(TextureUsage::COPY_SRC) && (!rgba8 || !evidence.copy_source) {
                    continue;
                }
                if usage.contains(TextureUsage::COPY_DST)
                    && !(rgba8 && evidence.copy_destination)
                    && !compressed
                {
                    continue;
                }
                if usage.contains(TextureUsage::COLOR_ATTACHMENT)
                    && (!evidence.renderable || depth_or_stencil)
                {
                    continue;
                }
                if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
                    && (!evidence.renderable || !depth_or_stencil)
                {
                    continue;
                }
                textures.push(GlTextureEvidence {
                    query: TextureSupportQuery::new(TextureDimension::D2, format, usage, 1),
                    limits: texture_limits,
                });
            }
        }
        GlCapabilitySnapshot {
            profile: discovery.context().profile(),
            feature_probe,
            limits,
            maximum_buffer_size: 9_007_199_254_740_991,
            formats,
            textures,
            lowering: GlLoweringClosure {
                buffers: true,
                textures: true,
                buffer_copy: true,
                texture_copy: true,
                // Shader ABI 1.0 does not carry the GL reflection names needed
                // to map logical (group, slot) bindings into a linked program.
                // Keep public binding facts closed until a later artifact ABI
                // provides that table; the backend must not guess identifiers.
                bindings: false,
                // WebGL2 exposes query objects, but no query-buffer route.
                // Synchronously polling a result during resolve would either
                // stall or violate the submission/completion contract, so the
                // public feature stays closed until an asynchronous resolve
                // sink exists.
                occlusion_query: false,
                sampler_anisotropy: true,
                // `compressedTexImage2D` accepts an exact whole-mip payload;
                // format evidence above remains the per-extension admission
                // gate, including ASTC HDR's separately observed profile.
                compressed_upload: true,
                // Upload/copy/query v13 actions are connected separately; no
                // feature may become public before that Phase-A/Phase-B route.
                ..Default::default()
            },
        }
    }
}

impl GlExecutionDriver for WebGl2ExecutionDriver {
    fn dispatch(&self, operation: &'static str) -> RhiResult<()> {
        self.with_executor(operation, |executor| {
            executor
                .assert_provider_ready(operation)
                .map_err(|error| map_gl_error(error, "WebGL2 dispatch"))
        })
    }

    fn install_loss_sink(&self, sink: Arc<dyn GlLossSink>) {
        *self.loss_sink.lock().unwrap_or_else(|p| p.into_inner()) = Some(sink);
    }

    fn submit(
        &self,
        plan: crate::backend::gl::platform::GlSubmissionPlan<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        // Phase A is deliberately complete before the first browser call.
        let submission = self.with_state("WebGl2ExecutionDriver::submit phase A", |state| {
            super::v13_submit::BrowserV13Submission::phase_a(plan, state)
        })?;
        let mut post_commit_loss = None;
        let outcome = self.with_state("WebGl2ExecutionDriver::submit phase B", |state| {
            // Exhaustion is still a true submit rejection: no browser call has
            // occurred yet. Every later failure is represented by this serial.
            let completion = state.reserve_completion_serial()?;
            match submission.phase_b(state) {
                Ok(entered) => {
                    if let Err(error) = state.accept_fence(completion) {
                        if error.kind() == RhiErrorKind::DeviceLost {
                            post_commit_loss = Some(DeviceLossInfo::new(error.to_string()));
                        }
                        state.publish_post_commit_failure(completion, &error);
                    }
                    Ok(crate::api::submission::backend::SubmissionOutcome {
                        completion,
                        points: entered
                            .into_iter()
                            .map(|point| (point, completion))
                            .collect(),
                    })
                }
                Err(failure) => {
                    // Browser calls may already have changed state.  Publish a
                    // terminal receipt rather than returning a false rollback.
                    if failure.error.kind() == RhiErrorKind::DeviceLost {
                        post_commit_loss = Some(DeviceLossInfo::new(failure.error.to_string()));
                    }
                    state.publish_post_commit_failure(completion, &failure.error);
                    Ok(crate::api::submission::backend::SubmissionOutcome {
                        completion,
                        points: failure
                            .entered_batches
                            .into_iter()
                            .map(|point| (point, completion))
                            .collect(),
                    })
                }
            }
        })?;
        if let Some(info) = post_commit_loss {
            self.report_context_loss(info);
        }
        Ok(outcome)
    }

    fn create_buffer(
        &self,
        descriptor: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<GlObjectName> {
        let descriptor = translate::buffer_descriptor(descriptor)?;
        self.with_state("WebGl2ExecutionDriver::create_buffer", |state| {
            let id = state
                .executor
                .create_buffer_resource(descriptor)
                .map_err(|error| map_gl_error(error, "create WebGL2 buffer"))?;
            object_name(id.slot, "WebGl2ExecutionDriver::create_buffer")
        })
    }

    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<GlObjectName> {
        let base_format = descriptor.format;
        let descriptor = translate::texture_descriptor(descriptor)?;
        self.with_state("WebGl2ExecutionDriver::create_texture", |state| {
            let id = state
                .executor
                .create_texture_resource(descriptor)
                .map_err(|error| map_gl_error(error, "create WebGL2 texture"))?;
            // The typed executor is authoritative for liveness/generation;
            // this is only the public-format fact a virtual view needs.
            debug_assert!(state.executor.textures.contains_key(&id.slot));
            state.texture_formats.insert(id.slot, base_format);
            object_name(id.slot, "WebGl2ExecutionDriver::create_texture")
        })
    }

    fn create_texture_view(
        &self,
        texture: GlTextureRef,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<GlObjectName> {
        let slot = texture.name.raw().checked_sub(1).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "WebGL texture carrier was zero",
            )
        })?;
        // Conversion is a preflight: WebGL2 has virtual views, but it must not
        self.with_state("WebGl2ExecutionDriver::create_texture_view", |state| {
            let entry = state.executor.textures.get(&slot).ok_or_else(|| {
                RhiError::new(RhiErrorKind::Unsupported, "texture backing is not live")
            })?;
            let texture_id = TextureId::new(state.executor.context_stamp(), slot, entry.generation);
            let base_format = *state.texture_formats.get(&slot).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL texture table lost its public base-format metadata",
                )
            })?;
            // WebGL2 has no independent view object.  Accept only the exact
            // whole-texture shape; changing base/max level would mutate shared
            // texture state and make sibling views alias.
            let dimension =
                translate::whole_compatible_virtual_view(descriptor, base_format, entry.desc)
                    .map_err(|error| error.at("WebGl2ExecutionDriver::create_texture_view"))?;
            // The portable layer has already checked format compatibility; the
            // virtual view retains the generation-safe base texture identity.
            let name = state.next_virtual;
            state.next_virtual = state.next_virtual.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL virtual-view namespace exhausted",
                )
            })?;
            state.views.insert(
                name,
                BrowserTextureView {
                    texture: texture_id,
                    target: browser_binding_target(dimension)?,
                    // `view_format` above has already verified the exact format
                    // route. The public portable validator owns reinterpretation
                    // compatibility; retain the immutable base fact here solely
                    // to make resolver-side liveness checks explicit.
                    base_format,
                },
            );
            GlObjectName::new(name, "WebGl2ExecutionDriver::create_texture_view")
        })
    }

    fn create_bind_group(&self, packet: &GlBindGroupPacket) -> RhiResult<GlObjectName> {
        self.with_state("WebGl2ExecutionDriver::create_bind_group", |state| {
            let name = state.next_virtual;
            state.next_virtual = state.next_virtual.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL bind-group namespace exhausted",
                )
            })?;
            state.bind_groups.insert(name, packet.clone());
            GlObjectName::new(name, "WebGl2ExecutionDriver::create_bind_group")
        })
    }

    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<GlObjectName> {
        let descriptor = translate::sampler_descriptor(descriptor)?;
        self.with_state("WebGl2ExecutionDriver::create_sampler", |state| {
            let id = state
                .executor
                .create_sampler(descriptor)
                .map_err(|error| map_gl_error(error, "create WebGL2 sampler"))?;
            object_name(id.slot, "WebGl2ExecutionDriver::create_sampler")
        })
    }

    fn create_shader(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<GlObjectName> {
        let source = translate::shader_source(artifact)?;
        self.with_state("WebGl2ExecutionDriver::create_shader", |state| {
            let id = state
                .executor
                .create_shader(&source)
                .map_err(|error| map_gl_error(error, "create WebGL2 shader"))?;
            object_name(id.slot, "WebGl2ExecutionDriver::create_shader")
        })
    }

    fn create_query_set(
        &self,
        descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<GlObjectName> {
        self.with_state("WebGl2ExecutionDriver::create_query_set", |state| {
            let count = usize::try_from(descriptor.count).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGL query-set count exceeds addressable range",
                )
            })?;
            let name = state.next_virtual;
            state.next_virtual = state.next_virtual.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL virtual query-set namespace exhausted",
                )
            })?;
            let mut queries = Vec::with_capacity(count);
            for _ in 0..count {
                match state.executor.create_query() {
                    Ok(query) => queries.push(query),
                    Err(error) => {
                        // The portable object is never published on partial
                        // creation. Release every successfully-created native
                        // object before reporting the failed allocation.
                        for query in queries {
                            let _ = state.executor.destroy_query(query);
                        }
                        return Err(map_gl_error(error, "create WebGL2 query"));
                    }
                }
            }
            state.query_sets.insert(name, queries);
            GlObjectName::new(name, "WebGl2ExecutionDriver::create_query_set")
        })
    }

    fn create_compute_pipeline(&self, _: GlComputePipelinePacket<'_>) -> RhiResult<GlObjectName> {
        // This is a capability boundary, not an unfinished WebGL implementation:
        // WebGL2 / ESSL 3.00 has no compute execution model.
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "WebGL2 has no compute-pipeline lowering",
        )
        .at("WebGl2ExecutionDriver::create_compute_pipeline"))
    }

    fn create_raster_pipeline(
        &self,
        shader_packet: GlRasterPipelinePacket<'_>,
    ) -> RhiResult<GlObjectName> {
        // Binding-layout translation remains deliberately conservative until
        // the shared layout adapter lands: an empty GL layout is valid only for
        // shader pairs whose reflection proves they declare no bind resources.
        // `create_program` performs that proof, so this is a real basic raster
        // route (position-only/color-only triangles), not a synthetic handle.
        let default_viewport = crate::backend::gl::api::GlViewport {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            min_depth: 0.0f32.to_bits(),
            max_depth: 1.0f32.to_bits(),
        };
        let packet = crate::backend::gl::translate_raster::raster_pipeline_packet(
            shader_packet.descriptor,
            default_viewport,
        )?;
        self.with_state("WebGl2ExecutionDriver::create_raster_pipeline", |state| {
            // Do not let a descriptor recompile itself after its declared
            // shader handles were dropped or came from another driver table.
            // The packet references are the native-ownership proof; sources
            // are only used to build the immutable program descriptor.
            for shader in [Some(shader_packet.vertex), shader_packet.fragment]
                .into_iter()
                .flatten()
            {
                let slot = shader.name.raw().checked_sub(1).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "WebGL shader carrier was zero",
                    )
                })?;
                if !state.executor.shaders.contains_key(&slot) {
                    return Err(RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "WebGL raster shader backing is not live",
                    ));
                }
            }
            let (program, _) = state
                .executor
                .create_program(&packet.program)
                .map_err(|error| map_gl_error(error, "link WebGL2 raster program"))?;
            let vertex_array = match state.executor.create_vertex_array(&packet.vertex_layout) {
                Ok(value) => value,
                Err(error) => {
                    let _ = state.executor.destroy_program(program);
                    return Err(map_gl_error(error, "create WebGL2 raster vertex array"));
                }
            };
            let name = state.next_virtual;
            state.next_virtual = state.next_virtual.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL raster-pipeline namespace exhausted",
                )
            })?;
            let identity =
                GlObjectName::new(name, "WebGl2ExecutionDriver::create_raster_pipeline")?;
            let pipeline = crate::backend::gl::api::GlRasterPipeline {
                program,
                vertex_array,
                state: packet.state,
            };
            let state_packet = StateRasterPipelinePacket {
                identity,
                blocks: state
                    .pipeline_blocks
                    .intern(&pipeline, "WebGL canonical pipeline namespace exhausted")
                    .map_err(|_| {
                        RhiError::new(
                            RhiErrorKind::BackendFailure,
                            "WebGL canonical pipeline namespace exhausted",
                        )
                    })?,
            };
            state
                .context_state
                .register_pipeline(state_packet)
                .map_err(|error| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        format!("WebGL canonical pipeline registration failed: {error:?}"),
                    )
                })?;
            state.raster_pipelines.insert(
                name,
                BrowserRasterPipeline {
                    program: pipeline.program,
                    vertex_array: pipeline.vertex_array,
                    state: pipeline.state,
                    state_packet,
                },
            );
            Ok(identity)
        })
    }

    fn destroy(&self, kind: GlObjectKind, name: GlObjectName) {
        // Drop cannot return an error.  A context loss makes native deletion
        // meaningless; otherwise retain typed generation checks before calling
        // the old executor's real delete route.
        let _ = self.with_state("WebGl2ExecutionDriver::destroy", |state| {
            if state.executor.lifecycle() == GlContextLifecycle::Lost {
                return Ok(());
            }
            let slot = name.raw().checked_sub(1).ok_or_else(|| {
                RhiError::new(RhiErrorKind::BackendFailure, "GL object carrier was zero")
            })?;
            let stamp = state.executor.context_stamp();
            let result = match kind {
                GlObjectKind::Buffer => state
                    .executor
                    .buffers
                    .get(&slot)
                    .map(|entry| entry.generation)
                    .map(|generation| {
                        state
                            .executor
                            .destroy_buffer_resource(BufferId::new(stamp, slot, generation))
                    }),
                GlObjectKind::Texture => state
                    .executor
                    .textures
                    .get(&slot)
                    .map(|entry| entry.generation)
                    .map(|generation| {
                        state
                            .executor
                            .destroy_texture_resource(TextureId::new(stamp, slot, generation))
                    }),
                GlObjectKind::Sampler => state
                    .executor
                    .samplers
                    .get(&slot)
                    .map(|entry| entry.generation)
                    .map(|generation| {
                        state
                            .executor
                            .destroy_sampler(SamplerId::new(stamp, slot, generation))
                    }),
                GlObjectKind::Shader => state
                    .executor
                    .shaders
                    .get(&slot)
                    .map(|entry| entry.generation)
                    .map(|generation| {
                        state
                            .executor
                            .destroy_shader(ShaderId::new(stamp, slot, generation))
                    }),
                GlObjectKind::QuerySet => {
                    let Some(queries) = state.query_sets.remove(&name.raw()) else {
                        return Ok(());
                    };
                    // A set is a virtual carrier for several WebGL objects.
                    // Drop every member even when one delete reports an error;
                    // retaining later entries would turn a failed logical drop
                    // into a permanent native leak.
                    let mut failure = None;
                    for query in queries {
                        state.context_state.event(StateEvent::QueryRetired(query));
                        if let Err(error) = state.executor.destroy_query(query) {
                            failure.get_or_insert(error);
                        }
                    }
                    if let Some(error) = failure {
                        return Err(map_gl_error(error, "destroy WebGL2 query-set"));
                    }
                    return Ok(());
                }
                GlObjectKind::TextureView => {
                    state.views.remove(&name.raw());
                    // A view is virtual, but an already-applied bind group
                    // can otherwise survive its retirement without another
                    // validation/flush.
                    state.context_state.event(StateEvent::DomainFailed(
                        crate::backend::gl::state::StateDomain::Bindings,
                    ));
                    return Ok(());
                }
                GlObjectKind::BindGroup => {
                    state.bind_groups.remove(&name.raw());
                    state.context_state.retire_bind_group(name);
                    return Ok(());
                }
                GlObjectKind::ComputePipeline => return Ok(()),
                GlObjectKind::RasterPipeline => {
                    if let Some(pipeline) = state.raster_pipelines.remove(&name.raw()) {
                        let pipeline_name = name;
                        state.retire_geometry(|key| key.pipeline == pipeline_name);
                        state
                            .context_state
                            .event(StateEvent::ProgramRetired(pipeline.program));
                        state
                            .context_state
                            .event(StateEvent::VertexArrayRetired(pipeline.vertex_array));
                        let _ = state.executor.destroy_vertex_array(pipeline.vertex_array);
                        let _ = state.executor.destroy_program(pipeline.program);
                    }
                    return Ok(());
                }
            };
            if let Some(result) = result {
                let retired_buffer = if matches!(kind, GlObjectKind::Buffer) {
                    state
                        .executor
                        .buffers
                        .get(&slot)
                        .map(|entry| BufferId::new(stamp, slot, entry.generation))
                } else {
                    None
                };
                let retired_texture = if matches!(kind, GlObjectKind::Texture) {
                    state
                        .executor
                        .textures
                        .get(&slot)
                        .map(|entry| TextureId::new(stamp, slot, entry.generation))
                } else {
                    None
                };
                let retired_sampler = if matches!(kind, GlObjectKind::Sampler) {
                    state
                        .executor
                        .samplers
                        .get(&slot)
                        .map(|entry| SamplerId::new(stamp, slot, entry.generation))
                } else {
                    None
                };
                // Publish retirement to the shared state authority before
                // calling WebGL deletion.  A driver exception is then
                // conservative (next use rebinds/revalidates), never a stale
                // cache hit into a deleted or slot-reused object.
                if let Some(id) = retired_buffer {
                    state.retire_geometry(|key| {
                        key.vertices.iter().any(|binding| binding.buffer == id)
                            || key.index.is_some_and(|index| index.buffer == id)
                    });
                    state.context_state.event(StateEvent::BufferRetired(id));
                }
                if let Some(id) = retired_texture {
                    state.context_state.event(StateEvent::TextureRetired(id));
                }
                if let Some(id) = retired_sampler {
                    state.context_state.event(StateEvent::SamplerRetired(id));
                }
                result.map_err(|error| map_gl_error(error, "destroy WebGL2 object"))?;
                if matches!(kind, GlObjectKind::Texture) {
                    state.texture_formats.remove(&slot);
                }
            }
            Ok(())
        });
    }

    fn completion(&self, serial: u64) -> CompletionState {
        let completion = self
            .with_state("WebGl2ExecutionDriver::completion", |state| {
                Ok(state.completion_state(serial, None))
            })
            .unwrap_or_else(|error| {
                if error.kind() == RhiErrorKind::DeviceLost {
                    CompletionState::DeviceLost(DeviceLossInfo::new(error.to_string()))
                } else {
                    CompletionState::Failed(CompletionFailure::new(error.to_string()))
                }
            });
        if let CompletionState::DeviceLost(info) = &completion {
            self.report_context_loss(info.clone());
        }
        completion
    }

    fn completion_or_register_waker(&self, serial: u64, waker: &Waker) -> CompletionState {
        let completion = self
            .with_state(
                "WebGl2ExecutionDriver::completion_or_register_waker",
                |state| Ok(state.completion_state(serial, Some(waker))),
            )
            .unwrap_or_else(|error| {
                if error.kind() == RhiErrorKind::DeviceLost {
                    CompletionState::DeviceLost(DeviceLossInfo::new(error.to_string()))
                } else {
                    CompletionState::Failed(CompletionFailure::new(error.to_string()))
                }
            });
        if let CompletionState::DeviceLost(info) = &completion {
            self.report_context_loss(info.clone());
        }
        if matches!(completion, CompletionState::Pending)
            && !schedule_completion_poll(self.registration, serial)
        {
            let failure = CompletionFailure::new(
                "the browser event loop could not schedule a WebGL2 completion poll",
            );
            let _ = self.with_state(
                "WebGl2ExecutionDriver::completion scheduler failure",
                |state| {
                    state.fail_completion_scheduler(serial);
                    Ok(())
                },
            );
            return CompletionState::Failed(failure);
        }
        completion
    }

    fn supports_presentation(
        &self,
        _: &crate::api::presentation::PresentationTarget,
    ) -> RhiResult<bool> {
        self.with_executor("WebGl2ExecutionDriver::supports_presentation", |executor| {
            executor
                .assert_provider_ready("WebGl2ExecutionDriver::supports_presentation")
                .map_err(|error| map_gl_error(error, "query WebGL2 presentation"))?;
            Ok(true)
        })
    }

    fn presentation_capabilities(&self, _: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        self.with_executor(
            "WebGl2ExecutionDriver::presentation_capabilities",
            |executor| {
                executor
                    .assert_provider_ready("WebGl2ExecutionDriver::presentation_capabilities")
                    .map_err(|error| {
                        map_gl_error(error, "query WebGL2 presentation capabilities")
                    })?;
                let current = match (
                    executor.raw.drawing_buffer_width(),
                    executor.raw.drawing_buffer_height(),
                ) {
                    (width, height) if width > 0 && height > 0 => Some(Extent2d {
                        width: width as u32,
                        height: height as u32,
                    }),
                    _ => None,
                };
                // WebGL's compositor owns pacing and the canvas drawable extent.
                // It cannot truthfully offer FIFO/Immediate semantics or RHI-sized
                // swapchain configuration, so Automatic + HostManaged is the whole
                // portable contract here.
                Ok(PresentationTargetCapabilities::new(
                    vec![crate::api::format::TextureFormat::Rgba8Unorm],
                    vec![PresentMode::Automatic],
                    PresentationExtentControl::HostManaged { current },
                ))
            },
        )
    }

    fn configure_presentation(
        &self,
        target: ObjectId,
        _: &PresentationConfiguration,
    ) -> RhiResult<GlPresentationLease> {
        self.with_state("WebGl2ExecutionDriver::configure_presentation", |state| {
            state
                .executor
                .assert_provider_ready("WebGl2ExecutionDriver::configure_presentation")
                .map_err(|error| map_gl_error(error, "configure WebGL2 presentation"))?;
            if state
                .presentation
                .leases
                .values()
                .any(|lease| lease.target == target)
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the WebGL2 target already has a configured presentation lease",
                ));
            }
            let id = state.presentation.next_lease;
            state.presentation.next_lease = id.checked_add(1).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "WebGL2 presentation lease space exhausted",
                )
            })?;
            state.presentation.leases.insert(
                id,
                BrowserPresentationLease {
                    target,
                    frame: None,
                    acquire_wakers: Vec::new(),
                    reconfigure_wakers: Vec::new(),
                },
            );
            Ok(GlPresentationLease(id))
        })
    }

    fn lease_capabilities(
        &self,
        lease: GlPresentationLease,
    ) -> RhiResult<PresentationTargetCapabilities> {
        self.with_state("WebGl2ExecutionDriver::lease_capabilities", |state| {
            if !state.presentation.leases.contains_key(&lease.0) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGL2 presentation lease is no longer live",
                ));
            }
            let current = match (
                state.executor.raw.drawing_buffer_width(),
                state.executor.raw.drawing_buffer_height(),
            ) {
                (width, height) if width > 0 && height > 0 => Some(Extent2d {
                    width: width as u32,
                    height: height as u32,
                }),
                _ => None,
            };
            Ok(PresentationTargetCapabilities::new(
                vec![crate::api::format::TextureFormat::Rgba8Unorm],
                vec![PresentMode::Automatic],
                PresentationExtentControl::HostManaged { current },
            ))
        })
    }

    fn reconfigure_or_register_waker(
        &self,
        lease: GlPresentationLease,
        _: &PresentationConfiguration,
        _: &Waker,
    ) -> Poll<RhiResult<()>> {
        // Browser presentation configuration has no asynchronous native commit:
        // the canvas's current drawing buffer is observed on acquire.
        match self.with_state("WebGl2ExecutionDriver::reconfigure_presentation", |state| {
            if state.presentation.leases.contains_key(&lease.0) {
                Ok(())
            } else {
                Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGL2 presentation lease is no longer live",
                ))
            }
        }) {
            Ok(()) => Poll::Ready(Ok(())),
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn try_acquire(
        &self,
        lease: GlPresentationLease,
    ) -> Result<Option<GlAcquiredFramebuffer>, AcquireError> {
        self.acquire_frame(lease)
    }

    fn acquire_or_register_waker(
        &self,
        lease: GlPresentationLease,
        waker: &Waker,
    ) -> Poll<Result<GlAcquiredFramebuffer, AcquireError>> {
        match self.acquire_frame(lease) {
            Ok(Some(frame)) => Poll::Ready(Ok(frame)),
            Ok(None) => {
                let _ = self.with_state(
                    "WebGl2ExecutionDriver::acquire_or_register_waker",
                    |state| {
                        if let Some(lease) = state.presentation.leases.get_mut(&lease.0) {
                            lease.acquire_wakers.push(waker.clone());
                        }
                        Ok(())
                    },
                );
                Poll::Pending
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn abandon(&self, lease: GlPresentationLease, frame: AcquiredFrameId) -> RhiResult<()> {
        self.end_frame(lease, frame.serial(), false)
    }

    fn abandon_no_throw(&self, lease: GlPresentationLease, frame: AcquiredFrameId) {
        let _ = self.end_frame(lease, frame.serial(), false);
    }

    fn release_presentation(&self, lease: GlPresentationLease) {
        let _ = self.with_state("WebGl2ExecutionDriver::release_presentation", |state| {
            state.presentation.leases.remove(&lease.0);
            Ok(())
        });
    }

    fn present(&self, frame: GlAcquiredFramebuffer, receipt: PresentReceiptId) {
        let _ = self.with_state("WebGl2ExecutionDriver::present", |state| {
            let outcome = match state.finish_frame(frame.lease, frame.serial, true) {
                Ok(()) => PresentState::Accepted,
                Err(error) if error.kind() == RhiErrorKind::DeviceLost => {
                    PresentState::DeviceLost(DeviceLossInfo::new(error.to_string()))
                }
                Err(error) => PresentState::Failed(crate::api::presentation::PresentFailure::new(
                    error.to_string(),
                )),
            };
            state.presentation.presents.insert(receipt, outcome);
            if let Some(wakers) = state.presentation.present_wakers.get_mut(&receipt) {
                BrowserDriverState::wake(wakers);
            }
            Ok(())
        });
    }

    fn terminate_present(
        &self,
        frame: GlAcquiredFramebuffer,
        receipt: PresentReceiptId,
        state: PresentState,
    ) {
        let _ = self.with_state("WebGl2ExecutionDriver::terminate_present", |driver| {
            let _ = driver.finish_frame(frame.lease, frame.serial, false);
            driver.presentation.presents.insert(receipt, state);
            if let Some(wakers) = driver.presentation.present_wakers.get_mut(&receipt) {
                BrowserDriverState::wake(wakers);
            }
            Ok(())
        });
    }

    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        self.with_state("WebGl2ExecutionDriver::present_state", |state| {
            state
                .presentation
                .presents
                .get(&receipt)
                .cloned()
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "WebGL2 present receipt was never accepted",
                    )
                })
        })
    }

    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        waker: &Waker,
    ) -> RhiResult<PresentState> {
        self.with_state(
            "WebGl2ExecutionDriver::present_state_or_register_waker",
            |state| {
                let present = state
                    .presentation
                    .presents
                    .get(&receipt)
                    .cloned()
                    .ok_or_else(|| {
                        RhiError::new(
                            RhiErrorKind::InvalidUsage,
                            "WebGL2 present receipt was never accepted",
                        )
                    })?;
                if matches!(present, PresentState::Pending) {
                    state
                        .presentation
                        .present_wakers
                        .entry(receipt)
                        .or_default()
                        .push(waker.clone());
                }
                Ok(present)
            },
        )
    }

    fn device_lost(&self, info: &DeviceLossInfo) {
        let _ = self.with_state("WebGl2ExecutionDriver::device_lost", |state| {
            state.query_sets.clear();
            state.fail_pending_for_loss(info);
            Ok(())
        });
    }

    fn poll(&self) -> RhiResult<()> {
        self.with_executor("WebGl2ExecutionDriver::poll", |executor| {
            executor
                .assert_provider_ready("WebGl2ExecutionDriver::poll")
                .map_err(|error| map_gl_error(error, "WebGL2 poll"))?;
            // `flush` submits the browser command stream without claiming a
            // completion; completion lowering installs a fence before it
            // publishes an accepted serial.
            executor.raw.flush();
            executor
                .driver_error("WebGl2ExecutionDriver::poll")
                .map_err(|error| map_gl_error(error, "WebGL2 poll"))
        })
    }

    fn wait_idle(&self) -> RhiResult<()> {
        self.with_executor("WebGl2ExecutionDriver::wait_idle", |executor| {
            executor
                .assert_provider_ready("WebGl2ExecutionDriver::wait_idle")
                .map_err(|error| map_gl_error(error, "WebGL2 idle wait"))?;
            // WebGL2's `finish` is deliberately confined to the shutdown /
            // recovery-only `wait_idle` route; normal completion uses fences.
            executor.raw.finish();
            executor
                .driver_error("WebGl2ExecutionDriver::wait_idle")
                .map_err(|error| map_gl_error(error, "WebGL2 idle wait"))
        })
    }
}

/// The old browser executor allocates zero-based Fluxel table slots, while the
/// v13 driver carrier reserves zero as "no object".  Encoding is deliberately
/// `slot + 1`, not a browser object address or a GL integer name.
fn browser_binding_target(
    dimension: crate::backend::gl::api::GlTextureDimension,
) -> RhiResult<crate::backend::gl::api::GlTextureTarget> {
    use crate::backend::gl::api::{GlTextureDimension as Dimension, GlTextureTarget as Target};
    match dimension {
        Dimension::D2 => Ok(Target::D2),
        // The WebGL2 resource allocator currently proves only single-sample
        // 2D storage.  Refuse all other targets here even if a caller somehow
        // bypassed capability validation; binding them as D2 is not a view.
        _ => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "WebGL2 has no verified native texture-view target for this shape",
        )
        .at("WebGl2ExecutionDriver::create_texture_view")),
    }
}

fn object_name(slot: u32, operation: &'static str) -> RhiResult<GlObjectName> {
    let encoded = slot.checked_add(1).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::BackendFailure,
            "WebGL2 object-table slot cannot be encoded as a v13 object name",
        )
        .at(operation)
    })?;
    GlObjectName::new(encoded, operation)
}

fn map_gl_error(error: crate::backend::gl::api::GlError, context: &'static str) -> RhiError {
    let kind = match error {
        crate::backend::gl::api::GlError::ContextLost { .. }
        | crate::backend::gl::api::GlError::Disposed { .. }
        | crate::backend::gl::api::GlError::Poisoned { .. }
        | crate::backend::gl::api::GlError::InvalidLifecycle {
            lifecycle:
                GlContextLifecycle::Lost
                | GlContextLifecycle::Restoring
                | GlContextLifecycle::Poisoned
                | GlContextLifecycle::Disposed,
            ..
        } => RhiErrorKind::DeviceLost,
        crate::backend::gl::api::GlError::Unsupported { .. } => RhiErrorKind::Unsupported,
        crate::backend::gl::api::GlError::OutOfMemory { .. } => RhiErrorKind::OutOfMemory,
        crate::backend::gl::api::GlError::WrongContext { .. }
        | crate::backend::gl::api::GlError::StaleObject { .. } => RhiErrorKind::WrongDevice,
        _ => RhiErrorKind::BackendFailure,
    };
    RhiError::new(kind, format!("{context}: {error:?}"))
}

#[cfg(test)]
mod tests {
    use super::map_gl_error;
    use crate::api::error::RhiErrorKind;
    use crate::backend::gl::api::GlError;

    #[test]
    fn browser_loss_maps_to_the_v13_terminal_error() {
        assert_eq!(
            map_gl_error(GlError::ContextLost { operation: "test" }, "test").kind(),
            RhiErrorKind::DeviceLost
        );
    }

    #[test]
    fn a_lost_executor_refusal_remains_device_lost() {
        assert_eq!(
            map_gl_error(
                GlError::InvalidLifecycle {
                    operation: "test",
                    lifecycle: crate::backend::gl::api::GlContextLifecycle::Lost,
                },
                "test",
            )
            .kind(),
            RhiErrorKind::DeviceLost
        );
    }

    #[test]
    fn unsupported_webgl_route_does_not_become_backend_success() {
        assert_eq!(
            map_gl_error(
                GlError::Unsupported {
                    operation: "test",
                    reason: "WebGL2 has no compute stage",
                },
                "test",
            )
            .kind(),
            RhiErrorKind::Unsupported
        );
    }
}
