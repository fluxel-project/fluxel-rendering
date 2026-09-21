//! The one mutable-state authority for one real GL context.
//!
//! This is deliberately a backend-private execution object.  It uses the v13
//! driver's stable object names and phase-A packets; no browser handle, raw GL
//! handle, `Arc` identity, or public context/session type participates in a
//! cache decision.  Native and browser owners each hold exactly one instance.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::api::identity::ObjectId;
use crate::backend::gl::api::{ContextStamp, GlObjectKind, ObjectIdentity};
use crate::backend::gl::api::{
    GlColorTargetState, GlCullMode, GlDepthStencilState, GlFrontFace, GlMultisampleState,
    GlPrimitiveTopology, GlRasterPipeline, GlScissorRect, GlViewport, ProgramId,
};
use crate::backend::gl::platform::GlObjectName;

use super::{DriverKnowledge, ExecutionMode, ResourceRef, StateDomain, StateEvent};

/// A canonical immutable block prepared during phase A.  It is assigned by
/// the private pipeline/binding table, never synthesized from a lossy hash.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum CanonicalBlockId {
    /// Private monotonically allocated immutable-state block.  Such blocks are
    /// owned by this one `ContextState`, which is discarded on context loss.
    Allocated(u64),
    /// A real GL object identity.  Do not reduce this to a native name or an
    /// allocation slot: browser registries deliberately reuse slots and a
    /// restored context may reuse the same native names.
    Object {
        context: ContextStamp,
        slot: u32,
        generation: u32,
    },
    /// `glBindBufferRange` state includes the selected range, not merely the
    /// buffer.  Keeping all fields here prevents an equal-buffer fast path
    /// from skipping a legitimate offset/size update.
    UniformRange {
        context: ContextStamp,
        slot: u32,
        generation: u32,
        offset: u32,
        size: u32,
    },
}
impl CanonicalBlockId {
    pub(crate) const fn new(value: u64) -> Self {
        Self::Allocated(value)
    }
    pub(crate) const fn object<K: GlObjectKind>(identity: ObjectIdentity<K>) -> Self {
        Self::Object {
            context: identity.context,
            slot: identity.slot,
            generation: identity.generation,
        }
    }
    pub(crate) const fn uniform_range<K: GlObjectKind>(
        buffer: ObjectIdentity<K>,
        offset: u32,
        size: u32,
    ) -> Self {
        Self::UniformRange {
            context: buffer.context,
            slot: buffer.slot,
            generation: buffer.generation,
            offset,
            size,
        }
    }
}

/// Canonical blocks that a pipeline installs independently in GL.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RasterPipelineBlocks {
    pub(crate) program: CanonicalBlockId,
    pub(crate) raster: CanonicalBlockId,
    pub(crate) depth_stencil: CanonicalBlockId,
    pub(crate) blend: CanonicalBlockId,
    pub(crate) multisample: CanonicalBlockId,
}

/// Per-context structural canonicalization for immutable raster pipeline
/// leaves.  These IDs are deliberately *not* creation serials: a pipeline
/// switch can only skip an unchanged GL leaf when equal descriptors intern to
/// the same value.  `HashMap` equality resolves collisions, so no lossy hash
/// ever becomes a state-cache identity.
#[derive(Default)]
pub(crate) struct RasterPipelineBlockInterner {
    next: u64,
    programs: HashMap<ProgramId, CanonicalBlockId>,
    rasters: HashMap<RasterBlock, CanonicalBlockId>,
    depth_stencils: HashMap<Option<GlDepthStencilState>, CanonicalBlockId>,
    blends: HashMap<BlendBlock, CanonicalBlockId>,
    multisamples: HashMap<GlMultisampleState, CanonicalBlockId>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RasterBlock {
    topology: GlPrimitiveTopology,
    cull_mode: GlCullMode,
    front_face: GlFrontFace,
    viewport: GlViewport,
    scissor: Option<GlScissorRect>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BlendBlock {
    color_targets: Vec<GlColorTargetState>,
    blend_constant: [u32; 4],
}

impl RasterPipelineBlockInterner {
    pub(crate) fn new() -> Self {
        Self {
            // Zero stays invalid so a malformed transport/default value cannot
            // accidentally compare equal to a real canonical block.
            next: 1,
            ..Self::default()
        }
    }

    pub(crate) fn intern(
        &mut self,
        pipeline: &GlRasterPipeline,
        operation: &'static str,
    ) -> Result<RasterPipelineBlocks, &'static str> {
        let state = &pipeline.state;
        Ok(RasterPipelineBlocks {
            program: self.intern_program(pipeline.program, operation)?,
            raster: self.intern_raster(
                RasterBlock {
                    topology: state.topology,
                    cull_mode: state.cull_mode,
                    front_face: state.front_face,
                    viewport: state.viewport,
                    scissor: state.scissor,
                },
                operation,
            )?,
            depth_stencil: self.intern_depth_stencil(state.depth_stencil, operation)?,
            blend: self.intern_blend(
                BlendBlock {
                    color_targets: state.color_targets.clone(),
                    blend_constant: state.blend_constant,
                },
                operation,
            )?,
            multisample: self.intern_multisample(state.multisample, operation)?,
        })
    }

    fn intern_program(
        &mut self,
        value: ProgramId,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.programs, value, operation)
    }
    fn intern_raster(
        &mut self,
        value: RasterBlock,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.rasters, value, operation)
    }
    fn intern_depth_stencil(
        &mut self,
        value: Option<GlDepthStencilState>,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.depth_stencils, value, operation)
    }
    fn intern_blend(
        &mut self,
        value: BlendBlock,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.blends, value, operation)
    }
    fn intern_multisample(
        &mut self,
        value: GlMultisampleState,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.multisamples, value, operation)
    }
}

fn intern<T: Eq + std::hash::Hash>(
    next: &mut u64,
    table: &mut HashMap<T, CanonicalBlockId>,
    value: T,
    operation: &'static str,
) -> Result<CanonicalBlockId, &'static str> {
    if let Some(id) = table.get(&value) {
        return Ok(*id);
    }
    let id = CanonicalBlockId::new(*next);
    *next = next.checked_add(1).ok_or(operation)?;
    table.insert(value, id);
    Ok(id)
}

/// Phase-A's immutable pipeline packet. `identity` is the actual v13 pipeline
/// object name, so the exact-hit fast path is O(1) and collision-free.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RasterPipelinePacket {
    pub(crate) identity: GlObjectName,
    pub(crate) blocks: RasterPipelineBlocks,
}

/// Pass-FBO state is not one opaque slot: read and draw targets as well as the
/// draw-buffer routing are independently mutable GL state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PassPacket {
    pub(crate) draw_framebuffer: CanonicalBlockId,
    pub(crate) read_framebuffer: CanonicalBlockId,
    pub(crate) draw_buffers: CanonicalBlockId,
}
/// Pixel transfer has independent PACK/UNPACK settings and PBO bindings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PixelTransferPacket {
    pub(crate) pack: CanonicalBlockId,
    pub(crate) unpack: CanonicalBlockId,
    pub(crate) pack_buffer: CanonicalBlockId,
    pub(crate) unpack_buffer: CanonicalBlockId,
}

/// Differences the owner must lower. An exact pipeline hit has every flag
/// false. Geometry stays separate because a draw can replace the VAO after a
/// pipeline bind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RasterPipelineDiff {
    pub(crate) program: bool,
    pub(crate) raster: bool,
    pub(crate) depth_stencil: bool,
    pub(crate) blend: bool,
    pub(crate) multisample: bool,
}

/// Phase A must not associate two immutable packets with the same v13 object
/// identity. Rejecting that malformed private input keeps the O(1) fast path
/// sound instead of silently trusting a changed packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContextStateError {
    UnknownPipeline,
    PipelineIdentityConflict,
}
impl RasterPipelineDiff {
    pub(crate) const fn is_empty(self) -> bool {
        !self.program && !self.raster && !self.depth_stencil && !self.blend && !self.multisample
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InstalledRasterPipeline {
    identity: GlObjectName,
    blocks: RasterPipelineBlocks,
}

/// A bound group is dirty until the owner has successfully flushed its private
/// packet. Resource uses deliberately do not appear here: visibility/barrier
/// tracking is not binding state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BoundGroupPacket {
    pub(crate) group: ObjectId,
    pub(crate) name: GlObjectName,
    pub(crate) index: u32,
    pub(crate) dynamic_offsets: Vec<u32>,
    /// Full program provenance, not a lossy `(slot, generation)` packing.
    /// Bind-group application must be revisited after a context replacement
    /// even where a provider happens to reuse both native name and slot.
    pub(crate) program_identity: CanonicalBlockId,
    pub(crate) dependencies: BTreeSet<ResourceRef>,
}

/// Immutable work to flush before a draw or dispatch. The owner calls
/// `acknowledge_bindings` only after all listed packets reached GL.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingFlush {
    pub(crate) groups: Vec<BoundGroupPacket>,
}
impl BindingFlush {
    pub(crate) const fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Private, structural keys for derived VAO/FBO cache entries. They are not
/// public GL objects and must be invalidated through their dependencies.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DerivedCacheKey {
    pub(crate) kind: DerivedCacheKind,
    pub(crate) canonical: CanonicalBlockId,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum DerivedCacheKind {
    VertexArray,
    DrawFramebuffer,
    ReadFramebuffer,
}

/// One real native/browser context's state authority. Submission, completion
/// and presentation remain separate execution concerns, but their context
/// owner calls `raw_access`/`event` whenever they touch mutable GL state.
pub(crate) struct ContextState {
    mode: ExecutionMode,
    pass: DriverKnowledge<PassPacket>,
    pipeline: DriverKnowledge<InstalledRasterPipeline>,
    current_program: DriverKnowledge<CanonicalBlockId>,
    geometry: DriverKnowledge<CanonicalBlockId>,
    compute: DriverKnowledge<CanonicalBlockId>,
    pixel_transfer: DriverKnowledge<PixelTransferPacket>,
    query: DriverKnowledge<CanonicalBlockId>,
    bindings: BTreeMap<u32, BoundGroupPacket>,
    dirty_bindings: BTreeSet<u32>,
    active_texture: DriverKnowledge<u32>,
    texture_slots: BTreeMap<u32, CanonicalBlockId>,
    // Samplers are independent texture-unit state in both desktop GL and
    // WebGL2.  They must not piggy-back on `texture_slots`: changing only a
    // compare/filter object is still a real `bind_sampler` mutation.
    sampler_slots: BTreeMap<u32, CanonicalBlockId>,
    uniform_slots: BTreeMap<u32, CanonicalBlockId>,
    storage_slots: BTreeMap<u32, CanonicalBlockId>,
    image_slots: BTreeMap<u32, CanonicalBlockId>,
    pipeline_packets: HashMap<GlObjectName, RasterPipelineBlocks>,
    derived: HashMap<DerivedCacheKey, BTreeSet<ResourceRef>>,
}

impl ContextState {
    pub(crate) fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            pass: DriverKnowledge::Unknown,
            pipeline: DriverKnowledge::Unknown,
            current_program: DriverKnowledge::Unknown,
            geometry: DriverKnowledge::Unknown,
            compute: DriverKnowledge::Unknown,
            pixel_transfer: DriverKnowledge::Unknown,
            query: DriverKnowledge::Unknown,
            bindings: BTreeMap::new(),
            dirty_bindings: BTreeSet::new(),
            active_texture: DriverKnowledge::Unknown,
            texture_slots: BTreeMap::new(),
            sampler_slots: BTreeMap::new(),
            uniform_slots: BTreeMap::new(),
            storage_slots: BTreeMap::new(),
            image_slots: BTreeMap::new(),
            pipeline_packets: HashMap::new(),
            derived: HashMap::new(),
        }
    }
    /// Pass begin carries load/clear/order semantics and is therefore never a
    /// cache hit. `pass_end`, draw, dispatch, query and barrier follow the same
    /// rule: this authority never elides execution commands.
    /// Returns `true` unconditionally so an owner cannot accidentally treat a
    /// structurally equal pass as a skipped GL command.
    pub(crate) const fn prepare_pass(&self, _: PassPacket) -> bool {
        true
    }
    pub(crate) fn commit_pass(&mut self, pass: PassPacket) {
        self.pass.set(pass);
    }
    pub(crate) fn pass_failed(&mut self) {
        self.pass.invalidate();
    }
    pub(crate) fn end_pass(&mut self) {
        self.pass.invalidate();
    }
    /// These are execution/visibility operations, never setters. Their `true`
    /// result is intentionally not cached and means the owner must emit GL.
    pub(crate) const fn draw(&self) -> bool {
        true
    }
    pub(crate) const fn dispatch(&self) -> bool {
        true
    }
    pub(crate) const fn clear(&self) -> bool {
        true
    }
    pub(crate) const fn query_command(&self) -> bool {
        true
    }
    pub(crate) const fn barrier(&self) -> bool {
        true
    }
    /// Registers a phase-A pipeline packet. Re-registration is legal only for
    /// exactly the same immutable packet, which makes identity hits sound.
    pub(crate) fn register_pipeline(
        &mut self,
        packet: RasterPipelinePacket,
    ) -> Result<(), ContextStateError> {
        match self.pipeline_packets.get(&packet.identity) {
            None => {
                self.pipeline_packets.insert(packet.identity, packet.blocks);
                Ok(())
            }
            Some(blocks) if *blocks == packet.blocks => Ok(()),
            Some(_) => Err(ContextStateError::PipelineIdentityConflict),
        }
    }
    pub(crate) fn prepare_pipeline(
        &self,
        wanted: RasterPipelinePacket,
    ) -> Result<RasterPipelineDiff, ContextStateError> {
        if self.pipeline_packets.get(&wanted.identity) != Some(&wanted.blocks) {
            return Err(if self.pipeline_packets.contains_key(&wanted.identity) {
                ContextStateError::PipelineIdentityConflict
            } else {
                ContextStateError::UnknownPipeline
            });
        }
        if self.mode.may_skip()
            && self
                .pipeline
                .get()
                .is_some_and(|have| have.identity == wanted.identity)
            && self.current_program.agrees(&wanted.blocks.program)
        {
            return Ok(RasterPipelineDiff::default());
        }
        let prior = self.pipeline.get().copied();
        let diff = RasterPipelineDiff {
            // Compute and raster share GL_CURRENT_PROGRAM. A pipeline identity
            // miss caused solely by compute must still restore this leaf.
            program: !self.current_program.agrees(&wanted.blocks.program),
            raster: prior.is_none_or(|have| have.blocks.raster != wanted.blocks.raster),
            depth_stencil: prior
                .is_none_or(|have| have.blocks.depth_stencil != wanted.blocks.depth_stencil),
            blend: prior.is_none_or(|have| have.blocks.blend != wanted.blocks.blend),
            multisample: prior
                .is_none_or(|have| have.blocks.multisample != wanted.blocks.multisample),
        };
        Ok(diff)
    }
    pub(crate) fn commit_pipeline(&mut self, wanted: RasterPipelinePacket) {
        self.current_program.set(wanted.blocks.program);
        self.pipeline.set(InstalledRasterPipeline {
            identity: wanted.identity,
            blocks: wanted.blocks,
        });
    }
    /// Must be called when lowering a reported leaf fails after a partial GL
    /// mutation. It never preserves a possibly stale exact-pipeline hit.
    pub(crate) fn pipeline_failed(&mut self) {
        self.pipeline.invalidate();
        self.current_program.invalidate();
    }
    pub(crate) fn prepare_geometry(&self, wanted: CanonicalBlockId) -> bool {
        if self.mode.may_skip() && self.geometry.agrees(&wanted) {
            false
        } else {
            true
        }
    }
    pub(crate) fn commit_geometry(&mut self, wanted: CanonicalBlockId) {
        self.geometry.set(wanted);
    }
    pub(crate) fn geometry_failed(&mut self) {
        self.geometry.invalidate();
    }
    pub(crate) fn prepare_compute_program(&self, wanted: CanonicalBlockId) -> bool {
        !(self.mode.may_skip() && self.current_program.agrees(&wanted))
    }
    pub(crate) fn commit_compute_program(&mut self, wanted: CanonicalBlockId) {
        self.current_program.set(wanted);
        self.compute.set(wanted);
    }
    pub(crate) fn prepare_pixel_transfer(&self, wanted: PixelTransferPacket) -> bool {
        if self.mode.may_skip() && self.pixel_transfer.agrees(&wanted) {
            false
        } else {
            true
        }
    }
    pub(crate) fn commit_pixel_transfer(&mut self, wanted: PixelTransferPacket) {
        self.pixel_transfer.set(wanted);
    }
    /// Query begin/end are commands and are always emitted. This tracks only
    /// the active target for validation after a successful native begin/end.
    pub(crate) const fn prepare_query_command(&self, _: CanonicalBlockId) -> bool {
        true
    }
    pub(crate) fn commit_query_begin(&mut self, target: CanonicalBlockId) {
        self.query.set(target);
    }
    pub(crate) fn commit_query_end(&mut self) {
        self.query.invalidate();
    }
    pub(crate) fn query_failed(&mut self) {
        self.query.invalidate();
    }
    pub(crate) fn stage_bind_group(&mut self, packet: BoundGroupPacket) {
        let changed = self.bindings.get(&packet.index) != Some(&packet);
        if changed || !self.mode.may_skip() {
            self.dirty_bindings.insert(packet.index);
        }
        self.bindings.insert(packet.index, packet);
    }
    /// `active_texture` has exactly one authority. The owner prepares this
    /// before a texture slot bind and commits it only after `glActiveTexture`.
    pub(crate) fn prepare_texture_slot(
        &self,
        unit: u32,
        binding: CanonicalBlockId,
    ) -> (bool, bool) {
        (
            !(self.mode.may_skip() && self.active_texture.agrees(&unit)),
            !(self.mode.may_skip()
                && self
                    .texture_slots
                    .get(&unit)
                    .is_some_and(|known| *known == binding)),
        )
    }
    pub(crate) fn commit_texture_slot(&mut self, unit: u32, binding: CanonicalBlockId) {
        self.active_texture.set(unit);
        self.texture_slots.insert(unit, binding);
    }
    /// `glBindSampler` has no active-texture prerequisite, but is otherwise
    /// exactly the same kind of generation-safe unit cache as a texture bind.
    /// The caller commits only after the native/browser operation succeeds.
    pub(crate) fn prepare_sampler_slot(&self, unit: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .sampler_slots
                .get(&unit)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_sampler_slot(&mut self, unit: u32, binding: CanonicalBlockId) {
        self.sampler_slots.insert(unit, binding);
    }
    pub(crate) fn prepare_uniform_slot(&self, index: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .uniform_slots
                .get(&index)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_uniform_slot(&mut self, index: u32, binding: CanonicalBlockId) {
        self.uniform_slots.insert(index, binding);
    }
    pub(crate) fn prepare_storage_slot(&self, index: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .storage_slots
                .get(&index)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_storage_slot(&mut self, index: u32, binding: CanonicalBlockId) {
        self.storage_slots.insert(index, binding);
    }
    pub(crate) fn prepare_image_slot(&self, index: u32, binding: CanonicalBlockId) -> bool {
        !(self.mode.may_skip()
            && self
                .image_slots
                .get(&index)
                .is_some_and(|known| *known == binding))
    }
    pub(crate) fn commit_image_slot(&mut self, index: u32, binding: CanonicalBlockId) {
        self.image_slots.insert(index, binding);
    }
    pub(crate) fn binding_flush(&self) -> BindingFlush {
        BindingFlush {
            groups: self
                .dirty_bindings
                .iter()
                .filter_map(|index| self.bindings.get(index).cloned())
                .collect(),
        }
    }
    pub(crate) fn acknowledge_bindings(&mut self, flushed: &BindingFlush) {
        for packet in &flushed.groups {
            self.dirty_bindings.remove(&packet.index);
        }
    }
    /// Removes a logical bind-group from the applied-state mirror before its
    /// private packet is retired.  Keeping it would let a later draw treat a
    /// destroyed packet as already flushed.
    pub(crate) fn retire_bind_group(&mut self, name: GlObjectName) {
        self.bindings.retain(|_, packet| packet.name != name);
        self.dirty_bindings
            .retain(|index| self.bindings.contains_key(index));
    }
    pub(crate) fn register_derived(
        &mut self,
        key: DerivedCacheKey,
        dependencies: BTreeSet<ResourceRef>,
    ) {
        self.derived.insert(key, dependencies);
    }
    pub(crate) fn event(&mut self, event: StateEvent) {
        match event {
            StateEvent::BufferRetired(id) => self.retire(ResourceRef::Buffer(id)),
            StateEvent::TextureRetired(id) => self.retire(ResourceRef::Texture(id)),
            StateEvent::RenderbufferRetired(id) => self.retire(ResourceRef::Renderbuffer(id)),
            StateEvent::FramebufferRetired(id) => self.retire(ResourceRef::Framebuffer(id)),
            StateEvent::QueryRetired(id) => self.retire(ResourceRef::Query(id)),
            StateEvent::SamplerRetired(id) => self.retire(ResourceRef::Sampler(id)),
            StateEvent::ProgramRetired(id) => self.retire(ResourceRef::Program(id)),
            StateEvent::VertexArrayRetired(id) => self.retire(ResourceRef::VertexArray(id)),
            StateEvent::DomainFailed(domain) => self.invalidate_domain(domain),
            StateEvent::ScopedRawAccess(access) => {
                for domain in access.domains().iter() {
                    self.invalidate_domain(domain);
                }
            }
            StateEvent::ContextLost
            | StateEvent::ContextRestored(_)
            | StateEvent::DeviceReplaced(_) => self.forget_context(),
        }
    }
    fn retire(&mut self, resource: ResourceRef) {
        self.derived
            .retain(|_, dependencies| !dependencies.contains(&resource));
        // A retired resource must not survive in a future flush packet.
        self.bindings
            .retain(|_, packet| !packet.dependencies.contains(&resource));
        self.dirty_bindings
            .retain(|index| self.bindings.contains_key(index));
        // Binding slots are state caches too.  Erase a matching object now;
        // although a complete identity would prevent a false hit after slot
        // reuse, eagerly forgetting it also prevents stale bookkeeping from
        // hiding a retirement while an application owns no replacement yet.
        self.texture_slots
            .retain(|_, value| !value.references(resource));
        self.sampler_slots
            .retain(|_, value| !value.references(resource));
        self.uniform_slots
            .retain(|_, value| !value.references(resource));
        self.storage_slots
            .retain(|_, value| !value.references(resource));
        self.image_slots
            .retain(|_, value| !value.references(resource));
        match resource {
            ResourceRef::Program(_) => {
                self.pipeline.invalidate();
                self.compute.invalidate();
                self.current_program.invalidate();
            }
            ResourceRef::VertexArray(_) | ResourceRef::Buffer(_) => self.geometry.invalidate(),
            ResourceRef::Texture(_)
            | ResourceRef::Renderbuffer(_)
            | ResourceRef::Framebuffer(_) => {
                self.pass.invalidate();
                self.pixel_transfer.invalidate();
            }
            ResourceRef::Query(_) => self.query.invalidate(),
            ResourceRef::Sampler(_) => self.pixel_transfer.invalidate(),
            _ => {}
        }
    }
    fn invalidate_domain(&mut self, domain: StateDomain) {
        match domain {
            StateDomain::PassFramebuffer => self.pass.invalidate(),
            StateDomain::RasterPipeline => {
                self.pipeline.invalidate();
                self.current_program.invalidate();
            }
            StateDomain::Geometry => self.geometry.invalidate(),
            StateDomain::Bindings => {
                self.dirty_bindings.extend(self.bindings.keys().copied());
                self.active_texture.invalidate();
                self.texture_slots.clear();
                self.sampler_slots.clear();
                self.uniform_slots.clear();
                self.storage_slots.clear();
                self.image_slots.clear();
            }
            StateDomain::Compute => {
                self.compute.invalidate();
                self.current_program.invalidate();
            }
            StateDomain::PixelTransferCopy => self.pixel_transfer.invalidate(),
            StateDomain::Query => self.query.invalidate(),
            StateDomain::DerivedCaches => self.derived.clear(),
        }
    }
    fn forget_context(&mut self) {
        self.pass.invalidate();
        self.pipeline.invalidate();
        self.current_program.invalidate();
        self.geometry.invalidate();
        self.compute.invalidate();
        self.pixel_transfer.invalidate();
        self.query.invalidate();
        self.dirty_bindings.extend(self.bindings.keys().copied());
        self.derived.clear();
        self.pipeline_packets.clear();
        self.active_texture.invalidate();
        self.texture_slots.clear();
        self.sampler_slots.clear();
        self.uniform_slots.clear();
        self.storage_slots.clear();
        self.image_slots.clear();
    }
}

impl CanonicalBlockId {
    fn references(self, resource: ResourceRef) -> bool {
        let (context, slot, generation) = match self {
            Self::Object {
                context,
                slot,
                generation,
            }
            | Self::UniformRange {
                context,
                slot,
                generation,
                ..
            } => (context, slot, generation),
            Self::Allocated(_) => return false,
        };
        // The cache is intentionally type-erased.  A same-number identity of
        // another GL object kind may cause an extra bind, never an unsafe skip.
        // Context/slot/generation equality is nevertheless mandatory.
        (match resource {
            ResourceRef::Buffer(id) => (id.context, id.slot, id.generation),
            ResourceRef::Texture(id) => (id.context, id.slot, id.generation),
            ResourceRef::Renderbuffer(id) => (id.context, id.slot, id.generation),
            ResourceRef::Sampler(id) => (id.context, id.slot, id.generation),
            ResourceRef::Shader(id) => (id.context, id.slot, id.generation),
            ResourceRef::Program(id) => (id.context, id.slot, id.generation),
            ResourceRef::VertexArray(id) => (id.context, id.slot, id.generation),
            ResourceRef::Framebuffer(id) => (id.context, id.slot, id.generation),
            ResourceRef::Query(id) => (id.context, id.slot, id.generation),
            ResourceRef::Sync(id) => (id.context, id.slot, id.generation),
        }) == (context, slot, generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gl::api::{
        BufferId, ContextEpoch, ContextStamp, DeviceIdentity, GlCullMode, GlFrontFace,
        GlMultisampleState, GlPrimitiveTopology, GlRasterPipeline, GlRasterState, GlViewport,
        ProgramId, SamplerId, VertexArrayId,
    };

    fn stamp(epoch: ContextEpoch) -> ContextStamp {
        ContextStamp::new(DeviceIdentity::new(7).unwrap(), epoch)
    }
    fn name(value: u32) -> GlObjectName {
        GlObjectName::new(value, "state test").unwrap()
    }
    fn packet(identity: u32, base: u64) -> RasterPipelinePacket {
        RasterPipelinePacket {
            identity: name(identity),
            blocks: RasterPipelineBlocks {
                program: CanonicalBlockId::new(base),
                raster: CanonicalBlockId::new(base + 1),
                depth_stencil: CanonicalBlockId::new(base + 2),
                blend: CanonicalBlockId::new(base + 3),
                multisample: CanonicalBlockId::new(base + 4),
            },
        }
    }
    fn pipeline(program_slot: u32, blend_constant: [u32; 4]) -> GlRasterPipeline {
        let context = stamp(ContextEpoch::INITIAL);
        GlRasterPipeline {
            program: ProgramId::new(context, program_slot, 1),
            vertex_array: VertexArrayId::new(context, program_slot, 1),
            state: GlRasterState {
                topology: GlPrimitiveTopology::Triangles,
                cull_mode: GlCullMode::None,
                front_face: GlFrontFace::CounterClockwise,
                depth_stencil: None,
                color_targets: vec![],
                multisample: GlMultisampleState {
                    sample_count: 1,
                    alpha_to_coverage_enabled: false,
                    sample_mask: u32::MAX,
                },
                viewport: GlViewport {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    min_depth: 0.0f32.to_bits(),
                    max_depth: 1.0f32.to_bits(),
                },
                scissor: None,
                blend_constant,
            },
        }
    }
    #[test]
    fn pipeline_identity_is_a_collision_free_first_gate() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        state.register_pipeline(packet(1, 10)).unwrap();
        assert!(!state.prepare_pipeline(packet(1, 10)).unwrap().is_empty());
        state.commit_pipeline(packet(1, 10));
        assert_eq!(
            state.prepare_pipeline(packet(1, 99)),
            Err(ContextStateError::PipelineIdentityConflict)
        );
    }
    #[test]
    fn pipeline_miss_diffs_canonical_blocks_not_the_whole_pipeline() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        state.register_pipeline(packet(1, 10)).unwrap();
        let replacement = RasterPipelinePacket {
            identity: name(2),
            blocks: RasterPipelineBlocks {
                blend: CanonicalBlockId::new(99),
                ..packet(1, 10).blocks
            },
        };
        state.register_pipeline(replacement).unwrap();
        state.commit_pipeline(packet(1, 10));
        let diff = state.prepare_pipeline(replacement).unwrap();
        assert_eq!(
            diff,
            RasterPipelineDiff {
                blend: true,
                ..RasterPipelineDiff::default()
            }
        );
    }
    #[test]
    fn structurally_equal_pipeline_leaves_share_canonical_ids() {
        let mut interner = RasterPipelineBlockInterner::new();
        let first = pipeline(1, [0; 4]);
        // Distinct pipeline object/VAO carriers must not prevent sharing the
        // immutable program and fixed-function leaves.
        let mut second = pipeline(1, [0; 4]);
        second.vertex_array = VertexArrayId::new(stamp(ContextEpoch::INITIAL), 99, 1);
        let a = interner.intern(&first, "test exhausted").unwrap();
        let b = interner.intern(&second, "test exhausted").unwrap();
        assert_eq!(a, b);
    }
    #[test]
    fn one_changed_leaf_produces_one_lowering_call_and_identity_hit_none() {
        let mut interner = RasterPipelineBlockInterner::new();
        let first = pipeline(1, [0; 4]);
        let changed_blend = pipeline(1, [1, 0, 0, 0]);
        let first_packet = RasterPipelinePacket {
            identity: name(41),
            blocks: interner.intern(&first, "test exhausted").unwrap(),
        };
        let changed_packet = RasterPipelinePacket {
            identity: name(42),
            blocks: interner.intern(&changed_blend, "test exhausted").unwrap(),
        };
        let mut state = ContextState::new(ExecutionMode::Optimized);
        state.register_pipeline(first_packet).unwrap();
        state.register_pipeline(changed_packet).unwrap();
        state.commit_pipeline(first_packet);
        let diff = state.prepare_pipeline(changed_packet).unwrap();
        assert_eq!(
            diff,
            RasterPipelineDiff {
                blend: true,
                ..RasterPipelineDiff::default()
            }
        );
        state.commit_pipeline(changed_packet);
        assert_eq!(
            state.prepare_pipeline(changed_packet).unwrap(),
            RasterPipelineDiff::default(),
            "exact pipeline identity fast path must issue no leaf install"
        );
    }
    #[test]
    fn dynamic_draw_invalidation_cannot_take_a_stale_pipeline_identity_hit() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let raster = packet(1, 10);
        state.register_pipeline(raster).unwrap();
        state.commit_pipeline(raster);
        assert!(state.prepare_pipeline(raster).unwrap().is_empty());
        // A viewport/scissor/blend/stencil value belongs to a draw, not the
        // immutable public pipeline packet. The native lowering invalidates
        // this domain before installing the dynamically modified state.
        state.event(StateEvent::DomainFailed(StateDomain::RasterPipeline));
        let diff = state.prepare_pipeline(raster).unwrap();
        assert!(
            diff.program && diff.raster && diff.depth_stencil && diff.blend && diff.multisample
        );
    }
    #[test]
    fn compute_program_change_forces_raster_program_restore() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let raster = packet(1, 10);
        state.register_pipeline(raster).unwrap();
        state.commit_pipeline(raster);
        state.commit_compute_program(CanonicalBlockId::new(80));
        assert!(state.prepare_pipeline(raster).unwrap().program);
    }
    #[test]
    fn compute_domain_invalidation_forces_raster_program_restore() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let raster = packet(1, 10);
        state.register_pipeline(raster).unwrap();
        state.commit_pipeline(raster);
        state.event(StateEvent::DomainFailed(StateDomain::Compute));
        assert!(state.prepare_pipeline(raster).unwrap().program);
    }
    #[test]
    fn bind_groups_flush_only_after_acknowledgement() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        state.stage_bind_group(BoundGroupPacket {
            group: ObjectId::new(3),
            name: name(3),
            index: 0,
            dynamic_offsets: vec![4],
            program_identity: CanonicalBlockId::new(1),
            dependencies: BTreeSet::new(),
        });
        let flush = state.binding_flush();
        assert_eq!(flush.groups.len(), 1);
        assert_eq!(state.binding_flush(), flush);
        state.acknowledge_bindings(&flush);
        assert!(state.binding_flush().is_empty());
    }
    #[test]
    fn pass_begin_is_never_elided() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let pass = PassPacket {
            draw_framebuffer: CanonicalBlockId::new(1),
            read_framebuffer: CanonicalBlockId::new(2),
            draw_buffers: CanonicalBlockId::new(3),
        };
        assert!(state.prepare_pass(pass));
        state.commit_pass(pass);
        assert!(state.prepare_pass(pass));
        assert!(state.pass.is_known());
    }
    #[test]
    fn uniform_cache_key_includes_buffer_generation_and_range() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let first = BufferId::new(stamp(ContextEpoch::INITIAL), 4, 1);
        let same_slot_new_generation = BufferId::new(stamp(ContextEpoch::INITIAL), 4, 2);
        let first_range = CanonicalBlockId::uniform_range(first, 256, 64);
        state.commit_uniform_slot(0, first_range);
        assert!(!state.prepare_uniform_slot(0, first_range));
        assert!(state.prepare_uniform_slot(0, CanonicalBlockId::uniform_range(first, 320, 64)));
        assert!(state.prepare_uniform_slot(0, CanonicalBlockId::uniform_range(first, 256, 128)));
        assert!(state.prepare_uniform_slot(
            0,
            CanonicalBlockId::uniform_range(same_slot_new_generation, 256, 64)
        ));
    }
    #[test]
    fn sampler_cache_is_unit_scoped_and_generation_safe() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let first = SamplerId::new(stamp(ContextEpoch::INITIAL), 5, 1);
        let recycled = SamplerId::new(stamp(ContextEpoch::INITIAL), 5, 2);
        let first_key = CanonicalBlockId::object(first);
        state.commit_sampler_slot(3, first_key);
        assert!(
            !state.prepare_sampler_slot(3, first_key),
            "identical sampler/unit is the only bind elision"
        );
        assert!(state.prepare_sampler_slot(4, first_key));
        assert!(state.prepare_sampler_slot(3, CanonicalBlockId::object(recycled)));
        state.event(StateEvent::SamplerRetired(first));
        assert!(state.prepare_sampler_slot(3, first_key));
    }
    #[test]
    fn raw_binding_access_forces_sampler_rebind() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let sampler = SamplerId::new(stamp(ContextEpoch::INITIAL), 5, 1);
        let key = CanonicalBlockId::object(sampler);
        state.commit_sampler_slot(3, key);
        state.event(StateEvent::DomainFailed(StateDomain::Bindings));
        assert!(state.prepare_sampler_slot(3, key));
    }
    #[test]
    fn retirement_erases_cached_object_binding_and_dependent_group() {
        let mut state = ContextState::new(ExecutionMode::Optimized);
        let buffer = BufferId::new(stamp(ContextEpoch::INITIAL), 8, 3);
        let key = CanonicalBlockId::uniform_range(buffer, 0, 64);
        state.commit_uniform_slot(2, key);
        let mut dependencies = BTreeSet::new();
        dependencies.insert(ResourceRef::Buffer(buffer));
        state.stage_bind_group(BoundGroupPacket {
            group: ObjectId::new(11),
            name: name(11),
            index: 0,
            dynamic_offsets: vec![],
            program_identity: CanonicalBlockId::new(1),
            dependencies,
        });
        state.acknowledge_bindings(&state.binding_flush());
        state.event(StateEvent::BufferRetired(buffer));
        assert!(state.prepare_uniform_slot(2, key));
        assert!(state.binding_flush().is_empty());
    }
}
