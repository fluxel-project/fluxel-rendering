//! The one mutable-state authority for one real GL context.
//!
//! This is deliberately a backend-private execution object.  It uses the v13
//! driver's stable object names and phase-A packets; no browser handle, raw GL
//! handle, `Arc` identity, or public context/session type participates in a
//! cache decision.  Native and browser owners each hold exactly one instance.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::api::identity::ObjectId;
use crate::backend::gl::platform::GlObjectName;

use super::{DriverKnowledge, ExecutionMode, ResourceRef, StateDomain, StateEvent};

/// A canonical immutable block prepared during phase A.  It is assigned by
/// the private pipeline/binding table, never synthesized from a lossy hash.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct CanonicalBlockId(u64);
impl CanonicalBlockId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
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
    pub(crate) program_link_epoch: u64,
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
        self.uniform_slots.clear();
        self.storage_slots.clear();
        self.image_slots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            program_link_epoch: 1,
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
}
