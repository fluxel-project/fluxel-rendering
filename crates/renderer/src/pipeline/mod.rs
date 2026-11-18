//! Frame policy and preparation for the first Forward consumer.

mod composition;
pub(crate) use composition::with_forward_feature_scope;
pub use composition::{ForwardCompositionContext, ForwardCompositionPass, ForwardFeaturePass};
mod composition_pair;
pub use composition_pair::{ForwardPairCompositionContext, ForwardPairCompositionPass};
mod policy;
pub use policy::FramePipeline;
pub(crate) use policy::read_all;
mod sorted_phase;
pub use sorted_phase::SortedForwardPass;
mod deferred;
pub use deferred::{
    DeferredCpuPipeline, DeferredDeformationPass, DeferredGeometryPass, DeferredLightingPass,
    DeferredPhaseContext,
};
mod forward_topology;
pub use forward_topology::FrameTopology;
mod oit;
pub use oit::DualDepthPeelingConfig;

use fluxel_material::{MaterialInstance, SurfaceSideOrientation};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use crate::{FrameSnapshot, RenderBounds, RenderId, RenderRecord, RenderViewId, SceneError};
use fluxel_rhi::backend::ResourceBackend;

/// A renderer-owned, single-use declaration context for the selected Forward
/// phase. It has no public constructor and exposes neither Graph resources nor
/// backend recording. A contribution declares accesses immediately on the
/// renderer's private Graph pass; it cannot retain a manifest or Graph handle.
pub struct ForwardPhaseContext<'context, 'graph, 'frame, B: ResourceBackend> {
    driver: &'context mut crate::graph::ForwardPhaseDriver<'graph, 'frame, B>,
}

/// The bounded opaque Forward access authority.
///
/// It is available only while [`ForwardPhaseContext::opaque`] is declaring the
/// real renderer-owned Graph pass. It neither exposes resource handles nor
/// permits command recording.
pub struct OpaqueForwardPass<'callback, 'pass, 'builder, 'frame, B: ResourceBackend> {
    pass: &'callback mut crate::graph::OpaqueForwardPass<'pass, 'builder, 'frame, B>,
}

impl<B: ResourceBackend> OpaqueForwardPass<'_, '_, '_, '_, B> {
    /// Declares the dependency on the frame upload pass.
    pub fn depends_on_uploads(&mut self) -> Result<(), PipelineError> {
        self.pass
            .depends_on_uploads()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the physical vertex-stream reads for the prepared draws.
    pub fn read_vertex_streams(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_vertex_streams()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the prepared frame-uniform reads.
    pub fn read_frame_uniforms(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_frame_uniforms()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the sampled texture reads for the prepared draws.
    pub fn read_sampled_textures(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_sampled_textures()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the depth attachment selected once during frame preparation.
    /// Disabled depth performs no resource access; enabled depth must be declared.
    pub fn write_selected_depth(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_selected_depth()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the main color attachment write.
    pub fn write_main_color(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_main_color()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}

/// The bounded clear-only Forward access authority.
///
/// It is available only while [`ForwardPhaseContext::clear`] is declaring the
/// real renderer-owned Graph pass.
pub struct ClearForwardPass<'callback, 'pass, 'builder, 'frame, B: ResourceBackend> {
    pass: &'callback mut crate::graph::ClearForwardPass<'pass, 'builder, 'frame, B>,
}

impl<B: ResourceBackend> ClearForwardPass<'_, '_, '_, '_, B> {
    /// Declares the dependency on the frame upload pass.
    pub fn depends_on_uploads(&mut self) -> Result<(), PipelineError> {
        self.pass
            .depends_on_uploads()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the main color attachment write.
    pub fn write_main_color(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_main_color()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}

impl<'context, 'graph, 'frame, B: ResourceBackend>
    ForwardPhaseContext<'context, 'graph, 'frame, B>
{
    pub(crate) fn new(
        driver: &'context mut crate::graph::ForwardPhaseDriver<'graph, 'frame, B>,
    ) -> Self {
        Self { driver }
    }

    /// The renderer-selected topology for this one real declaration. Default
    /// policy methods must consume this fact rather than evaluate selection a
    /// second time.
    pub(crate) const fn topology(&self) -> FrameTopology {
        self.driver.expected
    }

    /// Declares the opaque Forward phase directly on the private Graph.
    ///
    /// The callback must declare each supplied access exactly once. On return,
    /// the Graph transaction rejects a missing, duplicate, or wrong-phase
    /// declaration before Graph compilation proceeds.
    pub fn opaque<F>(&mut self, declare: F) -> Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut OpaqueForwardPass<'callback, 'pass, 'builder, 'frame, B>,
        ) -> Result<(), PipelineError>,
    {
        self.driver.declare_opaque(|pass| {
            let mut pass = OpaqueForwardPass { pass };
            declare(&mut pass)
        })
    }

    /// Declares the selected sorted Forward phase on the same real Graph
    /// transaction. Calling `opaque` for this topology is rejected.
    pub fn sorted<F>(&mut self, declare: F) -> Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut SortedForwardPass<'callback, 'pass, 'builder, 'frame, B>,
        ) -> Result<(), PipelineError>,
    {
        self.driver.declare_sorted(|pass| {
            let mut pass = SortedForwardPass { pass };
            declare(&mut pass)
        })
    }

    /// Declares the clear-only Forward phase directly on the private Graph.
    pub fn clear<F>(&mut self, declare: F) -> Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut ClearForwardPass<'callback, 'pass, 'builder, 'frame, B>,
        ) -> Result<(), PipelineError>,
    {
        self.driver.declare_clear(|pass| {
            let mut pass = ClearForwardPass { pass };
            declare(&mut pass)
        })
    }
}

/// A statically selected provider that filters scene candidates for one view.
///
/// The provider may choose a local visibility algorithm. It has no Graph,
/// resource, or submission authority. The pipeline validates every returned ID
/// against the supplied immutable snapshot before it publishes a frame.
pub trait VisibilityProvider<R> {
    /// Filters the deterministic list of live record identities.
    fn visible(
        &self,
        snapshot: &FrameSnapshot<R>,
        volume: RenderBounds,
        candidates: &[RenderId],
    ) -> Result<Vec<RenderId>, PipelineError>;
}

/// The depth contract selected for one Forward frame.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ForwardDepthMode {
    /// Preserve the existing color-only Forward product.
    #[default]
    Disabled,
    /// Own `Depth32Float` and rasterize with write + `LessEqual`.
    Depth32LessEqualWrite,
}

/// A controlled CPU Forward pipeline with a concrete visibility provider.
#[derive(Clone, Debug, Default)]
pub struct ForwardCpuPipeline<V = BoundsVisibility> {
    visibility: V,
    depth_mode: ForwardDepthMode,
    sorted_transparency: bool,
    material_instances: Option<Arc<HashMap<crate::ForwardDrawSource, MaterialInstance>>>,
    mesh_side_orientations: Option<Arc<HashMap<crate::ForwardDrawSource, SurfaceSideOrientation>>>,
    mesh_visibilities: Option<Arc<HashMap<crate::ForwardDrawSource, crate::MeshVisibility>>>,
}

impl<V> ForwardCpuPipeline<V> {
    /// Creates a pipeline from one concrete, statically dispatched provider.
    pub const fn new(visibility: V) -> Self {
        Self {
            visibility,
            depth_mode: ForwardDepthMode::Disabled,
            sorted_transparency: false,
            material_instances: None,
            mesh_side_orientations: None,
            mesh_visibilities: None,
        }
    }

    /// Supplies per-draw material instances for the prepared frame.
    ///
    /// Scene and Feature draws are looked up by their exact owner identities.
    /// Selected instances must be compiled for the draw's material generation.
    pub fn with_material_instances(
        mut self,
        instances: HashMap<crate::ForwardDrawSource, MaterialInstance>,
    ) -> Self {
        self.material_instances = Some(Arc::new(instances));
        self
    }

    /// Supplies mesh-side winding for scene records or typed Feature draws.
    ///
    /// Draws without an explicit winding use counterclockwise. A material can
    /// override the winding of the mesh using it.
    pub fn with_mesh_side_orientations(
        mut self,
        orientations: HashMap<crate::ForwardDrawSource, SurfaceSideOrientation>,
    ) -> Self {
        self.mesh_side_orientations = Some(Arc::new(orientations));
        self
    }

    /// Supplies per-draw mesh visibility for scene records or typed Feature draws.

    /// Partial visibility uses transparent composition. Enable sorted transparency
    /// for ordinary Forward submissions, or choose the OIT submission route.
    pub fn with_mesh_visibilities(
        mut self,
        visibilities: HashMap<crate::ForwardDrawSource, crate::MeshVisibility>,
    ) -> Self {
        self.mesh_visibilities = Some(Arc::new(visibilities));
        self
    }

    /// Selects the bounded opaque depth product.
    pub const fn with_depth32_less_equal(mut self) -> Self {
        self.depth_mode = ForwardDepthMode::Depth32LessEqualWrite;
        self
    }

    /// Selects the Forward product with globally merged sorted transparency.
    /// It owns the depth attachment required to depth-test blend draws.
    pub const fn with_sorted_transparency(mut self) -> Self {
        self.depth_mode = ForwardDepthMode::Depth32LessEqualWrite;
        self.sorted_transparency = true;
        self
    }
}

impl<R, V> FramePipeline<R> for ForwardCpuPipeline<V>
where
    V: VisibilityProvider<R>,
{
    fn forward_depth_mode(&self, _frame: &PreparedFrame<'_, R>) -> ForwardDepthMode {
        self.depth_mode
    }
    fn topology(&self, _frame: &PreparedFrame<'_, R>) -> Result<FrameTopology, PipelineError> {
        Ok(if self.sorted_transparency {
            FrameTopology::SortedForward
        } else {
            FrameTopology::OpaqueForward
        })
    }
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        view: RenderViewId,
        visibility: RenderBounds,
    ) -> Result<PreparedFrame<'a, R>, PipelineError> {
        // Validate before the provider runs so an external provider cannot
        // observe or act on a stale or foreign view.
        snapshot.view(view).map_err(PipelineError::Scene)?;
        let candidates: Vec<_> = snapshot.record_ids().collect();
        let mut visible = self.visibility.visible(snapshot, visibility, &candidates)?;
        let mut observed = HashSet::with_capacity(visible.len());
        for id in &visible {
            if !observed.insert(*id) {
                return Err(PipelineError::DuplicateCandidate {
                    slot: id.slot(),
                    generation: id.generation(),
                });
            }
            snapshot.record(*id).map_err(PipelineError::Scene)?;
        }
        visible.sort_unstable_by_key(|id| (id.slot(), id.generation()));
        Ok(PreparedFrame {
            snapshot,
            view,
            candidates: visible,
            material_instances: self.material_instances.clone(),
            mesh_side_orientations: self.mesh_side_orientations.clone(),
            mesh_visibilities: self.mesh_visibilities.clone(),
        })
    }
}

/// The reference provider that intersects resolved world-space AABBs.
///
/// The supplied volume is already in the same resolved space as record bounds;
/// this provider does not define camera projection semantics.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundsVisibility;

impl<R> VisibilityProvider<R> for BoundsVisibility {
    fn visible(
        &self,
        snapshot: &FrameSnapshot<R>,
        volume: RenderBounds,
        candidates: &[RenderId],
    ) -> Result<Vec<RenderId>, PipelineError> {
        let mut visible = Vec::new();
        for id in candidates {
            let record = snapshot.record(*id).map_err(PipelineError::Scene)?;
            if intersects(volume, record.bounds) {
                visible.push(*id);
            }
        }
        Ok(visible)
    }
}

/// A prepared CPU-only view of one immutable scene snapshot.
///
/// It borrows the snapshot, so selected records cannot outlive the version that
/// was validated. It contains no mutable scheduler, Graph, or backend handle.
pub struct PreparedFrame<'a, R> {
    snapshot: &'a FrameSnapshot<R>,
    view: RenderViewId,
    candidates: Vec<RenderId>,
    material_instances: Option<Arc<HashMap<crate::ForwardDrawSource, MaterialInstance>>>,
    mesh_side_orientations: Option<Arc<HashMap<crate::ForwardDrawSource, SurfaceSideOrientation>>>,
    mesh_visibilities: Option<Arc<HashMap<crate::ForwardDrawSource, crate::MeshVisibility>>>,
}

impl<'a, R> PreparedFrame<'a, R> {
    /// Supplies the material instances selected by a custom pipeline for this frame.
    pub fn with_material_instances(
        mut self,
        instances: HashMap<crate::ForwardDrawSource, MaterialInstance>,
    ) -> Self {
        self.material_instances = Some(Arc::new(instances));
        self
    }

    /// Supplies mesh-side winding for scene records or typed Feature draws.
    ///
    /// Draws without an explicit winding use counterclockwise. A material can
    /// override the winding of the mesh using it.
    pub fn with_mesh_side_orientations(
        mut self,
        orientations: HashMap<crate::ForwardDrawSource, SurfaceSideOrientation>,
    ) -> Self {
        self.mesh_side_orientations = Some(Arc::new(orientations));
        self
    }

    /// Supplies per-draw mesh visibility for scene records or typed Feature draws.
    pub fn with_mesh_visibilities(
        mut self,
        visibilities: HashMap<crate::ForwardDrawSource, crate::MeshVisibility>,
    ) -> Self {
        self.mesh_visibilities = Some(Arc::new(visibilities));
        self
    }

    pub(crate) fn mesh_side_orientation(
        &self,
        source: crate::ForwardDrawSource,
    ) -> SurfaceSideOrientation {
        self.mesh_side_orientations
            .as_ref()
            .and_then(|orientations| orientations.get(&source))
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn mesh_visibility(
        &self,
        source: crate::ForwardDrawSource,
    ) -> crate::MeshVisibility {
        self.mesh_visibilities
            .as_ref()
            .and_then(|visibilities| visibilities.get(&source))
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn material_instance(
        &self,
        source: crate::ForwardDrawSource,
    ) -> Option<&MaterialInstance> {
        self.material_instances.as_ref()?.get(&source)
    }

    pub(crate) fn validate_for(
        &self,
        snapshot: &FrameSnapshot<R>,
        view: RenderViewId,
    ) -> Result<(), PipelineError> {
        if self.snapshot.provenance() != snapshot.provenance()
            || self.scene_version() != snapshot.version()
            || self.view != view
        {
            return Err(PipelineError::PreparedFrameMismatch);
        }
        Ok(())
    }
    /// Returns the immutable scene version that supplied this frame.
    pub fn scene_version(&self) -> u64 {
        self.snapshot.version()
    }

    /// Returns the validated view identity.
    pub fn view(&self) -> RenderViewId {
        self.view
    }

    /// Returns the validated immutable view used for this preparation.
    pub fn view_record(&self) -> &crate::RenderView {
        // Construction validates this identity against the same borrowed root.
        self.snapshot
            .view(self.view)
            .expect("prepared view remains in its immutable snapshot")
    }

    /// Returns selected records in deterministic slot and generation order.
    pub fn candidates(&self) -> &[RenderId] {
        &self.candidates
    }

    /// Looks up a selected record in the retained immutable snapshot.
    pub fn record(&self, id: RenderId) -> Result<&RenderRecord<R>, PipelineError> {
        if !self.candidates.contains(&id) {
            return Err(PipelineError::CandidateNotPrepared {
                slot: id.slot(),
                generation: id.generation(),
            });
        }
        self.snapshot.record(id).map_err(PipelineError::Scene)
    }
}

/// A structured refusal while preparing a CPU frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineError {
    /// The bounded typed-feature product requires an opaque or sorted Forward phase.
    FeatureRequiresOpaque,
    /// The final ordering was not a permutation of this frame's exact draws.
    InvalidDrawOrder,
    /// A pipeline returned preparation for a different scene, version, or view.
    PreparedFrameMismatch,
    /// The contribution did not match the requested topology or phase order.
    InvalidPhaseContribution,
    /// Sorted transparency requires an eye position for the selected view.
    MissingTransparencyCamera {
        /// Transparent draw whose distance cannot be derived.
        source: crate::ForwardDrawSource,
    },
    /// A transparent draw has no finite world-space centre for sorting.
    InvalidTransparencyGeometry {
        /// Transparent draw with invalid geometry facts.
        source: crate::ForwardDrawSource,
    },
    /// The scene snapshot rejected an input identity.
    Scene(SceneError),
    /// A provider returned an ID more than once.
    DuplicateCandidate {
        /// Record slot.
        slot: u32,
        /// Record identity generation.
        generation: u32,
    },
    /// A caller requested a record outside the prepared candidate set.
    CandidateNotPrepared {
        /// Record slot.
        slot: u32,
        /// Record identity generation.
        generation: u32,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureRequiresOpaque => formatter
                .write_str("typed Forward features require an opaque or sorted Forward phase"),
            Self::InvalidDrawOrder => {
                formatter.write_str("pipeline ordering changed or duplicated a prepared draw")
            }
            Self::PreparedFrameMismatch => formatter
                .write_str("pipeline returned a different scene, snapshot version, or view"),
            Self::InvalidPhaseContribution => {
                formatter.write_str("pipeline made an invalid phase contribution")
            }
            Self::MissingTransparencyCamera { source } => {
                write!(
                    formatter,
                    "transparent draw {source:?} needs a camera position"
                )
            }
            Self::InvalidTransparencyGeometry { source } => {
                write!(
                    formatter,
                    "transparent draw {source:?} has invalid geometry for sorting"
                )
            }
            Self::Scene(error) => error.fmt(formatter),
            Self::DuplicateCandidate { slot, generation } => {
                write!(
                    formatter,
                    "visibility provider returned record {slot}:{generation} twice"
                )
            }
            Self::CandidateNotPrepared { slot, generation } => {
                write!(
                    formatter,
                    "record {slot}:{generation} is not in this prepared frame"
                )
            }
        }
    }
}

impl std::error::Error for PipelineError {}

fn intersects(left: RenderBounds, right: RenderBounds) -> bool {
    left.minimum()
        .iter()
        .zip(left.maximum())
        .zip(right.minimum())
        .zip(right.maximum())
        .all(
            |(((left_minimum, left_maximum), right_minimum), right_maximum)| {
                left_minimum <= &right_maximum && right_minimum <= left_maximum
            },
        )
}

#[cfg(test)]
#[path = "../tests/pipeline/contract.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/pipeline/test_helpers.rs"]
mod test_helpers;
