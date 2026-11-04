//! Frame policy and preparation for the first Forward consumer.

use std::{collections::HashSet, fmt};

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

/// The fixed set of GPU products that the first Forward consumer can build.
///
/// This is deliberately smaller than a Graph API: a pipeline may select an
/// existing product, while the renderer retains the declarations, resources,
/// recording, submission and completion ownership for that product.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FrameTopology {
    /// Upload the prepared opaque draws, rasterize them, then read the target.
    OpaqueForward,
    /// Clear and read the target without resolving scene assets or recording draws.
    ClearOnly,
}

/// The minimal public frame-policy seam exercised by the first Forward consumer.
///
/// Implementations are selected statically.  They prepare candidates and choose
/// one concrete GPU product.  They cannot obtain Graph state, command encoders,
/// allocators, or submission authority.
pub trait FramePipeline<R> {
    /// Prepares the records visible to an existing scene view.
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        view: RenderViewId,
        visibility: RenderBounds,
    ) -> Result<PreparedFrame<'a, R>, PipelineError>;

    /// Selects the GPU product contributed for this already-validated frame.
    ///
    /// The renderer calls this before resolving assets or issuing backend work.
    /// The returned value participates in the concrete Forward plan shape.
    fn topology(&self, _frame: &PreparedFrame<'_, R>) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::OpaqueForward)
    }

    /// Declares the selected phase directly on the renderer-owned private Graph.
    /// The default contributes the complete opaque Forward access set; a
    /// clear-only pipeline must override it with [`ForwardPhaseContext::clear`].
    fn contribute<B: ResourceBackend>(
        &self,
        _frame: &PreparedFrame<'_, R>,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
    ) -> Result<(), PipelineError> {
        context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        })
    }
}

/// A controlled CPU Forward pipeline with a concrete visibility provider.
#[derive(Clone, Debug, Default)]
pub struct ForwardCpuPipeline<V = BoundsVisibility> {
    visibility: V,
}

impl<V> ForwardCpuPipeline<V> {
    /// Creates a pipeline from one concrete, statically dispatched provider.
    pub const fn new(visibility: V) -> Self {
        Self { visibility }
    }
}

impl<R, V> FramePipeline<R> for ForwardCpuPipeline<V>
where
    V: VisibilityProvider<R>,
{
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
}

impl<'a, R> PreparedFrame<'a, R> {
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
    /// A pipeline returned preparation for a different scene, version, or view.
    PreparedFrameMismatch,
    /// The contribution did not match the requested topology or phase order.
    InvalidPhaseContribution,
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
            Self::PreparedFrameMismatch => formatter
                .write_str("pipeline returned a different scene, snapshot version, or view"),
            Self::InvalidPhaseContribution => {
                formatter.write_str("pipeline made an invalid Forward phase contribution")
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
