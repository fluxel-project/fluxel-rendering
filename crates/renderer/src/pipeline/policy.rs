//! Statically selected frame-policy contracts and their default Graph routes.

use super::{
    DeferredPhaseContext, ForwardCompositionContext, ForwardDepthMode,
    ForwardPairCompositionContext, ForwardPhaseContext, FrameTopology, PipelineError,
    PreparedFrame,
};
use crate::{FrameSnapshot, RenderBounds, RenderViewId};
use fluxel_rhi::backend::ResourceBackend;

/// The minimal public frame-policy seam exercised by Forward consumers.
///
/// Implementations are statically selected. They prepare candidates and choose
/// one concrete GPU product without Graph, command, allocation, or submission
/// authority.
pub trait FramePipeline<R> {
    /// Selects depth once for the already validated frame.
    fn forward_depth_mode(&self, _frame: &PreparedFrame<'_, R>) -> ForwardDepthMode {
        ForwardDepthMode::Disabled
    }

    /// Declares a typed GPU producer before the Deferred geometry consumer.
    fn contribute_deformed<F: crate::DeferredPositionDeformer>(
        &self,
        frame: &PreparedFrame<'_, R>,
        feature: &F,
        context: &mut DeferredPhaseContext<'_, '_>,
    ) -> Result<(), PipelineError> {
        context.deformation(|pass| feature.contribute(pass))?;
        self.contribute_deferred(frame, context)
    }

    /// Declares bounded Deferred passes on the real Graph.
    fn contribute_deferred(
        &self,
        _frame: &PreparedFrame<'_, R>,
        _context: &mut DeferredPhaseContext<'_, '_>,
    ) -> Result<(), PipelineError> {
        Err(PipelineError::InvalidPhaseContribution)
    }

    /// Defines order within the opaque and mask queues across built-in and
    /// feature-owned draws.
    ///
    /// The result must remain a permutation of the exact supplied metadata.
    /// In Sorted Forward this hook sees only the opaque/mask segment; the
    /// renderer runs the opaque queue before the mask queue and orders the
    /// complete transparent segment after both.
    fn order_forward(
        &self,
        _frame: &PreparedFrame<'_, R>,
        draws: &mut [crate::ForwardDrawOrder],
    ) -> Result<(), PipelineError> {
        draws.sort_by_key(|draw| draw.sort_key());
        Ok(())
    }

    /// Returns the primary transparent ordering key for one draw.
    ///
    /// Called once per transparent draw in Sorted Forward. Scene draws
    /// default to index zero; feature draws default to their explicit order.
    /// Override this to choose a scene draw's alpha index from its source and
    /// retained frame. Renderer sorts ascending index and descending camera
    /// distance across all contributions. Equal keys retain received draw
    /// order.
    fn transparent_alpha_index(
        &self,
        _frame: &PreparedFrame<'_, R>,
        draw: crate::ForwardDrawOrder,
    ) -> i64 {
        draw.order()
    }

    /// Declares one shared pass for meshes and one typed feature.
    fn contribute_composed<B: ResourceBackend, F: crate::RenderFeature>(
        &self,
        _frame: &PreparedFrame<'_, R>,
        context: &mut ForwardCompositionContext<'_, '_, '_, B, F>,
    ) -> Result<(), PipelineError> {
        match context.topology() {
            FrameTopology::OpaqueForward => context.opaque(|pass| {
                pass.depends_on_uploads()?;
                pass.meshes(read_all)?;
                pass.feature()?;
                pass.write_selected_depth()?;
                pass.write_main_color()
            }),
            FrameTopology::SortedForward => context.sorted(|pass| {
                pass.depends_on_uploads()?;
                pass.meshes(read_all)?;
                pass.feature()?;
                pass.write_selected_depth()?;
                pass.read_write_main_color()
            }),
            _ => Err(PipelineError::InvalidPhaseContribution),
        }
    }

    /// Declares one shared pass for meshes and two independently typed features.
    ///
    /// All three read scopes and the selected attachment accesses must be
    /// consumed in this real Graph callback. Each feature receives only its
    /// own scope; a missing or repeated contribution refuses recording.
    fn contribute_composed_pair<
        B: ResourceBackend,
        F: crate::RenderFeature,
        G: crate::RenderFeature,
    >(
        &self,
        _frame: &PreparedFrame<'_, R>,
        context: &mut ForwardPairCompositionContext<'_, '_, '_, B, F, G>,
    ) -> Result<(), PipelineError> {
        match context.topology() {
            FrameTopology::OpaqueForward => context.opaque(|pass| {
                pass.depends_on_uploads()?;
                pass.meshes(read_all)?;
                pass.first_feature()?;
                pass.second_feature()?;
                pass.write_selected_depth()?;
                pass.write_main_color()
            }),
            FrameTopology::SortedForward => context.sorted(|pass| {
                pass.depends_on_uploads()?;
                pass.meshes(read_all)?;
                pass.first_feature()?;
                pass.second_feature()?;
                pass.write_selected_depth()?;
                pass.read_write_main_color()
            }),
            _ => Err(PipelineError::InvalidPhaseContribution),
        }
    }

    /// Prepares records visible to an existing scene view.
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        view: RenderViewId,
        visibility: RenderBounds,
    ) -> Result<PreparedFrame<'a, R>, PipelineError>;

    /// Selects the GPU product contributed for the validated frame.
    fn topology(&self, _frame: &PreparedFrame<'_, R>) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::OpaqueForward)
    }

    /// Declares the selected phase directly on the renderer-owned Graph.
    fn contribute<B: ResourceBackend>(
        &self,
        _frame: &PreparedFrame<'_, R>,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
    ) -> Result<(), PipelineError> {
        match context.topology() {
            FrameTopology::OpaqueForward => context.opaque(|pass| {
                pass.depends_on_uploads()?;
                pass.read_vertex_streams()?;
                pass.read_frame_uniforms()?;
                pass.read_sampled_textures()?;
                pass.write_selected_depth()?;
                pass.write_main_color()
            }),
            FrameTopology::SortedForward => context.sorted(|pass| {
                pass.depends_on_uploads()?;
                pass.read_vertex_streams()?;
                pass.read_frame_uniforms()?;
                pass.read_sampled_textures()?;
                pass.write_selected_depth()?;
                pass.read_write_main_color()
            }),
            _ => Err(PipelineError::InvalidPhaseContribution),
        }
    }
}

pub(crate) fn read_all(pass: &mut super::ForwardFeaturePass<'_, '_>) -> Result<(), PipelineError> {
    pass.read_vertex_streams()?;
    pass.read_frame_uniforms()?;
    pass.read_sampled_textures()
}
