//! A single real opaque pass with disjoint mesh and typed-feature read scopes.

use super::{ForwardPhaseContext, PipelineError};
use crate::RenderFeature;
use fluxel_rhi::backend::ResourceBackend;

/// Read authority for exactly one prepared Forward batch in a real Graph pass.
pub struct ForwardFeaturePass<'scope, 'builder> {
    scope: crate::graph::ForwardReadScope<'scope, 'builder>,
}

impl ForwardFeaturePass<'_, '_> {
    /// Declares every physical vertex footprint in this batch.
    pub fn read_vertex_streams(&mut self) -> Result<(), PipelineError> {
        self.scope
            .read_vertex_streams()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    /// Declares every frame-uniform range in this batch.
    pub fn read_frame_uniforms(&mut self) -> Result<(), PipelineError> {
        self.scope
            .read_frame_uniforms()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    /// Declares every sampled texture in this batch.
    pub fn read_sampled_textures(&mut self) -> Result<(), PipelineError> {
        self.scope
            .read_sampled_textures()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    fn finish(self) -> Result<(), PipelineError> {
        self.scope
            .finish()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}

/// Controlled frame context for an opaque mesh and one concrete typed feature.
///
/// This is an additive Forward consumer, not a general Graph API. The selected
/// pipeline owns phase/output policy; each contributor owns only its read scope.
pub struct ForwardCompositionContext<'context, 'graph, 'frame, B: ResourceBackend, F: RenderFeature>
{
    phase: ForwardPhaseContext<'context, 'graph, 'frame, B>,
    feature: &'context F,
    prepared: &'context F::Prepared,
    meshes: &'context [usize],
    feature_draws: &'context [usize],
}

impl<'context, 'graph, 'frame, B: ResourceBackend, F: RenderFeature>
    ForwardCompositionContext<'context, 'graph, 'frame, B, F>
{
    pub(crate) fn new(
        phase: &'context mut ForwardPhaseContext<'_, 'graph, 'frame, B>,
        feature: &'context F,
        prepared: &'context F::Prepared,
        meshes: &'context [usize],
        feature_draws: &'context [usize],
    ) -> Self {
        Self {
            phase: ForwardPhaseContext::new(&mut *phase.driver),
            feature,
            prepared,
            meshes,
            feature_draws,
        }
    }

    /// Declares the single opaque pass and both of its required read scopes.
    pub fn opaque<D>(&mut self, declare: D) -> Result<(), PipelineError>
    where
        D: for<'access, 'callback, 'pass, 'builder> FnOnce(
            &'access mut ForwardCompositionPass<'callback, 'pass, 'builder, 'frame, B, F>,
        ) -> Result<(), PipelineError>,
    {
        let feature = self.feature;
        let prepared = self.prepared;
        let meshes = self.meshes;
        let feature_draws = self.feature_draws;
        self.phase.driver.declare_opaque(|pass| {
            let mut composed = ForwardCompositionPass {
                pass,
                feature,
                prepared,
                meshes,
                feature_draws,
                meshes_declared: false,
                feature_declared: false,
            };
            declare(&mut composed)?;
            if !composed.meshes_declared || !composed.feature_declared {
                return Err(PipelineError::InvalidPhaseContribution);
            }
            Ok(())
        })
    }
}

/// Pipeline authority over one opaque pass containing disjoint contributions.
pub struct ForwardCompositionPass<
    'callback,
    'pass,
    'builder,
    'frame,
    B: ResourceBackend,
    F: RenderFeature,
> {
    pass: &'callback mut crate::graph::OpaqueForwardPass<'pass, 'builder, 'frame, B>,
    feature: &'callback F,
    prepared: &'callback F::Prepared,
    meshes: &'callback [usize],
    feature_draws: &'callback [usize],
    meshes_declared: bool,
    feature_declared: bool,
}

impl<B: ResourceBackend, F: RenderFeature> ForwardCompositionPass<'_, '_, '_, '_, B, F> {
    /// Declares the common producer dependency for all contributions.
    pub fn depends_on_uploads(&mut self) -> Result<(), PipelineError> {
        self.pass
            .depends_on_uploads()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the built-in mesh batch's read accesses once.
    pub fn meshes<D>(&mut self, declare: D) -> Result<(), PipelineError>
    where
        D: FnOnce(&mut ForwardFeaturePass<'_, '_>) -> Result<(), PipelineError>,
    {
        if self.meshes_declared {
            return Err(PipelineError::InvalidPhaseContribution);
        }
        self.meshes_declared = true;
        let scope = self
            .pass
            .read_scope(self.meshes)
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        let mut pass = ForwardFeaturePass { scope };
        declare(&mut pass)?;
        pass.finish()
    }

    /// Calls the external feature on its own actual Graph read scope once.
    pub fn feature(&mut self) -> Result<(), PipelineError> {
        if self.feature_declared {
            return Err(PipelineError::InvalidPhaseContribution);
        }
        self.feature_declared = true;
        let scope = self
            .pass
            .read_scope(self.feature_draws)
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        let mut pass = ForwardFeaturePass { scope };
        self.feature.contribute(self.prepared, &mut pass)?;
        pass.finish()
    }

    /// Declares the shared output attachment write.
    pub fn write_main_color(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_main_color()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}
