//! One real Forward pass with mesh and two independently typed feature scopes.

use super::{ForwardFeaturePass, ForwardPhaseContext, PipelineError};
use crate::RenderFeature;
use fluxel_rhi::backend::ResourceBackend;

/// Controlled declaration context for meshes and two concrete typed features.
///
/// Every scope is disjoint by prepared-draw index and must be consumed once in
/// the selected real Graph pass. The feature preparation types remain separate.
pub struct ForwardPairCompositionContext<
    'context,
    'graph,
    'frame,
    B: ResourceBackend,
    F: RenderFeature,
    G: RenderFeature,
> {
    phase: ForwardPhaseContext<'context, 'graph, 'frame, B>,
    first: &'context F,
    first_prepared: &'context F::Prepared,
    second: &'context G,
    second_prepared: &'context G::Prepared,
    meshes: &'context [usize],
    first_draws: &'context [usize],
    second_draws: &'context [usize],
}

impl<'context, 'graph, 'frame, B: ResourceBackend, F: RenderFeature, G: RenderFeature>
    ForwardPairCompositionContext<'context, 'graph, 'frame, B, F, G>
{
    pub(crate) fn new(
        phase: &'context mut ForwardPhaseContext<'_, 'graph, 'frame, B>,
        first: &'context F,
        first_prepared: &'context F::Prepared,
        second: &'context G,
        second_prepared: &'context G::Prepared,
        meshes: &'context [usize],
        first_draws: &'context [usize],
        second_draws: &'context [usize],
    ) -> Self {
        Self {
            phase: ForwardPhaseContext::new(&mut *phase.driver),
            first,
            first_prepared,
            second,
            second_prepared,
            meshes,
            first_draws,
            second_draws,
        }
    }

    pub(crate) const fn topology(&self) -> crate::FrameTopology {
        self.phase.topology()
    }

    /// Declares the selected opaque pass and all three required read scopes.
    pub fn opaque<D>(&mut self, declare: D) -> Result<(), PipelineError>
    where
        D: for<'access, 'callback, 'pass, 'builder> FnOnce(
            &'access mut ForwardPairCompositionPass<'callback, 'pass, 'builder, 'frame, B, F, G>,
        ) -> Result<(), PipelineError>,
    {
        let first = self.first;
        let first_prepared = self.first_prepared;
        let second = self.second;
        let second_prepared = self.second_prepared;
        let meshes = self.meshes;
        let first_draws = self.first_draws;
        let second_draws = self.second_draws;
        self.phase.driver.declare_opaque(|pass| {
            let mut composed = ForwardPairCompositionPass::new(
                pass,
                first,
                first_prepared,
                second,
                second_prepared,
                meshes,
                first_draws,
                second_draws,
            );
            declare(&mut composed)?;
            composed.finish()
        })
    }

    /// Declares the selected sorted Forward pass and all three required scopes.
    pub fn sorted<D>(&mut self, declare: D) -> Result<(), PipelineError>
    where
        D: for<'access, 'callback, 'pass, 'builder> FnOnce(
            &'access mut ForwardPairCompositionPass<'callback, 'pass, 'builder, 'frame, B, F, G>,
        ) -> Result<(), PipelineError>,
    {
        let first = self.first;
        let first_prepared = self.first_prepared;
        let second = self.second;
        let second_prepared = self.second_prepared;
        let meshes = self.meshes;
        let first_draws = self.first_draws;
        let second_draws = self.second_draws;
        self.phase.driver.declare_sorted(|pass| {
            let mut composed = ForwardPairCompositionPass::new(
                pass,
                first,
                first_prepared,
                second,
                second_prepared,
                meshes,
                first_draws,
                second_draws,
            );
            declare(&mut composed)?;
            composed.finish()
        })
    }
}

/// Pipeline authority over one Forward pass with three disjoint draw scopes.
pub struct ForwardPairCompositionPass<
    'callback,
    'pass,
    'builder,
    'frame,
    B: ResourceBackend,
    F: RenderFeature,
    G: RenderFeature,
> {
    pass: &'callback mut crate::graph::OpaqueForwardPass<'pass, 'builder, 'frame, B>,
    first: &'callback F,
    first_prepared: &'callback F::Prepared,
    second: &'callback G,
    second_prepared: &'callback G::Prepared,
    meshes: &'callback [usize],
    first_draws: &'callback [usize],
    second_draws: &'callback [usize],
    meshes_declared: bool,
    first_declared: bool,
    second_declared: bool,
}

impl<'callback, 'pass, 'builder, 'frame, B: ResourceBackend, F: RenderFeature, G: RenderFeature>
    ForwardPairCompositionPass<'callback, 'pass, 'builder, 'frame, B, F, G>
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        pass: &'callback mut crate::graph::OpaqueForwardPass<'pass, 'builder, 'frame, B>,
        first: &'callback F,
        first_prepared: &'callback F::Prepared,
        second: &'callback G,
        second_prepared: &'callback G::Prepared,
        meshes: &'callback [usize],
        first_draws: &'callback [usize],
        second_draws: &'callback [usize],
    ) -> Self {
        Self {
            pass,
            first,
            first_prepared,
            second,
            second_prepared,
            meshes,
            first_draws,
            second_draws,
            meshes_declared: false,
            first_declared: false,
            second_declared: false,
        }
    }

    fn finish(&self) -> Result<(), PipelineError> {
        if self.meshes_declared && self.first_declared && self.second_declared {
            Ok(())
        } else {
            Err(PipelineError::InvalidPhaseContribution)
        }
    }

    /// Declares the shared upload producer dependency.
    pub fn depends_on_uploads(&mut self) -> Result<(), PipelineError> {
        self.pass
            .depends_on_uploads()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the built-in mesh scope exactly once.
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
        let mut pass = ForwardFeaturePass::new(scope);
        declare(&mut pass)?;
        pass.finish()
    }

    /// Invokes the first feature in its own scope exactly once.
    pub fn first_feature(&mut self) -> Result<(), PipelineError> {
        if self.first_declared {
            return Err(PipelineError::InvalidPhaseContribution);
        }
        self.first_declared = true;
        let scope = self
            .pass
            .read_scope(self.first_draws)
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        let mut pass = ForwardFeaturePass::new(scope);
        self.first.contribute(self.first_prepared, &mut pass)?;
        pass.finish()
    }

    /// Invokes the second feature in its own scope exactly once.
    pub fn second_feature(&mut self) -> Result<(), PipelineError> {
        if self.second_declared {
            return Err(PipelineError::InvalidPhaseContribution);
        }
        self.second_declared = true;
        let scope = self
            .pass
            .read_scope(self.second_draws)
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        let mut pass = ForwardFeaturePass::new(scope);
        self.second.contribute(self.second_prepared, &mut pass)?;
        pass.finish()
    }

    /// Declares the depth attachment selected for this Forward frame.
    pub fn write_selected_depth(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_selected_depth()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the opaque output color write.
    pub fn write_main_color(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_main_color()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the sorted pass's read/write color contract.
    pub fn read_write_main_color(&mut self) -> Result<(), PipelineError> {
        self.write_main_color()
    }
}
