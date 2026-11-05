//! Narrow public declaration authority for the private Deferred probe.

use super::{
    ForwardCpuPipeline, FramePipeline, FrameTopology, PipelineError, PreparedFrame,
    VisibilityProvider,
};
use crate::{FrameSnapshot, RenderBounds, RenderViewId};

/// Single-use context for the two private Deferred Graph transactions.
pub struct DeferredPhaseContext<'context, 'graph> {
    driver: &'context mut crate::graph::DeferredPhaseDriver<'graph>,
}

/// Public geometry-pass declaration capability without Graph handles.
pub struct DeferredGeometryPass<'callback, 'pass, 'builder> {
    pass: &'callback mut crate::graph::DeferredGeometryPass<'pass, 'builder>,
}

/// Public lighting-pass declaration capability without Graph handles.
pub struct DeferredLightingPass<'callback, 'pass, 'builder> {
    pass: &'callback mut crate::graph::DeferredLightingPass<'pass, 'builder>,
}

/// Public compute-pass declaration capability without Graph handles.
pub struct DeferredDeformationPass<'callback, 'pass, 'builder> {
    pass: &'callback mut crate::graph::DeferredDeformationPass<'pass, 'builder>,
}

impl DeferredGeometryPass<'_, '_, '_> {
    /// Makes the geometry transaction depend on the already-declared upload pass.
    pub fn depends_on_uploads(&mut self) -> Result<(), PipelineError> {
        self.pass
            .depends_on_uploads()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the physical vertex-stream reads for the fixed Deferred draw.
    pub fn read_vertex_streams(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_vertex_streams()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the per-draw tint and transform uniform reads.
    pub fn read_frame_uniforms(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_frame_uniforms()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the fixed material texture read.
    pub fn read_sampled_textures(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_sampled_textures()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the three G-buffer attachment writes.
    pub fn write_gbuffer(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_gbuffer()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}

impl DeferredLightingPass<'_, '_, '_> {
    /// Declares sampling reads of all three G-buffer attachments.
    pub fn read_gbuffer(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_gbuffer()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the final offscreen color write.
    pub fn write_main_color(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_main_color()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}

impl DeferredDeformationPass<'_, '_, '_> {
    /// Declares the storage read of the uploaded source vertices.
    pub fn read_source(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_source()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the uniform read of the deformation parameters.
    pub fn read_parameters(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_parameters()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Declares the storage write of the generated vertex product.
    pub fn write_vertices(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_vertices()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}

impl<'context, 'graph> DeferredPhaseContext<'context, 'graph> {
    pub(crate) fn new(driver: &'context mut crate::graph::DeferredPhaseDriver<'graph>) -> Self {
        Self { driver }
    }

    /// Opens the optional deformation transaction before geometry.
    pub fn deformation<F>(&mut self, declare: F) -> Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut DeferredDeformationPass<'callback, 'pass, 'builder>,
        ) -> Result<(), PipelineError>,
    {
        self.driver
            .deformation(|pass| {
                let mut pass = DeferredDeformationPass { pass };
                declare(&mut pass).map_err(|_| ())
            })
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Opens the one geometry declaration transaction.
    pub fn geometry<F>(&mut self, declare: F) -> Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut DeferredGeometryPass<'callback, 'pass, 'builder>,
        ) -> Result<(), PipelineError>,
    {
        self.driver
            .geometry(|pass| {
                let mut pass = DeferredGeometryPass { pass };
                declare(&mut pass).map_err(|_| ())
            })
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    /// Opens the one lighting declaration transaction after geometry completed.
    pub fn lighting<F>(&mut self, declare: F) -> Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut DeferredLightingPass<'callback, 'pass, 'builder>,
        ) -> Result<(), PipelineError>,
    {
        self.driver
            .lighting(|pass| {
                let mut pass = DeferredLightingPass { pass };
                declare(&mut pass).map_err(|_| ())
            })
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}

/// CPU visibility policy selecting the bounded Deferred product.
#[derive(Clone, Debug, Default)]
pub struct DeferredCpuPipeline<V = super::BoundsVisibility> {
    forward: ForwardCpuPipeline<V>,
}

impl<V> DeferredCpuPipeline<V> {
    /// Creates a Deferred policy using one static visibility provider.
    pub const fn new(visibility: V) -> Self {
        Self {
            forward: ForwardCpuPipeline::new(visibility),
        }
    }
}

impl<R, V> FramePipeline<R> for DeferredCpuPipeline<V>
where
    V: VisibilityProvider<R>,
{
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        view: RenderViewId,
        visibility: RenderBounds,
    ) -> Result<PreparedFrame<'a, R>, PipelineError> {
        self.forward.prepare(snapshot, view, visibility)
    }
    fn topology(&self, _frame: &PreparedFrame<'_, R>) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::Deferred)
    }
    fn contribute_deferred(
        &self,
        _frame: &PreparedFrame<'_, R>,
        context: &mut DeferredPhaseContext<'_, '_>,
    ) -> Result<(), PipelineError> {
        context.geometry(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_gbuffer()
        })?;
        context.lighting(|pass| {
            pass.read_gbuffer()?;
            pass.write_main_color()
        })
    }
}
