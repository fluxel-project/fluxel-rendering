//! Particle specialization for the ordinary sorted Forward material route.

use std::{
    collections::HashMap,
    sync::{Arc, atomic::AtomicBool},
};

use super::*;
use crate::{GpuParticleDraw, renderer::particle_update::ParticleStep};

pub(super) struct PreparedParticles<B: ResourceBackend> {
    pub(super) steps: Vec<(usize, ParticleStep<B>)>,
    pub(super) uses: Vec<(usize, Arc<AtomicBool>)>,
}

impl<B: ResourceBackend> Default for PreparedParticles<B> {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
            uses: Vec::new(),
        }
    }
}

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    /// Advances selected particle systems and draws their updated billboards in
    /// the same sorted Forward Graph as the other selected materials.
    #[allow(clippy::too_many_arguments)] // Scene and mutable particle buffers have different ownership.
    pub async fn submit_forward_with_particles<P: crate::FramePipeline<crate::ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        particles: &mut [GpuParticleDraw<'_, B>],
    ) -> std::result::Result<super::Submission, RendererError> {
        self.submit_forward_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            &[],
            Some(particles),
            None,
        )
        .await
    }

    /// Advances selected particle systems and presents the sorted Forward result.
    #[allow(clippy::too_many_arguments)] // The caller retains the particles and owns the surface lease.
    pub async fn submit_forward_surface_with_particles<
        P: crate::FramePipeline<crate::ForwardResources>,
    >(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        particles: &mut [GpuParticleDraw<'_, B>],
        image: SurfaceImage<B>,
    ) -> std::result::Result<super::Submission, RendererError> {
        self.submit_forward_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            &[],
            Some(particles),
            Some(image),
        )
        .await
    }

    pub(super) fn prepare_particles(
        &mut self,
        snapshot: &crate::FrameSnapshot<crate::ForwardResources>,
        topology: crate::FrameTopology,
        inputs: &mut [crate::ForwardDrawInput],
        requests: &[GpuParticleDraw<'_, B>],
    ) -> std::result::Result<PreparedParticles<B>, RendererError> {
        if requests.is_empty() {
            return Ok(PreparedParticles::default());
        }
        let mut by_draw = HashMap::with_capacity(requests.len());
        for (index, request) in requests.iter().enumerate() {
            snapshot
                .record(request.record)
                .map_err(RendererError::ParticleScene)?;
            if by_draw.insert(request.record, index).is_some() {
                return Err(RendererError::DuplicateParticleDraw(request.record));
            }
        }
        let mut composer = fluxel_shader::ShaderComposer::new();
        let mut selected = Vec::new();
        for (draw, input) in inputs.iter_mut().enumerate() {
            let Some(&request_index) = input.record().and_then(|record| by_draw.get(&record))
            else {
                continue;
            };
            if topology != crate::FrameTopology::SortedForward {
                return Err(Error::Unsupported(
                    "GPU particles require sorted Forward transparency",
                )
                .into());
            }
            let request = &requests[request_index];
            self.validate_particle_step(request.system, request.update)?;
            crate::preparation::prepare_particle_draw(
                input,
                request.camera,
                request.system.count(),
                &mut composer,
            )
            .map_err(RendererError::ForwardPreparation)?;
            selected.push((draw, request_index));
        }
        let mut products = PreparedParticles::default();
        for (draw, request_index) in selected {
            let request = &requests[request_index];
            let step = self
                .prepare_particle_step(request.system, request.update)
                .map_err(|error| self.observe_renderer_error(error))?;
            products.uses.push((request_index, step.use_marker()));
            products.steps.push((draw, step));
        }
        Ok(products)
    }
}
