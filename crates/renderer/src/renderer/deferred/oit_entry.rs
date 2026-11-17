//! Explicit finite OIT for the built-in Deferred scene consumer.

use super::*;
use std::time::Instant;

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
    /// Lights opaque/masked geometry through the G-buffer, then resolves
    /// selected transparent geometry with finite dual-depth peeling.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_oit<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_oit_target(pipeline, snapshot, view, visibility, assets, config, None)
            .await
    }

    /// Presents the same Deferred OIT product without a CPU readback.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_oit_surface<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_oit_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_deferred_oit_target<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        if let Some(image) = surface.as_ref() {
            if image.device() != self.device.id() {
                return Err(Error::WrongDevice.into());
            }
            if !image.is_current() {
                return Err(Error::SurfaceOutdated.into());
            }
        }
        let started = Instant::now();
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::DeferredPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::DeferredPipeline)?;
        let extent = selected.view_record().extent;
        if surface
            .as_ref()
            .is_some_and(|image| image.extent() != extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        if inputs.is_empty() {
            return Err(Error::Unsupported("Deferred requires visible geometry").into());
        }
        let partition =
            super::super::forward::order_forward_oit_draws(pipeline, &selected, inputs, config)
                .map_err(RendererError::DeferredPipeline)?;
        if partition.opaque_count == 0 {
            return self
                .submit_prepared_oit(
                    snapshot.clone(),
                    partition,
                    extent,
                    assets,
                    config,
                    surface,
                    started,
                    |context| {
                        context.sorted(|pass| {
                            pass.depends_on_uploads()?;
                            pass.read_vertex_streams()?;
                            pass.read_frame_uniforms()?;
                            pass.read_sampled_textures()?;
                            pass.write_selected_depth()?;
                            pass.read_write_main_color()
                        })
                    },
                    |context| context.all_reads(),
                )
                .await;
        }
        if partition.peel_count == 0 {
            if partition.draws.len() != partition.opaque_count {
                return self
                    .submit_prepared_deferred_transparent(
                        snapshot.clone(),
                        partition.draws,
                        extent,
                        assets,
                        surface,
                        |context| pipeline.contribute_deferred(&selected, context),
                    )
                    .await;
            }
            self.deferred_residency.retain_assets(assets);
            let retained = PendingInputs::Frame {
                _snapshot: snapshot.clone(),
                _inputs: partition.draws,
            };
            let PendingInputs::Frame {
                _inputs: inputs, ..
            } = &retained
            else {
                unreachable!()
            };
            return match surface {
                Some(image) => {
                    let frame = self
                        .prepare_deferred_surface_frames(inputs, image.format())
                        .await?;
                    self.execute_deferred_surface(retained, frame, image, |context| {
                        pipeline.contribute_deferred(&selected, context)
                    })
                    .await
                }
                None => {
                    let frame = self.prepare_deferred_frames(inputs).await?;
                    self.execute_deferred_probe(retained, frame, |context| {
                        pipeline.contribute_deferred(&selected, context)
                    })
                    .await
                }
            };
        }
        // Known OIT capacity facts are refused before the dependent frame
        // allocations; a selected-route failure never becomes ordinary alpha.
        config
            .rounds()
            .get()
            .checked_mul(3)
            .and_then(|passes| passes.checked_add(5))
            .ok_or(Error::InvalidRange)?;
        crate::residency::OitPlan::admit(&self.device, extent)?;
        self.prepare_deferred_oit(
            snapshot.clone(),
            partition,
            extent,
            assets,
            config,
            surface,
            started,
            |context| pipeline.contribute_deferred(&selected, context),
        )
        .await
    }
}
