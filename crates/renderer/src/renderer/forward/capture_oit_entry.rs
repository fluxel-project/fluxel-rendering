//! Same-Graph capture with independent finite OIT selection on each side.

use super::assembly::{AssembledForwardTarget, ForwardAllocationStats};
use super::*;
use crate::{FramePipeline, preparation::OitDrawPartition};
use std::{num::NonZeroU32, time::Instant};

pub(crate) struct CaptureOitSide<B: ResourceBackend> {
    pub(in crate::renderer) frame: crate::residency::ForwardFrame<B>,
    pub(in crate::renderer) oit: Option<OitFrameResources<B>>,
    pub(in crate::renderer) retained: PendingInputs,
    pub(in crate::renderer) publication: ForwardPublication,
    pub(in crate::renderer) opaque_count: usize,
    pub(in crate::renderer) peel_count: usize,
    pub(in crate::renderer) rounds: NonZeroU32,
    pub(in crate::renderer) allocations: ForwardAllocationStats,
}

impl<B: ResourceBackend> CaptureOitSide<B> {
    pub(in crate::renderer) fn graph_input(&self) -> Option<crate::graph::ForwardOitInput<'_, B>> {
        self.oit.as_ref().map(|oit| crate::graph::ForwardOitInput {
            owners: &oit.owners,
            output: &oit.output,
            opaque_count: self.opaque_count,
            peel_count: self.peel_count,
            rounds: self.rounds,
        })
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
    /// Captures the producer's final color, including its selected OIT work,
    /// then shades the receiver and reads its final color in one submission.
    ///
    /// Each side independently selects peeling candidates and excluded ordinary
    /// blends. An empty candidate set takes the ordinary sorted path without
    /// requiring OIT capabilities. Admission failures never select that path.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: &crate::DualDepthPeelingConfig,
        receiver_config: &crate::DualDepthPeelingConfig,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
    {
        self.submit_forward_oit_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            receiver,
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Presents the receiver's final captured-material color to this acquired
    /// Surface in the same submission, without an intermediate CPU readback.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_surface<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: &crate::DualDepthPeelingConfig,
        receiver_config: &crate::DualDepthPeelingConfig,
        image: SurfaceImage<B>,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
    {
        self.submit_forward_oit_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            receiver,
            producer_config,
            receiver_config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_forward_oit_capture_target<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: &crate::DualDepthPeelingConfig,
        receiver_config: &crate::DualDepthPeelingConfig,
        surface: Option<SurfaceImage<B>>,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
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
            .map_err(RendererError::ForwardPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::ForwardPipeline)?;
        let extent = selected.view_record().extent;
        if surface
            .as_ref()
            .is_some_and(|image| image.extent() != extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let mut composer = fluxel_shader::ShaderComposer::new();
        let receiver = receiver
            .with_lighting(
                &mut composer,
                self.environment_irradiance.is_some(),
                self.environment_specular.is_some(),
            )
            .map_err(RendererError::ForwardPreparation)?
            .into_draw_input();
        super::capture_entry::validate_receiver_sources(&receiver, &producer_slot)?;
        if receiver.extent() != extent {
            return Err(Error::InvalidRange.into());
        }
        let assets = super::capture_entry::capture_assets(assets, &receiver);
        let inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            &assets,
            &mut composer,
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        let producer =
            super::oit_order::order_oit_draws(pipeline, &selected, inputs, producer_config)
                .map_err(RendererError::ForwardPipeline)?;
        // The receiver has exactly one draw, so excluded transparency has no
        // inter-object sorting requirement or dependency on a producer eye.
        let receiver = crate::preparation::partition_oit_draws(vec![receiver], receiver_config);
        for (partition, config) in [(&producer, producer_config), (&receiver, receiver_config)] {
            if partition.peel_count != 0 {
                config
                    .rounds()
                    .get()
                    .checked_mul(3)
                    .and_then(|passes| passes.checked_add(5))
                    .ok_or(Error::InvalidRange)?;
                crate::residency::OitPlan::admit(&self.device, extent)?;
            }
        }
        let producer = self
            .assemble_capture_oit_side(
                snapshot.clone(),
                producer,
                extent,
                &assets,
                None,
                true,
                producer_config.rounds(),
                started,
            )
            .await?;
        let capture_owner = match producer.oit.as_ref() {
            Some(oit) => super::captured::CaptureTextureOwner::from_oit_output(
                &self.device,
                producer_slot,
                &oit.output,
            ),
            None => super::captured::CaptureTextureOwner::from_capture_frame(
                &self.device,
                producer_slot,
                &producer.frame,
            ),
        }
        .map_err(|error| self.observe_forward_error("OIT capture product admission", error))?;
        let receiver = self
            .assemble_capture_oit_side(
                snapshot,
                receiver,
                extent,
                &assets,
                Some(&capture_owner),
                surface.is_some(),
                receiver_config.rounds(),
                started,
            )
            .await?;
        let target = match surface {
            Some(image) => {
                let composite = match receiver.oit.as_ref() {
                    Some(oit) => {
                        self.oit_surface_composite_owner(&oit.output, extent, &image)
                            .await?
                    }
                    None => {
                        self.surface_composite_owner(&receiver.frame, &image)
                            .await?
                    }
                };
                super::oit_execute::OitExecutionTarget::Surface { image, composite }
            }
            None => super::oit_execute::OitExecutionTarget::Readback,
        };
        self.execute_capture_oit(producer, receiver, target, started)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) async fn assemble_capture_oit_side(
        &mut self,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        partition: OitDrawPartition,
        extent: [u32; 2],
        assets: &crate::ForwardAssets,
        capture: Option<&super::captured::CaptureTextureOwner<B>>,
        sampled_target: bool,
        rounds: NonZeroU32,
        started: Instant,
    ) -> std::result::Result<CaptureOitSide<B>, RendererError> {
        let OitDrawPartition {
            draws,
            opaque_count,
            peel_count,
        } = partition;
        let environment = self.environment_irradiance.clone().filter(|_| {
            draws
                .iter()
                .any(|draw| draw.prepared().uses_environment_irradiance())
        });
        let assembled = self
            .assemble_prepared_forward(
                snapshot,
                draws,
                extent,
                crate::FrameTopology::SortedForward,
                crate::ForwardDepthMode::Depth32LessEqualWrite,
                assets,
                None,
                TextureFormat::Rgba8Unorm,
                started,
                capture,
                sampled_target || peel_count != 0,
                environment,
            )
            .await?;
        let AssembledForwardTarget::Offscreen { frame } = assembled.target else {
            return Err(Error::InvalidState.into());
        };
        let PendingInputs::Frame {
            _inputs: inputs, ..
        } = &assembled.retained
        else {
            return Err(Error::InvalidState.into());
        };
        let oit = if peel_count == 0 {
            None
        } else {
            let end = opaque_count
                .checked_add(peel_count)
                .ok_or(Error::InvalidRange)?;
            Some(
                self.assemble_oit_resources(
                    &frame,
                    inputs,
                    &(opaque_count..end).collect::<Vec<_>>(),
                    &mut fluxel_shader::ShaderComposer::new(),
                )
                .await?,
            )
        };
        Ok(CaptureOitSide {
            frame,
            oit,
            retained: assembled.retained,
            publication: assembled.publication,
            opaque_count,
            peel_count,
            rounds,
            allocations: assembled.allocations,
        })
    }
}
