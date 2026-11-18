//! Same-Graph capture with independent finite OIT selection on each side.

use super::assembly::{AssembledForwardTarget, ForwardAllocationStats};
use super::capture_batch::{CaptureReceivers, order_capture_producer_draws};
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

    pub(in crate::renderer) fn capture_owner(
        &self,
        device: &Device<B>,
        slot: CaptureTextureSlot,
        scene_refraction: bool,
    ) -> Result<CaptureTextureOwner<B>> {
        let mut capture = match self.oit.as_ref() {
            Some(oit) => CaptureTextureOwner::from_oit_output(device, slot, &oit.output),
            None => CaptureTextureOwner::from_capture_frame(device, slot, &self.frame),
        }?;
        if scene_refraction {
            capture.refraction_depth = Some(
                self.frame
                    .frame
                    .depth
                    .as_ref()
                    .ok_or(Error::InvalidState)?
                    .retain(),
            );
        }
        Ok(capture)
    }
}

pub(in crate::renderer) fn capture_oit_receiver_partition(
    input: crate::ForwardDrawInput,
    config: Option<&crate::DualDepthPeelingConfig>,
) -> OitDrawPartition {
    let scene_refraction = input.prepared().uses_scene_refraction();
    let mut partition = match config {
        Some(config) => crate::preparation::partition_oit_draws(vec![input], config),
        None => OitDrawPartition {
            draws: vec![input],
            opaque_count: 0,
            peel_count: 0,
        },
    };
    if scene_refraction {
        // A projected receiver inherits the producer's background and depth.
        // Even an unblended receiver must not write into that opaque prefix.
        partition.opaque_count = 0;
    }
    partition
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
            CaptureReceivers::Single(receiver),
            Some(producer_config),
            Some(receiver_config),
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
            CaptureReceivers::Single(receiver),
            Some(producer_config),
            Some(receiver_config),
            Some(image),
        )
        .await
    }

    /// Submits projected Blend or MaskBlend receivers against the same captured scene.
    /// Each optional config independently selects finite peeling on that side.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture_batch<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
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
            CaptureReceivers::RefractionBatch(receivers),
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Presents the same refraction batch to an acquired Surface in one submission.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture_batch_surface<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
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
            CaptureReceivers::RefractionBatch(receivers),
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
        receivers: CaptureReceivers,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
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
        let environment = self.environment_irradiance.is_some();
        let specular = self.environment_specular.is_some();
        let (receiver, assets, scene_refraction) = receivers.prepare(
            &producer_slot,
            selected.view_record(),
            assets,
            environment,
            specular,
            &mut composer,
            receiver_config,
        )?;
        let inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            &assets,
            &mut composer,
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        let producer = order_capture_producer_draws(pipeline, &selected, inputs, producer_config)
            .map_err(RendererError::ForwardPipeline)?;
        let producer_rounds =
            producer_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
        let receiver_rounds =
            receiver_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
        if scene_refraction {
            self.admit_oit_capture_refraction(
                extent,
                surface.is_some() || receiver.peel_count != 0,
            )?;
        }
        for (partition, rounds) in [(&producer, producer_rounds), (&receiver, receiver_rounds)] {
            if partition.peel_count != 0 {
                rounds
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
                producer_rounds,
                started,
            )
            .await?;
        let capture_owner = producer
            .capture_owner(&self.device, producer_slot, scene_refraction)
            .map_err(|error| self.observe_forward_error("OIT capture product admission", error))?;
        let receiver = self
            .assemble_capture_oit_side(
                snapshot,
                receiver,
                extent,
                &assets,
                Some(&capture_owner),
                surface.is_some(),
                receiver_rounds,
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

    pub(super) fn admit_oit_capture_refraction(
        &self,
        extent: [u32; 2],
        sampled_target: bool,
    ) -> Result<()> {
        let hdr = self.output_processing().is_some();
        super::target_admission::admit(
            &self.device,
            crate::FrameTopology::SortedForward,
            extent,
            if hdr {
                TextureFormat::Rgba16Float
            } else {
                TextureFormat::Rgba8Unorm
            },
            false,
            sampled_target || hdr,
            true,
        )
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
                if self.output_processing().is_some() {
                    TextureFormat::Rgba16Float
                } else {
                    TextureFormat::Rgba8Unorm
                },
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
