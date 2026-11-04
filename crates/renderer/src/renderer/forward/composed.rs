//! Typed feature preparation and final opaque ordering before device work.

use super::*;
use crate::{
    FeatureBinding, ForwardDrawInput, ForwardDrawOrder, ForwardDrawSource, FramePipeline,
    RenderFeature,
};
use std::{collections::HashMap, time::Instant};

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
    /// Submits built-in mesh and one concrete typed feature in the same frame.
    ///
    /// Both products share validation, residency, the actual opaque Graph pass,
    /// frame allocations and completion ownership. Feature input stays borrowed
    /// during CPU preparation/declaration; resolved asset snapshots and copied
    /// frame data are retained independently for every submitted GPU use.
    #[allow(clippy::too_many_arguments)] // The feature binding adds its own typed input owner.
    pub async fn submit_forward_with_feature<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
    ) -> Result<super::Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            None,
            TextureFormat::Rgba8Unorm,
        )
        .await
    }

    /// Submits the same typed composition to a host-acquired Surface image.
    #[allow(clippy::too_many_arguments)] // Scene, typed feature input, and surface lease have independent ownership.
    pub async fn submit_forward_surface_with_feature<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        image: SurfaceImage<B>,
    ) -> Result<super::Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
    {
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        let format = image.format();
        let extent = image.extent();
        self.submit_composed_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            Some((image, extent)),
            format,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)] // Shared offscreen/surface admission preserves the same typed path.
    async fn submit_composed_target<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        surface: Option<(SurfaceImage<B>, [u32; 2])>,
        color_format: TextureFormat,
    ) -> Result<super::Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        let preparation_started = Instant::now();
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::ForwardPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::ForwardPipeline)?;
        let topology = pipeline
            .topology(&selected)
            .map_err(RendererError::ForwardPipeline)?;
        if topology != crate::FrameTopology::OpaqueForward {
            return Err(RendererError::ForwardPipeline(
                crate::PipelineError::FeatureRequiresOpaque,
            ));
        }
        let extent = selected.view_record().extent;
        if extent.contains(&0)
            || surface
                .as_ref()
                .is_some_and(|(_, expected)| *expected != extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let handle = feature.feature.handle();
        let context = feature
            .issuer
            .context(&snapshot, crate::CpuPhase::Reference, handle)
            .map_err(RendererError::ForwardFeature)?;
        crate::validate_context(&context, feature.input, handle)
            .map_err(RendererError::ForwardFeature)?;
        let prepared = feature
            .feature
            .prepare(&context, feature.input, &selected, visibility)
            .map_err(RendererError::ForwardFeature)?;
        let mut composer = fluxel_shader::ShaderComposer::new();
        let mut inputs = crate::prepare_forward(&selected, assets, &mut composer)
            .map_err(RendererError::ForwardPreparation)?;
        inputs.extend(
            crate::preparation::prepare_feature_draws(
                &selected,
                handle,
                feature.feature.draws(&prepared),
                assets,
                &mut composer,
            )
            .map_err(RendererError::ForwardPreparation)?,
        );
        let inputs =
            order_draws(pipeline, &selected, inputs).map_err(RendererError::ForwardPipeline)?;
        let mut meshes = Vec::new();
        let mut feature_draws = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            match input.source() {
                ForwardDrawSource::Scene(_) => meshes.push(index),
                ForwardDrawSource::Feature { .. } => feature_draws.push(index),
            }
        }
        // Every referenced asset is now cloned in `inputs`; transform/tint
        // values are copied into frame-owned upload bytes by realization.
        self.submit_prepared_forward(
            snapshot.clone(),
            inputs,
            extent,
            topology,
            assets,
            surface,
            color_format,
            preparation_started,
            |phase| {
                let mut context = crate::ForwardCompositionContext::new(
                    phase,
                    feature.feature,
                    &prepared,
                    &meshes,
                    &feature_draws,
                );
                pipeline.contribute_composed(&selected, &mut context)
            },
        )
        .await
    }
}

type Result<T> = std::result::Result<T, RendererError>;

fn order_draws<P: FramePipeline<crate::ForwardResources>>(
    pipeline: &P,
    frame: &crate::PreparedFrame<'_, crate::ForwardResources>,
    inputs: Vec<ForwardDrawInput>,
) -> std::result::Result<Vec<ForwardDrawInput>, crate::PipelineError> {
    let mut order: Vec<_> = inputs
        .iter()
        .map(|input| ForwardDrawOrder::new(input.source(), input.order()))
        .collect();
    let mut by_source = HashMap::with_capacity(inputs.len());
    for (metadata, input) in order.iter().copied().zip(inputs) {
        if by_source
            .insert(metadata.source(), (metadata, input))
            .is_some()
        {
            return Err(crate::PipelineError::InvalidDrawOrder);
        }
    }
    pipeline.order_forward(frame, &mut order)?;
    let mut ordered = Vec::with_capacity(order.len());
    for metadata in order {
        let (original, input) = by_source
            .remove(&metadata.source())
            .ok_or(crate::PipelineError::InvalidDrawOrder)?;
        if metadata != original {
            return Err(crate::PipelineError::InvalidDrawOrder);
        }
        ordered.push(input);
    }
    Ok(ordered)
}

#[cfg(test)]
#[path = "../../tests/renderer/forward_composed_order.rs"]
mod tests;
