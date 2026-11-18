//! Typed producer submission through Deferred capture and its real Forward fallback.

use super::{capture_feature::FeatureCaptureProducer, *};
use crate::{
    FeatureBinding, PreparedCaptureReceiver, RenderFeature,
    renderer::forward::{CaptureReceivers, CaptureTextureSlot},
};

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
    /// Captures scene meshes and one typed Feature for an offscreen receiver.
    /// Supplies one prepared receiver or a batch of projected Blend/MaskBlend receivers.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture_with_feature<V, F>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
        F: RenderFeature,
    {
        self.submit_deferred_feature_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            receivers,
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Surface form of [`Self::submit_deferred_capture_with_feature`].
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture_surface_with_feature<V, F>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
        F: RenderFeature,
    {
        self.submit_deferred_feature_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            receivers,
            producer_config,
            receiver_config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_deferred_feature_capture_target<V, F>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        mut receivers: Vec<PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
        F: RenderFeature,
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
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::DeferredPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::DeferredPipeline)?;
        if surface
            .as_ref()
            .is_some_and(|image| image.extent() != selected.view_record().extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let receivers = if receivers.len() == 1 {
            CaptureReceivers::Single(receivers.pop().ok_or(Error::InvalidState)?)
        } else {
            CaptureReceivers::RefractionBatch(receivers)
        };
        let environment = self.environment_irradiance.is_some();
        let specular = self.environment_specular.is_some();
        let mut composer = fluxel_shader::ShaderComposer::new();
        let (receiver, retained_assets, scene_refraction) = receivers.prepare(
            &producer_slot,
            selected.view_record(),
            assets,
            environment,
            specular,
            &mut composer,
            receiver_config,
        )?;
        let handle = feature.feature.handle();
        let context = feature
            .issuer
            .context(&snapshot, crate::CpuPhase::Reference, handle)
            .map_err(RendererError::DeferredFeature)?;
        crate::validate_context(&context, feature.input, handle)
            .map_err(RendererError::DeferredFeature)?;
        let prepared = feature
            .feature
            .prepare(&context, feature.input, &selected, visibility)
            .map_err(RendererError::DeferredFeature)?;
        let draws = crate::preparation::prepare_feature_draws_with_lighting(
            &selected,
            handle,
            feature.feature.draws(&prepared),
            assets,
            &mut composer,
            environment,
            specular,
        )
        .map_err(RendererError::ForwardPreparation)?;
        let producer = FeatureCaptureProducer {
            pipeline,
            selected: &selected,
            feature: feature.feature,
            handle,
            prepared: &prepared,
        };
        self.submit_deferred_capture_prepared(
            pipeline,
            snapshot.clone(),
            &selected,
            assets,
            producer_slot,
            receiver,
            scene_refraction,
            retained_assets,
            producer_config,
            receiver_config,
            surface,
            draws,
            &producer,
        )
        .await
    }
}
