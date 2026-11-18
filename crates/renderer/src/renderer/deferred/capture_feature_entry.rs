//! Typed producer submission through Deferred capture and its real Forward fallback.

use super::{
    capture_feature::{FeatureCaptureProducer, PairCaptureProducer, PreparedCaptureFeature},
    *,
};
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

    /// Captures scene meshes and two distinct typed Features for an offscreen receiver.
    /// Supplies one prepared receiver or a batch of projected Blend/MaskBlend receivers.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receivers: Vec<PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_deferred_feature_capture_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            producer_slot,
            receivers,
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Surface form of [`Self::submit_deferred_capture_with_features`].
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture_surface_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receivers: Vec<PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_deferred_feature_capture_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
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
        receivers: Vec<PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
        F: RenderFeature,
    {
        let selected = self.prepare_feature_capture_frame(
            pipeline,
            &snapshot,
            view,
            visibility,
            surface.as_ref(),
        )?;
        let mut composer = fluxel_shader::ShaderComposer::new();
        let (receiver, retained_assets, scene_refraction) = self
            .prepare_feature_capture_receivers(
                receivers,
                &producer_slot,
                &selected,
                assets,
                &mut composer,
                receiver_config,
            )?;
        let handle = feature.feature.handle();
        let (feature, draws) = self.prepare_capture_feature(
            feature,
            handle,
            &snapshot,
            &selected,
            visibility,
            assets,
            &mut composer,
        )?;
        let producer = FeatureCaptureProducer {
            pipeline,
            selected: &selected,
            feature,
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

    #[allow(clippy::too_many_arguments)]
    async fn submit_deferred_feature_capture_pair_target<V, F, G>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receivers: Vec<PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        let selected = self.prepare_feature_capture_frame(
            pipeline,
            &snapshot,
            view,
            visibility,
            surface.as_ref(),
        )?;
        let (first, second) = features;
        let first_handle = first.feature.handle();
        let second_handle = second.feature.handle();
        if first_handle == second_handle {
            return Err(RendererError::DeferredFeature(
                crate::FeatureError::DuplicateFeatureBinding {
                    feature: first_handle,
                },
            ));
        }
        let mut composer = fluxel_shader::ShaderComposer::new();
        let (receiver, retained_assets, scene_refraction) = self
            .prepare_feature_capture_receivers(
                receivers,
                &producer_slot,
                &selected,
                assets,
                &mut composer,
                receiver_config,
            )?;
        let (first, mut draws) = self.prepare_capture_feature(
            first,
            first_handle,
            &snapshot,
            &selected,
            visibility,
            assets,
            &mut composer,
        )?;
        let (second, second_draws) = self.prepare_capture_feature(
            second,
            second_handle,
            &snapshot,
            &selected,
            visibility,
            assets,
            &mut composer,
        )?;
        draws.extend(second_draws);
        let producer = PairCaptureProducer {
            pipeline,
            selected: &selected,
            handles: [first.handle, second.handle],
            first,
            second,
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

    fn prepare_feature_capture_frame<'a, V>(
        &self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: &'a FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        surface: Option<&SurfaceImage<B>>,
    ) -> Result<crate::PreparedFrame<'a, ForwardResources>, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        if let Some(image) = surface {
            if image.device() != self.device.id() {
                return Err(Error::WrongDevice.into());
            }
            if !image.is_current() {
                return Err(Error::SurfaceOutdated.into());
            }
        }
        let selected = pipeline
            .prepare(snapshot, view, visibility)
            .map_err(RendererError::DeferredPipeline)?;
        selected
            .validate_for(snapshot, view)
            .map_err(RendererError::DeferredPipeline)?;
        if surface.is_some_and(|image| image.extent() != selected.view_record().extent) {
            return Err(Error::InvalidRange.into());
        }
        Ok(selected)
    }

    fn prepare_feature_capture_receivers(
        &self,
        mut receivers: Vec<PreparedCaptureReceiver>,
        producer_slot: &CaptureTextureSlot,
        selected: &crate::PreparedFrame<'_, ForwardResources>,
        assets: &ForwardAssets,
        composer: &mut fluxel_shader::ShaderComposer,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
    ) -> Result<(crate::preparation::OitDrawPartition, ForwardAssets, bool), RendererError> {
        let receivers = if receivers.len() == 1 {
            CaptureReceivers::Single(receivers.pop().ok_or(Error::InvalidState)?)
        } else {
            CaptureReceivers::RefractionBatch(receivers)
        };
        receivers.prepare(
            producer_slot,
            selected.view_record(),
            assets,
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
            composer,
            receiver_config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_capture_feature<'feature, F: RenderFeature>(
        &self,
        feature: FeatureBinding<'feature, F>,
        handle: crate::FeatureHandle,
        snapshot: &FrameSnapshot<ForwardResources>,
        selected: &crate::PreparedFrame<'_, ForwardResources>,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        composer: &mut fluxel_shader::ShaderComposer,
    ) -> Result<
        (
            PreparedCaptureFeature<'feature, F>,
            Vec<crate::ForwardDrawInput>,
        ),
        RendererError,
    > {
        let context = feature
            .issuer
            .context(snapshot, crate::CpuPhase::Reference, handle)
            .map_err(RendererError::DeferredFeature)?;
        crate::validate_context(&context, feature.input, handle)
            .map_err(RendererError::DeferredFeature)?;
        let prepared = feature
            .feature
            .prepare(&context, feature.input, selected, visibility)
            .map_err(RendererError::DeferredFeature)?;
        let draws = crate::preparation::prepare_feature_draws_with_lighting(
            selected,
            handle,
            feature.feature.draws(&prepared),
            assets,
            composer,
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        Ok((
            PreparedCaptureFeature {
                feature: feature.feature,
                handle,
                prepared,
            },
            draws,
        ))
    }
}
