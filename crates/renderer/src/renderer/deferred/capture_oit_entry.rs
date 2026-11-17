//! Public finite-OIT capture producer entries for Deferred receivers.

use super::*;
use crate::renderer::forward::CaptureTextureSlot;

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
    /// Captures a producer prefix plus selected finite-OIT transparency, then
    /// shades the receiver. A nonempty producer with no opaque prefix remains
    /// in the actual Forward namespace instead of creating an empty G-buffer.
    /// A blended receiver uses the Forward capture-receiver branch and can
    /// select its own finite OIT work.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_oit_capture<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: &crate::DualDepthPeelingConfig,
        receiver_config: &crate::DualDepthPeelingConfig,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            receiver,
            Some(producer_config),
            Some(receiver_config),
            None,
        )
        .await
    }

    /// Presents the capture receiver after its producer's selected finite-OIT
    /// transparency has been composed in the same Graph. A nonempty producer
    /// with no opaque prefix remains in the actual Forward namespace; blended
    /// receivers use the Forward capture-receiver branch.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_oit_capture_surface<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: &crate::DualDepthPeelingConfig,
        receiver_config: &crate::DualDepthPeelingConfig,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            receiver,
            Some(producer_config),
            Some(receiver_config),
            Some(image),
        )
        .await
    }
}
