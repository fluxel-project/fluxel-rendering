//! Ordinary Forward assembly entry retaining the owned-attachment contract.

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
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn assemble_prepared_forward(
        &mut self,
        retained_snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        prepared_inputs: Vec<crate::ForwardDrawInput>,
        extent: [u32; 2],
        topology: crate::FrameTopology,
        depth_mode: crate::ForwardDepthMode,
        assets: &crate::ForwardAssets,
        surface: Option<(SurfaceImage<B>, [u32; 2])>,
        color_format: TextureFormat,
        preparation_started: Instant,
        capture_texture: Option<&super::captured::CaptureTextureOwner<B>>,
        sampled_target: bool,
        environment_snapshot: Option<
            fluxel_assets::AssetSnapshot<
                crate::EnvironmentIrradianceAsset,
                crate::EnvironmentIrradianceData,
            >,
        >,
    ) -> std::result::Result<super::assembly::AssembledForward<B>, RendererError> {
        self.assemble_prepared_forward_with_attachments(
            retained_snapshot,
            prepared_inputs,
            extent,
            topology,
            depth_mode,
            assets,
            surface,
            color_format,
            preparation_started,
            capture_texture,
            sampled_target,
            None,
            environment_snapshot,
        )
        .await
    }
}
