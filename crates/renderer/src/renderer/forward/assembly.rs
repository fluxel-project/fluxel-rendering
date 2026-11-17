//! Execution of an already assembled Forward frame.

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
    pub(super) async fn submit_prepared_forward<F>(
        &mut self,
        retained_snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        prepared_inputs: Vec<crate::ForwardDrawInput>,
        extent: [u32; 2],
        topology: crate::FrameTopology,
        assets: &crate::ForwardAssets,
        surface: Option<(SurfaceImage<B>, [u32; 2])>,
        color_format: TextureFormat,
        preparation_started: Instant,
        environment_snapshot: Option<
            fluxel_assets::AssetSnapshot<
                crate::EnvironmentIrradianceAsset,
                crate::EnvironmentIrradianceData,
            >,
        >,
        contribute: F,
    ) -> std::result::Result<super::Submission, RendererError>
    where
        F: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let assembled = self
            .assemble_prepared_forward(
                retained_snapshot,
                prepared_inputs,
                extent,
                topology,
                assets,
                surface,
                color_format,
                preparation_started,
                None,
                false,
                environment_snapshot,
            )
            .await?;
        let result = match assembled.target {
            AssembledForwardTarget::Offscreen { frame } => {
                self.execute_forward_with_contribution(
                    assembled.retained,
                    frame,
                    assembled.publication,
                    assembled.topology,
                    assembled.batch_stats,
                    contribute,
                )
                .await
            }
            AssembledForwardTarget::Surface { frame, image } => {
                self.execute_forward_surface(
                    assembled.retained,
                    frame,
                    assembled.publication,
                    assembled.topology,
                    assembled.batch_stats,
                    image,
                    contribute,
                )
                .await
            }
        };
        if result.is_ok() {
            self.forward_stats.preparation_elapsed_ns = self
                .forward_stats
                .preparation_elapsed_ns
                .saturating_add(assembled.preparation_elapsed_ns);
            self.forward_stats.accepted_buffer_allocations = self
                .forward_stats
                .accepted_buffer_allocations
                .saturating_add(assembled.buffer_allocations);
            self.forward_stats.accepted_buffer_allocation_bytes = self
                .forward_stats
                .accepted_buffer_allocation_bytes
                .saturating_add(assembled.buffer_allocation_bytes);
            self.forward_stats.accepted_texture_allocations = self
                .forward_stats
                .accepted_texture_allocations
                .saturating_add(assembled.texture_allocations);
        }
        result
    }
}

pub(super) enum AssembledForwardTarget<B: ResourceBackend> {
    Offscreen {
        frame: crate::residency::ForwardFrame<B>,
    },
    Surface {
        frame: crate::residency::SurfaceForwardFrame<B>,
        image: SurfaceImage<B>,
    },
}

pub(super) struct AssembledForward<B: ResourceBackend> {
    pub(super) retained: PendingInputs,
    pub(super) target: AssembledForwardTarget<B>,
    pub(super) publication: ForwardPublication,
    pub(super) topology: crate::FrameTopology,
    pub(super) batch_stats: crate::preparation::ForwardBatchStats,
    pub(super) preparation_elapsed_ns: u128,
    pub(super) buffer_allocations: u64,
    pub(super) buffer_allocation_bytes: u64,
    pub(super) texture_allocations: u64,
}
