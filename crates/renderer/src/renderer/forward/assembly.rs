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
    pub(in crate::renderer) async fn submit_prepared_forward<F>(
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
                depth_mode,
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
                .saturating_add(assembled.allocations.buffer_count);
            self.forward_stats.accepted_buffer_allocation_bytes = self
                .forward_stats
                .accepted_buffer_allocation_bytes
                .saturating_add(assembled.allocations.buffer_bytes);
            self.forward_stats.accepted_texture_allocations = self
                .forward_stats
                .accepted_texture_allocations
                .saturating_add(assembled.allocations.texture_count);
        }
        result
    }
}

pub(in crate::renderer) enum AssembledForwardTarget<B: ResourceBackend> {
    Offscreen {
        frame: crate::residency::ForwardFrame<B>,
    },
    Surface {
        frame: crate::residency::SurfaceForwardFrame<B>,
        image: SurfaceImage<B>,
    },
}

pub(in crate::renderer) struct AssembledForward<B: ResourceBackend> {
    pub(in crate::renderer) retained: PendingInputs,
    pub(in crate::renderer) target: AssembledForwardTarget<B>,
    pub(in crate::renderer) publication: ForwardPublication,
    pub(in crate::renderer) topology: crate::FrameTopology,
    pub(in crate::renderer) batch_stats: crate::preparation::ForwardBatchStats,
    pub(in crate::renderer) preparation_elapsed_ns: u128,
    pub(in crate::renderer) allocations: ForwardAllocationStats,
}

/// Allocation facts produced while assembling a Forward frame.
///
/// Depth is a texture allocation whenever the selected depth policy owns one;
/// the other counts already represent the concrete resident and frame plans.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::renderer) struct ForwardAllocationStats {
    pub(in crate::renderer) buffer_count: u64,
    pub(in crate::renderer) buffer_bytes: u64,
    pub(in crate::renderer) texture_count: u64,
}

impl ForwardAllocationStats {
    /// Counts owners requested by a concrete resident upload plan before that
    /// plan is consumed by device allocation.
    pub(super) fn from_resident_descriptors(
        streams: &[fluxel_rhi::api::BufferDesc],
        upload_bytes: u64,
        texture_count: usize,
    ) -> Self {
        Self {
            buffer_count: u64::try_from(streams.len())
                .unwrap_or(u64::MAX)
                .saturating_add(1),
            buffer_bytes: streams
                .iter()
                .fold(upload_bytes, |total, desc| total.saturating_add(desc.size)),
            texture_count: u64::try_from(texture_count).unwrap_or(u64::MAX),
        }
    }

    /// Adds allocations selected by the concrete frame target and depth mode.
    pub(super) fn with_frame(
        self,
        frame_buffer_count: u64,
        frame_buffer_bytes: u64,
        frame_texture_count: u64,
        owns_depth: bool,
    ) -> Self {
        Self {
            buffer_count: self.buffer_count.saturating_add(frame_buffer_count),
            buffer_bytes: self.buffer_bytes.saturating_add(frame_buffer_bytes),
            texture_count: self
                .texture_count
                .saturating_add(frame_texture_count)
                .saturating_add(u64::from(owns_depth)),
        }
    }
}
