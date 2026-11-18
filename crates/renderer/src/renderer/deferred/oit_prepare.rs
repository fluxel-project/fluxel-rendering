//! Preparation of Deferred opaque geometry and its finite OIT material suffix.

use super::*;
use crate::renderer::forward::{AssembledForwardTarget, BorrowedForwardAttachments};
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
    pub(super) async fn prepare_deferred_oit<F>(
        &mut self,
        snapshot: FrameSnapshot<ForwardResources>,
        partition: crate::preparation::OitDrawPartition,
        extent: [u32; 2],
        assets: &ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
        surface: Option<SurfaceImage<B>>,
        started: Instant,
        contribute: F,
    ) -> Result<Submission, RendererError>
    where
        F: FnOnce(&mut crate::DeferredPhaseContext<'_, '_>) -> Result<(), crate::PipelineError>,
    {
        let crate::preparation::OitDrawPartition {
            mut draws,
            opaque_count,
            peel_count,
        } = partition;
        let transparent_inputs = draws.split_off(opaque_count);
        let opaque_inputs = draws;
        self.deferred_residency.retain_assets(assets);
        let opaque = self.prepare_deferred_frames(&opaque_inputs).await?;
        let environment = self.environment_irradiance.clone().filter(|_| {
            transparent_inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        let assembled = self
            .assemble_prepared_forward_with_attachments(
                snapshot.clone(),
                transparent_inputs,
                extent,
                crate::FrameTopology::SortedForward,
                crate::ForwardDepthMode::Depth32LessEqualWrite,
                assets,
                None,
                // The overlay continues Deferred's lit scene color rather
                // than starting a second display-space target. Its OIT
                // result remains linear HDR until the Deferred terminal.
                opaque.scene_color.desc.format,
                started,
                None,
                true,
                Some(BorrowedForwardAttachments::new(
                    opaque.scene_color.clone(),
                    opaque.gbuffer[2].clone(),
                )),
                1,
                environment,
            )
            .await?;
        let AssembledForwardTarget::Offscreen { frame: overlay } = assembled.target else {
            return Err(Error::InvalidState.into());
        };
        let PendingInputs::Frame {
            _inputs: transparent_inputs,
            ..
        } = assembled.retained
        else {
            return Err(Error::InvalidState.into());
        };
        let oit = self
            .assemble_oit_resources(
                &overlay,
                &transparent_inputs,
                &(0..peel_count).collect::<Vec<_>>(),
                &mut fluxel_shader::ShaderComposer::new(),
            )
            .await?;
        let composite = match surface.as_ref() {
            Some(image) => {
                self.oit_surface_composite_owner(&oit.output, extent, image)
                    .await?
            }
            None => {
                self.oit_offscreen_composite_owner(
                    &oit.output,
                    extent,
                    opaque.frame.target_desc.format,
                )
                .await?
            }
        };
        let mut inputs = opaque_inputs;
        inputs.extend(transparent_inputs);
        let allocations = assembled.allocations;
        let batches = assembled.batch_stats;
        let uniform_bytes = u64::try_from(overlay.uniform_bytes.len()).unwrap_or(u64::MAX);
        let resident_bytes = overlay
            .resident_upload
            .vertices
            .iter()
            .map(|copy| copy.destination.end - copy.destination.start)
            .chain(
                overlay
                    .resident_upload
                    .textures
                    .iter()
                    .map(|copy| u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image)),
            )
            .fold(0_u64, u64::saturating_add);
        let result = self
            .execute_deferred_oit(
                PendingInputs::Frame {
                    _snapshot: snapshot,
                    _inputs: inputs,
                },
                opaque,
                overlay,
                oit,
                assembled.publication,
                composite,
                peel_count,
                config.rounds(),
                surface,
                contribute,
            )
            .await;
        if result.is_ok() {
            let stats = &mut self.forward_stats;
            stats.accepted_buffer_allocations = stats
                .accepted_buffer_allocations
                .saturating_add(allocations.buffer_count);
            stats.accepted_buffer_allocation_bytes = stats
                .accepted_buffer_allocation_bytes
                .saturating_add(allocations.buffer_bytes);
            stats.accepted_texture_allocations = stats
                .accepted_texture_allocations
                .saturating_add(allocations.texture_count.saturating_add(8));
            stats.frame_upload_bytes = stats.frame_upload_bytes.saturating_add(uniform_bytes);
            stats.resident_copy_bytes = stats.resident_copy_bytes.saturating_add(resident_bytes);
            stats.opaque_candidates = stats
                .opaque_candidates
                .saturating_add(u64::try_from(batches.candidates).unwrap_or(u64::MAX));
            stats.opaque_draws = stats
                .opaque_draws
                .saturating_add(u64::try_from(batches.candidates).unwrap_or(u64::MAX));
            stats.opaque_batches = stats
                .opaque_batches
                .saturating_add(u64::try_from(batches.batches).unwrap_or(u64::MAX));
            stats.preparation_elapsed_ns = stats
                .preparation_elapsed_ns
                .saturating_add(assembled.preparation_elapsed_ns);
            self.forward_stats.pending_peak = self
                .forward_stats
                .pending_peak
                .max(u64::try_from(self.pending()).unwrap_or(u64::MAX));
        }
        result
    }
}
