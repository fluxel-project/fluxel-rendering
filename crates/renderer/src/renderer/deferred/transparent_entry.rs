//! Material partition and actual attachments for Deferred transparent work.

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
    pub(super) async fn submit_prepared_deferred_transparent<F>(
        &mut self,
        snapshot: FrameSnapshot<ForwardResources>,
        inputs: Vec<crate::ForwardDrawInput>,
        extent: [u32; 2],
        assets: &ForwardAssets,
        surface: Option<SurfaceImage<B>>,
        contribute: F,
    ) -> Result<Submission, RendererError>
    where
        F: FnOnce(&mut crate::DeferredPhaseContext<'_, '_>) -> Result<(), crate::PipelineError>,
    {
        let started = Instant::now();
        // Environment transmission needs the Forward PBR route even when its material is
        // opaque or masked: Deferred G-buffer programs cannot consume the
        // selected specular IBL transmission bindings.  The ordered overlay
        // therefore contains all transmission plus the normal blend segment.
        // Its non-blended members retain depth writes, before the sorted blend
        // segment, so overlapping transmissive shells still occlude correctly.
        let (overlay, opaque): (Vec<_>, Vec<_>) = inputs
            .into_iter()
            .partition(super::requires_forward_overlay);
        let environment = self.environment_irradiance.clone().filter(|_| {
            overlay
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        if opaque.is_empty() {
            // No selected opaque records means there is no G-buffer product.
            // This is an empty partition, not recovery from failed admission.
            let format = surface
                .as_ref()
                .map_or(TextureFormat::Rgba8Unorm, SurfaceImage::format);
            return self
                .submit_prepared_forward(
                    snapshot,
                    overlay,
                    extent,
                    crate::FrameTopology::SortedForward,
                    crate::ForwardDepthMode::Depth32LessEqualWrite,
                    assets,
                    surface.map(|image| (image, extent)),
                    format,
                    started,
                    environment,
                    |context| {
                        context.sorted(|pass| {
                            pass.depends_on_uploads()?;
                            pass.read_vertex_streams()?;
                            pass.read_frame_uniforms()?;
                            pass.read_sampled_textures()?;
                            pass.write_selected_depth()?;
                            pass.read_write_main_color()
                        })
                    },
                )
                .await;
        }
        if overlay.is_empty() {
            return Err(Error::InvalidState.into());
        }
        // This is the same descriptor used by actual lighting allocation.
        // Ordinary opaque Deferred does not acquire a blending requirement.
        let desc = crate::residency::deferred_scene_color_desc_with_format(
            extent,
            self.deferred_scene_color_format(),
        );
        super::super::require_blendable_target(
            &self.device,
            &desc,
            crate::FrameTopology::SortedForward,
        )
        .map_err(|error| self.observe_renderer_error(error.into()))?;
        self.deferred_residency.retain_assets(assets);
        let opaque_frame = self
            .prepare_deferred_frames(&opaque)
            .await
            .map_err(|error| self.observe_renderer_error(error))?;
        let attachments = BorrowedForwardAttachments::new(
            opaque_frame.scene_color.clone(),
            opaque_frame.gbuffer[2].clone(),
        );
        let assembled = self
            .assemble_prepared_forward_with_attachments(
                snapshot.clone(),
                overlay,
                extent,
                crate::FrameTopology::SortedForward,
                crate::ForwardDepthMode::Depth32LessEqualWrite,
                assets,
                None,
                opaque_frame.scene_color.desc.format,
                started,
                None,
                true,
                Some(attachments),
                environment,
            )
            .await?;
        let AssembledForwardTarget::Offscreen { frame: overlay } = assembled.target else {
            return Err(Error::InvalidState.into());
        };
        let PendingInputs::Frame { mut _inputs, .. } = assembled.retained else {
            return Err(Error::InvalidState.into());
        };
        let mut inputs = opaque;
        inputs.append(&mut _inputs);
        let target = match surface {
            Some(image) => {
                let composite = self.surface_composite_owner(&overlay, &image).await?;
                super::super::forward::OitExecutionTarget::Surface { image, composite }
            }
            None => super::super::forward::OitExecutionTarget::Readback,
        };
        let allocation = assembled.allocations;
        let batch_stats = assembled.batch_stats;
        let preparation_ns = assembled.preparation_elapsed_ns;
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
            .execute_deferred_transparent(
                PendingInputs::Frame {
                    _snapshot: snapshot,
                    _inputs: inputs,
                },
                opaque_frame,
                overlay,
                assembled.publication,
                target,
                contribute,
            )
            .await;
        if result.is_ok() {
            let stats = &mut self.forward_stats;
            stats.accepted_buffer_allocations = stats
                .accepted_buffer_allocations
                .saturating_add(allocation.buffer_count);
            stats.accepted_buffer_allocation_bytes = stats
                .accepted_buffer_allocation_bytes
                .saturating_add(allocation.buffer_bytes);
            stats.accepted_texture_allocations = stats
                .accepted_texture_allocations
                .saturating_add(allocation.texture_count);
            stats.resident_copy_bytes = stats.resident_copy_bytes.saturating_add(resident_bytes);
            stats.frame_upload_bytes = stats.frame_upload_bytes.saturating_add(uniform_bytes);
            stats.preparation_elapsed_ns =
                stats.preparation_elapsed_ns.saturating_add(preparation_ns);
            stats.opaque_candidates = stats
                .opaque_candidates
                .saturating_add(u64::try_from(batch_stats.candidates).unwrap_or(u64::MAX));
            stats.opaque_batches = stats
                .opaque_batches
                .saturating_add(u64::try_from(batch_stats.batches).unwrap_or(u64::MAX));
            stats.opaque_draws = stats
                .opaque_draws
                .saturating_add(u64::try_from(batch_stats.candidates).unwrap_or(u64::MAX));
            self.forward_stats.pending_peak = self
                .forward_stats
                .pending_peak
                .max(u64::try_from(self.pending()).unwrap_or(u64::MAX));
        }
        result
    }
}
