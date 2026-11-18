//! Shared owned frame assembly for scene and typed-feature OIT consumers.

use super::assembly::AssembledForwardTarget;
use super::oit_execute::OitExecutionTarget;
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
    pub(in crate::renderer) async fn submit_prepared_oit<N, C>(
        &mut self,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        partition: crate::preparation::OitDrawPartition,
        extent: [u32; 2],
        assets: &crate::ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
        surface: Option<SurfaceImage<B>>,
        preparation_started: Instant,
        ordinary: N,
        contribute: C,
    ) -> std::result::Result<Submission, RendererError>
    where
        N: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
        C: FnMut(
            &mut crate::graph::ForwardOitReadContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let crate::preparation::OitDrawPartition {
            draws: inputs,
            opaque_count,
            peel_count,
        } = partition;
        let environment = self.environment_irradiance.clone().filter(|_| {
            inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        if peel_count == 0 {
            // No selected OIT work is a normal empty candidate set. It is not
            // a recovery route for failed capability or program admission.
            let format = surface
                .as_ref()
                .map_or(TextureFormat::Rgba8Unorm, SurfaceImage::format);
            return self
                .submit_prepared_forward(
                    snapshot.clone(),
                    inputs,
                    extent,
                    crate::FrameTopology::SortedForward,
                    crate::ForwardDepthMode::Depth32LessEqualWrite,
                    assets,
                    surface.map(|image| (image, extent)),
                    format,
                    preparation_started,
                    environment,
                    ordinary,
                )
                .await;
        }
        config
            .rounds()
            .get()
            .checked_mul(3)
            .and_then(|passes| passes.checked_add(5))
            .ok_or(Error::InvalidRange)?;
        crate::residency::OitPlan::admit(&self.device, extent)?;
        let assembled = self
            .assemble_prepared_forward(
                snapshot.clone(),
                inputs,
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
                preparation_started,
                None,
                true,
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
        let end = opaque_count
            .checked_add(peel_count)
            .ok_or(Error::InvalidRange)?;
        let candidates: Vec<_> = (opaque_count..end).collect();
        let oit = self
            .assemble_oit_resources(
                &frame,
                inputs,
                &candidates,
                &mut fluxel_shader::ShaderComposer::new(),
            )
            .await?;
        let (target, hdr_composite) = match surface {
            Some(image) => {
                let composite = self
                    .oit_surface_composite_owner(&oit.output, extent, &image)
                    .await?;
                (OitExecutionTarget::Surface { image, composite }, None)
            }
            None if frame.frame.target_desc.format == TextureFormat::Rgba16Float => {
                let composite = self
                    .oit_offscreen_composite_owner(&oit.output, extent, TextureFormat::Rgba8Unorm)
                    .await?;
                (OitExecutionTarget::Readback, Some(composite))
            }
            None => (OitExecutionTarget::Readback, None),
        };
        let preparation_elapsed_ns = preparation_started.elapsed().as_nanos();
        let result = self
            .execute_forward_oit(
                assembled.retained,
                frame,
                oit,
                assembled.publication,
                opaque_count,
                peel_count,
                config.rounds(),
                assembled.batch_stats,
                target,
                hdr_composite,
                contribute,
            )
            .await;
        if result.is_ok() {
            self.forward_stats.preparation_elapsed_ns = self
                .forward_stats
                .preparation_elapsed_ns
                .saturating_add(preparation_elapsed_ns);
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
                .saturating_add(assembled.allocations.texture_count.saturating_add(8));
        }
        result
    }
}
