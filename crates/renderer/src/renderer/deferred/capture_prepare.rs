//! Deferred capture-producer preparation with an optional Forward overlay.

use super::*;
use crate::renderer::forward::{
    AssembledForwardTarget, BorrowedForwardAttachments, ForwardPublication, OitFrameResources,
};
use std::{num::NonZeroU32, time::Instant};

/// Forward work layered on a Deferred capture producer's real lighting target.
///
/// The opaque Deferred frame is retained separately by the capture execution
/// owner. This value exists only when the producer has a transparent suffix.
pub(crate) struct DeferredCaptureComposition<B: ResourceBackend> {
    pub(in crate::renderer) frame: crate::residency::ForwardFrame<B>,
    pub(in crate::renderer) oit: Option<OitFrameResources<B>>,
    pub(in crate::renderer) publication: ForwardPublication,
    pub(in crate::renderer) peel_count: usize,
    pub(in crate::renderer) rounds: NonZeroU32,
}

impl<B: ResourceBackend> DeferredCaptureComposition<B> {
    pub(in crate::renderer) fn graph_input(&self) -> crate::graph::DeferredCaptureOverlay<'_, B> {
        crate::graph::DeferredCaptureOverlay {
            frame: &self.frame,
            oit: self.oit.as_ref().map(|oit| crate::graph::ForwardOitInput {
                owners: &oit.owners,
                output: &oit.output,
                opaque_count: 0,
                peel_count: self.peel_count,
                rounds: self.rounds,
            }),
        }
    }
}

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
    /// Builds the mandatory Deferred opaque prefix and, when present, its
    /// transparent Forward suffix against the prefix's actual attachments.
    pub(super) async fn prepare_deferred_capture_producer(
        &mut self,
        snapshot: FrameSnapshot<ForwardResources>,
        draws: Vec<crate::ForwardDrawInput>,
        opaque_count: usize,
        peel_count: usize,
        rounds: NonZeroU32,
        assets: &ForwardAssets,
        started: Instant,
        copy_scene_color: bool,
    ) -> Result<
        (
            crate::residency::DeferredFrame<B>,
            Option<DeferredCaptureComposition<B>>,
            Vec<crate::ForwardDrawInput>,
        ),
        RendererError,
    > {
        if opaque_count == 0
            || opaque_count > draws.len()
            || peel_count > draws.len() - opaque_count
        {
            return Err(Error::Unsupported("Deferred capture requires opaque geometry").into());
        }
        if peel_count != 0 {
            rounds
                .get()
                .checked_mul(3)
                .and_then(|passes| passes.checked_add(5))
                .ok_or(Error::InvalidRange)?;
        }
        let mut draws = draws;
        let transparent = draws.split_off(opaque_count);
        let opaque = draws;
        self.deferred_residency.retain_assets(assets);
        let deferred = self
            .prepare_deferred_frames_with_color_copy(
                &opaque,
                None,
                copy_scene_color && peel_count == 0,
            )
            .await?;
        if transparent.is_empty() {
            return Ok((deferred, None, opaque));
        }
        let environment = self.environment_irradiance.clone().filter(|_| {
            transparent
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        let assembled = self
            .assemble_prepared_forward_with_attachments(
                snapshot,
                transparent,
                deferred.extent,
                crate::FrameTopology::SortedForward,
                crate::ForwardDepthMode::Depth32LessEqualWrite,
                assets,
                None,
                TextureFormat::Rgba8Unorm,
                started,
                None,
                true,
                Some(BorrowedForwardAttachments::new(
                    deferred.scene_color.clone(),
                    deferred.gbuffer[2].clone(),
                )),
                environment,
            )
            .await?;
        let AssembledForwardTarget::Offscreen { frame } = assembled.target else {
            return Err(Error::InvalidState.into());
        };
        let PendingInputs::Frame {
            _inputs: transparent,
            ..
        } = assembled.retained
        else {
            return Err(Error::InvalidState.into());
        };
        let oit = if peel_count == 0 {
            None
        } else {
            Some(
                self.assemble_oit_resources(
                    &frame,
                    &transparent,
                    &(0..peel_count).collect::<Vec<_>>(),
                    &mut fluxel_shader::ShaderComposer::new(),
                )
                .await?,
            )
        };
        let mut retained = opaque;
        retained.extend(transparent);
        Ok((
            deferred,
            Some(DeferredCaptureComposition {
                frame,
                oit,
                publication: assembled.publication,
                peel_count,
                rounds,
            }),
            retained,
        ))
    }
}
