//! Device realization for the material-backed Forward consumer.
use super::{
    NEXT_PLAN_ORDINAL, PendingInputs, PendingWork, Renderer, RendererError, SLOT_COUNT, Submission,
    admission::{self, PipelineCapabilityFacts},
};
use crate::PreparedMaterial;
use fluxel_rhi::{
    api::*,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use std::sync::Arc;

mod borrowed_attachments;
pub(in crate::renderer) use borrowed_attachments::BorrowedForwardAttachments;

/// Cumulative traffic and completed-product reuse in the Forward coordinator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ForwardCacheStats {
    /// Accepted preparation elapsed time before Graph execution starts.
    ///
    /// This wall-clock boundary includes synchronous API calls made while
    /// resolving, allocating and assembling the Forward frame. It is not a
    /// CPU-cycle counter and excludes Graph execution, GPU completion and
    /// readback.
    pub preparation_elapsed_ns: u128,
    /// Accepted cache-miss Graph compilation elapsed time.
    pub graph_compile_elapsed_ns: u128,
    /// Buffers created for accepted Forward submissions.
    pub accepted_buffer_allocations: u64,
    /// Requested bytes of those newly created buffer owners.
    pub accepted_buffer_allocation_bytes: u64,
    /// Textures created for accepted Forward submissions.
    pub accepted_texture_allocations: u64,
    /// Highest Renderer pending count observed after accepting Forward work.
    pub pending_peak: u64,
    /// Completed geometry products reused during preparation.
    pub geometry_hits: u64,
    /// Completed texture products reused during preparation.
    pub texture_hits: u64,
    /// Static upload source footprints in accepted submissions, including row padding.
    pub resident_copy_bytes: u64,
    /// Per-frame uniform bytes copied, including physical alignment padding.
    pub frame_upload_bytes: u64,
    /// Completed frame allocation sets reused after CPU readback finished.
    pub frame_reuses: u64,
    /// Prepared opaque candidates accepted for Forward submission.
    pub opaque_candidates: u64,
    /// Contiguous compatible raster batches accepted for Forward submission.
    pub opaque_batches: u64,
    /// Opaque draw calls accepted for Forward submission.
    pub opaque_draws: u64,
}

#[derive(Default)]
pub(crate) struct ForwardPublication {
    geometry: Vec<(crate::residency::ForwardGeometryCacheKey, Vec<usize>)>,
    textures: Vec<(crate::residency::ForwardTextureCacheKey, usize)>,
}

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    fn observe_forward_error(&mut self, operation: &'static str, error: Error) -> RendererError {
        self.note_device_loss(&error);
        RendererError::ForwardOperation { operation, error }
    }

    /// Returns Forward upload traffic and completed resource cache hits.
    pub fn forward_cache_stats(&self) -> ForwardCacheStats {
        self.forward_stats
    }

    pub(super) fn publish_forward_completion(&mut self, work: &super::PendingWork<B>) {
        let publish =
            |frame: &crate::residency::ForwardResidentBuffers<B>,
             publication: &ForwardPublication,
             cache: &mut crate::residency::ForwardResidentCache<B>| {
                for (key, mapping) in &publication.geometry {
                    let streams: Vec<_> =
                        mapping.iter().map(|index| &frame.streams[*index]).collect();
                    cache.publish_geometry(*key, &streams);
                }
                for (key, index) in &publication.textures {
                    cache.publish_texture(*key, &frame.textures[*index]);
                }
            };
        match work {
            super::PendingWork::ForwardDeferredCapture { _producer, .. }
            | super::PendingWork::ForwardDeferredCaptureSurface { _producer, .. } => {
                publish(
                    &_producer.frame.resident,
                    &_producer.publication,
                    &mut self.forward_residency,
                );
            }
            super::PendingWork::DeferredForwardCapture {
                _composition,
                _receiver,
                ..
            }
            | super::PendingWork::DeferredForwardCaptureSurface {
                _composition,
                _receiver,
                ..
            } => {
                if let Some(composition) = _composition {
                    publish(
                        &composition.frame.resident,
                        &composition.publication,
                        &mut self.forward_residency,
                    );
                }
                publish(
                    &_receiver.frame.resident,
                    &_receiver.publication,
                    &mut self.forward_residency,
                );
            }
            super::PendingWork::DeferredCapture {
                _composition: Some(composition),
                ..
            }
            | super::PendingWork::DeferredCaptureSurface {
                _composition: Some(composition),
                ..
            } => {
                publish(
                    &composition.frame.resident,
                    &composition.publication,
                    &mut self.forward_residency,
                );
            }
            super::PendingWork::Forward {
                _frame,
                publication,
                ..
            }
            | super::PendingWork::ForwardOitSurface {
                _frame,
                publication,
                ..
            } => {
                publish(&_frame.resident, publication, &mut self.forward_residency);
            }
            super::PendingWork::DeferredTransparent {
                _transparent,
                publication,
                ..
            }
            | super::PendingWork::DeferredOit {
                _transparent,
                publication,
                ..
            }
            | super::PendingWork::DeferredOitSurface {
                _transparent,
                publication,
                ..
            }
            | super::PendingWork::DeferredTransparentSurface {
                _transparent,
                publication,
                ..
            } => {
                publish(
                    &_transparent.resident,
                    publication,
                    &mut self.forward_residency,
                );
            }
            super::PendingWork::ForwardSurface {
                _frame,
                publication,
                ..
            } => {
                publish(&_frame.resident, publication, &mut self.forward_residency);
            }
            super::PendingWork::CaptureForward {
                _capture,
                _receiver,
                capture_publication,
                receiver_publication,
                ..
            }
            | super::PendingWork::CaptureOit {
                _capture,
                _receiver,
                capture_publication,
                receiver_publication,
                ..
            }
            | super::PendingWork::CaptureOitSurface {
                _capture,
                _receiver,
                capture_publication,
                receiver_publication,
                ..
            }
            | super::PendingWork::CaptureForwardSurface {
                _capture,
                _receiver,
                capture_publication,
                receiver_publication,
                ..
            } => {
                publish(
                    &_capture.resident,
                    capture_publication,
                    &mut self.forward_residency,
                );
                publish(
                    &_receiver.resident,
                    receiver_publication,
                    &mut self.forward_residency,
                );
            }
            _ => {}
        }
    }
    pub(super) fn recycle_completed_readback(&mut self, work: super::PendingWork<B>, slot: usize) {
        match work {
            super::PendingWork::Forward { _frame, _oit, .. } => {
                if _oit.is_some() {
                    // OIT samples the opaque target; the ordinary pool expects
                    // a final CopySrc target. Retire this frame at completion.
                    return;
                }
                let frame = *_frame;
                let had_draws = !frame.draws.is_empty();
                self.forward_frames[slot] =
                    Some(crate::frame_data::pool::recycle(frame.frame, had_draws));
            }
            super::PendingWork::Deferred { _frame, .. } => {
                let frame = *_frame;
                self.deferred_frames[slot] = Some(crate::frame_data::deferred_pool::recycle(
                    frame.frame,
                    frame.gbuffer,
                    frame.effects,
                ));
            }
            _ => {}
        }
    }

    pub(super) fn recycle_forward_surface(&mut self, work: super::PendingWork<B>, slot: usize) {
        if let super::PendingWork::ForwardSurface { _frame, .. } = work {
            // Surface leases retire at completion, but these buffers are not
            // compatible with the offscreen readback pool.
            drop(_frame);
            let _ = slot;
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct ForwardProgramKey {
    program: fluxel_shader::ShaderArtifactKey,
    layout: Vec<crate::preparation::VertexLayoutKey>,
    color_format: TextureFormat,
    transform_uniform_bytes: u64,
    depth_mode: crate::ForwardDepthMode,
    surface: crate::renderer::RasterSurfaceState,
}
impl ForwardProgramKey {
    fn for_material(
        prepared: &PreparedMaterial,
        color_format: TextureFormat,
        transform_uniform_bytes: u64,
        depth_mode: crate::ForwardDepthMode,
        surface: crate::renderer::RasterSurfaceState,
    ) -> Self {
        Self {
            program: prepared.shader().cache_key().clone(),
            layout: prepared
                .geometry()
                .streams
                .iter()
                .map(crate::preparation::vertex_layout_key)
                .collect(),
            color_format,
            transform_uniform_bytes,
            depth_mode,
            surface,
        }
    }
}
pub(super) struct ForwardPipelineObjects<B: ResourceBackend> {
    /// Immutable facts of the concrete cached pipeline and its layouts.
    /// OIT must compare against these, not reconstruct expected facts.
    pub(super) key: ForwardProgramKey,
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) material_layout: BindGroupLayout<B>,
    pub(super) transform_layout: BindGroupLayout<B>,
    /// Present only for the explicit lat-long diffuse-irradiance shader route.
    pub(super) environment_layout: Option<BindGroupLayout<B>>,
}

mod assembly;
pub(in crate::renderer) use assembly::AssembledForwardTarget;
pub(in crate::renderer) use composed::order_draws as order_forward_draws;
mod capture_composed;
mod capture_entry;
mod capture_execute;
mod capture_oit_entry;
mod capture_oit_execute;
mod captured;
mod composed;
mod composed_pair;
mod depth;
mod entry;
mod execute;
mod oit_composed;
mod oit_composed_pair;
mod oit_entry;
mod oit_execute;
pub(in crate::renderer) use oit_execute::OitExecutionTarget;
mod oit_order;
pub(in crate::renderer) use oit_order::order_oit_draws as order_forward_oit_draws;
mod oit_prepared;
mod oit_program;
mod oit_resources;
pub(super) use oit_resources::OitFrameResources;
mod prepare;
mod prepare_entry;
mod prepare_frames;
mod sampler;
mod surface_composite;
mod target_admission;
mod texture_upload;
#[cfg(test)]
pub(crate) use sampler::validate_sampled_texture_owner;
pub(crate) use sampler::{
    default_material_sampler, material_requires_ordinary_sampler, material_sampler,
    material_sampler_descriptors, material_sampler_for_input,
    require_filterable_material_texture_with_mip_levels, same_material_sampler,
    validate_capture_material_texture, validate_material_sampler, validate_material_texture_owner,
    validate_ordinary_sampler,
};
mod program;
mod program_selection;
pub(in crate::renderer) use capture_entry::{capture_assets, validate_receiver_sources};
pub(in crate::renderer) use capture_oit_entry::CaptureOitSide;
pub(in crate::renderer) use captured::CaptureTextureOwner;
pub use captured::CaptureTextureSlot;
pub(crate) use surface_composite::SurfaceCompositeOwner;

fn aligned_forward_row(width: u32) -> std::result::Result<u32, RendererError> {
    let row = u64::from(width).checked_mul(4).ok_or(Error::InvalidRange)?;
    let aligned = row.checked_add(255).ok_or(Error::InvalidRange)? / 256 * 256;
    u32::try_from(aligned).map_err(|_| Error::InvalidRange.into())
}
