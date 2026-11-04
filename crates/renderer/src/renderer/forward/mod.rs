//! Device realization for the material-backed Forward consumer.
use super::{
    NEXT_PLAN_ORDINAL, PendingWork, Renderer, RendererError, SLOT_COUNT, Submission,
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
            super::PendingWork::Forward {
                _frame,
                publication,
                ..
            } => {
                publish(&_frame.resident, publication, &mut self.forward_residency);
            }
            super::PendingWork::ForwardSurface {
                _frame,
                publication,
                ..
            } => {
                publish(&_frame.resident, publication, &mut self.forward_residency);
            }
            _ => {}
        }
    }
    pub(super) fn recycle_forward_readback(&mut self, work: super::PendingWork<B>, slot: usize) {
        if let super::PendingWork::Forward { _frame, .. } = work {
            let frame = *_frame;
            let had_draws = !frame.draws.is_empty();
            self.forward_frames[slot] =
                Some(crate::frame_data::pool::recycle(frame.frame, had_draws));
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
}
impl ForwardProgramKey {
    fn for_material(prepared: &PreparedMaterial, color_format: TextureFormat) -> Self {
        Self {
            program: prepared.shader().cache_key().clone(),
            layout: prepared
                .geometry()
                .streams
                .iter()
                .map(crate::preparation::vertex_layout_key)
                .collect(),
            color_format,
        }
    }
}
pub(super) struct ForwardPipelineObjects<B: ResourceBackend> {
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) material_layout: BindGroupLayout<B>,
    pub(super) transform_layout: BindGroupLayout<B>,
}

mod composed;
mod entry;
mod execute;
mod prepare;
mod program;

fn aligned_forward_row(width: u32) -> std::result::Result<u32, RendererError> {
    let row = u64::from(width).checked_mul(4).ok_or(Error::InvalidRange)?;
    let aligned = row.checked_add(255).ok_or(Error::InvalidRange)? / 256 * 256;
    u32::try_from(aligned).map_err(|_| Error::InvalidRange.into())
}
