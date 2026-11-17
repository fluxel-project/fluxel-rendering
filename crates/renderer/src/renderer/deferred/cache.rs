//! Deferred cache observations and completion-gated publication.

use super::*;

/// Counters for Deferred preparation and reuse on this Renderer device.
///
/// Hits are CPU cache lookups, including preparations later refused by Graph.
/// Allocation/upload counters describe successfully assembled frames, not failed
/// partial allocation attempts or measured GPU transfer time.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeferredCacheStats {
    /// Completed immutable geometry generations selected from the cache.
    pub geometry_hits: u64,
    /// Completed immutable texture generations selected from the cache.
    pub texture_hits: u64,
    /// Realized G-buffer programs selected without device program creation.
    pub gbuffer_program_hits: u64,
    /// Realized lighting programs selected without device program creation.
    pub lighting_program_hits: u64,
    /// Plans reused after this frame's real declaration and owner validation.
    pub graph_hits: u64,
    /// Completed frame/G-buffer allocation sets reused after CPU readback.
    pub frame_reuses: u64,
    /// Newly allocated buffers in successfully assembled frames.
    pub buffer_allocations: u64,
    /// Newly allocated textures in successfully assembled frames.
    pub texture_allocations: u64,
    /// Static copy-source bytes prepared for successfully assembled frames.
    pub resident_upload_bytes: u64,
    /// Dynamic uniform upload bytes prepared for successfully assembled frames.
    pub frame_upload_bytes: u64,
}

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Returns Deferred preparation and reuse counters for this device owner.
    pub fn deferred_cache_stats(&self) -> DeferredCacheStats {
        self.deferred_stats
    }

    pub(in crate::renderer) fn publish_deferred_completion(
        &mut self,
        work: &super::super::PendingWork<B>,
    ) {
        let (publication, deformation, resident) = match work {
            super::super::PendingWork::Deferred { _frame: frame, .. } => (
                &frame.publication,
                frame.deformation.as_ref(),
                &frame.resident,
            ),
            super::super::PendingWork::DeferredSurface { _frame: frame, .. } => (
                &frame.publication,
                frame.deformation.as_ref(),
                &frame.resident,
            ),
            _ => return,
        };
        // Deformation reads its fresh packed source and product. The otherwise
        // unused original resident stream is not necessarily in vertex-read
        // state, so it must not be published with that outgoing-state promise.
        if deformation.is_none() {
            for (key, mapping) in &publication.geometry {
                let streams: Option<Vec<_>> = mapping
                    .iter()
                    .map(|&index| resident.streams.get(index))
                    .collect();
                if let Some(streams) = streams {
                    self.deferred_residency.publish_geometry(*key, &streams);
                }
            }
        }
        for (key, index) in &publication.textures {
            if let Some(texture) = resident.textures.get(*index) {
                self.deferred_residency.publish_texture(*key, texture);
            }
        }
    }

    pub(in crate::renderer) fn clear_deferred_caches(&mut self) {
        self.deferred_programs.clear();
        self.deferred_plans.clear();
        self.deferred_surface_plans.clear();
        self.deferred_residency.clear();
        self.deferred_frames = std::array::from_fn(|_| None);
    }
}
