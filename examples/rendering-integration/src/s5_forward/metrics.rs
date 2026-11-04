//! Persisted, consumer-observed S5 Forward measurement facts.

use std::{fs, io, path::Path};

use fluxel_renderer::ForwardCacheStats;

pub(super) fn write(
    output: &Path,
    initial: ForwardCacheStats,
    final_stats: ForwardCacheStats,
    initial_compilations: usize,
    final_compilations: usize,
) -> io::Result<()> {
    fs::write(
        output.join("s5-forward-measurements.json"),
        format!(
            concat!(
                "{{\"schemaVersion\":1,",
                "\"preparationBoundary\":\"accepted submit_forward preparation before Graph execution; includes synchronous API calls, CPU preparation and owner allocation; excludes Graph execution, GPU completion and readback\",",
                "\"preparationElapsedNs\":{},",
                "\"graphCompileElapsedNs\":{},",
                "\"acceptedBufferAllocations\":{},",
                "\"acceptedBufferAllocationBytes\":{},",
                "\"acceptedTextureAllocations\":{},",
                "\"pendingPeak\":{},",
                "\"residentStaticUploadBytes\":{},",
                "\"frameUploadBytes\":{},",
                "\"geometryCacheHits\":{},",
                "\"textureCacheHits\":{},",
                "\"frameAllocationReuses\":{},",
                "\"planCompilations\":{}",
                "}}\n"
            ),
            final_stats
                .preparation_elapsed_ns
                .saturating_sub(initial.preparation_elapsed_ns),
            final_stats
                .graph_compile_elapsed_ns
                .saturating_sub(initial.graph_compile_elapsed_ns),
            final_stats
                .accepted_buffer_allocations
                .saturating_sub(initial.accepted_buffer_allocations),
            final_stats
                .accepted_buffer_allocation_bytes
                .saturating_sub(initial.accepted_buffer_allocation_bytes),
            final_stats
                .accepted_texture_allocations
                .saturating_sub(initial.accepted_texture_allocations),
            final_stats.pending_peak.max(initial.pending_peak),
            final_stats
                .resident_copy_bytes
                .saturating_sub(initial.resident_copy_bytes),
            final_stats
                .frame_upload_bytes
                .saturating_sub(initial.frame_upload_bytes),
            final_stats
                .geometry_hits
                .saturating_sub(initial.geometry_hits),
            final_stats
                .texture_hits
                .saturating_sub(initial.texture_hits),
            final_stats
                .frame_reuses
                .saturating_sub(initial.frame_reuses),
            final_compilations.saturating_sub(initial_compilations),
        ),
    )
}
