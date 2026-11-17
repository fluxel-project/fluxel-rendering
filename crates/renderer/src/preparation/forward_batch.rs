//! Opaque Forward raster-program compatibility and stable batching.

use std::ops::Range;

use crate::preparation::{ForwardDrawInput, VertexLayoutKey, vertex_layout_key};
use crate::renderer::{RasterSurfaceState, raster_surface_state};

/// One exact device-program compatibility class for opaque Forward draws.
///
/// Dynamic bindings remain per draw. The fixed surface state is included
/// because a negative world determinant changes front-face selection and a
/// double-sided material changes culling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ForwardCompatibilityKey {
    shader: fluxel_shader::ShaderArtifactKey,
    layouts: Vec<VertexLayoutKey>,
    surface: RasterSurfaceState,
}

impl ForwardCompatibilityKey {
    /// Actual ordinary pipeline selection retained when the program is built.
    pub(crate) fn is_blended(&self) -> bool {
        self.surface.blend_state().is_some()
    }

    pub(crate) fn from_draw(draw: &ForwardDrawInput) -> Self {
        Self {
            shader: draw.prepared().shader().cache_key().clone(),
            layouts: draw
                .prepared()
                .geometry()
                .streams
                .iter()
                .map(vertex_layout_key)
                .collect(),
            surface: raster_surface_state(draw),
        }
    }
}

/// A contiguous compatible range in [`ForwardPreparedBatches::draws`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ForwardDrawBatch {
    pub(crate) key: ForwardCompatibilityKey,
    pub(crate) draws: Range<usize>,
}

/// Deterministic Forward opaque ordering and its program-compatible ranges.
#[derive(Clone)]
pub(crate) struct ForwardPreparedBatches {
    draws: Vec<ForwardDrawInput>,
    batches: Vec<ForwardDrawBatch>,
}

impl ForwardPreparedBatches {
    /// Returns counts suitable for renderer workload reporting.
    pub(crate) fn stats(&self) -> ForwardBatchStats {
        ForwardBatchStats {
            candidates: self.draws.len(),
            batches: self.batches.len(),
        }
    }

    /// Consumes the container into ordered draws and their contiguous ranges.
    pub(crate) fn into_draws_and_batches(self) -> (Vec<ForwardDrawInput>, Vec<ForwardDrawBatch>) {
        (self.draws, self.batches)
    }
}

/// Counts derived from one Forward opaque preparation result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ForwardBatchStats {
    /// Number of prepared opaque candidates.
    pub(crate) candidates: usize,
    /// Number of contiguous raster-program compatibility classes.
    pub(crate) batches: usize,
}

/// Orders opaque Forward draws into contiguous raster-program batches.
///
/// This preserves candidate order and joins only neighbouring draws selecting
/// identical fixed-function and Shader facts.
pub(crate) fn batch_forward_draws(draws: Vec<ForwardDrawInput>) -> ForwardPreparedBatches {
    let runs = coalesce_adjacent(draws, ForwardCompatibilityKey::from_draw);
    let mut ordered = Vec::new();
    let mut batches = Vec::with_capacity(runs.len());
    for (key, members) in runs {
        let start = ordered.len();
        ordered.extend(members);
        batches.push(ForwardDrawBatch {
            key,
            draws: start..ordered.len(),
        });
    }
    ForwardPreparedBatches {
        draws: ordered,
        batches,
    }
}

fn coalesce_adjacent<T, K>(items: Vec<T>, key_for: impl Fn(&T) -> K) -> Vec<(K, Vec<T>)>
where
    K: Eq,
{
    let mut runs: Vec<(K, Vec<T>)> = Vec::new();
    for item in items {
        let key = key_for(&item);
        match runs.last_mut() {
            Some((last_key, members)) if *last_key == key => members.push(item),
            _ => runs.push((key, vec![item])),
        }
    }
    runs
}

#[cfg(test)]
#[path = "../tests/preparation/forward_batches.rs"]
mod tests;
