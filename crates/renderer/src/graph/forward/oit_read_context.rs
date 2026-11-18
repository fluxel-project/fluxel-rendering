//! Stage-bounded typed-feature reads for one OIT material pass.

use super::{ForwardGraphDraw, PipelineError, read_scope::ForwardReadState};
use crate::graph::definition::{BufferHandle, PassBuilder, TextureHandle};
use std::ops::Range;

/// A real Graph read authority restricted to one OIT draw stage.  Callers use
/// the ordinary full-frame draw namespace; this context performs the only
/// permitted full-to-local remapping before exposing a feature pass.
pub(crate) struct ForwardOitReadContext<'pass, 'builder> {
    pass: &'pass mut PassBuilder<'builder>,
    reads: ForwardReadState,
    stage: Range<usize>,
    full_draw_count: usize,
}

impl<'pass, 'builder> ForwardOitReadContext<'pass, 'builder> {
    pub(crate) fn new(
        pass: &'pass mut PassBuilder<'builder>,
        streams: Vec<BufferHandle>,
        uniform: BufferHandle,
        textures: Vec<TextureHandle>,
        stage: Range<usize>,
        draws: &[ForwardGraphDraw],
    ) -> Result<Self, PipelineError> {
        let local = draws
            .get(stage.clone())
            .ok_or(PipelineError::InvalidPhaseContribution)?
            .to_vec();
        Ok(Self {
            pass,
            reads: ForwardReadState::new(streams, uniform, textures, local)?,
            stage,
            full_draw_count: draws.len(),
        })
    }

    /// Gives a typed feature its ordinary `ForwardFeaturePass`, restricted to
    /// the intersection of its full-frame namespace and this OIT stage.
    pub(crate) fn scope(
        &mut self,
        full_indices: &[usize],
        declare: impl FnOnce(
            &mut crate::pipeline::ForwardFeaturePass<'_, '_>,
        ) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        let local = self.local_indices(full_indices)?;
        let scope = self
            .reads
            .scope(&mut *self.pass, &local)
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        crate::pipeline::with_forward_feature_scope(scope, declare)
    }

    /// Built-in scene-only declaration of every read in the bounded stage.
    pub(crate) fn all_reads(&mut self) -> Result<(), PipelineError> {
        self.reads
            .begin_global_reads()
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        let local: Vec<_> = (0..(self.stage.end - self.stage.start)).collect();
        self.reads
            .read_vertex_streams_for(&mut *self.pass, &local)
            .and_then(|_| self.reads.read_frame_uniforms_for(&mut *self.pass, &local))
            .and_then(|_| {
                self.reads
                    .read_sampled_textures_for(&mut *self.pass, &local)
            })
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }

    pub(crate) fn finish(self) -> Result<(), PipelineError> {
        if self.reads.complete() {
            Ok(())
        } else {
            Err(PipelineError::InvalidPhaseContribution)
        }
    }

    fn local_indices(&self, full_indices: &[usize]) -> Result<Vec<usize>, PipelineError> {
        let mut local = Vec::with_capacity(full_indices.len());
        for &index in full_indices {
            if index >= self.full_draw_count {
                return Err(PipelineError::InvalidPhaseContribution);
            }
            if self.stage.contains(&index) {
                local.push(index - self.stage.start);
            }
        }
        Ok(local)
    }
}
