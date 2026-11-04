//! Per-draw Forward read declaration capabilities.
use super::phase::{TEXTURE_READS, UNIFORM_READS, VERTEX_READS};
use super::*;

pub(super) struct ForwardReadState {
    pub(super) streams: Vec<crate::graph::definition::BufferHandle>,
    pub(super) uniform: crate::graph::definition::BufferHandle,
    pub(super) textures: Vec<crate::graph::definition::TextureHandle>,
    pub(super) draws: Vec<ForwardGraphDraw>,
    vertex_covered: Vec<bool>,
    uniform_covered: Vec<bool>,
    texture_covered: Vec<bool>,
    scope_claimed: Vec<bool>,
    global_reads_started: bool,
    scopes_started: usize,
    scopes_pending_finish: usize,
}

/// Private authority for exactly one disjoint draw subset in the active pass.
pub(crate) struct ForwardReadScope<'scope, 'builder> {
    pass: &'scope mut crate::graph::definition::PassBuilder<'builder>,
    state: &'scope mut ForwardReadState,
    indices: Vec<usize>,
    declared: u8,
}

impl ForwardReadState {
    pub(super) fn new(
        streams: Vec<crate::graph::definition::BufferHandle>,
        uniform: crate::graph::definition::BufferHandle,
        textures: Vec<crate::graph::definition::TextureHandle>,
        draws: Vec<ForwardGraphDraw>,
    ) -> Self {
        let count = draws.len();
        Self {
            streams,
            uniform,
            textures,
            draws,
            vertex_covered: vec![false; count],
            uniform_covered: vec![false; count],
            texture_covered: vec![false; count],
            scope_claimed: vec![false; count],
            global_reads_started: false,
            scopes_started: 0,
            scopes_pending_finish: 0,
        }
    }
    pub(super) fn begin_global_reads(&mut self) -> std::result::Result<(), ()> {
        if self.scopes_started != 0 {
            return Err(());
        }
        self.global_reads_started = true;
        Ok(())
    }
    fn select(&mut self, indices: &[usize]) -> std::result::Result<Vec<usize>, ()> {
        if self.global_reads_started {
            return Err(());
        }
        let mut selected = Vec::with_capacity(indices.len());
        for &index in indices {
            if index >= self.draws.len() || self.scope_claimed[index] || selected.contains(&index) {
                return Err(());
            }
            selected.push(index);
        }
        for &index in &selected {
            self.scope_claimed[index] = true;
        }
        self.scopes_started += 1;
        self.scopes_pending_finish += 1;
        Ok(selected)
    }
    pub(super) fn read_vertex_streams_for(
        &mut self,
        pass: &mut crate::graph::definition::PassBuilder<'_>,
        indices: &[usize],
    ) -> std::result::Result<(), ()> {
        Self::ensure_uncovered(indices, &self.vertex_covered)?;
        for &draw_index in indices {
            let draw = self.draws.get(draw_index).ok_or(())?;
            validate_graph_draw(draw, self.streams.len()).map_err(|_| ())?;
            for (index, range) in draw.vertex_streams.iter().zip(&draw.accesses) {
                if let Some(range) = range {
                    pass.read_with_use(
                        *self.streams.get(*index).ok_or(())?,
                        range.clone(),
                        vertex_read_uses().uses,
                        BufferUsage::VERTEX,
                    )
                    .map_err(|_| ())?;
                }
            }
        }
        Self::mark_covered(indices, &mut self.vertex_covered);
        Ok(())
    }
    pub(super) fn read_frame_uniforms_for(
        &mut self,
        pass: &mut crate::graph::definition::PassBuilder<'_>,
        indices: &[usize],
    ) -> std::result::Result<(), ()> {
        Self::ensure_uncovered(indices, &self.uniform_covered)?;
        for &draw_index in indices {
            let draw = self.draws.get(draw_index).ok_or(())?;
            for range in &draw.uniform_ranges {
                pass.read_with_use(
                    self.uniform,
                    range.clone(),
                    shader_read_uses().uses,
                    BufferUsage::UNIFORM,
                )
                .map_err(|_| ())?;
            }
        }
        Self::mark_covered(indices, &mut self.uniform_covered);
        Ok(())
    }
    pub(super) fn read_sampled_textures_for(
        &mut self,
        pass: &mut crate::graph::definition::PassBuilder<'_>,
        indices: &[usize],
    ) -> std::result::Result<(), ()> {
        Self::ensure_uncovered(indices, &self.texture_covered)?;
        for &draw_index in indices {
            let draw = self.draws.get(draw_index).ok_or(())?;
            pass.read_texture_with_use(
                *self.textures.get(draw.texture).ok_or(())?,
                whole(),
                texture_read_uses().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )
            .map_err(|_| ())?;
        }
        Self::mark_covered(indices, &mut self.texture_covered);
        Ok(())
    }
    pub(super) fn complete(&self) -> bool {
        self.scopes_pending_finish == 0
            && self.vertex_covered.iter().all(|x| *x)
            && self.uniform_covered.iter().all(|x| *x)
            && self.texture_covered.iter().all(|x| *x)
    }
    pub(super) fn scopes_started(&self) -> bool {
        self.scopes_started != 0
    }
    pub(super) fn finish_scope(&mut self) -> std::result::Result<(), ()> {
        if self.scopes_pending_finish == 0 {
            return Err(());
        }
        self.scopes_pending_finish -= 1;
        Ok(())
    }
    fn ensure_uncovered(indices: &[usize], covered: &[bool]) -> std::result::Result<(), ()> {
        if indices
            .iter()
            .any(|&index| index >= covered.len() || covered[index])
        {
            Err(())
        } else {
            Ok(())
        }
    }
    fn mark_covered(indices: &[usize], covered: &mut [bool]) {
        for &index in indices {
            covered[index] = true;
        }
    }
}

impl<'pass, 'builder, 'frame, B: ResourceBackend> OpaqueForwardPass<'pass, 'builder, 'frame, B> {
    pub(crate) fn read_scope<'scope>(
        &'scope mut self,
        indices: &[usize],
    ) -> std::result::Result<ForwardReadScope<'scope, 'builder>, ()> {
        let selected = self.reads.select(indices)?;
        Ok(ForwardReadScope {
            pass: &mut *self.pass,
            state: &mut self.reads,
            indices: selected,
            declared: 0,
        })
    }
}

impl ForwardReadScope<'_, '_> {
    fn mark(&mut self, bit: u8) -> std::result::Result<(), ()> {
        if self.declared & bit != 0 {
            Err(())
        } else {
            self.declared |= bit;
            Ok(())
        }
    }
    pub(crate) fn read_vertex_streams(&mut self) -> std::result::Result<(), ()> {
        self.mark(VERTEX_READS)?;
        self.state.read_vertex_streams_for(self.pass, &self.indices)
    }
    pub(crate) fn read_frame_uniforms(&mut self) -> std::result::Result<(), ()> {
        self.mark(UNIFORM_READS)?;
        self.state.read_frame_uniforms_for(self.pass, &self.indices)
    }
    pub(crate) fn read_sampled_textures(&mut self) -> std::result::Result<(), ()> {
        self.mark(TEXTURE_READS)?;
        self.state
            .read_sampled_textures_for(self.pass, &self.indices)
    }
    pub(crate) fn finish(self) -> std::result::Result<(), ()> {
        if self.declared == VERTEX_READS | UNIFORM_READS | TEXTURE_READS {
            self.state.finish_scope()
        } else {
            Err(())
        }
    }
}

#[cfg(test)]
#[path = "../../tests/graph/forward_scopes.rs"]
mod tests;
