//! Transaction-scoped Forward phase and access capabilities.
use super::*;

/// Private state behind the public, deliberately narrow phase context.
pub(crate) struct ForwardPhaseDriver<'graph, 'frame, B: ResourceBackend> {
    pub(super) graph: Option<&'graph mut GraphBuilder>,
    pub(super) marker: PhantomData<&'frame ForwardFrame<B>>,
    pub(super) expected: FrameTopology,
    pub(super) upload: Option<GraphPass>,
    pub(super) streams: Vec<crate::graph::definition::BufferHandle>,
    pub(super) uniform: Option<crate::graph::definition::BufferHandle>,
    pub(super) textures: Vec<crate::graph::definition::TextureHandle>,
    // A surface target is an RHI lease, not a persistent Graph resource.  The
    // phase still declares its write on the real Graph transaction as a side
    // effect, but has no texture handle to version.
    pub(super) target: Option<crate::graph::definition::TextureHandle>,
    pub(super) surface_target: bool,
    pub(super) draws: Vec<ForwardGraphDraw>,
    pub(super) declared: bool,
    pub(super) output: Option<(GraphPass, Option<crate::graph::definition::TextureHandle>)>,
}

pub(super) const UPLOAD_DEPENDENCY: u8 = 1;
pub(super) const VERTEX_READS: u8 = 2;
pub(super) const UNIFORM_READS: u8 = 4;
pub(super) const TEXTURE_READS: u8 = 8;
pub(super) const COLOR_WRITE: u8 = 16;

/// Graph-private opaque pass capability, exposed only through the public
/// policy wrapper for the duration of a real `GraphBuilder::add_pass` call.
pub(crate) struct OpaqueForwardPass<'pass, 'builder, 'frame, B: ResourceBackend> {
    pub(super) pass: &'pass mut crate::graph::definition::PassBuilder<'builder>,
    pub(super) upload: GraphPass,
    pub(super) target: Option<crate::graph::definition::TextureHandle>,
    pub(super) surface_target: bool,
    pub(super) declared: u8,
    pub(super) reads: ForwardReadState,
    pub(super) marker: PhantomData<&'frame ForwardFrame<B>>,
}

impl<B: ResourceBackend> OpaqueForwardPass<'_, '_, '_, B> {
    fn mark(&mut self, bit: u8) -> std::result::Result<(), ()> {
        if self.declared & bit == 0 {
            self.declared |= bit;
            Ok(())
        } else {
            Err(())
        }
    }

    pub(crate) fn depends_on_uploads(&mut self) -> std::result::Result<(), ()> {
        self.mark(UPLOAD_DEPENDENCY)?;
        self.pass.depends_on(self.upload).map_err(|_| ())
    }

    pub(crate) fn read_vertex_streams(&mut self) -> std::result::Result<(), ()> {
        self.mark(VERTEX_READS)?;
        self.reads.begin_global_reads()?;
        let indices: Vec<_> = (0..self.reads.draws.len()).collect();
        self.reads.read_vertex_streams_for(self.pass, &indices)
    }

    pub(crate) fn read_frame_uniforms(&mut self) -> std::result::Result<(), ()> {
        self.mark(UNIFORM_READS)?;
        self.reads.begin_global_reads()?;
        let indices: Vec<_> = (0..self.reads.draws.len()).collect();
        self.reads.read_frame_uniforms_for(self.pass, &indices)
    }

    pub(crate) fn read_sampled_textures(&mut self) -> std::result::Result<(), ()> {
        self.mark(TEXTURE_READS)?;
        self.reads.begin_global_reads()?;
        let indices: Vec<_> = (0..self.reads.draws.len()).collect();
        self.reads.read_sampled_textures_for(self.pass, &indices)
    }

    pub(crate) fn write_main_color(&mut self) -> std::result::Result<(), ()> {
        self.mark(COLOR_WRITE)?;
        if let Some(target) = self.target.take() {
            self.target = Some(
                self.pass
                    .write_texture_with_use(
                        target,
                        whole(),
                        color_write_uses().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .map_err(|_| ())?,
            );
        } else if !self.surface_target {
            return Err(());
        }
        Ok(())
    }

    fn finish(
        &mut self,
    ) -> std::result::Result<Option<crate::graph::definition::TextureHandle>, ()> {
        if !opaque_requirements_met(self.declared, &self.reads) {
            return Err(());
        }
        if self.target.is_none() && !self.surface_target {
            return Err(());
        }
        self.pass.side_effect();
        Ok(self.target)
    }
}

pub(super) fn opaque_requirements_met(declared: u8, reads: &ForwardReadState) -> bool {
    declared & (UPLOAD_DEPENDENCY | COLOR_WRITE) == UPLOAD_DEPENDENCY | COLOR_WRITE
        && reads.complete()
        && if reads.scopes_started() {
            declared & (VERTEX_READS | UNIFORM_READS | TEXTURE_READS) == 0
        } else {
            declared & (VERTEX_READS | UNIFORM_READS | TEXTURE_READS)
                == VERTEX_READS | UNIFORM_READS | TEXTURE_READS
        }
}

pub(crate) struct ClearForwardPass<'pass, 'builder, 'frame, B: ResourceBackend> {
    pass: &'pass mut crate::graph::definition::PassBuilder<'builder>,
    upload: GraphPass,
    target: Option<crate::graph::definition::TextureHandle>,
    surface_target: bool,
    declared: u8,
    marker: PhantomData<&'frame ForwardFrame<B>>,
}

impl<B: ResourceBackend> ClearForwardPass<'_, '_, '_, B> {
    pub(crate) fn depends_on_uploads(&mut self) -> std::result::Result<(), ()> {
        if self.declared & UPLOAD_DEPENDENCY != 0 {
            return Err(());
        }
        self.declared |= UPLOAD_DEPENDENCY;
        self.pass.depends_on(self.upload).map_err(|_| ())
    }
    pub(crate) fn write_main_color(&mut self) -> std::result::Result<(), ()> {
        if self.declared & COLOR_WRITE != 0 {
            return Err(());
        }
        self.declared |= COLOR_WRITE;
        if let Some(target) = self.target.take() {
            self.target = Some(
                self.pass
                    .write_texture_with_use(
                        target,
                        whole(),
                        color_write_uses().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .map_err(|_| ())?,
            );
        } else if !self.surface_target {
            return Err(());
        }
        Ok(())
    }
    fn finish(
        &mut self,
    ) -> std::result::Result<Option<crate::graph::definition::TextureHandle>, ()> {
        if self.declared != UPLOAD_DEPENDENCY | COLOR_WRITE {
            return Err(());
        }
        if self.target.is_none() && !self.surface_target {
            return Err(());
        }
        self.pass.side_effect();
        Ok(self.target)
    }
}

impl<'graph, 'frame, B: ResourceBackend> ForwardPhaseDriver<'graph, 'frame, B> {
    fn require(&self, requested: FrameTopology) -> std::result::Result<(), ()> {
        if self.expected == requested && !self.declared {
            Ok(())
        } else {
            Err(())
        }
    }

    pub(crate) fn declare_opaque<F>(&mut self, declare: F) -> std::result::Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut OpaqueForwardPass<'pass, 'builder, 'frame, B>,
        )
            -> std::result::Result<(), PipelineError>,
    {
        self.require(FrameTopology::OpaqueForward)
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        if self.graph.is_none() {
            // A phase declaration is only meaningful while it directly
            // records its required accesses into the real Graph transaction.
            // Do not claim success for a preflight that cannot execute the
            // user callback or validate its exact access set.
            return Err(PipelineError::InvalidPhaseContribution);
        }
        let streams = self.streams.clone();
        let uniform = self
            .uniform
            .ok_or(PipelineError::InvalidPhaseContribution)?;
        let textures = self.textures.clone();
        let target = self.target;
        if target.is_none() && !self.surface_target {
            return Err(PipelineError::InvalidPhaseContribution);
        }
        let upload = self.upload.ok_or(PipelineError::InvalidPhaseContribution)?;
        let draws = self.draws.clone();
        let (pass, target) = self
            .graph
            .as_deref_mut()
            .ok_or(PipelineError::InvalidPhaseContribution)?
            .add_pass("forward-opaque", |pass| {
                let mut opaque = OpaqueForwardPass {
                    pass,
                    upload,
                    target,
                    surface_target: self.surface_target,
                    declared: 0,
                    reads: ForwardReadState::new(streams, uniform, textures, draws),
                    marker: PhantomData,
                };
                declare(&mut opaque)
                    .map_err(|_| crate::graph::definition::GraphError::InvalidRange)?;
                opaque
                    .finish()
                    .map_err(|_| crate::graph::definition::GraphError::InvalidRange)
            })
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        self.declared = true;
        self.output = Some((pass, target));
        Ok(())
    }

    pub(crate) fn declare_clear<F>(&mut self, declare: F) -> std::result::Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut ClearForwardPass<'pass, 'builder, 'frame, B>,
        )
            -> std::result::Result<(), PipelineError>,
    {
        self.require(FrameTopology::ClearOnly)
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        if self.graph.is_none() {
            return Err(PipelineError::InvalidPhaseContribution);
        }
        let upload = self.upload.ok_or(PipelineError::InvalidPhaseContribution)?;
        let target = self.target;
        if target.is_none() && !self.surface_target {
            return Err(PipelineError::InvalidPhaseContribution);
        }
        let (pass, target) = self
            .graph
            .as_deref_mut()
            .ok_or(PipelineError::InvalidPhaseContribution)?
            .add_pass("forward-clear", |pass| {
                let mut clear = ClearForwardPass {
                    pass,
                    upload,
                    target,
                    surface_target: self.surface_target,
                    declared: 0,
                    marker: PhantomData,
                };
                declare(&mut clear)
                    .map_err(|_| crate::graph::definition::GraphError::InvalidRange)?;
                clear
                    .finish()
                    .map_err(|_| crate::graph::definition::GraphError::InvalidRange)
            })
            .map_err(|_| PipelineError::InvalidPhaseContribution)?;
        self.declared = true;
        self.output = Some((pass, target));
        Ok(())
    }

    pub(super) fn finish(
        self,
    ) -> std::result::Result<(GraphPass, crate::graph::definition::TextureHandle), ()> {
        let (pass, target) = self.output.ok_or(())?;
        Ok((pass, target.ok_or(())?))
    }

    pub(super) fn finish_surface(self) -> std::result::Result<GraphPass, ()> {
        self.output.map(|(pass, _)| pass).ok_or(())
    }
}
