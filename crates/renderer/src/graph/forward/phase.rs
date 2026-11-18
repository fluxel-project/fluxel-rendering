//! Transaction-scoped Forward phase and access capabilities.
use super::*;

/// Private state behind the public, deliberately narrow phase context.
pub(crate) struct ForwardPhaseDriver<'graph, 'frame, B: ResourceBackend> {
    pub(super) graph: Option<&'graph mut GraphBuilder>,
    pub(super) marker: PhantomData<&'frame ForwardFrame<B>>,
    pub(crate) expected: FrameTopology,
    pub(super) upload: Option<GraphPass>,
    pub(super) streams: Vec<crate::graph::definition::BufferHandle>,
    pub(super) uniform: Option<crate::graph::definition::BufferHandle>,
    pub(super) textures: Vec<crate::graph::definition::TextureHandle>,
    pub(super) environment: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_radiance: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_brdf_lut: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_view_uniform: Option<crate::graph::definition::BufferHandle>,
    pub(super) specular_mip_levels: Option<u32>,
    /// Either a persistent target or an RHI-managed surface lease. Both are
    /// real Graph resources and must receive the terminal color write.
    pub(super) target: crate::graph::definition::TextureHandle,
    /// The single-sample target written by the native raster resolve, if the
    /// raster attachment above is multisampled.
    pub(super) resolve_target: Option<crate::graph::definition::TextureHandle>,
    pub(super) depth: Option<crate::graph::definition::TextureHandle>,
    pub(super) draws: Vec<ForwardGraphDraw>,
    pub(super) declared: bool,
    pub(super) output: Option<(
        GraphPass,
        crate::graph::definition::TextureHandle,
        Option<crate::graph::definition::TextureHandle>,
    )>,
}

pub(super) const UPLOAD_DEPENDENCY: u8 = 1;
pub(super) const VERTEX_READS: u8 = 2;
pub(super) const UNIFORM_READS: u8 = 4;
pub(super) const TEXTURE_READS: u8 = 8;
pub(super) const COLOR_WRITE: u8 = 16;
pub(super) const DEPTH_WRITE: u8 = 32;
pub(super) const DEPTH_READ: u8 = 64;

/// Graph-private opaque pass capability, exposed only through the public
/// policy wrapper for the duration of a real `GraphBuilder::add_pass` call.
pub(crate) struct OpaqueForwardPass<'pass, 'builder, 'frame, B: ResourceBackend> {
    pub(super) pass: &'pass mut crate::graph::definition::PassBuilder<'builder>,
    pub(super) upload: GraphPass,
    pub(super) target: crate::graph::definition::TextureHandle,
    pub(super) resolve_target: Option<crate::graph::definition::TextureHandle>,
    pub(super) depth: Option<crate::graph::definition::TextureHandle>,
    pub(super) declared: u8,
    pub(super) reads: ForwardReadState,
    pub(super) environment: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_radiance: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_brdf_lut: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_view_uniform: Option<crate::graph::definition::BufferHandle>,
    pub(super) specular_mip_levels: Option<u32>,
    /// The sorted product needs read/write attachment footprints for blending.
    pub(super) sorted: bool,
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
        let indices: Vec<_> = (0..self.reads.draw_count()).collect();
        self.reads.read_vertex_streams_for(self.pass, &indices)
    }

    pub(crate) fn read_frame_uniforms(&mut self) -> std::result::Result<(), ()> {
        self.mark(UNIFORM_READS)?;
        self.reads.begin_global_reads()?;
        let indices: Vec<_> = (0..self.reads.draw_count()).collect();
        self.reads.read_frame_uniforms_for(self.pass, &indices)
    }

    pub(crate) fn read_sampled_textures(&mut self) -> std::result::Result<(), ()> {
        self.mark(TEXTURE_READS)?;
        self.reads.begin_global_reads()?;
        let indices: Vec<_> = (0..self.reads.draw_count()).collect();
        self.reads.read_sampled_textures_for(self.pass, &indices)
    }

    pub(crate) fn write_main_color(&mut self) -> std::result::Result<(), ()> {
        self.mark(COLOR_WRITE)?;
        self.target = self
            .pass
            .write_texture_with_use(
                self.target,
                whole(),
                if self.sorted {
                    color_read_write_uses().uses
                } else {
                    color_write_uses().uses
                },
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .map_err(|_| ())?;
        if let Some(resolve_target) = self.resolve_target {
            self.resolve_target = Some(
                self.pass
                    .write_texture_with_use(
                        resolve_target,
                        whole(),
                        color_write_uses().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .map_err(|_| ())?,
            );
        }
        Ok(())
    }

    /// Declares the optional owner-selected depth attachment. With depth
    /// disabled this is intentionally a no-op; with it enabled the exact
    /// depth-aspect write is mandatory before the pass can finish.
    pub(crate) fn write_selected_depth(&mut self) -> std::result::Result<(), ()> {
        let Some(depth) = self.depth else {
            return Ok(());
        };
        self.mark(DEPTH_WRITE)?;
        self.depth = Some(
            self.pass
                .write_texture_with_use(
                    depth,
                    whole_depth(),
                    if self.sorted {
                        depth_read_write_uses().uses
                    } else {
                        depth_write_uses().uses
                    },
                    ImageLayout::DepthStencilAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )
                .map_err(|_| ())?,
        );
        Ok(())
    }

    /// Keeps the selected depth attachment as a read-only attachment for a
    /// later composition pass. The source depth must therefore have been
    /// retained by its producer rather than cleared or re-created.
    pub(crate) fn read_selected_depth(&mut self) -> std::result::Result<(), ()> {
        let Some(depth) = self.depth else {
            return Err(());
        };
        self.mark(DEPTH_READ)?;
        self.pass
            .read_texture_with_use(
                depth,
                whole_depth(),
                depth_read_uses().uses,
                ImageLayout::DepthStencilAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .map_err(|_| ())?;
        Ok(())
    }

    fn finish(
        &mut self,
    ) -> std::result::Result<
        (
            crate::graph::definition::TextureHandle,
            Option<crate::graph::definition::TextureHandle>,
        ),
        (),
    > {
        if !opaque_requirements_met(self.declared, &self.reads, self.depth.is_some()) {
            return Err(());
        }
        if let Some(environment) = self.environment {
            self.pass
                .read_texture_with_use(
                    environment,
                    whole(),
                    texture_read_uses().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )
                .map_err(|_| ())?;
        }
        if let (Some(radiance), Some(brdf_lut), Some(view_uniform)) = (
            self.specular_radiance,
            self.specular_brdf_lut,
            self.specular_view_uniform,
        ) {
            self.pass
                .read_texture_with_use(
                    radiance,
                    all_color_mips(self.specular_mip_levels.ok_or(())?),
                    texture_read_uses().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )
                .map_err(|_| ())?;
            self.pass
                .read_texture_with_use(
                    brdf_lut,
                    whole(),
                    texture_read_uses().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )
                .map_err(|_| ())?;
            self.pass
                .read_with_use(
                    view_uniform,
                    0..112,
                    shader_read_uses().uses,
                    BufferUsage::UNIFORM,
                )
                .map_err(|_| ())?;
        } else if self.specular_radiance.is_some()
            || self.specular_brdf_lut.is_some()
            || self.specular_view_uniform.is_some()
        {
            return Err(());
        }
        self.pass.side_effect();
        Ok((self.resolve_target.unwrap_or(self.target), self.depth))
    }
}

pub(super) fn opaque_requirements_met(
    declared: u8,
    reads: &ForwardReadState,
    depth_enabled: bool,
) -> bool {
    declared & (UPLOAD_DEPENDENCY | COLOR_WRITE) == UPLOAD_DEPENDENCY | COLOR_WRITE
        && reads.complete()
        && if reads.scopes_started() {
            declared & (VERTEX_READS | UNIFORM_READS | TEXTURE_READS) == 0
        } else {
            declared & (VERTEX_READS | UNIFORM_READS | TEXTURE_READS)
                == VERTEX_READS | UNIFORM_READS | TEXTURE_READS
        }
        && if depth_enabled {
            declared & (DEPTH_READ | DEPTH_WRITE) == DEPTH_READ
                || declared & (DEPTH_READ | DEPTH_WRITE) == DEPTH_WRITE
        } else {
            declared & (DEPTH_READ | DEPTH_WRITE) == 0
        }
}

pub(crate) struct ClearForwardPass<'pass, 'builder, 'frame, B: ResourceBackend> {
    pass: &'pass mut crate::graph::definition::PassBuilder<'builder>,
    upload: GraphPass,
    target: crate::graph::definition::TextureHandle,
    resolve_target: Option<crate::graph::definition::TextureHandle>,
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
        self.target = self
            .pass
            .write_texture_with_use(
                self.target,
                whole(),
                color_write_uses().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .map_err(|_| ())?;
        if let Some(resolve_target) = self.resolve_target {
            self.resolve_target = Some(
                self.pass
                    .write_texture_with_use(
                        resolve_target,
                        whole(),
                        color_write_uses().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .map_err(|_| ())?,
            );
        }
        Ok(())
    }
    fn finish(&mut self) -> std::result::Result<crate::graph::definition::TextureHandle, ()> {
        if self.declared != UPLOAD_DEPENDENCY | COLOR_WRITE {
            return Err(());
        }
        self.pass.side_effect();
        Ok(self.resolve_target.unwrap_or(self.target))
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
        self.declare_draw_phase(
            FrameTopology::OpaqueForward,
            "forward-opaque",
            false,
            declare,
        )
    }

    /// Declares the single actual Forward pass that contains opaque/masked
    /// draws followed by the renderer's globally ordered blend draws.
    pub(crate) fn declare_sorted<F>(&mut self, declare: F) -> std::result::Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut OpaqueForwardPass<'pass, 'builder, 'frame, B>,
        )
            -> std::result::Result<(), PipelineError>,
    {
        self.declare_draw_phase(
            FrameTopology::SortedForward,
            "forward-sorted",
            true,
            declare,
        )
    }

    fn declare_draw_phase<F>(
        &mut self,
        expected: FrameTopology,
        label: &'static str,
        sorted: bool,
        declare: F,
    ) -> std::result::Result<(), PipelineError>
    where
        F: for<'callback, 'pass, 'builder> FnOnce(
            &'callback mut OpaqueForwardPass<'pass, 'builder, 'frame, B>,
        )
            -> std::result::Result<(), PipelineError>,
    {
        self.require(expected)
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
        let environment = self.environment;
        let specular_radiance = self.specular_radiance;
        let specular_brdf_lut = self.specular_brdf_lut;
        let specular_view_uniform = self.specular_view_uniform;
        let specular_mip_levels = self.specular_mip_levels;
        let target = self.target;
        let resolve_target = self.resolve_target;
        let depth = self.depth;
        let upload = self.upload.ok_or(PipelineError::InvalidPhaseContribution)?;
        let draws = self.draws.clone();
        let (pass, (target, depth)) = self
            .graph
            .as_deref_mut()
            .ok_or(PipelineError::InvalidPhaseContribution)?
            .add_pass(label, |pass| {
                let mut opaque = OpaqueForwardPass {
                    pass,
                    upload,
                    target,
                    resolve_target,
                    depth,
                    declared: 0,
                    reads: ForwardReadState::new(streams, uniform, textures, draws)
                        .map_err(|_| crate::graph::definition::GraphError::InvalidRange)?,
                    environment,
                    specular_radiance,
                    specular_brdf_lut,
                    specular_view_uniform,
                    specular_mip_levels,
                    sorted,
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
        self.output = Some((pass, target, depth));
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
        let resolve_target = self.resolve_target;
        let (pass, target) = self
            .graph
            .as_deref_mut()
            .ok_or(PipelineError::InvalidPhaseContribution)?
            .add_pass("forward-clear", |pass| {
                let mut clear = ClearForwardPass {
                    pass,
                    upload,
                    target,
                    resolve_target,
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
        self.output = Some((pass, target, None));
        Ok(())
    }

    pub(super) fn finish(
        self,
    ) -> std::result::Result<
        (
            GraphPass,
            crate::graph::definition::TextureHandle,
            Option<crate::graph::definition::TextureHandle>,
        ),
        (),
    > {
        self.output.ok_or(())
    }
}
