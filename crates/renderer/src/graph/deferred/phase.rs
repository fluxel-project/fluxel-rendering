//! Transaction-scoped Deferred phase capabilities.

use super::*;

const UPLOAD: u8 = 1;
const VERTEX: u8 = 2;
const UNIFORM: u8 = 4;
const MATERIAL: u8 = 8;
const GBUFFER: u8 = 16;
const GBUFFER_READ: u8 = 1;
const COLOR: u8 = 2;
const DEFORMATION_SOURCE: u8 = 1;
const DEFORMATION_PARAMETERS: u8 = 2;
const DEFORMATION_PRODUCT: u8 = 4;

type DeferredPhaseDeclaration = (
    Option<GraphPass>,
    GraphPass,
    [super::super::definition::TextureHandle; 3],
    GraphPass,
    Option<super::super::definition::TextureHandle>,
);

pub(super) struct DeferredDeformationGraph {
    pub(super) source: super::super::definition::BufferHandle,
    pub(super) parameters: super::super::definition::BufferHandle,
    pub(super) product: super::super::definition::BufferHandle,
    pub(super) source_range: std::ops::Range<u64>,
    pub(super) parameter_range: std::ops::Range<u64>,
    pub(super) product_range: std::ops::Range<u64>,
    pub(super) product_stream: usize,
}

/// Private owner of the two real Deferred Graph transactions.
pub(crate) struct DeferredPhaseDriver<'graph> {
    pub(super) graph: &'graph mut GraphBuilder,
    pub(super) upload: GraphPass,
    pub(super) streams: Vec<super::super::definition::BufferHandle>,
    pub(super) stream_reads: Vec<Option<std::ops::Range<u64>>>,
    pub(super) uniform: super::super::definition::BufferHandle,
    pub(super) tint: std::ops::Range<u64>,
    pub(super) transform: std::ops::Range<u64>,
    pub(super) material: super::super::definition::TextureHandle,
    pub(super) gbuffer: [super::super::definition::TextureHandle; 3],
    pub(super) target: Option<super::super::definition::TextureHandle>,
    pub(super) deformation: Option<DeferredDeformationGraph>,
    pub(super) deformation_pass: Option<(GraphPass, super::super::definition::BufferHandle)>,
    pub(super) geometry: Option<(GraphPass, [super::super::definition::TextureHandle; 3])>,
    pub(super) lighting: Option<(GraphPass, Option<super::super::definition::TextureHandle>)>,
}

pub(crate) struct DeferredDeformationPass<'pass, 'builder> {
    pass: &'pass mut super::super::definition::PassBuilder<'builder>,
    source: super::super::definition::BufferHandle,
    parameters: super::super::definition::BufferHandle,
    product: super::super::definition::BufferHandle,
    source_range: std::ops::Range<u64>,
    parameter_range: std::ops::Range<u64>,
    product_range: std::ops::Range<u64>,
    declared: u8,
}

pub(crate) struct DeferredGeometryPass<'pass, 'builder> {
    pass: &'pass mut super::super::definition::PassBuilder<'builder>,
    upload: GraphPass,
    streams: &'pass [super::super::definition::BufferHandle],
    stream_reads: &'pass [Option<std::ops::Range<u64>>],
    uniform: super::super::definition::BufferHandle,
    tint: std::ops::Range<u64>,
    transform: std::ops::Range<u64>,
    material: super::super::definition::TextureHandle,
    gbuffer: [super::super::definition::TextureHandle; 3],
    declared: u8,
}

pub(crate) struct DeferredLightingPass<'pass, 'builder> {
    pass: &'pass mut super::super::definition::PassBuilder<'builder>,
    gbuffer: [super::super::definition::TextureHandle; 3],
    target: Option<super::super::definition::TextureHandle>,
    declared: u8,
}

impl DeferredPhaseDriver<'_> {
    pub(crate) fn deformation<F>(&mut self, declare: F) -> std::result::Result<(), ()>
    where
        F: for<'pass, 'builder> FnOnce(
            &mut DeferredDeformationPass<'pass, 'builder>,
        ) -> std::result::Result<(), ()>,
    {
        if self.deformation_pass.is_some() || self.geometry.is_some() || self.lighting.is_some() {
            return Err(());
        }
        let Some(deformation) = self.deformation.as_ref() else {
            return Err(());
        };
        let upload = self.upload;
        let source = deformation.source;
        let parameters = deformation.parameters;
        let product = deformation.product;
        let source_range = deformation.source_range.clone();
        let parameter_range = deformation.parameter_range.clone();
        let product_range = deformation.product_range.clone();
        let (pass, output) = self
            .graph
            .add_pass("deferred-deformation", |pass| {
                pass.depends_on(upload)?;
                let mut deformation = DeferredDeformationPass {
                    pass,
                    source,
                    parameters,
                    product,
                    source_range,
                    parameter_range,
                    product_range,
                    declared: 0,
                };
                declare(&mut deformation).map_err(|_| GraphError::InvalidRange)?;
                deformation.finish().map_err(|_| GraphError::InvalidRange)
            })
            .map_err(|_| ())?;
        self.deformation_pass = Some((pass, output));
        Ok(())
    }

    pub(crate) fn geometry<F>(&mut self, declare: F) -> std::result::Result<(), ()>
    where
        F: for<'pass, 'builder> FnOnce(
            &mut DeferredGeometryPass<'pass, 'builder>,
        ) -> std::result::Result<(), ()>,
    {
        if self.geometry.is_some() || self.lighting.is_some() {
            return Err(());
        }
        let upload = self.upload;
        let mut streams = self.streams.clone();
        let stream_reads = self.stream_reads.clone();
        if let Some(deformation) = self.deformation.as_ref() {
            let Some((_, product)) = self.deformation_pass else {
                return Err(());
            };
            let stream = streams.get_mut(deformation.product_stream).ok_or(())?;
            *stream = product;
        }
        let uniform = self.uniform;
        let tint = self.tint.clone();
        let transform = self.transform.clone();
        let material = self.material;
        let gbuffer = self.gbuffer;
        let (pass, output) = self
            .graph
            .add_pass("deferred-geometry", |pass| {
                let mut geometry = DeferredGeometryPass {
                    pass,
                    upload,
                    streams: &streams,
                    stream_reads: &stream_reads,
                    uniform,
                    tint,
                    transform,
                    material,
                    gbuffer,
                    declared: 0,
                };
                declare(&mut geometry).map_err(|_| GraphError::InvalidRange)?;
                geometry.finish().map_err(|_| GraphError::InvalidRange)
            })
            .map_err(|_| ())?;
        self.geometry = Some((pass, output));
        Ok(())
    }

    pub(crate) fn lighting<F>(&mut self, declare: F) -> std::result::Result<(), ()>
    where
        F: for<'pass, 'builder> FnOnce(
            &mut DeferredLightingPass<'pass, 'builder>,
        ) -> std::result::Result<(), ()>,
    {
        let Some((geometry, gbuffer)) = self.geometry else {
            return Err(());
        };
        if self.lighting.is_some() {
            return Err(());
        }
        let target = self.target;
        let (pass, output) = self
            .graph
            .add_pass("deferred-lighting", |pass| {
                pass.depends_on(geometry)?;
                let mut lighting = DeferredLightingPass {
                    pass,
                    gbuffer,
                    target,
                    declared: 0,
                };
                declare(&mut lighting).map_err(|_| GraphError::InvalidRange)?;
                lighting.finish().map_err(|_| GraphError::InvalidRange)
            })
            .map_err(|_| ())?;
        self.lighting = Some((pass, output));
        Ok(())
    }

    pub(crate) fn finish(self) -> std::result::Result<DeferredPhaseDeclaration, ()> {
        let deformation = self.deformation_pass.map(|(pass, _)| pass);
        if self.deformation.is_some() != deformation.is_some() {
            return Err(());
        }
        let (geometry, gbuffer) = self.geometry.ok_or(())?;
        let (lighting, target) = self.lighting.ok_or(())?;
        Ok((deformation, geometry, gbuffer, lighting, target))
    }
}

impl DeferredDeformationPass<'_, '_> {
    fn mark(&mut self, bit: u8) -> std::result::Result<(), ()> {
        if self.declared & bit == 0 {
            self.declared |= bit;
            Ok(())
        } else {
            Err(())
        }
    }

    pub(crate) fn read_source(&mut self) -> std::result::Result<(), ()> {
        self.mark(DEFORMATION_SOURCE)?;
        self.pass
            .read_with_use(
                self.source,
                self.source_range.clone(),
                compute_read().uses,
                BufferUsage::STORAGE,
            )
            .map_err(|_| ())
    }

    pub(crate) fn read_parameters(&mut self) -> std::result::Result<(), ()> {
        self.mark(DEFORMATION_PARAMETERS)?;
        self.pass
            .read_with_use(
                self.parameters,
                self.parameter_range.clone(),
                compute_parameter_read().uses,
                BufferUsage::UNIFORM,
            )
            .map_err(|_| ())
    }

    pub(crate) fn write_vertices(&mut self) -> std::result::Result<(), ()> {
        self.mark(DEFORMATION_PRODUCT)?;
        self.product = self
            .pass
            .write_with_use(
                self.product,
                self.product_range.clone(),
                compute_write().uses,
                BufferUsage::STORAGE,
            )
            .map_err(|_| ())?;
        Ok(())
    }

    fn finish(&mut self) -> std::result::Result<super::super::definition::BufferHandle, ()> {
        if self.declared != DEFORMATION_SOURCE | DEFORMATION_PARAMETERS | DEFORMATION_PRODUCT {
            return Err(());
        }
        Ok(self.product)
    }
}

impl DeferredGeometryPass<'_, '_> {
    fn mark(&mut self, bit: u8) -> std::result::Result<(), ()> {
        if self.declared & bit == 0 {
            self.declared |= bit;
            Ok(())
        } else {
            Err(())
        }
    }
    pub(crate) fn depends_on_uploads(&mut self) -> std::result::Result<(), ()> {
        self.mark(UPLOAD)?;
        self.pass.depends_on(self.upload).map_err(|_| ())
    }
    pub(crate) fn read_vertex_streams(&mut self) -> std::result::Result<(), ()> {
        self.mark(VERTEX)?;
        for (stream, range) in self.streams.iter().zip(self.stream_reads) {
            if let Some(range) = range {
                self.pass
                    .read_with_use(
                        *stream,
                        range.clone(),
                        vertex_read().uses,
                        BufferUsage::VERTEX,
                    )
                    .map_err(|_| ())?;
            }
        }
        Ok(())
    }
    pub(crate) fn read_frame_uniforms(&mut self) -> std::result::Result<(), ()> {
        self.mark(UNIFORM)?;
        self.pass
            .read_with_use(
                self.uniform,
                self.tint.clone(),
                uniform_read().uses,
                BufferUsage::UNIFORM,
            )
            .map_err(|_| ())?;
        self.pass
            .read_with_use(
                self.uniform,
                self.transform.clone(),
                uniform_read().uses,
                BufferUsage::UNIFORM,
            )
            .map_err(|_| ())
    }
    pub(crate) fn read_sampled_textures(&mut self) -> std::result::Result<(), ()> {
        self.mark(MATERIAL)?;
        self.pass
            .read_texture_with_use(
                self.material,
                whole_color(),
                sampled_read().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )
            .map_err(|_| ())
    }
    pub(crate) fn write_gbuffer(&mut self) -> std::result::Result<(), ()> {
        self.mark(GBUFFER)?;
        self.gbuffer[0] = self
            .pass
            .write_texture_with_use(
                self.gbuffer[0],
                whole_color(),
                color_write().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .map_err(|_| ())?;
        self.gbuffer[1] = self
            .pass
            .write_texture_with_use(
                self.gbuffer[1],
                whole_color(),
                color_write().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .map_err(|_| ())?;
        self.gbuffer[2] = self
            .pass
            .write_texture_with_use(
                self.gbuffer[2],
                whole_depth(),
                depth_write().uses,
                ImageLayout::DepthStencilAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .map_err(|_| ())?;
        Ok(())
    }
    fn finish(&mut self) -> std::result::Result<[super::super::definition::TextureHandle; 3], ()> {
        if self.declared != UPLOAD | VERTEX | UNIFORM | MATERIAL | GBUFFER {
            return Err(());
        }
        Ok(self.gbuffer)
    }
}

impl DeferredLightingPass<'_, '_> {
    fn mark(&mut self, bit: u8) -> std::result::Result<(), ()> {
        if self.declared & bit == 0 {
            self.declared |= bit;
            Ok(())
        } else {
            Err(())
        }
    }
    pub(crate) fn read_gbuffer(&mut self) -> std::result::Result<(), ()> {
        self.mark(GBUFFER_READ)?;
        for (index, texture) in self.gbuffer.iter().enumerate() {
            self.pass
                .read_texture_with_use(
                    *texture,
                    if index == 2 {
                        whole_depth()
                    } else {
                        whole_color()
                    },
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )
                .map_err(|_| ())?;
        }
        Ok(())
    }
    pub(crate) fn write_main_color(&mut self) -> std::result::Result<(), ()> {
        self.mark(COLOR)?;
        if let Some(target) = self.target {
            self.target = Some(
                self.pass
                    .write_texture_with_use(
                        target,
                        whole_color(),
                        color_write().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .map_err(|_| ())?,
            );
        }
        Ok(())
    }
    fn finish(
        &mut self,
    ) -> std::result::Result<Option<super::super::definition::TextureHandle>, ()> {
        if self.declared != GBUFFER_READ | COLOR {
            return Err(());
        }
        self.pass.side_effect();
        Ok(self.target)
    }
}
