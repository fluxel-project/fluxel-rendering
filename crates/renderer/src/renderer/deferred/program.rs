//! Private shader, layout, and raster-pipeline realization for Deferred.

use std::sync::Arc;

use fluxel_rhi::api::*;
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderCompositionRequest,
    ShaderStage as LogicalShaderStage, SourceUnit,
};

use crate::{
    PreparedMaterial,
    renderer::admission::{self, PipelineCapabilityFacts},
};

use super::templates::LIGHTING_TEMPLATE;
use super::*;

#[path = "program_cache.rs"]
mod program_cache;

pub(crate) use program_cache::DeferredProgramCache;

pub(crate) struct GbufferProgram<B: ResourceBackend> {
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) material_layout: BindGroupLayout<B>,
    pub(super) transform_layout: BindGroupLayout<B>,
}

pub(crate) struct LightingProgram<B: ResourceBackend> {
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) layout: BindGroupLayout<B>,
}

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    pub(super) async fn deferred_gbuffer_program(
        &mut self,
        prepared: &PreparedMaterial,
        pbr: bool,
    ) -> std::result::Result<Arc<GbufferProgram<B>>, RendererError> {
        self.ensure_open()?;
        let transform_uniform_bytes = if pbr { 128 } else { 64 };
        let key = program_cache::GbufferProgramKey::for_material(
            self.device.id(),
            prepared,
            transform_uniform_bytes,
            pbr,
        );
        if let Some(program) = self.deferred_programs.get_gbuffer(&key) {
            self.deferred_stats.gbuffer_program_hits += 1;
            return Ok(program);
        }
        let artifact = prepared.shader();
        let vertex = artifact
            .entry_point(LogicalShaderStage::Vertex)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let fragment = artifact
            .entry_point(LogicalShaderStage::Fragment)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let descriptor = ShaderDesc {
            label: Some("Fluxel Deferred G-buffer material program".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(artifact.source().as_bytes()),
            entries: vec![
                ShaderEntryPoint {
                    name: vertex.clone(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: fragment.clone(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: artifact.requirements().names.iter().cloned().collect(),
        };
        self.admit_deferred_shader(
            &descriptor,
            artifact
                .requirements()
                .naga_capabilities
                .iter()
                .map(|capability| capability.name()),
        )?;
        let shader = self
            .device
            .shader(descriptor)
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        let material_layout =
            self.deferred_material_layout("Deferred G-buffer material bindings")?;
        let transform_layout = self.deferred_transform_layout(
            "Deferred G-buffer transform bindings",
            transform_uniform_bytes,
        )?;
        let layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Deferred G-buffer pipeline layout"),
                groups: &[&material_layout, &transform_layout],
                push_constants: &[],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline = self
            .device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("Deferred G-buffer pipeline".into()),
                layout,
                vertex: PipelineStage {
                    module: shader.clone(),
                    entry: vertex,
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: shader,
                    entry: fragment,
                    constants: vec![],
                }),
                vertex_buffers: prepared.geometry().streams.clone(),
                color_targets: vec![Some(color_target()), Some(color_target())],
                primitive: opaque_triangles(),
                depth_stencil: Some(depth_state()),
                multisample: single_sample(),
                multiview: None,
                cache: None,
            })
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        // Do not publish a partially realized program: every shader, layout,
        // and pipeline operation above must have completed successfully first.
        let program = Arc::new(GbufferProgram {
            pipeline,
            material_layout,
            transform_layout,
        });
        self.deferred_programs
            .publish_gbuffer(key, Arc::clone(&program));
        Ok(program)
    }

    pub(super) async fn deferred_lighting_program(
        &mut self,
        pbr: bool,
        color_format: TextureFormat,
    ) -> std::result::Result<Arc<LightingProgram<B>>, RendererError> {
        self.ensure_open()?;
        let request = if pbr {
            crate::pbr::lighting_request()
        } else {
            ShaderCompositionRequest::new(vec![SourceUnit::new(
                "deferred-lighting",
                LIGHTING_TEMPLATE,
            )])
            .with_entry_point(EntryPoint::new(
                "deferred_lighting_vs",
                LogicalShaderStage::Vertex,
            ))
            .with_entry_point(EntryPoint::new(
                "deferred_lighting_fs",
                LogicalShaderStage::Fragment,
            ))
            .with_binding(
                BindingDeclaration::new(0, 0, BindingKind::Texture)
                    .with_visibility([LogicalShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(0, 1, BindingKind::Texture)
                    .with_visibility([LogicalShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(0, 2, BindingKind::Texture)
                    .with_visibility([LogicalShaderStage::Fragment]),
            )
        };
        let artifact = fluxel_shader::ShaderComposer::new()
            .compose(&request)
            .map_err(|_| Error::InvalidState)?;
        let key = program_cache::LightingProgramKey::for_artifact(
            self.device.id(),
            artifact.cache_key().clone(),
            pbr,
            color_format,
        );
        if let Some(program) = self.deferred_programs.get_lighting(&key) {
            self.deferred_stats.lighting_program_hits += 1;
            return Ok(program);
        }
        let vertex = artifact
            .entry_point(LogicalShaderStage::Vertex)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let fragment = artifact
            .entry_point(LogicalShaderStage::Fragment)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let descriptor = ShaderDesc {
            label: Some("Fluxel Deferred lighting program".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(artifact.source().as_bytes()),
            entries: vec![
                ShaderEntryPoint {
                    name: vertex.clone(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: fragment.clone(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: artifact.requirements().names.iter().cloned().collect(),
        };
        self.admit_deferred_shader(
            &descriptor,
            artifact
                .requirements()
                .naga_capabilities
                .iter()
                .map(|capability| capability.name()),
        )?;
        let shader = self
            .device
            .shader(descriptor)
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        let layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some("Deferred lighting bindings".into()),
                entries: vec![
                    sampled_texture(0, TextureSampleType::Float),
                    sampled_texture(1, TextureSampleType::Float),
                    sampled_texture(2, TextureSampleType::Depth),
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Deferred lighting pipeline layout"),
                groups: &[&layout],
                push_constants: &[],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline = self
            .device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("Deferred lighting pipeline".into()),
                layout: pipeline_layout,
                vertex: PipelineStage {
                    module: shader.clone(),
                    entry: vertex,
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: shader,
                    entry: fragment,
                    constants: vec![],
                }),
                vertex_buffers: vec![],
                color_targets: vec![Some(color_target_with_format(color_format))],
                primitive: opaque_triangles(),
                depth_stencil: None,
                multisample: single_sample(),
                multiview: None,
                cache: None,
            })
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        // As with G-buffer programs, compilation failures never replace a
        // usable cache entry. Submitted frames retain their own Arc owner.
        let program = Arc::new(LightingProgram { pipeline, layout });
        self.deferred_programs
            .publish_lighting(key, Arc::clone(&program));
        Ok(program)
    }

    fn admit_deferred_shader<'a>(
        &self,
        descriptor: &ShaderDesc,
        demands: impl IntoIterator<Item = &'a str>,
    ) -> std::result::Result<(), RendererError> {
        admission::admit(
            &demands.into_iter().map(str::to_owned).collect::<Vec<_>>(),
            self.device.capabilities(),
            PipelineCapabilityFacts {
                multiview_views: None,
                push_constant_bytes: 0,
                multisample_count: 1,
                nonuniform: Default::default(),
            },
        )
        .map_err(RendererError::ShaderAdmission)?;
        if matches!(
            self.device.shader_route(descriptor),
            ShaderRoute::Unsupported
        ) {
            return Err(Error::Unsupported("Deferred shader route").into());
        }
        Ok(())
    }

    fn deferred_material_layout(
        &mut self,
        label: &str,
    ) -> std::result::Result<BindGroupLayout<B>, RendererError> {
        self.device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some(label.into()),
                entries: vec![
                    uniform(0, 16, ShaderStage::Fragment),
                    sampled_texture(1, TextureSampleType::Float),
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))
    }

    fn deferred_transform_layout(
        &mut self,
        label: &str,
        bytes: u64,
    ) -> std::result::Result<BindGroupLayout<B>, RendererError> {
        self.device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some(label.into()),
                entries: vec![uniform(0, bytes, ShaderStage::Vertex)],
            })
            .map_err(|error| self.observe_rhi_error(error))
    }
}

fn uniform(binding: u32, min_size: u64, visibility: ShaderStage) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![visibility],
        ty: BindingType::UniformBuffer {
            dynamic_offset: true,
            min_size,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
fn sampled_texture(binding: u32, sample: TextureSampleType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::SampledTexture {
            sample,
            dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
fn color_target() -> ColorTargetState {
    color_target_with_format(TextureFormat::Rgba8Unorm)
}

fn color_target_with_format(format: TextureFormat) -> ColorTargetState {
    ColorTargetState {
        format,
        blend: None,
        write_mask: 0xf,
    }
}
fn opaque_triangles() -> PrimitiveState {
    PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull_mode: CullMode::None,
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
    }
}
fn single_sample() -> MultisampleState {
    MultisampleState {
        count: 1,
        mask: u64::from(u32::MAX),
        alpha_to_coverage_enabled: false,
    }
}
fn depth_state() -> DepthStencilState {
    let stencil = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil_front: stencil,
        stencil_back: stencil,
        stencil_read_mask: 0,
        stencil_write_mask: 0,
        depth_bias_constant: 0,
        depth_bias_slope_scale: 0.0,
        depth_bias_clamp: 0.0,
    }
}
