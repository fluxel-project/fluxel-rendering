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

use super::templates::{LIGHTING_TEMPLATE, SCENE_COLOR_COMPOSITE_TEMPLATE};
use super::*;

#[path = "program_cache.rs"]
mod program_cache;
#[path = "program_fixed.rs"]
mod program_fixed;
#[path = "program_layout.rs"]
mod program_layout;
use program_fixed::{
    color_target, color_target_with_format, depth_state, opaque_triangles, single_sample,
    surface_triangles,
};

use program_layout::{ordinary_sampler, sampled_texture, uniform};

pub(crate) use program_cache::DeferredProgramCache;

pub(crate) struct GbufferProgram<B: ResourceBackend> {
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) material_layout: BindGroupLayout<B>,
    pub(super) transform_layout: BindGroupLayout<B>,
}

pub(crate) struct LightingProgram<B: ResourceBackend> {
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) layout: BindGroupLayout<B>,
    pub(super) environment_layout: Option<BindGroupLayout<B>>,
}

pub(crate) struct SceneColorCompositeProgram<B: ResourceBackend> {
    pub(crate) pipeline: RasterPipeline<B>,
    pub(crate) layout: BindGroupLayout<B>,
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
    pub(super) fn retain_deferred_gbuffer_programs(
        &mut self,
        inputs: &[crate::ForwardDrawInput],
        materials: &[PreparedMaterial],
        pbr: bool,
        effects: bool,
    ) {
        let active = inputs
            .iter()
            .zip(materials)
            .map(|(input, material)| {
                program_cache::GbufferProgramKey::for_material(
                    self.device.id(),
                    material,
                    if pbr { 192 } else { 64 },
                    pbr,
                    effects,
                    crate::renderer::raster_surface_state(input),
                )
            })
            .collect();
        self.deferred_programs.retain_gbuffer(&active);
    }

    pub(super) async fn deferred_gbuffer_program(
        &mut self,
        prepared: &PreparedMaterial,
        pbr: bool,
        effects: bool,
        surface: crate::renderer::RasterSurfaceState,
    ) -> std::result::Result<Arc<GbufferProgram<B>>, RendererError> {
        self.ensure_open()?;
        let transform_uniform_bytes = if pbr { 192 } else { 64 };
        let key = program_cache::GbufferProgramKey::for_material(
            self.device.id(),
            prepared,
            transform_uniform_bytes,
            pbr,
            effects,
            surface,
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
        let material_layout = self.deferred_material_layout(
            "Deferred G-buffer material bindings",
            prepared.binding_schema().ok_or(Error::InvalidState)?,
        )?;
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
                color_targets: if effects {
                    vec![
                        Some(color_target()),
                        Some(color_target()),
                        Some(color_target_with_format(TextureFormat::Rgba16Float)),
                    ]
                } else {
                    vec![Some(color_target()), Some(color_target())]
                },
                primitive: surface_triangles(surface),
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
        use_environment: bool,
        use_specular: bool,
        effects: bool,
        color_format: TextureFormat,
    ) -> std::result::Result<Arc<LightingProgram<B>>, RendererError> {
        self.ensure_open()?;
        let request = if pbr {
            if effects && use_environment {
                crate::pbr::lighting_effect_irradiance_request()
            } else if effects {
                crate::pbr::lighting_effect_request()
            } else if use_environment {
                crate::pbr::lighting_irradiance_request()
            } else {
                crate::pbr::lighting_request()
            }
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
        let request = if use_specular {
            crate::pbr::specular::deferred_specular_request(request)
        } else {
            request
        };
        let artifact = fluxel_shader::ShaderComposer::new()
            .compose(&request)
            .map_err(RendererError::DeferredShader)?;
        let key = program_cache::LightingProgramKey::for_artifact(
            self.device.id(),
            artifact.cache_key().clone(),
            pbr,
            effects,
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
                entries: {
                    let mut entries = vec![
                        sampled_texture(0, TextureSampleType::Float),
                        sampled_texture(1, TextureSampleType::Float),
                        sampled_texture(2, TextureSampleType::Depth),
                    ];
                    if effects {
                        entries.push(sampled_texture(3, TextureSampleType::Float));
                    }
                    entries
                },
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let environment_layout = if use_environment || use_specular {
            Some(
                self.device
                    .create_bind_group_layout(&BindGroupLayoutDesc {
                        label: Some("Deferred PBR environment bindings".into()),
                        entries: crate::renderer::lighting::layout_entries(
                            use_environment,
                            use_specular,
                        ),
                    })
                    .map_err(|error| self.observe_rhi_error(error))?,
            )
        } else {
            None
        };
        let groups: Vec<_> = std::iter::once(&layout)
            .chain(environment_layout.iter())
            .collect();
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Deferred lighting pipeline layout"),
                groups: &groups,
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
        let program = Arc::new(LightingProgram {
            pipeline,
            layout,
            environment_layout,
        });
        self.deferred_programs
            .publish_lighting(key, Arc::clone(&program));
        Ok(program)
    }

    pub(crate) async fn deferred_scene_color_composite_program(
        &mut self,
        color_format: TextureFormat,
    ) -> std::result::Result<SceneColorCompositeProgram<B>, RendererError> {
        self.ensure_open()?;
        let request = ShaderCompositionRequest::new(vec![SourceUnit::new(
            "deferred-scene-color-composite",
            SCENE_COLOR_COMPOSITE_TEMPLATE,
        )])
        .with_entry_point(EntryPoint::new(
            "deferred_composite_vs",
            LogicalShaderStage::Vertex,
        ))
        .with_entry_point(EntryPoint::new(
            "deferred_composite_fs",
            LogicalShaderStage::Fragment,
        ))
        .with_binding(
            BindingDeclaration::new(0, 0, BindingKind::Texture)
                .with_visibility([LogicalShaderStage::Fragment]),
        )
        .with_binding(
            BindingDeclaration::new(0, 1, BindingKind::Sampler)
                .with_visibility([LogicalShaderStage::Fragment]),
        );
        let artifact = fluxel_shader::ShaderComposer::new()
            .compose(&request)
            .map_err(RendererError::DeferredShader)?;
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
            label: Some("Fluxel Deferred scene-color composite program".into()),
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
                label: Some("Deferred scene-color composite bindings".into()),
                entries: vec![
                    sampled_texture(0, TextureSampleType::Float),
                    ordinary_sampler(1),
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Deferred scene-color composite pipeline layout"),
                groups: &[&layout],
                push_constants: &[],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline = self
            .device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("Deferred scene-color composite pipeline".into()),
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
        Ok(SceneColorCompositeProgram { pipeline, layout })
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
        schema: &crate::preparation::MaterialBindingSchema,
    ) -> std::result::Result<BindGroupLayout<B>, RendererError> {
        let mut entries = vec![uniform(schema.tint, 16, ShaderStage::Fragment)];
        entries.extend(
            schema
                .textures
                .iter()
                .map(|texture| sampled_texture(texture.binding, TextureSampleType::Float)),
        );
        if let Some(binding) = schema.sampler {
            entries.push(ordinary_sampler(binding));
        }
        entries.sort_by_key(|entry| entry.binding);
        self.device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some(label.into()),
                entries,
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
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Vertex],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: true,
                        min_size: bytes,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .map_err(|error| self.observe_rhi_error(error))
    }
}
