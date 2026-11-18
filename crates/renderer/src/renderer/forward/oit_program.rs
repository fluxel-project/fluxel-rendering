//! Frame-owned dual-depth-peeling fullscreen programs.
//!
//! These programs deliberately have no renderer cache: their layouts describe
//! the exact frame-local OIT products retained through submission completion.

use super::*;

pub(super) struct OitFullscreenProgram<B: ResourceBackend> {
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) layout: BindGroupLayout<B>,
}

pub(super) struct OitFullscreenPrograms<B: ResourceBackend> {
    pub(super) initialize_back: OitFullscreenProgram<B>,
    pub(super) accumulate_back: OitFullscreenProgram<B>,
    pub(super) final_composite: OitFullscreenProgram<B>,
}

/// Frame-owned material programs for one dual-depth-peeling draw family.
pub(super) struct OitMaterialProgram<B: ResourceBackend> {
    pub(super) initial: RasterPipeline<B>,
    pub(super) peel: RasterPipeline<B>,
    pub(super) preliminary_initial: Option<RasterPipeline<B>>,
    pub(super) preliminary_peel: Option<RasterPipeline<B>>,
    pub(super) initial_layout: BindGroupLayout<B>,
    pub(super) peel_layout: BindGroupLayout<B>,
    pub(super) _empty_environment_layout: Option<BindGroupLayout<B>>,
    pub(super) empty_environment_group: Option<BindGroup<B>>,
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
    pub(super) async fn oit_fullscreen_programs(
        &mut self,
        final_format: TextureFormat,
    ) -> std::result::Result<OitFullscreenPrograms<B>, RendererError> {
        self.ensure_open()?;
        Ok(OitFullscreenPrograms {
            initialize_back: self
                .oit_fullscreen_program(
                    "OIT initialize back accumulator",
                    INIT_SOURCE,
                    1,
                    TextureFormat::Rgba16Float,
                    None,
                )
                .await?,
            accumulate_back: self
                .oit_fullscreen_program(
                    "OIT accumulate back layers",
                    BACK_SOURCE,
                    1,
                    TextureFormat::Rgba16Float,
                    Some(back_blend()),
                )
                .await?,
            final_composite: self
                .oit_fullscreen_program("OIT final composite", FINAL_SOURCE, 2, final_format, None)
                .await?,
        })
    }

    async fn oit_fullscreen_program(
        &mut self,
        label: &'static str,
        source: &'static str,
        textures: u32,
        target: TextureFormat,
        blend: Option<BlendState>,
    ) -> std::result::Result<OitFullscreenProgram<B>, RendererError> {
        if self.device.capabilities().limits.max_bind_groups < 1 {
            return Err(Error::Unsupported("OIT fullscreen bind group").into());
        }
        let descriptor = ShaderDesc {
            label: Some(label.into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![
                ShaderEntryPoint {
                    name: "oit_vs".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "oit_fs".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: Vec::new(),
        };
        // The fixed fullscreen WGSL uses no optional Naga capabilities, but
        // still consumes the same device-fact admission path as material code.
        admission::admit(
            &[],
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
            self.device.shader_route(&descriptor),
            ShaderRoute::Unsupported
        ) {
            return Err(Error::Unsupported("OIT fullscreen shader route").into());
        }
        let shader = self
            .device
            .shader(descriptor)
            .await
            .map_err(|error| self.observe_forward_error("OIT fullscreen shader", error))?;
        let entries = (0..textures).map(unfilterable_texture).collect();
        let layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some(label.into()),
                entries,
            })
            .map_err(|error| self.observe_forward_error("OIT fullscreen layout", error))?;
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some(label),
                groups: &[&layout],
                push_constants: &[],
            })
            .map_err(|error| self.observe_forward_error("OIT fullscreen pipeline layout", error))?;
        let pipeline = self
            .device
            .raster_pipeline(RasterPipelineDesc {
                label: Some(label.into()),
                layout: pipeline_layout,
                vertex: PipelineStage {
                    module: shader.clone(),
                    entry: "oit_vs".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: shader,
                    entry: "oit_fs".into(),
                    constants: vec![],
                }),
                vertex_buffers: vec![],
                color_targets: vec![Some(ColorTargetState {
                    format: target,
                    blend,
                    write_mask: 0xf,
                })],
                primitive: fullscreen_triangles(),
                depth_stencil: None,
                multisample: single_sample(),
                multiview: None,
                cache: None,
            })
            .await
            .map_err(|error| self.observe_forward_error("OIT fullscreen pipeline", error))?;
        Ok(OitFullscreenProgram { pipeline, layout })
    }

    /// Creates the two OIT MRT pipelines from stage-specialized prepared
    /// artifacts. Groups 0--2 are the actual ordinary program layouts, so
    /// existing material/transform/environment bind groups remain valid.
    pub(super) async fn oit_material_program(
        &mut self,
        ordinary: &PreparedMaterial,
        base: &ForwardPipelineObjects<B>,
        initial: &PreparedMaterial,
        peel: &PreparedMaterial,
        min_max_format: TextureFormat,
        transform_uniform_bytes: u64,
    ) -> std::result::Result<OitMaterialProgram<B>, RendererError> {
        let Some((initial_stage, initial_precision)) = initial.oit_fragment() else {
            return Err(Error::InvalidState.into());
        };
        let Some((peel_stage, peel_precision)) = peel.oit_fragment() else {
            return Err(Error::InvalidState.into());
        };
        if initial_stage != crate::preparation::forward_fragment::OitDrawStage::InitialDepth
            || peel_stage != crate::preparation::forward_fragment::OitDrawStage::Peel
            || initial_precision != peel_precision
            || !matches!(
                (initial_precision, min_max_format),
                (
                    crate::preparation::forward_fragment::OitDepthPrecision::Rg32Float,
                    TextureFormat::Rg32Float
                ) | (
                    crate::preparation::forward_fragment::OitDepthPrecision::Rg16Float,
                    TextureFormat::Rg16Float
                )
            )
            || ordinary.binding_schema() != initial.binding_schema()
            || ordinary.binding_schema() != peel.binding_schema()
            || base.key.program != *ordinary.shader().cache_key()
            || base.key.transform_uniform_bytes != transform_uniform_bytes
            || base.key.layout
                != ordinary
                    .geometry()
                    .streams
                    .iter()
                    .map(crate::preparation::vertex_layout_key)
                    .collect::<Vec<_>>()
            || base.key.layout
                != initial
                    .geometry()
                    .streams
                    .iter()
                    .map(crate::preparation::vertex_layout_key)
                    .collect::<Vec<_>>()
            || base.key.layout
                != peel
                    .geometry()
                    .streams
                    .iter()
                    .map(crate::preparation::vertex_layout_key)
                    .collect::<Vec<_>>()
            || ordinary.uses_environment_irradiance() != initial.uses_environment_irradiance()
            || ordinary.uses_environment_irradiance() != peel.uses_environment_irradiance()
            || ordinary.uses_specular_ibl() != initial.uses_specular_ibl()
            || ordinary.uses_specular_ibl() != peel.uses_specular_ibl()
            || self.device.capabilities().limits.max_bind_groups < 4
            || self.device.capabilities().limits.max_color_attachments < 3
        {
            return Err(Error::InvalidState.into());
        }
        let initial_layout = self.oit_draw_layout("OIT initial bindings", false)?;
        let peel_layout = self.oit_draw_layout("OIT peel bindings", true)?;
        let environment = base.environment_layout.as_ref();
        let empty_layout = if environment.is_none() {
            Some(
                self.device
                    .create_bind_group_layout(&BindGroupLayoutDesc {
                        label: Some("OIT empty environment gap".into()),
                        entries: vec![],
                    })
                    .map_err(|e| self.observe_forward_error("OIT environment gap", e))?,
            )
        } else {
            None
        };
        let empty_environment_group = empty_layout
            .as_ref()
            .map(|layout| {
                self.device.create_bind_group(&BindGroupDesc {
                    label: Some("OIT empty environment gap"),
                    layout,
                    entries: &[],
                })
            })
            .transpose()
            .map_err(|e| self.observe_forward_error("OIT environment gap group", e))?;
        let group2 = environment
            .or(empty_layout.as_ref())
            .ok_or(Error::InvalidState)?;
        let surface = base
            .key
            .surface
            .for_oit(ordinary.material().surface_policy().separate_culling_pass);
        let (initial_pipeline, preliminary_initial) = self
            .oit_draw_pipeline(
                "OIT initial depth",
                initial,
                base,
                group2,
                &initial_layout,
                surface,
                vec![oit_target(min_max_format)],
                Some(read_only_depth()),
            )
            .await?;
        let (peel_pipeline, preliminary_peel) = self
            .oit_draw_pipeline(
                "OIT peel",
                peel,
                base,
                group2,
                &peel_layout,
                surface,
                vec![
                    oit_target(min_max_format),
                    oit_target(TextureFormat::Rgba16Float),
                    oit_target(TextureFormat::Rgba16Float),
                ],
                None,
            )
            .await?;
        Ok(OitMaterialProgram {
            initial: initial_pipeline,
            peel: peel_pipeline,
            preliminary_initial,
            preliminary_peel,
            initial_layout,
            peel_layout,
            _empty_environment_layout: empty_layout,
            empty_environment_group,
        })
    }

    fn oit_draw_layout(
        &mut self,
        label: &'static str,
        peel: bool,
    ) -> std::result::Result<BindGroupLayout<B>, RendererError> {
        let mut entries = vec![unfilterable_texture(0)];
        if peel {
            entries.push(unfilterable_texture(1));
        }
        self.device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some(label.into()),
                entries,
            })
            .map_err(|e| self.observe_forward_error("OIT draw layout", e))
    }

    async fn oit_draw_pipeline(
        &mut self,
        label: &'static str,
        prepared: &PreparedMaterial,
        base: &ForwardPipelineObjects<B>,
        group2: &BindGroupLayout<B>,
        oit: &BindGroupLayout<B>,
        surface: crate::renderer::RasterSurfaceState,
        targets: Vec<Option<ColorTargetState>>,
        depth: Option<DepthStencilState>,
    ) -> std::result::Result<(RasterPipeline<B>, Option<RasterPipeline<B>>), RendererError> {
        let artifact = prepared.shader();
        let vertex = artifact
            .entry_point(fluxel_shader::ShaderStage::Vertex)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let fragment = artifact
            .entry_point(fluxel_shader::ShaderStage::Fragment)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let desc = ShaderDesc {
            label: Some(label.into()),
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
        let demands: Vec<_> = artifact
            .requirements()
            .naga_capabilities
            .iter()
            .map(|capability| capability.name().to_owned())
            .collect();
        admission::admit(
            &demands,
            self.device.capabilities(),
            PipelineCapabilityFacts {
                multiview_views: None,
                push_constant_bytes: 0,
                multisample_count: 1,
                nonuniform: Default::default(),
            },
        )
        .map_err(RendererError::ShaderAdmission)?;
        if matches!(self.device.shader_route(&desc), ShaderRoute::Unsupported) {
            return Err(Error::Unsupported("OIT material shader route").into());
        }
        let shader = self
            .device
            .shader(desc)
            .await
            .map_err(|e| self.observe_forward_error("OIT material shader", e))?;
        let layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some(label),
                groups: &[&base.material_layout, &base.transform_layout, group2, oit],
                push_constants: &[],
            })
            .map_err(|e| self.observe_forward_error("OIT material pipeline layout", e))?;
        let mut primitive = fullscreen_triangles();
        surface.apply_to(&mut primitive);
        let raster = |label: &str, primitive| RasterPipelineDesc {
            label: Some(label.into()),
            layout: layout.clone(),
            vertex: PipelineStage {
                module: shader.clone(),
                entry: vertex.clone(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader.clone(),
                entry: fragment.clone(),
                constants: vec![],
            }),
            vertex_buffers: prepared.geometry().streams.clone(),
            color_targets: targets.clone(),
            primitive,
            depth_stencil: depth.clone(),
            multisample: single_sample(),
            multiview: None,
            cache: None,
        };
        let preliminary = if let Some(primitive) = surface.preliminary_primitive(primitive) {
            Some(
                self.device
                    .raster_pipeline(raster(label, primitive))
                    .await
                    .map_err(|e| self.observe_forward_error("OIT preliminary pipeline", e))?,
            )
        } else {
            None
        };
        let primary = self
            .device
            .raster_pipeline(raster(label, primitive))
            .await
            .map_err(|e| self.observe_forward_error("OIT material pipeline", e))?;
        Ok((primary, preliminary))
    }
}

fn unfilterable_texture(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::SampledTexture {
            sample: TextureSampleType::UnfilterableFloat,
            dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}

fn back_blend() -> BlendState {
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }
}
fn oit_target(format: TextureFormat) -> Option<ColorTargetState> {
    Some(ColorTargetState {
        format,
        blend: Some(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Max,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Max,
            },
        }),
        write_mask: 0xf,
    })
}
fn read_only_depth() -> DepthStencilState {
    DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: false,
        depth_compare: CompareFunction::LessEqual,
        stencil_front: StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        },
        stencil_back: StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        },
        stencil_read_mask: 0,
        stencil_write_mask: 0,
        depth_bias_constant: 0,
        depth_bias_slope_scale: 0.,
        depth_bias_clamp: 0.,
    }
}

fn fullscreen_triangles() -> PrimitiveState {
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

const INIT_SOURCE: &str = r#"@group(0) @binding(0) var opaque: texture_2d<f32>;
struct O { @builtin(position) p: vec4<f32>, };
@vertex fn oit_vs(@builtin(vertex_index) i:u32)->O { var v=array<vec2<f32>,3>(vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.)); var o:O;o.p=vec4(v[i],0.,1.);return o; }
@fragment fn oit_fs(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32>{return textureLoad(opaque,vec2<i32>(p.xy),0);}"#;
const BACK_SOURCE: &str = r#"@group(0) @binding(0) var back: texture_2d<f32>;
struct O { @builtin(position) p: vec4<f32>, };
@vertex fn oit_vs(@builtin(vertex_index) i:u32)->O { var v=array<vec2<f32>,3>(vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.)); var o:O;o.p=vec4(v[i],0.,1.);return o; }
@fragment fn oit_fs(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32>{return textureLoad(back,vec2<i32>(p.xy),0);}"#;
// Babylon's oitFinal fragment uses additive front.a + back.a for final alpha.
const FINAL_SOURCE: &str = r#"@group(0) @binding(0) var front: texture_2d<f32>; @group(0) @binding(1) var back: texture_2d<f32>;
struct O { @builtin(position) p: vec4<f32>, };
@vertex fn oit_vs(@builtin(vertex_index) i:u32)->O { var v=array<vec2<f32>,3>(vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.)); var o:O;o.p=vec4(v[i],0.,1.);return o; }
@fragment fn oit_fs(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32>{let c=vec2<i32>(p.xy);let f=textureLoad(front,c,0);let b=textureLoad(back,c,0);return vec4(f.rgb+(1.-f.a)*b.rgb,f.a+b.a);}"#;
