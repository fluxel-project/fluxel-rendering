//! Realization of the Deferred sample-zero G-buffer resolve.

use super::super::{Renderer, RendererError};
use crate::residency::{DeferredGbufferPlan, DeferredMsaa};
use fluxel_rhi::{
    api::*,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderComposer, ShaderCompositionRequest,
    ShaderStage as LogicalStage, SourceUnit,
};
use std::sync::Arc;

pub(super) struct MsaaResolveProgram<B: ResourceBackend> {
    pipeline: RasterPipeline<B>,
    layout: BindGroupLayout<B>,
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
    pub(super) async fn realize_deferred_msaa(
        &mut self,
        plan: Option<DeferredGbufferPlan>,
        pbr: bool,
        effects: bool,
        diffuse_roughness: bool,
        ambient_occlusion: bool,
    ) -> std::result::Result<Option<DeferredMsaa<B>>, RendererError> {
        let Some(plan) = plan else {
            return Ok(None);
        };
        let program = self
            .deferred_msaa_resolve_program(pbr, effects, diffuse_roughness, ambient_occlusion)
            .await?;
        let (gbuffer, effects, diffuse_roughness, ambient_occlusion) = plan
            .allocate_with_aux(&self.device)
            .map_err(|error| self.observe_rhi_error(error))?;
        let entries = gbuffer
            .iter()
            .enumerate()
            .map(|(binding, texture)| BindGroupEntry {
                binding: binding as u32,
                resource: BindingResource::TextureView(&texture.view),
            })
            .chain(effects.iter().map(|texture| BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&texture.view),
            }))
            .chain(diffuse_roughness.iter().map(|texture| BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&texture.view),
            }))
            .chain(ambient_occlusion.iter().map(|texture| BindGroupEntry {
                binding: 5,
                resource: BindingResource::TextureView(&texture.view),
            }))
            .collect::<Vec<_>>();
        let group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Deferred multisampled G-buffer"),
                layout: &program.layout,
                entries: &entries,
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        self.deferred_stats.texture_allocations =
            self.deferred_stats.texture_allocations.saturating_add(
                3 + u64::from(effects.is_some())
                    + u64::from(diffuse_roughness.is_some())
                    + u64::from(ambient_occlusion.is_some()),
            );
        Ok(Some(DeferredMsaa {
            gbuffer,
            effects,
            diffuse_roughness,
            ambient_occlusion,
            pipeline: program.pipeline.clone(),
            group,
        }))
    }

    async fn deferred_msaa_resolve_program(
        &mut self,
        pbr: bool,
        effects: bool,
        diffuse_roughness: bool,
        ambient_occlusion: bool,
    ) -> std::result::Result<Arc<MsaaResolveProgram<B>>, RendererError> {
        if let Some(program) = self.deferred_programs.get_msaa_resolve(
            pbr,
            effects,
            diffuse_roughness,
            ambient_occlusion,
        ) {
            return Ok(program);
        }
        let request = resolve_request(effects, diffuse_roughness, ambient_occlusion);
        let artifact = ShaderComposer::new()
            .compose(&request)
            .map_err(RendererError::DeferredShader)?;
        let descriptor = ShaderDesc {
            label: Some("Deferred G-buffer resolve shader".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(artifact.source().as_bytes()),
            entries: [
                ("deferred_resolve_vs", ShaderStage::Vertex),
                ("deferred_resolve_fs", ShaderStage::Fragment),
            ]
            .into_iter()
            .map(|(name, stage)| ShaderEntryPoint {
                name: name.into(),
                stage,
                workgroup_size: None,
            })
            .collect(),
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
                label: Some("Deferred resolve layout".into()),
                entries: (0..3)
                    .chain(effects.then_some(3))
                    .chain(diffuse_roughness.then_some(4))
                    .chain(ambient_occlusion.then_some(5))
                    .map(|binding| BindGroupLayoutEntry {
                        binding,
                        visibility: vec![ShaderStage::Fragment],
                        ty: BindingType::SampledTexture {
                            sample: if binding == 2 {
                                TextureSampleType::Depth
                            } else {
                                TextureSampleType::UnfilterableFloat
                            },
                            dimension: TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: BindingCount::One,
                        partially_bound: false,
                        nonuniform_indexing: false,
                    })
                    .collect(),
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Deferred resolve pipeline layout"),
                groups: &[&layout],
                push_constants: &[],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        use super::program::program_fixed::{
            color_target_with_format, depth_state, opaque_triangles, single_sample,
        };
        let mut colors = vec![
            Some(color_target_with_format(
                crate::residency::deferred_albedo_format(pbr),
            )),
            Some(color_target_with_format(
                crate::residency::deferred_normal_format(pbr),
            )),
        ];
        if effects {
            colors.push(Some(color_target_with_format(
                crate::residency::deferred_effects_format(ambient_occlusion),
            )));
        }
        if diffuse_roughness {
            colors.push(Some(color_target_with_format(TextureFormat::R32Float)));
        }
        if ambient_occlusion {
            colors.push(Some(color_target_with_format(TextureFormat::Rgba32Float)));
        }
        let mut depth = depth_state();
        depth.depth_compare = CompareFunction::Always;
        let pipeline = self
            .device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("Deferred G-buffer resolve pipeline".into()),
                layout: pipeline_layout,
                vertex: PipelineStage {
                    module: shader.clone(),
                    entry: "deferred_resolve_vs".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: shader,
                    entry: "deferred_resolve_fs".into(),
                    constants: vec![],
                }),
                vertex_buffers: vec![],
                color_targets: colors,
                primitive: opaque_triangles(),
                depth_stencil: Some(depth),
                multisample: single_sample(),
                multiview: None,
                cache: None,
            })
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        let program = Arc::new(MsaaResolveProgram { pipeline, layout });
        self.deferred_programs.publish_msaa_resolve(
            pbr,
            effects,
            diffuse_roughness,
            ambient_occlusion,
            Arc::clone(&program),
        );
        Ok(program)
    }
}

fn resolve_request(
    effects: bool,
    diffuse_roughness: bool,
    ambient_occlusion: bool,
) -> ShaderCompositionRequest {
    let (effects_binding, effects_member, effects_load) = if effects {
        (
            "@group(0) @binding(3) var effects_msaa: texture_multisampled_2d<f32>;",
            "@location(2) effects: vec4<f32>,",
            "output.effects = textureLoad(effects_msaa, pixel, 0);",
        )
    } else {
        ("", "", "")
    };
    let diffuse_roughness_location = 2 + u32::from(effects);
    let (diffuse_roughness_binding, diffuse_roughness_member, diffuse_roughness_load) =
        if diffuse_roughness {
            (
                "@group(0) @binding(4) var diffuse_roughness_msaa: texture_multisampled_2d<f32>;",
                format!("@location({diffuse_roughness_location}) diffuse_roughness: f32,"),
                "output.diffuse_roughness = textureLoad(diffuse_roughness_msaa, pixel, 0).r;",
            )
        } else {
            ("", String::new(), "")
        };
    let ambient_occlusion_location = 2 + u32::from(effects) + u32::from(diffuse_roughness);
    let (ambient_occlusion_binding, ambient_occlusion_member, ambient_occlusion_load) =
        if ambient_occlusion {
            (
                "@group(0) @binding(5) var ambient_occlusion_msaa: texture_multisampled_2d<f32>;",
                format!("@location({ambient_occlusion_location}) ambient_occlusion: vec4<f32>,"),
                "output.ambient_occlusion = textureLoad(ambient_occlusion_msaa, pixel, 0);",
            )
        } else {
            ("", String::new(), "")
        };
    // Babylon WebGPU resolves depth from sample zero by default. Load all
    // G-buffer fields from that same surface so lighting never reconstructs a
    // position from one sample while shading a different sample's material.
    let source = format!(
        r#"
@group(0) @binding(0) var albedo_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(1) var normal_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(2) var depth_msaa: texture_depth_multisampled_2d;
{effects_binding}
{diffuse_roughness_binding}
{ambient_occlusion_binding}

@vertex fn deferred_resolve_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {{
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}}

struct ResolveOutput {{
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    {effects_member}
    {diffuse_roughness_member}
    {ambient_occlusion_member}
    @builtin(frag_depth) depth: f32,
}}

@fragment fn deferred_resolve_fs(@builtin(position) position: vec4<f32>) -> ResolveOutput {{
    let pixel = vec2<i32>(position.xy);
    var output: ResolveOutput;
    output.albedo = textureLoad(albedo_msaa, pixel, 0);
    output.normal = textureLoad(normal_msaa, pixel, 0);
    output.depth = textureLoad(depth_msaa, pixel, 0);
    {effects_load}
    {diffuse_roughness_load}
    {ambient_occlusion_load}
    return output;
}}
"#
    );
    let mut request =
        ShaderCompositionRequest::new(vec![SourceUnit::new("deferred-resolve", source)])
            .with_entry_point(EntryPoint::new("deferred_resolve_vs", LogicalStage::Vertex))
            .with_entry_point(EntryPoint::new(
                "deferred_resolve_fs",
                LogicalStage::Fragment,
            ));
    for binding in (0..3)
        .chain(effects.then_some(3))
        .chain(diffuse_roughness.then_some(4))
        .chain(ambient_occlusion.then_some(5))
    {
        request = request.with_binding(
            BindingDeclaration::new(0, binding, BindingKind::Texture)
                .with_visibility([LogicalStage::Fragment]),
        );
    }
    request
}
