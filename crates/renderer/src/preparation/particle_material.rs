//! Unlit material consumer for Babylon-style GPU particle billboards.

use fluxel_material::SurfaceOutput;
use fluxel_rhi::api::{Error, Result, VertexStepMode};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ScalarType, ShaderCompositionRequest, ShaderStage,
    SourceUnit, VertexInputDeclaration, VertexType,
};

use crate::feature::particles::{
    GPU_PARTICLE_STATE_STRIDE, GPU_PARTICLE_VERTEX_ATTRIBUTES, PARTICLE_BILLBOARD_WGSL,
    PARTICLE_STATE_WGSL,
};

use super::{
    GeometryAttribute, GeometryLayout, GeometryStream,
    forward_fragment::{ForwardFragmentInterface, with_ordinary_forward_fragment},
};

pub(crate) const PARTICLE_TRANSFORM_BYTES: u64 = 240;

pub(crate) fn request(
    output: &SurfaceOutput,
    quad: &GeometryLayout,
) -> Result<(GeometryLayout, ShaderCompositionRequest)> {
    let (material, vertex_uv) = match output {
        SurfaceOutput::ColorUv0 => (
            "evaluate_material_uv0(input.uv)",
            ", @location(1) source_uv: vec2<f32>",
        ),
        SurfaceOutput::Color => ("evaluate_material()", ""),
        _ => {
            return Err(Error::Unsupported(
                "GPU particles require an unlit color material",
            ));
        }
    };
    let mut geometry = quad.clone();
    let state_attributes = &GPU_PARTICLE_VERTEX_ATTRIBUTES[..4];
    geometry.streams.push(GeometryStream {
        stride: GPU_PARTICLE_STATE_STRIDE,
        step_mode: VertexStepMode::Instance,
        attributes: state_attributes
            .iter()
            .map(|attribute| GeometryAttribute {
                semantic: attribute.semantic.into(),
                offset: attribute.offset,
                format: attribute.format,
            })
            .collect(),
    });
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("particle-state", PARTICLE_STATE_WGSL),
        SourceUnit::new("particle-spherical-billboard", PARTICLE_BILLBOARD_WGSL),
        SourceUnit::new(
            "particle-material",
            TEMPLATE
                .replace("PARTICLE_MATERIAL", material)
                .replace("/* particle_uv_input */", vertex_uv)
                .replace(
                    "/* particle_uv_value */",
                    if matches!(output, SurfaceOutput::ColorUv0) {
                        "source_uv"
                    } else {
                        "corner.xy + vec2<f32>(0.5)"
                    },
                ),
        ),
    ])
    .with_entry_point(EntryPoint::new("forward_vs", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new("forward_fs", ShaderStage::Fragment))
    .with_binding(
        BindingDeclaration::new(1, 0, BindingKind::Uniform)
            .with_visibility([ShaderStage::Vertex, ShaderStage::Fragment]),
    )
    .with_vertex_input(VertexInputDeclaration::new(
        "POSITION",
        0,
        VertexType::Vec3(ScalarType::F32),
    ))
    .with_static_option("renderer.particle", "spherical");
    if matches!(output, SurfaceOutput::ColorUv0) {
        request = request.with_vertex_input(VertexInputDeclaration::new(
            "UV0",
            1,
            VertexType::Vec2(ScalarType::F32),
        ));
    }
    for (location, attribute) in (2..6).zip(state_attributes) {
        request = request.with_vertex_input(VertexInputDeclaration::new(
            attribute.semantic,
            location,
            VertexType::Vec4(ScalarType::F32),
        ));
    }
    Ok((
        geometry,
        with_ordinary_forward_fragment(request, ForwardFragmentInterface::PositionUvColor),
    ))
}

const TEMPLATE: &str = r#"
struct ParticleTransform {
    clip_from_world: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    visibility: vec4<f32>,
    pivot: vec4<f32>,
    dead_color: vec4<f32>,
};
@group(1) @binding(0) var<uniform> forward_transform: ParticleTransform;

struct ForwardVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn forward_vs(
    @location(0) corner: vec3<f32>/* particle_uv_input */,
    @location(2) position_age: vec4<f32>,
    @location(3) velocity_lifetime: vec4<f32>,
    @location(4) size_axes_angle: vec4<f32>,
    @location(5) color: vec4<f32>,
) -> ForwardVertexOutput {
    let uv = /* particle_uv_value */;
    let particle = FluxelParticle(
        position_age, velocity_lifetime, size_axes_angle, color, vec4<f32>(0.0),
    );
    let billboard = fluxel_particle_billboard(
        particle,
        corner.xy,
        forward_transform.pivot.xy,
        forward_transform.dead_color,
        forward_transform.view_from_world,
        forward_transform.world_from_view,
    );
    if billboard.inactive {
        return ForwardVertexOutput(vec4<f32>(0.0, 0.0, 0.0, 1.0), uv, vec4<f32>(0.0));
    }
    let position = forward_transform.clip_from_world * vec4<f32>(billboard.world_position, 1.0);
    return ForwardVertexOutput(position, uv, billboard.color);
}

fn renderer_forward_shade(input: ForwardVertexOutput) -> vec4<f32> {
    let material = PARTICLE_MATERIAL * input.color;
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(material.rgb, material_output_alpha(material.a));
}
"#;
