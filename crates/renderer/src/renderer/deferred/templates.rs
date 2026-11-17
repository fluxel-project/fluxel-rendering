//! Private WGSL templates for the initial position-only Deferred probe.

use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ScalarType, ShaderCompositionRequest, ShaderStage,
    SourceUnit, VertexInputDeclaration, VertexType,
};

use crate::GeometryLayout;

/// Builds the one supported G-buffer material request.
///
/// The first Deferred probe deliberately accepts a single `POSITION: vec3<f32>`
/// vertex stream only. Rejecting other layouts prevents this template from
/// silently discarding normals, UVs, colours, or instance data. The fragment
/// derives a geometric normal from interpolated local position derivatives,
/// avoiding an invented fixed normal for the position-only fixture.
pub(super) fn gbuffer_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    let stream = layout.streams.as_slice();
    let [stream] = stream else {
        return None;
    };
    if !matches!(stream.step_mode, VertexStepMode::Vertex) || stream.attributes.len() != 1 {
        return None;
    }
    let attribute = &stream.attributes[0];
    if attribute.semantic != "POSITION"
        || attribute.offset != 0
        || !matches!(attribute.format, VertexFormat::Float32(3))
    {
        return None;
    }
    Some(
        ShaderCompositionRequest::new(vec![SourceUnit::new(
            "deferred-gbuffer-position",
            GBUFFER_TEMPLATE,
        )])
        .with_entry_point(EntryPoint::new("deferred_gbuffer_vs", ShaderStage::Vertex))
        .with_entry_point(EntryPoint::new(
            "deferred_gbuffer_fs",
            ShaderStage::Fragment,
        ))
        .with_binding(
            BindingDeclaration::new(1, 0, BindingKind::Uniform)
                .with_visibility([ShaderStage::Vertex]),
        )
        .with_vertex_input(VertexInputDeclaration::new(
            "POSITION",
            0,
            VertexType::Vec3(ScalarType::F32),
        ))
        .with_static_option("deferred.vertex_inputs", "position"),
    )
}

/// Builds the unlit UV0 G-buffer request.
///
/// Unlike the position-only compatibility route, this names `UV0` explicitly
/// and passes it to the same typed Material root used by Forward. Extra
/// geometry attributes remain legal because this consumer only declares the
/// physical footprint it reads.
pub(super) fn gbuffer_uv0_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    let mut position = false;
    let mut uv0 = false;
    for attribute in layout.streams.iter().flat_map(|stream| &stream.attributes) {
        position |= attribute.semantic == "POSITION"
            && matches!(attribute.format, VertexFormat::Float32(3));
        uv0 |= attribute.semantic == "UV0" && matches!(attribute.format, VertexFormat::Float32(2));
    }
    (position && uv0).then(|| {
        ShaderCompositionRequest::new(vec![SourceUnit::new(
            "deferred-gbuffer-color-uv0",
            GBUFFER_UV0_TEMPLATE,
        )])
        .with_entry_point(EntryPoint::new("deferred_gbuffer_vs", ShaderStage::Vertex))
        .with_entry_point(EntryPoint::new(
            "deferred_gbuffer_fs",
            ShaderStage::Fragment,
        ))
        .with_binding(
            BindingDeclaration::new(1, 0, BindingKind::Uniform)
                .with_visibility([ShaderStage::Vertex]),
        )
        .with_vertex_input(VertexInputDeclaration::new(
            "POSITION",
            0,
            VertexType::Vec3(ScalarType::F32),
        ))
        .with_vertex_input(VertexInputDeclaration::new(
            "UV0",
            1,
            VertexType::Vec2(ScalarType::F32),
        ))
        .with_static_option("deferred.vertex_inputs", "position-uv0")
    })
}

pub(super) const LIGHTING_TEMPLATE: &str = r#"
@group(0) @binding(0) var deferred_albedo: texture_2d<f32>;
@group(0) @binding(1) var deferred_normal: texture_2d<f32>;
@group(0) @binding(2) var deferred_depth: texture_depth_2d;

struct LightingVertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn deferred_lighting_vs(@builtin(vertex_index) index: u32) -> LightingVertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    var output: LightingVertexOutput;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    return output;
}

@fragment
fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let coordinate = vec2<i32>(position.xy);
    let depth = textureLoad(deferred_depth, coordinate, 0);
    if depth >= 1.0 {
        return vec4<f32>(0.0);
    }
    let albedo = textureLoad(deferred_albedo, coordinate, 0);
    let normal = textureLoad(deferred_normal, coordinate, 0).xyz * 2.0 - vec3<f32>(1.0);
    let diffuse = 0.2 + 0.6 * abs(normal.z) + 0.2 * (1.0 - depth);
    return vec4<f32>(albedo.rgb * diffuse, albedo.a);
}
"#;

pub(super) const SCENE_COLOR_COMPOSITE_TEMPLATE: &str = r#"
@group(0) @binding(0) var deferred_scene_color: texture_2d<f32>;
@group(0) @binding(1) var deferred_scene_sampler: sampler;

struct CompositeVertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn deferred_composite_vs(@builtin(vertex_index) index: u32) -> CompositeVertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    var output: CompositeVertexOutput;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    return output;
}

@fragment
fn deferred_composite_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(deferred_scene_color, 0));
    // Fragment position already addresses the pixel centre. Adding another
    // half pixel would blend adjacent texels in this one-to-one composite.
    let uv = position.xy / size;
    return textureSampleLevel(deferred_scene_color, deferred_scene_sampler, uv, 0.0);
}
"#;

const GBUFFER_TEMPLATE: &str = r#"
@group(1) @binding(0) var<uniform> deferred_clip_from_local: mat4x4<f32>;

struct DeferredVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_position: vec3<f32>,
};

struct GbufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@vertex
fn deferred_gbuffer_vs(@location(0) position: vec3<f32>) -> DeferredVertexOutput {
    var output: DeferredVertexOutput;
    output.position = deferred_clip_from_local * vec4<f32>(position, 1.0);
    output.local_position = position;
    return output;
}

@fragment
fn deferred_gbuffer_fs(input: DeferredVertexOutput, @builtin(front_facing) front_facing: bool) -> GbufferOutput {
    let derivative_normal = normalize(cross(dpdx(input.local_position), dpdy(input.local_position)));
    let material = evaluate_material();
    if material_alpha_rejected(material.a) { discard; }
    var output: GbufferOutput;
    output.albedo = vec4<f32>(material.rgb, material_output_alpha(material.a));
    output.normal = vec4<f32>(select(derivative_normal, -derivative_normal, material_two_sided_lighting() && !front_facing) * 0.5 + vec3<f32>(0.5), 1.0);
    return output;
}
"#;

const GBUFFER_UV0_TEMPLATE: &str = r#"
@group(1) @binding(0) var<uniform> deferred_clip_from_local: mat4x4<f32>;

struct DeferredUv0VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_position: vec3<f32>,
    @location(1) uv0: vec2<f32>,
};

struct GbufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@vertex
fn deferred_gbuffer_vs(
    @location(0) position: vec3<f32>,
    @location(1) uv0: vec2<f32>,
) -> DeferredUv0VertexOutput {
    var output: DeferredUv0VertexOutput;
    output.position = deferred_clip_from_local * vec4<f32>(position, 1.0);
    output.local_position = position;
    output.uv0 = uv0;
    return output;
}

@fragment
fn deferred_gbuffer_fs(input: DeferredUv0VertexOutput, @builtin(front_facing) front_facing: bool) -> GbufferOutput {
    let derivative_normal = normalize(cross(dpdx(input.local_position), dpdy(input.local_position)));
    let material = evaluate_material_uv0(input.uv0);
    if material_alpha_rejected(material.a) { discard; }
    var output: GbufferOutput;
    output.albedo = vec4<f32>(material.rgb, material_output_alpha(material.a));
    output.normal = vec4<f32>(select(derivative_normal, -derivative_normal, material_two_sided_lighting() && !front_facing) * 0.5 + vec3<f32>(0.5), 1.0);
    return output;
}
"#;

#[cfg(test)]
#[path = "../../tests/renderer/deferred_templates.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/renderer/deferred_pbr_templates.rs"]
mod pbr_tests;
