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
fn deferred_gbuffer_fs(input: DeferredVertexOutput) -> GbufferOutput {
    let derivative_normal = normalize(cross(dpdx(input.local_position), dpdy(input.local_position)));
    var output: GbufferOutput;
    output.albedo = evaluate_material();
    output.normal = vec4<f32>(derivative_normal * 0.5 + vec3<f32>(0.5), 1.0);
    return output;
}
"#;

#[cfg(test)]
#[path = "../../tests/renderer/deferred_templates.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/renderer/deferred_pbr_templates.rs"]
mod pbr_tests;
