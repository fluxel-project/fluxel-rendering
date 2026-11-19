//! Babylon-style cotangent reconstruction when the mesh has no tangents.

use fluxel_shader::{ShaderCompositionRequest, SourceUnit};

const BASIS_OPTION: &str = "renderer.normal_basis";
const DERIVATIVE: &str = "derivative";

pub(super) fn mark_derivative(request: &mut ShaderCompositionRequest) {
    request
        .static_options
        .insert(BASIS_OPTION.into(), DERIVATIVE.into());
}

pub(super) fn uses_derivatives(request: &ShaderCompositionRequest) -> bool {
    request
        .static_options
        .get(BASIS_OPTION)
        .is_some_and(|basis| basis == DERIVATIVE)
}

/// Applies the normal basis after the pass and instance variants are chosen.
pub(crate) fn specialize_normal_basis(
    request: &mut ShaderCompositionRequest,
    settings: fluxel_material::NormalMapSettings,
) {
    if !uses_derivatives(request) {
        return;
    }
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-tangent-basis");
    for unit in &mut request.source_units {
        if !unit.id.0.starts_with("pbr-forward") && !unit.id.0.starts_with("pbr-gbuffer") {
            continue;
        }
        unit.source = unit.source
            .replace(", @location(3) tangent: vec4<f32>", "")
            .replace("pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */)", "mat3x3<f32>()")
            .replace("pbr_vertex_tangent_basis(normal, tangent , instance_world)", "mat3x3<f32>()")
            .replace(
                "pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0))",
                "pbr_derivative_normal(pbr_normal(input.geometric_world_normal), input.world_position, input.uv0, front_facing, evaluate_normal_uv0(input.uv0))",
            );
    }
    let tangent_sign = if settings.invert_x { "-1.0" } else { "1.0" };
    let bitangent_sign = if settings.invert_y { "-1.0" } else { "1.0" };
    request.source_units.push(SourceUnit::new(
        "pbr-derivative-normal",
        COTANGENT_FRAME
            .replace("NORMAL_TANGENT_SIGN", tangent_sign)
            .replace("NORMAL_BITANGENT_SIGN", bitangent_sign),
    ));
}

// Babylon.js bumpFragment.fx and bumpFragmentMainFunctions.fx,
// revision daae8d9483dcbb6308ebd0f6ca5b218783614771.
// Copyright 2023 The Babylon.js team. Licensed under Apache-2.0.
const COTANGENT_FRAME: &str = r#"
fn pbr_derivative_normal(
    normal: vec3<f32>, position: vec3<f32>, uv: vec2<f32>, front_facing: bool,
    tangent_normal: vec3<f32>,
) -> vec3<f32> {
    let face_uv = select(-uv, uv, front_facing);
    let dp1 = dpdx(position);
    let dp2 = dpdy(position);
    let duv1 = dpdx(face_uv);
    let duv2 = dpdy(face_uv);
    let dp2perp = cross(dp2, normal);
    let dp1perp = cross(normal, dp1);
    let tangent = (dp2perp * duv1.x + dp1perp * duv2.x) * NORMAL_TANGENT_SIGN;
    let bitangent = (dp2perp * duv1.y + dp1perp * duv2.y) * NORMAL_BITANGENT_SIGN;
    let magnitude = max(dot(tangent, tangent), dot(bitangent, bitangent));
    let invmax = select(inverseSqrt(magnitude), 0.0, magnitude == 0.0);
    let basis = mat3x3<f32>(tangent * invmax, bitangent * invmax, normal);
    let world_normal = basis * tangent_normal;
    return select(normal, pbr_normal(world_normal), any(world_normal != vec3<f32>(0.0)));
}
"#;
