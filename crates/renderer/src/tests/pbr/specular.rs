use crate::pbr::specular::{WGSL, deferred_specular_request, with_specular_ibl};
use crate::{GeometryAttribute, GeometryLayout, GeometryStream};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderComposer, ShaderCompositionRequest,
    ShaderStage, SourceUnit,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpecularIblOracle {
    lod: f32,
    brdf_uv: [f32; 2],
    f0: [f32; 3],
    reflectance: [f32; 3],
    diffuse_energy: [f32; 3],
}

/// Independent scalar reference for Babylon's selected linear-roughness and
/// height-correlated BRDF branch; it deliberately does not call production Rust.
fn oracle(
    base: [f32; 3],
    metallic: f32,
    roughness: f32,
    ndotv: f32,
    dielectric_f0: [f32; 3],
    lut: [f32; 2],
    max_lod: f32,
    lod_scale: f32,
    lod_offset: f32,
) -> SpecularIblOracle {
    let clamp = |value: f32| value.clamp(0.0, 1.0);
    let metal = clamp(metallic);
    let f0 = std::array::from_fn(|index| {
        clamp(dielectric_f0[index]) * (1.0 - metal) + clamp(base[index]) * metal
    });
    let lod = ((max_lod * clamp(roughness)) * lod_scale + lod_offset).clamp(0.0, max_lod);
    let reflectance = std::array::from_fn(|index| {
        ((1.0 - f0[index]) * clamp(lut[0]) + f0[index] * clamp(lut[1])).clamp(0.0, 1.0)
    });
    let base_f0 = (1.0 - metal) * dielectric_f0.into_iter().fold(0.0f32, f32::max) + metal;
    let base_reflectance =
        ((1.0 - base_f0) * clamp(lut[0]) + base_f0 * clamp(lut[1])).clamp(0.0, 1.0);
    SpecularIblOracle {
        lod,
        brdf_uv: [clamp(ndotv), clamp(roughness)],
        f0,
        reflectance,
        diffuse_energy: [1.0 - base_reflectance; 3],
    }
}

/// Independent Deferred coordinate oracle: fragment coordinates are already
/// pixel centres in top-left space, whereas clip-space Y increases upward.
fn deferred_pixel_center_to_ndc(position: [f32; 2], extent: [f32; 2]) -> [f32; 2] {
    [
        2.0 * position[0] / extent[0] - 1.0,
        1.0 - 2.0 * position[1] / extent[1],
    ]
}

fn position_normal_layout() -> GeometryLayout {
    GeometryLayout {
        streams: vec![GeometryStream {
            stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                GeometryAttribute {
                    semantic: "POSITION".into(),
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                GeometryAttribute {
                    semantic: "NORMAL".into(),
                    offset: 12,
                    format: VertexFormat::Float32(3),
                },
            ],
        }],
    }
}

fn uv0_layout() -> GeometryLayout {
    let mut layout = position_normal_layout();
    layout.streams[0].stride = 32;
    layout.streams[0].attributes.push(GeometryAttribute {
        semantic: "UV0".into(),
        offset: 24,
        format: VertexFormat::Float32(2),
    });
    layout
}

fn uv0_normal_layout() -> GeometryLayout {
    let mut layout = uv0_layout();
    layout.streams[0].stride = 48;
    layout.streams[0].attributes.push(GeometryAttribute {
        semantic: "TANGENT".into(),
        offset: 32,
        format: VertexFormat::Float32(4),
    });
    layout
}

fn actual_material_evaluators(request: &ShaderCompositionRequest) -> SourceUnit {
    let uv0 = request
        .vertex_inputs
        .iter()
        .any(|input| input.semantic == "UV0");
    let normal = request
        .vertex_inputs
        .iter()
        .any(|input| input.semantic == "TANGENT");
    let effects = request
        .static_options
        .contains_key("renderer.surface_effects");
    let source = if uv0 {
        let normal = if normal {
            "fn evaluate_normal_uv0(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0, 0.0, 1.0 + uv.x * 0.0); }"
        } else {
            ""
        };
        let effects = if effects {
            "fn evaluate_occlusion_uv0(uv: vec2<f32>) -> f32 { return 0.5 + uv.x * 0.0; } fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.1 + uv.x * 0.0); }"
        } else {
            ""
        };
        format!(
            "struct FluxelPbrSurface {{ base: vec4<f32>, properties: vec2<f32>, world_normal: vec3<f32>, has_world_normal: bool, occlusion: f32, }}; fn evaluate_pbr_vertex_uv0(uv: vec2<f32>) -> vec2<f32> {{ return uv; }} fn evaluate_pbr_surface_uv0(uv: vec2<f32>, world_position: vec3<f32>, world_normal: vec3<f32>, camera_position: vec3<f32>, vertex_uv0: vec2<f32>) -> FluxelPbrSurface {{ return FluxelPbrSurface(vec4<f32>(0.8 + uv.x * 0.0 + world_position.x * 0.0 + world_normal.x * 0.0 + camera_position.x * 0.0 + vertex_uv0.x * 0.0, 0.6, 0.4, 1.0), vec2<f32>(0.2 + uv.x * 0.0, 0.7), vec3<f32>(0.0), false, 1.0); }} {normal} {effects}"
        )
    } else {
        "fn evaluate_material() -> vec4<f32> { return vec4<f32>(0.8, 0.6, 0.4, 1.0); } fn evaluate_metallic_roughness() -> vec2<f32> { return vec2<f32>(0.2, 0.7); }".into()
    };
    SourceUnit::new("pbr-specular-real-builder-material", source)
}

fn compose_actual_forward(mut request: ShaderCompositionRequest) {
    request
        .source_units
        .push(actual_material_evaluators(&request));
    ShaderComposer::new()
        .compose(&with_forward_specular(request))
        .expect("the transformed real Forward builder must validate through Naga");
}

fn with_forward_specular(request: ShaderCompositionRequest) -> ShaderCompositionRequest {
    crate::pbr::specular::forward_specular_request(request)
}

#[test]
fn oracle_uses_selected_babylon_linear_roughness_lod_and_height_correlated_channels() {
    let result = oracle(
        [0.8, 0.2, 0.1],
        0.5,
        0.25,
        0.75,
        [0.04; 3],
        [0.25, 0.5],
        8.0,
        0.75,
        0.5,
    );
    assert!((result.lod - 2.0).abs() < 1e-6);
    assert_eq!(result.brdf_uv, [0.75, 0.25]);
    for (actual, expected) in result.f0.into_iter().zip([0.42, 0.12, 0.07]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert!((result.reflectance[0] - 0.355).abs() < 1e-6);
    assert!((result.reflectance[1] - 0.28).abs() < 1e-6);
    assert!((result.reflectance[2] - 0.2675).abs() < 1e-6);
    assert_eq!(result.diffuse_energy, [0.62; 3]);
}

#[test]
fn deferred_reconstruction_maps_top_left_pixel_centers_to_clip_space() {
    assert_eq!(
        deferred_pixel_center_to_ndc([0.5, 0.5], [4.0, 4.0]),
        [-0.75, 0.75]
    );
    assert_eq!(
        deferred_pixel_center_to_ndc([3.5, 3.5], [4.0, 4.0]),
        [0.75, -0.75]
    );
}

#[test]
fn source_composes_with_the_declared_group_two_contract() {
    let entry = r#"
struct Vertex { @builtin(position) position: vec4<f32> };
@vertex fn vs(@builtin(vertex_index) index: u32) -> Vertex {
    var output: Vertex;
    let positions = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    output.position = vec4(positions[index], 0.0, 1.0);
    return output;
}
@fragment fn fs() -> @location(0) vec4<f32> {
    let result = pbr_specular_ibl_bound(
        vec3(0.0), vec3(0.0, 0.0, 1.0), vec3(0.8, 0.2, 0.1), 0.5, 0.25,
        vec3(0.04));
    return vec4(result.radiance + result.diffuse_energy * 0.0, 1.0);
}
"#;
    let request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("pbr-normal", crate::pbr::templates::NORMAL),
        SourceUnit::new("pbr-specular-entry", entry),
    ])
    .with_entry_point(EntryPoint::new("vs", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new("fs", ShaderStage::Fragment));
    let artifact = ShaderComposer::new()
        .compose(&with_specular_ibl(request))
        .expect("specular IBL WGSL must compose and validate through Naga");
    assert!(artifact.source().contains("textureSampleLevel"));
    assert!(WGSL.contains("vec2<f32>(clamp(ndotv"));
}

#[test]
fn deferred_route_remaps_specular_bindings_without_a_group_one_hole() {
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("pbr-normal", crate::pbr::templates::NORMAL),
        SourceUnit::new("pbr-deferred-lighting", crate::pbr::templates::LIGHTING),
    ])
    .with_entry_point(EntryPoint::new("deferred_lighting_vs", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new(
        "deferred_lighting_fs",
        ShaderStage::Fragment,
    ));
    for binding in 0..3 {
        request = request.with_binding(
            BindingDeclaration::new(0, binding, BindingKind::Texture)
                .with_visibility([ShaderStage::Fragment]),
        );
    }
    let artifact = ShaderComposer::new()
        .compose(&deferred_specular_request(request))
        .expect("deferred specular route must validate through Naga");
    assert!(artifact.source().contains("@group(1) @binding(6)"));
}

#[test]
fn deferred_specular_transform_replaces_each_real_lighting_entry_once() {
    for request in [
        crate::pbr::lighting_request(),
        crate::pbr::lighting_irradiance_request(),
        crate::pbr::lighting_effect_request(),
        crate::pbr::lighting_effect_irradiance_request(),
    ] {
        let transformed = deferred_specular_request(request);
        assert_eq!(
            transformed
                .source_units
                .iter()
                .filter(|unit| unit.source.contains("fn deferred_lighting_fs"))
                .count(),
            1,
            "the selected Deferred entry is replaced rather than duplicated",
        );
        let artifact = ShaderComposer::new()
            .compose(&transformed)
            .expect("the transformed real Deferred builder must validate through Naga");
        assert!(artifact.source().contains("@group(1) @binding(6)"));
    }
}

#[test]
fn forward_specular_transform_composes_real_material_contexts_and_environment_variants() {
    let position_normal = position_normal_layout();
    let uv0 = uv0_layout();
    let uv0_normal = uv0_normal_layout();
    for request in [
        crate::pbr::forward_request(&position_normal),
        crate::pbr::forward_irradiance_request(&position_normal),
        crate::pbr::forward_uv0_request(&uv0),
        crate::pbr::forward_uv0_irradiance_request(&uv0),
        crate::pbr::forward_uv0_normal_request(&uv0_normal),
        crate::pbr::forward_uv0_normal_irradiance_request(&uv0_normal),
        crate::pbr::forward_uv0_normal_effect_request(&uv0_normal),
        crate::pbr::forward_uv0_normal_effect_irradiance_request(&uv0_normal),
    ] {
        let request = request.expect("the real material layout is supported");
        compose_actual_forward(request);
    }
}
