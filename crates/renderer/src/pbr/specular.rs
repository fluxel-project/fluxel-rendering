//! Shared minimum specular-IBL shader contract.
//!
//! Babylon's `LINEARSPECULARREFLECTION` branch maps roughness using the actual
//! base-level texture width, then applies the authored convolution scale and
//! offset. The available mip range limits the resulting sample.

use crate::preparation::forward_fragment::{
    ForwardFragmentInterface, with_ordinary_forward_fragment,
};
use fluxel_shader::{
    BindingDeclaration, BindingKind, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

#[path = "specular/forward_standard.rs"]
mod forward_standard;
#[path = "specular/transmission.rs"]
mod transmission;

/// The additional group-two bindings for the first specular-IBL product.
///
/// Group two bindings zero and one remain the existing diffuse irradiance
/// texture and sampler. `world_from_clip` is retained for the shared
/// camera ABI even though this fragment helper consumes the already supplied
/// world position; a deferred caller may use it when reconstructing that
/// position before calling the helper.
pub(crate) const WGSL: &str = r#"
struct PbrSpecularIblCamera {
    // ABI shared by Forward and Deferred.
    world_from_clip: mat4x4<f32>,
    // xyz is a perspective eye when w=1, or a normalized direction toward
    // the eye for orthographic rendering when w=0.
    camera: vec4<f32>,
    // x/y are target extent; z is the validated radiance maximum mip LOD.
    target_size_and_mips: vec4<f32>,
    // Babylon convolution scale/offset are x/y.
    lod_params: vec4<f32>,
    light_direction_and_roughness: vec4<f32>,
    light_radiance: vec4<f32>,
};
struct PbrSpecularIblResult {
    radiance: vec3<f32>,
    // Babylon's diffuse conservation uses base (non-coloured) specular
    // reflectance, not the coloured radiance reflectance.
    diffuse_energy: vec3<f32>,
    specular_energy: vec3<f32>,
    refraction_energy: f32,
};

@group(2) @binding(2) var pbr_specular_radiance: texture_2d<f32>;
@group(2) @binding(3) var pbr_specular_radiance_sampler: sampler;
@group(2) @binding(4) var pbr_specular_brdf_lut: texture_2d<f32>;
@group(2) @binding(5) var pbr_specular_brdf_lut_sampler: sampler;
@group(2) @binding(6) var<uniform> pbr_specular_camera: PbrSpecularIblCamera;

fn pbr_specular_view_direction(
    world_position: vec3<f32>,
    camera_position: vec3<f32>,
    orthographic_view_direction: vec3<f32>,
    orthographic: bool,
) -> vec3<f32> {
    let perspective = pbr_normal(camera_position - world_position);
    let orthographic_direction = pbr_normal(orthographic_view_direction);
    return select(perspective, orthographic_direction, orthographic);
}

fn pbr_specular_latlong_uv(direction: vec3<f32>) -> vec2<f32> {
    let unit = pbr_normal(direction);
    return vec2<f32>(fract(atan2(unit.z, unit.x) / 6.283185307179586 + 0.5),
                      acos(clamp(unit.y, -1.0, 1.0)) / 3.141592653589793);
}

fn pbr_specular_linear_lod(
    roughness: f32,
    radiance: texture_2d<f32>,
    max_lod: f32,
    lod_scale: f32,
    lod_offset: f32,
) -> f32 {
    // Babylon binds the sampled texture's width separately from mip availability.
    let base_width = f32(textureDimensions(radiance, 0).x);
    let babylon_lod = log2(base_width) * roughness;
    return clamp(babylon_lod * lod_scale + lod_offset, 0.0, max_lod);
}

fn pbr_specular_brdf_uv(ndotv: f32, perceptual_roughness: f32) -> vec2<f32> {
    // Babylon pbrBRDFFunctions indexes (cos(theta), roughness), with no Y flip.
    return vec2<f32>(clamp(ndotv, 0.0, 1.0), clamp(perceptual_roughness, 0.0, 1.0));
}

fn pbr_specular_height_correlated_reflectance(
    f0: vec3<f32>,
    f90: vec3<f32>,
    lut: vec2<f32>,
) -> vec3<f32> {
    // Babylon BRDF_V_HEIGHT_CORRELATED: x weights (F90 - F0), y weights F0.
    return (f90 - f0) * lut.x + f0 * lut.y;
}

fn pbr_specular_metallic_roughness_f0(
    base_color: vec3<f32>,
    metallic: f32,
    dielectric_f0: vec3<f32>,
) -> vec3<f32> {
    return mix(clamp(dielectric_f0, vec3<f32>(0.0), vec3<f32>(1.0)),
               max(base_color, vec3<f32>(0.0)),
               vec3<f32>(clamp(metallic, 0.0, 1.0)));
}

fn pbr_specular_ibl(
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    ambient_occlusion: f32,
    base_color: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    camera_position: vec3<f32>,
    orthographic_view_direction: vec3<f32>,
    orthographic: bool,
    radiance: texture_2d<f32>,
    radiance_sampler: sampler,
    brdf_lut: texture_2d<f32>,
    brdf_lut_sampler: sampler,
    lod_scale: f32,
    lod_offset: f32,
    max_lod: f32,
    dielectric_f0: vec3<f32>,
) -> PbrSpecularIblResult {
    let normal = pbr_normal(world_normal);
    let view = pbr_specular_view_direction(
        world_position, camera_position, orthographic_view_direction, orthographic);
    let ndotv_unclamped = dot(normal, view);
    let ndotv = abs(ndotv_unclamped) + 0.0000001;
    let f0 = pbr_specular_metallic_roughness_f0(base_color, metallic, dielectric_f0);
    let f90 = vec3<f32>(1.0);
    let reflection = reflect(-view, normal);
    let lod = pbr_specular_linear_lod(
        clamp(perceptual_roughness, 0.0, 1.0), radiance, max_lod, lod_scale, lod_offset);
    let environment = textureSampleLevel(
        radiance, radiance_sampler, pbr_specular_latlong_uv(reflection), lod).rgb;
    let lut = textureSample(brdf_lut, brdf_lut_sampler,
        pbr_specular_brdf_uv(ndotv, perceptual_roughness)).rg;
    // Babylon environmentRadianceOcclusion and the RADIANCEOCCLUSION
    // branch in pbrBlockReflectance. Use the signed NdotV from geometry.
    let occlusion_input = ndotv_unclamped + ambient_occlusion;
    let radiance_occlusion = clamp(occlusion_input * occlusion_input - 1.0 + ambient_occlusion, 0.0, 1.0);
    let reflectance = pbr_specular_height_correlated_reflectance(f0, f90, lut)
        * radiance_occlusion;
    // Babylon getEnergyConservationFactor. A zero Y coefficient carries no
    // normalization information, so keep unit energy for that authored LUT sample.
    let reciprocal_y = 1.0 / select(1.0, lut.y, lut.y > 0.0);
    let specular_energy = vec3<f32>(1.0) + f0 * (reciprocal_y - 1.0);
    let max_reflectance = max(reflectance.x, max(reflectance.y, reflectance.z));
    return PbrSpecularIblResult(
        environment * reflectance * specular_energy,
        vec3<f32>(1.0) - dielectric_f0,
        specular_energy,
        1.0 - max_reflectance,
    );
}

fn pbr_specular_ibl_bound(
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    ambient_occlusion: f32,
    base_color: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    dielectric_f0: vec3<f32>,
) -> PbrSpecularIblResult {
    return pbr_specular_ibl(
        world_position, world_normal, ambient_occlusion, base_color, metallic, perceptual_roughness,
        pbr_specular_camera.camera.xyz,
        pbr_specular_camera.camera.xyz,
        pbr_specular_camera.camera.w == 0.0,
        pbr_specular_radiance, pbr_specular_radiance_sampler,
        pbr_specular_brdf_lut, pbr_specular_brdf_lut_sampler,
        pbr_specular_camera.lod_params.x, pbr_specular_camera.lod_params.y,
        pbr_specular_camera.target_size_and_mips.z, dielectric_f0);
}
"#;

const FORWARD_DIFFUSE_BINDINGS: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;
"#;

/// Adds the shared source and its fragment-visible group-two contract.
pub(crate) fn with_specular_ibl(mut request: ShaderCompositionRequest) -> ShaderCompositionRequest {
    request
        .source_units
        .push(SourceUnit::new("pbr-specular-ibl", WGSL));
    for (binding, kind) in [
        (2, BindingKind::Texture),
        (3, BindingKind::Sampler),
        (4, BindingKind::Texture),
        (5, BindingKind::Sampler),
        (6, BindingKind::Uniform),
    ] {
        request = request.with_binding(
            BindingDeclaration::new(2, binding, kind).with_visibility([ShaderStage::Fragment]),
        );
    }
    request.with_static_option(
        "renderer.specular_ibl",
        "latlong-linear-roughness-lod-height-correlated-v1",
    )
}

/// Replaces one existing Forward PBR body with an actual view-dependent
/// specular-IBL consumer. The request must originate from one of the four
/// standard metallic/roughness Forward builders.
pub(crate) fn forward_specular_request(
    request: ShaderCompositionRequest,
) -> ShaderCompositionRequest {
    forward_specular_request_inner(request, false, None)
}

/// Selects environment refraction from the same prefiltered radiance chain as
/// specular IBL. It intentionally has no scene-color or capture binding.
pub(crate) fn forward_transmission_request(
    request: ShaderCompositionRequest,
) -> ShaderCompositionRequest {
    forward_specular_request_inner(request, true, None)
}

pub(crate) fn forward_scene_transmission_request(
    request: ShaderCompositionRequest,
    texture: &str,
    sampler: &str,
) -> ShaderCompositionRequest {
    forward_specular_request_inner(request, true, Some((texture, sampler)))
}

fn forward_specular_request_inner(
    mut request: ShaderCompositionRequest,
    transmission: bool,
    scene_color: Option<(&str, &str)>,
) -> ShaderCompositionRequest {
    let diffuse_irradiance = has_static_option(
        &request,
        "renderer.environment",
        "latlong-diffuse-irradiance-v1",
    );
    let effects = has_static_option(
        &request,
        "renderer.surface_effects",
        "emissive-ao-rgba16float-v1",
    );
    let uv0 = has_vertex_semantic(&request, "UV0");
    let normal_map = uv0 && has_vertex_semantic(&request, "TANGENT");
    let source = if uv0 {
        forward_source(
            normal_map,
            effects,
            diffuse_irradiance,
            transmission,
            scene_color,
        )
    } else {
        forward_standard::source(diffuse_irradiance, transmission)
    };
    // Only renderer-owned bodies are replaceable; Material source is never
    // inspected or rewritten.
    request
        .source_units
        .retain(|unit| !unit.id.0.starts_with("pbr-forward"));
    request
        .source_units
        .push(SourceUnit::new("pbr-forward-specular", source));
    if diffuse_irradiance {
        if !request
            .source_units
            .iter()
            .any(|unit| unit.id.0 == "pbr-diffuse-irradiance")
        {
            request.source_units.push(SourceUnit::new(
                "pbr-diffuse-irradiance",
                super::templates::DIFFUSE_IRRADIANCE,
            ));
        }
        request.source_units.push(SourceUnit::new(
            "pbr-specular-diffuse-bindings",
            FORWARD_DIFFUSE_BINDINGS,
        ));
        for (binding, kind) in [(0, BindingKind::Texture), (1, BindingKind::Sampler)] {
            if !request
                .bindings
                .iter()
                .any(|declared| declared.group == 2 && declared.binding == binding)
            {
                request = request.with_binding(
                    BindingDeclaration::new(2, binding, kind)
                        .with_visibility([ShaderStage::Fragment]),
                );
            }
        }
    }
    let mut request = with_specular_ibl(request);
    if let Some(binding) = request
        .bindings
        .iter_mut()
        .find(|binding| binding.group == 1 && binding.binding == 0)
    {
        *binding = binding.clone().with_visibility([ShaderStage::Vertex, ShaderStage::Fragment]);
    }
    let request = if transmission {
        request.source_units.push(SourceUnit::new(
            "pbr-specular-transmission",
            transmission::TRANSMISSION,
        ));
        if scene_color.is_some() {
            request
                .source_units
                .retain(|unit| unit.id.0 != "pbr-transforms");
            request.source_units.push(SourceUnit::new(
                "pbr-scene-refraction-transforms",
                super::templates::SCENE_REFRACTION_TRANSFORMS,
            ));
            request.source_units.push(SourceUnit::new(
                "pbr-scene-transmission",
                super::scene_refraction::WGSL,
            ));
            request.with_static_option("renderer.transmission", "scene-color-projected-v1")
        } else {
            request.with_static_option(
                "renderer.transmission",
                "environment-refraction-beer-lambert-v1",
            )
        }
    } else {
        request
    };
    with_ordinary_forward_fragment(request, ForwardFragmentInterface::PbrSpecular)
}

/// Replaces the actual Deferred lighting body. Deferred keeps its established
/// group-one diffuse bindings at zero/one; the shared specular contract is
/// remapped from Forward group two to group one bindings two through six.
pub(crate) fn deferred_specular_request(
    mut request: ShaderCompositionRequest,
) -> ShaderCompositionRequest {
    let effects = has_static_option(
        &request,
        "renderer.surface_effects",
        "emissive-ao-rgba16float-v1",
    );
    let diffuse = has_static_option(
        &request,
        "renderer.environment",
        "latlong-diffuse-irradiance-v1",
    );
    // The plain route still contributes the reusable diffuse helper. Only
    // this entry source owns deferred_lighting_fs and must be replaced.
    request.source_units.retain(|unit| {
        !unit.source.contains("fn deferred_lighting_fs") && unit.id.0 != "pbr-basic-light-view"
    });
    request.bindings.retain(|binding| {
        binding.group != 0 || binding.binding != super::basic_deferred_view_binding(effects)
    });
    request.source_units.push(SourceUnit::new(
        "pbr-deferred-lighting-specular",
        deferred_source(effects, diffuse),
    ));
    let remapped = WGSL.replace("@group(2)", "@group(1)");
    request
        .source_units
        .push(SourceUnit::new("pbr-specular-ibl-deferred", remapped));
    for (binding, kind) in [
        (2, BindingKind::Texture),
        (3, BindingKind::Sampler),
        (4, BindingKind::Texture),
        (5, BindingKind::Sampler),
        (6, BindingKind::Uniform),
    ] {
        request = request.with_binding(
            BindingDeclaration::new(1, binding, kind).with_visibility([ShaderStage::Fragment]),
        );
    }
    request.with_static_option(
        "renderer.specular_ibl",
        "latlong-linear-roughness-lod-height-correlated-v1",
    )
}

fn has_static_option(request: &ShaderCompositionRequest, key: &str, value: &str) -> bool {
    request
        .static_options
        .get(key)
        .is_some_and(|actual| actual == value)
}

fn has_vertex_semantic(request: &ShaderCompositionRequest, semantic: &str) -> bool {
    request
        .vertex_inputs
        .iter()
        .any(|input| input.semantic == semantic)
}

fn deferred_source(effects: bool, diffuse: bool) -> String {
    let effect_binding = if effects {
        "@group(0) @binding(3) var pbr_emissive_ao: texture_2d<f32>;"
    } else {
        ""
    };
    let diffuse_binding = if diffuse {
        "@group(1) @binding(0) var pbr_environment_irradiance: texture_2d<f32>; @group(1) @binding(1) var pbr_environment_sampler: sampler;"
    } else {
        ""
    };
    let effects_source = if effects {
        "textureLoad(pbr_emissive_ao, coordinate, 0)"
    } else {
        "vec4(0.0, 0.0, 0.0, 1.0)"
    };
    let indirect = if diffuse {
        " + pbr_diffuse_irradiance(base_metal.rgb, base_metal.a, normal, pbr_environment_irradiance, pbr_environment_sampler, spec.diffuse_energy) * surface_effects.a"
    } else {
        ""
    };
    format!(
        r#"
@group(0) @binding(0) var pbr_base_metal: texture_2d<f32>;
@group(0) @binding(1) var pbr_normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var pbr_depth: texture_depth_2d;
{effect_binding}
{diffuse_binding}
fn pbr_directional_light() -> PbrDirectionalLight {{
    return PbrDirectionalLight(pbr_specular_camera.light_direction_and_roughness, pbr_specular_camera.light_radiance);
}}
@vertex fn deferred_lighting_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {{
    let positions = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(positions[index], 0.0, 1.0);
}}
@fragment fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{
    let coordinate = vec2<i32>(position.xy);
    let nr = textureLoad(pbr_normal_roughness, coordinate, 0);
    if !pbr_gbuffer_is_covered(nr.xyz) {{ return vec4(0.0); }}
    let depth = textureLoad(pbr_depth, coordinate, 0);
    let world = pbr_world_position_from_depth(
        position.xy, depth, pbr_specular_camera.target_size_and_mips.xy,
        pbr_specular_camera.world_from_clip,
    );
    let base_metal = textureLoad(pbr_base_metal, coordinate, 0);
    let normal = nr.xyz * 2.0 - vec3(1.0);
    let view = pbr_specular_view_direction(world, pbr_specular_camera.camera.xyz,
        pbr_specular_camera.camera.xyz, pbr_specular_camera.camera.w == 0.0);
    let surface_effects = {effects_source};
    let spec = pbr_specular_ibl_bound(world, normal, surface_effects.a, base_metal.rgb, base_metal.a, nr.a, vec3(0.04));
    let terms = pbr_direct_light_components(base_metal.rgb, vec2(base_metal.a, nr.a), normal, view, vec3(0.04));
    let direct = terms.specular * spec.specular_energy + terms.diffuse;
    return vec4(direct + spec.radiance{indirect} + surface_effects.rgb, 1.0);
}}
"#
    )
}

fn forward_source(
    normal_map: bool,
    effects: bool,
    diffuse: bool,
    transmission: bool,
    scene_color: Option<(&str, &str)>,
) -> String {
    let input = if normal_map {
        r#"
struct PbrSpecularVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) tangent_world: vec3<f32>,
    @location(2) bitangent_world: vec3<f32>,
    @location(3) normal_world: vec3<f32>,
    @location(4) uv0: vec2<f32>,
};
@vertex fn forward_vs(
    @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>,
) -> PbrSpecularVertexOutput {
    var output: PbrSpecularVertexOutput;
    output.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0);
    output.world_position = (pbr_transforms.world_from_local * vec4<f32>(position, 1.0)).xyz;
    let basis = pbr_vertex_tangent_basis(normal, tangent);
    output.tangent_world = basis[0];
    output.bitangent_world = basis[1];
    output.normal_world = basis[2];
    output.uv0 = uv0;
    return output;
}
"#
    } else {
        r#"
struct PbrSpecularVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
};
@vertex fn forward_vs(
    @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
) -> PbrSpecularVertexOutput {
    var output: PbrSpecularVertexOutput;
    output.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0);
    output.world_position = (pbr_transforms.world_from_local * vec4<f32>(position, 1.0)).xyz;
    output.world_normal = (pbr_transforms.normal_from_local * vec4<f32>(normal, 0.0)).xyz;
    output.uv0 = uv0;
    return output;
}
"#
    };
    let normal = if normal_map {
        "pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0))"
    } else {
        "input.world_normal"
    };
    let base = "material.rgb";
    let dielectric_f0 = if transmission {
        "vec3<f32>(pbr_surface_dielectric_f0(material_surface_ior()))"
    } else {
        "vec3<f32>(0.04)"
    };
    let transmission_values = "let transmission_intensity = evaluate_transmission_uv0(input.uv0);\n    let transmission_thickness = evaluate_thickness_uv0(input.uv0);";
    let transmission_values = if transmission {
        transmission_values
    } else {
        "let transmission_intensity = 0.0;\n    let transmission_thickness = 0.0;"
    };
    let diffuse_weight = if transmission {
        " * (1.0 - transmission_intensity)"
    } else {
        ""
    };
    let direct = format!(
        "let terms = pbr_direct_light_components(base, properties, normal, view, dielectric_f0);\n    let direct = terms.specular * spec.specular_energy + terms.diffuse{diffuse_weight};"
    );
    let ambient_occlusion = if effects {
        "clamp(evaluate_occlusion_uv0(input.uv0), 0.0, 1.0)"
    } else {
        "1.0"
    };
    let indirect = if diffuse {
        format!(
            "let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, spec.diffuse_energy) * ambient_occlusion{diffuse_weight};"
        )
    } else {
        String::from("let indirect = vec3<f32>(0.0);")
    };
    let emissive = if effects {
        "let emissive = max(evaluate_emissive_uv0(input.uv0), vec3<f32>(0.0));"
    } else {
        "let emissive = vec3<f32>(0.0);"
    };
    let refraction = if let Some((texture, sampler)) = scene_color {
        format!(
            "let refraction = pbr_scene_transmission({texture}, {sampler}, input.world_position, normal, view, transmission_intensity, transmission_thickness) * spec.refraction_energy;"
        )
    } else if transmission {
        String::from(
            "let refraction = pbr_environment_transmission(normal, view, properties.y, transmission_intensity, transmission_thickness) * spec.refraction_energy;",
        )
    } else {
        String::from("let refraction = vec3<f32>(0.0);")
    };
    format!(
        r#"{input}
fn renderer_forward_shade(input: PbrSpecularVertexOutput, front_facing: bool) -> vec4<f32> {{
    let material = evaluate_material_uv0(input.uv0);
    let base = {base};
    let properties = evaluate_metallic_roughness_uv0(input.uv0);
    let normal = pbr_surface_normal({normal}, front_facing);
    let dielectric_f0 = {dielectric_f0};
    {transmission_values}
    let view = pbr_specular_view_direction(input.world_position,
        pbr_specular_camera.camera.xyz, pbr_specular_camera.camera.xyz,
        pbr_specular_camera.camera.w == 0.0);
    let ambient_occlusion = {ambient_occlusion};
    let spec = pbr_specular_ibl_bound(input.world_position, normal, ambient_occlusion, base,
        properties.x, properties.y, dielectric_f0);
    {direct}
    {indirect}
    {refraction}
    {emissive}
    if material_alpha_rejected(material.a) {{ discard; }}
    return vec4<f32>(direct + spec.radiance + indirect + refraction + emissive, material_output_alpha(material.a));
}}
"#
    )
}

#[cfg(test)]
#[path = "../tests/pbr/specular.rs"]
mod tests;
