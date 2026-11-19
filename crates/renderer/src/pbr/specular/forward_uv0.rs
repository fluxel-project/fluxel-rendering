//! Forward UV0 shading shared by specular IBL and material refraction.

use super::ForwardRefraction;

pub(super) fn source(
    normal_map: bool,
    effects: bool,
    diffuse: bool,
    refraction_source: Option<ForwardRefraction<'_>>,
    specular: bool,
) -> String {
    let transmission = refraction_source.is_some();
    let input = if normal_map {
        r#"
struct PbrSpecularVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) tangent_world: vec3<f32>,
    @location(2) bitangent_world: vec3<f32>,
    @location(3) normal_world: vec3<f32>,
    @location(4) uv0: vec2<f32>,
    @location(5) geometric_world_normal: vec3<f32>,
    @location(6) @interpolate(flat) camera_position: vec3<f32>,
    @location(7) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-8 */
};
@vertex fn forward_vs(
    @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32> /* instance_parameters */ /* fluxel:uv1-input */,
) -> PbrSpecularVertexOutput {
    /* instance_setup */
    var output: PbrSpecularVertexOutput;
    output.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0);
    output.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz;
    let basis = pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */);
    output.tangent_world = basis[0];
    output.bitangent_world = basis[1];
    output.normal_world = basis[2];
    output.geometric_world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz;
    output.camera_position = pbr_transforms.surface_camera_position.xyz;
    output.uv0 = uv0;
    /* fluxel:vertex-varying-write-output */output.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0);
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
    @location(3) @interpolate(flat) camera_position: vec3<f32>,
    @location(4) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-5 */
};
@vertex fn forward_vs(
    @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32> /* instance_parameters */ /* fluxel:uv1-input */,
) -> PbrSpecularVertexOutput {
    /* instance_setup */
    var output: PbrSpecularVertexOutput;
    output.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0);
    output.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz;
    output.world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz;
    output.camera_position = pbr_transforms.surface_camera_position.xyz;
    output.uv0 = uv0;
    /* fluxel:vertex-varying-write-output */output.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0);
    return output;
}
"#
    };
    let normal = if normal_map {
        "pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0))"
    } else {
        "select(input.world_normal, surface.world_normal, surface.has_world_normal)"
    };
    let geometric_normal = if normal_map {
        "pbr_normal(input.geometric_world_normal)"
    } else {
        "pbr_normal(input.world_normal)"
    };
    let base = "material.rgb";
    let dielectric_f0 = if transmission {
        "vec3<f32>(pbr_surface_dielectric_f0(material_surface_ior()))"
    } else {
        "vec3<f32>(0.04)"
    };
    let transmission_values = "let transmission_intensity = evaluate_transmission_uv0(/* fluxel:uv1-argument */input.uv0);\n    let transmission_thickness = evaluate_thickness_uv0(/* fluxel:uv1-argument */input.uv0);";
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
    let specular_energy = if specular {
        "spec.specular_energy"
    } else {
        "vec3<f32>(1.0)"
    };
    let diffuse_energy = if specular {
        "spec.diffuse_energy"
    } else {
        "(vec3<f32>(1.0) - dielectric_f0)"
    };
    let direct = format!(
        "let terms = pbr_direct_light_components(base, properties, normal, view, dielectric_f0);\n    let direct = terms.specular * {specular_energy} + terms.diffuse{diffuse_weight};"
    );
    let ambient_occlusion = if effects {
        "clamp(surface.occlusion * evaluate_occlusion_uv0(/* fluxel:uv1-argument */input.uv0), 0.0, 1.0)"
    } else {
        "surface.occlusion"
    };
    let indirect = if diffuse {
        format!(
            "let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, {diffuse_energy}) * ambient_occlusion{diffuse_weight};"
        )
    } else {
        String::from("let indirect = vec3<f32>(0.0);")
    };
    let emissive = if effects {
        "let emissive = pbr_emissive(/* fluxel:uv1-argument */input.uv0);"
    } else {
        "let emissive = vec3<f32>(0.0);"
    };
    let (view, specular_terms, radiance, refraction_energy) = if specular {
        (
            "pbr_specular_view_direction(input.world_position, pbr_specular_camera.camera.xyz, pbr_specular_camera.camera.xyz, pbr_specular_camera.camera.w == 0.0)",
            "let spec = pbr_specular_ibl_bound(input.world_position, normal, ambient_occlusion, base, properties.x, properties.y, dielectric_f0);",
            "spec.radiance",
            "spec.refraction_energy",
        )
    } else {
        (
            "pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera)",
            "",
            "vec3<f32>(0.0)",
            "pbr_analytic_refraction_energy(base, properties, normal, view, dielectric_f0)",
        )
    };
    let refraction = match refraction_source {
        Some(ForwardRefraction::Scene(texture, sampler)) => format!(
            "let refraction = pbr_scene_transmission({texture}, {sampler}, input.world_position, normal, view, properties.y, transmission_intensity, transmission_thickness) * {refraction_energy};"
        ),
        Some(ForwardRefraction::Environment) => format!(
            "let refraction = pbr_environment_transmission(normal, view, properties.y, transmission_intensity, transmission_thickness) * {refraction_energy};"
        ),
        Some(ForwardRefraction::Material(_)) => format!(
            "let refraction = pbr_material_transmission(normal, view, properties.y, transmission_intensity, transmission_thickness) * {refraction_energy};"
        ),
        None => String::from("let refraction = vec3<f32>(0.0);"),
    };
    format!(
        r#"{input}
fn renderer_forward_shade(input: PbrSpecularVertexOutput, front_facing: bool) -> vec4<f32> {{
    let surface = evaluate_pbr_surface_uv0(/* fluxel:uv1-argument */input.uv0, input.world_position, {geometric_normal}, input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let base = {base};
    let properties = surface.properties;
    let normal = pbr_surface_normal({normal}, front_facing);
    let dielectric_f0 = {dielectric_f0};
    {transmission_values}
    let view = {view};
    let ambient_occlusion = {ambient_occlusion};
    {specular_terms}
    {direct}
    {indirect}
    {refraction}
    {emissive}
    if material_alpha_rejected(material.a) {{ discard; }}
    let color = max(direct + {radiance} + indirect + refraction + emissive, vec3<f32>(0.0));
    return vec4<f32>(color, material_output_alpha(material.a));
}}
"#
    )
}
