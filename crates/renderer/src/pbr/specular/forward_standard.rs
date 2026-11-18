//! Forward specular source for PBR without UV0 normal mapping.

pub(super) fn source(diffuse: bool, transmission: bool) -> String {
    let dielectric_f0 = if transmission {
        "vec3<f32>(pbr_surface_dielectric_f0(material_surface_ior()))"
    } else {
        "vec3<f32>(0.04)"
    };
    let transmission_values = transmission.then_some(
        "let transmission_intensity = evaluate_transmission_uv0(vec2<f32>(0.0));\n    let transmission_thickness = evaluate_thickness_uv0(vec2<f32>(0.0));",
    ).unwrap_or("let transmission_intensity = 0.0;\n    let transmission_thickness = 0.0;");
    let diffuse_weight = if transmission {
        " * (1.0 - transmission_intensity)"
    } else {
        ""
    };
    let direct = format!(
        "let terms = pbr_direct_light_components(base, properties, normal, view, dielectric_f0);\n    let direct = terms.specular * spec.specular_energy + terms.diffuse{diffuse_weight};"
    );
    let indirect = if diffuse {
        format!(
            "let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler) * spec.diffuse_energy{diffuse_weight};"
        )
    } else {
        String::from("let indirect = vec3<f32>(0.0);")
    };
    let refraction = if transmission {
        "let refraction = pbr_environment_transmission(normal, view, properties.y, transmission_intensity, transmission_thickness) * spec.refraction_energy;"
    } else {
        "let refraction = vec3<f32>(0.0);"
    };
    format!(
        r#"
struct PbrSpecularVertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}};
@vertex fn forward_vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> PbrSpecularVertexOutput {{
    var output: PbrSpecularVertexOutput;
    output.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0);
    output.world_position = (pbr_transforms.world_from_local * vec4<f32>(position, 1.0)).xyz;
    output.world_normal = (pbr_transforms.normal_from_local * vec4<f32>(normal, 0.0)).xyz;
    return output;
}}
fn renderer_forward_shade(input: PbrSpecularVertexOutput, front_facing: bool) -> vec4<f32> {{
    let material = evaluate_material();
    let base = material.rgb;
    let properties = evaluate_metallic_roughness();
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    let dielectric_f0 = {dielectric_f0};
    {transmission_values}
    let view = pbr_specular_view_direction(input.world_position,
        pbr_specular_camera.camera.xyz, pbr_specular_camera.camera.xyz,
        pbr_specular_camera.camera.w == 0.0);
    let spec = pbr_specular_ibl_bound(input.world_position, normal, base,
        properties.x, properties.y, dielectric_f0);
    {direct}
    {indirect}
    {refraction}
    if material_alpha_rejected(material.a) {{ discard; }}
    return vec4<f32>(direct + spec.radiance + indirect + refraction, material_output_alpha(material.a));
}}
"#
    )
}
