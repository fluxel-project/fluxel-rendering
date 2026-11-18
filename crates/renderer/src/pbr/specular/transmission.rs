//! Surface transmission WGSL added only to the environment-refraction route.

pub(super) const TRANSMISSION: &str = r#"
fn pbr_surface_dielectric_f0(ior: f32) -> f32 {
    let relative = (max(ior, 1.0) - 1.0) / (max(ior, 1.0) + 1.0);
    return relative * relative;
}

fn pbr_stable_length(value: vec3<f32>) -> f32 {
    let maximum = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if maximum == 0.0 { return 0.0; }
    return length(value / maximum) * maximum;
}

fn pbr_world_max_scale() -> f32 {
    return max(
        pbr_stable_length(pbr_transforms.world_from_local[0].xyz),
        max(
            pbr_stable_length(pbr_transforms.world_from_local[1].xyz),
            pbr_stable_length(pbr_transforms.world_from_local[2].xyz),
        ),
    );
}

fn pbr_environment_transmission(
    normal: vec3<f32>,
    view: vec3<f32>,
    roughness: f32,
    intensity: f32,
    thickness: f32,
) -> vec3<f32> {
    let volume_ior = material_volume_ior();
    let refracted = refract(-view, normal, 1.0 / volume_ior);
    // Babylon uploads reciprocal surface/volume IOR. SS_HAS_THICKNESS selects
    // the volume ratio for roughness; refraction direction always uses volume.
    let roughness_ior = select(material_surface_ior(), volume_ior, material_refraction_has_thickness());
    let eta = 1.0 / roughness_ior;
    let perceptual_roughness = clamp(roughness, 0.0, 1.0);
    let alpha_g = perceptual_roughness * perceptual_roughness + 0.0005;
    let ior_roughness = mix(
        alpha_g,
        0.0,
        clamp(eta * 3.0 - 2.0, 0.0, 1.0),
    );
    let lod = pbr_specular_linear_lod(
        ior_roughness,
        pbr_specular_radiance,
        pbr_specular_camera.target_size_and_mips.z,
        pbr_specular_camera.lod_params.x,
        pbr_specular_camera.lod_params.y,
    );
    let environment = textureSampleLevel(
        pbr_specular_radiance,
        pbr_specular_radiance_sampler,
        pbr_specular_latlong_uv(pbr_normal(refracted)),
        lod,
    ).rgb;
    let sigma = -log(material_attenuation_color())
        / max(material_attenuation_distance(), 1e-5);
    let attenuation = exp(-sigma * thickness * pbr_world_max_scale());
    return environment * attenuation * intensity;
}
"#;
