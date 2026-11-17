//! Surface transmission WGSL added only to the environment-refraction route.

pub(super) const DIRECT_LIGHT: &str = r#"
struct PbrDirectLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn pbr_direct_light_components(
    base_input: vec3<f32>, properties: vec2<f32>, world_normal: vec3<f32>, view: vec3<f32>, dielectric_f0: vec3<f32>,
) -> PbrDirectLight {
    let base = clamp(base_input, vec3<f32>(0.0), vec3<f32>(1.0));
    let metal = clamp(properties.x, 0.0, 1.0);
    let roughness = clamp(properties.y, 0.04, 1.0);
    let normal = pbr_normal(world_normal);
    let light = vec3<f32>(0.6, 0.0, 0.8);
    let half_direction = normalize(light + view);
    let no_l = clamp(dot(normal, light), 0.0, 1.0);
    let no_v = clamp(dot(normal, view), 0.0, 1.0);
    if no_l <= 0.0 || no_v <= 0.0 { return PbrDirectLight(vec3<f32>(0.0), vec3<f32>(0.0)); }
    let no_h = clamp(dot(normal, half_direction), 0.0, 1.0);
    let vo_h = clamp(dot(view, half_direction), 0.0, 1.0);
    let alpha = roughness * roughness;
    let alpha_squared = alpha * alpha;
    let distribution_term = no_h * no_h * (alpha_squared - 1.0) + 1.0;
    let distribution = alpha_squared / (3.141592653589793 * distribution_term * distribution_term);
    let visibility = 0.5 / (
        no_l * sqrt(no_v * no_v * (1.0 - alpha_squared) + alpha_squared) +
        no_v * sqrt(no_l * no_l * (1.0 - alpha_squared) + alpha_squared));
    let f0 = mix(dielectric_f0, base, vec3<f32>(metal));
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vo_h, 5.0);
    let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metal) * base / 3.141592653589793;
    return PbrDirectLight(
        diffuse * no_l * 3.0,
        distribution * visibility * fresnel * no_l * 3.0,
    );
}
fn pbr_direct_light_view(
    base_input: vec3<f32>, properties: vec2<f32>, world_normal: vec3<f32>, view: vec3<f32>, dielectric_f0: vec3<f32>,
) -> vec3<f32> {
    let terms = pbr_direct_light_components(base_input, properties, world_normal, view, dielectric_f0);
    return terms.diffuse + terms.specular;
}
"#;

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
        pbr_specular_camera.target_size_and_mips.z,
        pbr_specular_camera.lod_params.x,
        pbr_specular_camera.lod_params.y,
    );
    let environment = textureSampleLevel(
        pbr_specular_radiance,
        pbr_specular_radiance_sampler,
        pbr_specular_latlong_uv(refracted),
        lod,
    ).rgb;
    let sigma = -log(material_attenuation_color())
        / max(material_attenuation_distance(), 1e-5);
    let attenuation = exp(-sigma * thickness * pbr_world_max_scale());
    return environment * attenuation * intensity;
}
"#;
