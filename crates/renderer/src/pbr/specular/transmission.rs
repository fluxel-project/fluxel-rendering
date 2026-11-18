//! Surface transmission WGSL added only to the environment-refraction route.

pub(super) const TRANSMISSION_MATH: &str = r#"
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

// Babylon's refraction path first converts perceptual roughness to alphaG,
// then reduces it for IORs close to one.  Both environment and projected
// scene-color refraction use that same material-side choice; only the sampled
// texture and its LOD range differ.
fn pbr_refraction_roughness(roughness: f32) -> f32 {
    // Babylon uploads reciprocal surface/volume IOR. SS_HAS_THICKNESS selects
    // the volume ratio for roughness; refraction direction always uses volume.
    let roughness_ior = select(material_surface_ior(), material_volume_ior(), material_refraction_has_thickness());
    let eta = 1.0 / roughness_ior;
    let perceptual_roughness = clamp(roughness, 0.0, 1.0);
    let alpha_g = perceptual_roughness * perceptual_roughness + 0.0005;
    return mix(
        alpha_g,
        0.0,
        clamp(eta * 3.0 - 2.0, 0.0, 1.0),
    );
}

fn pbr_analytic_refraction_energy(
    base: vec3<f32>, properties: vec2<f32>, normal: vec3<f32>,
    view: vec3<f32>, dielectric_f0: vec3<f32>,
) -> f32 {
    // Babylon's Jones reflectance when an environment BRDF LUT is absent.
    let f0 = mix(dielectric_f0, max(base, vec3<f32>(0.0)), clamp(properties.x, 0.0, 1.0));
    let ndotv = abs(dot(normal, view)) + 0.0000001;
    let one_minus_ndotv = clamp(1.0 - ndotv, 0.0, 1.0);
    let squared = one_minus_ndotv * one_minus_ndotv;
    let weight = mix(0.25, 1.0, sqrt(1.0 - clamp(properties.y, 0.0, 1.0)));
    let reflectance = f0 + weight * (vec3<f32>(1.0) - f0) * squared * squared * one_minus_ndotv;
    return 1.0 - max(max(reflectance.x, reflectance.y), reflectance.z);
}

fn pbr_refraction_latlong_uv(direction: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(fract(atan2(direction.z, direction.x) / 6.283185307179586 + 0.5),
                      acos(clamp(direction.y, -1.0, 1.0)) / 3.141592653589793);
}

fn pbr_environment_attenuation(thickness: f32) -> vec3<f32> {
    let sigma = -log(material_attenuation_color())
        / max(material_attenuation_distance(), 1e-5);
    return exp(-sigma * thickness * pbr_world_max_scale());
}
"#;

pub(super) const ENVIRONMENT_TRANSMISSION: &str = r#"
fn pbr_environment_transmission(
    normal: vec3<f32>,
    view: vec3<f32>,
    roughness: f32,
    intensity: f32,
    thickness: f32,
) -> vec3<f32> {
    let volume_ior = material_volume_ior();
    let refracted = refract(-view, normal, 1.0 / volume_ior);
    let lod = pbr_specular_linear_lod(
        pbr_refraction_roughness(roughness),
        pbr_specular_radiance,
        pbr_specular_camera.target_size_and_mips.z,
        pbr_specular_camera.lod_params.x,
        pbr_specular_camera.lod_params.y,
    );
    let environment = textureSampleLevel(
        pbr_specular_radiance,
        pbr_specular_radiance_sampler,
        pbr_refraction_latlong_uv((pbr_specular_camera.reflection_matrix
            * vec4<f32>(pbr_normal(refracted), 0.0)).xyz),
        lod,
    ).rgb;
    return environment * pbr_environment_attenuation(thickness) * intensity;
}
"#;

/// Specializes the material texture and its authored linear-LOD conversion.
pub(super) fn material_transmission(source: &fluxel_material::RefractionEnvironment) -> String {
    let literal = |value: f32| format!("bitcast<f32>({}u)", value.to_bits());
    let [x, y, z] = source.direction_from_world.map(|column| {
        let [x, y, z] = column.map(literal);
        format!("vec3<f32>({x}, {y}, {z})")
    });
    let scale = literal(source.lod_scale);
    let offset = literal(source.lod_offset);
    let texture = &source.texture;
    let sampler = &source.sampler;
    format!(
        r#"
fn pbr_material_transmission(
    normal: vec3<f32>, view: vec3<f32>, roughness: f32, intensity: f32, thickness: f32,
) -> vec3<f32> {{
    let refracted = refract(-view, normal, 1.0 / material_volume_ior());
    let direction = mat3x3<f32>({x}, {y}, {z}) * pbr_normal(refracted);
    let lod = log2(f32(textureDimensions({texture}, 0).x)) * pbr_refraction_roughness(roughness);
    let max_lod = f32(textureNumLevels({texture})) - 1.0;
    let color = textureSampleLevel({texture}, {sampler}, pbr_refraction_latlong_uv(direction),
        clamp(lod * {scale} + {offset}, 0.0, max_lod)).rgb;
    return color * pbr_environment_attenuation(thickness) * intensity;
}}
"#
    )
}
