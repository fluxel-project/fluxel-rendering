//! Direct lighting for the selected Babylon.js metallic-roughness branch.
//!
//! Adapted from Babylon.js at daae8d9483dcbb6308ebd0f6ca5b218783614771.
//! See ShadersWGSL/ShadersInclude pbrDirectLightingSetupFunctions.fx,
//! pbrDirectLightingFunctions.fx, pbrBRDFFunctions.fx, pbrHelperFunctions.fx,
//! pbrBlockReflectivity.fx, lightFragment.fx, and pbrBlockFinalUnlitComponents.fx.
//! Copyright 2023 The Babylon.js team. Licensed under Apache-2.0.

pub(super) const WGSL: &str = r#"
const PI: f32 = 3.141592653589793;
const RECIPROCAL_PI: f32 = 1.0 / PI;
const PBR_EPSILON: f32 = 0.0000001;

const constant1_FON: f32 = 0.5f - 2.0f / (3.0f * PI);
const constant2_FON: f32 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// Fujii Oren-Nayar (FON) directional albedo (approximate).
fn E_FON_approx(mu: f32, roughness: f32) -> f32
{
    var sigma: f32 = roughness; // FON sigma prime
    var mucomp: f32 = 1.0f - mu;
    var mucomp2: f32 = mucomp * mucomp;
    const Gcoeffs: mat2x2f = mat2x2f(0.0571085289f, -0.332181442f,
                               0.491881867f, 0.0714429953f);
    var GoverPi: f32 = dot(Gcoeffs * vec2f(mucomp, mucomp2), vec2f(1.0f, mucomp2));
    return (1.0f + sigma * GoverPi) / (1.0f + constant1_FON * sigma);
}

fn diffuseBRDF_EON(albedo: vec3f, roughness: f32, NdotL: f32, NdotV: f32, LdotV: f32) -> vec3f
{
    var rho: vec3f = albedo;
    var sigma: f32 = roughness;                            // FON sigma prime
    var mu_i: f32 = NdotL;                            // input angle cos
    var mu_o: f32 = NdotV;                            // output angle cos
    var s: f32 = LdotV - mu_i * mu_o;    // QON s term
    var sovertF: f32 = select(s, s / max(mu_i, mu_o), s > 0.0f); // FON s/t
    var AF: f32 = 1.0f / (1.0f + constant1_FON * sigma);   // FON A coeff.
    var f_ss: vec3f = (rho * RECIPROCAL_PI) * AF * (1.0f + sigma * sovertF); // single-scatter
    var EFo: f32 = E_FON_approx(mu_o, sigma);      // FON wo albedo (approx)
    var EFi: f32 = E_FON_approx(mu_i, sigma);      // FON wi albedo (approx)
    var avgEF: f32 = AF * (1.0f + constant2_FON * sigma);  // avg. albedo
    var rho_ms: vec3f = (rho * rho) * avgEF / (vec3f(1.0f) - rho * (1.0f - avgEF));
    const eps: f32 = 1.0e-7f;
    var f_ms: vec3f = (rho_ms * RECIPROCAL_PI) * max(eps, 1.0f - EFo) // multi-scatter lobe
                                 * max(eps, 1.0f - EFi)
                                 / max(eps, 1.0f - avgEF);
    return (f_ss + f_ms);
}

struct PbrDirectionalLight {
    direction_and_roughness: vec4<f32>,
    radiance: vec4<f32>,
}

struct PbrDirectLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn pbr_direct_light_brdf(
    base: vec3<f32>, properties: vec2<f32>, normal: vec3<f32>, view: vec3<f32>,
    dielectric_f0: vec3<f32>, directional: PbrDirectionalLight,
) -> PbrDirectLight {
    // The authored direction follows emitted rays, as in Babylon's vLightData.
    let light = pbr_normal(-directional.direction_and_roughness.xyz);
    let light_color = directional.radiance.rgb;
    let half_sum = view + light;
    let half_direction = select(normal, pbr_normal(half_sum), any(half_sum != vec3<f32>(0.0)));
    let no_l = clamp(dot(normal, light), PBR_EPSILON, 1.0);
    let no_v = abs(dot(normal, view)) + PBR_EPSILON;
    let no_h = clamp(dot(normal, half_direction), PBR_EPSILON, 1.0);
    let vo_h = clamp(dot(view, half_direction), 0.0, 1.0);

    // PBRBRDFConfiguration defaults to EON and legacy energy conservation.
    // The current surface supplies no separate diffuse-roughness texture/value.
    let surface_albedo = base * (vec3<f32>(1.0) - dielectric_f0) * (1.0 - properties.x);
    let eon_albedo = clamp(surface_albedo, vec3<f32>(0.1), vec3<f32>(1.0));
    let diffuse_term = diffuseBRDF_EON(eon_albedo, 0.0, no_l, no_v, dot(light, view)) / eon_albedo;
    let diffuse = max(diffuse_term * no_l * light_color * surface_albedo, vec3<f32>(0.0));

    // glTF dielectric and conductor Schlick branches with unit specular weight.
    let f0 = mix(dielectric_f0, base, vec3<f32>(properties.x));
    let grazing = 1.0 - vo_h;
    let grazing_squared = grazing * grazing;
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * (grazing_squared * grazing_squared * grazing);
    let roughness = clamp(properties.y + directional.direction_and_roughness.w, 0.0, 1.0);
    let alpha_g = roughness * roughness + 0.0005;
    let a2 = alpha_g * alpha_g;
    let d = no_h * no_h * (a2 - 1.0) + 1.0;
    let distribution = a2 / (PI * d * d);
    let ggx_v = no_l * sqrt(no_v * (no_v - a2 * no_v) + a2);
    let ggx_l = no_v * sqrt(no_l * (no_l - a2 * no_l) + a2);
    let visibility = 0.5 / (ggx_v + ggx_l);
    let specular = max(fresnel * distribution * visibility * no_l * light_color, vec3<f32>(0.0));
    return PbrDirectLight(diffuse, specular);
}

fn pbr_direct_light_components(
    base: vec3<f32>, properties: vec2<f32>, world_normal: vec3<f32>, view: vec3<f32>, dielectric_f0: vec3<f32>,
) -> PbrDirectLight {
    return pbr_direct_light_brdf(
        max(base, vec3<f32>(0.0)), clamp(properties, vec2<f32>(0.0), vec2<f32>(1.0)), pbr_normal(world_normal),
        view, dielectric_f0, pbr_directional_light(),
    );
}

fn pbr_direct_light_view(
    base: vec3<f32>, properties: vec2<f32>, world_normal: vec3<f32>, view: vec3<f32>, dielectric_f0: vec3<f32>,
) -> vec3<f32> {
    let terms = pbr_direct_light_components(base, properties, world_normal, view, dielectric_f0);
    return terms.diffuse + terms.specular;
}

fn pbr_gbuffer_is_covered(normal: vec3<f32>) -> bool {
    // The all-zero clear value cannot be a unit surface normal.
    return any(normal != vec3<f32>(0.0));
}

fn pbr_world_position_from_depth(
    position: vec2<f32>, depth: f32, extent: vec2<f32>, world_from_clip: mat4x4<f32>,
) -> vec3<f32> {
    // Fragment position is already the pixel center in a top-left origin.
    let uv = position / extent;
    let clip = vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    var world = world_from_clip * clip;
    if world.w == 0.0 {
        // An infinite far plane can produce w = 0 after depth rounds to one.
        // Lighting only needs the view ray, which an interior point also defines.
        world = world_from_clip * vec4<f32>(clip.xy, 0.5, 1.0);
    }
    return world.xyz / world.w;
}

fn pbr_shading_view_direction(world_position: vec3<f32>, camera: vec4<f32>) -> vec3<f32> {
    let direction = select(camera.xyz, camera.xyz - world_position, camera.w != 0.0);
    return pbr_normal(direction);
}

"#;
