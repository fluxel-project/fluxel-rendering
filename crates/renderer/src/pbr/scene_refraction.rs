//! Projected scene-color transmission for the selected Babylon.js refraction route.
//!
//! Adapted from Babylon.js at daae8d9483dcbb6308ebd0f6ca5b218783614771.
//! See `ShadersWGSL/ShadersInclude/pbrBlockSubSurface.fx`, specifically the
//! `SS_USE_THICKNESS_AS_DEPTH` 2D-refraction branch and final transmittance.
//! Copyright 2023 The Babylon.js team. Licensed under Apache-2.0.

/// This unit deliberately declares no binding. Its caller supplies the previous
/// scene-color capture and sampler, so the Graph remains responsible for the
/// capture producer and the real dependency between opaque color and refraction.
/// `pbr_transforms.clip_from_world` is the camera's Fluxel clip transform:
/// projective XY is NDC, with +Y up, and is therefore mapped to texture UV here.
pub(super) const WGSL: &str = r#"
fn pbr_scene_refraction_uv(
    world_position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    thickness: f32,
) -> vec2<f32> {
    // This is Babylon's SS_USE_THICKNESS_AS_DEPTH branch. Babylon's `view`
    // followed by `refractionMatrix` produces texture-projective coordinates;
    // Fluxel carries their composed equivalent as clip_from_world.
    let refracted = refract(-view, normal, 1.0 / material_volume_ior());
    var projected = pbr_transforms.clip_from_world
        * vec4<f32>(world_position + refracted * thickness, 1.0);
    // A displaced point behind the perspective camera has no scene sample.
    // Use the surface coordinate from the same capture in that case.
    if projected.w <= 0.0000001 {
        projected = pbr_transforms.clip_from_world * vec4<f32>(world_position, 1.0);
    }
    let ndc = projected.xy / max(projected.w, 0.0000001);
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn pbr_scene_transmission(
    scene: texture_2d<f32>,
    scene_sampler: sampler,
    world_position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    intensity: f32,
    thickness: f32,
) -> vec3<f32> {
    let world_thickness = thickness * pbr_world_max_scale();
    let capture_uv = pbr_scene_refraction_uv(world_position, normal, view, world_thickness);
    // Edge texels bound the projected image even if the shared material
    // sampler repeats its other, ordinarily authored textures.
    let half_texel = vec2<f32>(0.5) / vec2<f32>(textureDimensions(scene));
    let uv = clamp(capture_uv, half_texel, vec2<f32>(1.0) - half_texel);
    // The captured scene color is a single resolved image, not a prefiltered
    // environment map. Its level zero is the prior Graph product.
    let scene_color = textureSampleLevel(scene, scene_sampler, uv, 0.0).rgb;
    let absorption = -log(material_attenuation_color())
        / max(material_attenuation_distance(), 1e-5);
    let attenuation = exp(-absorption * world_thickness);
    return scene_color * attenuation * intensity;
}
"#;
