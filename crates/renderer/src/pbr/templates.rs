//! WGSL for the standard Forward and Deferred PBR consumers.

pub(super) const NORMAL: &str = r#"
fn pbr_normal(value: vec3<f32>) -> vec3<f32> {
    let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    // Opposite vertex normals may cancel at interpolation. Use the unlit
    // back-facing direction so both routes remain black after UNORM encoding.
    if scale == 0.0 { return vec3<f32>(0.0, 0.0, -1.0); }
    return normalize(value / scale);
}
"#;

/// Surface-only normal orientation. Lighting-only composition deliberately
/// omits this unit because it has no Material contribution to supply the
/// static two-sided policy helper.
pub(super) const SURFACE_NORMAL: &str = r#"
fn pbr_surface_normal(normal: vec3<f32>, front_facing: bool) -> vec3<f32> {
    let normalized = pbr_normal(normal);
    if material_two_sided_lighting() && pbr_flip_back_normal(front_facing) { return -normalized; }
    return normalized;
}
"#;

pub(super) const DIFFUSE_IRRADIANCE: &str = r#"
fn pbr_latlong_uv(direction: vec3<f32>) -> vec2<f32> {
    let unit = pbr_normal(direction);
    let u = fract(atan2(unit.z, unit.x) / 6.283185307179586 + 0.5);
    let v = acos(clamp(unit.y, -1.0, 1.0)) / 3.141592653589793;
    return vec2<f32>(u, v);
}
fn pbr_diffuse_irradiance(base_input: vec3<f32>, metal_input: f32, world_normal: vec3<f32>, environment: texture_2d<f32>, environment_sampler: sampler, dielectric_energy: vec3<f32>) -> vec3<f32> {
    let base = max(base_input, vec3<f32>(0.0));
    let metal = clamp(metal_input, 0.0, 1.0);
    let irradiance = textureSample(environment, environment_sampler, pbr_latlong_uv(world_normal)).rgb;
    let surface_albedo = base * dielectric_energy * (1.0 - metal);
    return irradiance * surface_albedo;
}
"#;

pub(super) const TRANSFORMS: &str = r#"
struct PbrTransforms {
    clip_from_local: mat4x4<f32>,
    normal_from_local: mat4x4<f32>,
    world_from_local: mat4x4<f32>,
    shading_camera: vec4<f32>,
    light_direction_and_roughness: vec4<f32>,
    light_radiance: vec4<f32>,
    surface_camera_position: vec4<f32>,
};
@group(1) @binding(0) var<uniform> pbr_transforms: PbrTransforms;
struct PbrVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
};
fn pbr_vertex(position: vec3<f32>, normal: vec3<f32> /* instance_basis_parameter */) -> PbrVertexOutput {
    var output: PbrVertexOutput;
    output.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0);
    output.world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz;
    output.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz;
    return output;
}
"#;

pub(super) const SCENE_REFRACTION_TRANSFORMS: &str = r#"
struct PbrSceneRefractionTransforms {
    clip_from_local: mat4x4<f32>,
    normal_from_local: mat4x4<f32>,
    world_from_local: mat4x4<f32>,
    shading_camera: vec4<f32>,
    light_direction_and_roughness: vec4<f32>,
    light_radiance: vec4<f32>,
    surface_camera_position: vec4<f32>,
    clip_from_world: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> pbr_transforms: PbrSceneRefractionTransforms;
struct PbrVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
};
fn pbr_vertex(position: vec3<f32>, normal: vec3<f32> /* instance_basis_parameter */) -> PbrVertexOutput {
    var output: PbrVertexOutput;
    output.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0);
    output.world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz;
    output.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz;
    return output;
}
"#;

pub(super) const FORWARD_LIGHT: &str = r#"
fn pbr_directional_light() -> PbrDirectionalLight {
    return PbrDirectionalLight(pbr_transforms.light_direction_and_roughness, pbr_transforms.light_radiance);
}
"#;

pub(super) const FORWARD: &str = r#"
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32> /* instance_parameters */) -> PbrVertexOutput {
    /* instance_setup */
    return pbr_vertex(position, normal /* instance_basis_argument */);
}
fn renderer_forward_shade(input: PbrVertexOutput, front_facing: bool) -> vec4<f32> {
    let material = evaluate_material();
    let properties = evaluate_metallic_roughness();
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(pbr_direct_light_view(material.rgb, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04)), material_output_alpha(material.a));
}
"#;

pub(super) const FORWARD_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32> /* instance_parameters */) -> PbrVertexOutput {
    /* instance_setup */
    return pbr_vertex(position, normal /* instance_basis_argument */);
}
fn renderer_forward_shade(input: PbrVertexOutput, front_facing: bool) -> vec4<f32> {
    let material = evaluate_material();
    let base = material.rgb;
    let properties = evaluate_metallic_roughness();
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    let direct = pbr_direct_light_view(base, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04));
    let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04));
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(direct + indirect, material_output_alpha(material.a));
}
"#;

pub(super) const GBUFFER: &str = r#"
struct PbrGbufferOutput {
    @location(0) base_metal: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
};
@vertex
fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32> /* instance_parameters */) -> PbrVertexOutput {
    /* instance_setup */
    return pbr_vertex(position, normal /* instance_basis_argument */);
}
@fragment
fn deferred_gbuffer_fs(input: PbrVertexOutput, @builtin(front_facing) front_facing: bool) -> PbrGbufferOutput {
    let material = evaluate_material();
    let base = max(material.rgb, vec3<f32>(0.0));
    let properties = evaluate_metallic_roughness();
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    if material_alpha_rejected(material.a) { discard; }
    var output: PbrGbufferOutput;
    output.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0));
    output.normal_roughness = vec4<f32>(normal, clamp(properties.y, 0.0, 1.0));
    return output;
}
"#;

pub(super) const UV0_VERTEX: &str = r#"
struct PbrUv0VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv0: vec2<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) @interpolate(flat) camera_position: vec3<f32>,
    @location(4) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-5 */
};
fn pbr_uv0_vertex(position: vec3<f32>, normal: vec3<f32> /* instance_basis_parameter */, uv0: vec2<f32>) -> PbrUv0VertexOutput {
    let base = pbr_vertex(position, normal /* instance_basis_argument */);
    var output: PbrUv0VertexOutput;
    output.position = base.position;
    output.world_normal = base.world_normal;
    output.world_position = base.world_position;
    output.camera_position = pbr_transforms.surface_camera_position.xyz;
    output.uv0 = uv0;
    /* fluxel:vertex-varying-write-output */output.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0);
    return output;
}
"#;

pub(super) const FORWARD_UV0: &str = r#"
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32> /* instance_parameters */) -> PbrUv0VertexOutput {
    /* instance_setup */
    return pbr_uv0_vertex(position, normal /* instance_basis_argument */, uv0);
}
fn renderer_forward_shade(input: PbrUv0VertexOutput, front_facing: bool) -> vec4<f32> {
    let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let properties = surface.properties;
    let normal = pbr_surface_normal(select(input.world_normal, surface.world_normal, surface.has_world_normal), front_facing);
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(pbr_direct_light_view(material.rgb, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04)), material_output_alpha(material.a));
}
"#;

pub(super) const GBUFFER_UV0: &str = r#"
struct PbrGbufferUv0Output {
    @location(0) base_metal: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
};
@vertex
fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32> /* instance_parameters */) -> PbrUv0VertexOutput {
    /* instance_setup */
    return pbr_uv0_vertex(position, normal /* instance_basis_argument */, uv0);
}
@fragment
fn deferred_gbuffer_fs(input: PbrUv0VertexOutput, @builtin(front_facing) front_facing: bool) -> PbrGbufferUv0Output {
    let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let base = max(material.rgb, vec3<f32>(0.0));
    let properties = surface.properties;
    let normal = pbr_surface_normal(select(input.world_normal, surface.world_normal, surface.has_world_normal), front_facing);
    if material_alpha_rejected(material.a) { discard; }
    var output: PbrGbufferUv0Output;
    output.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0));
    output.normal_roughness = vec4<f32>(normal, clamp(properties.y, 0.0, 1.0));
    return output;
}
"#;

pub(super) const FORWARD_UV0_EFFECT: &str = r#"
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32> /* instance_parameters */) -> PbrUv0VertexOutput {
    /* instance_setup */
    return pbr_uv0_vertex(position, normal /* instance_basis_argument */, uv0);
}
fn renderer_forward_shade(input: PbrUv0VertexOutput, front_facing: bool) -> vec4<f32> {
    let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let base = material.rgb;
    let properties = surface.properties;
    let normal = pbr_surface_normal(select(input.world_normal, surface.world_normal, surface.has_world_normal), front_facing);
    let emissive = pbr_emissive(input.uv0);
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(max(pbr_direct_light_view(base, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04)) + emissive, vec3<f32>(0.0)), material_output_alpha(material.a));
}
"#;

pub(super) const FORWARD_UV0_EFFECT_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32> /* instance_parameters */) -> PbrUv0VertexOutput {
    /* instance_setup */
    return pbr_uv0_vertex(position, normal /* instance_basis_argument */, uv0);
}
fn renderer_forward_shade(input: PbrUv0VertexOutput, front_facing: bool) -> vec4<f32> {
    let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let base = material.rgb;
    let properties = surface.properties;
    let normal = pbr_surface_normal(select(input.world_normal, surface.world_normal, surface.has_world_normal), front_facing);
    let direct = pbr_direct_light_view(base, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04));
    let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04)) * clamp(surface.occlusion * evaluate_occlusion_uv0(input.uv0), 0.0, 1.0);
    let emissive = pbr_emissive(input.uv0);
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(max(direct + indirect + emissive, vec3<f32>(0.0)), material_output_alpha(material.a));
}
"#;

pub(super) const GBUFFER_UV0_EFFECT: &str = r#"
struct PbrUv0EffectGbufferOutput {
    @location(0) base_metal: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
    @location(2) emissive_ao: vec4<f32>,
};
@vertex
fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32> /* instance_parameters */) -> PbrUv0VertexOutput {
    /* instance_setup */
    return pbr_uv0_vertex(position, normal /* instance_basis_argument */, uv0);
}
@fragment
fn deferred_gbuffer_fs(input: PbrUv0VertexOutput, @builtin(front_facing) front_facing: bool) -> PbrUv0EffectGbufferOutput {
    let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let base = max(material.rgb, vec3<f32>(0.0));
    let properties = surface.properties;
    let normal = pbr_surface_normal(select(input.world_normal, surface.world_normal, surface.has_world_normal), front_facing);
    let emissive = pbr_emissive(input.uv0);
    let occlusion = clamp(surface.occlusion * evaluate_occlusion_uv0(input.uv0), 0.0, 1.0);
    if material_alpha_rejected(material.a) { discard; }
    var output: PbrUv0EffectGbufferOutput;
    output.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0));
    output.normal_roughness = vec4<f32>(normal, clamp(properties.y, 0.0, 1.0));
    output.emissive_ao = vec4<f32>(emissive, occlusion);
    return output;
}
"#;

/// Explicit tangent branch of Babylon.js bumpVertex.fx and
/// bumpFragmentMainFunctions.fx at daae8d9483dcbb6308ebd0f6ca5b218783614771.
/// Copyright 2023 The Babylon.js team. Licensed under Apache-2.0.
pub(super) const TANGENT_BASIS: &str = r#"
fn pbr_vertex_tangent_basis(local_normal: vec3<f32>, tangent: vec4<f32> /* instance_basis_parameter */) -> mat3x3<f32> {
    let n = pbr_normal(local_normal);
    let t = pbr_normal(tangent.xyz);
    let b = cross(n, t) * tangent.w;
    let world = mat3x3<f32>(
        pbr_transforms.world_from_local[0].xyz,
        pbr_transforms.world_from_local[1].xyz,
        pbr_transforms.world_from_local[2].xyz,
    );
    return world /* instance_basis_matrix */ * mat3x3<f32>(t, b, n);
}

fn pbr_apply_normal_map(t: vec3<f32>, b: vec3<f32>, n: vec3<f32>, sampled: vec3<f32>) -> vec3<f32> {
    return pbr_normal(mat3x3<f32>(t, b, n) * sampled);
}
"#;

pub(super) const FORWARD_UV0_NORMAL: &str = r#"
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) tangent_world: vec3<f32>, @location(1) bitangent_world: vec3<f32>, @location(2) normal_world: vec3<f32>, @location(3) uv0: vec2<f32>, @location(4) world_position: vec3<f32>, @location(5) geometric_world_normal: vec3<f32>, @location(6) @interpolate(flat) camera_position: vec3<f32>, @location(7) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-8 */ };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32> /* instance_parameters */) -> PbrNormalVertexOutput { /* instance_setup */ var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0); let basis = pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */); o.tangent_world = basis[0]; o.bitangent_world = basis[1]; o.normal_world = basis[2]; o.uv0 = uv0; /* fluxel:vertex-varying-write-o */o.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0); o.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz; o.geometric_world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz; o.camera_position = pbr_transforms.surface_camera_position.xyz; return o; }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> { let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.geometric_world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */); let material = surface.base; let properties = surface.properties; let normal = pbr_surface_normal(pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0)), front_facing); if material_alpha_rejected(material.a) { discard; } return vec4<f32>(pbr_direct_light_view(material.rgb, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04)), material_output_alpha(material.a)); }
"#;

pub(super) const GBUFFER_UV0_NORMAL: &str = r#"
struct PbrNormalGbufferOutput { @location(0) base_metal: vec4<f32>, @location(1) normal_roughness: vec4<f32>, };
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) tangent_world: vec3<f32>, @location(1) bitangent_world: vec3<f32>, @location(2) normal_world: vec3<f32>, @location(3) uv0: vec2<f32>, @location(4) world_position: vec3<f32>, @location(5) geometric_world_normal: vec3<f32>, @location(6) @interpolate(flat) camera_position: vec3<f32>, @location(7) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-8 */ };
@vertex fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32> /* instance_parameters */) -> PbrNormalVertexOutput { /* instance_setup */ var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0); let basis = pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */); o.tangent_world = basis[0]; o.bitangent_world = basis[1]; o.normal_world = basis[2]; o.uv0 = uv0; /* fluxel:vertex-varying-write-o */o.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0); o.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz; o.geometric_world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz; o.camera_position = pbr_transforms.surface_camera_position.xyz; return o; }
@fragment fn deferred_gbuffer_fs(input: PbrNormalVertexOutput, @builtin(front_facing) front_facing: bool) -> PbrNormalGbufferOutput { let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.geometric_world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */); let material = surface.base; let base = max(material.rgb, vec3<f32>(0.0)); let properties = surface.properties; let normal = pbr_surface_normal(pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0)), front_facing); if material_alpha_rejected(material.a) { discard; } var o: PbrNormalGbufferOutput; o.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0)); o.normal_roughness = vec4<f32>(normal, clamp(properties.y, 0.0, 1.0)); return o; }
"#;

pub(super) const FORWARD_UV0_NORMAL_EFFECT: &str = r#"
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) tangent_world: vec3<f32>, @location(1) bitangent_world: vec3<f32>, @location(2) normal_world: vec3<f32>, @location(3) uv0: vec2<f32>, @location(4) world_position: vec3<f32>, @location(5) geometric_world_normal: vec3<f32>, @location(6) @interpolate(flat) camera_position: vec3<f32>, @location(7) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-8 */ };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32> /* instance_parameters */) -> PbrNormalVertexOutput { /* instance_setup */ var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0); let basis = pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */); o.tangent_world = basis[0]; o.bitangent_world = basis[1]; o.normal_world = basis[2]; o.uv0 = uv0; /* fluxel:vertex-varying-write-o */o.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0); o.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz; o.geometric_world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz; o.camera_position = pbr_transforms.surface_camera_position.xyz; return o; }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> { let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.geometric_world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */); let material = surface.base; let base = material.rgb; let properties = surface.properties; let normal = pbr_surface_normal(pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0)), front_facing); let emissive = pbr_emissive(input.uv0); if material_alpha_rejected(material.a) { discard; } return vec4<f32>(max(pbr_direct_light_view(base, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04)) + emissive, vec3<f32>(0.0)), material_output_alpha(material.a)); }
"#;

pub(super) const FORWARD_UV0_NORMAL_EFFECT_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) tangent_world: vec3<f32>, @location(1) bitangent_world: vec3<f32>, @location(2) normal_world: vec3<f32>, @location(3) uv0: vec2<f32>, @location(4) world_position: vec3<f32>, @location(5) geometric_world_normal: vec3<f32>, @location(6) @interpolate(flat) camera_position: vec3<f32>, @location(7) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-8 */ };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32> /* instance_parameters */) -> PbrNormalVertexOutput { /* instance_setup */ var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0); let basis = pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */); o.tangent_world = basis[0]; o.bitangent_world = basis[1]; o.normal_world = basis[2]; o.uv0 = uv0; /* fluxel:vertex-varying-write-o */o.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0); o.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz; o.geometric_world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz; o.camera_position = pbr_transforms.surface_camera_position.xyz; return o; }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> { let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.geometric_world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */); let material = surface.base; let base = material.rgb; let properties = surface.properties; let normal = pbr_surface_normal(pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0)), front_facing); let direct = pbr_direct_light_view(base, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04)); let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04)) * clamp(surface.occlusion * evaluate_occlusion_uv0(input.uv0), 0.0, 1.0); let emissive = pbr_emissive(input.uv0); if material_alpha_rejected(material.a) { discard; } return vec4<f32>(max(direct + indirect + emissive, vec3<f32>(0.0)), material_output_alpha(material.a)); }
"#;

/// Effect-enabled tangent-space-normal G-buffer. RGB retains signed linear
/// emissive values until the lighting pass sums the terms, following Babylon's
/// PBR final color composition. Alpha carries scalar ambient occlusion.
pub(super) const GBUFFER_UV0_NORMAL_EFFECT: &str = r#"
struct PbrNormalEffectGbufferOutput { @location(0) base_metal: vec4<f32>, @location(1) normal_roughness: vec4<f32>, @location(2) emissive_ao: vec4<f32>, };
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) tangent_world: vec3<f32>, @location(1) bitangent_world: vec3<f32>, @location(2) normal_world: vec3<f32>, @location(3) uv0: vec2<f32>, @location(4) world_position: vec3<f32>, @location(5) geometric_world_normal: vec3<f32>, @location(6) @interpolate(flat) camera_position: vec3<f32>, @location(7) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-8 */ };
@vertex fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32> /* instance_parameters */) -> PbrNormalVertexOutput { /* instance_setup */ var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0); let basis = pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */); o.tangent_world = basis[0]; o.bitangent_world = basis[1]; o.normal_world = basis[2]; o.uv0 = uv0; /* fluxel:vertex-varying-write-o */o.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0); o.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz; o.geometric_world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz; o.camera_position = pbr_transforms.surface_camera_position.xyz; return o; }
@fragment fn deferred_gbuffer_fs(input: PbrNormalVertexOutput, @builtin(front_facing) front_facing: bool) -> PbrNormalEffectGbufferOutput { let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.geometric_world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */); let material = surface.base; let base = max(material.rgb, vec3<f32>(0.0)); let properties = surface.properties; let normal = pbr_surface_normal(pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0)), front_facing); let emissive = pbr_emissive(input.uv0); let occlusion = clamp(surface.occlusion * evaluate_occlusion_uv0(input.uv0), 0.0, 1.0); if material_alpha_rejected(material.a) { discard; } var o: PbrNormalEffectGbufferOutput; o.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0)); o.normal_roughness = vec4<f32>(normal, clamp(properties.y, 0.0, 1.0)); o.emissive_ao = vec4<f32>(emissive, occlusion); return o; }
"#;

pub(super) const BASIC_LIGHT_VIEW: &str = r#"
struct PbrBasicLightingView {
    world_from_clip: mat4x4<f32>,
    camera: vec4<f32>,
    extent: vec4<f32>,
    light: PbrDirectionalLight,
};
@group(0) @binding(4) var<uniform> pbr_basic_view: PbrBasicLightingView;

fn pbr_directional_light() -> PbrDirectionalLight {
    return pbr_basic_view.light;
}

fn pbr_deferred_view_direction(position: vec2<f32>, depth: f32) -> vec3<f32> {
    if pbr_basic_view.camera.w == 0.0 {
        return pbr_normal(pbr_basic_view.camera.xyz);
    }
    let world = pbr_world_position_from_depth(
        position, depth, pbr_basic_view.extent.xy, pbr_basic_view.world_from_clip,
    );
    return pbr_shading_view_direction(world, pbr_basic_view.camera);
}
"#;

pub(super) const LIGHTING: &str = r#"
@group(0) @binding(0) var pbr_base_metal: texture_2d<f32>;
@group(0) @binding(1) var pbr_normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var pbr_depth: texture_depth_2d;
@vertex
fn deferred_lighting_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}
@fragment
fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let coordinate = vec2<i32>(position.xy);
    let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0);
    if !pbr_gbuffer_is_covered(normal_roughness.xyz) { return vec4<f32>(0.0); }
    let depth = textureLoad(pbr_depth, coordinate, 0);
    let base_metal = textureLoad(pbr_base_metal, coordinate, 0);
    let normal = normal_roughness.xyz;
    return vec4<f32>(pbr_direct_light_view(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal, pbr_deferred_view_direction(position.xy, depth), vec3<f32>(0.04)), 1.0);
}
"#;

pub(super) const LIGHTING_IRRADIANCE: &str = r#"
@group(0) @binding(0) var pbr_base_metal: texture_2d<f32>;
@group(0) @binding(1) var pbr_normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var pbr_depth: texture_depth_2d;
@group(1) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(1) @binding(1) var pbr_environment_sampler: sampler;
@vertex
fn deferred_lighting_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}
@fragment
fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let coordinate = vec2<i32>(position.xy);
    let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0);
    if !pbr_gbuffer_is_covered(normal_roughness.xyz) { return vec4<f32>(0.0); }
    let depth = textureLoad(pbr_depth, coordinate, 0);
    let base_metal = textureLoad(pbr_base_metal, coordinate, 0);
    let normal = normal_roughness.xyz;
    let direct = pbr_direct_light_view(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal, pbr_deferred_view_direction(position.xy, depth), vec3<f32>(0.04));
    let indirect = pbr_diffuse_irradiance(base_metal.rgb, base_metal.a, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04));
    return vec4<f32>(direct + indirect, 1.0);
}
"#;

pub(super) const LIGHTING_EFFECT: &str = r#"
@group(0) @binding(0) var pbr_base_metal: texture_2d<f32>;
@group(0) @binding(1) var pbr_normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var pbr_depth: texture_depth_2d;
@group(0) @binding(3) var pbr_emissive_ao: texture_2d<f32>;
@vertex fn deferred_lighting_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> { var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)); return vec4<f32>(positions[index], 0.0, 1.0); }
@fragment fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> { let coordinate = vec2<i32>(position.xy); let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0); if !pbr_gbuffer_is_covered(normal_roughness.xyz) { return vec4<f32>(0.0); } let depth = textureLoad(pbr_depth, coordinate, 0); let base_metal = textureLoad(pbr_base_metal, coordinate, 0); let effect = textureLoad(pbr_emissive_ao, coordinate, 0); let normal = normal_roughness.xyz; return vec4<f32>(max(pbr_direct_light_view(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal, pbr_deferred_view_direction(position.xy, depth), vec3<f32>(0.04)) + effect.rgb, vec3<f32>(0.0)), 1.0); }
"#;

pub(super) const LIGHTING_EFFECT_IRRADIANCE: &str = r#"
@group(0) @binding(0) var pbr_base_metal: texture_2d<f32>;
@group(0) @binding(1) var pbr_normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var pbr_depth: texture_depth_2d;
@group(0) @binding(3) var pbr_emissive_ao: texture_2d<f32>;
@group(1) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(1) @binding(1) var pbr_environment_sampler: sampler;
@vertex fn deferred_lighting_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> { var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)); return vec4<f32>(positions[index], 0.0, 1.0); }
@fragment fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> { let coordinate = vec2<i32>(position.xy); let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0); if !pbr_gbuffer_is_covered(normal_roughness.xyz) { return vec4<f32>(0.0); } let depth = textureLoad(pbr_depth, coordinate, 0); let base_metal = textureLoad(pbr_base_metal, coordinate, 0); let effect = textureLoad(pbr_emissive_ao, coordinate, 0); let normal = normal_roughness.xyz; let direct = pbr_direct_light_view(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal, pbr_deferred_view_direction(position.xy, depth), vec3<f32>(0.04)); let indirect = pbr_diffuse_irradiance(base_metal.rgb, base_metal.a, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04)) * effect.a; return vec4<f32>(max(direct + indirect + effect.rgb, vec3<f32>(0.0)), 1.0); }
"#;

pub(super) const FORWARD_UV0_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32> /* instance_parameters */) -> PbrUv0VertexOutput {
    /* instance_setup */
    return pbr_uv0_vertex(position, normal /* instance_basis_argument */, uv0);
}
fn renderer_forward_shade(input: PbrUv0VertexOutput, front_facing: bool) -> vec4<f32> {
    let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let base = material.rgb;
    let properties = surface.properties;
    let normal = pbr_surface_normal(select(input.world_normal, surface.world_normal, surface.has_world_normal), front_facing);
    let direct = pbr_direct_light_view(base, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04));
    let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04));
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(direct + indirect, material_output_alpha(material.a));
}
"#;

pub(super) const FORWARD_UV0_NORMAL_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;

struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) tangent_world: vec3<f32>, @location(1) bitangent_world: vec3<f32>, @location(2) normal_world: vec3<f32>, @location(3) uv0: vec2<f32>, @location(4) world_position: vec3<f32>, @location(5) geometric_world_normal: vec3<f32>, @location(6) @interpolate(flat) camera_position: vec3<f32>, @location(7) vertex_uv0: vec2<f32>, /* fluxel:vertex-varying-fields-8 */ };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32> /* instance_parameters */) -> PbrNormalVertexOutput { /* instance_setup */ var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local /* instance_matrix */ * vec4<f32>(position, 1.0); let basis = pbr_vertex_tangent_basis(normal, tangent /* instance_basis_argument */); o.tangent_world = basis[0]; o.bitangent_world = basis[1]; o.normal_world = basis[2]; o.uv0 = uv0; /* fluxel:vertex-varying-write-o */o.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0); o.world_position = (pbr_transforms.world_from_local /* instance_matrix */ * vec4<f32>(position, 1.0)).xyz; o.geometric_world_normal = (pbr_transforms.normal_from_local /* instance_normal_matrix */ * vec4<f32>(normal, 0.0)).xyz; o.camera_position = pbr_transforms.surface_camera_position.xyz; return o; }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> {
    let surface = evaluate_pbr_surface_uv0(input.uv0, input.world_position, pbr_normal(input.geometric_world_normal), input.camera_position, input.vertex_uv0 /* fluxel:vertex-varying-arguments */);
    let material = surface.base;
    let base = material.rgb;
    let properties = surface.properties;
    let normal = pbr_surface_normal(pbr_apply_normal_map(input.tangent_world, input.bitangent_world, input.normal_world, evaluate_normal_uv0(input.uv0)), front_facing);
    let direct = pbr_direct_light_view(base, properties, normal, pbr_shading_view_direction(input.world_position, pbr_transforms.shading_camera), vec3<f32>(0.04));
    let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04));
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(direct + indirect, material_output_alpha(material.a));
}
"#;
