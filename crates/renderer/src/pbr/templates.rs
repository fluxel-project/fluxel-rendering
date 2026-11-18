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
    if material_two_sided_lighting() && !front_facing { return -normalized; }
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
    let base = clamp(base_input, vec3<f32>(0.0), vec3<f32>(1.0));
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
};
@group(1) @binding(0) var<uniform> pbr_transforms: PbrTransforms;
struct PbrVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};
fn pbr_vertex(position: vec3<f32>, normal: vec3<f32>) -> PbrVertexOutput {
    var output: PbrVertexOutput;
    output.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0);
    output.world_normal = (pbr_transforms.normal_from_local * vec4<f32>(normal, 0.0)).xyz;
    return output;
}
"#;

pub(super) const FORWARD: &str = r#"
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> PbrVertexOutput {
    return pbr_vertex(position, normal);
}
fn renderer_forward_shade(input: PbrVertexOutput, front_facing: bool) -> vec4<f32> {
    let material = evaluate_material();
    let properties = evaluate_metallic_roughness();
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(pbr_direct_light(material.rgb, properties, normal), material_output_alpha(material.a));
}
"#;

pub(super) const FORWARD_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> PbrVertexOutput {
    return pbr_vertex(position, normal);
}
fn renderer_forward_shade(input: PbrVertexOutput, front_facing: bool) -> vec4<f32> {
    let material = evaluate_material();
    let base = material.rgb;
    let properties = evaluate_metallic_roughness();
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    let direct = pbr_direct_light(base, properties, normal);
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
fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> PbrVertexOutput {
    return pbr_vertex(position, normal);
}
@fragment
fn deferred_gbuffer_fs(input: PbrVertexOutput, @builtin(front_facing) front_facing: bool) -> PbrGbufferOutput {
    let material = evaluate_material();
    let base = clamp(material.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let properties = evaluate_metallic_roughness();
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    if material_alpha_rejected(material.a) { discard; }
    var output: PbrGbufferOutput;
    output.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0));
    output.normal_roughness = vec4<f32>(normal * 0.5 + vec3<f32>(0.5), clamp(properties.y, 0.0, 1.0));
    return output;
}
"#;

pub(super) const FORWARD_UV0: &str = r#"
struct PbrUv0VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv0: vec2<f32>,
};
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>) -> PbrUv0VertexOutput {
    let base = pbr_vertex(position, normal);
    var output: PbrUv0VertexOutput;
    output.position = base.position;
    output.world_normal = base.world_normal;
    output.uv0 = uv0;
    return output;
}
fn renderer_forward_shade(input: PbrUv0VertexOutput, front_facing: bool) -> vec4<f32> {
    let material = evaluate_material_uv0(input.uv0);
    let properties = evaluate_metallic_roughness_uv0(input.uv0);
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(pbr_direct_light(material.rgb, properties, normal), material_output_alpha(material.a));
}
"#;

pub(super) const GBUFFER_UV0: &str = r#"
struct PbrGbufferUv0Output {
    @location(0) base_metal: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
};
struct PbrUv0VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv0: vec2<f32>,
};
@vertex
fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>) -> PbrUv0VertexOutput {
    let base = pbr_vertex(position, normal);
    var output: PbrUv0VertexOutput;
    output.position = base.position;
    output.world_normal = base.world_normal;
    output.uv0 = uv0;
    return output;
}
@fragment
fn deferred_gbuffer_fs(input: PbrUv0VertexOutput, @builtin(front_facing) front_facing: bool) -> PbrGbufferUv0Output {
    let material = evaluate_material_uv0(input.uv0);
    let base = clamp(material.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let properties = evaluate_metallic_roughness_uv0(input.uv0);
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    if material_alpha_rejected(material.a) { discard; }
    var output: PbrGbufferUv0Output;
    output.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0));
    output.normal_roughness = vec4<f32>(normal * 0.5 + vec3<f32>(0.5), clamp(properties.y, 0.0, 1.0));
    return output;
}
"#;

pub(super) const FORWARD_UV0_NORMAL: &str = r#"
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> { let material = evaluate_material_uv0(input.uv0); let properties = evaluate_metallic_roughness_uv0(input.uv0); let normal = pbr_surface_normal(pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0)), front_facing); if material_alpha_rejected(material.a) { discard; } return vec4<f32>(pbr_direct_light(material.rgb, properties, normal), material_output_alpha(material.a)); }
"#;

pub(super) const GBUFFER_UV0_NORMAL: &str = r#"
struct PbrNormalGbufferOutput { @location(0) base_metal: vec4<f32>, @location(1) normal_roughness: vec4<f32>, };
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
@fragment fn deferred_gbuffer_fs(input: PbrNormalVertexOutput, @builtin(front_facing) front_facing: bool) -> PbrNormalGbufferOutput { let material = evaluate_material_uv0(input.uv0); let base = clamp(material.rgb, vec3<f32>(0.0), vec3<f32>(1.0)); let properties = evaluate_metallic_roughness_uv0(input.uv0); let normal = pbr_surface_normal(pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0)), front_facing); if material_alpha_rejected(material.a) { discard; } var o: PbrNormalGbufferOutput; o.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0)); o.normal_roughness = vec4<f32>(normal * 0.5 + vec3<f32>(0.5), clamp(properties.y, 0.0, 1.0)); return o; }
"#;

pub(super) const FORWARD_UV0_NORMAL_EFFECT: &str = r#"
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> { let material = evaluate_material_uv0(input.uv0); let base = material.rgb; let properties = evaluate_metallic_roughness_uv0(input.uv0); let normal = pbr_surface_normal(pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0)), front_facing); let emissive = max(evaluate_emissive_uv0(input.uv0), vec3<f32>(0.0)); if material_alpha_rejected(material.a) { discard; } return vec4<f32>(pbr_direct_light(base, properties, normal) + emissive, material_output_alpha(material.a)); }
"#;

pub(super) const FORWARD_UV0_NORMAL_EFFECT_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> { let material = evaluate_material_uv0(input.uv0); let base = material.rgb; let properties = evaluate_metallic_roughness_uv0(input.uv0); let normal = pbr_surface_normal(pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0)), front_facing); let direct = pbr_direct_light(base, properties, normal); let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04)) * clamp(evaluate_occlusion_uv0(input.uv0), 0.0, 1.0); let emissive = max(evaluate_emissive_uv0(input.uv0), vec3<f32>(0.0)); if material_alpha_rejected(material.a) { discard; } return vec4<f32>(direct + indirect + emissive, material_output_alpha(material.a)); }
"#;

/// Effect-enabled tangent-space-normal G-buffer. RGB preserves linear HDR
/// emissive radiance; alpha carries scalar ambient occlusion.
pub(super) const GBUFFER_UV0_NORMAL_EFFECT: &str = r#"
struct PbrNormalEffectGbufferOutput { @location(0) base_metal: vec4<f32>, @location(1) normal_roughness: vec4<f32>, @location(2) emissive_ao: vec4<f32>, };
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
@fragment fn deferred_gbuffer_fs(input: PbrNormalVertexOutput, @builtin(front_facing) front_facing: bool) -> PbrNormalEffectGbufferOutput { let material = evaluate_material_uv0(input.uv0); let base = clamp(material.rgb, vec3<f32>(0.0), vec3<f32>(1.0)); let properties = evaluate_metallic_roughness_uv0(input.uv0); let normal = pbr_surface_normal(pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0)), front_facing); let emissive = max(evaluate_emissive_uv0(input.uv0), vec3<f32>(0.0)); let occlusion = clamp(evaluate_occlusion_uv0(input.uv0), 0.0, 1.0); if material_alpha_rejected(material.a) { discard; } var o: PbrNormalEffectGbufferOutput; o.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0)); o.normal_roughness = vec4<f32>(normal * 0.5 + vec3<f32>(0.5), clamp(properties.y, 0.0, 1.0)); o.emissive_ao = vec4<f32>(emissive, occlusion); return o; }
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
    if textureLoad(pbr_depth, coordinate, 0) >= 1.0 { return vec4<f32>(0.0); }
    let base_metal = textureLoad(pbr_base_metal, coordinate, 0);
    let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0);
    let normal = normal_roughness.xyz * 2.0 - vec3<f32>(1.0);
    return vec4<f32>(pbr_direct_light(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal), 1.0);
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
    if textureLoad(pbr_depth, coordinate, 0) >= 1.0 { return vec4<f32>(0.0); }
    let base_metal = textureLoad(pbr_base_metal, coordinate, 0);
    let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0);
    let normal = normal_roughness.xyz * 2.0 - vec3<f32>(1.0);
    let direct = pbr_direct_light(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal);
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
@fragment fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> { let coordinate = vec2<i32>(position.xy); if textureLoad(pbr_depth, coordinate, 0) >= 1.0 { return vec4<f32>(0.0); } let base_metal = textureLoad(pbr_base_metal, coordinate, 0); let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0); let effect = textureLoad(pbr_emissive_ao, coordinate, 0); let normal = normal_roughness.xyz * 2.0 - vec3<f32>(1.0); return vec4<f32>(pbr_direct_light(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal) + effect.rgb, 1.0); }
"#;

pub(super) const LIGHTING_EFFECT_IRRADIANCE: &str = r#"
@group(0) @binding(0) var pbr_base_metal: texture_2d<f32>;
@group(0) @binding(1) var pbr_normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var pbr_depth: texture_depth_2d;
@group(0) @binding(3) var pbr_emissive_ao: texture_2d<f32>;
@group(1) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(1) @binding(1) var pbr_environment_sampler: sampler;
@vertex fn deferred_lighting_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> { var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)); return vec4<f32>(positions[index], 0.0, 1.0); }
@fragment fn deferred_lighting_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> { let coordinate = vec2<i32>(position.xy); if textureLoad(pbr_depth, coordinate, 0) >= 1.0 { return vec4<f32>(0.0); } let base_metal = textureLoad(pbr_base_metal, coordinate, 0); let normal_roughness = textureLoad(pbr_normal_roughness, coordinate, 0); let effect = textureLoad(pbr_emissive_ao, coordinate, 0); let normal = normal_roughness.xyz * 2.0 - vec3<f32>(1.0); let direct = pbr_direct_light(base_metal.rgb, vec2<f32>(base_metal.a, normal_roughness.a), normal); let indirect = pbr_diffuse_irradiance(base_metal.rgb, base_metal.a, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04)) * effect.a; return vec4<f32>(direct + indirect + effect.rgb, 1.0); }
"#;

pub(super) const FORWARD_UV0_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;

struct PbrUv0VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv0: vec2<f32>,
};
@vertex
fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>) -> PbrUv0VertexOutput {
    let base = pbr_vertex(position, normal);
    var output: PbrUv0VertexOutput;
    output.position = base.position;
    output.world_normal = base.world_normal;
    output.uv0 = uv0;
    return output;
}
fn renderer_forward_shade(input: PbrUv0VertexOutput, front_facing: bool) -> vec4<f32> {
    let material = evaluate_material_uv0(input.uv0);
    let base = material.rgb;
    let properties = evaluate_metallic_roughness_uv0(input.uv0);
    let normal = pbr_surface_normal(input.world_normal, front_facing);
    let direct = pbr_direct_light(base, properties, normal);
    let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04));
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(direct + indirect, material_output_alpha(material.a));
}
"#;

pub(super) const FORWARD_UV0_NORMAL_IRRADIANCE: &str = r#"
@group(2) @binding(0) var pbr_environment_irradiance: texture_2d<f32>;
@group(2) @binding(1) var pbr_environment_sampler: sampler;

struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
fn renderer_forward_shade(input: PbrNormalVertexOutput, front_facing: bool) -> vec4<f32> {
    let material = evaluate_material_uv0(input.uv0);
    let base = material.rgb;
    let properties = evaluate_metallic_roughness_uv0(input.uv0);
    let normal = pbr_surface_normal(pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0)), front_facing);
    let direct = pbr_direct_light(base, properties, normal);
    let indirect = pbr_diffuse_irradiance(base, properties.x, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04));
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(direct + indirect, material_output_alpha(material.a));
}
"#;
