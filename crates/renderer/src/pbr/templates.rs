//! WGSL implementing one shared direct-light BRDF and its two consumers.
//!
//! GGX distribution, height-correlated Smith visibility and Schlick Fresnel:
//! https://google.github.io/filament/main/filament.html
//! Diffuse uses Lambert with the remaining (1-F) dielectric energy. Inputs and
//! outputs are linear; RGBA8 targets clamp HDR radiance without tone mapping.

pub(super) const NORMAL: &str = r#"
fn pbr_normal(value: vec3<f32>) -> vec3<f32> {
    let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    // Opposite vertex normals may cancel at interpolation. Use the unlit
    // back-facing direction so both routes remain black after UNORM encoding.
    if scale == 0.0 { return vec3<f32>(0.0, 0.0, -1.0); }
    return normalize(value / scale);
}
"#;

pub(super) const DIRECT_LIGHT: &str = r#"
fn pbr_direct_light(base_input: vec3<f32>, properties: vec2<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let base = clamp(base_input, vec3<f32>(0.0), vec3<f32>(1.0));
    let metal = clamp(properties.x, 0.0, 1.0);
    let roughness = clamp(properties.y, 0.04, 1.0);
    let normal = pbr_normal(world_normal);
    let light = vec3<f32>(0.6, 0.0, 0.8);
    let view = vec3<f32>(0.0, 0.0, 1.0);
    let half_direction = normalize(light + view);
    let no_l = clamp(dot(normal, light), 0.0, 1.0);
    let no_v = clamp(dot(normal, view), 0.0, 1.0);
    if no_l <= 0.0 || no_v <= 0.0 { return vec3<f32>(0.0); }
    let no_h = clamp(dot(normal, half_direction), 0.0, 1.0);
    let vo_h = clamp(dot(view, half_direction), 0.0, 1.0);
    let alpha = roughness * roughness;
    let alpha_squared = alpha * alpha;
    let distribution_term = no_h * no_h * (alpha_squared - 1.0) + 1.0;
    let distribution = alpha_squared / (3.141592653589793 * distribution_term * distribution_term);
    let visibility = 0.5 / (
        no_l * sqrt(no_v * no_v * (1.0 - alpha_squared) + alpha_squared) +
        no_v * sqrt(no_l * no_l * (1.0 - alpha_squared) + alpha_squared)
    );
    let f0 = mix(vec3<f32>(0.04), base, vec3<f32>(metal));
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vo_h, 5.0);
    let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metal) * base / 3.141592653589793;
    return (diffuse + distribution * visibility * fresnel) * no_l * 3.0;
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
@fragment
fn forward_fs(input: PbrVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(pbr_direct_light(evaluate_material().rgb, evaluate_metallic_roughness(), input.world_normal), 1.0);
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
fn deferred_gbuffer_fs(input: PbrVertexOutput) -> PbrGbufferOutput {
    let base = clamp(evaluate_material().rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let properties = evaluate_metallic_roughness();
    var output: PbrGbufferOutput;
    output.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0));
    output.normal_roughness = vec4<f32>(pbr_normal(input.world_normal) * 0.5 + vec3<f32>(0.5), clamp(properties.y, 0.04, 1.0));
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
@fragment
fn forward_fs(input: PbrUv0VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(pbr_direct_light(evaluate_material_uv0(input.uv0).rgb, evaluate_metallic_roughness_uv0(input.uv0), input.world_normal), 1.0);
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
fn deferred_gbuffer_fs(input: PbrUv0VertexOutput) -> PbrGbufferUv0Output {
    let base = clamp(evaluate_material_uv0(input.uv0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let properties = evaluate_metallic_roughness_uv0(input.uv0);
    var output: PbrGbufferUv0Output;
    output.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0));
    output.normal_roughness = vec4<f32>(pbr_normal(input.world_normal) * 0.5 + vec3<f32>(0.5), clamp(properties.y, 0.04, 1.0));
    return output;
}
"#;

pub(super) const FORWARD_UV0_NORMAL: &str = r#"
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn forward_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
@fragment fn forward_fs(input: PbrNormalVertexOutput) -> @location(0) vec4<f32> { return vec4<f32>(pbr_direct_light(evaluate_material_uv0(input.uv0).rgb, evaluate_metallic_roughness_uv0(input.uv0), pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0))), 1.0); }
"#;

pub(super) const GBUFFER_UV0_NORMAL: &str = r#"
struct PbrNormalGbufferOutput { @location(0) base_metal: vec4<f32>, @location(1) normal_roughness: vec4<f32>, };
struct PbrNormalVertexOutput { @builtin(position) position: vec4<f32>, @location(0) local_normal: vec3<f32>, @location(1) tangent: vec4<f32>, @location(2) uv0: vec2<f32>, };
@vertex fn deferred_gbuffer_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv0: vec2<f32>, @location(3) tangent: vec4<f32>) -> PbrNormalVertexOutput { var o: PbrNormalVertexOutput; o.position = pbr_transforms.clip_from_local * vec4<f32>(position, 1.0); o.local_normal = normal; o.tangent = tangent; o.uv0 = uv0; return o; }
fn pbr_world_normal(local_normal: vec3<f32>, tangent: vec4<f32>, sampled: vec3<f32>) -> vec3<f32> { let n = pbr_normal((pbr_transforms.normal_from_local * vec4<f32>(local_normal, 0.0)).xyz); let raw_t = (pbr_transforms.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz; let orthogonal = raw_t - n * dot(n, raw_t); let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); let t = select(pbr_normal(cross(n, axis)), pbr_normal(orthogonal), dot(orthogonal, orthogonal) > 1e-12); let tangent_sign = select(1.0, -1.0, tangent.w < 0.0); let handedness = tangent_sign * pbr_transforms.normal_from_local[3].x; let b = cross(n, t) * handedness; return pbr_normal(mat3x3<f32>(t, b, n) * sampled); }
@fragment fn deferred_gbuffer_fs(input: PbrNormalVertexOutput) -> PbrNormalGbufferOutput { let base = clamp(evaluate_material_uv0(input.uv0).rgb, vec3<f32>(0.0), vec3<f32>(1.0)); let properties = evaluate_metallic_roughness_uv0(input.uv0); var o: PbrNormalGbufferOutput; o.base_metal = vec4<f32>(base, clamp(properties.x, 0.0, 1.0)); o.normal_roughness = vec4<f32>(pbr_normal(pbr_world_normal(input.local_normal, input.tangent, evaluate_normal_uv0(input.uv0))) * 0.5 + vec3<f32>(0.5), clamp(properties.y, 0.04, 1.0)); return o; }
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
