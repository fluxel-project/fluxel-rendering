//! Real-device proof for UV0 normal-surface occlusion and emissive effects.
//!
//! All proof cases deliberately remain below final RGBA8 saturation.  HDR
//! storage remains a Deferred attachment contract, not an inferred claim from
//! a saturated final target.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceExtensions,
    SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, EnvironmentIrradianceAsset,
    EnvironmentIrradianceData, ForwardAssets, ForwardCpuPipeline, ForwardResources, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds, RenderChange,
    RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView, Renderer,
    SceneAssetRef,
};
use fluxel_rhi::{
    api::{Device, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::SourceUnit;

use crate::block_on;

const EXTENT: [u32; 2] = [64, 64];
const BASE: [f64; 3] = [0.10, 0.15, 0.20];
// Linear E/pi; a uniform map removes sampling ambiguity from this focused proof.
const ENVIRONMENT: [u8; 4] = [64, 96, 128, 255];
const FORWARD_TOLERANCE: u8 = 3;
const DEFERRED_TOLERANCE: u8 = 6;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    rgba: [u8; 4],
    environment: bool,
}

pub(crate) fn run<B>(device: Device<B>, output: &Path) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(output)?;
    let geometries = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let environments =
        AssetStore::<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>::new();
    let geometry = snapshot_geometry(&geometries)?;
    let material = snapshot_material(&materials)?;
    let environment = snapshot_environment(&environments)?;
    // AO is texture alpha, while emissive is RGB times two. The selected
    // source remains below one after final lighting, so every component is
    // observable in final RGBA8 rather than hidden by saturation.
    let cases = cases();
    fs::write(
        output.join("m1-surface-effects-geometry.bin"),
        &geometry.value().streams[0],
    )?;
    fs::write(
        output.join("m1-surface-effects-material.txt"),
        format!("{:#?}\n", material.value()),
    )?;
    fs::write(
        output.join("m1-surface-effects-environment-e-over-pi.rgba"),
        ENVIRONMENT,
    )?;
    fs::write(
        output.join("m1-surface-effects-input.json"),
        input_json(&cases),
    )?;

    let mut renderer = Renderer::new(device)?;
    let mut observed = Vec::new();
    for case in cases {
        let texture = snapshot_texture(&textures, case.rgba)?;
        let forward = oracle(case, false);
        let deferred = oracle(case, true);
        require_nonsaturated(&forward, case.name, "Forward")?;
        require_nonsaturated(&deferred, case.name, "Deferred")?;
        // Inputs, geometry, material and independent expectations all exist
        // before the first GPU allocation/upload/recording of this case.
        fs::write(
            output.join(format!("m1-surface-effects-{}-texture.rgba", case.name)),
            case.rgba,
        )?;
        fs::write(
            output.join(format!("m1-surface-effects-{}-oracle.json", case.name)),
            oracle_json(case, &forward, &deferred),
        )?;
        renderer.set_environment_irradiance(if case.environment {
            Some(environment.clone())
        } else {
            None
        })?;
        let (snapshot, view, assets) = scene(&geometry, &material, &texture)?;
        let bounds = RenderBounds::new([-1.0; 3], [1.0; 3])?;
        let deferred_submission = block_on(renderer.submit_deferred(
            &ExternalDeferredPipeline::new(),
            snapshot.clone(),
            view,
            bounds,
            &assets,
        ))?;
        let deferred_actual = block_on(renderer.readback(&deferred_submission))?;
        fs::write(
            output.join(format!(
                "m1-surface-effects-{}-deferred-readback.rgba",
                case.name
            )),
            &deferred_actual,
        )?;
        verify_center(
            &deferred_actual,
            deferred.encoded,
            DEFERRED_TOLERANCE,
            case.name,
            "Deferred",
        )?;
        let forward_submission = block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds,
            &assets,
        ))?;
        let forward_actual = block_on(renderer.readback(&forward_submission))?;
        fs::write(
            output.join(format!(
                "m1-surface-effects-{}-forward-readback.rgba",
                case.name
            )),
            &forward_actual,
        )?;
        verify_center(
            &forward_actual,
            forward.encoded,
            FORWARD_TOLERANCE,
            case.name,
            "Forward",
        )?;
        observed.push((case.name, center(&forward_actual)?));
    }
    require_effects(&observed)?;
    fs::write(
        output.join("m1-surface-effects-result.json"),
        "{\"schemaVersion\":1,\"result\":\"forward-and-deferred-ao-and-emissive-readbacks-matched-independent-oracles\",\"aoScope\":\"AO multiplies only environment diffuse irradiance; direct diffuse and specular remain unchanged\",\"emissiveScope\":\"linear RGB times two, alpha ignored, additive after direct and indirect\",\"finalTarget\":\"all cases are deliberately below RGBA8 saturation\",\"toleranceBudget\":\"Forward 3 codes: final UNORM quantization plus cross-backend f32 raster/BRDF rounding; Deferred 6 codes: that budget plus base/normal UNORM8 and Rgba16Float effects round-trips propagated through GGX\"}\n",
    )?;
    Ok(())
}

fn cases() -> [Case; 5] {
    [
        Case {
            name: "ao-one-environment",
            rgba: [0, 0, 0, 255],
            environment: true,
        },
        Case {
            name: "ao-zero-environment",
            rgba: [0, 0, 0, 0],
            environment: true,
        },
        Case {
            name: "emissive-no-environment",
            rgba: [64, 32, 16, 255],
            environment: false,
        },
        Case {
            name: "emissive-ao-zero-environment",
            rgba: [64, 32, 16, 0],
            environment: true,
        },
        Case {
            name: "combined-nonsaturated",
            rgba: [64, 32, 16, 255],
            environment: true,
        },
    ]
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(Vec::new(), "unused", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "tint_u".into() }], vec![
        MaterialResource { name: "tint_u".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
        MaterialResource { name: "effects_tex".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
        MaterialResource { name: "effects_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
    ], BTreeMap::new(), SurfaceOutput::MetallicRoughnessUv0Normal { properties: "properties".into(), normal: "normal".into() })
    .with_code_node(node("base", MaterialType::Vec4, "fn base(uv: vec2<f32>) -> vec4<f32> { return tint; }"))
    .with_code_node(node("properties", MaterialType::Vec2, "fn properties(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(0.0, 0.5); }"))
    .with_code_node(node("normal", MaterialType::Vec3, "fn normal(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0, 0.0, 1.0); }"))
    .with_code_node(node("ao", MaterialType::F32, "fn ao(uv: vec2<f32>) -> f32 { return textureSample(effects_tex, effects_sampler, uv).a; }"))
    .with_code_node(node("emission", MaterialType::Vec3, "fn emission(uv: vec2<f32>) -> vec3<f32> { return textureSample(effects_tex, effects_sampler, uv).rgb * 2.0; }"))
    .with_surface_extensions(SurfaceExtensions { occlusion: Some("ao".into()), emissive: Some("emission".into()) })
    .with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(61), callable: "base".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(61) })
}
fn node(name: &str, output: MaterialType, source: &str) -> MaterialCodeNode {
    MaterialCodeNode {
        name: name.into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output,
        calls: Vec::new(),
        source: SourceUnit::new(format!("m1-effects-{name}"), source),
        resources: Vec::new(),
    }
}
fn geometry() -> GeometryData {
    let mut bytes = Vec::new();
    for (position, uv) in [
        [-1., -1., 0.5],
        [1., -1., 0.5],
        [-1., 1., 0.5],
        [-1., 1., 0.5],
        [1., -1., 0.5],
        [1., 1., 0.5],
    ]
    .into_iter()
    .zip([[0., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [1., 1.]])
    {
        for x in position
            .into_iter()
            .chain([0.0_f32, 0.0, 1.0])
            .chain(uv)
            .chain([1.0, 0.0, 0.0, 1.0])
        {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 48,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "NORMAL".into(),
                        offset: 12,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "UV0".into(),
                        offset: 24,
                        format: VertexFormat::Float32(2),
                    },
                    GeometryAttribute {
                        semantic: "TANGENT".into(),
                        offset: 32,
                        format: VertexFormat::Float32(4),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..6,
    }
}
fn snapshot_geometry(
    s: &AssetStore<GeometryAsset, GeometryData, String>,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(geometry(), ResidentBytes::new(288))?),
        _ => Err("effects geometry producer missing".into()),
    }
}
fn snapshot_material(
    s: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(material(), ResidentBytes::new(0))?),
        _ => Err("effects material producer missing".into()),
    }
}
fn snapshot_texture(
    s: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    rgba: [u8; 4],
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(
            ColorTextureData {
                encoding: fluxel_renderer::TextureEncoding::Linear,
                mip_levels: 1,
                extent: [1, 1],
                rgba8: rgba.to_vec(),
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("effects texture producer missing".into()),
    }
}
fn snapshot_environment(
    s: &AssetStore<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>,
) -> Result<AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(
            EnvironmentIrradianceData {
                extent: [1, 1],
                rgba8: ENVIRONMENT.to_vec(),
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("effects environment producer missing".into()),
    }
}
fn scene(
    g: &AssetSnapshot<GeometryAsset, GeometryData>,
    m: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    t: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
) -> Result<
    (
        fluxel_renderer::FrameSnapshot<ForwardResources>,
        fluxel_renderer::RenderViewId,
        ForwardAssets,
    ),
    Box<dyn Error>,
> {
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(g),
            material: SceneAssetRef::from_snapshot(m),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "effects_tex",
                SceneAssetRef::from_snapshot(t),
            )],
            samplers: Vec::new(),
            tint: [BASE[0] as f32, BASE[1] as f32, BASE[2] as f32, 1.],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.; 3], [1.; 3])?,
    }));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    Ok((
        scene.snapshot(),
        receipt.inserted_views()[0],
        ForwardAssets {
            geometry: vec![g.clone()],
            materials: vec![m.clone()],
            textures: vec![t.clone()],
        },
    ))
}

struct Expected {
    linear: [f64; 3],
    encoded: [u8; 4],
}
fn oracle(c: Case, deferred: bool) -> Expected {
    let unorm = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0;
    let base = if deferred { BASE.map(unorm) } else { BASE };
    let normal = if deferred {
        normalize([
            unorm(0.5) * 2.0 - 1.0,
            unorm(0.5) * 2.0 - 1.0,
            unorm(1.0) * 2.0 - 1.0,
        ])
    } else {
        [0.0, 0.0, 1.0]
    };
    let ao = if deferred {
        half_round(unorm(f64::from(c.rgba[3]) / 255.0))
    } else {
        f64::from(c.rgba[3]) / 255.0
    };
    let direct = direct(base, normal);
    let indirect = if c.environment {
        [0, 1, 2].map(|i| f64::from(ENVIRONMENT[i]) / 255.0 * base[i])
    } else {
        [0.0; 3]
    };
    let emission = [c.rgba[0], c.rgba[1], c.rgba[2]].map(|v| {
        let value = 2.0 * f64::from(v) / 255.0;
        if deferred {
            half_round(value)
        } else {
            value
        }
    });
    let linear = [0, 1, 2].map(|i| direct[i] + indirect[i] * ao + emission[i]);
    Expected {
        linear,
        encoded: linear
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .map_alpha(255),
    }
}
fn direct(base: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let l = [0.6, 0.0, 0.8];
    let h = normalize([0.6, 0.0, 1.8]);
    let nl = dot(n, l).max(0.0);
    let nv = n[2].max(0.0);
    let nh = dot(n, h).max(0.0);
    let vh = h[2].max(0.0);
    let a2 = 0.5_f64.powi(4);
    let term = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (std::f64::consts::PI * term * term);
    let v =
        0.5 / (nl * (nv * nv * (1.0 - a2) + a2).sqrt() + nv * (nl * nl * (1.0 - a2) + a2).sqrt());
    base.map(|b| {
        let f = 0.04 + 0.96 * (1.0 - vh).powi(5);
        (((1.0 - f) * b / std::f64::consts::PI) + d * v * f) * nl * 3.0
    })
}

/// Models the independent IEEE-754 binary16 store/load boundary of Deferred's
/// effects attachment, without consulting renderer code or a GPU conversion.
fn half_round(value: f64) -> f64 {
    let value = value as f32;
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    let half = if exponent >= 31 {
        sign | 0x7c00
    } else if exponent <= 0 {
        if exponent < -10 {
            sign
        } else {
            let shifted = (mantissa | 0x80_0000) >> (14 - exponent);
            sign | ((shifted + ((shifted >> 13) & 1)) >> 13)
        }
    } else {
        let rounded = mantissa + 0x1000 + ((mantissa >> 13) & 1);
        if rounded & 0x80_0000 != 0 {
            sign | (((exponent + 1) as u32) << 10)
        } else {
            sign | ((exponent as u32) << 10) | (rounded >> 13)
        }
    };
    let exponent = ((half >> 10) & 0x1f) as i32;
    let mantissa = half & 0x3ff;
    let result = if exponent == 0 {
        (mantissa as f32) * 2_f32.powi(-24)
    } else if exponent == 31 {
        f32::INFINITY
    } else {
        (1.0 + mantissa as f32 / 1024.0) * 2_f32.powi(exponent - 15)
    };
    if half & 0x8000 != 0 {
        -f64::from(result)
    } else {
        f64::from(result)
    }
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn normalize(x: [f64; 3]) -> [f64; 3] {
    let d = dot(x, x).sqrt();
    x.map(|v| v / d)
}
trait Alpha {
    fn map_alpha(self, a: u8) -> [u8; 4];
}
impl Alpha for [u8; 3] {
    fn map_alpha(self, a: u8) -> [u8; 4] {
        [self[0], self[1], self[2], a]
    }
}
fn center(bytes: &[u8]) -> Result<[u8; 4], Box<dyn Error>> {
    let i = ((EXTENT[1] / 2 * EXTENT[0] + EXTENT[0] / 2) * 4) as usize;
    Ok(bytes[i..i + 4].try_into()?)
}
fn verify_center(
    actual: &[u8],
    expected: [u8; 4],
    tolerance: u8,
    name: &str,
    route: &str,
) -> Result<(), Box<dyn Error>> {
    let found = center(actual)?;
    if found[3] != 255
        || found[..3]
            .iter()
            .zip(&expected[..3])
            .any(|(a, e)| a.abs_diff(*e) > tolerance)
    {
        return Err(format!("{name} {route} centre {found:?} differs from independent surface-effects oracle {expected:?}").into());
    }
    Ok(())
}
fn require_nonsaturated(
    expected: &Expected,
    name: &str,
    route: &str,
) -> Result<(), Box<dyn Error>> {
    if expected
        .linear
        .iter()
        .any(|value| !(0.0..1.0).contains(value))
    {
        return Err(format!(
            "{name} {route} oracle reached final RGBA8 saturation; this workload requires observable non-saturated channels"
        )
        .into());
    }
    Ok(())
}
fn require_effects(found: &[(&str, [u8; 4])]) -> Result<(), Box<dyn Error>> {
    let get = |name| {
        found
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, x)| *x)
            .ok_or("missing case")
    };
    let ao_one = get("ao-one-environment")?;
    let ao_zero = get("ao-zero-environment")?;
    let no_environment = get("emissive-no-environment")?;
    let emissive_ao_zero = get("emissive-ao-zero-environment")?;
    let combined = get("combined-nonsaturated")?;
    let within = |left: [u8; 4], right: [u8; 4], tolerance| {
        left.iter()
            .zip(right)
            .all(|(left, right)| left.abs_diff(right) <= tolerance)
    };
    if within(ao_one, ao_zero, FORWARD_TOLERANCE)
        || within(no_environment, ao_one, FORWARD_TOLERANCE)
        || !within(no_environment, emissive_ao_zero, FORWARD_TOLERANCE)
        || within(combined, emissive_ao_zero, FORWARD_TOLERANCE)
    {
        Err("AO/emissive combination or no-environment/AO-zero control was not observable in Forward readback".into())
    } else {
        Ok(())
    }
}
fn input_json(cases: &[Case]) -> String {
    let cases = cases
        .iter()
        .map(|c| {
            format!(
                "{{\"name\":\"{}\",\"effectsRgba\":{:?},\"environment\":{}}}",
                c.name, c.rgba, c.environment
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"schemaVersion\":1,\"workload\":\"m1-surface-effects-forward-deferred\",\"extent\":{:?},\"baseUniform\":{:?},\"bindingLayout\":[\"g0b0 uniform\",\"g0b1 texture_2d<f32>\",\"g0b2 sampler\"],\"effectSourceWgsl\":{},\"deferredEffectsAttachment\":{},\"environmentEOverPi\":{:?},\"cases\":[{}]}}\n", EXTENT, BASE, r#"{\"occlusion\":\"textureSample(effects_tex, effects_sampler, uv).a\",\"emissive\":\"textureSample(effects_tex, effects_sampler, uv).rgb * 2.0\"}"#.replace("\\", ""), r#"{\"format\":\"Rgba16Float\",\"rgb\":\"emissive\",\"a\":\"occlusion\"}"#.replace("\\", ""), ENVIRONMENT, cases)
}
#[cfg(test)]
pub(crate) fn input_evidence_for_test() -> String {
    input_json(&cases())
}
/// Exposes the production oracle outputs to the external test module while the
/// expected values remain independently fixed there.  This keeps test code out
/// of the workload module and avoids turning the JSON wording into its oracle.
#[cfg(test)]
pub(crate) fn oracle_codes_for_test() -> Vec<(&'static str, [u8; 4], [u8; 4])> {
    cases()
        .into_iter()
        .map(|case| {
            (
                case.name,
                oracle(case, false).encoded,
                oracle(case, true).encoded,
            )
        })
        .collect()
}
fn oracle_json(c: Case, f: &Expected, d: &Expected) -> String {
    format!(
        "{{\"schemaVersion\":1,\"case\":\"{}\",\"forwardLinearUnclamped\":{:?},\"forwardEncoded\":{:?},\"forwardToleranceCodes\":{},\"deferredLinearUnclamped\":{:?},\"deferredEncoded\":{:?},\"deferredToleranceCodes\":{},\"deferredModel\":\"RGBA8 UNORM G-buffer base and normal; Rgba16Float effects RGB emissive/AO alpha with independent binary16 round-trip\",\"toleranceBudget\":\"Forward 3 codes from final UNORM plus f32 BRDF/raster rounding; Deferred 6 adds UNORM8 G-buffer and binary16 effects propagation\",\"finalTarget\":\"non-saturated RGBA8 UNORM\"}}\n",
        c.name, f.linear, f.encoded, FORWARD_TOLERANCE, d.linear, d.encoded, DEFERRED_TOLERANCE
    )
}
