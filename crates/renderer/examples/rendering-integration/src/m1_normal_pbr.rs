//! M1 normal-map fixture.  GPU execution is deliberately kept separate from
//! the UV0 color fixture: its three cases retain flat, positive- and
//! negative-handed tangent inputs for later DX12/WGL evidence attempts.

use crate::block_on;
use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout,
    GeometryStream, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, Renderer, SceneAssetRef,
};
use fluxel_rhi::{
    api::{Device, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::SourceUnit;
use std::{collections::BTreeMap, error::Error, fs, path::Path};

const EXTENT: [u32; 2] = [64, 64];
const FLAT: [u8; 4] = [128, 128, 255, 255];
const MAPPED: [u8; 4] = [204, 64, 230, 255];
pub(crate) fn run<B>(device: Device<B>, out: &Path) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(out)?;
    let gs = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let ms = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let ts = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let mat = snap_mat(&ms)?;
    let flat = snap_tex(&ts, FLAT)?;
    let map = snap_tex(&ts, MAPPED)?;
    fs::write(
        out.join("m1-normal-material.txt"),
        format!("{:#?}", mat.value()),
    )?;
    let cases = [
        ("flat", FLAT, 1., false),
        ("mapped-handed-positive", MAPPED, 1., false),
        ("mapped-handed-negative", MAPPED, -1., false),
        ("mapped-negative-determinant", MAPPED, 1., true),
    ];
    let mut observed = Vec::new();
    let mut renderer = Renderer::new(device)?;
    for (name, tex, w, mirrored) in cases {
        let is_flat = tex == FLAT;
        let geo = snap_geo(&gs, w)?;
        let tex = if tex == FLAT {
            flat.clone()
        } else {
            map.clone()
        };
        let record = RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&geo),
                material: SceneAssetRef::from_snapshot(&mat),
                textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                    "normal_tex",
                    SceneAssetRef::from_snapshot(&tex),
                )],
                samplers: Vec::new(),
                tint: [1.; 4],
            },
            transform: transform(mirrored)?,
            bounds: RenderBounds::new([-2.; 3], [2.; 3])?,
        };
        let mut scene = RenderScene::new()?;
        let mut c = RenderChangeSet::new();
        c.push(RenderChange::Insert(record));
        c.push(RenderChange::InsertView(RenderView::new(
            RenderTransform::identity(),
            EXTENT,
        )?));
        let r = scene.apply(&c)?;
        let view = r.inserted_views()[0];
        let assets = ForwardAssets {
            geometry: vec![geo],
            materials: vec![mat.clone()],
            textures: vec![flat.clone(), map.clone()],
        };
        let snap = scene.snapshot();
        let bounds = RenderBounds::new([-2.; 3], [2.; 3])?;
        let expected = oracle(w, mirrored, is_flat);
        let deferred_expected = deferred_oracle(w, mirrored, is_flat);
        // Persist the actual input and independent oracle before either route
        // can submit. A failed submission therefore still has auditable input.
        fs::write(
            out.join(format!("m1-normal-{name}-input.json")),
            format!(
                "{{\"schemaVersion\":1,\"case\":\"{name}\",\"normalTexel\":{:?},\"tangent\":[1.0,0.0,0.0,{}],\"worldColumns\":[[{}, {},0.0,0.0],[-0.707,0.707,0.0,0.0],[0.0,0.0,0.5,0.0],[0.0,0.0,0.0,1.0]],\"determinantSign\":{},\"forwardExpectedCenter\":{:?},\"deferredExpectedCenter\":{:?},\"forwardTolerance\":3,\"deferredTolerance\":8}}\n",
                tex.value().rgba8,
                w,
                if mirrored { -1.414 } else { 1.414 },
                if mirrored { -1.414 } else { 1.414 },
                if mirrored { -1 } else { 1 },
                expected,
                deferred_expected
            ),
        )?;
        fs::write(
            out.join(format!("m1-normal-{name}-geometry.bin")),
            &assets.geometry[0].value().streams[0],
        )?;
        fs::write(
            out.join(format!("m1-normal-{name}-oracle.json")),
            format!(
                "{{\"forward\":{:?},\"deferred\":{:?},\"deferredNormalPath\":\"world normal -> UNORM8 encode -> decode -> normalize -> direct light\"}}\n",
                expected, deferred_expected
            ),
        )?;
        let d = block_on(renderer.submit_deferred(
            &ExternalDeferredPipeline::new(),
            snap.clone(),
            view,
            bounds,
            &assets,
        ))?;
        let db = block_on(renderer.readback(&d))?;
        fs::write(out.join(format!("m1-normal-{name}-deferred.rgba")), &db)?;
        let f = block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snap,
            view,
            bounds,
            &assets,
        ))?;
        let fb = block_on(renderer.readback(&f))?;
        fs::write(out.join(format!("m1-normal-{name}-forward.rgba")), &fb)?;
        verify_center(&fb, expected, 3, name, "Forward")?;
        verify_center(&db, deferred_expected, 8, name, "Deferred")?;
        observed.push((name, fb, db));
    }
    if observed[0].1 == observed[1].1
        || observed[1].1 == observed[2].1
        || observed[1].1 == observed[3].1
    {
        return Err("normal-map flat or handedness cases did not change Forward readback".into());
    };
    fs::write(
        out.join("m1-normal-input.json"),
        "{\"schemaVersion\":1,\"oracle\":\"N=normalize(normalMatrix*Nlocal); T=normalize(worldLinear*Tlocal-N*dot(N,worldLinear*Tlocal)); B=cross(N,T)*(handedness*sign(det(worldLinear))); world=normalize(T*sample.x+B*sample.y+N*sample.z)\",\"transform\":\"nonuniform-scale-plus-z-rotation; fourth case mirrors X\",\"cases\":[\"flat\",\"mapped-handed-positive\",\"mapped-handed-negative\",\"mapped-negative-determinant\"]}\n",
    )?;
    fs::write(
        out.join("m1-normal-result.json"),
        "{\"schemaVersion\":1,\"result\":\"normal-map-and-handedness-forward-deferred-readbacks-recorded\"}\n",
    )?;
    Ok(())
}
fn material() -> MaterialDefinition {
    MaterialDefinition::new(Vec::new(),"unused",vec![MaterialParameter{name:"tint".into(),ty:MaterialType::Vec4,resource:"tint_u".into()}],vec![MaterialResource{name:"tint_u".into(),kind:MaterialResourceKind::UniformVec4,location:BindingLocation{group:0,binding:0}},MaterialResource{name:"normal_tex".into(),kind:MaterialResourceKind::Texture2dF32,location:BindingLocation{group:0,binding:1}},MaterialResource{name:"normal_sampler".into(),kind:MaterialResourceKind::Sampler{comparison:false},location:BindingLocation{group:0,binding:2}}],BTreeMap::new(),SurfaceOutput::MetallicRoughnessUv0Normal{properties:"props".into(),normal:"normal".into()}).with_code_node(node("base",MaterialType::Vec4,"fn base(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.65,0.35,0.15,1.0)*tint; }")).with_code_node(node("props",MaterialType::Vec2,"fn props(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(0.0,0.5); }")).with_code_node(node("normal",MaterialType::Vec3,"fn normal(uv: vec2<f32>) -> vec3<f32> { return textureSample(normal_tex, normal_sampler, uv).xyz*2.0-vec3<f32>(1.0); }")).with_node_graph(MaterialNodeGraph{nodes:vec![MaterialNode{id:MaterialNodeId(31),callable:"base".into(),inputs:vec![MaterialNodeInput{input:"uv".into(),source:MaterialNodeInputSource::Uv0}]}],output:MaterialNodeId(31)})
}
fn node(n: &str, o: MaterialType, s: &str) -> MaterialCodeNode {
    MaterialCodeNode {
        name: n.into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output: o,
        calls: Vec::new(),
        source: SourceUnit::new(format!("m1-normal-{n}"), s),
        resources: Vec::new(),
    }
}
fn geo(w: f32) -> GeometryData {
    let mut b = Vec::new();
    for (p, u) in [
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
        for x in p {
            b.extend((x as f32).to_le_bytes())
        }
        for x in [0., 0., 1.] {
            b.extend((x as f32).to_le_bytes())
        }
        for x in u {
            b.extend((x as f32).to_le_bytes())
        }
        for x in [1., 0., 0., w] {
            b.extend((x as f32).to_le_bytes())
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
        streams: vec![b],
        vertices: 0..6,
    }
}
fn snap_geo(
    s: &AssetStore<GeometryAsset, GeometryData, String>,
    w: f32,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(geo(w), ResidentBytes::new(288))?),
        _ => Err("geometry producer missing".into()),
    }
}
fn snap_mat(
    s: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(material(), ResidentBytes::new(0))?),
        _ => Err("material producer missing".into()),
    }
}
fn snap_tex(
    s: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    v: [u8; 4],
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(
            ColorTextureData {
                encoding: fluxel_renderer::TextureEncoding::Linear,
                mip_levels: 1,
                extent: [1, 1],
                rgba8: v.to_vec(),
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("texture producer missing".into()),
    }
}
fn transform(mirrored: bool) -> Result<RenderTransform, Box<dyn Error>> {
    let mirror = if mirrored { -1.0 } else { 1.0 };
    Ok(RenderTransform::from_columns([
        [1.414 * mirror, 1.414 * mirror, 0., 0.],
        [-0.707, 0.707, 0., 0.],
        [0., 0., 0.5, 0.],
        [0., 0., 0., 1.],
    ])?)
}

/// Independent world-space TBN and direct-light oracle for the centre pixel.
/// It intentionally uses the transform's mathematical columns rather than
/// renderer shader helpers: N is inverse-transpose transformed, T is linear
/// transformed and orthogonalized, and the determinant sign participates in B.
fn world_normal(handedness: f32, mirrored: bool, flat: bool) -> [f32; 3] {
    let sign = if mirrored { -1.0 } else { 1.0 };
    let t = [sign * 0.70710677, sign * 0.70710677, 0.0];
    let b = [-t[1] * handedness * sign, t[0] * handedness * sign, 0.0];
    let s = if flat {
        [0.0039216, 0.0039216, 1.0]
    } else {
        [0.6_f32, -0.49803922, 0.8039216]
    };
    normalize([t[0] * s[0] + b[0] * s[1], t[1] * s[0] + b[1] * s[1], s[2]])
}

fn oracle(handedness: f32, mirrored: bool, flat: bool) -> [u8; 4] {
    shade(
        world_normal(handedness, mirrored, flat),
        [0.65, 0.35, 0.15],
        0.5,
    )
}

fn shade(n: [f32; 3], base: [f32; 3], roughness: f32) -> [u8; 4] {
    let l = [0.6_f32, 0.0, 0.8];
    let no_l = dot(n, l).max(0.0);
    let no_v = n[2].max(0.0);
    let h = normalize([0.6, 0.0, 1.8]);
    let no_h = dot(n, h).max(0.0);
    let vo_h = h[2].max(0.0);
    let a2 = roughness.powi(4);
    let term = no_h * no_h * (a2 - 1.0) + 1.0;
    let d = a2 / (std::f32::consts::PI * term * term);
    let v = 0.5
        / (no_l * (no_v * no_v * (1.0 - a2) + a2).sqrt()
            + no_v * (no_l * no_l * (1.0 - a2) + a2).sqrt());
    let f = 0.04 + 0.96 * (1.0 - vo_h).powi(5);
    let scale = (1.0 - f) * (1.0 / std::f32::consts::PI) * no_l * 3.0;
    let spec = d * v * f * no_l * 3.0;
    base.map(|c| ((c * scale + spec).clamp(0.0, 1.0) * 255.0).round() as u8)
        .map_alpha(255)
}
fn deferred_oracle(handedness: f32, mirrored: bool, flat: bool) -> [u8; 4] {
    let unorm = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() / 255.0;
    let n = world_normal(handedness, mirrored, flat)
        .map(|component| unorm(component * 0.5 + 0.5) * 2.0 - 1.0);
    shade(normalize(n), [0.65, 0.35, 0.15].map(unorm), unorm(0.5))
}
trait Alpha {
    fn map_alpha(self, a: u8) -> [u8; 4];
}
impl Alpha for [u8; 3] {
    fn map_alpha(self, a: u8) -> [u8; 4] {
        [self[0], self[1], self[2], a]
    }
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let d = dot(v, v).sqrt();
    [v[0] / d, v[1] / d, v[2] / d]
}
fn verify_center(
    actual: &[u8],
    expected: [u8; 4],
    tol: u8,
    case: &str,
    route: &str,
) -> Result<(), Box<dyn Error>> {
    let i = ((EXTENT[1] / 2 * EXTENT[0] + EXTENT[0] / 2) * 4) as usize;
    let got: [u8; 4] = actual[i..i + 4].try_into()?;
    if got[..3]
        .iter()
        .zip(expected[..3].iter())
        .any(|(a, b)| a.abs_diff(*b) > tol)
    {
        return Err(format!(
            "{case} {route} centre {got:?} differs from independent TBN/PBR oracle {expected:?}"
        )
        .into());
    }
    Ok(())
}
