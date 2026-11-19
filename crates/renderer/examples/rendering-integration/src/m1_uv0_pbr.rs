//! M1 real-device UV0 metallic-roughness material proof.
//!
//! A six-vertex screen quad carries spatially varying UV0.  One material node
//! graph routes that UV0 to a CodeNode which samples a non-constant texture;
//! the same immutable scene snapshot is then read back through Forward and
//! Deferred.  The CPU oracle deliberately implements linear sampling and the
//! documented direct-light equation itself rather than reusing renderer code.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
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

use crate::block_on;

const EXTENT: [u32; 2] = [64, 64];
const TEXTURE: [u8; 16] = [
    // Public texture row zero (top): dark blue through warm orange.
    32, 48, 80, 255, 224, 144, 48, 255,
    // Public texture row one (bottom): green through violet.
    48, 200, 160, 255, 196, 48, 210, 255,
];
// Raw readback row zero is the RHI Core top row. The top of the fullscreen
// quad has UV0.y = 1, so the independent oracle flips readback Y before
// texture sampling. Both DX12 and WGL are expected to honor this Core order.
const PROBES: [(u32, u32); 4] = [(8, 8), (40, 8), (8, 56), (40, 56)];

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
    let geometry = snapshot_geometry(&geometries)?;
    let material = snapshot_material(&materials)?;
    let texture = snapshot_texture(&textures)?;
    fs::write(
        output.join("m1-uv0-pbr-geometry.txt"),
        format!("{:#?}\n", geometry.value()),
    )?;
    fs::write(
        output.join("m1-uv0-pbr-material.txt"),
        format!("{:#?}\n", material.value()),
    )?;
    fs::write(output.join("m1-uv0-pbr-texture.rgba"), TEXTURE)?;

    let probes = expected_probes();
    fs::write(output.join("m1-uv0-pbr-oracle.json"), oracle_json(&probes))?;
    fs::write(
        output.join("m1-uv0-pbr-input.json"),
        include_str!("../assets/m1-uv0-pbr-input.json"),
    )?;

    let record = RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&texture),
            )],
            samplers: Vec::new(),
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.0; 3], [1.0; 3])?,
    };
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let assets = ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: vec![texture],
    };
    let view = receipt.inserted_views()[0];
    let snapshot = scene.snapshot();
    let visibility = RenderBounds::new([-1.0; 3], [1.0; 3])?;
    let mut renderer = Renderer::new(device)?;

    let deferred = block_on(renderer.submit_deferred(
        &ExternalDeferredPipeline::new(),
        snapshot.clone(),
        view,
        visibility,
        &assets,
    ))?;
    require_pending(&renderer, 1, "Deferred submit")?;
    let deferred_actual = block_on(renderer.readback(&deferred))?;
    fs::write(
        output.join("m1-uv0-pbr-deferred-readback.rgba"),
        &deferred_actual,
    )?;
    fs::write(
        output.join("m1-uv0-pbr-deferred-readback.ppm"),
        ppm(&deferred_actual)?,
    )?;
    require_pending(&renderer, 0, "Deferred readback")?;
    verify_probes(&deferred_actual, &probes, 5, "Deferred")?;

    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        visibility,
        &assets,
    ))?;
    require_pending(&renderer, 1, "Forward submit")?;
    let forward_actual = block_on(renderer.readback(&forward))?;
    fs::write(
        output.join("m1-uv0-pbr-forward-readback.rgba"),
        &forward_actual,
    )?;
    fs::write(
        output.join("m1-uv0-pbr-forward-readback.ppm"),
        ppm(&forward_actual)?,
    )?;
    require_pending(&renderer, 0, "Forward readback")?;
    verify_probes(&forward_actual, &probes, 3, "Forward")?;
    fs::write(
        output.join("m1-uv0-pbr-result.json"),
        "{\"schemaVersion\":1,\"result\":\"forward-and-deferred-uv0-pbr-probes-matched-independent-oracle\",\"scope\":\"same node graph and CodeNode; ordinary sampled nonconstant texture; no transparency, IBL, shadows, or resolve claim\"}\n",
    )?;
    Ok(())
}

fn require_pending<B>(
    renderer: &Renderer<B>,
    expected: usize,
    stage: &str,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    if renderer.pending() == expected {
        Ok(())
    } else {
        Err(format!("M1 UV0 PBR {stage} pending count differed from expected {expected}").into())
    }
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(Vec::new(), "unused", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }], vec![
        MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
        MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
        MaterialResource { name: "material_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
    ], BTreeMap::new(), SurfaceOutput::MetallicRoughnessUv0 { properties: "m1_properties".into() })
    .with_code_node(MaterialCodeNode { name: "m1_sample".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec4, calls: Vec::new(), source: SourceUnit::new("m1-uv0-sample", "fn m1_sample(uv: vec2<f32>) -> vec4<f32> { return textureSample(material_texture, material_sampler, uv) * tint; }"), resources: Vec::new() })
    .with_code_node(MaterialCodeNode { name: "m1_properties".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec2, calls: Vec::new(), source: SourceUnit::new("m1-uv0-properties", "fn m1_properties(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(0.0, 0.5); }"), resources: Vec::new() })
    .with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(17), callable: "m1_sample".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(17) })
}

fn geometry() -> GeometryData {
    let positions: [[f32; 3]; 6] = [
        [-1., -1., 0.5],
        [1., -1., 0.5],
        [-1., 1., 0.5],
        [-1., 1., 0.5],
        [1., -1., 0.5],
        [1., 1., 0.5],
    ];
    let uvs: [[f32; 2]; 6] = [[0., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [1., 1.]];
    let mut bytes = Vec::new();
    for (position, uv) in positions.into_iter().zip(uvs) {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0_f32, 0., 1.] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in uv {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 32,
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
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..6,
    }
}

fn snapshot_geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
) -> Result<fluxel_assets::AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(geometry(), ResidentBytes::new(192))?),
        _ => Err("M1 geometry producer missing".into()),
    }
}
fn snapshot_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(material(), ResidentBytes::new(0))?),
        _ => Err("M1 material producer missing".into()),
    }
}
fn snapshot_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<fluxel_assets::AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(
            ColorTextureData {
                encoding: fluxel_renderer::TextureEncoding::Linear,
                mip_levels: 1,
                extent: [2, 2],
                rgba8: TEXTURE.to_vec(),
            },
            ResidentBytes::new(16),
        )?),
        _ => Err("M1 texture producer missing".into()),
    }
}

#[derive(Clone, Copy)]
struct Probe {
    x: u32,
    y: u32,
    expected: [u8; 4],
}
fn expected_probes() -> Vec<Probe> {
    PROBES
        .into_iter()
        .map(|(x, y)| Probe {
            x,
            y,
            expected: pbr(linear_sample(
                (x as f64 + 0.5) / f64::from(EXTENT[0]),
                1.0 - (y as f64 + 0.5) / f64::from(EXTENT[1]),
            )),
        })
        .collect()
}
fn linear_sample(u: f64, v: f64) -> [f64; 3] {
    let x = (u * 2.0 - 0.5).clamp(0.0, 1.0);
    let y = (v * 2.0 - 0.5).clamp(0.0, 1.0);
    let top_left = [32., 48., 80.];
    let top_right = [224., 144., 48.];
    let bottom_left = [48., 200., 160.];
    let bottom_right = [196., 48., 210.];
    [0, 1, 2].map(|channel| {
        let top = top_left[channel] + (top_right[channel] - top_left[channel]) * x;
        let bottom = bottom_left[channel] + (bottom_right[channel] - bottom_left[channel]) * x;
        (top + (bottom - top) * y) / 255.0
    })
}
fn pbr(base: [f64; 3]) -> [u8; 4] {
    let rough = 0.5_f64;
    let alpha2 = rough.powi(4);
    let no_l = 0.8_f64;
    let no_v = 1.0_f64;
    let no_h = ((0.8_f64 + 1.0) / 2.0).sqrt();
    let vo_h = no_h;
    let term = no_h * no_h * (alpha2 - 1.) + 1.;
    let distribution = alpha2 / (std::f64::consts::PI * term * term);
    let visibility = 0.5
        / (no_l * (no_v * no_v * (1. - alpha2) + alpha2).sqrt()
            + no_v * (no_l * no_l * (1. - alpha2) + alpha2).sqrt());
    let fresnel = 0.04 + 0.96 * (1. - vo_h).powi(5);
    let scale = ((1. - fresnel) / std::f64::consts::PI) * no_l * 3.;
    let spec = distribution * visibility * fresnel * no_l * 3.;
    [
        ((base[0] * scale + spec).clamp(0., 1.) * 255.).round() as u8,
        ((base[1] * scale + spec).clamp(0., 1.) * 255.).round() as u8,
        ((base[2] * scale + spec).clamp(0., 1.) * 255.).round() as u8,
        255,
    ]
}
fn verify_probes(
    actual: &[u8],
    probes: &[Probe],
    tolerance: u8,
    route: &str,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != (EXTENT[0] * EXTENT[1] * 4) as usize {
        return Err(format!("M1 {route} readback length is invalid").into());
    }
    for probe in probes {
        let index = ((probe.y * EXTENT[0] + probe.x) * 4) as usize;
        let found: [u8; 4] = actual[index..index + 4].try_into()?;
        if found[3] != 255
            || found[..3]
                .iter()
                .zip(&probe.expected[..3])
                .any(|(a, e)| a.abs_diff(*e) > tolerance)
        {
            return Err(format!(
                "M1 {route} probe x={}, y={} actual={found:?} expected={:?}",
                probe.x, probe.y, probe.expected
            )
            .into());
        }
    }
    if probes
        .windows(2)
        .any(|pair| pair[0].expected == pair[1].expected)
    {
        return Err("M1 oracle probes are not spatially distinct".into());
    }
    Ok(())
}
fn oracle_json(probes: &[Probe]) -> String {
    format!(
        "{{\"schemaVersion\":1,\"readbackOrigin\":\"top-left Core row zero\",\"uvAtReadbackPixel\":\"u=(x+0.5)/width; v=1-(y+0.5)/height\",\"sampler\":\"clamp-to-edge linear no-mip\",\"directLight\":\"independent CPU implementation\",\"forwardTolerance\":3,\"deferredTolerance\":5,\"probes\":[{}]}}\n",
        probes
            .iter()
            .map(|p| format!(
                "{{\"x\":{},\"y\":{},\"expected\":{:?}}}",
                p.x, p.y, p.expected
            ))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() != (EXTENT[0] * EXTENT[1] * 4) as usize {
        return Err("M1 image byte count invalid".into());
    }
    let mut out = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        out.extend_from_slice(&pixel[..3]);
    }
    Ok(out)
}
