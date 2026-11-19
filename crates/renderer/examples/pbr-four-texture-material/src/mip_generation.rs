//! Explicit-LOD consumer for GPU-generated RGBA8 mip chains.
//!
//! This deliberately narrow acceptance input owns one Linear 4x4 base level.
//! The renderer receives those sixteen texels and `mip_levels == 3`; the CPU
//! oracle creates no upload payload for the generated levels.

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::*;
use fluxel_renderer::*;
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::SourceUnit;
use std::{collections::BTreeMap, error::Error, fs, path::Path};

use crate::scene::EXTENT;

pub const TOLERANCE: u8 = 2;
pub const MIP_LEVELS: u32 = 3;
pub const RESOURCE: &str = "mip_image";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Case {
    Lod0,
    Lod1,
    Lod15,
    Lod2,
    SourceOnly,
}

impl Case {
    pub const ALL: [Self; 5] = [
        Self::Lod0,
        Self::Lod1,
        Self::Lod15,
        Self::Lod2,
        Self::SourceOnly,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Lod0 => "lod0",
            Self::Lod1 => "lod1",
            Self::Lod15 => "lod1-5",
            Self::Lod2 => "lod2",
            Self::SourceOnly => "source-only",
        }
    }

    fn lod(self) -> f32 {
        match self {
            Self::Lod0 => 0.,
            Self::Lod1 | Self::SourceOnly => 1.,
            Self::Lod15 => 1.5,
            Self::Lod2 => 2.,
        }
    }

    fn changed_source(self) -> bool {
        self == Self::SourceOnly
    }
}

pub struct Input {
    pub snapshot: FrameSnapshot<ForwardResources>,
    pub view: RenderViewId,
    pub assets: ForwardAssets,
    pub geometry_bytes: Vec<u8>,
    pub material: MaterialDefinition,
    pub case: Case,
    pub source: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Probe {
    pub x: u32,
    pub y: u32,
    pub uv: [f64; 2],
    pub expected: [u8; 4],
}

/// Makes an unlit, complete POSITION/UV0 quad. `tint.x` is deliberately the
/// explicit shader LOD, not a color multiplier.
pub fn input(case: Case) -> Result<Input, Box<dyn Error>> {
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let geometry_bytes = quad_bytes();
    let geometry = commit(
        &geometry_store,
        geometry_data(geometry_bytes.clone()),
        geometry_bytes.len() as u64,
    )?;
    let material = definition();
    let material_snapshot = commit(&material_store, material.clone(), 0)?;
    let source = source_bytes(case.changed_source());
    let texture = commit(
        &texture_store,
        ColorTextureData {
            extent: [4, 4],
            encoding: TextureEncoding::Linear,
            mip_levels: MIP_LEVELS,
            rgba8: source.clone(),
        },
        source.len() as u64,
    )?;
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material_snapshot),
            textures: vec![ForwardTextureBinding::asset(
                RESOURCE,
                SceneAssetRef::from_snapshot(&texture),
            )],
            samplers: Vec::new(),
            tint: [case.lod(), 1., 1., 1.],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.; 3], [1.; 3])?,
    }));
    let receipt = scene.apply(&changes)?;
    Ok(Input {
        snapshot: scene.snapshot(),
        view: receipt.inserted_views()[0],
        assets: ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material_snapshot],
            textures: vec![texture],
        },
        geometry_bytes,
        material,
        case,
        source,
    })
}

pub fn reference(input: &Input) -> Result<Vec<Probe>, String> {
    if input.source.len() != 4 * 4 * 4 || input.assets.textures[0].value().mip_levels != MIP_LEVELS
    {
        return Err("mip oracle requires the one 4x4 base-level-only source".into());
    }
    let levels = generated_levels(&input.source, [4, 4]);
    if levels.len() != MIP_LEVELS as usize {
        return Err("mip chain length differs from the declared chain".into());
    }
    let targets = [
        [0.125, 0.125],
        [0.875, 0.125],
        [0.125, 0.875],
        [0.875, 0.875],
        [0.375, 0.625],
    ];
    targets
        .iter()
        .map(|&target| {
            let (x, y, uv) = pixel_for_uv(target);
            Ok(Probe {
                x,
                y,
                uv,
                expected: trilinear(&levels, input.case.lod(), uv),
            })
        })
        .collect()
}

pub fn verify(raw: &[u8], probes: &[Probe]) -> Result<(), Box<dyn Error>> {
    if raw.len() != (EXTENT[0] * EXTENT[1] * 4) as usize {
        return Err("mip readback byte length mismatch".into());
    }
    for probe in probes {
        let offset = ((probe.y * EXTENT[0] + probe.x) * 4) as usize;
        let found = &raw[offset..offset + 4];
        if found[..3]
            .iter()
            .zip(probe.expected)
            .any(|(a, b)| a.abs_diff(b) > TOLERANCE)
        {
            return Err(format!(
                "mip probe ({},{}) UV {:?} found {found:?}, expected {:?}, tolerance {TOLERANCE}",
                probe.x, probe.y, probe.uv, probe.expected
            )
            .into());
        }
    }
    Ok(())
}

/// Saves facts before GPU work: only mip0 bytes, exact asset generation,
/// material/geometry, LOD, expected probes and the package lock file.
pub fn write_inputs(output: &Path, input: &Input, probes: &[Probe]) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output)?;
    fs::write(output.join("geometry.bin"), &input.geometry_bytes)?;
    fs::write(output.join("mip0-source.rgba"), &input.source)?;
    fs::write(
        output.join("material.txt"),
        format!("{:#?}\n", input.material),
    )?;
    fs::write(output.join("Cargo.lock"), include_bytes!("../Cargo.lock"))?;
    let geometry = &input.assets.geometry[0];
    let material = &input.assets.materials[0];
    fs::write(
        output.join("owner-identities.json"),
        format!(
            "{{\"geometry\":{{\"id\":{:?},\"generation\":{}}},\"material\":{{\"id\":{:?},\"generation\":{}}}}}\n",
            format!("{:?}", geometry.id()),
            geometry.generation().get(),
            format!("{:?}", material.id()),
            material.generation().get(),
        ),
    )?;
    let texture = &input.assets.textures[0];
    fs::write(
        output.join("mip-input.json"),
        format!(
            "{{\"case\":{:?},\"extent\":[4,4],\"encoding\":\"Linear\",\"mipLevels\":{MIP_LEVELS},\"basePayloadBytes\":{},\"texture\":{{\"id\":{:?},\"generation\":{}}},\"tint\":[{},1,1,1],\"vertexStride\":20,\"sampling\":\"textureSampleLevel; linear mag/min/mip, clamp-to-edge; GPU blits mip0->1->2\"}}\n",
            input.case.name(),
            input.source.len(),
            format!("{:?}", texture.id()),
            texture.generation().get(),
            input.case.lod()
        ),
    )?;
    let entries = probes
        .iter()
        .map(|p| {
            format!(
                "{{\"x\":{},\"y\":{},\"uv\":{:?},\"expected\":{:?}}}",
                p.x, p.y, p.uv, p.expected
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        output.join("mip-oracle.json"),
        format!(
            "{{\"rgbTolerance\":{TOLERANCE},\"independent\":\"quantized iterative linear box mip generation, bilinear and trilinear sampling\",\"probes\":[{entries}]}}\n"
        ),
    )?;
    Ok(())
}

pub fn definition() -> MaterialDefinition {
    MaterialDefinition::new(
        Vec::new(), "unused",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "mip_lod".into() }],
        vec![
            MaterialResource { name: "mip_lod".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: RESOURCE.into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "mip_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
        ],
        BTreeMap::new(), SurfaceOutput::ColorUv0,
    )
    .with_code_node(MaterialCodeNode {
        name: "sample_explicit_mip".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec4,
        calls: Vec::new(), resources: Vec::new(),
        source: SourceUnit::new("renderer-mip-explicit-lod", "fn sample_explicit_mip(uv: vec2<f32>) -> vec4<f32> { return textureSampleLevel(mip_image, mip_sampler, uv, tint.x); }"),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sample_explicit_mip".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }],
        output: MaterialNodeId(1),
    })
}

fn geometry_data(bytes: Vec<u8>) -> GeometryData {
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 20,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "UV0".into(),
                        offset: 12,
                        format: VertexFormat::Float32(2),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..6,
    }
}

fn quad_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(6 * 20);
    for (position, uv) in [
        ([-1.0_f32, -1., 0.], [0., 1.]),
        ([1., -1., 0.], [1., 1.]),
        ([-1., 1., 0.], [0., 0.]),
        ([-1., 1., 0.], [0., 0.]),
        ([1., -1., 0.], [1., 1.]),
        ([1., 1., 0.], [1., 0.]),
    ] {
        for value in position.into_iter().chain(uv) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn source_bytes(changed: bool) -> Vec<u8> {
    // The top-left 2x2 is deliberately nonconstant: at UV .125, LOD0 reads
    // its first texel while LOD1 reads the generated four-texel average.
    let blocks = if changed {
        [
            [240, 24, 32, 255],
            [20, 200, 40, 255],
            [32, 48, 220, 255],
            [224, 224, 80, 255],
        ]
    } else {
        [
            [80, 24, 32, 255],
            [20, 200, 40, 255],
            [32, 48, 220, 255],
            [224, 224, 80, 255],
        ]
    };
    let mut bytes = Vec::with_capacity(64);
    for y in 0..4 {
        for x in 0..4 {
            let pixel = if x < 2 && y < 2 {
                match (x, y) {
                    (0, 0) => blocks[0],
                    (1, 0) => [192, 48, 80, 255],
                    (0, 1) => [48, 160, 112, 255],
                    _ => [128, 96, 208, 255],
                }
            } else {
                blocks[(y / 2) * 2 + x / 2]
            };
            bytes.extend_from_slice(&pixel);
        }
    }
    bytes
}

fn generated_levels(base: &[u8], extent: [u32; 2]) -> Vec<([u32; 2], Vec<u8>)> {
    let mut levels = vec![(extent, base.to_vec())];
    while levels.last().is_some_and(|(size, _)| *size != [1, 1]) {
        let (old_size, old) = levels.last().unwrap();
        let size = [(old_size[0] / 2).max(1), (old_size[1] / 2).max(1)];
        let mut next = Vec::with_capacity((size[0] * size[1] * 4) as usize);
        for y in 0..size[1] {
            for x in 0..size[0] {
                for channel in 0..4 {
                    let mut sum = 0u32;
                    for oy in 0..2 {
                        for ox in 0..2 {
                            let sx = (x * 2 + ox).min(old_size[0] - 1);
                            let sy = (y * 2 + oy).min(old_size[1] - 1);
                            sum += u32::from(old[((sy * old_size[0] + sx) * 4 + channel) as usize]);
                        }
                    }
                    next.push(((sum + 2) / 4) as u8);
                }
            }
        }
        levels.push((size, next));
    }
    levels
}

fn trilinear(levels: &[([u32; 2], Vec<u8>)], lod: f32, uv: [f64; 2]) -> [u8; 4] {
    let low = lod.floor().clamp(0., (levels.len() - 1) as f32) as usize;
    let high = lod.ceil().clamp(0., (levels.len() - 1) as f32) as usize;
    let mix = f64::from(lod - low as f32);
    let a = bilinear(&levels[low].1, levels[low].0, uv);
    let b = bilinear(&levels[high].1, levels[high].0, uv);
    std::array::from_fn(|c| (a[c] * (1. - mix) + b[c] * mix).round() as u8)
}

fn bilinear(bytes: &[u8], extent: [u32; 2], uv: [f64; 2]) -> [f64; 4] {
    let p = [
        uv[0] * f64::from(extent[0]) - 0.5,
        uv[1] * f64::from(extent[1]) - 0.5,
    ];
    let floor = [p[0].floor() as i32, p[1].floor() as i32];
    let weight = [p[0] - f64::from(floor[0]), p[1] - f64::from(floor[1])];
    let sample = |x: i32, y: i32, c: usize| {
        f64::from(
            bytes[((y.clamp(0, extent[1] as i32 - 1) as u32 * extent[0]
                + x.clamp(0, extent[0] as i32 - 1) as u32)
                * 4) as usize
                + c],
        )
    };
    std::array::from_fn(|c| {
        let top = sample(floor[0], floor[1], c) * (1. - weight[0])
            + sample(floor[0] + 1, floor[1], c) * weight[0];
        let bottom = sample(floor[0], floor[1] + 1, c) * (1. - weight[0])
            + sample(floor[0] + 1, floor[1] + 1, c) * weight[0];
        top * (1. - weight[1]) + bottom * weight[1]
    })
}

fn pixel_for_uv(target: [f64; 2]) -> (u32, u32, [f64; 2]) {
    let x = (target[0] * f64::from(EXTENT[0]) - 0.5).round() as u32;
    let y = (target[1] * f64::from(EXTENT[1]) - 0.5).round() as u32;
    (
        x,
        y,
        [
            (f64::from(x) + 0.5) / f64::from(EXTENT[0]),
            (f64::from(y) + 0.5) / f64::from(EXTENT[1]),
        ],
    )
}

fn commit<K: AssetKind, T: Send + Sync + 'static>(
    store: &AssetStore<K, T, String>,
    value: T,
    bytes: u64,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>> {
    match store.acquire(&store.create()?)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(bytes))?),
        _ => Err("new mip asset has no producer".into()),
    }
}

#[cfg(test)]
#[path = "tests/mip_generation.rs"]
mod tests;
