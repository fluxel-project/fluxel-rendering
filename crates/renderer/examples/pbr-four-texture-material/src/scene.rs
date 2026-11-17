//! Two real records with four named image bindings and controlled source edits.
use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::*;
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use std::error::Error;

#[path = "material.rs"]
mod material;
pub const EXTENT: [u32; 2] = [512, 512];
pub const TINTS: [[f32; 4]; 2] = [[0.8, 0.9, 0.7, 1.0], [0.6, 0.75, 1.0, 1.0]];
pub const ROLE_NAMES: [&str; 4] = ["base_color", "normal", "orm", "emissive"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    Baseline,
    Base,
    Normal,
    Ao,
    Roughness,
    Metallic,
    Emissive,
    Alias,
}
impl Case {
    pub const ALL: [Self; 8] = [
        Self::Baseline,
        Self::Base,
        Self::Normal,
        Self::Ao,
        Self::Roughness,
        Self::Metallic,
        Self::Emissive,
        Self::Alias,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Base => "base-only",
            Self::Normal => "normal-only",
            Self::Ao => "ao-only",
            Self::Roughness => "roughness-only",
            Self::Metallic => "metallic-only",
            Self::Emissive => "emissive-only",
            Self::Alias => "aliased-source",
        }
    }
}
pub struct Input {
    pub snapshot: FrameSnapshot<ForwardResources>,
    pub view: RenderViewId,
    pub assets: ForwardAssets,
    pub diffuse: AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>,
    pub geometry_bytes: Vec<u8>,
    pub material: MaterialDefinition,
    /// Object then role -> actual source asset index; aliasing is explicit.
    pub sources: [[usize; 4]; 2],
    pub models: [[[f32; 4]; 4]; 2],
    pub case: Case,
}

pub fn input(case: Case) -> Result<Input, Box<dyn Error>> {
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let diffuse_store =
        AssetStore::<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>::new();
    let data = geometry();
    let geometry_bytes = data.streams[0].clone();
    let geometry = commit(&geometry_store, data, geometry_bytes.len() as u64)?;
    let definition = material::definition();
    let material = commit(&material_store, definition.clone(), 0)?;
    let mut payloads = payloads();
    let (index, channel, value) = match case {
        Case::Base => (0, 0, 200),
        Case::Normal => (2, 0, 208),
        Case::Ao => (3, 0, 24),
        Case::Roughness => (3, 1, 220),
        Case::Metallic => (3, 2, 235),
        Case::Emissive => (5, 1, 180),
        _ => (0, 0, 0),
    };
    if !matches!(case, Case::Baseline | Case::Alias) {
        for pixel in payloads[index].rgba8.chunks_exact_mut(4) {
            pixel[channel] = value;
        }
    }
    let textures = payloads
        .into_iter()
        .map(|data| commit(&texture_store, data, 16))
        .collect::<Result<Vec<_>, _>>()?;
    let mut sources = [[0, 2, 3, 5], [1, 2, 4, 5]];
    if case == Case::Alias {
        sources[1][3] = 1;
    }
    // Explicit consumer role facts; generic texture names do not infer encoding.
    validate_roles(&textures, sources)?;
    let diffuse = commit(
        &diffuse_store,
        EnvironmentIrradianceData {
            extent: [1, 1],
            rgba8: vec![48, 64, 80, 255],
        },
        4,
    )?;
    let models = [-0.5, 0.5].map(|x| {
        [
            [0.45, 0., 0., 0.],
            [0., 0.85, 0., 0.],
            [0., 0., 1., 0.],
            [x, 0., 0.5, 1.],
        ]
    });
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    for object in 0..2 {
        // Deliberately supply sources in a different order from bindings.
        let textures = [3, 0, 2, 1]
            .map(|role| {
                ForwardTextureBinding::asset(
                    ROLE_NAMES[role],
                    SceneAssetRef::from_snapshot(&textures[sources[object][role]]),
                )
            })
            .to_vec();
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&geometry),
                material: SceneAssetRef::from_snapshot(&material),
                textures,
                sampler: None,
                tint: TINTS[object],
            },
            transform: RenderTransform::from_columns(models[object])?,
            bounds: RenderBounds::new([-1.; 3], [1.; 3])?,
        }));
    }
    let receipt = scene.apply(&changes)?;
    Ok(Input {
        snapshot: scene.snapshot(),
        view: receipt.inserted_views()[0],
        assets: ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures,
        },
        diffuse,
        geometry_bytes,
        material: definition,
        sources,
        models,
        case,
    })
}

fn validate_roles(
    textures: &[AssetSnapshot<ColorTextureAsset, ColorTextureData>],
    sources: [[usize; 4]; 2],
) -> Result<(), Box<dyn Error>> {
    for source in sources {
        for (role, index) in source.into_iter().enumerate() {
            let expected = if role == 0 || role == 3 {
                TextureEncoding::Srgb
            } else {
                TextureEncoding::Linear
            };
            let texture = textures
                .get(index)
                .ok_or("source role index out of range")?;
            if texture.value().encoding != expected {
                return Err("source role encoding mismatch".into());
            }
        }
    }
    Ok(())
}

fn payloads() -> Vec<ColorTextureData> {
    let color = |bytes: [u8; 16]| ColorTextureData {
        extent: [2, 2],
        encoding: TextureEncoding::Srgb,
        mip_levels: 1,
        rgba8: bytes.to_vec(),
    };
    let data = |bytes: [u8; 16]| ColorTextureData {
        extent: [2, 2],
        encoding: TextureEncoding::Linear,
        mip_levels: 1,
        rgba8: bytes.to_vec(),
    };
    vec![
        color([
            128, 96, 72, 255, 176, 128, 96, 255, 96, 160, 128, 255, 144, 112, 176, 255,
        ]),
        color([
            96, 128, 176, 255, 128, 160, 112, 255, 160, 96, 128, 255, 112, 144, 176, 255,
        ]),
        data([
            128, 128, 255, 255, 160, 112, 250, 255, 112, 150, 245, 255, 144, 136, 252, 255,
        ]),
        data([
            200, 96, 40, 255, 160, 160, 96, 255, 224, 128, 144, 255, 180, 192, 64, 255,
        ]),
        data([
            128, 144, 96, 255, 208, 112, 176, 255, 176, 176, 64, 255, 224, 96, 144, 255,
        ]),
        color([
            48, 32, 16, 255, 72, 48, 24, 255, 32, 64, 48, 255, 64, 24, 72, 255,
        ]),
    ]
}
fn geometry() -> GeometryData {
    let mut bytes = Vec::new();
    for (p, uv) in [
        ([-1.0_f32, -1., 0.], [0., 1.]),
        ([1., -1., 0.], [1., 1.]),
        ([-1., 1., 0.], [0., 0.]),
        ([-1., 1., 0.], [0., 0.]),
        ([1., -1., 0.], [1., 1.]),
        ([1., 1., 0.], [1., 0.]),
    ] {
        for value in p
            .into_iter()
            .chain([0., 0., 1.])
            .chain(uv)
            .chain([1., 0., 0., 1.])
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 48,
                step_mode: VertexStepMode::Vertex,
                attributes: [
                    ("POSITION", 0, 3),
                    ("NORMAL", 12, 3),
                    ("UV0", 24, 2),
                    ("TANGENT", 32, 4),
                ]
                .map(|(name, offset, n)| GeometryAttribute {
                    semantic: name.into(),
                    offset,
                    format: VertexFormat::Float32(n),
                })
                .to_vec(),
            }],
        },
        streams: vec![bytes],
        vertices: 0..6,
    }
}
fn commit<K: AssetKind, T: Send + Sync + 'static>(
    store: &AssetStore<K, T, String>,
    value: T,
    bytes: u64,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>> {
    match store.acquire(&store.create()?)? {
        Acquire::Producer(p) => Ok(p.commit(value, ResidentBytes::new(bytes))?),
        _ => Err("new asset has no producer".into()),
    }
}

#[cfg(test)]
#[path = "tests/scene.rs"]
mod tests;
