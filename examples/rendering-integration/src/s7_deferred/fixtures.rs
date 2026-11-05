//! Real position-only assets for the provisional native Deferred consumer.

use external_renderer::ChannelMapVariant;
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds, RenderRecord,
    RenderTransform, SceneAssetRef,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

pub(super) const ALBEDO: [u8; 4] = [144, 80, 32, 255];
pub(super) const EXTENT: [u32; 2] = [96, 96];

pub(super) struct Assets {
    pub(super) flat: AssetSnapshot<GeometryAsset, GeometryData>,
    pub(super) tilt: AssetSnapshot<GeometryAsset, GeometryData>,
    pub(super) material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    pub(super) code_first: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    pub(super) code_second: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    pub(super) texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
}

impl Assets {
    pub(super) fn new() -> Result<Self, Box<dyn Error>> {
        let geometry_store = AssetStore::new();
        let material_store = AssetStore::new();
        let texture_store = AssetStore::new();
        Ok(Self {
            flat: geometry_snapshot(&geometry_store, geometry(flat_positions()))?,
            tilt: geometry_snapshot(&geometry_store, geometry(tilt_positions()))?,
            material: material_snapshot(&material_store, material())?,
            code_first: material_snapshot(&material_store, ChannelMapVariant::First.material())?,
            code_second: material_snapshot(&material_store, ChannelMapVariant::Second.material())?,
            texture: texture_snapshot(&texture_store)?,
        })
    }

    pub(super) fn forward(&self) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.flat.clone(), self.tilt.clone()],
            materials: vec![
                self.material.clone(),
                self.code_first.clone(),
                self.code_second.clone(),
            ],
            textures: vec![self.texture.clone()],
        }
    }

    pub(super) fn record(&self, tilted: bool) -> RenderRecord<ForwardResources> {
        let geometry = if tilted { &self.tilt } else { &self.flat };
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(geometry),
                material: SceneAssetRef::from_snapshot(&self.material),
                texture: SceneAssetRef::from_snapshot(&self.texture),
                tint: [1.0; 4],
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([-0.62, -0.56, 0.15], [0.58, 0.66, 0.82])
                .expect("finite ordered fixture bounds"),
        }
    }

    pub(super) fn code_record(&self, variant: ChannelMapVariant) -> RenderRecord<ForwardResources> {
        let material = match variant {
            ChannelMapVariant::First => &self.code_first,
            ChannelMapVariant::Second => &self.code_second,
        };
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.flat),
                material: SceneAssetRef::from_snapshot(material),
                texture: SceneAssetRef::from_snapshot(&self.texture),
                tint: [1.0; 4],
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([-0.62, -0.56, 0.15], [0.58, 0.66, 0.82])
                .expect("finite ordered fixture bounds"),
        }
    }

    pub(super) fn write(&self, output: &Path) -> Result<(), Box<dyn Error>> {
        for (name, snapshot) in [("flat", &self.flat), ("tilt", &self.tilt)] {
            fs::write(
                output.join(format!("s7-{name}-position.bin")),
                &snapshot.value().streams[0],
            )?;
            fs::write(
                output.join(format!("s7-{name}-geometry.txt")),
                format!("{:#?}\n", snapshot.value()),
            )?;
        }
        fs::write(
            output.join("s7-material.txt"),
            format!("{:#?}\n", self.material.value()),
        )?;
        for (name, material) in [("first", &self.code_first), ("second", &self.code_second)] {
            fs::write(
                output.join(format!("s7-code-node-{name}-material.txt")),
                format!("{:#?}\n", material.value()),
            )?;
        }
        fs::write(output.join("s7-albedo.rgba"), &self.texture.value().rgba8)?;
        Ok(())
    }
}

pub(super) fn flat_positions() -> [[f32; 3]; 3] {
    [
        [-0.62, -0.56, 0.50],
        [0.58, -0.42, 0.50],
        [-0.12, 0.66, 0.50],
    ]
}

pub(super) fn tilt_positions() -> [[f32; 3]; 3] {
    [
        [-0.62, -0.56, 0.20],
        [0.58, -0.42, 0.60],
        [-0.12, 0.66, 0.82],
    ]
}

fn geometry(positions: [[f32; 3]; 3]) -> GeometryData {
    let mut bytes = Vec::with_capacity(36);
    for position in positions {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 12,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![GeometryAttribute {
                    semantic: "POSITION".into(),
                    offset: 0,
                    format: VertexFormat::Float32(3),
                }],
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    }
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "s7_albedo".into(),
            inputs: vec![],
            output: MaterialType::Vec4,
            calls: vec![],
            wgsl: "fn s7_albedo() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
        }],
        "s7_albedo",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation { group: 0, binding: 0 },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation { group: 0, binding: 1 },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
}

fn geometry_snapshot(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
    value: GeometryData,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let handle = store.create()?;
    let bytes = value.streams.iter().map(|stream| stream.len() as u64).sum();
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(bytes))?),
        _ => Err("S7 geometry producer missing".into()),
    }
}

fn material_snapshot(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    value: MaterialDefinition,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(0))?),
        _ => Err("S7 material producer missing".into()),
    }
}

fn texture_snapshot(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    let value = ColorTextureData {
        extent: [1, 1],
        rgba8: ALBEDO.to_vec(),
    };
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(4))?),
        _ => Err("S7 texture producer missing".into()),
    }
}
