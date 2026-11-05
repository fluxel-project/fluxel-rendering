//! Exact Color assets for the Deferred Surface consumer.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

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

pub(super) const COMPANION_EXTENT: [u32; 2] = [96, 96];
pub(super) const POSITIONS: [[f32; 3]; 3] = [
    [-0.62, -0.56, 0.50],
    [0.58, -0.42, 0.50],
    [-0.12, 0.66, 0.50],
];

pub(super) struct Assets {
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
}

impl Assets {
    pub(super) fn new() -> Result<Self, Box<dyn Error>> {
        let geometries = AssetStore::new();
        let materials = AssetStore::new();
        let textures = AssetStore::new();
        Ok(Self {
            geometry: geometry(&geometries)?,
            material: material(&materials)?,
            texture: texture(&textures)?,
        })
    }

    pub(super) fn forward(&self) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.geometry.clone()],
            materials: vec![self.material.clone()],
            textures: vec![self.texture.clone()],
        }
    }

    pub(super) fn record(&self) -> RenderRecord<ForwardResources> {
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.geometry),
                material: SceneAssetRef::from_snapshot(&self.material),
                texture: SceneAssetRef::from_snapshot(&self.texture),
                tint: [1.0; 4],
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([-0.62, -0.56, 0.15], [0.58, 0.66, 0.82])
                .expect("fixed finite Surface fixture bounds"),
        }
    }

    pub(super) fn write(&self, output: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            output.join("s7-surface-position.bin"),
            &self.geometry.value().streams[0],
        )?;
        fs::write(
            output.join("s7-surface-geometry.txt"),
            format!("{:#?}\n", self.geometry.value()),
        )?;
        fs::write(
            output.join("s7-surface-material.txt"),
            format!("{:#?}\n", self.material.value()),
        )?;
        fs::write(
            output.join("s7-surface-albedo.rgba"),
            &self.texture.value().rgba8,
        )?;
        Ok(())
    }
}

fn geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(36);
    for position in POSITIONS {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let value = GeometryData {
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
    };
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(36))?),
        _ => Err("S7 Deferred Surface geometry producer missing".into()),
    }
}

fn material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let value = MaterialDefinition::new(
        vec![MaterialFragment {
            name: "surface_color".into(), inputs: Vec::new(), output: MaterialType::Vec4, calls: Vec::new(),
            wgsl: "fn surface_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
        }],
        "surface_color",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        vec![
            MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
        ],
        BTreeMap::new(), SurfaceOutput::Color,
    );
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(0))?),
        _ => Err("S7 Deferred Surface material producer missing".into()),
    }
}

fn texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    let value = ColorTextureData {
        extent: [1, 1],
        rgba8: vec![144, 80, 32, 255],
    };
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(4))?),
        _ => Err("S7 Deferred Surface texture producer missing".into()),
    }
}
