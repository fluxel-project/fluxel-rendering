//! Actual immutable asset and scene construction for the centre-plane controls.

use std::{collections::BTreeMap, error::Error};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialCodeNode, MaterialDefinition, MaterialNode,
    MaterialNodeGraph, MaterialNodeId, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, EnvironmentIrradianceAsset, EnvironmentIrradianceData,
    EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, ForwardAssets, ForwardResources,
    GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout, GeometryStream,
    LinearSpecularLod, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, SceneAssetRef, ShadingCamera, SpecularIblImage, SpecularIblMip,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::SourceUnit;

use crate::oracle::Control;

pub const EXTENT: [u32; 2] = [512, 512];

pub struct Input {
    pub snapshot: fluxel_renderer::FrameSnapshot<ForwardResources>,
    pub view: fluxel_renderer::RenderViewId,
    pub assets: ForwardAssets,
    pub environment: AssetSnapshot<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData>,
    pub diffuse: Option<AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>>,
    pub diffuse_bytes: Option<Vec<u8>>,
    pub geometry_bytes: Vec<u8>,
    pub material: MaterialDefinition,
    pub texture_bytes: Vec<u8>,
}

pub fn input(control: Control) -> Result<Input, Box<dyn Error>> {
    let geometries = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let environments =
        AssetStore::<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, String>::new();
    let diffuse_store =
        AssetStore::<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>::new();
    let geometry_data = plane();
    let geometry_bytes = geometry_data.streams[0].clone();
    let material_definition = material(control);
    let texture_bytes = vec![179, 102, 51, (control.metallic * 255.0).round() as u8];
    let geometry = commit(
        &geometries,
        geometry_data,
        ResidentBytes::new(geometry_bytes.len() as u64),
    )?;
    let material = commit(
        &materials,
        material_definition.clone(),
        ResidentBytes::new(0),
    )?;
    let texture = commit(
        &textures,
        ColorTextureData {
            encoding: fluxel_renderer::TextureEncoding::Linear,
            mip_levels: 1,
            extent: [1, 1],
            rgba8: texture_bytes.clone(),
        },
        ResidentBytes::new(4),
    )?;
    let environment = commit(&environments, environment(control), ResidentBytes::new(192))?;
    let diffuse_bytes = control.diffuse.then(|| {
        vec![
            32, 16, 8, 255, 64, 32, 16, 255, 32, 16, 8, 255, 64, 32, 16, 255,
        ]
    });
    let diffuse = match &diffuse_bytes {
        Some(bytes) => Some(commit(
            &diffuse_store,
            EnvironmentIrradianceData {
                extent: [2, 2],
                rgba8: bytes.clone(),
            },
            ResidentBytes::new(bytes.len() as u64),
        )?),
        None => None,
    };
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&texture),
            )],
            samplers: Vec::new(),
            tint: [1.0, 1.0, 1.0, control.roughness],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.0; 3], [1.0; 3])?,
    }));
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), EXTENT)?
            .with_shading_camera(ShadingCamera::orthographic(control.camera)?),
    ));
    let receipt = scene.apply(&changes)?;
    Ok(Input {
        snapshot: scene.snapshot(),
        view: receipt.inserted_views()[0],
        assets: ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![texture],
        },
        environment,
        diffuse,
        diffuse_bytes,
        geometry_bytes,
        material: material_definition,
        texture_bytes,
    })
}

pub fn environment(control: Control) -> EnvironmentSpecularIblData {
    let radiance = if control.lowest_mip_only {
        [
            [0x4000, 0x3c00, 0x3800],
            [0x3c00, 0x3800, 0x3400],
            [0x3400, 0x3000, 0x2c00],
        ]
    } else {
        [
            [0x4000, 0x3c00, 0x3800],
            [0x3c00, 0x3800, 0x3400],
            [0x3800, 0x3400, 0x3000],
        ]
    };
    EnvironmentSpecularIblData {
        radiance_mips: vec![
            mip([4, 2], radiance[0]),
            mip([2, 1], radiance[1]),
            mip([1, 1], radiance[2]),
        ],
        brdf_lut: image(
            [2, 2],
            if control.lut_variant {
                [0x3a00, 0x3400]
            } else {
                [0x3400, 0x3a00]
            },
        ),
        lod: LinearSpecularLod {
            scale: 1.0,
            offset: 0.0,
        },
    }
}

fn mip(extent: [u32; 2], rgb: [u16; 3]) -> SpecularIblMip {
    SpecularIblMip {
        extent,
        rgba16f_le: pixels(extent, rgb),
    }
}
fn image(extent: [u32; 2], rg: [u16; 2]) -> SpecularIblImage {
    let mut bytes = Vec::with_capacity((extent[0] * extent[1] * 8) as usize);
    for _ in 0..extent[0] * extent[1] {
        for value in [rg[0], rg[1], 0, 0x3c00] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    SpecularIblImage {
        extent,
        rgba16f_le: bytes,
    }
}
fn pixels(extent: [u32; 2], rgb: [u16; 3]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for texel in 0..extent[0] * extent[1] {
        let column = texel % extent[0];
        let value = if column == 0 {
            rgb
        } else if column + 1 == extent[0] {
            [0x3000, 0x2c00, 0x2800]
        } else {
            [0x3800, 0x3400, 0x3000]
        };
        for value in [value[0], value[1], value[2], 0x3c00] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn plane() -> GeometryData {
    let mut bytes = Vec::new();
    for position in [
        [-1_f32, -1., 0.5],
        [1., -1., 0.5],
        [-1., 1., 0.5],
        [-1., 1., 0.5],
        [1., -1., 0.5],
        [1., 1., 0.5],
    ] {
        for value in position.into_iter().chain([0., 0., 1.]) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 24,
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
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..6,
    }
}

fn material(control: Control) -> MaterialDefinition {
    MaterialDefinition::new(Vec::new(), "base", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "tint_u".into() }], vec![MaterialResource { name: "tint_u".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } }, MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } }], BTreeMap::new(), SurfaceOutput::MetallicRoughness { properties: "properties".into() })
        .with_code_node(node("base", MaterialType::Vec4, "fn base() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }"))
        .with_code_node(node("properties", MaterialType::Vec2, &format!("fn properties() -> vec2<f32> {{ return vec2<f32>({}, {}); }}", control.metallic, control.roughness)))
        .with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "base".into(), inputs: Vec::new() }], output: MaterialNodeId(1) })
}
fn node(name: &str, output: MaterialType, source: &str) -> MaterialCodeNode {
    MaterialCodeNode {
        name: name.into(),
        inputs: Vec::new(),
        output,
        calls: Vec::new(),
        source: SourceUnit::new(format!("ibl-{name}"), source),
        resources: Vec::new(),
    }
}
fn commit<K: AssetKind, T: Send + Sync + 'static>(
    store: &AssetStore<K, T, String>,
    value: T,
    bytes: ResidentBytes,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, bytes)?),
        _ => Err("new asset did not provide a producer".into()),
    }
}

#[cfg(test)]
#[path = "tests/scene.rs"]
mod tests;
