//! Renderer scene assembly for the source-faithful cube grid.

use std::{collections::BTreeMap, error::Error};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialCodeNode, MaterialDefinition, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds, RenderChange,
    RenderChangeSet, RenderScene, RenderTransform, RenderView, SceneAssetRef,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::SourceUnit;

use crate::scene::{camera, model, CUBE_COUNT, CUBE_INDICES, CUBE_VERTICES};

pub const EXTENT: [u32; 2] = [512, 512];

pub struct GridInput {
    pub snapshot: fluxel_renderer::FrameSnapshot<ForwardResources>,
    pub view: fluxel_renderer::RenderViewId,
    pub assets: ForwardAssets,
    pub bounds: RenderBounds,
    pub transforms: Vec<[[f32; 4]; 4]>,
    pub camera: [[f32; 4]; 4],
}

/// Builds 125 actual scene records sharing one geometry/material/white texture.
/// Reverse insertion assigns source identities in reversed cube order, so the
/// ordinary stable source ordering really reverses the cube draw sequence.
pub fn grid(time: f32, reverse_insert: bool) -> Result<GridInput, Box<dyn Error>> {
    let geometries = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let geometry = commit(&geometries, cube_geometry(), 36 * 36)?;
    let material = commit(&materials, vertex_color_material(), 0)?;
    let texture = commit(
        &textures,
        ColorTextureData {
            encoding: fluxel_renderer::TextureEncoding::Linear,
            mip_levels: 1,
            extent: [1, 1],
            rgba8: vec![255, 255, 255, 255],
        },
        4,
    )?;
    let assets = ForwardAssets {
        geometry: vec![geometry.clone()],
        materials: vec![material.clone()],
        textures: vec![texture.clone()],
    };
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    let indices: Vec<u32> = if reverse_insert {
        (0..CUBE_COUNT).rev().collect()
    } else {
        (0..CUBE_COUNT).collect()
    };
    let mut transforms = vec![[[0.0; 4]; 4]; CUBE_COUNT as usize];
    for index in indices {
        let transform = model(index, time);
        transforms[index as usize] = transform;
        changes.push(RenderChange::Insert(fluxel_renderer::RenderRecord {
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
            transform: RenderTransform::from_columns(transform)?,
            bounds: grid_bounds(),
        }));
    }
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::from_columns(camera(EXTENT))?,
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    Ok(GridInput {
        snapshot: scene.snapshot(),
        view: receipt.inserted_views()[0],
        assets,
        bounds: grid_bounds(),
        transforms,
        camera: camera(EXTENT),
    })
}

pub fn cube_geometry() -> GeometryData {
    let mut bytes = Vec::with_capacity(CUBE_INDICES.len() * 36);
    for index in CUBE_INDICES {
        let vertex = CUBE_VERTICES[index as usize];
        for value in vertex[..3]
            .iter()
            .copied()
            .chain([0.0, 0.0])
            .chain([vertex[3], vertex[4], vertex[5], 1.0])
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 36,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "UV".into(),
                        offset: 12,
                        format: VertexFormat::Float32(2),
                    },
                    GeometryAttribute {
                        semantic: "COLOR".into(),
                        offset: 20,
                        format: VertexFormat::Float32(4),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..36,
    }
}

pub fn vertex_color_material() -> MaterialDefinition {
    MaterialDefinition::new(
        Vec::new(),
        "vertex_color_white",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation {
                    group: 0,
                    binding: 1,
                },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
    .with_code_node(MaterialCodeNode {
        name: "vertex_color_white".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(
            "renderer-04-vertex-color-root",
            "fn vertex_color_white() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }",
        ),
        resources: Vec::new(),
    })
}

fn grid_bounds() -> RenderBounds {
    RenderBounds::new([-12.0; 3], [12.0; 3]).expect("fixed finite cube-grid bounds")
}

/// Two coincident projected triangles with independent constant material tints.
/// The nearer cyan draw must win even when red is drawn last.
pub fn overlap(reverse: bool) -> Result<GridInput, Box<dyn Error>> {
    let geometries = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let mut geometry = cube_geometry();
    geometry.vertices = 0..3;
    geometry.streams[0].clear();
    for position in [[-0.8_f32, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0]] {
        for value in position.into_iter().chain([0., 0., 1., 1., 1., 1.]) {
            geometry.streams[0].extend_from_slice(&value.to_le_bytes());
        }
    }
    let geometry = commit(&geometries, geometry, 108)?;
    let material = commit(&materials, vertex_color_material(), 0)?;
    let texture = commit(
        &textures,
        ColorTextureData {
            encoding: fluxel_renderer::TextureEncoding::Linear,
            mip_levels: 1,
            extent: [1, 1],
            rgba8: vec![255; 4],
        },
        4,
    )?;
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    let mut layers = [(0.75, [1., 0., 0., 1.]), (0.25, [0., 1., 1., 1.])];
    if reverse {
        layers.reverse();
    }
    let mut transforms = Vec::new();
    for (depth, tint) in layers {
        let mut transform = RenderTransform::identity().columns();
        transform[3][2] = depth;
        transforms.push(transform);
        changes.push(RenderChange::Insert(fluxel_renderer::RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&geometry),
                material: SceneAssetRef::from_snapshot(&material),
                textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                    "material_texture",
                    SceneAssetRef::from_snapshot(&texture),
                )],
                samplers: Vec::new(),
                tint,
            },
            transform: RenderTransform::from_columns(transform)?,
            bounds: grid_bounds(),
        }));
    }
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    Ok(GridInput {
        snapshot: scene.snapshot(),
        view: receipt.inserted_views()[0],
        assets: ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![texture],
        },
        bounds: grid_bounds(),
        transforms,
        camera: RenderTransform::identity().columns(),
    })
}

fn commit<K, T>(
    store: &AssetStore<K, T, String>,
    value: T,
    bytes: u64,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>>
where
    K: AssetKind,
    T: Send + Sync + 'static,
{
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(bytes))?),
        _ => Err("new source asset did not yield a producer".into()),
    }
}
