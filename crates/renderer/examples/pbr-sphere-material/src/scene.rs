//! Immutable PBR scene inputs for the single-mesh and multi-record consumers.

#[path = "scene/mesh_data.rs"]
mod mesh_data;
pub use mesh_data::{mesh, sphere_centres};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceExtensions,
    SurfaceOutput,
};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, EnvironmentIrradianceAsset, EnvironmentIrradianceData,
    EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, ForwardAssets, ForwardResources,
    GeometryAsset, GeometryData, LinearSpecularLod, RenderBounds, RenderChange, RenderChangeSet,
    RenderRecord, RenderScene, RenderTransform, RenderView, SceneAssetRef, ShadingCamera,
    SpecularIblImage, SpecularIblMip,
};
use fluxel_shader::SourceUnit;
use std::{collections::BTreeMap, error::Error};

pub const EXTENT: [u32; 2] = [512, 512];
pub const EYE: [f32; 3] = [0., 0.35, 7.5];
pub const ORM: [[u8; 4]; 6] = [
    [255, 38, 0, 255],
    [255, 38, 128, 255],
    [255, 38, 255, 255],
    [96, 204, 0, 255],
    [255, 204, 128, 255],
    [255, 204, 255, 255],
];

pub struct Input {
    pub snapshot: fluxel_renderer::FrameSnapshot<ForwardResources>,
    pub view: fluxel_renderer::RenderViewId,
    pub assets: ForwardAssets,
    pub environment: AssetSnapshot<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData>,
    pub diffuse: AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>,
    pub geometry_bytes: Vec<u8>,
    pub orm_bytes: Vec<u8>,
    pub diffuse_bytes: Vec<u8>,
    pub radiance_mip_bytes: Vec<Vec<u8>>,
    pub brdf_lut_bytes: Vec<u8>,
    pub eye: [f32; 3],
    pub tint: [f32; 4],
    pub base_color: [f32; 3],
    pub vp: [[f32; 4]; 4],
    pub material: MaterialDefinition,
    /// Authoritative per-record source mapping for consumers that must retain
    /// exact asset ownership instead of assuming a shared combined mesh.
    pub object_sources: Vec<SourceObject>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceObject {
    pub geometry_index: usize,
    pub material_index: usize,
    pub texture_index: usize,
    pub world_from_local: [[f32; 4]; 4],
}

pub fn input(ao_variant: bool, eye: [f32; 3]) -> Result<Input, Box<dyn Error>> {
    let gs = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let ms = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let ts = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let es = AssetStore::<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, String>::new();
    let ds = AssetStore::<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>::new();
    let geo = mesh();
    let bytes = geo.streams[0].clone();
    let orm = orm_bytes(ao_variant);
    let definition = material();
    let g = commit(&gs, geo, ResidentBytes::new(bytes.len() as u64))?;
    let m = commit(&ms, definition.clone(), ResidentBytes::new(0))?;
    let t = commit(
        &ts,
        ColorTextureData {
            encoding: fluxel_renderer::TextureEncoding::Linear,
            mip_levels: 1,
            extent: [6, 1],
            rgba8: orm.clone(),
        },
        ResidentBytes::new(24),
    )?;
    let (e, radiance_mip_bytes, brdf_lut_bytes) = commit_environment(&es)?;
    let (d, diffuse_bytes) = commit_diffuse(&ds)?;
    let vp = perspective(eye);
    let mut scene = RenderScene::new()?;
    let mut c = RenderChangeSet::new();
    c.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&g),
            material: SceneAssetRef::from_snapshot(&m),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&t),
            )],
            samplers: Vec::new(),
            tint: [0.42, 0.18, 0.08, 1.],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-3.2, -2., -1.], [3.2, 2., 1.])?,
    }));
    c.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::from_columns(vp)?, EXTENT)?
            .with_shading_camera(ShadingCamera::perspective(eye)?),
    ));
    let r = scene.apply(&c)?;
    Ok(Input {
        snapshot: scene.snapshot(),
        view: r.inserted_views()[0],
        assets: ForwardAssets {
            geometry: vec![g],
            materials: vec![m],
            textures: vec![t],
        },
        environment: e,
        diffuse: d,
        geometry_bytes: bytes,
        orm_bytes: orm,
        diffuse_bytes,
        radiance_mip_bytes,
        brdf_lut_bytes,
        eye,
        tint: [0.42, 0.18, 0.08, 1.],
        base_color: [0.42, 0.18, 0.08],
        vp,
        material: definition,
        object_sources: vec![SourceObject {
            geometry_index: 0,
            material_index: 0,
            texture_index: 0,
            world_from_local: identity_columns(),
        }],
    })
}

/// Six independent scene records retaining their own geometry and ORM owner.
/// The baked positions and identity transforms deliberately preserve the
/// single-mesh scene's world-space geometry and oracle input bytes.
pub fn multi_input(ao_variant: bool, eye: [f32; 3]) -> Result<Input, Box<dyn Error>> {
    let gs = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let ms = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let ts = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let es = AssetStore::<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, String>::new();
    let ds = AssetStore::<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>::new();
    // This is the legacy world-space oracle artifact, not the bytes uploaded
    // by the multi-record geometry owners below.
    let geometry_bytes = mesh().streams[0].clone();
    let definition = material();
    let no_effects = material_without_effects();
    let geometries = mesh_data::individual_meshes()
        .into_iter()
        .map(|geometry| {
            commit(
                &gs,
                geometry,
                ResidentBytes::new(
                    (mesh_data::VERTICES_PER_SPHERE as usize * mesh_data::VERTEX_BYTES) as u64,
                ),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let materials = vec![
        commit(&ms, definition.clone(), ResidentBytes::new(0))?,
        commit(&ms, no_effects, ResidentBytes::new(0))?,
    ];
    let actual_textures = multi_orm_textures(ao_variant);
    let orm_bytes = actual_textures
        .iter()
        .enumerate()
        .flat_map(|(index, texture)| active_orm(texture, index))
        .collect();
    let textures = actual_textures
        .into_iter()
        .map(|texture| commit(&ts, texture, ResidentBytes::new(24)))
        .collect::<Result<Vec<_>, _>>()?;
    let vp = perspective(eye);
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    let mut object_sources = Vec::with_capacity(6);
    for index in 0..6 {
        let material_index = if index == 3 { 1 } else { 0 };
        let world_from_local = translation_columns(sphere_centres()[index]);
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&geometries[index]),
                material: SceneAssetRef::from_snapshot(&materials[material_index]),
                textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                    "material_texture",
                    SceneAssetRef::from_snapshot(&textures[index]),
                )],
                samplers: Vec::new(),
                tint: [0.42, 0.18, 0.08, 1.],
            },
            transform: RenderTransform::from_columns(world_from_local)?,
            bounds: sphere_bounds(sphere_centres()[index])?,
        }));
        object_sources.push(SourceObject {
            geometry_index: index,
            material_index,
            texture_index: index,
            world_from_local,
        });
    }
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::from_columns(vp)?, EXTENT)?
            .with_shading_camera(ShadingCamera::perspective(eye)?),
    ));
    let records = scene.apply(&changes)?;
    let (environment, radiance_mip_bytes, brdf_lut_bytes) = commit_environment(&es)?;
    let (diffuse, diffuse_bytes) = commit_diffuse(&ds)?;
    Ok(Input {
        snapshot: scene.snapshot(),
        view: records.inserted_views()[0],
        assets: ForwardAssets {
            geometry: geometries,
            materials,
            textures,
        },
        environment,
        diffuse,
        geometry_bytes,
        orm_bytes,
        diffuse_bytes,
        radiance_mip_bytes,
        brdf_lut_bytes,
        eye,
        tint: [0.42, 0.18, 0.08, 1.],
        base_color: [0.42, 0.18, 0.08],
        vp,
        material: definition,
        object_sources,
    })
}

fn multi_orm_textures(ao_variant: bool) -> Vec<ColorTextureData> {
    (0..6)
        .map(|index| {
            let mut rgba8 = vec![255, 255, 0, 255].repeat(6);
            let mut value = ORM[index];
            if index == 0 && ao_variant {
                value[0] = 24;
            }
            // Sphere three selects its own fourth texel.  Its no-effects material
            // must retain physical AO=1 even while effects are active elsewhere.
            if index == 3 {
                value[0] = 255;
            }
            rgba8[index * 4..index * 4 + 4].copy_from_slice(&value);
            ColorTextureData {
                encoding: fluxel_renderer::TextureEncoding::Linear,
                mip_levels: 1,
                extent: [6, 1],
                rgba8,
            }
        })
        .collect()
}

fn active_orm(texture: &ColorTextureData, index: usize) -> [u8; 4] {
    texture.rgba8[index * 4..index * 4 + 4]
        .try_into()
        .expect("six texels")
}

fn sphere_bounds(centre: [f32; 3]) -> Result<RenderBounds, Box<dyn Error>> {
    Ok(RenderBounds::new(
        [centre[0] - 0.72, centre[1] - 0.72, centre[2] - 0.72],
        [centre[0] + 0.72, centre[1] + 0.72, centre[2] + 0.72],
    )?)
}

fn identity_columns() -> [[f32; 4]; 4] {
    [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]
}

fn translation_columns(translation: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [translation[0], translation[1], translation[2], 1.],
    ]
}
fn orm_bytes(ao_variant: bool) -> Vec<u8> {
    ORM.into_iter()
        .enumerate()
        .flat_map(|(i, mut p)| {
            if ao_variant && i == 0 {
                p[0] = 24;
            }
            p
        })
        .collect()
}
fn commit_environment(
    store: &AssetStore<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, String>,
) -> Result<
    (
        AssetSnapshot<EnvironmentSpecularIblAsset, EnvironmentSpecularIblData>,
        Vec<Vec<u8>>,
        Vec<u8>,
    ),
    Box<dyn Error>,
> {
    let data = environment();
    let radiance_mip_bytes = data
        .radiance_mips
        .iter()
        .map(|mip| mip.rgba16f_le.clone())
        .collect();
    let brdf_lut_bytes = data.brdf_lut.rgba16f_le.clone();
    let bytes = data
        .radiance_mips
        .iter()
        .map(|mip| mip.rgba16f_le.len() as u64)
        .sum::<u64>()
        + data.brdf_lut.rgba16f_le.len() as u64;
    Ok((
        commit(store, data, ResidentBytes::new(bytes))?,
        radiance_mip_bytes,
        brdf_lut_bytes,
    ))
}

fn commit_diffuse(
    store: &AssetStore<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>,
) -> Result<
    (
        AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>,
        Vec<u8>,
    ),
    Box<dyn Error>,
> {
    let rgba8 = vec![
        48, 32, 20, 255, 32, 48, 64, 255, 48, 32, 20, 255, 32, 48, 64, 255,
    ];
    let snapshot = commit(
        store,
        EnvironmentIrradianceData {
            extent: [2, 2],
            rgba8: rgba8.clone(),
        },
        ResidentBytes::new(16),
    )?;
    Ok((snapshot, rgba8))
}
fn material() -> MaterialDefinition {
    let parameters = vec![MaterialParameter {
        name: "tint".into(),
        ty: MaterialType::Vec4,
        resource: "tint_u".into(),
    }];
    let resources = vec![
        MaterialResource {
            name: "tint_u".into(),
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
        MaterialResource {
            name: "material_sampler".into(),
            kind: MaterialResourceKind::Sampler { comparison: false },
            location: BindingLocation {
                group: 0,
                binding: 2,
            },
        },
    ];
    let graph = MaterialNodeGraph {
        nodes: vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "base".into(),
            inputs: vec![MaterialNodeInput {
                input: "uv".into(),
                source: MaterialNodeInputSource::Uv0,
            }],
        }],
        output: MaterialNodeId(1),
    };
    MaterialDefinition::new(
        Vec::new(),
        "base",
        parameters,
        resources,
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughnessUv0Normal {
            properties: "orm".into(),
            normal: "normal".into(),
        },
    )
    .with_code_node(node(
        "base",
        MaterialType::Vec4,
        "fn base(uv: vec2<f32>) -> vec4<f32> { return tint; }",
    ))
    .with_code_node(node(
        "orm",
        MaterialType::Vec2,
        "fn orm(uv: vec2<f32>) -> vec2<f32> { let p=textureSample(material_texture,material_sampler,uv); return vec2<f32>(p.b,p.g); }",
    ))
    .with_code_node(node(
        "normal",
        MaterialType::Vec3,
        "fn normal(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0,0.0,1.0); }",
    ))
    .with_code_node(node(
        "ao",
        MaterialType::F32,
        "fn ao(uv: vec2<f32>) -> f32 { return textureSample(material_texture,material_sampler,uv).r; }",
    ))
    .with_surface_extensions(SurfaceExtensions {
        occlusion: Some("ao".into()),
        emissive: None,
    })
    .with_node_graph(graph)
}
fn material_without_effects() -> MaterialDefinition {
    let mut definition = material();
    definition = definition.with_surface_extensions(SurfaceExtensions {
        occlusion: None,
        emissive: None,
    });
    definition
}
fn node(name: &str, output: MaterialType, source: &str) -> MaterialCodeNode {
    MaterialCodeNode {
        name: name.into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output,
        calls: vec![],
        resources: vec![],
        source: SourceUnit::new(format!("sphere-{name}"), source),
    }
}
pub fn environment() -> EnvironmentSpecularIblData {
    EnvironmentSpecularIblData {
        radiance_mips: vec![
            mip(
                [4, 2],
                &[
                    [0x4000, 0x3c00, 0x3800],
                    [0x3800, 0x3400, 0x3c00],
                    [0x3000, 0x3800, 0x4000],
                    [0x3c00, 0x3000, 0x3400],
                    [0x3800, 0x3c00, 0x4000],
                    [0x3c00, 0x3800, 0x3400],
                    [0x4000, 0x3400, 0x3000],
                    [0x3400, 0x4000, 0x3800],
                ],
            ),
            mip(
                [2, 1],
                &[[0x3c00, 0x3800, 0x3400], [0x3400, 0x3800, 0x3c00]],
            ),
            mip([1, 1], &[[0x3800, 0x3400, 0x3000]]),
        ],
        brdf_lut: SpecularIblImage {
            extent: [2, 2],
            rgba16f_le: [
                [0x3a00_u16, 0x3000, 0, 0x3c00],
                [0x3000, 0x3a00, 0, 0x3c00],
                [0x3800, 0x3400, 0, 0x3c00],
                [0x3400, 0x3800, 0, 0x3c00],
            ]
            .into_iter()
            .flatten()
            .flat_map(u16::to_le_bytes)
            .collect(),
        },
        lod: LinearSpecularLod {
            scale: 1.,
            offset: 0.,
        },
    }
}
fn mip(extent: [u32; 2], rgb: &[[u16; 3]]) -> SpecularIblMip {
    SpecularIblMip {
        extent,
        rgba16f_le: rgb
            .iter()
            .flat_map(|color| {
                [color[0], color[1], color[2], 0x3c00]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
            })
            .collect(),
    }
}
pub fn perspective(eye: [f32; 3]) -> [[f32; 4]; 4] {
    let f = 1. / (40f32.to_radians() / 2.).tan();
    let a = 1.;
    let n = 0.1;
    let z = 20.;
    let view = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [-eye[0], -eye[1], -eye[2], 1.],
    ];
    let p = [
        [f / a, 0., 0., 0.],
        [0., f, 0., 0.],
        [0., 0., z / (n - z), -1.],
        [0., 0., n * z / (n - z), 0.],
    ];
    mul(p, view)
}
fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    core::array::from_fn(|c| core::array::from_fn(|r| (0..4).map(|k| a[k][r] * b[c][k]).sum()))
}
fn commit<K: AssetKind, T: Send + Sync + 'static>(
    s: &AssetStore<K, T, String>,
    v: T,
    b: ResidentBytes,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(v, b)?),
        _ => Err("new asset did not provide producer".into()),
    }
}
#[cfg(test)]
#[path = "tests/scene.rs"]
mod tests;
