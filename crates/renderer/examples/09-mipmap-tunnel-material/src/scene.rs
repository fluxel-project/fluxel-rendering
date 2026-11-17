//! Strict base-image and tunnel-geometry scene construction.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::*;
use fluxel_renderer::*;
use fluxel_rhi::api::{AddressMode, FilterMode, SamplerDesc};
use fluxel_shader::SourceUnit;

#[path = "../../07-offscreen-material/src/gltf.rs"]
mod gltf;

pub const EXTENT: [u32; 2] = [512, 512];
pub const IMAGE: &str = "tunnel_image";
pub const SAMPLER: &str = "tunnel_sampler";

pub struct Input {
    pub snapshot: FrameSnapshot<ForwardResources>,
    pub view: RenderViewId,
    pub assets: ForwardAssets,
    pub geometry: GeometryData,
    pub image: Vec<u8>,
    pub extent: [u32; 2],
    pub material: MaterialDefinition,
    pub sampler: SamplerDesc,
    pub model_bytes: Vec<u8>,
    pub asset_bytes: Vec<u8>,
    pub vp: [[f32; 4]; 4],
}

pub fn input(
    model_path: &Path,
    image_path: &Path,
    anisotropy: u16,
) -> Result<Input, Box<dyn Error>> {
    let model_bytes = fs::read(model_path)?;
    let asset_bytes = fs::read(image_path)?;
    let mut geometry = gltf::load(&model_bytes)?;
    // RHI example 09 uses PRETRANSFORM_FLIP_Y. The shared decoder already
    // applies flat-node transforms; that RHI option then negates only Y for
    // position and normal, not UVs or winding. Initial culling is disabled.
    flip_y(&mut geometry)?;
    let (extent, image) = rgba8_ktx_base(&asset_bytes)?;
    let levels = 1 + extent[0].max(extent[1]).ilog2();
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let geometry_snapshot = commit(
        &geometry_store,
        geometry.clone(),
        geometry.streams[0].len() as u64,
    )?;
    let material = definition();
    let material_snapshot = commit(&material_store, material.clone(), 0)?;
    let texture = commit(
        &texture_store,
        ColorTextureData {
            extent,
            encoding: TextureEncoding::Linear,
            mip_levels: levels,
            rgba8: image.clone(),
        },
        image.len() as u64,
    )?;
    let sampler = SamplerDesc {
        label: Some("09 tunnel material sampler".into()),
        address: [AddressMode::MirrorRepeat; 3],
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Linear,
        lod: 0.0..levels as f32,
        anisotropy,
        compare: None,
        border: None,
    };
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry_snapshot),
            material: SceneAssetRef::from_snapshot(&material_snapshot),
            textures: vec![ForwardTextureBinding::asset(
                IMAGE,
                SceneAssetRef::from_snapshot(&texture),
            )],
            sampler: Some(ForwardSamplerBinding::new(SAMPLER, sampler.clone())),
            tint: [0.0, 1.0, 1.0, 1.0],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-64.; 3], [64.; 3])?,
    }));
    let vp = tunnel_vp();
    let view = RenderView::new(RenderTransform::from_columns(vp)?, EXTENT)?;
    changes.push(RenderChange::InsertView(view));
    let receipt = scene.apply(&changes)?;
    Ok(Input {
        snapshot: scene.snapshot(),
        view: receipt.inserted_views()[0],
        assets: ForwardAssets {
            geometry: vec![geometry_snapshot],
            materials: vec![material_snapshot],
            textures: vec![texture],
        },
        geometry,
        image,
        extent,
        material,
        sampler,
        model_bytes,
        asset_bytes,
        vp,
    })
}

pub fn definition() -> MaterialDefinition {
    MaterialDefinition::new(Vec::new(), "unused", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "values".into() }], vec![
        MaterialResource { name: "values".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
        MaterialResource { name: IMAGE.into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
        MaterialResource { name: SAMPLER.into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
    ], BTreeMap::new(), SurfaceOutput::ColorUv0)
    .with_code_node(MaterialCodeNode { name: "tunnel_sample".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec4, calls: Vec::new(), resources: Vec::new(), source: SourceUnit::new("renderer-09-tunnel-sample", "fn tunnel_sample(uv: vec2<f32>) -> vec4<f32> { let sampled = textureSampleBias(tunnel_image, tunnel_sampler, uv * vec2<f32>(2.0, 1.0), tint.x); return vec4<f32>(sampled.rgb, 1.0); }") })
    .with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "tunnel_sample".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(1) })
}

fn rgba8_ktx_base(bytes: &[u8]) -> Result<([u32; 2], Vec<u8>), Box<dyn Error>> {
    if bytes.len() < 64 || &bytes[..12] != b"\xABKTX 11\xBB\r\n\x1A\n" {
        return Err("expected KTX1 input".into());
    }
    let u = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    if u(12) != 0x0403_0201
        || u(16) != 0x1401
        || u(20) != 1
        || u(24) != 0x1908
        || u(28) != 0x8058
        || u(32) != 0x1908
        || u(44) != 0
        || u(48) != 0
        || u(52) != 1
        || u(56) != 1
    {
        return Err("expected little-endian RGBA8 two-dimensional one-face one-level KTX1".into());
    }
    let extent = [u(36), u(40)];
    let offset = 64usize
        .checked_add(u(60) as usize)
        .ok_or("KTX metadata overflow")?;
    let length_end = offset
        .checked_add(4)
        .ok_or("KTX image length offset overflow")?;
    if length_end > bytes.len() {
        return Err("KTX metadata exceeds input".into());
    }
    let length = u(offset) as usize;
    let start = length_end;
    let end = start.checked_add(length).ok_or("KTX image overflow")?;
    let required = usize::try_from(extent[0])
        .ok()
        .and_then(|width| {
            usize::try_from(extent[1])
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("KTX extent overflow")?;
    if extent.contains(&0) || length != required || end > bytes.len() {
        return Err("KTX base image is not exact RGBA8".into());
    }
    Ok((extent, bytes[start..end].to_vec()))
}

fn tunnel_vp() -> [[f32; 4]; 4] {
    let aspect = EXTENT[0] as f32 / EXTENT[1] as f32;
    let f = 1.0 / (60f32.to_radians() * 0.5).tan();
    let near = 0.1;
    let far = 1024.0;
    let projection = [
        [f / aspect, 0., 0., 0.],
        [0., f, 0., 0.],
        [0., 0., far / (near - far), -1.],
        [0., 0., far * near / (near - far), 0.],
    ];
    let view = [
        [0., 0., -1., 0.],
        [0., 1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., -40.75, 1.],
    ];
    core::array::from_fn(|c| {
        core::array::from_fn(|r| (0..4).map(|k| projection[k][r] * view[c][k]).sum())
    })
}

fn flip_y(geometry: &mut GeometryData) -> Result<(), Box<dyn Error>> {
    let stream = geometry
        .streams
        .get_mut(0)
        .ok_or("tunnel geometry has no stream")?;
    if stream.len() % 32 != 0 {
        return Err("tunnel geometry ABI is not 32 bytes per vertex".into());
    }
    for vertex in stream.chunks_exact_mut(32) {
        for offset in [4, 16] {
            let mut bytes = [0; 4];
            bytes.copy_from_slice(&vertex[offset..offset + 4]);
            vertex[offset..offset + 4].copy_from_slice(&(-f32::from_le_bytes(bytes)).to_le_bytes());
        }
    }
    Ok(())
}

fn commit<K: AssetKind, T: Send + Sync + 'static>(
    store: &AssetStore<K, T, String>,
    value: T,
    bytes: u64,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>> {
    match store.acquire(&store.create()?)? {
        Acquire::Producer(p) => Ok(p.commit(value, ResidentBytes::new(bytes))?),
        _ => Err("new input did not produce an asset".into()),
    }
}
