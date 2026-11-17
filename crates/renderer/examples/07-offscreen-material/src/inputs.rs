//! Real CPU inputs for the 07 same-Graph capture consumer.
//!
//! The only CPU texture belongs to the dragon producer. The receiver declares
//! `captured_dragon` as a capture slot, so it cannot accidentally sample an
//! uploaded copy of a readback.

use std::{collections::BTreeMap, error::Error};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    prepare_forward_capture_receiver, CaptureTextureSlot, ColorTextureAsset, ColorTextureData,
    ForwardAssets, ForwardCaptureResources, ForwardDrawSource, ForwardResources, FrameSnapshot,
    GeometryAsset, GeometryData, PreparedCaptureReceiver, RenderBounds, RenderChange,
    RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView, RenderViewId,
    SceneAssetRef,
};
use fluxel_shader::{ShaderComposer, SourceUnit};

use crate::scene::{mirror_material, CAPTURE_TEXTURE_SLOT};

/// Shared output size for the producer target and the receiver target.
pub const EXTENT: [u32; 2] = [256, 256];

/// Builds the primary producer-color control input.
pub fn build_capture_scene(
    dragon: GeometryData,
    receiver_plane: GeometryData,
) -> Result<CaptureSceneInputs, Box<dyn Error>> {
    build_capture_scene_with_albedo(dragon, receiver_plane, crate::oracle::PRIMARY_ALBEDO)
}

/// All immutable inputs required by `Renderer::submit_forward_capture`.
pub struct CaptureSceneInputs {
    /// Immutable producer scene selected by the capture pass.
    pub capture_snapshot: FrameSnapshot<ForwardResources>,
    /// Producer camera selected from `capture_snapshot`.
    pub capture_view: RenderViewId,
    /// Receiver view supplied to Material preparation.
    pub receiver_view: RenderView,
    /// Receiver local-to-world transform supplied to Material preparation.
    pub receiver_transform: RenderTransform,
    /// Receiver uniform tint supplied to Material preparation.
    pub receiver_tint: [f32; 4],
    /// Conservative visibility volume for the imported dragon.
    pub visibility: RenderBounds,
    /// Exact immutable asset generations for both passes.
    pub assets: ForwardAssets,
    /// Receiver that declares the renderer-owned capture slot.
    pub receiver: PreparedCaptureReceiver,
    /// The only slot through which the receiver can observe the producer.
    pub slot: CaptureTextureSlot,
}

/// Builds an otherwise identical producer/receiver input with a selected
/// producer-only albedo. The receiver remains capture-slot-only.
pub fn build_capture_scene_with_albedo(
    dragon: GeometryData,
    receiver_plane: GeometryData,
    albedo: [u8; 4],
) -> Result<CaptureSceneInputs, Box<dyn Error>> {
    let geometries = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();

    let dragon_geometry = commit(&geometries, dragon, 0)?;
    let receiver_geometry = commit(&geometries, receiver_plane, 0)?;
    let dragon_material = commit(&materials, dragon_material(), 0)?;
    let receiver_material = commit(&materials, mirror_material(), 0)?;
    let dragon_albedo = commit(
        &textures,
        ColorTextureData {
            encoding: fluxel_renderer::TextureEncoding::Linear,
            mip_levels: 1,
            extent: [1, 1],
            rgba8: albedo.to_vec(),
        },
        4,
    )?;
    let assets = ForwardAssets {
        geometry: vec![dragon_geometry.clone(), receiver_geometry.clone()],
        materials: vec![dragon_material.clone(), receiver_material.clone()],
        textures: vec![dragon_albedo.clone()],
    };

    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&dragon_geometry),
            material: SceneAssetRef::from_snapshot(&dragon_material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "dragon_albedo",
                SceneAssetRef::from_snapshot(&dragon_albedo),
            )],
            sampler: None,
            tint: [1.0; 4],
        },
        transform: capture_fit(),
        bounds: visibility(),
    }));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let capture_view = receipt.inserted_views()[0];
    let receiver_view = RenderView::new(RenderTransform::identity(), EXTENT)?;
    let receiver_transform = receiver_fit();
    let receiver_tint = [1.0; 4];
    let feature = scene
        .feature_issuer()?
        .issue("renderer-07-capture-receiver")?;
    let slot = CaptureTextureSlot::new(CAPTURE_TEXTURE_SLOT)?;
    let receiver = prepare_forward_capture_receiver(
        &receiver_view,
        ForwardDrawSource::Feature {
            feature,
            local_id: 0,
        },
        0,
        &ForwardCaptureResources {
            geometry: SceneAssetRef::from_snapshot(&receiver_geometry),
            material: SceneAssetRef::from_snapshot(&receiver_material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::capture(
                CAPTURE_TEXTURE_SLOT,
                slot.name(),
            )],
            sampler: None,
            tint: receiver_tint,
        },
        receiver_transform,
        &assets,
        &mut ShaderComposer::new(),
    )?;
    Ok(CaptureSceneInputs {
        capture_snapshot: scene.snapshot(),
        capture_view,
        receiver_view,
        receiver_transform,
        receiver_tint,
        visibility: visibility(),
        assets,
        receiver,
        slot,
    })
}

fn capture_fit() -> RenderTransform {
    // RHI 07's dragon spans beyond normalized clip coordinates after glTF TRS.
    // This deterministic acceptance framing is deliberately a local-to-clip fit,
    // not a claim that the old RHI camera matrix is Renderer camera semantics.
    RenderTransform::from_columns([
        [0.55, 0.0, 0.0, 0.0],
        [0.0, 0.55, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 1.0],
    ])
    .expect("finite capture fit")
}

fn receiver_fit() -> RenderTransform {
    // The retained RHI 07 plane lies in XZ. Map XZ to clip XY, force a valid
    // clip depth, and retain the imported source geometry unchanged.
    RenderTransform::from_columns([
        [0.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.5, 1.0],
    ])
    .expect("finite receiver fit")
}

fn visibility() -> RenderBounds {
    RenderBounds::new([-32.0; 3], [32.0; 3]).expect("finite scene bounds")
}

fn dragon_material() -> MaterialDefinition {
    MaterialDefinition::new(
        Vec::new(),
        "sample_dragon_albedo",
        vec![MaterialParameter {
            name: "dragon_tint".into(),
            ty: MaterialType::Vec4,
            resource: "dragon_tint".into(),
        }],
        vec![
            MaterialResource { name: "dragon_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "dragon_albedo".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "dragon_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
        ],
        BTreeMap::new(),
        SurfaceOutput::ColorUv0,
    )
    .with_code_node(MaterialCodeNode {
        name: "sample_dragon_albedo".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new("renderer-07-dragon-albedo", "fn sample_dragon_albedo(uv: vec2<f32>) -> vec4<f32> { return textureSample(dragon_albedo, dragon_sampler, uv) * dragon_tint; }"),
        resources: Vec::new(),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sample_dragon_albedo".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }],
        output: MaterialNodeId(1),
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
        _ => Err("a newly created asset did not yield a producer".into()),
    }
}
