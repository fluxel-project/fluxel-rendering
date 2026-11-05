//! S8 high-cardinality Forward lifecycle witnesses.
//!
//! These submit real Forward scenes, so the optimized texture manifest and
//! pass-access checks run before the witness backend supplies a lifecycle
//! outcome. They deliberately do not claim native-GPU fault injection.

use super::*;

use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::Ordering},
};

use fluxel_assets::{Acquire, AssetStore, Production, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout,
    GeometryStream, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, RendererError, SceneAssetRef,
};

const TEXTURES: usize = 12;

struct Assets {
    initial: ForwardAssets,
    replacement: ForwardAssets,
}

fn s8_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        // A real Forward Color program owns a material and a transform group.
        // The generic lifecycle fixture deliberately advertises only one.
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 1;
        caps.limits.max_vertex_attributes = 2;
    })
}

fn bounds() -> RenderBounds {
    RenderBounds::new([-1.0; 3], [1.0; 3]).unwrap()
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "s8_color".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn s8_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
        }],
        "s8_color",
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

fn geometry() -> GeometryData {
    let mut bytes = Vec::new();
    for point in [[-0.8_f32, -0.8, 0.5], [0.8, -0.8, 0.5], [0.0, 0.8, 0.5]] {
        for value in point {
            bytes.extend(value.to_le_bytes());
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

fn assets() -> Assets {
    let geometries = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometries.create().unwrap();
    let geometry = match geometries.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(geometry(), ResidentBytes::new(36)).unwrap(),
        _ => unreachable!(),
    };
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material_handle = materials.create().unwrap();
    let material = match materials.acquire(&material_handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(material(), ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    let store = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let mut initial = Vec::with_capacity(TEXTURES);
    let mut handles = Vec::with_capacity(TEXTURES);
    for index in 0..TEXTURES {
        let handle = store.create().unwrap();
        let snapshot = match store.acquire(&handle).unwrap() {
            Acquire::Producer(producer) => producer
                .commit(
                    ColorTextureData {
                        extent: [1, 1],
                        rgba8: vec![index as u8 + 1, 64, 192, 255],
                    },
                    ResidentBytes::new(4),
                )
                .unwrap(),
            _ => unreachable!(),
        };
        handles.push(handle);
        initial.push(snapshot);
    }
    let mut replacement = initial.clone();
    replacement[0] = match store.request_replacement(&handles[0]).unwrap() {
        Production::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![250, 31, 53, 255],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        Production::Waiter(_) => unreachable!(),
    };
    Assets {
        initial: ForwardAssets {
            geometry: vec![geometry.clone()],
            materials: vec![material.clone()],
            textures: initial,
        },
        replacement: ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: replacement,
        },
    }
}

fn scene(assets: &ForwardAssets) -> (crate::FrameSnapshot<ForwardResources>, crate::RenderViewId) {
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    for texture in &assets.textures {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
                material: SceneAssetRef::from_snapshot(&assets.materials[0]),
                texture: SceneAssetRef::from_snapshot(texture),
                tint: [1.0; 4],
            },
            transform: RenderTransform::identity(),
            bounds: bounds(),
        }));
    }
    let receipt = scene.apply(&changes).unwrap();
    (scene.snapshot(), receipt.inserted_views()[0])
}

fn submit(
    renderer: &mut Renderer<WitnessBackend>,
    assets: &ForwardAssets,
) -> std::result::Result<Submission, RendererError> {
    let (snapshot, view) = scene(assets);
    block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        assets,
    ))
}

fn retire(renderer: &mut Renderer<WitnessBackend>, submission: &Submission) {
    let pending_before = renderer.pending();
    let bytes = block_on(renderer.readback(submission)).expect("completed Forward readback");
    // The real Forward target is 8x8 RGBA. Verifying its shape ensures this is
    // not merely a successful completion wait used as a cache shortcut.
    assert_eq!(bytes.len(), 8 * 8 * 4);
    assert_eq!(renderer.pending(), pending_before - 1);
}

#[test]
fn s8_many_textures_replace_while_old_generation_is_pending_then_reuse_old_return() {
    let (device, _) = s8_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let assets = assets();
    let old = submit(&mut renderer, &assets.initial).unwrap();
    let replacement = submit(&mut renderer, &assets.replacement).unwrap();
    assert_eq!(renderer.pending(), 2);
    assert_eq!(renderer.forward_cache_stats().texture_hits, 0);
    retire(&mut renderer, &old);
    assert_eq!(renderer.pending(), 1);
    retire(&mut renderer, &replacement);
    let before = renderer.forward_cache_stats().texture_hits;
    let returned = submit(&mut renderer, &assets.initial).unwrap();
    assert!(renderer.forward_cache_stats().texture_hits >= before + TEXTURES as u64);
    retire(&mut renderer, &returned);
}

#[test]
fn s8_many_textures_cancelled_wait_retains_pending_owner() {
    let (device, native) = s8_device(Script::AcceptedWaitPending);
    let mut renderer = Renderer::new(device).unwrap();
    let assets = assets();
    let retained_texture = Arc::downgrade(&assets.initial.textures[0].value_arc());
    let submission = submit(&mut renderer, &assets.initial).unwrap();
    drop(assets);
    assert_eq!(renderer.forward_cache_stats().opaque_draws, TEXTURES as u64);
    poll_once_pending(renderer.readback(&submission));
    assert_eq!(renderer.pending(), 1);
    assert!(retained_texture.upgrade().is_some());
    assert!(renderer.forward_frames.iter().all(Option::is_none));
    assert_eq!(renderer.forward_cache_stats().texture_hits, 0);
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
}

#[test]
fn s8_many_textures_upload_or_recording_failure_does_not_publish_cache() {
    for script in [
        Script::MapDeviceLost,
        Script::FlushDeviceLost,
        Script::RecordingFailureAfterPrefix,
    ] {
        let (device, _) = s8_device(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let assets = assets();
        let result = submit(&mut renderer, &assets.initial);
        if matches!(script, Script::RecordingFailureAfterPrefix) {
            assert!(matches!(
                result,
                Err(RendererError::RetainedExecution { .. })
            ));
            assert_eq!(renderer.pending(), 1);
        } else {
            assert!(result.is_err());
            assert_eq!(renderer.pending(), 0);
        }
        assert_eq!(renderer.forward_cache_stats().texture_hits, 0);
        assert!(renderer.forward_frames.iter().all(Option::is_none));
        for texture in &assets.initial.textures {
            let key = crate::residency::ForwardTextureCacheKey::from_snapshot(texture, device.id());
            assert!(renderer.forward_residency.texture(key).is_none());
        }
    }
}

#[test]
fn s8_many_textures_devices_are_isolated_and_old_device_reuses_its_own_products() {
    let (old_device, _) = s8_device(Script::AcceptedExact);
    let (new_device, _) = s8_device(Script::AcceptedExact);
    let mut old_renderer = Renderer::new(old_device).unwrap();
    let mut new_renderer = Renderer::new(new_device).unwrap();
    let assets = assets();
    let old = submit(&mut old_renderer, &assets.initial).unwrap();
    assert!(matches!(
        block_on(new_renderer.readback(&old)),
        Err(RendererError::ForeignSubmission)
    ));
    retire(&mut old_renderer, &old);
    let new = submit(&mut new_renderer, &assets.initial).unwrap();
    assert_eq!(new_renderer.forward_cache_stats().texture_hits, 0);
    retire(&mut new_renderer, &new);
    let old_return = submit(&mut old_renderer, &assets.initial).unwrap();
    assert!(old_renderer.forward_cache_stats().texture_hits >= TEXTURES as u64);
    retire(&mut old_renderer, &old_return);
}
