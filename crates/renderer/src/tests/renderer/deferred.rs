//! Admission and completion ownership tests for the first Deferred probe.

use super::*;

use std::{collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{TextureFormat, VertexFormat, VertexStepMode};

use crate::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds, RenderChange,
    RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView, SceneAssetRef,
};

fn bounds() -> RenderBounds {
    RenderBounds::new([-1.0; 3], [1.0; 3]).expect("test bounds")
}

fn deferred_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = 2;
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 1;
        caps.limits.max_vertex_attributes = 1;
    })
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "tinted_texture".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn tinted_texture() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
        }],
        "tinted_texture",
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
}

fn scene_and_assets() -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().expect("geometry handle");
    let geometry = match geometry_store
        .acquire(&geometry_handle)
        .expect("geometry producer")
    {
        Acquire::Producer(producer) => producer
            .commit(
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
                    streams: vec![vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0,
                    ]],
                    vertices: 0..3,
                },
                ResidentBytes::new(36),
            )
            .expect("geometry snapshot"),
        _ => unreachable!(),
    };
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material_handle = material_store.create().expect("material handle");
    let material = match material_store
        .acquire(&material_handle)
        .expect("material producer")
    {
        Acquire::Producer(producer) => producer
            .commit(material(), ResidentBytes::new(0))
            .expect("material snapshot"),
        _ => unreachable!(),
    };
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let texture_handle = texture_store.create().expect("texture handle");
    let texture = match texture_store
        .acquire(&texture_handle)
        .expect("texture producer")
    {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![192, 96, 48, 255],
                },
                ResidentBytes::new(4),
            )
            .expect("texture snapshot"),
        _ => unreachable!(),
    };
    let resources = ForwardResources {
        geometry: SceneAssetRef::from_snapshot(&geometry),
        material: SceneAssetRef::from_snapshot(&material),
        texture: SceneAssetRef::from_snapshot(&texture),
        tint: [0.75, 0.5, 0.25, 1.0],
    };
    let assets = ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: vec![texture],
    };
    let mut scene = RenderScene::<ForwardResources>::new().expect("scene");
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).expect("view"),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources,
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let receipt = scene.apply(&changes).expect("scene insert");
    (scene.snapshot(), receipt.inserted_views()[0], assets)
}

fn submit(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit_deferred_probe(snapshot, view, bounds(), assets))
}

fn counters(native: &Native) -> (usize, usize, usize) {
    (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    )
}

#[test]
fn deferred_probe_records_four_graph_passes_and_retires_after_readback() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();

    let submission = submit(&mut renderer, snapshot, view, &assets).expect("submission");
    let facts = renderer.deferred_probe_facts().expect("deferred facts");
    assert_eq!(renderer.compilations(), 1);
    assert_eq!(renderer.plan_id(), Some(facts.plan));
    assert_eq!(facts.passes, 4);
    assert_eq!(
        facts.gbuffer_formats,
        [
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Depth32Float,
        ]
    );
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 4);
    assert_eq!(renderer.pending(), 1);

    let readback = block_on(renderer.readback(&submission)).expect("readback");
    assert_eq!(readback.len(), 8 * 8 * 4);
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    block_on(renderer.shutdown()).expect("successful shutdown");
    assert!(renderer.deferred_probe_facts().is_some());
    assert_eq!(renderer.plan_id(), None);
}

#[test]
fn deferred_probe_refuses_mrt_assets_and_foreign_view_before_device_work() {
    let (limited_device, limited_native) = fixture(Script::AcceptedExact);
    let mut limited = Renderer::new(limited_device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let before = counters(&limited_native);
    assert!(matches!(
        submit(&mut limited, snapshot, view, &assets),
        Err(RendererError::Rhi(Error::Unsupported(_)))
    ));
    assert_eq!(counters(&limited_native), before);

    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, _) = scene_and_assets();
    let before = counters(&native);
    assert!(matches!(
        submit(&mut renderer, snapshot, view, &ForwardAssets::default()),
        Err(RendererError::ForwardPreparation(_))
    ));
    assert_eq!(counters(&native), before);

    let (snapshot, _, assets) = scene_and_assets();
    let (_, foreign_view, _) = scene_and_assets();
    let before = counters(&native);
    assert!(matches!(
        submit(&mut renderer, snapshot, foreign_view, &assets),
        Err(RendererError::DeferredPipeline(_))
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn deferred_probe_fourth_submission_is_backpressure_before_allocation() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let mut submissions = Vec::new();
    for _ in 0..3 {
        submissions.push(submit(&mut renderer, snapshot.clone(), view, &assets).expect("submit"));
    }
    assert_eq!(renderer.pending(), 3);
    let before = counters(&native);
    assert!(matches!(
        submit(&mut renderer, snapshot, view, &assets),
        Err(RendererError::Backpressure)
    ));
    assert_eq!(counters(&native), before);

    for submission in submissions {
        block_on(renderer.readback(&submission)).expect("retire submission");
    }
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn deferred_completion_loss_keeps_pending_owner_and_closes_renderer() {
    let (device, native) = deferred_device(Script::AcceptedCompletionDeviceLost);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let submission = submit(&mut renderer, snapshot.clone(), view, &assets).expect("submit");

    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    let before = counters(&native);
    assert!(matches!(
        submit(&mut renderer, snapshot, view, &assets),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn deferred_cpu_readback_failure_releases_completed_owner() {
    let (device, _) = deferred_device(Script::AcceptedReadbackDeviceLost);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let submission = submit(&mut renderer, snapshot.clone(), view, &assets).expect("submit");

    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 0);
    assert!(matches!(
        submit(&mut renderer, snapshot, view, &assets),
        Err(RendererError::RendererClosed)
    ));
}

#[test]
fn deferred_probe_diagnostic_is_pending_owner_scoped() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let submission = submit(&mut renderer, snapshot, view, &assets).expect("submit");

    assert!(
        renderer
            .deferred_probe_diagnostic(&submission)
            .expect("pending diagnostic")
            .is_some()
    );
    block_on(renderer.readback(&submission)).expect("readback");
    assert!(matches!(
        renderer.deferred_probe_diagnostic(&submission),
        Err(RendererError::StaleSubmission)
    ));
}

#[test]
fn deferred_probe_diagnostic_rejects_a_foreign_renderer_submission() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let mut owner = Renderer::new(device).expect("owner renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let submission = submit(&mut owner, snapshot, view, &assets).expect("submit");
    let (other_device, _) = deferred_device(Script::AcceptedExact);
    let other = Renderer::new(other_device).expect("other renderer");

    assert!(matches!(
        other.deferred_probe_diagnostic(&submission),
        Err(RendererError::ForeignSubmission)
    ));
    block_on(owner.readback(&submission)).expect("owner readback");
}

#[path = "deferred_cache.rs"]
mod cache_tests;
#[path = "deferred_code_nodes.rs"]
mod code_node_tests;
#[path = "deferred_deformation.rs"]
mod deformation_tests;
#[path = "deferred_owners.rs"]
mod owner_tests;
#[path = "deferred_pbr.rs"]
mod pbr_tests;
#[path = "deferred_pipeline.rs"]
mod pipeline_tests;
#[path = "deferred_surface.rs"]
mod surface_tests;
