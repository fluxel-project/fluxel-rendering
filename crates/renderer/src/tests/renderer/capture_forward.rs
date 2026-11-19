//! Same-Graph capture submission witness coverage.

use super::*;
use crate::{BoundsVisibility, ForwardCpuPipeline};
use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialCodeNode, MaterialNode, MaterialNodeGraph,
    MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
};
use fluxel_shader::{ShaderComposer, SourceUnit};

#[path = "capture_mixed_fixture.rs"]
pub(crate) mod mixed_fixture;

pub(super) fn receiver() -> (crate::PreparedCaptureReceiver, ForwardAssets) {
    receiver_at([0, 1, 2])
}

fn receiver_at(locations: [u32; 3]) -> (crate::PreparedCaptureReceiver, ForwardAssets) {
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry = match geometry_store
        .acquire(&geometry_store.create().unwrap())
        .unwrap()
    {
        Acquire::Producer(p) => p
            .commit(
                GeometryData {
                    layout: GeometryLayout {
                        streams: vec![GeometryStream {
                            stride: 20,
                            step_mode: VertexStepMode::Vertex,
                            attributes: vec![
                                GeometryAttribute {
                                    semantic: "POSITION".into(),
                                    offset: 0,
                                    format: VertexFormat::Float32(3),
                                },
                                GeometryAttribute {
                                    semantic: "UV0".into(),
                                    offset: 12,
                                    format: VertexFormat::Float32(2),
                                },
                            ],
                        }],
                    },
                    streams: vec![vec![0; 60]],
                    vertices: 0..3,
                },
                ResidentBytes::new(60),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material = match material_store.acquire(&material_store.create().unwrap()).unwrap() {
        Acquire::Producer(p) => p.commit(MaterialDefinition::new(Vec::new(), "unused", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "tint".into() }], vec![MaterialResource { name: "tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: locations[0] } }, MaterialResource { name: "captured".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: locations[1] } }, MaterialResource { name: "capture_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: locations[2] } }], BTreeMap::new(), SurfaceOutput::ColorUv0).with_code_node(MaterialCodeNode { name: "sample".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec4, calls: Vec::new(), resources: Vec::new(), source: SourceUnit::new("capture", "fn sample(uv: vec2<f32>) -> vec4<f32> { return textureSample(captured, capture_sampler, uv) * tint; }") }).with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sample".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(1) }), ResidentBytes::new(0)).unwrap(), _ => unreachable!(),
    };
    let assets = ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: Vec::new(),
    };
    let resources = crate::ForwardCaptureResources {
        geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
        material: SceneAssetRef::from_snapshot(&assets.materials[0]),
        textures: vec![crate::ForwardTextureBinding::capture(
            "captured", "captured",
        )],
        samplers: Vec::new(),
        tint: [1.0; 4],
    };
    let prepared = crate::prepare_forward_capture_receiver(
        &RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
        crate::ForwardDrawSource::Feature {
            feature: crate::FeatureIssuer::new(0, 0).issue("capture").unwrap(),
            local_id: 1,
        },
        0,
        &resources,
        RenderTransform::identity(),
        &assets,
        &mut ShaderComposer::new(),
    )
    .unwrap();
    (prepared, assets)
}

pub(super) fn capture_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 7;
        caps.limits.max_vertex_attributes = 2;
    })
}

#[test]
fn capture_submission_records_five_scopes_and_retains_until_receiver_readback() {
    let (device, native) = capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets();
    let (receiver, receiver_assets) = receiver();
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("captured").unwrap(),
        receiver,
    ))
    .unwrap();
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 5);
    assert!(block_on(renderer.readback(&submission)).is_ok());
    assert_eq!(renderer.pending(), 0);
    drop(receiver_assets);
}

#[test]
fn capture_completion_failure_keeps_both_frames_pending() {
    let (device, _) = capture_device(Script::AcceptedCompletionDeviceLost);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = mixed_fixture::receiver(&mut assets, false);
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .unwrap();
    assert!(block_on(renderer.readback(&submission)).is_err());
    assert_eq!(renderer.pending(), 1);
}

#[test]
fn capture_receiver_consumes_permuted_material_locations_without_a_cpu_asset() {
    let (device, native) = capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets();
    let (receiver, receiver_assets) = receiver_at([2, 0, 1]);
    assert!(receiver_assets.textures.is_empty());
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("captured").unwrap(),
        receiver,
    ))
    .expect("permuted capture receiver bindings");
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 5);
    block_on(renderer.readback(&submission)).expect("combined completion");
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn mixed_receiver_keeps_nonfirst_capture_and_asset_slots_in_schema_order() {
    let (device, _) = capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = mixed_fixture::receiver(&mut assets, true);
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .expect("mixed capture submission");
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::CaptureForward { _receiver, .. } = &pending.workload else {
        panic!("capture pending work")
    };
    let draw = _receiver.draws.first().unwrap();
    assert_eq!(draw.textures.len(), 5);
    assert_eq!(draw.capture_textures, vec![false, false, false, true, true]);
    assert_eq!(draw.textures[3], draw.textures[4]);
    block_on(renderer.readback(&submission)).unwrap();
}

#[test]
fn mixed_capture_rejects_a_missing_explicit_producer_before_backend_work() {
    let (device, native) = capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = mixed_fixture::receiver(&mut assets, false);
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.created_textures.load(Ordering::SeqCst),
        native.created_samplers.load(Ordering::SeqCst),
        native.pipeline_color_formats.lock().unwrap().len(),
    );
    assert!(matches!(
        block_on(renderer.submit_forward_capture(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
            crate::CaptureTextureSlot::new("wrong_producer").unwrap(),
            receiver,
        )),
        Err(RendererError::ForwardPreparation(
            crate::ForwardPreparationError::MissingCaptureSlot { .. }
        ))
    ));
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        (
            native.created_buffers.load(Ordering::SeqCst),
            native.created_textures.load(Ordering::SeqCst),
            native.created_samplers.load(Ordering::SeqCst),
            native.pipeline_color_formats.lock().unwrap().len(),
        ),
        before,
        "missing producer is rejected before backend allocation or pipeline creation"
    );
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn mixed_capture_allows_a_nonfilterable_texture_load_receiver() {
    let (device, native) = capture_device(Script::AcceptedExact);
    native.texture_filterable.store(false, Ordering::SeqCst);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = mixed_fixture::texture_load_receiver(&mut assets);
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .expect("textureLoad capture does not require filtering");
    block_on(renderer.readback(&submission)).unwrap();
}

#[test]
fn mixed_capture_publishes_shared_assets_only_after_completion_then_warms() {
    let (device, _) = capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = mixed_fixture::receiver(&mut assets, false);
    let cold = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver.clone(),
    ))
    .unwrap();
    assert_eq!(renderer.forward_cache_stats().texture_hits, 0);
    block_on(renderer.readback(&cold)).unwrap();
    let before_warm = renderer.forward_cache_stats().texture_hits;
    let warm = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .unwrap();
    assert!(renderer.forward_cache_stats().texture_hits > before_warm);
    let pending = renderer.pending_for(&warm).unwrap();
    let crate::renderer::PendingWork::CaptureForward {
        _capture,
        _receiver,
        ..
    } = &pending.workload
    else {
        panic!("capture pending work")
    };
    assert_eq!(
        _capture.resident.streams[0].instance, _receiver.resident.streams[0].instance,
        "warm capture and receiver share the published geometry owner"
    );
    assert_eq!(
        _capture.resident.textures[_capture.draws[0].textures[0]].instance,
        _receiver.resident.textures[_receiver.draws[0].textures[0]].instance,
        "warm capture and receiver share the published immutable texture owner"
    );
    block_on(renderer.readback(&warm)).unwrap();
}

#[test]
fn mixed_capture_declares_depth_for_both_producer_and_receiver() {
    let (device, native) = capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = mixed_fixture::receiver(&mut assets, false);
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .expect("capture receiver contributes the selected depth attachment");
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::CaptureForward {
        _capture,
        _receiver,
        ..
    } = &pending.workload
    else {
        panic!("capture")
    };
    assert!(_capture.frame.depth.is_some());
    assert!(_receiver.frame.depth.is_some());
    assert!(
        native
            .pipeline_depth_states
            .lock()
            .unwrap()
            .iter()
            .all(|state| state.is_some())
    );
    block_on(renderer.readback(&submission)).unwrap();
}
