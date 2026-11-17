//! Witness coverage for environment-lit PBR draws emitted by a typed feature.

use super::pbr_tests::{environment_device, geometry, irradiance_snapshot, scene_and_assets};
use super::*;

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_shader::ShaderComposer;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use crate::{
    BoundsVisibility, CpuPhase, FeatureBinding, FeatureError, ForwardCpuPipeline,
    ForwardDrawRequest, ForwardResources, RenderBounds, RenderChange, RenderChangeSet,
    RenderFeature, RenderRecord, RenderScene, RenderTransform, RenderView, SceneAssetRef,
};

struct PbrFeature {
    handle: crate::FeatureHandle,
}

impl RenderFeature for PbrFeature {
    type Input = Vec<ForwardDrawRequest>;
    type Prepared = Vec<ForwardDrawRequest>;

    fn handle(&self) -> crate::FeatureHandle {
        self.handle
    }

    fn prepare(
        &self,
        _: &crate::FeatureContext<'_, ForwardResources>,
        input: &crate::TypedFeatureSnapshot<Self::Input>,
        _: &crate::PreparedFrame<'_, ForwardResources>,
        _: RenderBounds,
    ) -> std::result::Result<Self::Prepared, FeatureError> {
        Ok(input.data().clone())
    }

    fn draws<'a>(&self, prepared: &'a Self::Prepared) -> &'a [ForwardDrawRequest] {
        prepared
    }
}

fn bounds() -> RenderBounds {
    RenderBounds::new([-1.0; 3], [1.0; 3]).expect("bounds")
}

fn feature_resources(assets: &crate::ForwardAssets) -> ForwardResources {
    ForwardResources {
        geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
        material: SceneAssetRef::from_snapshot(&assets.materials[0]),
        textures: vec![crate::ForwardTextureBinding::asset(
            "material_texture",
            SceneAssetRef::from_snapshot(&assets.textures[0]),
        )],
        tint: [1.0; 4],
    }
}

fn scene_with_view() -> (
    RenderScene<ForwardResources>,
    crate::FeatureIssuer,
    crate::RenderViewId,
) {
    let mut scene = RenderScene::new().expect("scene");
    let issuer = scene.feature_issuer().expect("feature issuer");
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).expect("view"),
    ));
    let view = scene
        .apply(&changes)
        .expect("view insertion")
        .inserted_views()[0];
    (scene, issuer, view)
}

fn feature_input(
    issuer: &crate::FeatureIssuer,
    scene: &RenderScene<ForwardResources>,
    handle: crate::FeatureHandle,
    request: ForwardDrawRequest,
) -> (
    crate::TypedFeatureStorage<Vec<ForwardDrawRequest>>,
    crate::TypedFeatureSnapshot<Vec<ForwardDrawRequest>>,
) {
    let mut storage = issuer.storage::<Vec<ForwardDrawRequest>>();
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("feature context");
    let input = storage
        .publish(&context, vec![request])
        .expect("feature input");
    (storage, input)
}

fn color_material_snapshot() -> fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition> {
    let store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let handle = store.create().expect("material handle");
    match store.acquire(&handle).expect("material producer") {
        Acquire::Producer(producer) => producer
            .commit(
                MaterialDefinition::new(
                    vec![MaterialFragment {
                        name: "color".into(),
                        inputs: Vec::new(),
                        output: MaterialType::Vec4,
                        calls: Vec::new(),
                        wgsl: "fn color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
                    }],
                    "color",
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
                ),
                ResidentBytes::new(0),
            )
            .expect("color material snapshot"),
        _ => unreachable!(),
    }
}

fn mixed_requests(assets: &mut crate::ForwardAssets) -> Vec<ForwardDrawRequest> {
    let color = color_material_snapshot();
    let color_resources = ForwardResources {
        geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
        material: SceneAssetRef::from_snapshot(&color),
        textures: vec![crate::ForwardTextureBinding::asset(
            "material_texture",
            SceneAssetRef::from_snapshot(&assets.textures[0]),
        )],
        tint: [1.0; 4],
    };
    assets.materials.push(color);
    vec![
        pbr_request(assets, 0, 0),
        ForwardDrawRequest {
            local_id: 1,
            order: 1,
            resources: color_resources,
            transform: RenderTransform::identity(),
        },
    ]
}

fn mixed_feature_input(
    issuer: &crate::FeatureIssuer,
    scene: &RenderScene<ForwardResources>,
    handle: crate::FeatureHandle,
    requests: Vec<ForwardDrawRequest>,
) -> (
    crate::TypedFeatureStorage<Vec<ForwardDrawRequest>>,
    crate::TypedFeatureSnapshot<Vec<ForwardDrawRequest>>,
) {
    let mut storage = issuer.storage::<Vec<ForwardDrawRequest>>();
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("feature context");
    let input = storage.publish(&context, requests).expect("feature input");
    (storage, input)
}

fn pbr_request(assets: &crate::ForwardAssets, local_id: u64, order: i64) -> ForwardDrawRequest {
    ForwardDrawRequest {
        local_id,
        order,
        resources: feature_resources(assets),
        transform: RenderTransform::identity(),
    }
}

#[test]
fn feature_only_pbr_environment_binds_group_two_copies_and_keeps_owner_pending() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (_, _, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let (scene, issuer, view) = scene_with_view();
    let handle = issuer.issue("pbr-feature").expect("feature handle");
    let feature = PbrFeature { handle };
    let (storage, input) = feature_input(&issuer, &scene, handle, pbr_request(&assets, 0, 0));
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .expect("environment selection");

    let submission = block_on(renderer.submit_forward_with_feature(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.snapshot(),
        view,
        bounds(),
        &assets,
        FeatureBinding::new(&issuer, &feature, &input),
    ))
    .expect("feature-only environment submit");
    let _ = storage;

    // PBR's environment route realizes group 2 (texture plus sampler) on this
    // three-group device; the sampler is distinct from the textureLoad material.
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    assert_eq!(native.texture_uploads.lock().expect("uploads").len(), 2);
    assert!(
        native
            .mapped_bytes
            .lock()
            .expect("mapped bytes")
            .iter()
            .any(|bytes| bytes.len() == 256 && bytes[..8] == [1, 2, 3, 255, 4, 5, 6, 255])
    );
    assert_eq!(renderer.forward_cache_stats().opaque_draws, 1);
    assert!(renderer.pending_for(&submission).is_ok());
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.readback(&submission)).expect("readback");
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
}

#[test]
fn scene_and_feature_pbr_environment_share_one_upload_and_pending_owner() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (_, _, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let (mut scene, issuer, view) = scene_with_view();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: feature_resources(&assets),
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    scene.apply(&changes).expect("scene draw insertion");
    let handle = issuer.issue("pbr-feature").expect("feature handle");
    let feature = PbrFeature { handle };
    let (storage, input) = feature_input(&issuer, &scene, handle, pbr_request(&assets, 3, 1));
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([9, 8, 7, 255, 6, 5, 4, 255])))
        .expect("environment selection");

    let submission = block_on(renderer.submit_forward_with_feature(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.snapshot(),
        view,
        bounds(),
        &assets,
        FeatureBinding::new(&issuer, &feature, &input),
    ))
    .expect("scene plus feature environment submit");
    let _ = storage;

    assert_eq!(renderer.forward_cache_stats().opaque_draws, 2);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    // The shared environment is copied once, alongside the shared material texture.
    assert_eq!(native.texture_uploads.lock().expect("uploads").len(), 2);
    assert!(renderer.pending_for(&submission).is_ok());
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.readback(&submission)).expect("readback");
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
}

#[test]
fn public_prepare_material_rejects_environment_static_option_without_group_two_bindings() {
    let (_, _, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let layout = geometry(true, [0.0, 0.0, 1.0]).layout;
    let request = crate::pbr::forward_request(&layout)
        .expect("forward PBR request")
        .with_static_option("renderer.environment", "latlong-diffuse-irradiance-v1");

    assert!(matches!(
        crate::prepare_material(
            &assets.materials[0],
            &layout,
            &mut ShaderComposer::new(),
            request,
        ),
        Err(crate::MaterialPreparationError::InvalidEnvironmentRoute)
    ));
}

#[test]
fn mixed_pbr_and_color_feature_draws_share_one_environment_upload_offscreen() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (_, _, mut assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let (scene, issuer, view) = scene_with_view();
    let handle = issuer.issue("mixed-feature").expect("feature handle");
    let feature = PbrFeature { handle };
    let (storage, input) =
        mixed_feature_input(&issuer, &scene, handle, mixed_requests(&mut assets));
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .expect("environment selection");

    let submission = block_on(renderer.submit_forward_with_feature(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.snapshot(),
        view,
        bounds(),
        &assets,
        FeatureBinding::new(&issuer, &feature, &input),
    ))
    .expect("mixed offscreen submission");
    let _ = storage;

    assert_eq!(renderer.forward_cache_stats().opaque_draws, 2);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    assert_eq!(native.texture_uploads.lock().expect("uploads").len(), 2);
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.readback(&submission)).expect("readback");
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
}

#[test]
fn mixed_pbr_and_color_feature_draws_share_one_environment_upload_on_surface() {
    let (device, native) = environment_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (_, _, mut assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let (scene, issuer, view) = scene_with_view();
    let handle = issuer.issue("mixed-feature").expect("feature handle");
    let feature = PbrFeature { handle };
    let (storage, input) =
        mixed_feature_input(&issuer, &scene, handle, mixed_requests(&mut assets));
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .expect("environment selection");
    let mut surface = super::surface_tests::surface(&device, [8, 8]);

    let submission = block_on(renderer.submit_forward_surface_with_feature(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.snapshot(),
        view,
        bounds(),
        &assets,
        FeatureBinding::new(&issuer, &feature, &input),
        block_on(surface.acquire()).expect("surface image"),
    ))
    .expect("mixed surface submission");
    let _ = storage;

    assert_eq!(renderer.forward_cache_stats().opaque_draws, 2);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
    assert_eq!(native.texture_uploads.lock().expect("uploads").len(), 2);
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.complete(&submission)).expect("present completion");
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
}
