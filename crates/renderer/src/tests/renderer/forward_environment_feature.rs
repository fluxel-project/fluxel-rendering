//! Witness coverage for environment-lit PBR draws emitted by a typed feature.

use super::pbr_tests::{environment_device, geometry, irradiance_snapshot, scene_and_assets};
use super::*;

use fluxel_shader::ShaderComposer;
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
        texture: SceneAssetRef::from_snapshot(&assets.textures[0]),
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
