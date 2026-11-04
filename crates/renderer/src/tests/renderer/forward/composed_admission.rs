//! Additional typed-composition admission boundaries.

use super::*;
use crate::{ForwardDrawRequest, ForwardPreparationError, ForwardResources, SceneAssetRef};
use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    MaterialAsset, MaterialDefinition, MaterialFragment, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use std::{cell::Cell, collections::BTreeMap, sync::atomic::Ordering};

struct InvalidContributionPipeline {
    mode: InvalidContribution,
}

#[derive(Clone, Copy)]
enum InvalidContribution {
    DuplicateMeshes,
    DuplicateFeature,
    MissingUpload,
}

impl FramePipeline<crate::ForwardResources> for InvalidContributionPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<crate::PreparedFrame<'a, crate::ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
    ) -> std::result::Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::OpaqueForward)
    }

    fn contribute_composed<B: fluxel_rhi::backend::ResourceBackend, F: RenderFeature>(
        &self,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
        context: &mut crate::ForwardCompositionContext<'_, '_, '_, B, F>,
    ) -> std::result::Result<(), PipelineError> {
        context.opaque(|pass| {
            if !matches!(self.mode, InvalidContribution::MissingUpload) {
                pass.depends_on_uploads()?;
            }
            pass.meshes(|mesh| {
                mesh.read_vertex_streams()?;
                mesh.read_frame_uniforms()?;
                mesh.read_sampled_textures()
            })?;
            if matches!(self.mode, InvalidContribution::DuplicateMeshes) {
                pass.meshes(|_| Ok(()))?;
            }
            pass.feature()?;
            if matches!(self.mode, InvalidContribution::DuplicateFeature) {
                pass.feature()?;
            }
            pass.write_main_color()
        })
    }
}

struct MissingAssetFeature {
    handle: crate::FeatureHandle,
    requests: Vec<ForwardDrawRequest>,
    prepared: Cell<usize>,
}

impl RenderFeature for MissingAssetFeature {
    type Input = ();
    type Prepared = Vec<ForwardDrawRequest>;

    fn handle(&self) -> crate::FeatureHandle {
        self.handle
    }

    fn prepare(
        &self,
        _: &crate::FeatureContext<'_, crate::ForwardResources>,
        _: &crate::TypedFeatureSnapshot<Self::Input>,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
        _: RenderBounds,
    ) -> std::result::Result<Self::Prepared, FeatureError> {
        self.prepared.set(self.prepared.get() + 1);
        Ok(self.requests.clone())
    }

    fn draws<'a>(&self, prepared: &'a Self::Prepared) -> &'a [ForwardDrawRequest] {
        prepared
    }
}

fn unavailable_resources() -> ForwardResources {
    let geometry_store = AssetStore::<crate::GeometryAsset, crate::GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().expect("geometry handle");
    let geometry = match geometry_store
        .acquire(&geometry_handle)
        .expect("geometry producer")
    {
        Acquire::Producer(producer) => producer
            .commit(
                crate::GeometryData {
                    layout: crate::GeometryLayout::default(),
                    streams: Vec::new(),
                    vertices: 0..0,
                },
                ResidentBytes::new(0),
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
            .commit(
                MaterialDefinition::new(
                    Vec::new(),
                    "unused",
                    Vec::new(),
                    Vec::new(),
                    BTreeMap::new(),
                    SurfaceOutput::Color,
                ),
                ResidentBytes::new(0),
            )
            .expect("material snapshot"),
        _ => unreachable!(),
    };
    let texture_store = AssetStore::<crate::ColorTextureAsset, crate::ColorTextureData, ()>::new();
    let texture_handle = texture_store.create().expect("texture handle");
    let texture = match texture_store
        .acquire(&texture_handle)
        .expect("texture producer")
    {
        Acquire::Producer(producer) => producer
            .commit(
                crate::ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![0; 4],
                },
                ResidentBytes::new(4),
            )
            .expect("texture snapshot"),
        _ => unreachable!(),
    };
    ForwardResources {
        geometry: SceneAssetRef::from_snapshot(&geometry),
        material: SceneAssetRef::from_snapshot(&material),
        texture: SceneAssetRef::from_snapshot(&texture),
        tint: [1.0; 4],
    }
}

fn assets_with_texture_payload(bytes: Vec<u8>) -> (ForwardResources, crate::ForwardAssets) {
    let geometry_store = AssetStore::<crate::GeometryAsset, crate::GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().expect("geometry handle");
    let geometry = match geometry_store
        .acquire(&geometry_handle)
        .expect("geometry producer")
    {
        Acquire::Producer(producer) => producer
            .commit(
                crate::GeometryData {
                    layout: crate::GeometryLayout {
                        streams: vec![crate::GeometryStream {
                            stride: 12,
                            step_mode: VertexStepMode::Vertex,
                            attributes: vec![crate::GeometryAttribute {
                                semantic: "POSITION".into(),
                                offset: 0,
                                format: VertexFormat::Float32(3),
                            }],
                        }],
                    },
                    streams: vec![vec![0; 12]],
                    vertices: 0..1,
                },
                ResidentBytes::new(12),
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
            .commit(
                MaterialDefinition::new(
                    vec![MaterialFragment {
                        name: "unused".into(),
                        inputs: Vec::new(),
                        output: MaterialType::Vec4,
                        calls: Vec::new(),
                        wgsl: "fn unused() -> vec4<f32> { return vec4<f32>(1.0); }".into(),
                    }],
                    "unused",
                    Vec::new(),
                    Vec::new(),
                    BTreeMap::new(),
                    SurfaceOutput::Color,
                ),
                ResidentBytes::new(0),
            )
            .expect("material snapshot"),
        _ => unreachable!(),
    };
    let texture_store = AssetStore::<crate::ColorTextureAsset, crate::ColorTextureData, ()>::new();
    let texture_handle = texture_store.create().expect("texture handle");
    let byte_len = u64::try_from(bytes.len()).expect("texture byte length");
    let texture = match texture_store
        .acquire(&texture_handle)
        .expect("texture producer")
    {
        Acquire::Producer(producer) => producer
            .commit(
                crate::ColorTextureData {
                    extent: [1, 1],
                    rgba8: bytes,
                },
                ResidentBytes::new(byte_len),
            )
            .expect("texture snapshot"),
        _ => unreachable!(),
    };
    let resources = ForwardResources {
        geometry: SceneAssetRef::from_snapshot(&geometry),
        material: SceneAssetRef::from_snapshot(&material),
        texture: SceneAssetRef::from_snapshot(&texture),
        tint: [1.0; 4],
    };
    let assets = crate::ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: vec![texture],
    };
    (resources, assets)
}

fn counters(native: &Native) -> (usize, usize, usize) {
    (
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
        native.finishes.load(Ordering::SeqCst),
    )
}

#[test]
fn same_scene_foreign_issuer_payload_is_rejected_before_prepare_or_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let other_issuer = scene.feature_issuer().expect("other issuer");
    let handle = issuer.issue("primary").expect("primary handle");
    let feature = EmptyFeature::new(handle);
    let foreign_handle = other_issuer.issue("other").expect("other handle");
    let snapshot = scene.snapshot();
    let foreign_context = other_issuer
        .context(&snapshot, CpuPhase::Reference, foreign_handle)
        .expect("other context");
    let mut foreign_storage = other_issuer.storage::<()>();
    let payload = foreign_storage
        .publish(&foreign_context, ())
        .expect("other payload");
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
        native.finishes.load(Ordering::SeqCst),
    );
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &SwitchPipeline {
                omit_color: Cell::new(false)
            },
            snapshot,
            view,
            bounds(),
            &assets(),
            FeatureBinding::new(&issuer, &feature, &payload),
        )),
        Err(RendererError::ForwardFeature(FeatureError::ForeignIssuer))
    ));
    assert_eq!(feature.prepared.get(), 0);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
}

#[test]
fn foreign_feature_handle_is_rejected_before_prepare_or_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let local_handle = issuer.issue("local").expect("local handle");
    let foreign_scene = RenderScene::<crate::ForwardResources>::new().expect("foreign scene");
    let foreign_handle = foreign_scene
        .feature_issuer()
        .expect("foreign issuer")
        .issue("foreign")
        .expect("foreign handle");
    let feature = EmptyFeature::new(foreign_handle);
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, local_handle)
        .expect("local context");
    let mut storage = issuer.storage::<()>();
    let payload = storage.publish(&context, ()).expect("local payload");
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
        native.finishes.load(Ordering::SeqCst),
    );
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &SwitchPipeline {
                omit_color: Cell::new(false)
            },
            snapshot,
            view,
            bounds(),
            &assets(),
            FeatureBinding::new(&issuer, &feature, &payload),
        )),
        Err(RendererError::ForwardFeature(
            FeatureError::FeatureHandleMismatch { .. }
        ))
    ));
    assert_eq!(feature.prepared.get(), 0);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
}

#[test]
fn missing_feature_resource_generation_is_rejected_before_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("missing-resource").expect("handle");
    let feature = MissingAssetFeature {
        handle,
        requests: vec![ForwardDrawRequest {
            local_id: 4,
            order: 0,
            resources: unavailable_resources(),
            transform: RenderTransform::identity(),
        }],
        prepared: Cell::new(0),
    };
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("context");
    let mut storage = issuer.storage::<()>();
    let payload = storage.publish(&context, ()).expect("payload");
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
        native.finishes.load(Ordering::SeqCst),
    );
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &SwitchPipeline {
                omit_color: Cell::new(false)
            },
            snapshot,
            view,
            bounds(),
            &assets(),
            FeatureBinding::new(&issuer, &feature, &payload),
        )),
        Err(RendererError::ForwardPreparation(
            ForwardPreparationError::MissingAsset {
                asset: "geometry",
                ..
            }
        ))
    ));
    assert_eq!(feature.prepared.get(), 1);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
}

#[test]
fn feature_request_with_present_invalid_texture_payload_is_rejected_before_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("invalid-texture").expect("handle");
    let (resources, feature_assets) = assets_with_texture_payload(vec![0; 3]);
    let feature = MissingAssetFeature {
        handle,
        requests: vec![ForwardDrawRequest {
            local_id: 9,
            order: 0,
            resources,
            transform: RenderTransform::identity(),
        }],
        prepared: Cell::new(0),
    };
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("context");
    let mut storage = issuer.storage::<()>();
    let payload = storage.publish(&context, ()).expect("payload");
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
        native.finishes.load(Ordering::SeqCst),
    );
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &SwitchPipeline {
                omit_color: Cell::new(false)
            },
            snapshot,
            view,
            bounds(),
            &feature_assets,
            FeatureBinding::new(&issuer, &feature, &payload),
        )),
        Err(RendererError::ForwardPreparation(
            ForwardPreparationError::InvalidTexture { .. }
        ))
    ));
    assert_eq!(feature.prepared.get(), 1);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
}

#[test]
fn feature_request_with_present_invalid_material_schema_is_rejected_before_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("invalid-material").expect("handle");
    let (resources, feature_assets) = assets_with_texture_payload(vec![0; 4]);
    let feature = MissingAssetFeature {
        handle,
        requests: vec![ForwardDrawRequest {
            local_id: 10,
            order: 0,
            resources,
            transform: RenderTransform::identity(),
        }],
        prepared: Cell::new(0),
    };
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("context");
    let mut storage = issuer.storage::<()>();
    let payload = storage.publish(&context, ()).expect("payload");
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
        native.finishes.load(Ordering::SeqCst),
    );
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &SwitchPipeline {
                omit_color: Cell::new(false)
            },
            snapshot,
            view,
            bounds(),
            &feature_assets,
            FeatureBinding::new(&issuer, &feature, &payload),
        )),
        Err(RendererError::ForwardPreparation(
            ForwardPreparationError::MaterialBindingSchema { .. }
        ))
    ));
    assert_eq!(feature.prepared.get(), 1);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
}

#[test]
fn invalid_composition_declarations_reject_before_map_record_or_submit() {
    for mode in [
        InvalidContribution::DuplicateMeshes,
        InvalidContribution::DuplicateFeature,
        InvalidContribution::MissingUpload,
    ] {
        let (device, native) = fixture(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).expect("renderer");
        let (scene, issuer, view) = setup_scene();
        let handle = issuer.issue("empty").expect("handle");
        let feature = EmptyFeature::new(handle);
        let snapshot = scene.snapshot();
        let context = issuer
            .context(&snapshot, CpuPhase::Reference, handle)
            .expect("context");
        let mut storage = issuer.storage::<()>();
        let payload = storage.publish(&context, ()).expect("payload");
        let before = counters(&native);
        assert!(matches!(
            block_on(renderer.submit_forward_with_feature(
                &InvalidContributionPipeline { mode },
                snapshot,
                view,
                bounds(),
                &assets(),
                FeatureBinding::new(&issuer, &feature, &payload),
            )),
            Err(RendererError::ForwardPipeline(
                PipelineError::InvalidPhaseContribution
            ))
        ));
        assert_eq!(counters(&native), before);
        assert_eq!(renderer.pending(), 0);
    }
}
