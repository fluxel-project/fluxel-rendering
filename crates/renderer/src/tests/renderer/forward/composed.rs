//! Typed composed Forward admission and cached-declaration refusal coverage.

use super::*;
use crate::{
    BoundsVisibility, CpuPhase, FeatureBinding, FeatureError, ForwardAssets, ForwardCpuPipeline,
    FramePipeline, FrameTopology, PipelineError, RenderBounds, RenderChange, RenderChangeSet,
    RenderFeature, RenderScene, RenderTransform, RenderView,
};
use std::{cell::Cell, result::Result, sync::atomic::Ordering};

struct EmptyFeature {
    handle: crate::FeatureHandle,
    prepared: Cell<usize>,
    contributed: Cell<usize>,
    omit_texture: Cell<bool>,
}

impl EmptyFeature {
    fn new(handle: crate::FeatureHandle) -> Self {
        Self {
            handle,
            prepared: Cell::new(0),
            contributed: Cell::new(0),
            omit_texture: Cell::new(false),
        }
    }
}

impl RenderFeature for EmptyFeature {
    type Input = ();
    type Prepared = ();

    fn handle(&self) -> crate::FeatureHandle {
        self.handle
    }

    fn prepare(
        &self,
        _: &crate::FeatureContext<'_, crate::ForwardResources>,
        _: &crate::TypedFeatureSnapshot<Self::Input>,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
        _: RenderBounds,
    ) -> Result<Self::Prepared, FeatureError> {
        self.prepared.set(self.prepared.get() + 1);
        Ok(())
    }

    fn draws<'a>(&self, _: &'a Self::Prepared) -> &'a [crate::ForwardDrawRequest] {
        &[]
    }

    fn contribute(
        &self,
        _: &Self::Prepared,
        pass: &mut crate::ForwardFeaturePass<'_, '_>,
    ) -> Result<(), PipelineError> {
        self.contributed.set(self.contributed.get() + 1);
        pass.read_vertex_streams()?;
        pass.read_frame_uniforms()?;
        if self.omit_texture.get() {
            Ok(())
        } else {
            pass.read_sampled_textures()
        }
    }
}

struct SwitchPipeline {
    omit_color: Cell<bool>,
}

struct ClearOnlyPipeline;

impl FramePipeline<crate::ForwardResources> for ClearOnlyPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> Result<crate::PreparedFrame<'a, crate::ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
    ) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::ClearOnly)
    }
}

struct FailingPrepareFeature {
    handle: crate::FeatureHandle,
    prepared: Cell<usize>,
}

impl RenderFeature for FailingPrepareFeature {
    type Input = ();
    type Prepared = ();

    fn handle(&self) -> crate::FeatureHandle {
        self.handle
    }

    fn prepare(
        &self,
        _: &crate::FeatureContext<'_, crate::ForwardResources>,
        _: &crate::TypedFeatureSnapshot<Self::Input>,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
        _: RenderBounds,
    ) -> Result<Self::Prepared, FeatureError> {
        self.prepared.set(self.prepared.get() + 1);
        Err(FeatureError::InvalidPreparedData {
            feature: self.handle.name(),
            detail: "scripted prepare refusal",
        })
    }

    fn draws<'a>(&self, _: &'a Self::Prepared) -> &'a [crate::ForwardDrawRequest] {
        &[]
    }
}

impl FramePipeline<crate::ForwardResources> for SwitchPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> Result<crate::PreparedFrame<'a, crate::ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
    ) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::OpaqueForward)
    }

    fn contribute_composed<B: fluxel_rhi::backend::ResourceBackend, F: RenderFeature>(
        &self,
        _: &crate::PreparedFrame<'_, crate::ForwardResources>,
        context: &mut crate::ForwardCompositionContext<'_, '_, '_, B, F>,
    ) -> Result<(), PipelineError> {
        context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.meshes(|mesh| {
                mesh.read_vertex_streams()?;
                mesh.read_frame_uniforms()?;
                mesh.read_sampled_textures()
            })?;
            pass.feature()?;
            if !self.omit_color.get() {
                pass.write_main_color()?;
            }
            Ok(())
        })
    }
}

fn setup_scene() -> (
    RenderScene<crate::ForwardResources>,
    crate::FeatureIssuer,
    crate::RenderViewId,
) {
    let mut scene = RenderScene::new().expect("scene");
    let issuer = scene.feature_issuer().expect("issuer");
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).expect("view"),
    ));
    let view = scene.apply(&changes).expect("insert view").inserted_views()[0];
    (scene, issuer, view)
}

fn assets() -> ForwardAssets {
    ForwardAssets {
        geometry: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
    }
}

fn bounds() -> RenderBounds {
    RenderBounds::new([-1.0; 3], [1.0; 3]).expect("bounds")
}

fn assert_no_backend_work(
    native: &Native,
    buffers: usize,
    maps: usize,
    commands: usize,
    finishes: usize,
) {
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), buffers);
    assert_eq!(native.map_calls.load(Ordering::SeqCst), maps);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), finishes);
}

#[test]
fn stale_typed_feature_input_is_rejected_before_prepare_or_backend_work() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (mut scene, issuer, view) = setup_scene();
    let handle = issuer.issue("empty").expect("handle");
    let feature = EmptyFeature::new(handle);
    let mut storage = issuer.storage::<()>();
    let first = scene.snapshot();
    let context = issuer
        .context(&first, CpuPhase::Reference, handle)
        .expect("first context");
    let typed = storage.publish(&context, ()).expect("typed input");
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).expect("second view"),
    ));
    scene.apply(&changes).expect("advance scene");
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
            scene.snapshot(),
            view,
            bounds(),
            &assets(),
            FeatureBinding::new(&issuer, &feature, &typed),
        )),
        Err(RendererError::ForwardFeature(
            FeatureError::SnapshotVersionMismatch { .. }
        ))
    ));
    assert_eq!(feature.prepared.get(), 0);
    assert_eq!(feature.contributed.get(), 0);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn foreign_typed_feature_input_is_rejected_before_prepare_or_backend_work() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("empty").expect("handle");
    let feature = EmptyFeature::new(handle);
    let (foreign_scene, foreign_issuer, _) = setup_scene();
    let foreign_handle = foreign_issuer.issue("foreign").expect("foreign handle");
    let mut foreign_storage = foreign_issuer.storage::<()>();
    let foreign_snapshot = foreign_scene.snapshot();
    let foreign_context = foreign_issuer
        .context(&foreign_snapshot, CpuPhase::Reference, foreign_handle)
        .expect("foreign context");
    let foreign_typed = foreign_storage
        .publish(&foreign_context, ())
        .expect("foreign typed");
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
            scene.snapshot(),
            view,
            bounds(),
            &assets(),
            FeatureBinding::new(&issuer, &feature, &foreign_typed),
        )),
        Err(RendererError::ForwardFeature(FeatureError::ForeignScene))
    ));
    assert_eq!(feature.prepared.get(), 0);
    assert_eq!(feature.contributed.get(), 0);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn cached_empty_composition_invokes_current_feature_and_rejects_missing_color_before_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("empty").expect("handle");
    let feature = EmptyFeature::new(handle);
    let mut storage = issuer.storage::<()>();
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("context");
    let typed = storage.publish(&context, ()).expect("typed input");
    let pipeline = SwitchPipeline {
        omit_color: Cell::new(false),
    };
    let asset_set = assets();
    for expected in 1..=2 {
        block_on(renderer.submit_forward_with_feature(
            &pipeline,
            snapshot.clone(),
            view,
            bounds(),
            &asset_set,
            FeatureBinding::new(&issuer, &feature, &typed),
        ))
        .expect("empty composition submits");
        assert_eq!(feature.prepared.get(), expected);
        assert_eq!(feature.contributed.get(), expected);
    }
    assert_eq!(renderer.compilations(), 1);
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    pipeline.omit_color.set(true);
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &pipeline,
            snapshot,
            view,
            bounds(),
            &asset_set,
            FeatureBinding::new(&issuer, &feature, &typed),
        )),
        Err(RendererError::ForwardPipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(feature.prepared.get(), 3);
    assert_eq!(feature.contributed.get(), 3);
    assert_eq!(renderer.compilations(), 1);
    assert_eq!(native.map_calls.load(Ordering::SeqCst), maps);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), finishes);
    // Refusing this frame must not retire either previously accepted frame.
    assert_eq!(renderer.pending(), 2);
}

#[test]
fn cached_feature_texture_omission_is_rejected_before_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("empty").expect("handle");
    let feature = EmptyFeature::new(handle);
    let mut storage = issuer.storage::<()>();
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("context");
    let typed = storage.publish(&context, ()).expect("typed input");
    let pipeline = SwitchPipeline {
        omit_color: Cell::new(false),
    };
    let asset_set = assets();
    block_on(renderer.submit_forward_with_feature(
        &pipeline,
        snapshot.clone(),
        view,
        bounds(),
        &asset_set,
        FeatureBinding::new(&issuer, &feature, &typed),
    ))
    .expect("first empty composition submits");
    assert_eq!(renderer.compilations(), 1);
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    feature.omit_texture.set(true);
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &pipeline,
            snapshot,
            view,
            bounds(),
            &asset_set,
            FeatureBinding::new(&issuer, &feature, &typed),
        )),
        Err(RendererError::ForwardPipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(feature.prepared.get(), 2);
    assert_eq!(feature.contributed.get(), 2);
    assert_eq!(renderer.compilations(), 1);
    assert_eq!(native.map_calls.load(Ordering::SeqCst), maps);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), finishes);
    // The prior accepted frame still owns its resources until completion.
    assert_eq!(renderer.pending(), 1);
}

#[test]
fn clear_only_composed_topology_is_rejected_before_feature_prepare_or_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("empty").expect("handle");
    let feature = EmptyFeature::new(handle);
    let mut storage = issuer.storage::<()>();
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("context");
    let typed = storage.publish(&context, ()).expect("typed input");
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
        native.finishes.load(Ordering::SeqCst),
    );
    assert!(matches!(
        block_on(renderer.submit_forward_with_feature(
            &ClearOnlyPipeline,
            snapshot,
            view,
            bounds(),
            &assets(),
            FeatureBinding::new(&issuer, &feature, &typed),
        )),
        Err(RendererError::ForwardPipeline(
            PipelineError::FeatureRequiresOpaque
        ))
    ));
    assert_eq!(feature.prepared.get(), 0);
    assert_eq!(feature.contributed.get(), 0);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
}

#[test]
fn feature_prepare_failure_is_rejected_before_allocation_or_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (scene, issuer, view) = setup_scene();
    let handle = issuer.issue("failing").expect("handle");
    let feature = FailingPrepareFeature {
        handle,
        prepared: Cell::new(0),
    };
    let mut storage = issuer.storage::<()>();
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, handle)
        .expect("context");
    let typed = storage.publish(&context, ()).expect("typed input");
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
            FeatureBinding::new(&issuer, &feature, &typed),
        )),
        Err(RendererError::ForwardFeature(
            FeatureError::InvalidPreparedData { .. }
        ))
    ));
    assert_eq!(feature.prepared.get(), 1);
    assert_no_backend_work(&native, before.0, before.1, before.2, before.3);
}

#[path = "composed_admission.rs"]
mod admission;
