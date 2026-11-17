//! Public Deferred pipeline callback tests over the real private Graph.

use super::*;

use std::cell::Cell;

use crate::{
    BoundsVisibility, DeferredPhaseContext, ForwardCpuPipeline, FramePipeline, FrameTopology,
    PipelineError, PreparedFrame,
};

#[test]
fn deferred_multi_record_refuses_an_incompatible_lighting_family_before_allocation() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (_, _, assets) = scene_and_assets();
    let mut scene = RenderScene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    for _ in 0..2 {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
                material: SceneAssetRef::from_snapshot(&assets.materials[0]),
                textures: vec![crate::ForwardTextureBinding::asset(
                    "material_texture",
                    SceneAssetRef::from_snapshot(&assets.textures[0]),
                )],
                tint: [1.; 4],
            },
            transform: RenderTransform::identity(),
            bounds: bounds(),
        }));
    }
    let receipt = scene.apply(&changes).unwrap();
    let before = (
        native.created_buffers.load(Ordering::SeqCst),
        native.created_textures.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    );
    let result = block_on(renderer.submit_deferred(
        &ExternalDeferredPipeline::new(),
        scene.snapshot(),
        receipt.inserted_views()[0],
        bounds(),
        &assets,
    ));
    assert!(
        matches!(
            result,
            Err(RendererError::Rhi(Error::Unsupported(
                "multiple Deferred draws require the UV0-normal MR family"
            )))
        ),
        "family admission must be the actual refusal: {result:?}"
    );
    assert_eq!(
        before,
        (
            native.created_buffers.load(Ordering::SeqCst),
            native.created_textures.load(Ordering::SeqCst),
            native.map_calls.load(Ordering::SeqCst),
            native.command_calls.load(Ordering::SeqCst)
        )
    );
}

struct ExternalDeferredPipeline {
    callbacks: Cell<usize>,
}

impl ExternalDeferredPipeline {
    fn new() -> Self {
        Self {
            callbacks: Cell::new(0),
        }
    }
}

impl FramePipeline<ForwardResources> for ExternalDeferredPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::Deferred)
    }

    fn contribute_deferred(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
        context: &mut DeferredPhaseContext<'_, '_>,
    ) -> std::result::Result<(), PipelineError> {
        self.callbacks.set(self.callbacks.get() + 1);
        context.geometry(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_gbuffer()
        })?;
        context.lighting(|pass| {
            pass.read_gbuffer()?;
            pass.write_main_color()
        })
    }
}

struct ForwardTopologyPipeline;

impl FramePipeline<ForwardResources> for ForwardTopologyPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }
}

struct MissingDeferredCallback;

impl FramePipeline<ForwardResources> for MissingDeferredCallback {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::Deferred)
    }
}

#[derive(Clone, Copy)]
enum InvalidDeclaration {
    GeometryMissingTextureRead,
    LightingMissingGbufferRead,
    DuplicateGeometry,
}

struct InvalidDeferredPipeline(InvalidDeclaration);

impl FramePipeline<ForwardResources> for InvalidDeferredPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::Deferred)
    }

    fn contribute_deferred(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
        context: &mut DeferredPhaseContext<'_, '_>,
    ) -> std::result::Result<(), PipelineError> {
        context.geometry(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            if !matches!(self.0, InvalidDeclaration::GeometryMissingTextureRead) {
                pass.read_sampled_textures()?;
            }
            pass.write_gbuffer()
        })?;
        if matches!(self.0, InvalidDeclaration::DuplicateGeometry) {
            context.geometry(|_| Ok(()))?;
        }
        context.lighting(|pass| {
            if !matches!(self.0, InvalidDeclaration::LightingMissingGbufferRead) {
                pass.read_gbuffer()?;
            }
            pass.write_main_color()
        })
    }
}

fn submit_with_pipeline<P: FramePipeline<ForwardResources>>(
    renderer: &mut Renderer<WitnessBackend>,
    pipeline: &P,
    snapshot: crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit_deferred(pipeline, snapshot, view, bounds(), assets))
}

fn device_counts(native: &Native) -> (usize, usize, usize) {
    (
        native.created_buffers.load(Ordering::SeqCst),
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    )
}

#[test]
fn external_deferred_pipeline_declares_both_real_passes_and_retires() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = ExternalDeferredPipeline::new();

    let submission = submit_with_pipeline(&mut renderer, &pipeline, snapshot, view, &assets)
        .expect("Deferred submission");
    assert_eq!(pipeline.callbacks.get(), 1);
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 5);
    assert_eq!(renderer.deferred_probe_facts().expect("facts").passes, 5);
    block_on(renderer.readback(&submission)).expect("readback");
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn deferred_entry_refuses_forward_topology_before_resource_allocation() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let before = device_counts(&native);

    assert!(matches!(
        submit_with_pipeline(
            &mut renderer,
            &ForwardTopologyPipeline,
            snapshot,
            view,
            &assets
        ),
        Err(RendererError::DeferredPipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(device_counts(&native), before);
}

#[test]
fn deferred_entry_refuses_missing_or_invalid_phase_callbacks_before_mapping_or_recording() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let before = device_counts(&native);
    assert!(matches!(
        submit_with_pipeline(
            &mut renderer,
            &MissingDeferredCallback,
            snapshot,
            view,
            &assets
        ),
        Err(RendererError::DeferredPipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(native.map_calls.load(Ordering::SeqCst), before.1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before.2);

    for invalid in [
        InvalidDeclaration::GeometryMissingTextureRead,
        InvalidDeclaration::LightingMissingGbufferRead,
        InvalidDeclaration::DuplicateGeometry,
    ] {
        let (snapshot, view, assets) = scene_and_assets();
        let before = device_counts(&native);
        assert!(
            submit_with_pipeline(
                &mut renderer,
                &InvalidDeferredPipeline(invalid),
                snapshot,
                view,
                &assets,
            )
            .is_err()
        );
        assert_eq!(native.map_calls.load(Ordering::SeqCst), before.1);
        assert_eq!(native.command_calls.load(Ordering::SeqCst), before.2);
    }
}
