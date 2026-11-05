//! Public Deferred pipeline callback tests over the real private Graph.

use super::*;

use std::cell::Cell;

use crate::{
    BoundsVisibility, DeferredPhaseContext, ForwardCpuPipeline, FramePipeline, FrameTopology,
    PipelineError, PreparedFrame,
};

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
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 4);
    assert_eq!(renderer.deferred_probe_facts().expect("facts").passes, 4);
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
