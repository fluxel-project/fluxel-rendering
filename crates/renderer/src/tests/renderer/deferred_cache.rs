//! Completion-gated Deferred cache behavior over the witness backend.

use super::*;

use std::{cell::Cell, sync::Arc};

use crate::{
    BoundsVisibility, DeferredPhaseContext, ForwardCpuPipeline, ForwardDrawInput, FramePipeline,
    FrameTopology, PipelineError, PreparedFrame, prepare_forward,
};
use fluxel_rhi::api::{Device, Error, PlanId};

struct CountingDeferredPipeline {
    callbacks: Cell<usize>,
    declare: bool,
}

impl CountingDeferredPipeline {
    fn complete() -> Self {
        Self {
            callbacks: Cell::new(0),
            declare: true,
        }
    }

    fn missing() -> Self {
        Self {
            callbacks: Cell::new(0),
            declare: false,
        }
    }
}

impl FramePipeline<ForwardResources> for CountingDeferredPipeline {
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
        if !self.declare {
            return Ok(());
        }
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

fn submit_with_counting_pipeline(
    renderer: &mut Renderer<WitnessBackend>,
    pipeline: &CountingDeferredPipeline,
    snapshot: crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit_deferred(pipeline, snapshot, view, bounds(), assets))
}

fn prepared_deferred_frame(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: &crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) -> crate::residency::DeferredFrame<WitnessBackend> {
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let selected = pipeline
        .prepare(snapshot, view, bounds())
        .expect("selected frame");
    let inputs: Vec<ForwardDrawInput> =
        prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new())
            .expect("prepared input");
    let [input] = inputs.as_slice() else {
        panic!("fixture has one draw");
    };
    block_on(renderer.prepare_deferred_test_frame(input)).expect("prepared Deferred frame")
}

fn compile(
    device: &Device<WitnessBackend>,
    ordinal: u64,
    frame: &crate::residency::DeferredFrame<WitnessBackend>,
) -> std::result::Result<crate::graph::DeferredPlan<WitnessBackend>, Error> {
    crate::graph::DeferredPlan::compile(
        device,
        PlanId {
            device: device.id(),
            ordinal,
        },
        frame,
    )
}

#[test]
fn deferred_cache_reuses_completed_products_but_declares_each_frame() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = CountingDeferredPipeline::complete();

    let cold =
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot.clone(), view, &assets)
            .expect("cold submission");
    block_on(renderer.readback(&cold)).expect("cold readback");
    assert_eq!(pipeline.callbacks.get(), 1);
    assert_eq!(renderer.deferred_cache_stats().geometry_hits, 0);
    assert_eq!(renderer.deferred_cache_stats().texture_hits, 0);

    let warm =
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot.clone(), view, &assets)
            .expect("warm submission");
    let warm_stats = renderer.deferred_cache_stats();
    assert_eq!(pipeline.callbacks.get(), 2);
    assert!(warm_stats.geometry_hits >= 1);
    assert!(warm_stats.texture_hits >= 1);
    assert!(warm_stats.gbuffer_program_hits >= 1);
    assert!(warm_stats.lighting_program_hits >= 1);
    block_on(renderer.readback(&warm)).expect("warm readback");

    let repeat = submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot, view, &assets)
        .expect("repeat submission");
    assert_eq!(pipeline.callbacks.get(), 3);
    assert!(renderer.deferred_cache_stats().graph_hits >= 1);
    block_on(renderer.readback(&repeat)).expect("repeat readback");
}

#[test]
fn deferred_pending_work_does_not_publish_static_products_early() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = CountingDeferredPipeline::complete();

    let first =
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot.clone(), view, &assets)
            .expect("first submission");
    let second =
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot.clone(), view, &assets)
            .expect("second submission");
    let before_completion = renderer.deferred_cache_stats();
    assert_eq!(before_completion.geometry_hits, 0);
    assert_eq!(before_completion.texture_hits, 0);
    assert_eq!(renderer.pending(), 2);

    block_on(renderer.readback(&first)).expect("first readback");
    let after_completion =
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot, view, &assets)
            .expect("completed-product submission");
    let stats = renderer.deferred_cache_stats();
    assert!(stats.geometry_hits >= 1);
    assert!(stats.texture_hits >= 1);
    block_on(renderer.readback(&second)).expect("second readback");
    block_on(renderer.readback(&after_completion)).expect("third readback");
}

#[test]
fn deferred_cache_does_not_replay_a_missing_declaration() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let complete = CountingDeferredPipeline::complete();
    let warmup =
        submit_with_counting_pipeline(&mut renderer, &complete, snapshot.clone(), view, &assets)
            .expect("warmup submission");
    block_on(renderer.readback(&warmup)).expect("warmup readback");

    let missing = CountingDeferredPipeline::missing();
    let before = counters(&native);
    let result = submit_with_counting_pipeline(&mut renderer, &missing, snapshot, view, &assets);
    assert!(
        matches!(result, Err(RendererError::Rhi(Error::InvalidState))),
        "cached declaration must be refused before mapping or recording: {result:?}"
    );
    assert_eq!(missing.callbacks.get(), 1);
    assert_eq!(native.map_calls.load(Ordering::SeqCst), before.1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before.2);
}

#[test]
fn deferred_upload_validation_accepts_actual_completed_partial_hits_only() {
    let (device, _) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = CountingDeferredPipeline::complete();

    let cold =
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot.clone(), view, &assets)
            .expect("cold submission");
    block_on(renderer.readback(&cold)).expect("cold readback");
    let completed = prepared_deferred_frame(&mut renderer, &snapshot, view, &assets);
    assert!(completed.resident_upload.bytes.is_empty());

    renderer.deferred_residency.clear();
    let mut cold_without_upload = prepared_deferred_frame(&mut renderer, &snapshot, view, &assets);
    cold_without_upload.resident_upload.vertices.clear();
    cold_without_upload.resident_upload.textures.clear();
    cold_without_upload.resident_upload.bytes.clear();
    assert!(matches!(
        compile(&device, 81_001, &cold_without_upload),
        Err(Error::InvalidRange)
    ));

    let mut texture_only = prepared_deferred_frame(&mut renderer, &snapshot, view, &assets);
    Arc::get_mut(&mut texture_only.resident)
        .expect("fresh frame has one resident owner")
        .streams = completed.resident.streams.clone();
    texture_only.resident_upload.vertices.clear();
    assert!(compile(&device, 81_002, &texture_only).is_ok());

    let mut geometry_only = prepared_deferred_frame(&mut renderer, &snapshot, view, &assets);
    Arc::get_mut(&mut geometry_only.resident)
        .expect("fresh frame has one resident owner")
        .textures = completed.resident.textures.clone();
    geometry_only.resident_upload.textures.clear();
    assert!(compile(&device, 81_003, &geometry_only).is_ok());
}

#[path = "deferred_cache_boundaries.rs"]
mod boundary_tests;
