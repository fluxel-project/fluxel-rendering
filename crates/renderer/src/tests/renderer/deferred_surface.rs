//! Public Deferred Surface admission and completion ownership behavior.

use super::*;
use crate::{DeferredPhaseContext, FramePipeline, PipelineError, PreparedFrame};
use fluxel_rhi::api::{
    ColorSpace, CompositeAlphaMode, PresentMode, Surface, SurfaceConfig, SurfaceImage, SurfaceUsage,
};

#[path = "deferred_surface_failures.rs"]
mod failure_tests;
use std::cell::Cell;

pub(super) fn config(extent: [u32; 2]) -> SurfaceConfig {
    SurfaceConfig {
        extent,
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    }
}

pub(super) fn surface(
    device: &Device<WitnessBackend>,
    extent: [u32; 2],
) -> Surface<WitnessBackend> {
    let mut surface = witness_surface(device);
    block_on(surface.configure(&config(extent))).expect("configured Deferred surface");
    surface
}

fn submit_surface(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
    image: SurfaceImage<WitnessBackend>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit_deferred_surface(
        &crate::DeferredCpuPipeline::new(crate::BoundsVisibility),
        snapshot,
        view,
        bounds(),
        assets,
        image,
    ))
}

#[test]
fn deferred_surface_retires_only_after_present_completion() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let mut surface = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let submission = submit_surface(
        &mut renderer,
        snapshot,
        view,
        &assets,
        block_on(surface.acquire()).expect("image"),
    )
    .expect("Deferred Surface submit");
    let facts = renderer
        .deferred_probe_facts()
        .expect("Deferred Surface facts");
    assert_eq!(renderer.compilations(), 1);
    assert_eq!(renderer.plan_id(), Some(facts.plan));
    assert_eq!(renderer.pending(), 1);
    assert!(
        block_on(renderer.readback(&submission)).is_err(),
        "Surface has no readback ticket"
    );
    assert_eq!(renderer.pending(), 1);
    block_on(renderer.complete(&submission)).expect("present completion");
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
}

#[test]
fn deferred_surface_refuses_foreign_stale_and_wrong_extent_before_device_work() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let (foreign_device, _) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let mut foreign = surface(&foreign_device, [8, 8]);
    let before = counters(&native);
    assert!(matches!(
        submit_surface(
            &mut renderer,
            snapshot.clone(),
            view,
            &assets,
            block_on(foreign.acquire()).expect("foreign image"),
        ),
        Err(RendererError::Rhi(Error::WrongDevice))
    ));
    assert_eq!(counters(&native), before);

    let mut target = surface(&device, [8, 8]);
    let stale = block_on(target.acquire()).expect("old lease");
    block_on(target.configure(&config([8, 8]))).expect("new surface generation");
    let before = counters(&native);
    assert!(matches!(
        submit_surface(&mut renderer, snapshot.clone(), view, &assets, stale,),
        Err(RendererError::Rhi(Error::SurfaceOutdated))
    ));
    assert_eq!(counters(&native), before);

    block_on(target.configure(&config([16, 8]))).expect("mismatched extent");
    let before = counters(&native);
    assert!(matches!(
        submit_surface(
            &mut renderer,
            snapshot,
            view,
            &assets,
            block_on(target.acquire()).expect("mismatched image"),
        ),
        Err(RendererError::Rhi(Error::InvalidRange))
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn deferred_surface_failed_completion_keeps_owner_and_blocks_reentry() {
    let (device, native) = deferred_device(Script::AcceptedCompletionDeviceLost);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let submission = submit_surface(
        &mut renderer,
        snapshot.clone(),
        view,
        &assets,
        block_on(target.acquire()).expect("image"),
    )
    .expect("accepted Surface frame");
    assert!(matches!(
        block_on(renderer.complete(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    let image = block_on(target.acquire()).expect("another lease");
    let before = counters(&native);
    assert!(matches!(
        submit_surface(&mut renderer, snapshot, view, &assets, image),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn deferred_surface_missing_present_frontier_cannot_retire_or_wait() {
    let (device, native) = deferred_device(Script::AcceptedWrongFrontier);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let submission = submit_surface(
        &mut renderer,
        snapshot,
        view,
        &assets,
        block_on(target.acquire()).expect("image"),
    )
    .expect("accepted execution with incomplete evidence");
    assert!(matches!(
        block_on(renderer.complete(&submission)),
        Err(RendererError::MissingCompletion)
    ));
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.waits.load(Ordering::SeqCst), 0);
}

struct ObservedPipeline {
    calls: Cell<usize>,
    omit_color: Cell<bool>,
}

#[test]
fn deferred_surface_rejects_static_copy_bytes_missing_from_the_actual_upload() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = ObservedPipeline {
        calls: Cell::new(0),
        omit_color: Cell::new(false),
    };
    let selected = pipeline
        .prepare(&snapshot, view, bounds())
        .expect("selected");
    let inputs = crate::prepare_forward(
        &selected,
        &assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .expect("resolved input");
    let mut frame = block_on(renderer.prepare_deferred_surface_test_frame(&inputs[0]))
        .expect("actual Surface owners");
    let declaration =
        crate::graph::DeferredPlan::declare_surface_with(&device, &frame, |context| {
            pipeline.contribute_deferred(&selected, context)
        })
        .expect("legal declaration");
    let plan = crate::graph::DeferredPlan::compile_declared(
        &device,
        fluxel_rhi::api::PlanId {
            device: device.id(),
            ordinal: 81_901,
        },
        declaration,
    )
    .expect("compiled Surface");
    plan.validate_surface_frame(&frame)
        .expect("legal before mutation");
    let before = counters(&native);
    frame.resident_upload.bytes.truncate(1);
    assert!(matches!(
        plan.validate_surface_frame(&frame),
        Err(Error::InvalidRange)
    ));
    assert_eq!(counters(&native), before);
}

impl FramePipeline<ForwardResources> for ObservedPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        crate::ForwardCpuPipeline::new(crate::BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<crate::FrameTopology, PipelineError> {
        Ok(crate::FrameTopology::Deferred)
    }

    fn contribute_deferred(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
        context: &mut DeferredPhaseContext<'_, '_>,
    ) -> std::result::Result<(), PipelineError> {
        self.calls.set(self.calls.get() + 1);
        context.geometry(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_gbuffer()
        })?;
        context.lighting(|pass| {
            pass.read_gbuffer()?;
            if self.omit_color.get() {
                Ok(())
            } else {
                pass.write_main_color()
            }
        })
    }
}

#[test]
fn deferred_surface_cache_consumes_the_current_color_declaration() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = ObservedPipeline {
        calls: Cell::new(0),
        omit_color: Cell::new(false),
    };
    for _ in 0..3 {
        let image = block_on(target.acquire()).expect("image");
        let submission = block_on(renderer.submit_deferred_surface(
            &pipeline,
            snapshot.clone(),
            view,
            bounds(),
            &assets,
            image,
        ))
        .expect("declared Surface frame");
        block_on(renderer.complete(&submission)).expect("completed Surface frame");
    }
    assert_eq!(pipeline.calls.get(), 3);
    assert!(
        renderer.deferred_cache_stats().graph_hits > 0,
        "settled Surface frame must actually reuse a plan"
    );
    pipeline.omit_color.set(true);
    let image = block_on(target.acquire()).expect("image for refusal");
    let before = counters(&native);
    let result = block_on(renderer.submit_deferred_surface(
        &pipeline,
        snapshot,
        view,
        bounds(),
        &assets,
        image,
    ));
    assert!(
        result.is_err(),
        "omitting the current color declaration must refuse"
    );
    assert_eq!(pipeline.calls.get(), 4);
    assert_eq!(native.map_calls.load(Ordering::SeqCst), before.1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before.2);
    assert_eq!(renderer.pending(), 0);
}
