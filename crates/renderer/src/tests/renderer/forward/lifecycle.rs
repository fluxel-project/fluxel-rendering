use super::*;
use crate::{FrameTopology, PipelineError};
use crate::{
    renderer::forward::ForwardPublication,
    residency::{ForwardFrame, ForwardFrameBuffers, ForwardResidentBuffers, SurfaceForwardFrame},
};

fn blank_frame(device: &Device<WitnessBackend>) -> ForwardFrame<WitnessBackend> {
    let (resident, resident_upload) =
        ForwardResidentBuffers::allocate(device, Vec::new(), Vec::new()).unwrap();
    let uniform_bytes = vec![0; 16];
    let (frame, uniform_upload) =
        ForwardFrameBuffers::allocate(device, [1, 1], 16, uniform_bytes.clone(), 0..16, 256)
            .unwrap();
    ForwardFrame {
        resident,
        resident_upload,
        frame,
        uniform_upload,
        uniform_bytes,
        draws: Vec::new(),
        draw_batches: Vec::new(),
        extent: [1, 1],
        readback_row_pitch: 256,
    }
}

fn execute(
    renderer: &mut Renderer<WitnessBackend>,
    retained: Arc<()>,
    frame: ForwardFrame<WitnessBackend>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.execute_forward(retained, frame, ForwardPublication::default()))
}

fn execute_surface(
    renderer: &mut Renderer<WitnessBackend>,
    retained: Arc<()>,
    frame: SurfaceForwardFrame<WitnessBackend>,
    topology: FrameTopology,
    image: SurfaceImage<WitnessBackend>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.execute_forward_surface(
        retained,
        frame,
        ForwardPublication::default(),
        topology,
        crate::preparation::ForwardBatchStats {
            candidates: 0,
            batches: 0,
        },
        image,
        |context| match topology {
            FrameTopology::OpaqueForward => context.opaque_forward(),
            FrameTopology::ClearOnly => context.clear_only(),
        },
    ))
}

#[test]
fn surface_forward_requires_present_frontier_before_retiring_owner() {
    let (device, native) = fixture(Script::AcceptedWrongFrontier);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    let image = block_on(surface.acquire()).unwrap();
    let submission = execute_surface(
        &mut renderer,
        retained,
        blank_surface_frame(&device),
        FrameTopology::OpaqueForward,
        image,
    )
    .unwrap();

    assert!(matches!(
        block_on(renderer.complete(&submission)),
        Err(RendererError::MissingCompletion)
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.waits.load(Ordering::SeqCst), 0);
}

#[test]
fn surface_forward_cache_keeps_topologies_separate() {
    let (device, _) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let initial = renderer.compilations();

    for topology in [FrameTopology::OpaqueForward, FrameTopology::ClearOnly] {
        let image = block_on(surface.acquire()).unwrap();
        let submission = execute_surface(
            &mut renderer,
            Arc::new(()),
            blank_surface_frame(&device),
            topology,
            image,
        )
        .unwrap();
        block_on(renderer.complete(&submission)).unwrap();
    }
    assert_eq!(renderer.compilations(), initial + 2);
}

#[test]
fn cached_surface_plan_rejects_an_illegal_current_contribution_before_mapping_or_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let stats = crate::preparation::ForwardBatchStats {
        candidates: 0,
        batches: 0,
    };
    let calls = std::cell::Cell::new(0);
    let first = block_on(renderer.execute_forward_surface(
        Arc::new(()),
        blank_surface_frame(&device),
        ForwardPublication::default(),
        FrameTopology::OpaqueForward,
        stats,
        block_on(surface.acquire()).unwrap(),
        |context| {
            calls.set(calls.get() + 1);
            context.opaque_forward()
        },
    ))
    .unwrap();
    block_on(renderer.complete(&first)).unwrap();
    assert_eq!(calls.get(), 1);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    let maps = native.map_calls.load(Ordering::SeqCst);

    assert!(matches!(
        block_on(renderer.execute_forward_surface(
            Arc::new(()),
            blank_surface_frame(&device),
            ForwardPublication::default(),
            FrameTopology::OpaqueForward,
            stats,
            block_on(surface.acquire()).unwrap(),
            |context| context.clear(|pass| {
                pass.depends_on_uploads()?;
                pass.write_main_color()
            }),
        )),
        Err(RendererError::ForwardPipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(native.map_calls.load(Ordering::SeqCst), maps);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), finishes);
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn stale_surface_forward_image_is_rejected_before_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let image = block_on(surface.acquire()).unwrap();
    let config = SurfaceConfig {
        extent: [1, 1],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    block_on(surface.configure(&config)).unwrap();

    assert!(matches!(
        execute_surface(
            &mut renderer,
            Arc::new(()),
            blank_surface_frame(&device),
            FrameTopology::OpaqueForward,
            image,
        ),
        Err(RendererError::Rhi(Error::SurfaceOutdated))
    ));
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.finishes.load(Ordering::SeqCst), 0);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn surface_forward_rejects_a_lease_with_a_different_extent_during_raster_recording() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let config = SurfaceConfig {
        extent: [2, 1],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    block_on(surface.configure(&config)).unwrap();
    let image = block_on(surface.acquire()).unwrap();

    assert!(matches!(
        execute_surface(
            &mut renderer,
            Arc::new(()),
            blank_surface_frame(&device),
            FrameTopology::OpaqueForward,
            image,
        ),
        Err(RendererError::RetainedExecution {
            recording_error: Some(Error::InvalidRange),
            ..
        })
    ));
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 2);
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 1);
}

#[test]
fn surface_forward_rejects_a_lease_with_a_different_pipeline_format_during_raster_recording() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let config = SurfaceConfig {
        extent: [1, 1],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    block_on(surface.configure(&config)).unwrap();
    let image = block_on(surface.acquire()).unwrap();

    assert!(matches!(
        execute_surface(
            &mut renderer,
            Arc::new(()),
            blank_surface_frame(&device),
            FrameTopology::OpaqueForward,
            image,
        ),
        Err(RendererError::RetainedExecution {
            recording_error: Some(Error::InvalidRange),
            ..
        })
    ));
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 2);
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 1);
}

#[test]
fn surface_forward_plan_rejects_a_different_access_shape() {
    let (device, _) = fixture(Script::AcceptedExact);
    let frame = blank_surface_frame(&device);
    let plan = match crate::graph::SurfaceForwardPlan::compile_with(
        &device,
        PlanId {
            device: device.id(),
            ordinal: 9_001,
        },
        &frame,
        FrameTopology::OpaqueForward,
        |context| context.opaque_forward(),
    ) {
        Ok(plan) => plan,
        Err(_) => panic!("blank surface plan must compile"),
    };
    let mut different_extent = blank_surface_frame(&device);
    different_extent.extent = [2, 1];
    assert!(matches!(
        plan.validate_frame(&different_extent),
        Err(Error::InvalidState)
    ));
    let mut different_format = blank_surface_frame(&device);
    different_format.color_format = TextureFormat::Bgra8Unorm;
    assert!(matches!(
        plan.validate_frame(&different_format),
        Err(Error::InvalidState)
    ));
}

#[test]
fn forward_executor_retires_blank_frame_after_confirmed_completion() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    let submission = execute(&mut renderer, retained, blank_frame(&device)).unwrap();

    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    // This witness intentionally has no pixel fixture for a 1x1 padded image.
    // Successful completion is still sufficient retirement evidence.
    assert!(block_on(renderer.readback(&submission)).is_err());
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
}

#[test]
fn forward_partial_or_indeterminate_execution_retains_real_frame_and_snapshot() {
    for script in [Script::PartialAccepted, Script::Indeterminate] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let retained = Arc::new(());
        let weak = Arc::downgrade(&retained);

        assert!(matches!(
            execute(&mut renderer, retained, blank_frame(&device)),
            Err(RendererError::RetainedExecution { .. })
        ));
        assert_eq!(renderer.pending(), 1);
        assert!(weak.upgrade().is_some());
        assert_eq!(native.finishes.load(Ordering::SeqCst), 1);

        let report = block_on(renderer.shutdown()).unwrap();
        assert_eq!(report.retired_submissions, 1);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn forward_completion_loss_keeps_owner_and_quarantines_renderer() {
    let (device, native) = fixture(Script::AcceptedCompletionDeviceLost);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    let submission = execute(&mut renderer, retained, blank_frame(&device)).unwrap();

    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::ForwardOperation {
            error: Error::DeviceLost,
            ..
        }) | Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    let commands = native.command_calls.load(Ordering::SeqCst);
    assert!(matches!(
        execute(&mut renderer, Arc::new(()), blank_frame(&device)),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
}

#[test]
fn forward_readback_loss_after_completion_releases_owner_but_quarantines_renderer() {
    let (device, native) = fixture(Script::AcceptedReadbackDeviceLost);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    let submission = execute(&mut renderer, retained, blank_frame(&device)).unwrap();

    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::ForwardOperation {
            error: Error::DeviceLost,
            ..
        }) | Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    // Completion was observed, so the submitted owner may release. The map
    // failure must not put its allocation into a reusable frame slot.
    assert!(renderer.forward_frames.iter().all(Option::is_none));
    assert!(matches!(
        execute(&mut renderer, Arc::new(()), blank_frame(&device)),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
}

#[test]
fn forward_recording_failure_retains_owner_until_successful_idle() {
    let (device, native) = fixture(Script::RecordingFailureAfterPrefix);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);

    assert!(matches!(
        execute(&mut renderer, retained, blank_frame(&device)),
        Err(RendererError::RetainedExecution {
            status: ExecutionStatus::PartiallyAccepted,
            ..
        })
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 2);
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert!(renderer.forward_frames.iter().all(Option::is_none));

    let report = block_on(renderer.shutdown()).unwrap();
    assert_eq!(report.retired_submissions, 1);
    assert!(weak.upgrade().is_none());
}

#[test]
fn forward_finish_loss_quarantines_and_preserves_uncertain_owner() {
    let (device, native) = fixture(Script::FinishDeviceLost);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);

    assert!(matches!(
        execute(&mut renderer, retained, blank_frame(&device)),
        Err(RendererError::RetainedExecution {
            execution_error: Some(Error::DeviceLost),
            ..
        })
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert!(renderer.forward_frames.iter().all(Option::is_none));
    let commands = native.command_calls.load(Ordering::SeqCst);

    assert!(matches!(
        execute(&mut renderer, Arc::new(()), blank_frame(&device)),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert!(matches!(
        block_on(renderer.shutdown()),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
}

#[test]
fn forward_clean_rejection_releases_real_owners_without_pending_slot() {
    let (device, native) = fixture(Script::CleanRejected);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let created_before = native.created_buffers.load(Ordering::SeqCst);

    assert!(matches!(
        execute(&mut renderer, Arc::new(()), blank_frame(&device)),
        Err(RendererError::ForwardOperation { .. })
    ));
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(
        native.created_buffers.load(Ordering::SeqCst) - created_before,
        native.dropped_buffers.load(Ordering::SeqCst)
    );
}

#[test]
fn forward_malformed_uniform_upload_refuses_before_execution_begins() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut frame = blank_frame(&device);
    // The owner allocation is 16 bytes, while this declared copy crosses it.
    frame.uniform_upload.destination = 1..17;
    let commands = native.command_calls.load(Ordering::SeqCst);

    assert!(matches!(
        execute(&mut renderer, Arc::new(()), frame),
        Err(RendererError::ForwardOperation { .. })
    ));
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), 0);
    assert_eq!(renderer.pending(), 0);
}
