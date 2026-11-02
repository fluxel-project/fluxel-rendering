use super::*;

fn triangle() -> AssetSnapshot<Bytes, Vec<u8>> {
    snapshot_bytes(vec![0; 3 * 24]).0
}

fn pipeline(device: &Device<WitnessBackend>) -> RasterPipeline<WitnessBackend> {
    // SAFETY: this no-op test pipeline belongs to the supplied witness device.
    unsafe { RasterPipeline::from_native(device.id(), ()) }
}

fn surface_image(
    device: &Device<WitnessBackend>,
) -> (Surface<WitnessBackend>, SurfaceImage<WitnessBackend>) {
    let mut surface = unsafe {
        Surface::from_native(
            device,
            SurfaceId::allocate().expect("surface identity"),
            WitnessSurface {
                generation: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            },
        )
    };
    let image = block_on(surface.acquire()).expect("witness surface image");
    (surface, image)
}

fn submit_surface(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: AssetSnapshot<Bytes, Vec<u8>>,
    pipeline: RasterPipeline<WitnessBackend>,
    image: SurfaceImage<WitnessBackend>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit_surface_raster(snapshot, pipeline, image, [4, 4]))
}

#[test]
fn accepted_surface_frame_retires_only_after_its_completion() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (_, image) = surface_image(&device);
    let submission = submit_surface(&mut renderer, triangle(), pipeline(&device), image).unwrap();

    assert_eq!(renderer.pending(), 1);
    block_on(renderer.complete(&submission)).unwrap();
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    assert_eq!(*native.waited_completions.lock().unwrap(), [1]);
}

#[test]
fn stale_surface_lease_is_rejected_before_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (mut surface, image) = surface_image(&device);
    let config = SurfaceConfig {
        extent: [8, 8],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    block_on(surface.configure(&config)).unwrap();

    assert!(matches!(
        submit_surface(&mut renderer, triangle(), pipeline(&device), image),
        Err(RendererError::Rhi(Error::SurfaceOutdated))
    ));
    assert_eq!(native.finishes.load(Ordering::SeqCst), 0);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn surface_present_rejection_has_no_effect_and_present_loss_quarantines_owner() {
    let (device, rejected_native) = fixture(Script::SurfacePresentRejected);
    let mut rejected = Renderer::new(device.clone()).unwrap();
    let (_, image) = surface_image(&device);
    assert!(matches!(
        submit_surface(&mut rejected, triangle(), pipeline(&device), image),
        Err(RendererError::Rhi(Error::SurfaceOutdated))
    ));
    assert_eq!(rejected.pending(), 0);
    assert_eq!(rejected_native.finishes.load(Ordering::SeqCst), 1);

    let (lost_device, lost_native) = fixture(Script::SurfacePresentDeviceLost);
    let mut lost = Renderer::new(lost_device.clone()).unwrap();
    let (_, image) = surface_image(&lost_device);
    let (source, weak) = snapshot_bytes(vec![0; 3 * 24]);
    assert!(matches!(
        submit_surface(&mut lost, source, pipeline(&lost_device), image),
        Err(RendererError::RetainedExecution {
            status: ExecutionStatus::Indeterminate,
            recording_error: Some(Error::DeviceLost),
            execution_error: Some(Error::DeviceLost),
        })
    ));
    assert_eq!(lost.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(lost_native.finishes.load(Ordering::SeqCst), 1);
    assert!(matches!(
        submit(&mut lost, snapshot().0),
        Err(RendererError::RendererClosed)
    ));
}

#[test]
fn nonaccepted_surface_outcomes_keep_the_handed_off_owner_until_successful_idle() {
    for (script, status) in [
        (Script::PartialAccepted, ExecutionStatus::PartiallyAccepted),
        (Script::PartialIssued, ExecutionStatus::Failed),
        (Script::Indeterminate, ExecutionStatus::Indeterminate),
    ] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let (_, image) = surface_image(&device);
        let (source, weak) = snapshot_bytes(vec![0; 3 * 24]);
        assert!(matches!(
            submit_surface(&mut renderer, source, pipeline(&device), image),
            Err(RendererError::RetainedExecution { status: actual, .. }) if actual == status
        ));
        assert_eq!(renderer.pending(), 1);
        assert!(weak.upgrade().is_some());
        let report = block_on(renderer.shutdown()).unwrap();
        assert_eq!(report.retired_submissions, 1);
        assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn incomplete_or_failed_surface_completion_never_retires_the_frame() {
    for script in [
        Script::AcceptedWaitPending,
        Script::AcceptedCompletionDeviceLost,
    ] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let (_, image) = surface_image(&device);
        let (source, weak) = snapshot_bytes(vec![0; 3 * 24]);
        let submission = submit_surface(&mut renderer, source, pipeline(&device), image).unwrap();
        if matches!(script, Script::AcceptedWaitPending) {
            poll_once_pending(renderer.complete(&submission));
        } else {
            assert!(matches!(
                block_on(renderer.complete(&submission)),
                Err(RendererError::Rhi(Error::DeviceLost))
            ));
        }
        assert_eq!(renderer.pending(), 1);
        assert!(weak.upgrade().is_some());
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn replacement_pipeline_is_rejected_before_old_device_execution() {
    let (old_device, old_native) = fixture(Script::AcceptedExact);
    let (new_device, new_native) = fixture(Script::AcceptedExact);
    let replacement_pipeline = pipeline(&new_device);
    let mut renderer = Renderer::new(old_device.clone()).unwrap();
    let (_, image) = surface_image(&old_device);

    assert!(matches!(
        submit_surface(&mut renderer, triangle(), replacement_pipeline, image),
        Err(RendererError::Rhi(Error::WrongDevice))
    ));
    assert_eq!(old_native.finishes.load(Ordering::SeqCst), 0);
    assert_eq!(old_native.command_calls.load(Ordering::SeqCst), 0);
    assert_eq!(new_native.finishes.load(Ordering::SeqCst), 0);
}

#[test]
fn replacement_device_does_not_adopt_old_surface_submission_or_snapshot() {
    let (old_device, old_native) = fixture(Script::AcceptedCompletionDeviceLost);
    let mut old = Renderer::new(old_device.clone()).unwrap();
    let (_, old_image) = surface_image(&old_device);
    let (old_source, old_weak) = snapshot_bytes(vec![0; 3 * 24]);
    let old_submission =
        submit_surface(&mut old, old_source, pipeline(&old_device), old_image).unwrap();
    assert!(matches!(
        block_on(old.complete(&old_submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert!(old_weak.upgrade().is_some());
    assert_eq!(old.pending(), 1);

    let (new_device, new_native) = fixture(Script::AcceptedExact);
    let mut replacement = Renderer::new(new_device.clone()).unwrap();
    assert!(matches!(
        block_on(replacement.complete(&old_submission)),
        Err(RendererError::ForeignSubmission)
    ));
    let (_, new_image) = surface_image(&new_device);
    let new_submission = submit_surface(
        &mut replacement,
        triangle(),
        pipeline(&new_device),
        new_image,
    )
    .unwrap();
    block_on(replacement.complete(&new_submission)).unwrap();
    assert_eq!(new_native.waits.load(Ordering::SeqCst), 1);
    assert!(old_native.waited_completions.lock().unwrap().len() == 1);
}
