use super::*;

fn triangle() -> AssetSnapshot<Bytes, Vec<u8>> {
    snapshot_bytes(vec![0; 3 * 24]).0
}

fn pipeline(device: &Device<WitnessBackend>) -> RasterPipeline<WitnessBackend> {
    // SAFETY: this inert pipeline belongs to the supplied witness device.
    unsafe { RasterPipeline::from_native(device.id(), ()) }
}

fn submit_raster(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: AssetSnapshot<Bytes, Vec<u8>>,
    pipeline: RasterPipeline<WitnessBackend>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit_raster(snapshot, pipeline, [4, 4]))
}

#[test]
fn raster_accepted_outcome_owns_and_retires_the_full_fixture() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (source, weak) = snapshot_bytes(vec![0; 3 * 24]);
    let _submission = submit_raster(&mut renderer, source, pipeline(&device)).unwrap();

    assert_eq!(renderer.pending(), 1);
    // This logical backend intentionally returns no pixel payload; use idle
    // only as ownership evidence, not as an output oracle.
    let report = block_on(renderer.shutdown()).unwrap();
    assert_eq!(report.retired_submissions, 1);
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
}

#[test]
fn raster_requirements_refuse_every_owned_resource_limit_before_cache_or_allocation() {
    for case in [
        "upload-and-vertex-buffer",
        "readback-buffer",
        "texture-dimension",
        "texture-array-layers",
        "color-attachment",
        "vertex-buffer",
        "vertex-attributes",
    ] {
        let (device, native) = fixture_with_caps(Script::AcceptedExact, |caps| match case {
            "upload-and-vertex-buffer" => caps.limits.max_buffer_bytes = 71,
            "readback-buffer" => caps.limits.max_buffer_bytes = 1023,
            "texture-dimension" => caps.limits.max_texture_dimension_2d = 3,
            "texture-array-layers" => caps.limits.max_texture_array_layers = 0,
            "color-attachment" => caps.limits.max_color_attachments = 0,
            "vertex-buffer" => caps.limits.max_vertex_buffers = 0,
            "vertex-attributes" => caps.limits.max_vertex_attributes = 1,
            _ => unreachable!(),
        });
        let mut renderer = Renderer::new(device.clone()).unwrap();
        assert!(
            matches!(
                submit_raster(&mut renderer, triangle(), pipeline(&device)),
                Err(RendererError::Rhi(Error::InvalidRange))
            ),
            "{case}"
        );
        assert_eq!(renderer.compilations(), 0, "{case}");
        assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0, "{case}");
        assert_eq!(native.command_calls.load(Ordering::SeqCst), 0, "{case}");
    }
}

#[test]
fn raster_clean_rejection_releases_without_retaining_a_submission() {
    let (device, native) = fixture(Script::CleanRejected);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (source, weak) = snapshot_bytes(vec![0; 3 * 24]);

    assert!(matches!(
        submit_raster(&mut renderer, source, pipeline(&device)),
        Err(RendererError::Rhi(Error::InvalidState))
    ));
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
}

#[test]
fn raster_nonaccepted_outcomes_retain_the_actual_fixture_until_idle() {
    for (script, status) in [
        (Script::PartialAccepted, ExecutionStatus::PartiallyAccepted),
        (Script::PartialIssued, ExecutionStatus::Failed),
        (Script::Indeterminate, ExecutionStatus::Indeterminate),
    ] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let (source, weak) = snapshot_bytes(vec![0; 3 * 24]);
        assert!(matches!(
            submit_raster(&mut renderer, source, pipeline(&device)),
            Err(RendererError::RetainedExecution { status: actual, .. }) if actual == status
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
fn raster_recording_loss_finishes_then_preserves_both_diagnostics_and_owner() {
    let (device, native) = fixture(Script::RecordingDeviceLost);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (source, weak) = snapshot_bytes(vec![0; 3 * 24]);
    assert!(matches!(
        submit_raster(&mut renderer, source, pipeline(&device)),
        Err(RendererError::RetainedExecution {
            status: ExecutionStatus::PartiallyAccepted,
            recording_error: Some(Error::DeviceLost),
            execution_error: None,
        })
    ));
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert!(matches!(
        submit_raster(&mut renderer, triangle(), pipeline(&device)),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
}
