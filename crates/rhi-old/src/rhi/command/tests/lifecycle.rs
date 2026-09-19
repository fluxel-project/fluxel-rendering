//! Section 29.2 and 29.3: the recorder's lifecycle and its two failure
//! policies.
//!
//! The distinction these tests exist to pin is between the two ways a command
//! can fail. A parameter or capability error rejects one command and leaves the
//! recorder usable; a backend failure, or a failure while finalizing a scope,
//! poisons the recorder so every later command is refused. The mock's journal
//! is what separates them: a command the recorder refused never reached the
//! backend, so it leaves no entry, and a command the backend refused does.

use super::*;
// ---------------------------------------------------------------------------
// Section 29.2: the recorder's lifecycle
// ---------------------------------------------------------------------------

#[test]
fn a_recorder_opens_closes_and_finishes() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let view = color_view(&mock);
    let descriptor = one_color_scope(&view);

    {
        let scope: RasterScope<'_> = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.end().expect("an empty raster scope closes");
    }
    {
        let scope: ComputeScope<'_> = open_compute(&mut recorder, &mock);
        scope.end().expect("an empty compute scope closes");
    }

    assert!(!recorder.is_poisoned());
    let work = recorder.finish().expect("an open recorder finalizes");

    assert!(work.work_domains().contains(LaneWorkDomains::RASTER));
    assert!(work.work_domains().contains(LaneWorkDomains::COMPUTE));
    assert!(!work.work_domains().contains(LaneWorkDomains::COPY));

    // Both scopes really reached the backend.
    assert!(mock.saw("recorder:begin_raster"));
    assert!(mock.saw("recorder:end_compute"));

    // A scope that draws nothing declares no resource use. Uses are derived
    // from the commands that produce them, not from the attachment set of a
    // scope that was opened.
    assert!(
        work.resource_uses().is_empty(),
        "an empty scope touches nothing: {:?}",
        work.resource_uses()
    );
    assert_eq!(
        journal_count(&mock, "recorder:record:Draw"),
        0,
        "no draw was recorded, so no attachment was used"
    );
}

#[test]
fn a_raster_scope_dropped_without_end_poisons_the_recorder() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let view = color_view(&mock);
    let descriptor = one_color_scope(&view);

    {
        let _scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        // Dropped without `end()`.
    }

    assert!(
        recorder.is_poisoned(),
        "a scope that is dropped without end() poisons the recorder"
    );
    let error = recorder
        .begin_compute()
        .expect_err("a poisoned recorder refuses every later command");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_compute_scope_dropped_without_end_poisons_the_recorder() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();

    {
        let _scope = open_compute(&mut recorder, &mock);
    }

    assert!(recorder.is_poisoned());
    let error = recorder
        .insert_debug_marker("after")
        .expect_err("a poisoned recorder refuses every later command");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

#[test]
fn ending_a_scope_with_an_open_debug_group_is_refused_and_then_poisons() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let view = color_view(&mock);
    let descriptor = one_color_scope(&view);

    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope
            .push_debug_group("open")
            .expect("a debug group opens on the scope");
        let error = scope
            .end()
            .expect_err("a scope with an open debug group cannot end");
        assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
        assert_eq!(error.operation(), Some("end"));
        // `end` refused before it marked the scope ended, so `Drop` poisons.
    }

    assert!(recorder.is_poisoned());
}

#[test]
fn an_unbalanced_recorder_pop_is_refused_without_poisoning() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();

    let error = recorder
        .pop_debug_group()
        .expect_err("no debug group is open on the recorder");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("pop_debug_group"));
    assert!(
        !recorder.is_poisoned(),
        "a parameter error rejects one command and leaves the recorder usable"
    );

    recorder
        .push_debug_group("balanced")
        .expect("a debug group opens");
    recorder
        .pop_debug_group()
        .expect("a balanced debug group closes");
    recorder
        .finish()
        .expect("the recorder is still usable after the refusal");
}

#[test]
fn finishing_with_an_open_debug_group_is_refused() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    recorder
        .push_debug_group("open")
        .expect("a debug group opens");

    let error = recorder
        .finish()
        .expect_err("a recorder with an open debug group cannot finalize");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("finish"));
}

// ---------------------------------------------------------------------------
// Section 29.3: two failure policies
// ---------------------------------------------------------------------------

#[test]
fn a_parameter_error_rejects_one_command_and_leaves_the_recorder_usable() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let (src, dst) = copy_pair(&mock, 64);

    let refused = BufferCopy {
        src: src.clone(),
        src_offset: 0,
        dst: dst.clone(),
        dst_offset: 0,
        size: 0,
    };
    let error = recorder
        .copy_buffer(&refused)
        .expect_err("a zero-length copy is refused");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("copy_buffer"));
    assert!(!recorder.is_poisoned());
    assert_eq!(
        journal_count(&mock, "recorder:record:CopyBuffer"),
        0,
        "the recorder refused the command before the backend saw it"
    );

    let accepted = BufferCopy {
        size: 64,
        ..refused
    };
    recorder
        .copy_buffer(&accepted)
        .expect("the recorder is still usable");
    assert_eq!(journal_count(&mock, "recorder:record:CopyBuffer"), 1);
    recorder.finish().expect("the recording finalizes");
}

#[test]
fn a_backend_record_failure_poisons_the_recorder() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let (src, dst) = copy_pair(&mock, 64);
    let copy = BufferCopy {
        src,
        src_offset: 0,
        dst,
        dst_offset: 0,
        size: 64,
    };

    mock.fail_next_record();
    let error = recorder
        .copy_buffer(&copy)
        .expect_err("an injected backend failure is reported");
    assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
    assert_eq!(
        journal_count(&mock, "recorder:record:CopyBuffer"),
        1,
        "the command reached the backend before it failed"
    );
    assert!(recorder.is_poisoned());

    let error = recorder
        .copy_buffer(&copy)
        .expect_err("a poisoned recorder refuses every later command");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_backend_scope_close_failure_poisons_the_recorder() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let view = color_view(&mock);
    let descriptor = one_color_scope(&view);

    let scope = recorder
        .begin_raster(&descriptor)
        .expect("a one-attachment scope opens");
    mock.fail_next_end_raster();
    let error = scope
        .end()
        .expect_err("a failed scope close is reported");
    assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
    assert!(
        recorder.is_poisoned(),
        "a scope-finalization failure poisons the recorder"
    );

    let error = recorder
        .begin_raster(&descriptor)
        .expect_err("a poisoned recorder refuses every later command");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_backend_compute_scope_close_failure_poisons_the_recorder() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();

    let scope = open_compute(&mut recorder, &mock);
    mock.fail_next_end_compute();
    let error = scope
        .end()
        .expect_err("a failed compute scope close is reported");
    assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
    assert!(
        recorder.is_poisoned(),
        "the compute path follows the same scope-finalization rule as the raster path"
    );

    // The failure is reported once, into the device's one log.
    let events = mock.drain_diagnostics();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].severity, DiagnosticSeverity::Error);
    assert_eq!(events[0].operation, Some("end_compute"));
}

#[test]
fn a_recorder_carries_the_descriptor_label_into_the_backend() {
    let mock = Mock::new();
    let descriptor = RecorderDescriptor::new().with_label("frame 7 commands");

    let mut recorder = mock
        .device()
        .create_recorder(&descriptor)
        .expect("the mock device creates a labelled recorder");
    recorder
        .insert_debug_marker("only")
        .expect("a marker is recorded");
    recorder.finish().expect("the recording finalizes");

    assert!(
        mock.saw("recorder:new:Label(Some(\"frame 7 commands\"))"),
        "the recorder's label reached the backend: {:?}",
        mock.journal()
    );
}

#[test]
fn a_failed_backend_finalization_is_reported_as_a_backend_failure() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    recorder
        .insert_debug_marker("only")
        .expect("a marker is recorded");

    mock.fail_next_finish();
    let error = recorder
        .finish()
        .expect_err("a failed finalization is reported");
    assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
    assert_eq!(error.operation(), Some("finish"));
}

#[test]
fn a_backend_failure_is_recorded_in_the_device_diagnostic_log() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let (src, dst) = copy_pair(&mock, 64);

    assert!(
        mock.drain_diagnostics().is_empty(),
        "a healthy device reports nothing"
    );

    mock.fail_next_record();
    let error = recorder
        .copy_buffer(&BufferCopy {
            src,
            src_offset: 0,
            dst,
            dst_offset: 0,
            size: 64,
        })
        .expect_err("an injected backend failure is reported");
    assert_eq!(error.kind(), RhiErrorKind::BackendFailure);

    let events = mock.drain_diagnostics();
    assert_eq!(
        events.len(),
        1,
        "the backend failure is the only diagnostic reported"
    );
    assert_eq!(events[0].severity, DiagnosticSeverity::Error);
    assert_eq!(events[0].operation, Some("record"));

    recorder.finish().expect_err("the recorder is poisoned");
}

#[test]
fn a_backend_record_failure_inside_a_scope_survives_the_scope_closing() {
    let mock = Mock::new();
    let mut recorder = mock.recorder();
    let view = color_view(&mock);
    let pipeline = plain_pipeline(&mock, COLOR_FORMAT);
    let descriptor = one_color_scope(&view);

    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a one-attachment scope opens");
        scope.set_pipeline(&pipeline).expect("the pipeline binds");

        mock.fail_next_record();
        let error = scope
            .draw(0..1, 0..1)
            .expect_err("an injected backend failure inside a scope is reported");
        assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
        assert_eq!(
            journal_count(&mock, "recorder:record:Draw"),
            1,
            "the draw passed validation and was refused by the backend, not by \
             the recorder"
        );

        // The scope's own close reaches the backend and *succeeds*: only the
        // draw failed. That success is what used to be mistaken for recovery.
        scope
            .end()
            .expect("the backend closes the scope it opened, even after a failed draw");
    }

    assert!(
        recorder.is_poisoned(),
        "section 29.3 makes a backend recording failure a one-way transition, so \
         a successful end() must not return the recorder to its open state"
    );
    let error = recorder
        .begin_compute()
        .expect_err("a poisoned recorder refuses every later command");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    let error = recorder
        .finish()
        .expect_err("and a poisoned recording can never be finalized");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}
