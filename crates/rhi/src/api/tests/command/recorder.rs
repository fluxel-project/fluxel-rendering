//! Sections 29 and 38: the recorder's lifetime, its debug-group stack, and the
//! work it produces.

use super::*;
use crate::api::command::RecordedWork;
use crate::api::command::{AccessMask, PipelineScope, ResourceUse};
use crate::api::submission::LaneWorkDomains;
use crate::api::tests::fixture;

#[test]
fn an_empty_recording_is_refused_at_finish() {
    // Section 10.1: a recording with no domain records nothing executable, and the
    // refusal belongs where it is reported rather than at the empty value.
    assert_kind(recorder().finish().map(|_| ()), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_debug_group_must_be_closed_before_finish() {
    let mut recorder = recorder();
    recorder
        .push_debug_group("pass")
        .expect("a push on an open recorder is legal");

    assert_kind(recorder.finish().map(|_| ()), RhiErrorKind::InvalidUsage);
}

#[test]
fn an_unmatched_pop_is_refused_without_poisoning() {
    let mut recorder = recorder();
    assert_kind(recorder.pop_debug_group(), RhiErrorKind::InvalidUsage);

    // Section 29.3 keeps parameter refusal out of the poisoning category: a marker
    // recorded afterwards still produces a usable recording.
    recorder
        .insert_debug_marker("after the refusal")
        .expect("the recorder is still open");
    let work = recorder.finish().expect("a marker is a command");
    assert!(work.work_domains().contains(LaneWorkDomains::COPY));
}

#[test]
fn a_recorder_prints_its_identity_and_progress_only() {
    // Adjudication A16: a hand-written `Debug` prints portable identity, never a
    // device-side handle.
    let mut recorder = recorder();
    recorder
        .insert_debug_marker("one command")
        .expect("an open recorder accepts a marker");

    let printed = format!("{recorder:?}");
    assert!(printed.contains("CommandRecorder"), "{printed}");
    assert!(printed.contains("commands: 1"), "{printed}");
    assert!(printed.contains("phase: Open"), "{printed}");
}

#[test]
fn a_poisoned_recorder_refuses_every_verb_and_the_finish() {
    let mut recorder = recorder();
    {
        let scope = recorder
            .begin_raster(&color_scope("dropped"))
            .expect("the attachment set is legal");
        drop(scope);
    }

    assert_kind(
        recorder.copy_buffer(&buffer_copy()),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(recorder.finish().map(|_| ()), RhiErrorKind::InvalidUsage);
}

#[test]
fn an_upload_records_a_copy_write_in_the_copy_domain() {
    // Both recorders are built before the first binding shadows the helper they
    // come from, which is also why they are named `first` and `second` rather
    // than `recorder`.
    let mut first = recorder();
    let mut second = recorder();

    first
        .encode_upload(&buffer_upload())
        .expect("a prepared upload is encodable");
    second
        .encode_upload(&buffer_upload())
        .expect("a prepared upload is encodable");

    let work = first.finish().expect("the recording is complete");
    assert_eq!(work.work_domains(), LaneWorkDomains::COPY);
    assert_eq!(work.device_identity(), device());

    // The id comes from the process-wide counter every object shares, so a literal
    // cannot be asserted: any other test creating an object concurrently moves it.
    // What can be asserted is the property that counter exists for, and it is the
    // one `finish` has to preserve — a recording's id is its own.
    let other = second.finish().expect("the recording is complete");
    assert_ne!(
        work.id(),
        other.id(),
        "two recordings are two pieces of work"
    );

    let use_ = work
        .resource_uses()
        .first()
        .expect("an upload produces a use");
    match use_ {
        ResourceUse::Buffer(buffer) => {
            assert_eq!(buffer.access, AccessMask::COPY_WRITE);
            assert_eq!(buffer.stages, PipelineScope::COPY);
            assert_eq!(buffer.range, BufferRange::new(0, 16));
        }
        ResourceUse::Texture(_) | ResourceUse::Frame(_) | ResourceUse::AccelerationStructure(_) => {
            panic!("a buffer upload must record a buffer use")
        }
    }
}

#[test]
fn an_upload_from_another_device_is_wrong_device() {
    let mut recorder = recorder();
    let foreign = UploadJob::new(
        object(81),
        other_device(),
        UploadDescriptor::Buffer(BufferUploadDescriptor {
            label: Label::default(),
            dst: fixture::buffer(
                object(12),
                other_device(),
                BufferDescriptor::new(64, BufferUsage::COPY_DST),
            ),
            dst_offset: 0,
            bytes: vec![7u8; 16].into(),
        }),
    );
    assert_kind(recorder.encode_upload(&foreign), RhiErrorKind::WrongDevice);
}

#[test]
fn recorded_work_prints_identity_and_domains_only() {
    let mut recorder = recorder();
    recorder
        .encode_upload(&buffer_upload())
        .expect("a prepared upload is encodable");
    let work = recorder.finish().expect("the recording is complete");

    let printed = format!("{work:?}");
    // What a caller may already read is what the printable form carries: the
    // identity, the device, the domain set, and the size of the recording. The
    // expectations are built from the accessors rather than spelled out, because
    // how `LaneWorkDomains` renders itself is the submission chapter's answer and
    // not this test's.
    assert!(printed.contains("RecordedWork"), "{printed}");
    assert!(printed.contains(&format!("{:?}", work.id())), "{printed}");
    assert!(
        printed.contains(&format!("{:?}", work.device_identity())),
        "{printed}"
    );
    assert!(
        printed.contains(&format!("{:?}", work.work_domains())),
        "{printed}"
    );
    // The recorded commands are the internal sequence (section 37.1), so they
    // appear as a count: a `Debug` that walked them would print every cloned
    // device-side handle the portable surface exists to hide.
    assert!(
        printed.contains(&format!("resource_uses: {}", work.resource_uses().len())),
        "{printed}"
    );
    assert!(!printed.contains("BufferUploadDescriptor"), "{printed}");
}

/// Section 38.1's accessors as a downstream consumer reaches them.
///
/// Compiled, never called: the same four accessors are exercised above, and what
/// this adds is the shape a consumer sees — a reference to the use list rather
/// than an owned copy, and a `LaneWorkDomains` value rather than a borrow of the
/// recorder, so a validator can compare them without holding the work mutably.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_consumer_reads_the_use_summary(work: &RecordedWork) {
    let _: ObjectId = work.id();
    let _: DeviceIdentity = work.device_identity();
    let _: LaneWorkDomains = work.work_domains();
    let _: &[ResourceUse] = work.resource_uses();
}
