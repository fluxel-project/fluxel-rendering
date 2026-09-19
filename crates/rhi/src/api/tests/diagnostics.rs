//! Contract tests for the diagnostics chapter (specification section 48).
//!
//! The chapter has one verb and two types, and the verb is not implemented, so
//! almost everything here is a **shape test**: an ordinary function compiled but
//! never called, written as a realistic call site. That is the right instrument
//! for this chapter, because what section 48 fixes is a *pull* model — a caller
//! drains into its own buffer and decides what to do — and the question worth
//! asking before a backend exists is whether that model is expressible at all:
//! whether an event can be built without a sink, kept after the queue forgets it,
//! and read without the reader learning a driver's vocabulary.
//!
//! What the tests *can* run is the event's own structure: what is required, what
//! is optional, and what the optional fields are made of. Those assertions are
//! the ones that would catch a field quietly becoming mandatory, which is the
//! failure that would break every emit site at once.
//!
//! None of this is GPU evidence and none of it stands in for a device.

use crate::api::diagnostics::{DiagnosticEvent, DiagnosticSeverity};
use crate::api::identity::ObjectId;
use crate::api::platform::device::Device;

/// An event about an object, written the way an emit site writes one.
fn event_about_an_object() -> DiagnosticEvent {
    DiagnosticEvent {
        severity: DiagnosticSeverity::Warning,
        message: "the descriptor uses an alternate view format the device did not enable"
            .to_owned(),
        object: Some(ObjectId::new(7)),
        label: Some("albedo".to_owned()),
        operation: Some("create_texture"),
        backend_detail: Some("VK_ERROR_FORMAT_NOT_SUPPORTED".to_owned()),
    }
}

/// Three levels, and the two the RHI deliberately does not have are the
/// interesting part.
///
/// There is no `Fatal`, because device loss is a state a caller reads from
/// `Device::status` rather than a severity a message may declare — a message that
/// could declare the device dead would let a backend announce a state transition
/// through a channel that is not a state machine. There is no `Trace`, because
/// per-command tracing is the tooling surface's job.
#[test]
fn there_are_three_severities_and_none_of_them_is_fatal() {
    let levels = [
        DiagnosticSeverity::Info,
        DiagnosticSeverity::Warning,
        DiagnosticSeverity::Error,
    ];

    for (index, level) in levels.iter().enumerate() {
        assert!(
            levels
                .iter()
                .enumerate()
                .all(|(other, candidate)| index == other || level != candidate),
            "level {index} repeats an earlier variant"
        );
    }

    assert_ne!(DiagnosticSeverity::Warning, DiagnosticSeverity::Info);
    assert_ne!(DiagnosticSeverity::Error, DiagnosticSeverity::Warning);
}

/// An event with no object is a statement about the device or the call, not a
/// statement about an unknown object.
///
/// That distinction is why the fields are `Option` rather than being omitted from
/// a separate "device event" type: a caller filtering "everything about this
/// texture" and "everything that happened" writes one filter either way, and two
/// record types would make the second one unable to see the first.
#[test]
fn an_event_that_names_no_object_is_still_a_complete_event() {
    let device_wide = DiagnosticEvent {
        severity: DiagnosticSeverity::Info,
        message: "device initialization completed".to_owned(),
        object: None,
        label: None,
        operation: Some("create_device"),
        backend_detail: None,
    };

    assert!(device_wide.object.is_none());
    assert!(device_wide.label.is_none());
    assert!(device_wide.backend_detail.is_none());

    // Only severity and message are required, and both are present.
    assert_eq!(device_wide.severity, DiagnosticSeverity::Info);
    assert!(!device_wide.message.is_empty());
}

/// The label travels with the event rather than being looked up by the reader.
///
/// Section 48 gives the reason and this test is that reason: a diagnostic stays
/// meaningful after the object it is about is reclaimed and its label is gone. A
/// reader that had to resolve `object` against a live inventory would find
/// nothing exactly in the case where it most wants to know what happened — an
/// object that was created and destroyed around the problem.
#[test]
fn an_event_keeps_its_objects_label_after_the_object_is_gone() {
    let mut event = event_about_an_object();

    // `DiagnosticEvent` has no lifetime parameter, which is the whole assertion
    // this test makes about ownership: the event holds `ObjectId` and its label
    // text itself, so there is no caller handle for it to borrow and nothing for
    // the caller to keep alive. If a future revision made `label` a
    // `&'a str`, this file would stop compiling — which is exactly the review
    // signal the shape tests exist to produce.
    assert_eq!(event.label.as_deref(), Some("albedo"));
    assert_eq!(event.object, Some(ObjectId::new(7)));

    // Cloning keeps the text, because the queue is emptied by the drain and a
    // caller that wants to keep an event must be able to keep it.
    let kept = event.clone();
    event.message.clear();
    assert!(!kept.message.is_empty());
    assert_eq!(kept.severity, DiagnosticSeverity::Warning);
}

/// `backend_detail` is a `String` for a reason section 48 states: backend detail
/// does not participate in portable correctness, so giving it structure would
/// invite a caller to branch on it.
///
/// The test cannot assert "no caller branches on this". What it can pin down is
/// the shape that makes branching unattractive — an untyped string a caller has to
/// parse, and an `Option` so that its absence is an ordinary state rather than a
/// missing field.
#[test]
fn backend_detail_is_free_text_and_is_optional() {
    let with_detail = event_about_an_object();
    assert_eq!(
        with_detail.backend_detail.as_deref(),
        Some("VK_ERROR_FORMAT_NOT_SUPPORTED")
    );

    // A different backend's wording is equally valid, which is the point: there
    // is no vocabulary here to depend on.
    let d3d12 = DiagnosticEvent {
        backend_detail: Some("D3D12: the resource state was not compatible".to_owned()),
        ..with_detail.clone()
    };
    assert_ne!(d3d12.backend_detail, with_detail.backend_detail);
    assert_eq!(d3d12.severity, with_detail.severity);
}

/// The operation name is a `&'static str`, for the reason the error model gives:
/// it names a fixed call site, costs no allocation on a path that may run per
/// command, and cannot drift into carrying caller data.
#[test]
fn the_operation_names_a_call_site_and_costs_no_allocation() {
    const SITE: &str = "create_texture";

    let event = DiagnosticEvent {
        operation: Some(SITE),
        ..event_about_an_object()
    };

    assert_eq!(event.operation, Some("create_texture"));

    // Two events from the same site compare equal on the name without either
    // having allocated one.
    let other = DiagnosticEvent {
        operation: Some(SITE),
        ..event_about_an_object()
    };
    assert_eq!(event.operation, other.operation);
}

/// The drain verb as a host calls it: into a buffer the caller owns, appended to
/// rather than replacing, so one call site can accumulate across devices.
///
/// Compiled, never called. The queue is filled by the code that detects the
/// problem and that code does not exist yet, so the verb panics; what this test
/// reviews is that the caller does not have to hand the RHI a callback, a sink, or
/// a buffer lifetime the caller cannot express. Section 48's own note on the verb
/// — *"Pull model; avoids imposing a callback threading policy"* — is the contract
/// this call site has to satisfy, and it does: no closure, no `Send` bound, no
/// `'static`, and the buffer is a plain `&mut Vec` the caller keeps.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_host_drains_into_its_own_buffer(device: &Device) {
    let mut buffer: Vec<DiagnosticEvent> = Vec::new();

    device.drain_diagnostics(&mut buffer);

    // Appending, not replacing: a caller accumulating across two devices gets
    // both devices' events in one buffer.
    let before = buffer.len();
    device.drain_diagnostics(&mut buffer);
    assert!(buffer.len() >= before);
}
