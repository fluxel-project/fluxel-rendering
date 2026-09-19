//! Contract tests for rhi-design section 48.

use super::super::platform::{Label, RhiError, RhiErrorKind};
use super::{DiagnosticEvent, DiagnosticLog, DiagnosticSeverity};

/// A log large enough that the tests below only evict when they mean to.
fn log(capacity: usize) -> DiagnosticLog {
    DiagnosticLog::new(capacity)
}

fn info(message: &str) -> DiagnosticEvent {
    DiagnosticEvent::new(DiagnosticSeverity::Info, message)
}

#[test]
fn drain_returns_pending_events_oldest_first() {
    let log = log(4);
    log.push(info("first"));
    log.push(info("second"));

    assert_eq!(log.pending(), 2);

    let mut out = Vec::new();
    log.drain(&mut out);

    let messages: Vec<&str> = out.iter().map(|event| event.message.as_str()).collect();
    assert_eq!(messages, ["first", "second"]);
    assert_eq!(log.pending(), 0);
}

#[test]
fn drain_appends_to_the_callers_vector_rather_than_replacing_it() {
    let log = log(4);
    log.push(info("new"));

    let mut out = vec![info("existing")];
    log.drain(&mut out);

    let messages: Vec<&str> = out.iter().map(|event| event.message.as_str()).collect();
    assert_eq!(messages, ["existing", "new"]);
}

#[test]
fn a_second_drain_is_empty() {
    let log = log(4);
    log.push(info("once"));

    let mut first = Vec::new();
    log.drain(&mut first);
    assert_eq!(first.len(), 1);

    let mut second = Vec::new();
    log.drain(&mut second);
    assert!(second.is_empty());
}

#[test]
fn a_full_log_evicts_the_oldest_event() {
    let log = log(2);
    log.push(info("first"));
    log.push(info("second"));
    log.push(info("third"));

    let mut out = Vec::new();
    log.drain(&mut out);

    // Two survivors plus the warning that says one was lost.
    let messages: Vec<&str> = out.iter().map(|event| event.message.as_str()).collect();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0], "second");
    assert_eq!(messages[1], "third");
    assert!(messages[2].contains('1'));
}

#[test]
fn eviction_is_reported_rather_than_silent() {
    let log = log(1);
    for index in 0..5 {
        log.push(info(&format!("event {index}")));
    }

    let mut out = Vec::new();
    log.drain(&mut out);

    let warning = out
        .last()
        .expect("a drain after eviction always reports it");
    assert_eq!(warning.severity, DiagnosticSeverity::Warning);
    assert!(
        warning.message.contains('4'),
        "the warning must say how many events were lost, got {:?}",
        warning.message
    );
}

#[test]
fn the_drop_count_is_reported_once() {
    let log = log(1);
    log.push(info("a"));
    log.push(info("b"));

    let mut first = Vec::new();
    log.drain(&mut first);
    assert_eq!(first.len(), 2, "one survivor plus one warning");

    let mut second = Vec::new();
    log.drain(&mut second);
    assert!(
        second.is_empty(),
        "the drop count is consumed by the drain that reported it"
    );
}

#[test]
fn a_zero_capacity_log_keeps_nothing_but_still_counts() {
    let log = log(0);
    log.push(info("a"));
    log.push(info("b"));

    assert_eq!(log.pending(), 0);

    let mut out = Vec::new();
    log.drain(&mut out);

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].severity, DiagnosticSeverity::Warning);
    assert!(out[0].message.contains('2'));
}

#[test]
fn a_portable_error_becomes_an_error_event() {
    let error = RhiError::new(RhiErrorKind::InvalidUsage, "range is inverted")
        .at("CommandRecorder::draw")
        .on(super::super::platform::next_object_id());

    let event = DiagnosticEvent::from_error(&error);

    assert_eq!(event.severity, DiagnosticSeverity::Error);
    assert_eq!(event.message, "range is inverted");
    assert_eq!(event.operation, Some("CommandRecorder::draw"));
    assert_eq!(event.object, error.object());
    assert!(event.backend_detail.is_none());
}

#[test]
fn a_label_is_copied_into_the_event() {
    let event = info("lost").with_label(&Label::new("main color target"));

    assert_eq!(event.label.as_deref(), Some("main color target"));

    let unlabelled = info("lost").with_label(&Label::none());
    assert!(unlabelled.label.is_none());
}

#[test]
fn backend_detail_is_carried_but_is_not_portable() {
    let event = info("fallback").with_backend_detail("VK_ERROR_DEVICE_LOST (0xfffffffe)");

    assert_eq!(
        event.backend_detail.as_deref(),
        Some("VK_ERROR_DEVICE_LOST (0xfffffffe)")
    );
    // The portable half is unchanged by native text.
    assert_eq!(event.message, "fallback");
    assert!(event.object.is_none());
}
