//! Error-model contract tests (specification section 4).

use crate::api::{ObjectId, RhiError, RhiErrorKind};

#[test]
fn every_section_4_kind_has_a_distinct_name() {
    // Section 4 lists exactly eleven kinds. This test is the tripwire for a
    // twelfth being added without the section being updated: `as_str` is a match
    // with no wildcard, so a new variant fails to compile there first, and this
    // count is what fails here.
    let kinds = [
        RhiErrorKind::InvalidUsage,
        RhiErrorKind::NoSuitableAdapter,
        RhiErrorKind::Unsupported,
        RhiErrorKind::IncompatibleInterface,
        RhiErrorKind::MissingDependency,
        RhiErrorKind::WrongDevice,
        RhiErrorKind::OutOfMemory,
        RhiErrorKind::TargetOutdated,
        RhiErrorKind::TargetLost,
        RhiErrorKind::DeviceLost,
        RhiErrorKind::BackendFailure,
    ];
    let mut names: Vec<&str> = kinds.iter().map(|kind| kind.as_str()).collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), count, "two kinds share a name: {names:?}");
    assert_eq!(count, 11, "section 4 names eleven kinds");
}

#[test]
fn the_operation_name_is_the_first_one_attached() {
    // The innermost call site names the error, so that a refusal reached from
    // `Device::create_buffer` through a backend still tells the caller which
    // verb they called while preserving where it was actually decided. The
    // façade relies on this when it attaches its own name to an error it did not
    // raise.
    let error = RhiError::new(RhiErrorKind::InvalidUsage, "size is zero")
        .at("backend::dx12::create_buffer")
        .at("Device::create_buffer");
    assert_eq!(error.operation(), Some("backend::dx12::create_buffer"));
}

#[test]
fn an_error_carries_the_object_it_concerns_when_there_is_one() {
    let object = ObjectId::new(42);
    let error = RhiError::new(RhiErrorKind::WrongDevice, "resource belongs elsewhere")
        .with_object(object)
        .at("Device::submit");

    assert_eq!(error.object(), Some(object));
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert!(error.to_string().contains("42"));
}

#[test]
fn an_error_without_an_operation_or_object_displays_plainly() {
    let error = RhiError::new(RhiErrorKind::Unsupported, "no direct route");
    assert_eq!(error.operation(), None);
    assert_eq!(error.object(), None);
    assert_eq!(error.to_string(), "Unsupported: no direct route");
}

#[test]
fn a_kind_is_matchable_through_a_wildcard_arm() {
    // `#[non_exhaustive]` is what a downstream crate sees. The assertion pins the
    // behaviour a caller depends on: a kind added in a later version must be
    // branchable through the wildcard rather than becoming a compile error at the
    // call site.
    fn classify(kind: RhiErrorKind) -> &'static str {
        match kind {
            RhiErrorKind::WrongDevice => "wrong device",
            RhiErrorKind::DeviceLost => "lost",
            _ => "other",
        }
    }
    assert_eq!(classify(RhiErrorKind::WrongDevice), "wrong device");
    assert_eq!(classify(RhiErrorKind::DeviceLost), "lost");
    assert_eq!(classify(RhiErrorKind::OutOfMemory), "other");
}

#[test]
fn the_error_type_is_usable_from_question_mark_in_a_boxed_caller() {
    // Section 4 does not show a `std::error::Error` impl, but every consumer of a
    // fallible device verb needs one to write `?` into a boxed error. This test
    // is what keeps that ergonomic completion from being dropped as unused.
    fn fallible() -> Result<(), Box<dyn std::error::Error>> {
        Err(RhiError::new(RhiErrorKind::DeviceLost, "device is gone"))?
    }

    let error = fallible().expect_err("the inner error propagates");
    assert!(error.to_string().contains("device is gone"));

    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<RhiError>();
}
