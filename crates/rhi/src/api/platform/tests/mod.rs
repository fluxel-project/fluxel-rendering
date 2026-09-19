//! Contract tests for the platform chapter (specification sections 5 through 7).
//!
//! Only the rules that are decidable without a backend are testable here, and
//! they are the ones that matter most: section 3.1 requires every public
//! operation to validate identity *before* touching a backend, so a wrong-device
//! or wrong-provider argument has to be refused by the façade rather than handed
//! down for a driver to discover. Everything else in this chapter panics with a
//! documented message and is covered by shape tests instead.

use crate::api::error::RhiErrorKind;
use crate::api::identity::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::platform::requirements::{DeviceRequirements, LimitKey};
use crate::api::platform::{
    AdapterId, AdapterSelection, BackendKind, Device, DeviceLossInfo, DeviceRequest,
    DeviceRequestDescriptor, DeviceStatus, PlatformProvider,
};
use crate::api::presentation::PresentationTarget;
use crate::api::resource::buffer::{BufferDescriptor, BufferUsage};

/// A device identity under the instance/generation pair section 3 defines.
fn identity(instance: u64, generation: u64) -> DeviceIdentity {
    DeviceIdentity::new(
        DeviceInstanceId::new(instance),
        DeviceGeneration::new(generation),
    )
}

/// A live device under the identity every other fixture in this file uses.
fn live_device() -> Device {
    Device::new(identity(1, 1))
}

/// An adapter that belongs to a different provider is refused by the façade.
///
/// This is section 3.1's O(1) identity check, and it is the reason the check runs
/// before the probing that would otherwise panic: a caller that passes a foreign
/// adapter gets a structured refusal, not an `unimplemented!()`.
#[test]
fn a_foreign_adapter_is_refused_before_any_probing() {
    let provider = PlatformProvider::new(BackendKind::Dx12, 1);
    let target = PresentationTarget::new(ObjectId::new(1));

    assert_eq!(provider.backend(), BackendKind::Dx12);

    let error = provider
        .supports_presentation(AdapterId::new(2, 0), &target)
        .expect_err("an adapter from another provider must not be accepted");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

/// The limit classification has no wildcard arm, so a new key fails to compile
/// until it is classified. This test is what keeps the classification honest in
/// the meantime: a key moved to the wrong side is caught here rather than by a
/// device request that silently accepts a weaker limit.
#[test]
fn limits_are_classified_by_the_direction_that_makes_them_stronger() {
    // `Max*` limits grow stronger with size.
    assert!(LimitKey::MaxBufferSize.larger_is_stronger());
    assert!(LimitKey::MaxBindGroups.larger_is_stronger());
    assert!(LimitKey::MaxComputeWorkgroupStorageSize.larger_is_stronger());

    // Alignment limits grow stronger as they shrink, which is why section 7.4
    // refuses to unify the two directions behind one `minimum_limit()`.
    assert!(!LimitKey::MinUniformBufferOffsetAlignment.larger_is_stronger());
    assert!(!LimitKey::MinStorageBufferOffsetAlignment.larger_is_stronger());
}

/// A descriptor answers back what was put into it, which is what makes the
/// request path reviewable from a call site.
#[test]
fn a_device_request_descriptor_round_trips_its_contents() {
    let target = PresentationTarget::new(ObjectId::new(7));

    let descriptor = DeviceRequestDescriptor::new(
        AdapterSelection::PreferHighPerformance,
        DeviceRequirements::new(),
    )
    .require_presentation_target(target.clone());

    assert_eq!(
        descriptor.selection(),
        AdapterSelection::PreferHighPerformance
    );
    assert_eq!(descriptor.presentation_targets().len(), 1);
    assert_eq!(
        descriptor.presentation_targets()[0].id(),
        target.id(),
        "the target comes back out under the same identity"
    );

    // A headless request is a request with no target, not an error.
    let headless =
        DeviceRequestDescriptor::new(AdapterSelection::Default, DeviceRequirements::new());
    assert!(headless.presentation_targets().is_empty());
}

/// A request is single-shot: once it has reported its outcome, asking again is a
/// usage error rather than another look at a finished result.
#[test]
fn a_completed_device_request_refuses_a_second_poll() {
    let mut request = DeviceRequest::new();
    request.mark_complete();

    let error = request
        .poll()
        .expect_err("a completed request must not appear to still be in flight");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

/// A lost device refuses creation itself, and the refusal carries the reason.
///
/// This is the rule section 6.5 states for the handles it lists: they "must
/// return `WrongDevice` when passed to that new Device, and return `DeviceLost`
/// when used through their lost original Device". Section 6.9 puts the verdict
/// here rather than below, and gives the reason — a release environment may not
/// have native validation on at all, so "let the driver notice" is not a legal
/// implementation of this rule.
///
/// It is reachable on today's tree for a reason worth naming: it returns before
/// the capability read that still panics. An *active* device stops inside
/// `Device::capabilities()`, so a lost one is the only state in which this verb
/// answers at all. The ownership refusals of the verbs that take a handle are
/// reachable the same way, for the same reason.
#[test]
fn a_lost_device_refuses_creation_and_says_why() {
    let mut device = live_device();
    device.mark_lost(DeviceLossInfo::new(
        "the driver reset the adapter".to_string(),
    ));

    assert_eq!(device.status(), DeviceStatus::Lost);
    assert_eq!(
        device.loss_info().map(|loss| loss.message().to_string()),
        Some("the driver reset the adapter".to_string())
    );

    let error = device
        .create_buffer(&BufferDescriptor::new(64, BufferUsage::COPY_DST))
        .expect_err("a lost device must refuse to create a resource");

    assert_eq!(
        error.kind(),
        RhiErrorKind::DeviceLost,
        "{}",
        error.message()
    );
    assert!(
        error.message().contains("the driver reset the adapter"),
        "the refusal must carry section 6.5's stable loss summary, got: {}",
        error.message()
    );
}
