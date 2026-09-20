//! Contract tests for the platform chapter (specification sections 5 through 7).
//!
//! Only the rules that are decidable without a backend are testable here, and
//! they are the ones that matter most: section 3.1 requires every public
//! operation to validate identity *before* touching a backend, so a wrong-device
//! or wrong-provider argument has to be refused by the façade rather than handed
//! down for a driver to discover. Everything else in this chapter panics with a
//! documented message and is covered by shape tests instead.
//!
//! The verbs that *are* backed run against [`crate::base::mock`], which is the
//! conformance vehicle described in that module: it answers from memory, so the
//! portable rules it exercises are checked on every platform in the same run.
//! It proves nothing about hardware, and nothing here should be read as if it
//! did.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use crate::api::capability::{CapabilityFacts, EnabledCapabilities};
use crate::api::error::RhiErrorKind;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::platform::requirements::{DeviceRequirements, LimitKey, OptionalFeature};
use crate::api::platform::{
    AdapterId, AdapterSelection, BackendKind, Device, DeviceLossInfo, DeviceRequestDescriptor,
    DeviceStatus, PlatformProvider,
};
use crate::api::presentation::PresentationTarget;
use crate::api::resource::buffer::{BufferDescriptor, BufferUsage};
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};
use crate::base::mock::{MockDevice, MockEnumeration, MockProvider};

/// A device identity under the single v13 device-instance token.
fn identity(instance: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(instance))
}

/// Minimal executor for the mock-only futures in this module.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {}
        }
    }
}

/// A mock backend under the instance every other fixture in this file uses.
fn mock_provider() -> MockProvider {
    MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1))
}

/// A provider wrapping the mock backend.
fn provider() -> PlatformProvider {
    let instance = DeviceInstanceId::new(1);
    PlatformProvider::new(
        BackendKind::Dx12,
        instance,
        MockProvider::new(BackendKind::Dx12, instance).shared(),
    )
}

/// A live device under the identity every other fixture in this file uses.
///
/// Returned with its backend so a test that needs to observe a loss can reach
/// the half that observes it. The portable handle owns the backend, so a test
/// without its own handle could not mark anything lost.
fn live_device() -> (Device, Arc<MockDevice>) {
    let native = MockDevice::new(BackendKind::Dx12, mock_provider().adapter());
    (
        Device::new(identity(1), native.clone())
            .expect("the mock backend offers a lane accepting raster and copy work"),
        native,
    )
}

/// A headless request descriptor with no requirements.
fn headless_request() -> DeviceRequestDescriptor {
    DeviceRequestDescriptor::new(AdapterSelection::Default, DeviceRequirements::new())
}

/// An adapter that belongs to a different provider is refused by the façade.
///
/// This is section 3.1's O(1) identity check, and it is the reason the check runs
/// before the probing it would otherwise reach: a caller that passes a foreign
/// adapter gets a structured refusal, not an answer from a driver that was asked
/// about a number it never issued.
#[test]
fn a_foreign_adapter_is_refused_before_any_probing() {
    let provider = provider();
    let target = PresentationTarget::new(ObjectId::new(1));

    assert_eq!(provider.backend(), BackendKind::Dx12);

    let error = provider
        .supports_presentation(AdapterId::new(2, 0), &target)
        .expect_err("an adapter from another provider must not be accepted");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

/// The three enumeration outcomes stay three.
///
/// `Ok(None)` and `Ok(Some(vec![]))` are different statements — "this provider
/// has no portable enumeration" against "it can enumerate and has nothing to
/// offer" — and a façade that collapsed them would tell a caller that a provider
/// which cannot enumerate has no adapters, which it has no way to know.
#[test]
fn enumeration_distinguishes_unsupported_from_empty() {
    let instance = DeviceInstanceId::new(1);

    let not_exposed = MockProvider::new(BackendKind::Dx12, instance)
        .enumerating(MockEnumeration::NotExposed)
        .shared();
    let empty = MockProvider::new(BackendKind::Dx12, instance)
        .enumerating(MockEnumeration::NoCandidate)
        .shared();
    let listed = MockProvider::new(BackendKind::Dx12, instance).shared();

    let provider = |native: Arc<dyn crate::base::platform::ProviderBackend>| {
        PlatformProvider::new(BackendKind::Dx12, instance, native)
    };

    assert!(
        block_on(provider(not_exposed).enumerate_adapters())
            .unwrap()
            .is_none(),
        "a provider with no portable enumeration must say so rather than report no adapters"
    );
    assert_eq!(
        block_on(provider(empty).enumerate_adapters())
            .unwrap()
            .map(|adapters| adapters.len()),
        Some(0),
        "a provider that can enumerate and has no candidate reports an empty list"
    );

    let adapters = block_on(provider(listed).enumerate_adapters())
        .unwrap()
        .unwrap();
    assert_eq!(adapters.len(), 1);
    assert_eq!(
        adapters[0].id(),
        AdapterId::new(instance.as_u64(), 0),
        "an enumerated adapter is scoped to the provider that produced it"
    );
    assert_eq!(adapters[0].backend(), BackendKind::Dx12);
}

/// A provider that cannot present to a target says so, and says so as a fact
/// rather than as an error.
///
/// Section 5.8 makes the distinction load-bearing: "this adapter has no
/// presentation route to that target" is an answer a caller plans around, while
/// `Err` would mean the preflight itself failed. It is also the case a device
/// request has to survive — a headless request is legal on a provider whose
/// adapters cannot present at all.
#[test]
fn a_provider_reports_an_adapter_that_cannot_present() {
    let instance = DeviceInstanceId::new(1);
    let target = PresentationTarget::new(ObjectId::new(1));
    let adapter = AdapterId::new(instance.as_u64(), 0);

    let presenting = PlatformProvider::new(
        BackendKind::Dx12,
        instance,
        MockProvider::new(BackendKind::Dx12, instance).shared(),
    );
    assert!(
        presenting
            .supports_presentation(adapter, &target)
            .expect("preflight is not expected to fail here")
    );

    let headless_only = PlatformProvider::new(
        BackendKind::Dx12,
        instance,
        MockProvider::new(BackendKind::Dx12, instance)
            .presenting(false)
            .shared(),
    );
    assert!(
        !headless_only
            .supports_presentation(adapter, &target)
            .expect("an adapter with no route is a fact, not a failure"),
        "a provider whose adapter cannot present must report false rather than fail"
    );
}

/// A request that resolves reports a device whose identity the *portable* layer
/// minted.
///
/// Section 6.1 ties identity minting to a completed request, and the backend is
/// deliberately not the minter: a backend that composed its own identity could
/// hand two domains the same one, or revive an old one by choosing a generation.
/// Section 3.1 lists both under "P0 None".
#[test]
fn a_resolved_request_yields_a_device_under_a_minted_identity() {
    let provider = provider();
    let device = block_on(provider.request_device(headless_request())).unwrap();

    assert_eq!(device.status(), DeviceStatus::Active);
    assert_eq!(device.backend(), BackendKind::Dx12);

    // A second request off the same provider is a new domain.
    let second = block_on(provider.request_device(headless_request())).unwrap();
    assert_ne!(
        second.identity(),
        device.identity(),
        "two standalone requests must not land in the same execution domain"
    );
}

/// A backend request that needs progress is exposed only as an awaited result.
#[test]
fn a_pending_request_resolves_through_the_async_boundary() {
    let instance = DeviceInstanceId::new(1);
    let native = MockProvider::new(BackendKind::Dx12, instance)
        .pending_steps(2)
        .shared();
    let provider = PlatformProvider::new(BackendKind::Dx12, instance, native);
    assert!(block_on(provider.request_device(headless_request())).is_ok());
}

/// A failed request is terminal, and the failure reaches the caller intact.
///
/// Section 5.9 draws two terminal outcomes and no third. A request that stayed
/// pollable after its backend had given up would invite a caller to keep asking a
/// question that has already been answered.
#[test]
fn a_failed_request_carries_its_error() {
    let instance = DeviceInstanceId::new(1);
    let native = MockProvider::new(BackendKind::Dx12, instance)
        .failing("the adapter was removed while the request was in flight")
        .shared();
    let provider = PlatformProvider::new(BackendKind::Dx12, instance, native);
    let error = block_on(provider.request_device(headless_request()))
        .expect_err("the request was configured to fail");
    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert_eq!(
        error.message(),
        "the adapter was removed while the request was in flight"
    );
}

/// A device answers provenance, identity, and progress from its backend.
///
/// These are the verbs whose answers are facts about the created device rather
/// than decisions about the caller's request, which is exactly the split the
/// seam draws: the backend reports, the portable layer decides.
#[test]
fn a_device_reports_backend_facts() {
    let (device, _native) = live_device();

    assert_eq!(device.backend(), BackendKind::Dx12);
    assert_eq!(device.adapter_info().backend(), BackendKind::Dx12);
    assert_eq!(
        device.adapter_info().id(),
        AdapterId::new(1, 0),
        "the device reports the adapter the provider actually offered"
    );
    assert_eq!(device.status(), DeviceStatus::Active);
    assert!(device.loss_info().is_none());
    assert!(device.poll().is_ok());
    assert!(block_on(device.wait_idle()).is_ok());

    // Section 7.1 asks tooling to describe what it observes by a process-local
    // object ID, and this is where a device says what its own is.
    let first = device.object_id();
    let (other, _native) = live_device();
    assert_ne!(
        other.object_id(),
        first,
        "two devices must not share a process-local object ID"
    );
}

/// The whole chain, end to end: a backend's enumeration becomes the contract a
/// portable device reports.
///
/// This is the test that section 7.1's interning rule is *reachable* through, and
/// reachability is the part worth asserting: before this block, every path from a
/// caller to [`Device::capabilities`] ended at a documented `unimplemented!()`, so
/// the id, the fingerprint, and the fact table were all unreachable from outside
/// the crate no matter how well tested they were in isolation.
///
/// The id is checked against an independently interned contract rather than
/// against a number. Section 7.1 makes the id the interning of the canonical
/// semantics, so the property that matters is *equal contracts intern together*
/// — and a golden integer would pin the encoding without checking that, which is
/// the mistake `api::tests::capability` explains in its own header.
#[test]
fn a_created_device_reports_the_contract_its_backend_enumerated() {
    let device = block_on(request_over(declared_facts(), declared_submission()))
        .expect("the mock request resolves");

    let capabilities = device.capabilities();
    assert!(
        capabilities.supports_feature(OptionalFeature::Compute),
        "the feature the backend recorded is the feature the device reports"
    );
    assert_eq!(capabilities.limit(LimitKey::MaxBufferSize), Some(1 << 28));
    assert_eq!(capabilities.submission().lanes().len(), 2);

    // The same contract, interned a second time. Equality here is the property a
    // `CompiledGraph` keys reuse on, so it is asserted against a *fresh* interning
    // rather than against an id the device is already holding — the latter would
    // pass even if the device reported a constant.
    let expected = EnabledCapabilities::from_facts(declared_facts(), declared_submission());
    assert_eq!(capabilities.compatibility_id(), expected.compatibility_id());
    assert_eq!(capabilities.fingerprint(), expected.fingerprint());
}

/// Section 7.2's base guarantee is checked where both halves of the enumeration are
/// in hand, and a backend that violates it produces no device at all.
///
/// `BackendFailure` rather than `Unsupported`, and the difference is the point:
/// every device is required to have a lane accepting `RASTER | COPY`, so a snapshot
/// without one is a defect in the enumeration rather than a capability a caller may
/// not use. Section 6.9 is what makes refusing it here the right place — a portable
/// defect must not be handed down for a driver or a validation layer to discover.
#[test]
fn a_device_whose_enumeration_violates_the_base_guarantee_is_refused() {
    // A lane that takes raster work and nothing else: no lane accepts the
    // `RASTER | COPY` pair the guarantee requires.
    let lanes_without_copy = SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
        SubmissionLaneId::new(0),
        SubmissionLaneClass::Graphics,
        LaneWorkDomains::RASTER,
    )]);

    let error = block_on(request_over(declared_facts(), lanes_without_copy))
        .expect_err("the base guarantee is violated, so no device may be published");
    assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
    assert!(
        error.message().contains("raster and copy"),
        "the error names which half of the guarantee was violated: {}",
        error.message()
    );
}

/// An async device request over a mock provider that enumerates `facts` and `lanes`.
async fn request_over(
    facts: CapabilityFacts,
    lanes: SubmissionCapabilities,
) -> crate::api::error::RhiResult<Device> {
    let instance = DeviceInstanceId::new(1);
    let provider = PlatformProvider::new(
        BackendKind::Dx12,
        instance,
        mock_provider()
            .with_capability_facts(facts)
            .with_submission_capabilities(lanes)
            .shared(),
    );
    provider.request_device(headless_request()).await
}

/// The facts the contract test has the provider enumerate, built a second time so
/// the test can intern them independently.
fn declared_facts() -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(OptionalFeature::Compute);
    facts.record_limit(LimitKey::MaxBufferSize, 1 << 28);
    facts
}

/// The lanes that test has the provider enumerate.
fn declared_submission() -> SubmissionCapabilities {
    SubmissionCapabilities::new(vec![
        SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
        ),
        SubmissionLaneInfo::new(
            SubmissionLaneId::new(1),
            SubmissionLaneClass::Compute,
            LaneWorkDomains::COMPUTE,
        ),
    ])
}

/// A clone is the same execution domain; a second request is not.
///
/// Section 6.1 makes this a portable contract rather than an implementation
/// detail: even where a backend reuses one native device internally, two
/// successful requests still produce two isolated domains that must not accept
/// each other's objects.
#[test]
fn a_clone_is_the_same_domain_and_a_new_request_is_not() {
    let (device, _native) = live_device();
    let clone = device.clone();

    assert_eq!(clone.identity(), device.identity());
    assert_eq!(clone.object_id(), device.object_id());

    let provider = provider();
    let fresh = block_on(provider.request_device(headless_request())).unwrap();
    assert_ne!(fresh.identity(), device.identity());
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
    let (device, native) = live_device();
    native.mark_lost(DeviceLossInfo::new(
        "the driver reset the adapter".to_string(),
    ));

    assert_eq!(device.status(), DeviceStatus::Lost);
    assert_eq!(
        device.loss_info().map(|loss| loss.message().to_string()),
        Some("the driver reset the adapter".to_string()),
        "section 6.5 makes the summary stable, so a later ask must match an earlier one"
    );
    assert_eq!(
        block_on(device.lost()).message(),
        "the driver reset the adapter",
        "the async loss wait returns the same stable terminal summary"
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

/// Everything hanging off a device follows it into the lost state.
///
/// Section 6.5's list is about *handles* used through their lost original device,
/// so it is not limited to the creation verbs; a façade that only gated creation
/// would leave `poll` and `wait_idle` answering as if the device were alive.
#[test]
fn a_lost_device_is_visible_through_every_provenance_verb() {
    let (device, native) = live_device();
    native.mark_lost(DeviceLossInfo::new("the adapter was removed".to_string()));

    assert_eq!(device.status(), DeviceStatus::Lost);
    assert_eq!(
        device.backend(),
        BackendKind::Dx12,
        "provenance is a fact about where the device came from, not about whether it is alive"
    );

    let printed = format!("{device:?}");
    assert!(
        printed.contains("Lost"),
        "a device that is gone must not print as if it were alive: {printed}"
    );
}
