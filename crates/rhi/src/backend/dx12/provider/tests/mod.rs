//! The DX12 provider's real-GPU evidence.
//!
//! These tests are not a mock conformance run and must not be read as one: they
//! call `CreateDXGIFactory2` and `D3D12CreateDevice` on whatever adapter the
//! machine actually has. `CLAUDE.md` section 4.8 is explicit that a `TestRhi` or
//! a mock cannot stand in for this, and section 9 is explicit about the reverse
//! substitution too — passing here proves that this backend's native path works
//! on this driver, not that the portable contract holds.
//!
//! What they deliberately do not cover, recorded rather than left to be assumed:
//! nothing is rasterized, dispatched, copied, uploaded or read back, because no
//! resource or command lowering exists yet. `version-plan.md` section 4 asks for
//! real headless raster/compute/copy/upload/readback on Windows DX12, and **that
//! requirement is not met by this module**.

use std::sync::Arc;

use super::*;
use crate::api::binding::vocabulary::StorageAccess;
use crate::api::format::TextureFormat;
use crate::api::platform::requirements::{DeviceRequirements, LimitKey, OptionalFeature};
use crate::api::platform::{PlatformProvider, RequestStatus};
use crate::api::resource::buffer::{BufferSupportQuery, BufferUsage};
use crate::api::submission::LaneWorkDomains;

/// A fresh provider instance identity, as host integration would mint.
fn instance() -> DeviceInstanceId {
    DeviceInstanceId::new(0x0D12_0001)
}

/// A provider over this machine's DXGI factory.
fn provider() -> Dx12Provider {
    Dx12Provider::new(instance()).expect("DXGI must be reachable on a Windows test host")
}

/// The same provider behind the portable façade host integration would build.
///
/// This is what makes the tests below evidence about the *portable* path rather
/// than only about this backend: the identity check, the request retirement and
/// the device handle all come from the API v1 layer.
fn portable_provider() -> PlatformProvider {
    PlatformProvider::new(
        BackendKind::Dx12,
        instance(),
        Arc::new(provider()) as Arc<dyn ProviderBackend>,
    )
}

/// Asks for a headless device satisfying nothing, which is the one request shape
/// this backend can currently honour.
fn headless_request(selection: AdapterSelection) -> DeviceRequestDescriptor {
    DeviceRequestDescriptor::new(selection, DeviceRequirements::new())
}

/// The candidate the default selection lands on, for tests that need to name one.
fn default_candidate(provider: &Dx12Provider) -> Candidate {
    provider
        .select(AdapterSelection::Default)
        .expect("a Windows host with DXGI exposes a usable adapter")
}

/// A portable device over this machine's default adapter, through the full path.
///
/// Built the way host integration builds one — the portable `PlatformProvider`
/// asked for a device, its request polled to completion — rather than by reaching
/// for the backend directly, so a test using this observes the capability facts
/// the *portable* layer would hand a caller. The capability table is filled
/// during device creation, so anything that wants to examine it has to come
/// through a created device.
fn portable_device() -> crate::api::platform::Device {
    let provider = portable_provider();
    let mut request = provider
        .request_device(headless_request(AdapterSelection::Default))
        .expect("a headless device request must succeed on a machine with DXGI");

    let RequestStatus::Ready(device) = request.poll().expect("the first poll succeeds") else {
        panic!("the DX12 path is synchronous, so the first poll must be Ready");
    };

    device
}

#[test]
fn a_provider_reports_the_adapters_dxgi_lists() {
    let candidates = provider()
        .candidates()
        .expect("adapter enumeration must not fail on a machine with DXGI");

    assert!(
        !candidates.is_empty(),
        "a Windows host with DXGI exposes at least one adapter"
    );
    for candidate in &candidates {
        assert!(
            !candidate.name.is_empty(),
            "DXGI must give a non-empty adapter description"
        );
    }

    // The mapping `Explicit` selection relies on: distinct adapters must produce
    // distinct serials, or two adapters could not be told apart by the id a
    // caller holds.
    let mut serials: Vec<u64> = candidates
        .iter()
        .map(|candidate| candidate.serial)
        .collect();
    serials.sort_unstable();
    let before = serials.len();
    serials.dedup();
    assert_eq!(before, serials.len(), "adapter serials must be distinct");
}

#[test]
fn a_headless_device_is_created_on_the_default_adapter() {
    let provider = provider();
    let mut request = provider
        .request_device(&headless_request(AdapterSelection::Default))
        .expect("a headless device request must succeed on a machine with DXGI");

    let progress = request
        .poll()
        .expect("a synchronous backend must answer on the first poll");
    let RequestProgress::Ready(device) = progress else {
        panic!("the DX12 path is synchronous, so the first poll must be Ready");
    };

    assert_eq!(device.backend_kind(), BackendKind::Dx12);
    assert_eq!(
        device.status(),
        DeviceStatus::Active,
        "a freshly created device is active"
    );
    assert!(
        device.loss_info().is_none(),
        "an active device has no loss to report"
    );
    assert!(
        !device.adapter_info().name().is_empty(),
        "the device must carry the adapter it was actually created on"
    );
    assert!(
        device.poll().is_ok(),
        "polling an idle device must not fail"
    );
    assert!(
        device.wait_idle().is_ok(),
        "a device with no submitted work is idle"
    );
}

#[test]
fn the_portable_path_produces_a_real_device_and_retires_its_request() {
    // The end-to-end shape: a portable `PlatformProvider` over this real DX12
    // backend, the portable `DeviceRequest` in the middle, and a portable
    // `Device` at the end that is backed by an actual `ID3D12Device`.
    //
    // The single-shot rule is checked *here* and not against the backend's own
    // `poll`, because section 5.9 makes it a portable contract: a backend's
    // `DeviceRequestBackend::poll` has no such rule and this one would happily
    // answer `Ready` twice. An earlier version of this test asserted it against
    // the backend and failed — correctly. The layer that owns an invariant is the
    // layer that must be asked about it.
    let provider = portable_provider();
    let mut request = provider
        .request_device(headless_request(AdapterSelection::Default))
        .expect("a headless device request must succeed on a machine with DXGI");

    let RequestStatus::Ready(device) = request.poll().expect("the first poll succeeds") else {
        panic!("the DX12 path is synchronous, so the first poll must be Ready");
    };

    assert_eq!(device.backend(), BackendKind::Dx12);
    assert_eq!(device.status(), DeviceStatus::Active);
    assert!(device.loss_info().is_none());
    assert!(
        !device.adapter_info().name().is_empty(),
        "the portable device must carry the adapter its native device was created on"
    );
    assert!(device.poll().is_ok());
    assert!(device.wait_idle().is_ok());

    // `let .. else` rather than `expect_err`: the `Ok` side holds a
    // `Box<dyn Device>`, which has no `Debug` for `expect_err` to print, and a
    // backend object deliberately has no portable rendering.
    let Err(error) = request.poll() else {
        panic!("a request is single-shot, so a second poll must be refused");
    };

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("DeviceRequest::poll"));
}

#[test]
fn a_loss_recorded_on_the_native_device_is_visible_through_the_shim() {
    // The reason `ArcDevice` exists rather than a copy: the backend's own handle
    // and the one the portable layer holds must be one device. This reaches the
    // concrete device directly because the channel that will record a loss in
    // production — a terminal `HRESULT` on a resource or command call — has no
    // lowering to be reached from yet.
    let provider = provider();
    let native = provider
        .create_native(AdapterSelection::Default)
        .expect("a headless device must be creatable");

    let shared = ArcDevice(Arc::clone(&native));
    assert_eq!(shared.status(), DeviceStatus::Active);
    assert!(shared.loss_info().is_none());

    native.mark_lost(DeviceLossInfo::new("DXGI_ERROR_DEVICE_REMOVED".to_string()));

    assert_eq!(shared.status(), DeviceStatus::Lost);
    assert_eq!(
        shared.status(),
        DeviceStatus::Lost,
        "loss is terminal: a second read must not revive it"
    );
    assert_eq!(
        shared.loss_info().map(|info| info.message().to_string()),
        Some("DXGI_ERROR_DEVICE_REMOVED".to_string()),
        "section 6.5 makes the summary stable rather than a one-shot notification"
    );
    assert_eq!(
        shared.object_id(),
        native.object_id(),
        "the shim must name the same device, not a second one"
    );
}

#[test]
fn adapter_enumeration_refuses_rather_than_publishing_a_hollow_snapshot() {
    let error = provider()
        .enumerate_adapters()
        .expect_err("a capability snapshot with holes in it is a bug, so enumeration must refuse");

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert_eq!(error.operation(), Some("Dx12Provider::enumerate_adapters"));
    assert!(
        error.message().contains("capability"),
        "the refusal must name what is missing rather than read as a generic failure: {}",
        error.message()
    );
}

#[test]
fn presentation_support_is_refused_rather_than_answered_no() {
    let provider = provider();
    let candidate = default_candidate(&provider);
    let adapter = AdapterId::new(instance().as_u64(), candidate.serial);
    let target = PresentationTarget::new(ObjectId::new(0x9001));

    let error = provider
        .supports_presentation(adapter, &target)
        .expect_err("there is no channel from an ObjectId to an HWND, so this cannot be answered");

    assert_eq!(
        error.kind(),
        RhiErrorKind::Unsupported,
        "Ok(false) would turn 'no way to ask' into 'the hardware says no'"
    );
}

#[test]
fn a_presentation_requirement_is_refused_rather_than_dropped() {
    let descriptor = headless_request(AdapterSelection::Default)
        .require_presentation_target(PresentationTarget::new(ObjectId::new(0x9002)));

    // `.err().expect(..)` rather than `expect_err`: a `DeviceRequest` holds a
    // `Box<dyn DeviceRequestBackend>`, which is deliberately not `Debug` — a
    // backend object has no portable rendering, and printing one would be the
    // native leak the seam exists to prevent.
    let error = provider().request_device(&descriptor).err().expect(
        "a device required to present cannot be created while presentation cannot be resolved",
    );

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
}

#[test]
fn a_device_requirement_is_refused_rather_than_dropped() {
    let requirements =
        DeviceRequirements::new().require_limit_at_least(LimitKey::MaxBufferSize, 1024);
    let descriptor = DeviceRequestDescriptor::new(AdapterSelection::Default, requirements);

    let error = provider()
        .request_device(&descriptor)
        .err()
        .expect("an unverifiable requirement must fail the request, not be ignored");

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert!(
        error.message().contains("requirement"),
        "the refusal must name the unmet contract: {}",
        error.message()
    );
}

#[test]
fn the_high_performance_preference_never_picks_the_software_adapter() {
    // A preference is not a guarantee — section 5.6 says so — but it must at
    // least not select the opposite of what was asked when a hardware adapter
    // exists. Which adapter that is depends on the machine, so this asserts the
    // property rather than a name.
    let provider = provider();
    let candidate = provider
        .select(AdapterSelection::PreferHighPerformance)
        .expect("a Windows host exposes a usable adapter");
    let hardware_exists = provider
        .candidates()
        .expect("enumeration must not fail")
        .iter()
        .any(|other| !other.software);

    assert!(
        !hardware_exists || !candidate.software,
        "a hardware adapter exists, so the high-performance preference must not resolve to WARP"
    );
}

#[test]
fn what_the_machine_actually_reported() {
    // Not an assertion about the hardware: `CLAUDE.md` section 4.8 requires the
    // exact backend, adapter and vendor/device pair to be recorded with the
    // evidence, and section 9 forbids recording an observation that was not made.
    // Run with `--nocapture` and this is what the run above used.
    let provider = provider();
    for candidate in provider.candidates().expect("enumeration must not fail") {
        println!(
            "dx12 evidence: name={:?} vendor=0x{:04x} device=0x{:04x} \
             dedicated_video_memory={} software={}",
            candidate.name,
            candidate.vendor,
            candidate.device,
            candidate.dedicated_video_memory,
            candidate.software
        );
    }
    assert_eq!(provider.instance(), instance());
}

/// The enumerated facts answer the one table that must be complete.
///
/// This is the test that closes the landmine `facts`' module documentation names.
/// Before the enumeration landed, `buffer_support` panicked for every query on a
/// real DX12 device, because the table was empty and its key space is one a
/// backend can walk in full. The assertions below therefore do two separate
/// things: they check the *answers*, and — by the mere fact of returning — they
/// check that a query against a real device no longer panics.
#[test]
fn a_real_device_answers_every_buffer_usage_combination() {
    let device = portable_device();

    for usage in BufferUsage::all() {
        let query = BufferSupportQuery::new(usage);
        let support = device.capabilities().buffer_support(&query);

        if usage.is_empty() {
            // Section 12.3 refuses to create a buffer with no usage bit at all,
            // so the honest answer is a refusal rather than a supported entry
            // with a ceiling nobody can reach.
            assert!(
                !support.is_supported(),
                "an empty usage set has no legal operation and must be refused"
            );
            continue;
        }

        assert!(
            support.is_supported(),
            "Direct3D 12 expresses {usage} as resource states rather than as creation \
             flags, so it must be reported creatable"
        );
        assert!(
            support.limits().is_some_and(|limits| limits.max_size() > 0),
            "a supported answer must carry a non-zero ceiling: {usage}"
        );
    }
}

/// The three optional features Direct3D 12 answers structurally are reported.
///
/// They are recorded without a probe, and the reason is at each call site in
/// `facts`. What matters for the portable contract is the consequence: a caller
/// reading this device's enabled capabilities must not be told that a D3D12
/// device cannot dispatch, cannot filter anisotropically, or cannot use binding
/// arrays. The lane assertion is the other half of the same fact — section 7.2's
/// base guarantee relates the `Compute` feature to a lane accepting compute work,
/// and the two are now consistent rather than deliberately under-reported.
#[test]
fn a_real_device_reports_the_features_direct3d_12_mandates() {
    let device = portable_device();
    let capabilities = device.capabilities();

    for feature in [
        OptionalFeature::Compute,
        OptionalFeature::SamplerAnisotropy,
        OptionalFeature::BindingArrays,
    ] {
        assert!(
            capabilities.supports_feature(feature),
            "{feature:?} is a property of Direct3D 12 rather than a driver's choice"
        );
    }

    let accepts_compute = capabilities
        .submission()
        .lanes()
        .iter()
        .any(|lane| lane.domains().contains(LaneWorkDomains::COMPUTE));

    assert!(
        accepts_compute,
        "section 7.2's base guarantee ties a compute-accepting lane to the Compute feature, \
         and this device reports both"
    );
}

/// A format table that covers the portable set, minus the two it cannot name.
///
/// The two omissions are the assertion worth reading: `Depth24Plus` and
/// `Depth24PlusStencil8` explicitly permit a driver to choose a bit layout, so
/// there is no single DXGI format that is the answer and `facts` returns none.
/// The portable accessor answers `Option`, so "not asked" and "asked and refused"
/// stay distinguishable — and this test pins which of the two a caller sees.
#[test]
fn a_real_device_reports_what_each_namable_format_can_do() {
    let device = portable_device();
    let capabilities = device.capabilities();

    // The two permitted-layout formats are the only ones this backend declines to
    // answer, and that is asserted as an exact set rather than one format at a
    // time: a third silent omission is the failure mode worth catching, and
    // checking only the two known names would not catch it.
    let unanswered: Vec<TextureFormat> = TextureFormat::all()
        .filter(|format| capabilities.format(*format).is_none())
        .collect();

    assert_eq!(
        unanswered,
        vec![
            TextureFormat::Depth24Plus,
            TextureFormat::Depth24PlusStencil8
        ],
        "exactly the two formats whose bit layout Direct3D 12 leaves to the driver \
         may go unanswered; anything else absent is a hole in the table"
    );

    // One answered format, read end to end: `None` above is only meaningful if
    // `Some` carries real facts, and `Rgba8Unorm` is the format every backend
    // must support for a storage write or the portable P0 set is not viable.
    let sampled = capabilities
        .format(TextureFormat::Rgba8Unorm)
        .expect("Rgba8Unorm is a DXGI format on every device");

    assert!(
        sampled.storage_access().supports(StorageAccess::ReadOnly),
        "an Rgba8Unorm texture is readable as a storage resource"
    );
}

/// The two limits whose Direct3D 12 source is unambiguous are recorded.
///
/// Not a claim that the limit table is complete — it is not, and `facts` records
/// which twenty-five keys are still unrecorded. This pins the two that are, so
/// that a later mapping change cannot quietly drop them.
#[test]
fn a_real_device_reports_the_two_limits_it_can_ground() {
    let device = portable_device();
    let capabilities = device.capabilities();

    assert_eq!(
        capabilities.limit(LimitKey::MaxSamplerAnisotropy),
        Some(16),
        "the D3D12 sampler descriptor clamps MaxAnisotropy to 1..=16"
    );

    let max_buffer = capabilities
        .limit(LimitKey::MaxBufferSize)
        .expect("the resource address space is always reported");
    assert!(
        max_buffer >= (1 << 32),
        "a D3D12 device addresses at least 32 bits per resource; got {max_buffer}"
    );
}

/// Records what this machine's device enumerated, for the evidence binding.
///
/// Run with `--nocapture`. `CLAUDE.md` section 9 forbids recording an observation
/// that was not made, so this prints rather than asserts a hardware-specific
/// value; the assertions that do bind are in the tests above.
#[test]
fn what_the_capability_enumeration_actually_reported() {
    let device = portable_device();
    let capabilities = device.capabilities();

    println!(
        "dx12 capability evidence: backend={:?} adapter={:?} id={:?} fingerprint={:?}",
        device.backend(),
        device.adapter_info().name(),
        capabilities.compatibility_id(),
        capabilities.fingerprint(),
    );

    for format in TextureFormat::all() {
        let Some(facts) = capabilities.format(format) else {
            continue;
        };
        println!(
            "dx12 format {format:?}: storage read={} write={} read_write={}",
            facts.storage_access().supports(StorageAccess::ReadOnly),
            facts.storage_access().supports(StorageAccess::WriteOnly),
            facts.storage_access().supports(StorageAccess::ReadWrite),
        );
    }

    println!(
        "dx12 limits: max_buffer_size={:?} max_sampler_anisotropy={:?}",
        capabilities.limit(LimitKey::MaxBufferSize),
        capabilities.limit(LimitKey::MaxSamplerAnisotropy),
    );
}
