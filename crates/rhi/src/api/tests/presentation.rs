//! Contract tests for the presentation chapter (specification sections 42 to 45).
//!
//! The review instrument for the surface side of the frame loop: which
//! configuration a target must refuse and with what kind, what an acquire refusal
//! is, and what a frame's lifecycle state permits. As with the sibling chapter,
//! there is no hardware behind these tests and none of them may be presented as
//! GPU evidence — a browser canvas, a swapchain, and a `CAMetalLayer` are all
//! absent by design, which is exactly why the *rules* have to be testable without
//! them.
//!
//! ```text
//! surface facts are one device plus one target        (42)
//! one active lease per target, one frame per lease    (42.1, 43.4)
//! a FrameAttachment is not a Texture                  (44.3)
//! acquire refusal, GPU completion, present outcome    (45.5)
//! ```
//!
//! Almost every rule here is decided from its arguments alone, so almost every one
//! of them is exercised through a `pub(crate) validate_*` function taking its facts
//! as parameters. The verbs that must reach a platform surface are exercised only
//! where they refuse before doing so; the rest of the loop is written out as
//! `shape_*` call sites — compiled, never called — which are the review instrument
//! for the interface rather than for the behaviour.

use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::platform::{Device, DeviceLossInfo};
use crate::api::presentation::{
    AcquireErrorKind, AcquiredFrame, AcquiredFrameId, AcquiredFrameState, CompositeAlphaMode,
    ConfiguredPresentation, DisplayHdrInfo, Extent2d, FrameAttachment, FrameLatencyRange,
    PresentFailure, PresentMode, PresentPlanId, PresentReceipt, PresentReceiptId, PresentState,
    PresentationColorSpace, PresentationConfiguration, PresentationExtent,
    PresentationExtentControl, PresentationFormat, PresentationTarget,
    PresentationTargetCapabilities, PresentationTimingCapabilities, configure, frame, target,
};
use crate::api::resource::texture::{Extent3d, TextureUsage};
use crate::api::submission::{CompletionPoint, SubmissionPlanId, SubmissionPoint};
use crate::api::tests::mock::paired_device_for_test;

// ---------------------------------------------------------------------------
// Section 43 — configuration.
// ---------------------------------------------------------------------------

/// The constructor spells every portable default explicitly. `Automatic` and a
/// host-managed extent are universally meaningful; the remaining baseline
/// choices are still checked against the target snapshot at configuration time.
#[test]
fn configuration_defaults_ask_for_nothing_a_target_must_refuse() {
    let config = PresentationConfiguration::new(TextureFormat::Bgra8Unorm);

    assert_eq!(config.format(), TextureFormat::Bgra8Unorm);
    assert_eq!(config.present_mode(), PresentMode::Automatic);
    assert_eq!(config.extent(), PresentationExtent::HostManaged);
    assert_eq!(config.color_space(), PresentationColorSpace::Srgb);
    assert_eq!(config.maximum_frame_latency(), 2);
    assert_eq!(config.composite_alpha_mode(), CompositeAlphaMode::Automatic);
    assert_eq!(config.usage(), TextureUsage::COLOR_ATTACHMENT);
    assert!(config.view_formats().is_empty());

    let asked =
        config
            .with_present_mode(PresentMode::Mailbox)
            .with_extent(PresentationExtent::Exact(Extent2d {
                width: 1280,
                height: 720,
            }));

    assert_eq!(asked.present_mode(), PresentMode::Mailbox);
    assert_eq!(
        asked.extent(),
        PresentationExtent::Exact(Extent2d {
            width: 1280,
            height: 720
        })
    );
    assert_eq!(
        asked.format(),
        TextureFormat::Bgra8Unorm,
        "the two `with_` methods leave the format the constructor was given"
    );
}

#[test]
fn acquired_frame_reports_suboptimal_without_changing_ownership() {
    let identity = device_identity(1);
    let regular = AcquiredFrame::new(
        AcquiredFrameId::new(identity, 1),
        identity,
        TextureFormat::Bgra8Unorm,
        Extent3d::d2(1, 1),
    );
    let suboptimal = AcquiredFrame::new_suboptimal(
        AcquiredFrameId::new(identity, 2),
        identity,
        TextureFormat::Bgra8Unorm,
        Extent3d::d2(1, 1),
    );
    assert!(!regular.suboptimal());
    assert!(suboptimal.suboptimal());
    assert_eq!(suboptimal.state(), AcquiredFrameState::Acquired);
}

#[test]
fn presentation_clock_has_supported_unsupported_and_loss_boundaries() {
    let (device, native) = paired_device_for_test(device_identity(41));
    let target = PresentationTarget::new(ObjectId::new(9));

    assert_eq!(
        device.presentation_timestamp(&target).unwrap_err().kind(),
        RhiErrorKind::Unsupported,
        "a target that did not report timing must be refused before sampling"
    );

    native.set_presentation_timing(true);
    let capabilities = device.presentation_capabilities(&target).unwrap();
    assert!(capabilities.timing().timestamps);
    assert_eq!(
        device.presentation_timestamp(&target).unwrap(),
        crate::api::presentation::PresentationTimestamp {
            value: 42,
            period_nanos: 0.5,
        }
    );

    native.mark_lost(DeviceLossInfo::new("presentation clock loss".into()));
    assert_eq!(
        device.presentation_timestamp(&target).unwrap_err().kind(),
        RhiErrorKind::DeviceLost
    );
    assert_eq!(
        device
            .presentation_capabilities(&target)
            .unwrap_err()
            .kind(),
        RhiErrorKind::DeviceLost,
        "surface fact queries are still Device operations after terminal loss"
    );
}

/// A format the target never reported is a capability gap, so it is `Unsupported`
/// rather than a usage error.
#[test]
fn a_format_the_target_never_reported_is_unsupported() {
    let caps = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        host_managed(Some((640, 480))),
    );

    assert!(
        configure::validate_presentation_configuration(
            &PresentationConfiguration::new(TextureFormat::Bgra8Unorm),
            &caps
        )
        .is_ok()
    );

    let error = configure::validate_presentation_configuration(
        &PresentationConfiguration::new(TextureFormat::Rgba16Float),
        &caps,
    )
    .unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
}

#[test]
fn complete_surface_configuration_accepts_reported_values() {
    let caps = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        host_managed(Some((640, 480))),
    )
    .with_format_color_spaces(vec![PresentationFormat {
        format: TextureFormat::Bgra8Unorm,
        color_space: PresentationColorSpace::DisplayP3,
    }])
    .with_surface_details(
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        vec![CompositeAlphaMode::Opaque],
        Some(FrameLatencyRange { min: 1, max: 3 }),
        vec![TextureFormat::Bgra8UnormSrgb],
    )
    .with_timing_and_hdr(
        PresentationTimingCapabilities { timestamps: true },
        Some(DisplayHdrInfo {
            min_luminance_nits: 0.01,
            max_luminance_nits: 1_000.0,
            max_full_frame_luminance_nits: 400.0,
        }),
    );
    assert!(caps.timing().timestamps);
    assert_eq!(caps.hdr_info().unwrap().max_luminance_nits, 1_000.0);
    for latency in [1, 3] {
        let config = PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
            .with_color_space(PresentationColorSpace::DisplayP3)
            .with_present_mode(PresentMode::Fifo)
            .with_maximum_frame_latency(latency)
            .with_composite_alpha_mode(CompositeAlphaMode::Opaque)
            .with_usage(TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC))
            .with_view_formats([TextureFormat::Bgra8UnormSrgb]);
        assert!(configure::validate_presentation_configuration(&config, &caps).is_ok());
    }
}

#[test]
fn surface_configuration_rejects_unreported_independent_facts() {
    let caps = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![],
        host_managed(Some((640, 480))),
    )
    .with_surface_details(
        TextureUsage::COLOR_ATTACHMENT,
        vec![CompositeAlphaMode::Opaque],
        Some(FrameLatencyRange { min: 1, max: 3 }),
        vec![],
    );

    let unsupported = [
        PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
            .with_color_space(PresentationColorSpace::Hdr10),
        PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
            .with_composite_alpha_mode(CompositeAlphaMode::PreMultiplied),
        PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
            .with_usage(TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC)),
        PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
            .with_view_formats([TextureFormat::Bgra8UnormSrgb]),
    ];
    for config in unsupported {
        assert_eq!(
            configure::validate_presentation_configuration(&config, &caps)
                .unwrap_err()
                .kind(),
            RhiErrorKind::Unsupported
        );
    }
}

#[test]
fn frame_latency_zero_and_values_past_the_reported_edge_are_invalid() {
    let caps = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![],
        host_managed(Some((640, 480))),
    )
    .with_surface_details(
        TextureUsage::COLOR_ATTACHMENT,
        vec![CompositeAlphaMode::Opaque],
        Some(FrameLatencyRange { min: 1, max: 3 }),
        vec![],
    );
    for latency in [0, 4, u32::MAX] {
        let config = PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
            .with_maximum_frame_latency(latency);
        assert_eq!(
            configure::validate_presentation_configuration(&config, &caps)
                .unwrap_err()
                .kind(),
            RhiErrorKind::InvalidUsage
        );
    }
}

/// `Automatic` needs no report and every other mode does — the one rule section
/// 42.2 freezes about present modes.
#[test]
fn automatic_needs_no_report_and_the_other_modes_do() {
    let silent = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        Vec::new(),
        host_managed(Some((640, 480))),
    );

    assert!(
        configure::validate_presentation_configuration(
            &PresentationConfiguration::new(TextureFormat::Bgra8Unorm),
            &silent
        )
        .is_ok(),
        "Automatic is the one mode every presentation backend must support"
    );

    for mode in [
        PresentMode::Fifo,
        PresentMode::Mailbox,
        PresentMode::Immediate,
    ] {
        let config =
            PresentationConfiguration::new(TextureFormat::Bgra8Unorm).with_present_mode(mode);
        assert_eq!(
            configure::validate_presentation_configuration(&config, &silent)
                .unwrap_err()
                .kind(),
            RhiErrorKind::Unsupported,
            "a mode the target never reported is a capability gap"
        );
    }

    let reported = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        host_managed(Some((640, 480))),
    );
    assert!(
        configure::validate_presentation_configuration(
            &PresentationConfiguration::new(TextureFormat::Bgra8Unorm)
                .with_present_mode(PresentMode::Fifo),
            &reported
        )
        .is_ok(),
        "a mode the target did report is accepted"
    );
}

/// An exact extent is legal only where the target says the RHI may choose one, and
/// out of the reported bounds or zero-sized it is the caller's own range error.
#[test]
fn an_exact_extent_needs_a_configurable_target_and_a_legal_size() {
    let exact = PresentationConfiguration::new(TextureFormat::Bgra8Unorm).with_extent(
        PresentationExtent::Exact(Extent2d {
            width: 640,
            height: 480,
        }),
    );

    let host = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        host_managed(Some((640, 480))),
    );
    assert_eq!(
        configure::validate_presentation_configuration(&exact, &host)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage,
        "this target's drawable size belongs to the host"
    );

    let configurable_target = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        configurable(16, 16, 4096, 4096),
    );
    assert!(
        configure::validate_presentation_configuration(&exact, &configurable_target).is_ok(),
        "within the reported bounds, an exact extent is legal"
    );

    let too_wide = PresentationConfiguration::new(TextureFormat::Bgra8Unorm).with_extent(
        PresentationExtent::Exact(Extent2d {
            width: 8192,
            height: 480,
        }),
    );
    assert_eq!(
        configure::validate_presentation_configuration(&too_wide, &configurable_target)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );

    let zero = PresentationConfiguration::new(TextureFormat::Bgra8Unorm).with_extent(
        PresentationExtent::Exact(Extent2d {
            width: 0,
            height: 480,
        }),
    );
    assert_eq!(
        configure::validate_presentation_configuration(&zero, &configurable_target)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

/// A host-managed request over a currently zero-sized surface is `TargetOutdated`:
/// nothing about the request is wrong and the surface has gone to zero size.
///
/// That is what keeps it apart from the `InvalidUsage` above — blaming the caller
/// for a minimized window would send them looking for a mistake they did not make
/// — and it is reported rather than guessed at: a target that cannot report a size
/// at all answers `None`, which is a different fact and is not refused.
#[test]
fn a_host_managed_request_over_a_zero_sized_surface_is_outdated() {
    let config = PresentationConfiguration::new(TextureFormat::Bgra8Unorm);

    let suspended = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        host_managed(Some((0, 0))),
    );
    assert_eq!(
        configure::validate_presentation_configuration(&config, &suspended)
            .unwrap_err()
            .kind(),
        RhiErrorKind::TargetOutdated
    );

    // Section 42.5 makes `None` a fact about this query, not a promise about later
    // ones, so a target that could not answer is not a zero-sized surface.
    let silent = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        host_managed(None),
    );
    assert!(configure::validate_presentation_configuration(&config, &silent).is_ok());

    // A host-managed request against a configurable target asks for nothing the
    // target has to agree to.
    let configurable_target = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        configurable(16, 16, 4096, 4096),
    );
    assert!(configure::validate_presentation_configuration(&config, &configurable_target).is_ok());
}

/// Section 43.4's two preconditions, which are the portable half of acquire and
/// reconfigure.
#[test]
fn an_outstanding_frame_blocks_both_the_next_acquire_and_a_reconfigure() {
    assert!(configure::validate_acquire_allowed(None).is_ok());
    assert!(configure::validate_reconfigure_allowed(None).is_ok());

    let frame = AcquiredFrameId::new(device_identity(1), 1);

    assert_eq!(
        configure::validate_acquire_allowed(Some(frame))
            .unwrap_err()
            .kind(),
        AcquireErrorKind::FrameOutstanding
    );
    assert_eq!(
        configure::validate_reconfigure_allowed(Some(frame))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage,
        "the frame that is out was acquired against the old configuration"
    );
}

/// A lease is the only thing a frame can be acquired from, and the record of the
/// frame it has out is what makes section 43.4's refusal decidable at all.
///
/// Built through the crate-private constructor the backend port will use, because a
/// lease exists only because a device configured a surface.
#[test]
fn a_lease_keeps_the_frame_it_has_outstanding_and_its_own_identity() {
    let device = device_identity(1);
    let target_id = ObjectId::new(4);
    let config = PresentationConfiguration::new(TextureFormat::Bgra8Unorm);
    let lease =
        ConfiguredPresentation::new_for_test(ObjectId::new(7), device, target_id, config.clone());

    // A reconfiguration is the same lease rather than a second one, so the identity
    // distinguishes this from dropping the lease and configuring again.
    assert_eq!(lease.device_identity(), device);
    assert_eq!(lease.target_id(), target_id);
    assert_eq!(lease.configuration().format(), config.format());
    assert_ne!(lease.id(), target_id);
    assert_eq!(lease.outstanding_frame(), None);

    let frame = AcquiredFrameId::new(device, 1);
    lease.set_outstanding_frame(Some(frame));
    assert_eq!(lease.outstanding_frame(), Some(frame));
    assert_eq!(
        configure::validate_acquire_allowed(lease.outstanding_frame())
            .unwrap_err()
            .kind(),
        AcquireErrorKind::FrameOutstanding,
        "the refusal is answered from the lease's own record, not from a guess"
    );

    // The next acquire is allowed only once that frame is terminal — presented,
    // abandoned, or terminated by target or device loss.
    lease.set_outstanding_frame(None);
    assert!(configure::validate_acquire_allowed(lease.outstanding_frame()).is_ok());

    assert!(
        format!("{lease:?}").contains("ConfiguredPresentation"),
        "a lease prints portable identity rather than the platform surface"
    );
}

/// Configuring a lost device is terminal, and the check runs before anything
/// native is touched.
#[test]
fn configure_presentation_on_a_lost_device_is_device_lost() {
    let (device, native) = paired_device_for_test(device_identity(1));
    native.mark_lost(DeviceLossInfo::new("simulated loss".into()));

    let target = PresentationTarget::new(ObjectId::new(1));
    let config = PresentationConfiguration::new(TextureFormat::Bgra8Unorm);

    let error = block_on(device.configure_presentation(&target, &config)).unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::DeviceLost);
    assert_eq!(error.operation(), Some("Device::configure_presentation"));
    assert_eq!(
        device.loss_info().map(|loss| loss.message().to_owned()),
        Some("simulated loss".to_owned()),
        "the loss summary is stable, so a refusal can say why"
    );
}

/// An acquire that suspended waiting for the host must not strand its future
/// when the device goes away.  This deliberately drives the public lease
/// future through the backend waker seam rather than relying on a scheduler
/// yield: loss is an event, so it must wake the task that is waiting for one.
#[test]
fn pending_acquire_is_woken_and_terminates_as_device_lost() {
    use core::future::Future;
    use core::task::{Context, Poll, Waker};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    struct WakeCounter(AtomicUsize);

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let backend = MockPendingAcquireBackend::new();
    let mut lease = ConfiguredPresentation::new(
        ObjectId::new(7),
        device_identity(1),
        ObjectId::new(8),
        PresentationConfiguration::new(TextureFormat::Bgra8Unorm),
        Box::new(backend.clone()),
    );
    let wakes = Arc::new(WakeCounter(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wakes));
    let mut context = Context::from_waker(&waker);
    let mut acquire = Box::pin(lease.acquire());

    assert!(matches!(acquire.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);

    backend.mark_device_lost();
    assert_eq!(
        wakes.0.load(Ordering::SeqCst),
        1,
        "device loss must wake the acquire future that registered for host progress"
    );

    let refusal = match acquire.as_mut().poll(&mut context) {
        Poll::Ready(Err(refusal)) => refusal,
        Poll::Ready(Ok(_)) => panic!("a lost device must not hand out a drawable"),
        Poll::Pending => panic!("a loss must terminally resolve a pending acquire"),
    };
    assert_eq!(refusal.kind(), AcquireErrorKind::DeviceLost);
    drop(acquire);

    assert_eq!(
        lease.try_acquire().unwrap_err().kind(),
        AcquireErrorKind::DeviceLost,
        "later acquire attempts remain terminal after loss"
    );
    assert_eq!(
        block_on(lease.reconfigure(&PresentationConfiguration::new(TextureFormat::Bgra8Unorm)))
            .unwrap_err()
            .kind(),
        RhiErrorKind::DeviceLost,
        "the same configured lease cannot be revived after loss"
    );
}

// ---------------------------------------------------------------------------
// Section 44.1 — the acquire refusal vocabulary.
// ---------------------------------------------------------------------------

/// A refused acquire is its own channel, and every kind has a stable name — which
/// is what lets a caller log one and branch on the other.
#[test]
fn an_acquire_refusal_is_printable_and_named() {
    let kinds = [
        AcquireErrorKind::NotReady,
        AcquireErrorKind::Timeout,
        AcquireErrorKind::FrameOutstanding,
        AcquireErrorKind::ZeroSizeOrSuspended,
        AcquireErrorKind::Outdated,
        AcquireErrorKind::TargetLost,
        AcquireErrorKind::DeviceLost,
        AcquireErrorKind::OutOfMemory,
    ];

    for kind in kinds {
        let error = frame::AcquireError::new(kind, "test refusal");
        assert_eq!(error.kind(), kind);
        assert_eq!(error.message(), "test refusal");
        assert_eq!(
            error.to_string(),
            format!("{}: test refusal", kind.as_str())
        );
        assert!(
            !kind.as_str().is_empty(),
            "a refusal a caller cannot name is a refusal they cannot log"
        );

        // The shape an application needs, and the reason `Display` and `Error` are
        // implemented at all: forwarding into a caller's own boxed error type.
        let boxed: Box<dyn std::error::Error> = Box::new(error);
        assert!(boxed.to_string().contains("test refusal"));
    }
}

/// Section 44.1 fixes eight names, and the two a frame loop treats as ordinary are
/// as distinct as the two terminal ones.
#[test]
fn not_ready_and_timeout_are_distinct_kinds() {
    assert_ne!(AcquireErrorKind::NotReady, AcquireErrorKind::Timeout);
    assert_ne!(AcquireErrorKind::TargetLost, AcquireErrorKind::DeviceLost);
    assert_eq!(AcquireErrorKind::NotReady.as_str(), "NotReady");
    assert_eq!(
        AcquireErrorKind::ZeroSizeOrSuspended.as_str(),
        "ZeroSizeOrSuspended"
    );
    assert_eq!(AcquireErrorKind::OutOfMemory.as_str(), "OutOfMemory");
}

// ---------------------------------------------------------------------------
// Sections 44.2 to 44.6 — the frame and its attachment.
// ---------------------------------------------------------------------------

/// A frame starts acquired, and its attachment describes the drawable without
/// conferring ownership.
#[test]
fn a_frame_starts_acquired_and_its_attachment_describes_the_drawable() {
    let device = device_identity(1);
    let frame = acquired_frame(3, device, 1280, 720);

    assert_eq!(frame.state(), AcquiredFrameState::Acquired);
    assert_eq!(frame.device_identity(), device);

    let attachment = frame.attachment();
    assert_eq!(attachment.frame_id(), frame.id());
    assert_eq!(attachment.device_identity(), device);
    assert_eq!(attachment.format(), TextureFormat::Bgra8Unorm);
    assert_eq!(attachment.extent(), Extent3d::d2(1280, 720));
    assert_eq!(
        attachment.sample_count(),
        1,
        "P0 has no multisampled presentation frame"
    );

    // A second attachment is a second view of the same drawable, and neither owns
    // it — which is why the frame's state can change under one that is still held,
    // and why every use is validated at record time.
    let another = frame.attachment();
    assert_eq!(another.frame_id(), attachment.frame_id());
    assert_eq!(frame.state(), AcquiredFrameState::Acquired);
}

/// Section 45.1's state change: `mark_planned_for_present` is what moves a frame to
/// `PlannedForPresent`, and a frame enters one plan.
///
/// It is crate-private because only the plan builder consumes a frame, so this
/// exercises it directly rather than through `present_after`; the builder's own
/// half of the rule is in `tests::submission`.
#[test]
fn a_planned_frame_cannot_be_planned_again_or_abandoned() {
    let device = device_identity(1);
    let mut frame = acquired_frame(1, device, 64, 64);

    assert!(frame.mark_planned_for_present().is_ok());
    assert_eq!(frame.state(), AcquiredFrameState::PlannedForPresent);

    assert_eq!(
        frame.mark_planned_for_present().unwrap_err().kind(),
        RhiErrorKind::InvalidUsage,
        "a frame enters one present plan"
    );

    // `abandon` consumes the token it refuses, so this is the last thing this test
    // can do with the frame — and the refusal is the point: a plan being built is
    // abandoned as a whole, not frame by frame.
    assert_eq!(
        block_on(frame.abandon()).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
}

/// Section 44.5 requires the no-throw path to exist because `Drop` cannot return an
/// error, so both states a caller can be in when it drops a frame must complete
/// silently.
///
/// The state a drop reaches is not observable from here — the token is gone — so
/// this states what is: the state the drop acts on, and that neither path panics or
/// reaches for a channel that does not exist yet.
#[test]
fn dropping_a_frame_performs_no_throw_abandonment() {
    let device = device_identity(1);

    // Section 44.5: acquired, never presented, never abandoned.
    let unplanned = acquired_frame(1, device, 64, 64);
    assert_eq!(unplanned.state(), AcquiredFrameState::Acquired);
    drop(unplanned);

    // Section 41.9: consumed by a plan that was then dropped without being
    // submitted. If this path did not exist the frame would be retained for the
    // life of the process, because nothing else holds it.
    let mut planned = acquired_frame(2, device, 64, 64);
    planned.mark_planned_for_present().unwrap();
    assert_eq!(planned.state(), AcquiredFrameState::PlannedForPresent);
    drop(planned);
}

/// Section 44.6's whole table, decided from the facts the recorder holds.
#[test]
fn a_frame_attachment_use_follows_the_frame_state_table() {
    let device = device_identity(1);
    let other = device_identity(2);
    let attachment = acquired_frame(1, device, 64, 64).attachment();

    assert!(
        frame::validate_frame_attachment_use(
            &attachment,
            AcquiredFrameState::Acquired,
            device,
            false
        )
        .is_ok()
    );
    assert!(
        frame::validate_frame_attachment_use(
            &attachment,
            AcquiredFrameState::PlannedForPresent,
            device,
            true
        )
        .is_ok(),
        "the closure that will present the frame may still record against it"
    );

    let refusals = [
        (
            AcquiredFrameState::PlannedForPresent,
            false,
            RhiErrorKind::InvalidUsage,
        ),
        (
            AcquiredFrameState::Abandoned,
            false,
            RhiErrorKind::InvalidUsage,
        ),
        (
            AcquiredFrameState::PresentAccepted,
            false,
            RhiErrorKind::InvalidUsage,
        ),
        (
            AcquiredFrameState::Outdated,
            false,
            RhiErrorKind::TargetOutdated,
        ),
        (
            AcquiredFrameState::TargetLost,
            false,
            RhiErrorKind::TargetLost,
        ),
        (
            AcquiredFrameState::DeviceLost,
            false,
            RhiErrorKind::DeviceLost,
        ),
    ];

    for (state, in_current_closure, expected) in refusals {
        assert_eq!(
            frame::validate_frame_attachment_use(&attachment, state, device, in_current_closure)
                .unwrap_err()
                .kind(),
            expected,
            "state {state:?} must be refused with its own kind"
        );
    }

    // Another device is refused before the state is considered: section 3.3 gives
    // P0 no implicit copy or staging bridge between devices.
    assert_eq!(
        frame::validate_frame_attachment_use(
            &attachment,
            AcquiredFrameState::Acquired,
            other,
            false
        )
        .unwrap_err()
        .kind(),
        RhiErrorKind::WrongDevice
    );
}

// ---------------------------------------------------------------------------
// Sections 42.4 and 45 — surface facts, present identity, present outcome.
// ---------------------------------------------------------------------------

/// The surface-facts snapshot is what section 42.4 froze, and nothing more.
#[test]
fn a_snapshot_reports_only_what_section_42_froze() {
    let caps = target_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        configurable(16, 16, 4096, 2160),
    );

    assert_eq!(caps.formats(), &[TextureFormat::Bgra8Unorm]);
    assert_eq!(caps.present_modes(), &[PresentMode::Fifo]);
    match caps.extent_control() {
        PresentationExtentControl::Configurable { min, max } => {
            assert_eq!(
                min,
                Extent2d {
                    width: 16,
                    height: 16
                }
            );
            assert_eq!(
                max,
                Extent2d {
                    width: 4096,
                    height: 2160
                }
            );
        }
        PresentationExtentControl::HostManaged { .. } => {
            panic!("this snapshot reported a configurable extent")
        }
    }
}

/// An empty format list is a real answer — this target offers nothing the RHI can
/// present in — and not a placeholder for "unknown".
#[test]
fn an_empty_format_list_refuses_every_configuration() {
    let caps = target_caps(Vec::new(), Vec::new(), host_managed(Some((640, 480))));
    let error = configure::validate_presentation_configuration(
        &PresentationConfiguration::new(TextureFormat::Bgra8Unorm),
        &caps,
    )
    .unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
}

/// A present identity names one presentation inside one plan, and the receipt
/// names it back.
#[test]
fn a_present_receipt_reports_the_presentation_it_is_about() {
    let device = device_identity(1);
    let plan = SubmissionPlanId::new(device, 1);
    let present = PresentPlanId::new(plan, 0);
    let receipt = PresentReceipt::new(PresentReceiptId::new(device, 5), present);

    assert_eq!(receipt.plan_id(), present);
    assert_eq!(receipt.id().device_identity(), device);
    assert_eq!(
        present.plan(),
        plan,
        "a present identity is plan-local: it is minted by the plan that presents"
    );
    assert_eq!(
        present.local(),
        0,
        "the index orders and counts the plan's presentations"
    );
    assert!(
        format!("{receipt:?}").contains("PresentReceipt"),
        "a receipt prints which plan it is about rather than what the driver was handed"
    );
}

/// A present that failed carries its reason, which is what section 45.4 adds so a
/// failed presentation is not just a state with no explanation.
#[test]
fn a_failed_presentation_carries_its_reason() {
    let failure = PresentFailure::new("the present system refused the image");
    assert_eq!(failure.message(), "the present system refused the image");

    let state = PresentState::Failed(failure);
    if let PresentState::Failed(reason) = state {
        assert_eq!(reason.message(), "the present system refused the image");
    } else {
        panic!("this state was built as a failure");
    }
}

/// Asking about another device's presentation is always `WrongDevice`.
#[test]
fn present_state_refuses_a_foreign_receipt_before_backend_state() {
    let identity = device_identity(1);
    let other = device_identity(2);
    let (device, native) = paired_device_for_test(identity);

    assert_eq!(
        device
            .present_state(PresentReceiptId::new(other, 1))
            .unwrap_err()
            .kind(),
        RhiErrorKind::WrongDevice,
        "another device has no such presentation"
    );

    native.mark_lost(DeviceLossInfo::new("simulated loss".into()));

    assert_eq!(
        device
            .present_state(PresentReceiptId::new(other, 1))
            .unwrap_err()
            .kind(),
        RhiErrorKind::WrongDevice,
        "identity is validated before receipt-specific terminal state"
    );
}

// ---------------------------------------------------------------------------
// Shape tests.
//
// Compiled, never called. This is the chapter as a caller writes it — the query,
// the lease, the frame, the attachment, and the outcomes — and it is the review
// instrument for the *interface*: if a step needs an extra construction, a
// lifetime, or a state precondition a caller cannot satisfy, this stops compiling
// and the fault is the interface's.
// ---------------------------------------------------------------------------

#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
async fn shape_frame_loop_through_presentation(
    device: &Device,
    target: &PresentationTarget,
    receipt: PresentReceiptId,
) -> Result<(), frame::AcquireError> {
    let caps = device
        .presentation_capabilities(target)
        .expect("a live target answers");
    let format = *caps.formats().first().expect("one presentable format");
    let config = PresentationConfiguration::new(format);

    // Section 42.5 makes the snapshot a query-time answer, so the configuration is
    // checked against it rather than trusted — and the check is a free function
    // precisely so a caller can ask before leasing.
    let _ = configure::validate_presentation_configuration(&config, &caps);

    let mut lease = device
        .configure_presentation(target, &config)
        .await
        .expect("the surface is still there");

    let frame = match lease.try_acquire()? {
        Some(frame) => frame,
        None => lease.acquire().await?,
    };
    let attachment: FrameAttachment = frame.attachment();
    let _ = (attachment.format(), attachment.extent());

    // Every state is nameable without a wildcard, so a caller can act on the one
    // it got without guessing at a representation it does not own.
    match frame.state() {
        AcquiredFrameState::Acquired => {}
        AcquiredFrameState::PlannedForPresent => {}
        AcquiredFrameState::PresentAccepted => {}
        AcquiredFrameState::Abandoned => {}
        AcquiredFrameState::Outdated => {}
        AcquiredFrameState::TargetLost => {}
        AcquiredFrameState::DeviceLost => {}
    }

    // Reconfiguring is a request like any other: it may be refused because the
    // surface moved, and the lease keeps its identity either way.
    let _ = lease.reconfigure(&config).await;
    let _ = lease.outstanding_frame();
    let _ = device.present_state(receipt);
    let _ = device.wait_present(receipt).await;
    Ok(())
}

/// Section 45.5's three outcomes are three different queries: acceptance is a
/// `SubmissionPoint`, GPU completion is a `CompletionPoint`, and frame ownership is
/// a `PresentReceipt`. A caller that has waited for one has not observed the other
/// two, and neither is derivable from the other.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_three_independent_outcomes(
    device: &Device,
    receipt: &crate::api::submission::SubmissionReceipt,
    accepted: SubmissionPoint,
    completed: CompletionPoint,
) {
    let _ = device.completion_state(completed);
    let _ = accepted.device_identity();

    for present in receipt.presents() {
        match device.present_state(present.id()) {
            Ok(PresentState::Pending) => {}
            Ok(PresentState::Accepted) => {}
            Ok(PresentState::Outdated) => {}
            Ok(PresentState::TargetLost) => {}
            Ok(PresentState::DeviceLost(loss)) => {
                let _ = loss.message();
            }
            Ok(PresentState::Failed(failure)) => {
                let _ = failure.message();
            }
            Err(error) => {
                let _ = error.kind();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test backends and helpers.
// ---------------------------------------------------------------------------

/// In-memory host acquire used to verify the configured-presentation waker
/// contract. It intentionally never produces a drawable: the test is about the
/// pending-to-loss transition before a drawable can be owned.
#[derive(Clone)]
struct MockPendingAcquireBackend {
    state: std::sync::Arc<std::sync::Mutex<MockPendingAcquireState>>,
}

struct MockPendingAcquireState {
    device_lost: bool,
    waiters: Vec<std::task::Waker>,
}

impl MockPendingAcquireBackend {
    fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(MockPendingAcquireState {
                device_lost: false,
                waiters: Vec::new(),
            })),
        }
    }

    fn mark_device_lost(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.device_lost = true;
        let waiters = core::mem::take(&mut state.waiters);
        drop(state);
        for waker in waiters {
            waker.wake();
        }
    }

    fn device_lost(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .device_lost
    }

    fn device_lost_refusal() -> frame::AcquireError {
        frame::AcquireError::new(
            AcquireErrorKind::DeviceLost,
            "the mock device was lost while waiting for a drawable",
        )
    }
}

impl crate::api::presentation::backend::ConfiguredPresentationBackend
    for MockPendingAcquireBackend
{
    fn capabilities(&self) -> crate::api::error::RhiResult<PresentationTargetCapabilities> {
        if self.device_lost() {
            return Err(crate::api::error::RhiError::new(
                RhiErrorKind::DeviceLost,
                "the mock device was lost",
            ));
        }
        Ok(target_caps(
            vec![TextureFormat::Bgra8Unorm],
            vec![PresentMode::Fifo],
            host_managed(Some((64, 64))),
        ))
    }

    fn reconfigure_or_register_waker(
        &self,
        _: &PresentationConfiguration,
        _: &std::task::Waker,
    ) -> std::task::Poll<crate::api::error::RhiResult<()>> {
        std::task::Poll::Ready(if self.device_lost() {
            Err(crate::api::error::RhiError::new(
                RhiErrorKind::DeviceLost,
                "the mock device was lost",
            ))
        } else {
            Ok(())
        })
    }

    fn try_acquire(
        &self,
        _: DeviceIdentity,
    ) -> Result<Option<crate::api::presentation::backend::AcquiredSurfaceFrame>, frame::AcquireError>
    {
        if self.device_lost() {
            Err(Self::device_lost_refusal())
        } else {
            Ok(None)
        }
    }

    fn acquire_or_register_waker(
        &self,
        _: DeviceIdentity,
        waker: &std::task::Waker,
    ) -> std::task::Poll<
        Result<crate::api::presentation::backend::AcquiredSurfaceFrame, frame::AcquireError>,
    > {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.device_lost {
            return std::task::Poll::Ready(Err(Self::device_lost_refusal()));
        }
        state.waiters.push(waker.clone());
        // The registration and this recheck share the same lock, so loss cannot
        // occur in between and leave the future asleep.
        if state.device_lost {
            std::task::Poll::Ready(Err(Self::device_lost_refusal()))
        } else {
            std::task::Poll::Pending
        }
    }

    fn abandon(&self, _: AcquiredFrameId) -> crate::api::error::RhiResult<()> {
        Ok(())
    }

    fn abandon_no_throw(&self, _: AcquiredFrameId) {}

    fn release(&self) {}
}

fn block_on<T>(future: impl core::future::Future<Output = T>) -> T {
    use core::task::{Context, Poll, Waker};

    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the test future unexpectedly suspended"),
    }
}

/// A device identity, as the platform layer mints one.
fn device_identity(instance: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(instance))
}

/// A surface-facts snapshot, as the presentation backend reports one.
fn target_caps(
    formats: Vec<TextureFormat>,
    present_modes: Vec<PresentMode>,
    extent_control: PresentationExtentControl,
) -> PresentationTargetCapabilities {
    target::PresentationTargetCapabilities::new(formats, present_modes, extent_control)
}

/// A target whose drawable size belongs to the host.
fn host_managed(current: Option<(u32, u32)>) -> PresentationExtentControl {
    PresentationExtentControl::HostManaged {
        current: current.map(|(width, height)| Extent2d { width, height }),
    }
}

/// A target the RHI may request an exact extent from.
fn configurable(
    min_width: u32,
    min_height: u32,
    max_width: u32,
    max_height: u32,
) -> PresentationExtentControl {
    PresentationExtentControl::Configurable {
        min: Extent2d {
            width: min_width,
            height: min_height,
        },
        max: Extent2d {
            width: max_width,
            height: max_height,
        },
    }
}

/// A frame the acquire path just produced, in the `Acquired` state.
fn acquired_frame(serial: u64, device: DeviceIdentity, width: u32, height: u32) -> AcquiredFrame {
    AcquiredFrame::new(
        AcquiredFrameId::new(device, serial),
        device,
        TextureFormat::Bgra8Unorm,
        Extent3d::d2(width, height),
    )
}
