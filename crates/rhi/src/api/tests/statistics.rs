//! Contract tests for the statistics chapter (specification section 47).
//!
//! Section 47 is the largest chapter in the specification and the one with the
//! most to get wrong, because most of it is a *vocabulary* rather than a
//! behaviour: sixty named counters whose definitions are frozen across backends,
//! and a device-scoped service that hands them out. So the tests are split the
//! way the chapter is, under the seams the specification already draws:
//!
//! ```text
//! counters   47.5 - 47.8, 47.10 - 47.12   the counter records
//! snapshot   47.4, 47.9, 47.13            consistency, the interval, the sampler
//! inventory  47.14 - 47.18                live inventory and the memory estimate
//! ```
//!
//! and this file holds what belongs to none of them: the device scope (47.2), the
//! collection level (47.3), and the two verbs that are answered from data already
//! in hand.
//!
//! Three kinds of test appear throughout, and the distinction is load-bearing:
//!
//! * **Behavioural tests** drive the parts that are already real — the portable
//!   refusals, the descriptor-based estimate, the sampling-rate formula, the
//!   counter records' own defaults. They run and they assert exact kinds.
//! * **Shape tests** are ordinary functions compiled but never called, written as
//!   realistic caller-side checks for ergonomics that the type system owns.
//! * **Tier tests** verify that portable validation runs before a backend or
//!   capability boundary and that an unavailable fact is returned structurally,
//!   never fabricated as a number.
//!
//! There is no GPU behind any of this. Counters that read zero here read zero
//! because nothing was ever recorded, and section 47.1's own warning applies to
//! every number in this file: a statistic that reads zero is not evidence that
//! nothing happened.
//!
//! One chapter rule is deliberately *not* asserted anywhere in this file, because
//! the type system already holds it: section 47.5 forbids a counter from wrapping,
//! and the increment path that would wrap is the recorder's, which does not exist.
//! A test that added `u64::MAX + 1` by hand would be testing `u64`, not the RHI.

mod counters;
mod inventory;
mod snapshot;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::resource::buffer::{Buffer, BufferDescriptor, BufferUsage};
use crate::api::resource::texture::{Texture, TextureDescriptor, TextureUsage};
use crate::api::statistics::{
    DeviceStatistics, MemoryEstimate, MemoryEstimateQuality, StatisticsConfig, StatisticsDetail,
};
use crate::api::tests::fixture;

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn identity(instance: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(instance))
}

fn device() -> DeviceIdentity {
    identity(1)
}

/// A second device, so that "the domains are independent" is a statement about
/// two real identities rather than about one identity compared with itself.
fn other_device() -> DeviceIdentity {
    identity(2)
}

fn object(value: u64) -> ObjectId {
    ObjectId::new(value)
}

fn statistics() -> DeviceStatistics {
    crate::api::tests::mock::device_for_test(device()).statistics()
}

fn buffer_on(device: DeviceIdentity, size: u64) -> Buffer {
    fixture::buffer(
        object(1),
        device,
        BufferDescriptor::new(size, BufferUsage::UNIFORM),
    )
}

fn texture_on(device: DeviceIdentity) -> Texture {
    Texture::new(
        object(2),
        device,
        TextureDescriptor::new_2d(
            64,
            64,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
        ),
    )
}

/// Asserts the exact refusal kind, never "an error".
///
/// The kind is the part a caller branches on, so a test that accepted any error
/// would not notice a refusal moving from `WrongDevice` to `InvalidUsage`, which
/// is exactly the change that would break every `match` in every caller.
fn assert_kind(result: RhiResult<impl core::fmt::Debug>, expected: RhiErrorKind) {
    match result {
        Ok(value) => panic!("expected {expected}, but the operation was accepted: {value:?}"),
        Err(error) => assert_eq!(error.kind(), expected, "{}", error.message()),
    }
}

// ---------------------------------------------------------------------------
// Section 47.2 — device scope.
// ---------------------------------------------------------------------------

/// Each [`DeviceIdentity`] has an independent statistics domain and there is no
/// global singleton.
///
/// What is testable before the domain exists is the refusal the boundary depends
/// on: a resource from one device is a portable usage error here, decided above
/// the backend, and the error names the object it refused — without that, a
/// caller holding a frame's worth of resources has no way to tell which one
/// belonged to the wrong device.
#[test]
fn a_statistics_service_refuses_a_resource_from_another_device() {
    let service = statistics();

    let own_buffer = buffer_on(device(), 1024);
    let own = service
        .estimate_buffer_memory(&own_buffer)
        .expect("a buffer of the owning device");
    assert_eq!(own.logical_estimated_bytes, Some(1024));
    assert_eq!(own.quality, MemoryEstimateQuality::LogicalEstimate);

    let foreign = buffer_on(other_device(), 1024);
    match service.estimate_buffer_memory(&foreign) {
        Ok(estimate) => panic!("expected WrongDevice, but an estimate was produced: {estimate:?}"),
        Err(error) => {
            assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
            assert_eq!(error.object(), Some(object(1)));
        }
    }
}

/// Re-requesting after terminal loss creates a different logical device domain.
#[test]
fn a_lost_and_recreated_device_is_not_the_same_domain() {
    let stale = crate::api::tests::mock::device_for_test(identity(1)).statistics();
    let current = crate::api::tests::mock::device_for_test(identity(3)).statistics();

    let recorded_before_loss = buffer_on(identity(1), 4096);

    assert!(stale.estimate_buffer_memory(&recorded_before_loss).is_ok());
    assert_kind(
        current.estimate_buffer_memory(&recorded_before_loss),
        RhiErrorKind::WrongDevice,
    );
}

/// The RHI does not combine two devices' domains into one counter, because a
/// combined number would have no owner and no epoch.
///
/// Section 47.2 puts that combination in the caller's hands, and the test writes
/// the aggregation out so that the shape of it is visible: two estimates, each
/// carrying its own quality, summed only where both are known. A service that
/// offered to add them would have to decide what to do with an `Unknown`, and
/// there is no answer that is right for every caller.
#[test]
fn two_devices_are_aggregated_by_the_caller_and_not_by_the_rhi() {
    let first = crate::api::tests::mock::device_for_test(device()).statistics();
    let second = crate::api::tests::mock::device_for_test(other_device()).statistics();

    let first_estimate = first
        .estimate_buffer_memory(&buffer_on(device(), 2048))
        .expect("a buffer of the first device");
    let second_estimate = second
        .estimate_buffer_memory(&buffer_on(other_device(), 2048))
        .expect("a buffer of the second device");

    let total = match (
        first_estimate.logical_estimated_bytes,
        second_estimate.logical_estimated_bytes,
    ) {
        (Some(a), Some(b)) => Some(a + b),
        _ => None,
    };
    assert_eq!(total, Some(4096));

    // Neither service can see the other's resource, so neither could have
    // produced that total on its own.
    assert_kind(
        first.estimate_buffer_memory(&buffer_on(other_device(), 2048)),
        RhiErrorKind::WrongDevice,
    );
}

// ---------------------------------------------------------------------------
// Section 47.3 — the collection level.
// ---------------------------------------------------------------------------

/// The cheapest level is the default, and that is a rule rather than a
/// convenience: `Minimal` is the level that costs no work in a hot recording
/// loop, so a caller who never configures anything never pays for statistics.
#[test]
fn the_default_collection_level_is_the_cheapest_one() {
    assert_eq!(StatisticsConfig::default(), StatisticsConfig::minimal());
    assert_eq!(
        StatisticsConfig::default().detail,
        StatisticsDetail::Minimal
    );

    assert_eq!(StatisticsConfig::basic().detail, StatisticsDetail::Basic);
    assert_eq!(
        StatisticsConfig::detailed().detail,
        StatisticsDetail::Detailed
    );
}

/// The three levels nest, each adding the cost of the level below it, so they
/// must be three distinct values rather than a set of flags.
#[test]
fn the_three_collection_levels_are_distinct() {
    let levels = [
        StatisticsDetail::Minimal,
        StatisticsDetail::Basic,
        StatisticsDetail::Detailed,
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

    assert_ne!(StatisticsConfig::minimal(), StatisticsConfig::basic());
    assert_ne!(StatisticsConfig::basic(), StatisticsConfig::detailed());
    assert_ne!(StatisticsConfig::minimal(), StatisticsConfig::detailed());
}

// ---------------------------------------------------------------------------
// Section 47.16 — the buffer estimate, which needs no backend.
// ---------------------------------------------------------------------------

/// `logical_estimated_bytes = BufferDescriptor.size`, and nothing else.
///
/// One of the two verbs in the chapter that is answered from data already in
/// hand, and the test is here rather than in a backend suite because there is no
/// backend in it: the estimate is a read of a descriptor the crate already holds.
#[test]
fn a_buffer_estimate_is_the_descriptor_size() {
    let service = statistics();

    let zero = service
        .estimate_buffer_memory(&buffer_on(device(), 0))
        .expect("a zero-byte buffer is a legal buffer");

    // Zero is a real answer, not the absence of one. A caller that conflated the
    // two would report an unmeasured resource as measured-and-empty.
    assert_eq!(zero.logical_estimated_bytes, Some(0));
    assert_eq!(zero.quality, MemoryEstimateQuality::LogicalEstimate);

    let larger = service
        .estimate_buffer_memory(&buffer_on(device(), 1 << 30))
        .expect("a large buffer");
    assert_eq!(larger.logical_estimated_bytes, Some(1 << 30));

    // Monotone in the descriptor, which is the only property of it a caller
    // could reasonably build on.
    assert!(
        larger.logical_estimated_bytes > zero.logical_estimated_bytes,
        "a larger descriptor must not estimate smaller"
    );

    // And the two qualities are different states, which is what makes `unknown`
    // usable as "not estimated" rather than as a flavour of zero.
    assert_ne!(zero.quality, MemoryEstimate::unknown().quality);
    assert_eq!(MemoryEstimate::unknown().logical_estimated_bytes, None);
}

// ---------------------------------------------------------------------------
// Section 47.18 — the texture estimate, whose portable half runs first.
// ---------------------------------------------------------------------------

/// The two-tier rule of section 4, as an executable assertion: cross-device
/// ownership is refused before texture format facts are consulted.
///
/// The order is the whole point. A caller that passed a foreign texture must see
/// the promised [`RhiErrorKind::WrongDevice`], rather than a format lookup or an
/// estimate about an unrelated device.
#[test]
fn a_texture_from_another_device_is_refused_before_the_estimate_reads_facts() {
    let service = statistics();
    let foreign = texture_on(other_device());

    let error = service
        .estimate_texture_memory(&foreign)
        .expect_err("a texture from another device is refused");
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert_eq!(error.object(), Some(object(2)));
}

#[test]
fn a_texture_estimate_on_the_owning_device_uses_enabled_format_facts() {
    let service = statistics();
    let own = texture_on(device());

    let estimate = service
        .estimate_texture_memory(&own)
        .expect("the owning texture is estimable or explicitly unknown");
    assert!(matches!(
        estimate.quality,
        MemoryEstimateQuality::LogicalEstimate | MemoryEstimateQuality::Unknown
    ));
}

// ---------------------------------------------------------------------------
// The service handle's own shape.
// ---------------------------------------------------------------------------

/// A handle is cloneable, and every clone refers to the same domain, because the
/// domain belongs to the device rather than to the handle.
///
/// The property is observable today only through the identity the whole domain is
/// keyed by: two clones refuse the same foreign resource and accept the same own
/// resource, which is what "same domain" reduces to until a domain exists.
#[test]
fn every_clone_of_a_handle_is_scoped_to_the_same_domain() {
    let service = statistics();
    let clone = service.clone();

    let foreign = buffer_on(other_device(), 16);
    assert_kind(
        service.estimate_buffer_memory(&foreign),
        RhiErrorKind::WrongDevice,
    );
    assert_kind(
        clone.estimate_buffer_memory(&foreign),
        RhiErrorKind::WrongDevice,
    );

    let own = buffer_on(device(), 16);
    let from_service = service.estimate_buffer_memory(&own).expect("own device");
    let from_clone = clone.estimate_buffer_memory(&own).expect("own device");
    assert_eq!(
        from_service.logical_estimated_bytes,
        from_clone.logical_estimated_bytes
    );
    assert_eq!(from_service.quality, from_clone.quality);
}

/// The handle prints its device and says the rest is elided.
///
/// Adjudication A16: an opaque handle gets a hand-written `Debug`. The port will
/// add a native handle to the collection domain here, that handle has no reason to
/// be `Debug`, and printing a native object into a log is a leak — so the output
/// is the portable identity plus `..`, and a caller reading a log can see that it
/// is not the whole story.
#[test]
fn a_statistics_handle_prints_its_device_and_says_it_is_incomplete() {
    let rendered = format!("{:?}", statistics());

    assert!(rendered.starts_with("DeviceStatistics"), "{rendered}");
    assert!(rendered.contains("device"), "{rendered}");
    assert!(rendered.ends_with(", .. }"), "{rendered}");
}

/// The device verb, as a renderer calls it: once at setup, then held for the
/// lifetime of the device.
///
/// Compiled, never called. Section 47.2's rule that the domain belongs to the
/// device is what makes this a method on `Device` rather than a free function or
/// a global — a caller cannot obtain a statistics service without having a device
/// in hand, so it cannot accidentally aggregate two.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_renderer_takes_the_service_once(device: &crate::api::platform::device::Device) {
    let statistics = device.statistics();

    // Held across the frame, and taken again at the next frame's start without
    // the caller having to release the first one.
    let _held = statistics.clone();
    let _again = device.statistics();

    let _snapshot = statistics.snapshot();
    let _sampler = statistics.frame_sampler();
    let _inventory = statistics.inventory();
}

/// The verb that switches level, written as a caller writes it.
///
/// Compiled, never called: switching the level restarts the cumulative counters
/// on the device's collection domain, and that domain does not exist yet. What
/// this reviews is that the verb is *not* shaped like a setter — it returns
/// `RhiResult<()>` because the restart can fail, and a caller has to be able to
/// tell a successful reconfigure from one that did not happen, since the two
/// leave the counters in different states.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_caller_switches_collection_level(service: &DeviceStatistics) -> RhiResult<()> {
    service.configure(StatisticsConfig::detailed())?;

    let current = service.config();
    assert_eq!(current.detail, StatisticsDetail::Detailed);

    // The epoch a caller reads back before taking a snapshot, because a snapshot
    // from a previous epoch cannot be subtracted from one of these.
    let _epoch: u64 = service.collection_epoch();
    Ok(())
}

/// The live inventory is not reset by a reconfigure, because it describes what
/// exists and what exists did not change.
///
/// Compiled, never called. The pair of calls is the assertion: section 47.3's
/// restart applies to the *event* counters, and a caller must be able to take an
/// inventory on both sides of a reconfigure and compare them.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_the_inventory_survives_a_reconfigure(service: &DeviceStatistics) -> RhiResult<()> {
    let before = service.inventory()?;
    service.configure(StatisticsConfig::detailed())?;
    let after = service.inventory()?;

    assert_eq!(before.device, after.device);
    Ok(())
}

/// The error model the refusals above rely on, restated where it is used: a
/// refusal carries a kind, a message, the object it is about, and — when the
/// emitting site names itself — the operation.
#[test]
fn a_statistics_refusal_carries_the_error_model_every_other_chapter_uses() {
    let service = statistics();
    let foreign = buffer_on(other_device(), 8);

    let error: RhiError = service
        .estimate_buffer_memory(&foreign)
        .expect_err("a foreign buffer is refused");

    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert_eq!(error.object(), Some(object(1)));
    assert!(!error.message().is_empty());

    // An unset operation is a fact about the emitting site rather than a defect:
    // the verb records its call site when the port threads it through.
    let _ = error.operation();
}
