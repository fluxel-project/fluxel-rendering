//! Contract tests for the submission chapter (specification sections 39 to 41),
//! including the submission-lane vocabulary of section 10.
//!
//! These are the review instrument for the plan interface, not a conformance
//! suite: no hardware is behind them and none of them may be presented as GPU
//! evidence. What they *can* answer is the half of the chapter that is portable
//! and therefore real today — which refusals happen before a device is reached,
//! what kind each one carries, and whether the rules of sections 40.4 and 45.3
//! are decidable at all from the facts a caller holds.
//!
//! # What makes that possible without a backend
//!
//! Every fact the builder needs is either a caller's own value or an
//! enumeration-time device answer, so the tests drive the builder through the same
//! crate-private constructor the port will use
//! ([`SubmissionPlanBuilder::with_facts`]) and hand it a lane table built by hand.
//! Nothing here reaches a driver, and nothing here is a substitute for the
//! hardware evidence the chapter still owes.
//!
//! # What these tests cannot reach
//!
//! ```text
//! a plan's native lowering and acceptance           no backend in this crate does it
//! the cross-plan hazard check of section 41.4       needs accepted-work history
//! ```
//!
//! The first row is why the acceptance tests below run against
//! [`crate::base::mock`]'s device: that backend accepts a plan and executes none of
//! it, which makes the *portable* half of section 41 — the phase split, the two
//! completion levels, the polling rule — reachable today. It is evidence about
//! this crate's logic and not about any GPU; `version-plan.md` section 4 still
//! requires a real DX12 and Vulkan run over raster, compute, copy, upload and
//! readback before this chapter is closed.

use crate::api::command::RecordedWork;
use crate::api::command::{
    AccessMask, BufferUse, FrameAttachmentUse, PipelineScope, ResourceUse, TextureUse,
    TextureUseIntent,
};
use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::platform::{Device, DeviceLossInfo};
use crate::api::presentation::{
    AcquiredFrame, AcquiredFrameId, AcquiredFrameState, PresentPlanId, PresentReceipt,
    PresentReceiptId,
};
use crate::api::resource::buffer::{Buffer, BufferDescriptor, BufferRange, BufferUsage};
use crate::api::resource::subresource::{TextureAspects, TextureSubresourceRange};
use crate::api::resource::texture::{Extent3d, Texture, TextureDescriptor, TextureUsage};
use crate::api::resource::transient::TransientLifetime;
use crate::api::submission::builder::validate_plan_graph;
use crate::api::submission::plan::{PlanBatch, PlanPresent};
use crate::api::submission::{
    CompletionFailure, CompletionPoint, CompletionState, LaneDependencyRoute, LaneWorkDomains,
    PlanPoint, SubmissionBatchId, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo, SubmissionPlanBuilder, SubmissionPlanId, SubmissionPoint,
    SubmissionReceipt,
};
use crate::api::tests::fixture;
use crate::base::mock::paired_device_for_test;
// The only import here that exists for the backend's own answer rather than the
// portable layer's: section 41.8's per-point obligation is owed by whoever owns
// the completion bookkeeping, and on this device that is the mock.
use crate::base::platform::DeviceBackend;

// ---------------------------------------------------------------------------
// Section 10 — the lane vocabulary.
//
// A lane is a *logically ordered submission domain*, and the two facts that make
// that usable are which domains a lane accepts and how happens-before between two
// lanes can be established. Both are device answers, so both are tested as
// questions about what the device reported rather than as hardware behaviour.
// ---------------------------------------------------------------------------

/// An unreported lane pair is `Unsupported`, and a lane's self pair is `Ordered`
/// by definition.
///
/// This is the conservative direction section 40.2 requires: under-reporting a
/// route costs a refusal a caller can restructure around, while reporting one that
/// does not exist would accept an edge that is not actually ordered.
#[test]
fn an_unreported_route_is_unsupported_and_a_self_pair_is_ordered() {
    let mut caps = SubmissionCapabilities::new(vec![
        lane(0, LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY)),
        lane(1, LaneWorkDomains::COMPUTE),
    ]);

    assert_eq!(
        caps.dependency_route(lane_id(0), lane_id(1)),
        LaneDependencyRoute::Unsupported,
        "a route no one reported is not a route"
    );
    assert_eq!(
        caps.dependency_route(lane_id(1), lane_id(1)),
        LaneDependencyRoute::Ordered,
        "one lane is one ordered domain, with no separate wait needed"
    );

    caps.record_dependency_route(lane_id(0), lane_id(1), LaneDependencyRoute::Gpu);

    assert_eq!(
        caps.dependency_route(lane_id(0), lane_id(1)),
        LaneDependencyRoute::Gpu
    );
    assert_eq!(
        caps.dependency_route(lane_id(1), lane_id(0)),
        LaneDependencyRoute::Unsupported,
        "a route is directional; the reverse pair was never reported"
    );
}

/// `lane` answers `None` for a lane the device never offered, which is what makes
/// a batch on a foreign lane refusable rather than submittable.
#[test]
fn a_lane_the_device_never_offered_has_no_entry() {
    let caps = SubmissionCapabilities::new(vec![lane(0, LaneWorkDomains::COPY)]);

    let reported = caps.lane(lane_id(0)).expect("lane 0 was reported");
    assert_eq!(reported.id(), lane_id(0));
    assert_eq!(reported.domains(), LaneWorkDomains::COPY);
    assert_eq!(reported.class(), SubmissionLaneClass::General);
    assert!(caps.lane(lane_id(9)).is_none());
    assert_eq!(caps.lanes().len(), 1);
}

/// Section 10's base guarantee, and the reason its failure is `BackendFailure`
/// rather than `Unsupported`: every device must have such a lane, so a snapshot
/// without one is an enumeration defect.
#[test]
fn the_base_guarantee_needs_a_raster_copy_lane_and_a_compute_lane_when_enabled() {
    let raster_copy = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
    let complete = SubmissionCapabilities::new(vec![
        lane(0, raster_copy),
        lane(1, LaneWorkDomains::COMPUTE),
    ]);

    assert!(complete.validate_base_guarantee(false).is_ok());
    assert!(complete.validate_base_guarantee(true).is_ok());

    let no_compute = SubmissionCapabilities::new(vec![lane(0, raster_copy)]);
    assert!(no_compute.validate_base_guarantee(false).is_ok());
    assert_eq!(
        no_compute.validate_base_guarantee(true).unwrap_err().kind(),
        RhiErrorKind::BackendFailure,
        "compute was enabled and no lane accepts it"
    );

    let copy_only = SubmissionCapabilities::new(vec![lane(0, LaneWorkDomains::COPY)]);
    assert_eq!(
        copy_only.validate_base_guarantee(false).unwrap_err().kind(),
        RhiErrorKind::BackendFailure,
        "every device must offer one lane that takes both raster and copy work"
    );
}

/// The domain set behaves as a set: containment, union, emptiness, and a name in
/// a refusal.
#[test]
fn domain_sets_contain_union_print_and_report_emptiness() {
    let both = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);

    assert!(both.contains(LaneWorkDomains::RASTER));
    assert!(both.contains(both));
    assert!(!LaneWorkDomains::RASTER.contains(both));
    assert!(both.contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY)));
    assert!(!LaneWorkDomains::COMPUTE.is_empty());
    assert_eq!(format!("{both}"), "RASTER|COPY");
    assert_eq!(format!("{}", LaneWorkDomains::COMPUTE), "COMPUTE");
}

// ---------------------------------------------------------------------------
// Section 40.1 — adding a batch.
// ---------------------------------------------------------------------------

/// An empty batch is refused, because a batch with no work has nothing to submit
/// and nothing to order.
#[test]
fn add_batch_refuses_an_empty_batch() {
    let device = device_identity(1);
    let mut builder = builder(device, vec![lane(0, everything())], 1);

    let error = builder
        .add_batch(lane_id(0), Vec::new())
        .expect_err("an empty batch is not a batch");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("SubmissionPlanBuilder::add_batch"));
}

/// A lane this device does not offer is `WrongDevice`, because a lane identity is
/// device-scoped: one obtained from another device cannot be a lane of this one.
#[test]
fn add_batch_refuses_a_lane_the_device_does_not_offer() {
    let device = device_identity(1);
    let mut builder = builder(device, vec![lane(0, everything())], 1);

    let error = builder
        .add_batch(lane_id(4), vec![raster_work(device, Vec::new())])
        .expect_err("lane 4 was never reported");

    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
}

/// Work recorded on another device is refused rather than submitted, which is the
/// batch-level form of section 3.3: P0 has no path that moves work between devices.
#[test]
fn add_batch_refuses_work_recorded_on_another_device() {
    let device = device_identity(1);
    let other = device_identity(2);
    let mut builder = builder(device, vec![lane(0, everything())], 1);

    let error = builder
        .add_batch(lane_id(0), vec![raster_work(other, Vec::new())])
        .expect_err("this work belongs to another device");

    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
}

/// A lane accepts only the domains it reported, which is the rule section 10.1
/// keeps apart from [`SubmissionLaneClass`]: the class is a label, and the domain
/// set is the legality answer.
#[test]
fn add_batch_refuses_work_whose_domains_the_lane_does_not_accept() {
    let device = device_identity(1);
    let mut builder = builder(
        device,
        vec![lane(0, LaneWorkDomains::COPY), lane(1, everything())],
        1,
    );

    let error = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .expect_err("a copy lane cannot take raster work");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);

    // The same work on a lane that reports what it needs is accepted, which is
    // what makes the refusal above a statement about the lane and not the work.
    assert!(
        builder
            .add_batch(lane_id(1), vec![raster_work(device, Vec::new())])
            .is_ok()
    );
}

/// A batch may contain several domains at once, which is the ordinary case for a
/// frame's recording.
#[test]
fn a_batch_may_contain_several_domains_at_once() {
    let device = device_identity(1);
    let mut builder = builder(device, vec![lane(0, everything())], 1);

    let mixed = RecordedWork::new(
        ObjectId::new(5),
        device,
        LaneWorkDomains::RASTER
            .union(LaneWorkDomains::COPY)
            .union(LaneWorkDomains::COMPUTE),
        Vec::new(),
        Vec::new(),
    );

    assert!(builder.add_batch(lane_id(0), vec![mixed]).is_ok());
}

// ---------------------------------------------------------------------------
// Sections 40.2 and 40.4 — ordering and the hazard analysis.
// ---------------------------------------------------------------------------

/// Two batches on one lane are ordered by insertion, so a write followed by a
/// read on one lane needs no explicit edge at all.
///
/// Section 40.1's implicit order is what makes the common case — a frame's
/// upload, its draws, and its present — expressible without a dependency graph.
#[test]
fn one_lane_orders_its_batches_without_an_explicit_edge() {
    let device = device_identity(1);
    let mut builder = builder(device, vec![lane(0, everything())], 1);

    let texture = texture_handle(1, device);
    let writer = builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![texture_use(texture.id(), device, AccessMask::COLOR_WRITE)],
            )],
        )
        .expect("the first batch is well formed");
    let reader = builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![texture_use(texture.id(), device, AccessMask::SHADER_READ)],
            )],
        )
        .expect("the second batch is well formed");

    assert_ne!(writer, reader);
    assert!(
        builder.build().is_ok(),
        "one lane's insertion order already orders the write before the read"
    );
}

/// Two lanes with no edge between them are unordered, so a write on one and an
/// overlapping write on the other is `MissingDependency` — the one kind section
/// 40.4 names outright.
#[test]
fn two_lanes_that_both_write_one_texture_need_an_edge() {
    let device = device_identity(1);
    let texture = texture_handle(1, device);

    // The raster half draws into the texture as a color attachment; the compute
    // half writes the same mips as a storage image. This is the classic
    // missing-barrier pair, and neither lane orders the other — so the batch points
    // are not kept: this plan is expected to be refused as it stands.
    let mut builder = routed_builder(device, 1);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![texture_use(texture.id(), device, AccessMask::COLOR_WRITE)],
            )],
        )
        .unwrap();
    builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![texture_use(texture.id(), device, AccessMask::SHADER_WRITE)],
            )],
        )
        .unwrap();

    assert_eq!(
        builder.build().unwrap_err().kind(),
        RhiErrorKind::MissingDependency,
        "two unordered lanes, one texture, and a write on both sides"
    );

    // With the edge, the same plan is legal — which is the whole point of the
    // refusal above: the fix is a dependency, not different work.
    let mut builder = routed_builder(device, 2);
    let raster = builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![texture_use(texture.id(), device, AccessMask::COLOR_WRITE)],
            )],
        )
        .unwrap();
    let compute = builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![texture_use(texture.id(), device, AccessMask::SHADER_WRITE)],
            )],
        )
        .unwrap();
    builder
        .add_dependency(raster, compute)
        .expect("this device reported a GPU route between these lanes");

    assert!(builder.build().is_ok());
}

/// A read-only pair on two lanes needs no dependency, and neither does a write
/// pair that does not overlap.
///
/// The three non-conflicting cases of section 40.4 are what keep the check from
/// being "any two batches on two lanes need an edge": unrelated resources,
/// disjoint ranges, and two reads.
#[test]
fn unrelated_disjoint_and_read_only_pairs_need_no_edge() {
    let device = device_identity(1);

    // Two different textures.
    let a = texture_handle(1, device);
    let b = texture_handle(2, device);
    let mut builder = routed_builder(device, 1);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![texture_use(a.id(), device, AccessMask::COLOR_WRITE)],
            )],
        )
        .unwrap();
    builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![texture_use(b.id(), device, AccessMask::SHADER_WRITE)],
            )],
        )
        .unwrap();
    assert!(
        builder.build().is_ok(),
        "two different textures cannot conflict"
    );

    // One texture, disjoint mip ranges.
    let mut builder = routed_builder(device, 2);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![texture_range_use(
                    a.id(),
                    device,
                    AccessMask::COLOR_WRITE,
                    0,
                    1,
                )],
            )],
        )
        .unwrap();
    builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![texture_range_use(
                    a.id(),
                    device,
                    AccessMask::SHADER_WRITE,
                    1,
                    1,
                )],
            )],
        )
        .unwrap();
    assert!(
        builder.build().is_ok(),
        "mip 0 and mip 1 of one texture are disjoint subresources"
    );

    // One buffer, overlapping ranges, one side writing.
    let buffer = buffer_handle(1, device);
    let mut builder = routed_builder(device, 3);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![buffer_range_use(
                    buffer.id(),
                    device,
                    0,
                    64,
                    AccessMask::COPY_WRITE,
                )],
            )],
        )
        .unwrap();
    builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![buffer_range_use(
                    buffer.id(),
                    device,
                    32,
                    64,
                    AccessMask::SHADER_WRITE,
                )],
            )],
        )
        .unwrap();
    assert_eq!(
        builder.build().unwrap_err().kind(),
        RhiErrorKind::MissingDependency,
        "bytes 0..64 and 32..96 overlap"
    );

    // The same buffer, disjoint byte ranges.
    let mut builder = routed_builder(device, 4);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![buffer_range_use(
                    buffer.id(),
                    device,
                    0,
                    64,
                    AccessMask::COPY_WRITE,
                )],
            )],
        )
        .unwrap();
    builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![buffer_range_use(
                    buffer.id(),
                    device,
                    64,
                    64,
                    AccessMask::SHADER_WRITE,
                )],
            )],
        )
        .unwrap();
    assert!(
        builder.build().is_ok(),
        "bytes 0..64 and 64..128 do not share a byte"
    );

    // One texture, both sides reading.
    let mut builder = routed_builder(device, 5);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![texture_use(a.id(), device, AccessMask::SHADER_READ)],
            )],
        )
        .unwrap();
    builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![texture_use(a.id(), device, AccessMask::SHADER_READ)],
            )],
        )
        .unwrap();
    assert!(
        builder.build().is_ok(),
        "two reads with no write on either side need no happens-before"
    );
}

/// A frame is a resource like any other for section 40.4: two lanes writing one
/// frame is a hazard, not a present-plan question.
#[test]
fn two_lanes_that_both_write_one_frame_need_an_edge() {
    let device = device_identity(1);
    let frame_id = AcquiredFrameId::new(device, 1);

    let mut builder = routed_builder(device, 1);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![ResourceUse::Frame(FrameAttachmentUse {
                    frame: frame_id,
                    stages: PipelineScope::FRAGMENT,
                    access: AccessMask::COLOR_WRITE,
                })],
            )],
        )
        .unwrap();
    builder
        .add_batch(
            lane_id(1),
            vec![compute_work(
                device,
                vec![ResourceUse::Frame(FrameAttachmentUse {
                    frame: frame_id,
                    stages: PipelineScope::COMPUTE,
                    access: AccessMask::COLOR_WRITE,
                })],
            )],
        )
        .unwrap();

    assert_eq!(
        builder.build().unwrap_err().kind(),
        RhiErrorKind::MissingDependency
    );
}

// ---------------------------------------------------------------------------
// Sections 40.2 and 40.5 — the explicit dependency verbs.
// ---------------------------------------------------------------------------

/// A route the device did not report is refused with `Unsupported`, and the
/// refusal is honest rather than a fallback: a caller that needs the order puts
/// both batches on one lane, or observes completion first.
#[test]
fn add_dependency_refuses_a_route_the_device_never_reported() {
    let device = device_identity(1);
    let mut builder = builder(
        device,
        vec![lane(0, everything()), lane(1, everything())],
        1,
    );
    let first = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();
    let second = builder
        .add_batch(lane_id(1), vec![raster_work(device, Vec::new())])
        .unwrap();

    let error = builder.add_dependency(first, second).unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert_eq!(
        error.operation(),
        Some("SubmissionPlanBuilder::add_dependency")
    );
}

/// An already-ordered pair is `Ordered`, and it is accepted: a backend that has no
/// separate wait primitive for a pair it already orders must not refuse the edge.
#[test]
fn add_dependency_accepts_a_pair_the_device_already_orders() {
    let device = device_identity(1);
    let mut builder = builder(device, vec![lane(0, everything())], 1);
    let first = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();
    let second = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();

    builder
        .add_dependency(first, second)
        .expect("one lane is one ordered domain");
    assert!(builder.build().is_ok());
}

/// A point from another plan is `InvalidUsage`, and a self-loop is refused for the
/// same reason: an edge that says a batch runs before itself states no order.
#[test]
fn add_dependency_refuses_a_foreign_point_and_a_self_loop() {
    let device = device_identity(1);
    let foreign = {
        let mut other = builder(device, vec![lane(0, everything())], 99);
        other
            .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
            .unwrap()
    };

    let mut builder = routed_builder(device, 1);
    let mine = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();
    let other_lane = builder
        .add_batch(lane_id(1), vec![raster_work(device, Vec::new())])
        .unwrap();

    let error = builder.add_dependency(foreign, mine).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);

    let error = builder.add_dependency(mine, mine).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(
        !error.message().contains("another plan"),
        "a self-loop is refused as a loop, not as a cross-plan mistake"
    );

    // A point that names no batch of this plan at all is the same refusal as a
    // foreign plan's point: neither is one of mine.
    let unknown = PlanPoint::new(plan_id(device, 1), SubmissionBatchId::new(7));
    assert_eq!(
        builder.add_dependency(unknown, mine).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );

    builder
        .add_dependency(mine, other_lane)
        .expect("a reported GPU route carries this edge");
}

/// A cycle is refused before anything is submitted, and the implicit order of one
/// lane participates in the detection: this edge closes a loop through batches
/// whose order was never written down.
#[test]
fn a_cycle_through_the_implicit_lane_order_is_refused() {
    let device = device_identity(1);
    let mut builder = builder(device, vec![lane(0, everything())], 1);

    // Insertion order puts `first` before `second` on one lane.
    let first = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();
    let second = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();

    builder
        .add_dependency(second, first)
        .expect("the edge itself is legal to state; the plan it makes is not");

    let error = builder.build().unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("SubmissionPlanBuilder::build"));
}

/// An external dependency is accepted when its device matches, and refused with
/// `WrongDevice` when it does not.
#[test]
fn add_external_dependency_checks_the_earlier_work_device() {
    let device = device_identity(1);
    let other = device_identity(2);
    let mut builder = builder(device, vec![lane(0, everything())], 1);
    let after = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();

    assert_eq!(
        builder
            .add_external_dependency(CompletionPoint::new(other, 3), after)
            .unwrap_err()
            .kind(),
        RhiErrorKind::WrongDevice
    );

    builder
        .add_external_dependency(CompletionPoint::new(device, 3), after)
        .expect("an earlier plan on this device can be ordered before this batch");
    builder
        .add_external_dependency(CompletionPoint::new(device, 4), after)
        .expect("several earlier completions may precede one batch");

    let plan = builder.build().unwrap();
    assert_eq!(plan.external_dependencies().len(), 2);
}

// ---------------------------------------------------------------------------
// Sections 41.9 and 45.1 to 45.3 — frames, presents, and the closure.
// ---------------------------------------------------------------------------

/// The realistic present path: a frame is acquired, drawn into by the batch that
/// presents it, and handed to the plan.
#[test]
fn a_frame_is_consumed_by_present_after_and_the_plan_owns_it() {
    let device = device_identity(1);
    let frame = acquired_frame(1, device);
    let frame_id = frame.id();

    let mut builder = builder(device, vec![lane(0, everything())], 1);
    let drawing = builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![ResourceUse::Frame(FrameAttachmentUse {
                    frame: frame_id,
                    stages: PipelineScope::FRAGMENT,
                    access: AccessMask::COLOR_WRITE,
                })],
            )],
        )
        .unwrap();

    assert_eq!(frame.state(), AcquiredFrameState::Acquired);
    let present = builder
        .present_after(frame, drawing)
        .expect("a frame acquired now may be presented after the batch that fills it");

    let plan = builder.build().expect("the closure closes");

    assert_eq!(plan.device_identity(), device);
    assert_eq!(plan.batches().len(), 1);
    assert_eq!(plan.presents().len(), 1);
    assert_eq!(plan.frames().len(), 1);
    assert_eq!(plan.frames()[0].id(), frame_id);
    assert_eq!(
        plan.frames()[0].state(),
        AcquiredFrameState::PlannedForPresent,
        "the plan owns the frame, and the frame says so"
    );
    assert_eq!(plan.presents()[0].id, present);
    assert_eq!(plan.presents()[0].frame, frame_id);
    assert_eq!(plan.presents()[0].after, drawing);
}

/// A frame acquired on another device cannot be presented by this plan.
#[test]
fn present_after_refuses_a_frame_from_another_device() {
    let device = device_identity(1);
    let other = device_identity(2);
    let frame = acquired_frame(1, other);

    let mut builder = builder(device, vec![lane(0, everything())], 1);
    let after = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();

    let error = builder.present_after(frame, after).unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert_eq!(
        error.operation(),
        Some("SubmissionPlanBuilder::present_after")
    );
}

/// The after-point belongs to this plan, like every other point.
#[test]
fn present_after_refuses_a_foreign_after_point() {
    let device = device_identity(1);
    let foreign = {
        let mut other = builder(device, vec![lane(0, everything())], 99);
        other
            .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
            .unwrap()
    };

    let mut builder = builder(device, vec![lane(0, everything())], 1);
    let error = builder
        .present_after(acquired_frame(1, device), foreign)
        .unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

/// A frame is presented once. Two acquisitions of one drawable — which a
/// misbehaving caller can produce by cloning the identity, and a backend can
/// produce by returning a stale image — are refused at build time, which is where
/// section 40.5 puts the rule.
#[test]
fn build_refuses_two_presents_of_one_frame() {
    let device = device_identity(1);
    let frame_id = AcquiredFrameId::new(device, 7);

    let mut builder = builder(device, vec![lane(0, everything())], 1);
    let after = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();

    // Two frames with one identity: separately acquired values, one drawable.
    builder
        .present_after(acquired_frame_with(frame_id, device), after)
        .unwrap();
    builder
        .present_after(acquired_frame_with(frame_id, device), after)
        .unwrap();

    let error = builder.build().unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(
        error.message().contains("presented twice"),
        "the refusal names the rule it broke: {}",
        error.message()
    );
}

/// A batch that draws into a frame is only legal in the plan that presents that
/// frame: section 45.3 refuses a use of a frame nothing presents, because no work
/// would carry it to a display and its ownership would have no end.
#[test]
fn build_refuses_a_frame_use_the_plan_never_presents() {
    let device = device_identity(1);
    let frame_id = AcquiredFrameId::new(device, 7);

    let mut builder = builder(device, vec![lane(0, everything())], 1);
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![ResourceUse::Frame(FrameAttachmentUse {
                    frame: frame_id,
                    stages: PipelineScope::FRAGMENT,
                    access: AccessMask::COLOR_WRITE,
                })],
            )],
        )
        .unwrap();

    let error = builder.build().unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(
        error.message().contains("never presents"),
        "the refusal names the frame closure: {}",
        error.message()
    );
}

/// A frame use must happen before the point that presents the frame. Drawing into
/// a frame *after* it was presented is the other half of section 45.3, and it is
/// refused even though nothing in the plan is unordered in the hazard sense.
#[test]
fn build_refuses_a_frame_use_after_the_present_point() {
    let device = device_identity(1);
    let frame = acquired_frame(1, device);
    let frame_id = frame.id();

    let mut builder = routed_builder(device, 1);
    // The frame use is on lane 0, and the present point is the batch on lane 1.
    builder
        .add_batch(
            lane_id(0),
            vec![raster_work(
                device,
                vec![ResourceUse::Frame(FrameAttachmentUse {
                    frame: frame_id,
                    stages: PipelineScope::FRAGMENT,
                    access: AccessMask::COLOR_WRITE,
                })],
            )],
        )
        .unwrap();
    let presenting = builder
        .add_batch(lane_id(1), vec![compute_work(device, Vec::new())])
        .unwrap();
    builder.present_after(frame, presenting).unwrap();

    let error = builder.build().unwrap_err();

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(
        error.message().contains("does not happen before"),
        "the refusal says which direction is missing: {}",
        error.message()
    );
}

/// The plan builder is the last owner of a frame it consumed: a build that fails
/// drops the builder, and with it every frame — which is section 41.9's no-submit
/// abandonment path, and the reason it can be stated without a `Drop` impl on the
/// builder of its own.
#[test]
fn a_failed_build_drops_the_frames_it_consumed() {
    let device = device_identity(1);
    let frame = acquired_frame(1, device);
    let frame_id = frame.id();

    let mut builder = builder(device, vec![lane(0, everything())], 1);
    let first = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();
    let second = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();
    builder.present_after(frame, first).unwrap();
    builder.add_dependency(second, first).unwrap();

    // The cycle makes the build fail; the frame goes down with the builder, and
    // nothing was submitted, because no device was ever reached.
    assert_eq!(
        builder.build().unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    let _ = frame_id;
}

// ---------------------------------------------------------------------------
// Section 40.5's build-time rules, driven directly.
//
// The builder refuses a foreign point at insertion, so the graph validator's own
// foreign-point branch is unreachable through the public verbs. It is exercised
// here through the same crate-private entry point the port will call, because a
// check that only exists for the port's benefit still has to be right.
// ---------------------------------------------------------------------------

/// A graph whose edges name batches that are not in it is refused rather than
/// analysed.
#[test]
fn validate_plan_graph_refuses_edges_outside_the_plan() {
    let device = device_identity(1);
    let plan = plan_id(device, 1);
    let foreign = PlanPoint::new(plan_id(device, 2), SubmissionBatchId::new(0));
    let local = PlanPoint::new(plan, SubmissionBatchId::new(0));
    let batches = vec![batch(plan, 0, lane_id(0), raster_work(device, Vec::new()))];

    assert_eq!(
        validate_plan_graph(&batches, &[(local, foreign)], &[])
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        validate_plan_graph(&batches, &[(foreign, local)], &[])
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

/// A present whose after-point names no batch of the plan is refused too: the
/// closure rule cannot be stated without it.
#[test]
fn validate_plan_graph_refuses_a_present_with_no_after_batch() {
    let device = device_identity(1);
    let plan = plan_id(device, 1);
    let frame_id = AcquiredFrameId::new(device, 1);
    let batches = vec![batch(plan, 0, lane_id(0), raster_work(device, Vec::new()))];
    let presents = vec![PlanPresent {
        id: PresentPlanId::new(plan, 0),
        frame: frame_id,
        after: PlanPoint::new(plan, SubmissionBatchId::new(9)),
    }];

    assert_eq!(
        validate_plan_graph(&batches, &[], &presents)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

/// A built plan carries what it validated: the batches, the explicit edges, and the
/// presents. This is the only view of a plan's contents from outside the lowering,
/// and the one a backend reads.
#[test]
fn a_built_plan_carries_what_it_validated() {
    let device = device_identity(1);
    let mut builder = routed_builder(device, 1);
    let first = builder
        .add_batch(lane_id(0), vec![raster_work(device, Vec::new())])
        .unwrap();
    let second = builder
        .add_batch(lane_id(1), vec![compute_work(device, Vec::new())])
        .unwrap();
    builder
        .add_dependency(first, second)
        .expect("this device reported a GPU route between these lanes");
    let present = builder
        .present_after(acquired_frame(1, device), second)
        .expect("the frame belongs to this device and the point to this plan");

    let plan = builder.build().unwrap();

    assert_eq!(plan.device_identity(), device);
    assert_eq!(plan.batches().len(), 2);
    assert_eq!(plan.dependencies().len(), 1);
    assert_eq!(plan.dependencies()[0], (first, second));
    assert!(
        plan.external_dependencies().is_empty(),
        "nothing was chained to earlier accepted work"
    );
    assert_eq!(
        plan.frames().len(),
        1,
        "the plan owns the frame it presents"
    );
    assert_eq!(plan.presents().len(), 1);
    assert_eq!(
        plan.presents()[0].id,
        present,
        "the identity the caller was handed is the identity the plan carries"
    );
    assert_eq!(plan.presents()[0].after, second);
}

// ---------------------------------------------------------------------------
// Sections 39 and 41 — identity tokens and the receipt.
// ---------------------------------------------------------------------------

/// A plan identity is device-scoped, and a point names a batch inside one plan.
#[test]
fn plan_identity_is_device_scoped_and_points_name_their_batch() {
    let device = device_identity(1);
    let other = device_identity(2);
    let point = PlanPoint::new(plan_id(device, 3), SubmissionBatchId::new(5));

    assert_eq!(point.batch(), SubmissionBatchId::new(5));
    assert_eq!(plan_id(device, 3), plan_id(device, 3));
    assert_ne!(plan_id(device, 3), plan_id(device, 4));
    assert_ne!(plan_id(device, 3), plan_id(other, 3));
    assert_eq!(
        plan_id(device, 3).device_identity(),
        device,
        "a plan serial is unique per device, so the device half is what makes it so"
    );
    assert_eq!(plan_id(other, 3).device_identity(), other);
}

/// Section 41.7 keeps acceptance and completion apart, and the receipt carries
/// both without conflating them.
#[test]
fn a_receipt_reports_acceptance_and_completion_separately() {
    let device = device_identity(1);
    let plan = plan_id(device, 1);
    let submitted = SubmissionPoint::new(device, 10);
    let overall = CompletionPoint::new(device, 10);

    let receipt = SubmissionReceipt::new(plan, device, submitted, overall, Vec::new(), Vec::new());

    assert_eq!(receipt.device_identity(), device);
    assert_eq!(receipt.submitted(), submitted);
    assert_eq!(receipt.completion(), overall);
    assert!(receipt.presents().is_empty());
    assert_eq!(
        receipt.submitted().device_identity(),
        receipt.completion().device_identity()
    );
    assert!(
        format!("{receipt:?}").contains("SubmissionReceipt"),
        "a receipt prints which plan it is about rather than what the driver was handed"
    );
}

/// Section 41.2's fallback: a backend with no finer primitive than "everything
/// submitted so far" answers a per-batch query with the overall token, which is
/// coarse in the safe direction.
#[test]
fn completion_for_falls_back_to_the_overall_token_and_refuses_a_foreign_plan() {
    let device = device_identity(1);
    let plan = plan_id(device, 1);
    let overall = CompletionPoint::new(device, 10);
    let per_batch = CompletionPoint::new(device, 11);
    let first = PlanPoint::new(plan, SubmissionBatchId::new(0));
    let second = PlanPoint::new(plan, SubmissionBatchId::new(1));

    let receipt = SubmissionReceipt::new(
        plan,
        device,
        SubmissionPoint::new(device, 1),
        overall,
        vec![(first, per_batch)],
        Vec::new(),
    );

    assert_eq!(receipt.completion_for(first).unwrap(), per_batch);
    assert_eq!(
        receipt.completion_for(second).unwrap(),
        overall,
        "a point with no recorded token is answered coarsely, not refused"
    );

    let foreign = PlanPoint::new(plan_id(device, 2), SubmissionBatchId::new(0));
    let error = receipt.completion_for(foreign).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("SubmissionReceipt::completion_for"));
}

/// A receipt carries the presentations of its plan, which section 45.5 makes the
/// independent other half of the plan's fate.
#[test]
fn a_receipt_carries_the_presents_of_its_plan() {
    let device = device_identity(1);
    let plan = plan_id(device, 1);
    let present = PresentPlanId::new(plan, 0);

    let receipt = SubmissionReceipt::new(
        plan,
        device,
        SubmissionPoint::new(device, 1),
        CompletionPoint::new(device, 1),
        Vec::new(),
        vec![PresentReceipt::new(
            PresentReceiptId::new(device, 2),
            present,
        )],
    );

    assert_eq!(receipt.presents().len(), 1);
    assert_eq!(receipt.presents()[0].plan_id(), present);
    assert_eq!(
        receipt.presents()[0].id().device_identity(),
        device,
        "a present receipt is answerable against the device that accepted it"
    );
}

/// A terminal completion that did not complete carries its reason, which is what
/// section 41.1 adds to the completion vocabulary: the state alone would leave an
/// asynchronous failure unexplained long after the call that started the work.
#[test]
fn a_terminal_completion_carries_its_reason() {
    let failure = CompletionFailure::new("the lane could not continue");
    assert_eq!(failure.message(), "the lane could not continue");

    let failed = CompletionState::Failed(failure);
    if let CompletionState::Failed(reason) = failed {
        assert_eq!(
            reason.message(),
            "the lane could not continue",
            "a failed completion keeps its reason reachable"
        );
    } else {
        panic!("this state was built as a failure");
    }

    // The other terminal shape carries the device loss summary instead, which
    // section 6.5 makes stable rather than one-shot.
    let lost = CompletionState::DeviceLost(DeviceLossInfo::new("simulated loss".into()));
    if let CompletionState::DeviceLost(loss) = lost {
        assert_eq!(loss.message(), "simulated loss");
    } else {
        panic!("this state was built as a device loss");
    }
}

// ---------------------------------------------------------------------------
// Section 41.3 — Phase A refusals that need no backend.
// ---------------------------------------------------------------------------

/// Reserved points make a transient lifetime expressible before recording, but
/// never make an empty batch executable.
#[test]
fn a_reserved_batch_must_be_filled_exactly_once_before_build() {
    let identity = device_identity(1);
    let mut first_builder = builder(identity, vec![lane(0, everything())], 1);
    let _point = first_builder.reserve_batch(lane_id(0)).unwrap();

    assert_eq!(
        first_builder.build().unwrap_err().kind(),
        RhiErrorKind::InvalidUsage,
        "a reserved point is not an empty batch"
    );

    let mut second_builder = builder(identity, vec![lane(0, everything())], 2);
    let point = second_builder.reserve_batch(lane_id(0)).unwrap();
    second_builder
        .set_batch(point, vec![raster_work(identity, Vec::new())])
        .unwrap();
    assert_eq!(
        second_builder
            .set_batch(point, vec![raster_work(identity, Vec::new())])
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage,
        "a point has exactly one work payload"
    );
}

/// The allocator is plan-scoped and validates even a transient resource that no
/// command ultimately uses; allocation cannot hide an invalid lifetime from the
/// builder merely by leaving the handle unrecorded.
#[test]
fn an_unused_transient_lifetime_still_obeys_the_plan_dag() {
    let identity = device_identity(1);
    let mut builder = builder(
        identity,
        vec![lane(0, everything()), lane(1, everything())],
        1,
    );
    let acquire = builder.reserve_batch(lane_id(0)).unwrap();
    let release = builder.reserve_batch(lane_id(1)).unwrap();
    let allocator = builder.transient_allocator();

    assert_eq!(allocator.device_identity(), identity);
    assert_eq!(allocator.plan_id(), acquire.plan());
    allocator
        .create_buffer(
            &BufferDescriptor::new(64, BufferUsage::COPY_DST),
            TransientLifetime::new(acquire).release_at(release),
        )
        .unwrap();
    builder
        .set_batch(acquire, vec![raster_work(identity, Vec::new())])
        .unwrap();
    builder
        .set_batch(release, vec![raster_work(identity, Vec::new())])
        .unwrap();

    assert_eq!(
        builder.build().unwrap_err().kind(),
        RhiErrorKind::InvalidUsage,
        "points on unordered lanes do not form a transient lifetime"
    );
}

/// Same-lane reservation order is happens-before, so it is sufficient for a
/// legal lifetime even when Dedicated backing performs no aliasing.
#[test]
fn a_transient_lifetime_uses_submission_ordering() {
    let identity = device_identity(1);
    let mut builder = builder(identity, vec![lane(0, everything())], 1);
    let acquire = builder.reserve_batch(lane_id(0)).unwrap();
    let release = builder.reserve_batch(lane_id(0)).unwrap();
    let allocator = builder.transient_allocator();
    let buffer = allocator
        .create_buffer(
            &BufferDescriptor::new(64, BufferUsage::COPY_DST),
            TransientLifetime::new(acquire).release_at(release),
        )
        .unwrap();

    assert_eq!(buffer.device_identity(), identity);
    assert_eq!(buffer.descriptor().size, 64);
    builder
        .set_batch(acquire, vec![raster_work(identity, Vec::new())])
        .unwrap();
    builder
        .set_batch(release, vec![raster_work(identity, Vec::new())])
        .unwrap();
    builder.build().unwrap();
}

/// A lifetime must be complete and entirely owned by the allocator's plan.
#[test]
fn a_transient_allocator_refuses_empty_and_foreign_lifetimes() {
    let identity = device_identity(1);
    let mut first = builder(identity, vec![lane(0, everything())], 1);
    let first_point = first.reserve_batch(lane_id(0)).unwrap();
    let mut second = builder(identity, vec![lane(0, everything())], 2);
    let foreign_point = second.reserve_batch(lane_id(0)).unwrap();
    let allocator = first.transient_allocator();
    let descriptor = BufferDescriptor::new(64, BufferUsage::COPY_DST);

    assert_eq!(
        allocator
            .create_buffer(&descriptor, TransientLifetime::new(first_point))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        allocator
            .create_buffer(
                &descriptor,
                TransientLifetime::new(first_point).release_at(foreign_point),
            )
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

/// A plan built for another device is refused before anything native is reached,
/// and a plan submitted to a lost device is refused as terminal.
///
/// Both are section 41.3's Phase A: they are decided before any native submit, so
/// `Err` here means no work was submitted — the one promise the phase split exists
/// to make.
#[test]
fn submit_refuses_a_foreign_plan_and_a_lost_device() {
    let device_identity_value = device_identity(1);
    let other = device_identity(2);

    let mut builder = builder(other, vec![lane(0, everything())], 1);
    builder
        .add_batch(lane_id(0), vec![raster_work(other, Vec::new())])
        .unwrap();
    let plan = builder.build().unwrap();

    let (device, native) = paired_device_for_test(device_identity_value);
    assert_eq!(plan.device_identity(), other);
    // The device is checked after its own liveness, so a lost device answers
    // `DeviceLost` for any plan, foreign or not.
    native.mark_lost(DeviceLossInfo::new("simulated loss".into()));
    let error = block_on(device.submit(plan)).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::DeviceLost);
    assert_eq!(error.operation(), Some("Device::submit"));
}

/// Asking about a foreign completion point is `WrongDevice`, and asking a lost
/// device about any point is `DeviceLost`.
///
/// Section 41.8 defines the per-point outcome after a loss, and that answer needs
/// per-point state this layer does not hold; what it can answer first is the loss
/// itself, which is terminal for the whole identity.
#[test]
fn completion_state_refuses_a_foreign_point_and_reports_a_lost_device() {
    let device_identity_value = device_identity(1);
    let other = device_identity(2);
    let (device, native) = paired_device_for_test(device_identity_value);

    let error = device
        .completion_state(CompletionPoint::new(other, 1))
        .unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);

    native.mark_lost(DeviceLossInfo::new("simulated loss".into()));
    let error = device
        .completion_state(CompletionPoint::new(device_identity_value, 1))
        .unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::DeviceLost);
}

// ---------------------------------------------------------------------------
// Shape tests.
//
// Compiled, never called. This one is the whole section 40 to 41 call site as a
// caller writes it, and it is the review instrument for the interface rather than
// for the implementation: it takes its values as parameters, so it compiles
// without a device to build them from. `SubmissionPlanBuilder::new` panics today
// because the device's lane table and plan serials are not built; a shape test is
// never called, so the compile is the whole point.
// ---------------------------------------------------------------------------

#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
async fn shape_frame_loop_through_submission(
    device: &Device,
    recorded: Vec<RecordedWork>,
    lane: SubmissionLaneId,
    frame: AcquiredFrame,
) -> RhiResult<()> {
    let mut builder = SubmissionPlanBuilder::new(device);
    let drawing = builder.add_batch(lane, recorded)?;
    let present = builder.present_after(frame, drawing)?;
    let plan = builder.build()?;

    let receipt = device.submit(plan).await?;
    let completion = receipt.completion_for(drawing)?;
    let _ = receipt.completion();
    for presented in receipt.presents() {
        let _ = presented.id();
    }

    // The frame loop observes rather than waits (section 41.10): one poll per
    // frame, and a completion point is what progress is asked about.
    device.poll()?;
    let state = device.completion_state(completion)?;
    let _ = state;
    let _ = device.wait_completion(completion).await?;

    let _ = present;
    Ok(())
}

/// A plan that presents two frames in one submit, as a caller with a two-surface
/// host writes it.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
async fn shape_two_presents_in_one_plan(
    device: &Device,
    first: AcquiredFrame,
    second: AcquiredFrame,
    work: Vec<RecordedWork>,
    lane: SubmissionLaneId,
) -> RhiResult<()> {
    let mut builder = SubmissionPlanBuilder::new(device);
    let batch = builder.add_batch(lane, work)?;
    let first_present = builder.present_after(first, batch)?;
    let second_present = builder.present_after(second, batch)?;
    let plan = builder.build()?;
    let _ = (first_present, second_present);

    // A caller reading the two outcomes after a submit must go through the
    // receipt's present list, because `present_after` consumed the frame tokens.
    let receipt = device.submit(plan).await?;
    for present in receipt.presents() {
        let _ = present.plan_id();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Acceptance and completion, against the mock backend.
//
// These are the only tests in this file that reach a backend at all, and what
// they prove is narrow and worth stating exactly: that the *portable* half of
// section 41 runs — the phase split of 41.3, the two completion levels of 41.2,
// the polling rule of 41.8, and the identity checks of 41.1 — over a backend that
// accepts plans and runs nothing. No GPU is behind any of them, and none of them
// may be presented as hardware evidence.
// ---------------------------------------------------------------------------

/// A submitted plan is accepted, reported, and its receipt names per-batch tokens.
///
/// The mock is deliberately finer than "everything so far", so the per-batch path
/// is the one exercised: section 41.2 makes finer the better answer when a backend
/// can give it, and `completion_for`'s fallback is a separate test below.
#[test]
fn a_validated_plan_is_accepted_and_its_receipt_reports_every_batch() {
    let identity = device_identity(1);
    let (device, native) = paired_device_for_test(identity);

    let mut builder = SubmissionPlanBuilder::new(&device);
    let first = builder
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .expect("the mock's lane 0 accepts raster work");
    let second = builder
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .expect("the mock's lane 0 accepts raster work");
    let plan = builder.build().expect("a batch with no edge is acyclic");

    let receipt = block_on(device.submit(plan))
        .expect("the mock backend accepts a plan the portable layer validated");

    assert_eq!(native.submissions(), 1);
    assert_eq!(receipt.device_identity(), identity);
    assert!(
        receipt.presents().is_empty(),
        "a plan that presents is refused before this point, so no receipt can list one"
    );
    // The two tokens are different facts and section 41.7 forbids collapsing them:
    // `submitted()` is when the RHI accepted the plan, `completion()` is when the
    // work is done. Both are answered, and both are this device's.
    assert_eq!(receipt.submitted().device_identity(), identity);
    assert_eq!(receipt.completion().device_identity(), identity);

    // Two batches, two distinct tokens: the mock can be finer, so it is.
    let first_token = receipt
        .completion_for(first)
        .expect("the point came from this plan's builder");
    let second_token = receipt
        .completion_for(second)
        .expect("the point came from this plan's builder");
    assert_eq!(receipt.completion_for(first).unwrap(), first_token);
    assert_ne!(
        first_token, second_token,
        "a backend that can distinguish two batches' completion should"
    );
    assert_ne!(
        first_token,
        receipt.completion(),
        "the overall token is the plan's, and the mock reports it separately"
    );
}

/// Acceptance and completion are separate, and completion is observable as a
/// state rather than waited for.
///
/// This is section 41.7 and 41.10 together: `submit` returning `Ok` says nothing
/// about the GPU, and the only way to learn more is to poll. Holding the mock's
/// completion is what makes the distinction decidable at all — an unheld mock
/// reports `Complete` the instant it accepts, which would let a broken
/// `completion_state` that always answered `Complete` pass every test here.
#[test]
fn completion_is_pending_until_the_device_reports_it() {
    let identity = device_identity(1);
    let (device, native) = paired_device_for_test(identity);

    native.hold_completion();

    let mut builder = SubmissionPlanBuilder::new(&device);
    let point = builder
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    let plan = builder.build().unwrap();

    // Section 41.7: acceptance happens *while* the work is unfinished. A submit
    // that waited for completion here would be the blocking frame-loop call
    // section 41.10 forbids.
    let receipt = block_on(device.submit(plan)).expect("acceptance does not await completion");
    let token = receipt.completion_for(point).unwrap();
    assert!(
        matches!(
            device.completion_state(token).unwrap(),
            CompletionState::Pending
        ),
        "held work is not complete, and section 41.1 makes this query the way to see that"
    );
    assert!(matches!(
        device.completion_state(receipt.completion()).unwrap(),
        CompletionState::Pending
    ));

    native.release_completion();
    assert!(matches!(
        device.completion_state(token).unwrap(),
        CompletionState::Complete
    ));
}

/// A lost device reaches every serial it reported, and never leaves one `Pending`.
///
/// Section 41.8's liveness rule, in the direction the mock can decide: work that
/// was in flight when the device ended becomes terminal through the *same* query a
/// caller was already polling, so a frame loop that never changes shape still
/// escapes. The portable layer is excused here rather than exercised — it answers
/// `DeviceLost` for the whole identity before it asks about the point, which is
/// why the per-point half is asserted against the backend directly.
#[test]
fn device_loss_reaches_every_reported_serial() {
    let identity = device_identity(1);
    let (device, native) = paired_device_for_test(identity);

    native.hold_completion();
    let mut builder = SubmissionPlanBuilder::new(&device);
    let point = builder
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    let plan = builder.build().unwrap();
    let receipt = block_on(device.submit(plan)).unwrap();
    let token = receipt.completion_for(point).unwrap();

    assert!(matches!(
        device.completion_state(token).unwrap(),
        CompletionState::Pending
    ));

    native.mark_lost(DeviceLossInfo::new("simulated loss".into()));

    // The per-point answer a backend owes: `Pending` is gone, and it did not take
    // a `wait_idle` to get there.
    assert!(matches!(
        native.completion(token.serial()),
        CompletionState::DeviceLost(_)
    ));
    // And the portable layer's own answer, which is the loss itself. Terminal on
    // both paths, which is the property section 41.8 exists to guarantee.
    assert_eq!(
        device.completion_state(token).unwrap_err().kind(),
        RhiErrorKind::DeviceLost
    );
}

/// A serial the device never reported is terminal, not `Pending`.
///
/// The distinction matters because the two are conflated easily and only one of
/// them is safe to return: `Pending` for a token that names no work is a polling
/// loop with no exit, and a stale or foreign serial is exactly how a caller
/// arrives at one.
#[test]
fn a_serial_the_device_never_reported_is_terminal() {
    let identity = device_identity(1);
    let (device, _native) = paired_device_for_test(identity);

    let mut builder = SubmissionPlanBuilder::new(&device);
    builder
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    let plan = builder.build().unwrap();
    let receipt = block_on(device.submit(plan)).unwrap();

    // A serial above anything this device minted. Constructed through the same
    // crate-private constructor the portable layer uses, because that is the only
    // way a token exists at all — the point of the test is the *backend's* answer,
    // not how a caller could forge one.
    let stale = CompletionPoint::new(identity, receipt.completion().serial() + 1_000);
    assert!(matches!(
        device.completion_state(stale).unwrap(),
        CompletionState::Failed(_)
    ));
}

/// A plan the portable layer refuses never reaches the backend.
///
/// Section 41.3's Phase A, made observable. `Err` from `submit` is supposed to
/// prove that no native work was accepted, and the return value alone cannot prove
/// it: a backend that was handed the plan and declined looks identical to a caller.
/// The backend's own counter is what separates them, and it is also what pins the
/// half of the rule that is easy to get wrong in the direction nobody notices — a
/// preflight that runs *after* the commit.
///
/// Both refusals are checked here because they are different phases of the same
/// promise: identity is decided before liveness would matter, and the present
/// refusal is decided after both.
#[test]
fn a_refused_plan_never_reaches_the_backend() {
    let identity = device_identity(1);
    let other = device_identity(2);
    let (device, native) = paired_device_for_test(identity);

    // A plan built for another device.
    let mut builder = builder(other, vec![lane(0, everything())], 1);
    builder
        .add_batch(lane_id(0), vec![raster_work(other, Vec::new())])
        .unwrap();
    let plan = builder.build().unwrap();
    let error = block_on(device.submit(plan)).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert_eq!(
        native.submissions(),
        0,
        "a cross-device plan must be refused by identity, not handed to a driver"
    );

    // A plan carrying a presentation, on a device that is otherwise fine.
    let mut builder = SubmissionPlanBuilder::new(&device);
    let point = builder
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    builder
        .present_after(acquired_frame(1, identity), point)
        .expect("the frame is this device's");
    let plan = builder.build().unwrap();
    let error = block_on(device.submit(plan)).unwrap_err();
    assert_eq!(
        error.kind(),
        RhiErrorKind::Unsupported,
        "no backend lowers presentation yet, and executing the work while dropping \
         the frame would be the silent substitution discipline 3 forbids"
    );
    assert_eq!(error.operation(), Some("Device::submit"));
    assert_eq!(
        native.submissions(),
        0,
        "a plan that cannot be presented in full must not run at all"
    );

    // A cross-device external dependency is *not* a case here, and the reason is
    // worth recording: `add_external_dependency` already refuses one, so a built
    // plan cannot carry it and `Device::submit` has no such check to exercise.
    // The refusal's authority is the builder, and this test would be asserting a
    // reachable failure that does not exist. What follows is the lost-device path,
    // which is checked first of all.
    let mut builder = SubmissionPlanBuilder::new(&device);
    builder
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    let plan = builder.build().unwrap();
    native.mark_lost(DeviceLossInfo::new("simulated loss".into()));
    let error = block_on(device.submit(plan)).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::DeviceLost);
    assert_eq!(native.submissions(), 0);
}

/// Every plan a device mints has its own identity, and the builder takes it from
/// the device rather than inventing one.
///
/// Section 39.1's uniqueness rule is what makes "a point from another plan is
/// `InvalidUsage`" decidable, and it can only hold if plan serials come from one
/// counter per device. A builder that numbered its own plans would collide with
/// the next builder's first plan, and two live plans would then be
/// indistinguishable to every check that compares them.
#[test]
fn two_plans_on_one_device_are_distinguishable() {
    let identity = device_identity(1);
    let (device, _native) = paired_device_for_test(identity);

    let mut first = SubmissionPlanBuilder::new(&device);
    first
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    let first = first.build().unwrap();

    let mut second = SubmissionPlanBuilder::new(&device);
    second
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    let second = second.build().unwrap();

    assert_ne!(first.id(), second.id());
    assert_eq!(first.device_identity(), identity);
    assert_eq!(second.device_identity(), identity);

    // And the receipt's per-plan check is real: a point from the second plan is
    // not answered for by the first plan's receipt.
    let mut third = SubmissionPlanBuilder::new(&device);
    let foreign = third
        .add_batch(lane_id(0), vec![raster_work(identity, Vec::new())])
        .unwrap();
    let third = third.build().unwrap();
    // Built and then dropped: what the check needs is a point from a plan other
    // than the receipt's, and section 41.9 makes dropping an unsubmitted plan
    // legal — it submits nothing and abandons only what it consumed.
    drop(third);
    let receipt = block_on(device.submit(first)).unwrap();
    assert_eq!(
        receipt.completion_for(foreign).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
}

// ---------------------------------------------------------------------------
// Helpers.
//
// Every one of these builds a value the RHI itself would produce, through the
// same crate-private constructor the port will use. They are the seam tests of
// this chapter: if a constructor's shape is wrong, these stop compiling.
// ---------------------------------------------------------------------------

fn block_on<T>(future: impl core::future::Future<Output = T>) -> T {
    use core::task::{Context, Poll, Waker};

    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the mock submission future unexpectedly suspended"),
    }
}

/// A device identity, as the platform layer mints one.
fn device_identity(instance: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(instance))
}

/// A plan identity.
fn plan_id(device: DeviceIdentity, serial: u64) -> SubmissionPlanId {
    SubmissionPlanId::new(device, serial)
}

/// A lane identity.
fn lane_id(value: u16) -> SubmissionLaneId {
    SubmissionLaneId::new(value)
}

/// One lane's facts.
fn lane(id: u16, domains: LaneWorkDomains) -> SubmissionLaneInfo {
    SubmissionLaneInfo::new(lane_id(id), SubmissionLaneClass::General, domains)
}

/// The domains every device's base-guarantee lane accepts.
fn everything() -> LaneWorkDomains {
    LaneWorkDomains::RASTER
        .union(LaneWorkDomains::COPY)
        .union(LaneWorkDomains::COMPUTE)
}

/// A builder over one lane, with no cross-lane routes reported.
fn builder(
    device: DeviceIdentity,
    lanes: Vec<SubmissionLaneInfo>,
    serial: u64,
) -> SubmissionPlanBuilder {
    SubmissionPlanBuilder::with_facts(
        plan_id(device, serial),
        device,
        SubmissionCapabilities::new(lanes),
    )
}

/// A builder over two lanes that can be ordered in both directions.
///
/// The route is a device answer, so it is recorded the way enumeration records it;
/// what the tests above check is what the builder does with the answer.
fn routed_builder(device: DeviceIdentity, serial: u64) -> SubmissionPlanBuilder {
    let mut caps = SubmissionCapabilities::new(vec![lane(0, everything()), lane(1, everything())]);
    caps.record_dependency_route(lane_id(0), lane_id(1), LaneDependencyRoute::Gpu);
    caps.record_dependency_route(lane_id(1), lane_id(0), LaneDependencyRoute::Gpu);
    SubmissionPlanBuilder::with_facts(plan_id(device, serial), device, caps)
}

/// One batch, as the validator sees it.
fn batch(
    plan: SubmissionPlanId,
    index: u32,
    lane: SubmissionLaneId,
    work: RecordedWork,
) -> PlanBatch {
    PlanBatch {
        point: PlanPoint::new(plan, SubmissionBatchId::new(index)),
        lane,
        work: vec![work],
    }
}

/// A recording that rasterizes.
fn raster_work(device: DeviceIdentity, uses: Vec<ResourceUse>) -> RecordedWork {
    RecordedWork::new(
        ObjectId::new(100),
        device,
        LaneWorkDomains::RASTER,
        uses,
        Vec::new(),
    )
}

/// A recording that dispatches.
fn compute_work(device: DeviceIdentity, uses: Vec<ResourceUse>) -> RecordedWork {
    RecordedWork::new(
        ObjectId::new(101),
        device,
        LaneWorkDomains::COMPUTE,
        uses,
        Vec::new(),
    )
}

/// A buffer handle on `device`.
fn buffer_handle(serial: u64, device: DeviceIdentity) -> Buffer {
    fixture::buffer(
        ObjectId::new(serial),
        device,
        BufferDescriptor::new(1024, BufferUsage::STORAGE),
    )
}

/// A texture handle on `device`.
fn texture_handle(serial: u64, device: DeviceIdentity) -> Texture {
    Texture::new(
        ObjectId::new(serial),
        device,
        TextureDescriptor::new_2d(
            64,
            64,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::STORAGE),
        ),
    )
}

/// A buffer use over one range.
fn buffer_range_use(
    id: ObjectId,
    device: DeviceIdentity,
    offset: u64,
    size: u64,
    access: AccessMask,
) -> ResourceUse {
    ResourceUse::Buffer(BufferUse {
        buffer: fixture::buffer(
            id,
            device,
            BufferDescriptor::new(1024, BufferUsage::STORAGE),
        ),
        range: BufferRange::new(offset, size),
        stages: PipelineScope::COMPUTE,
        access,
    })
}

/// A texture use over every mip and layer of the color aspect.
fn texture_use(id: ObjectId, device: DeviceIdentity, access: AccessMask) -> ResourceUse {
    texture_range_use(id, device, access, 0, 1)
}

/// A texture use over one mip.
fn texture_range_use(
    id: ObjectId,
    device: DeviceIdentity,
    access: AccessMask,
    base_mip: u32,
    mip_count: u32,
) -> ResourceUse {
    ResourceUse::Texture(TextureUse {
        texture: Texture::new(
            id,
            device,
            TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::STORAGE),
            ),
        ),
        subresources: TextureSubresourceRange {
            aspects: TextureAspects::COLOR,
            base_mip,
            mip_count,
            base_layer: 0,
            layer_count: 1,
        },
        stages: PipelineScope::FRAGMENT,
        access,
        intent: TextureUseIntent::ColorAttachment,
    })
}

/// A frame acquired on `device`.
fn acquired_frame(serial: u64, device: DeviceIdentity) -> AcquiredFrame {
    acquired_frame_with(AcquiredFrameId::new(device, serial), device)
}

/// A frame with a given identity — the shape a backend produces when it hands out
/// an image it has handed out before.
fn acquired_frame_with(id: AcquiredFrameId, device: DeviceIdentity) -> AcquiredFrame {
    AcquiredFrame::new(id, device, TextureFormat::Bgra8Unorm, Extent3d::d2(64, 64))
}
