//! The submission plan builder and the validation that produces a plan
//! (specification section 40).
//!
//! What a caller assembles before anything is submitted: batches on logical lanes,
//! happens-before edges between them, edges from already-submitted work, and the
//! frames a plan presents. It does not own the plan's identity or its validated
//! shape ([`super::plan`]), the accepted-work bookkeeping a submission produces
//! ([`super::completion`]), or the frames themselves
//! ([`crate::api::presentation`] owns the frame lifecycle, and this builder only
//! takes ownership of one).
//!
//! Invariant, and the reason this type is not just a `Vec` of batches:
//!
//! ```text
//! a plan that exists has passed every check build() can make        (40.5)
//! two batches with no happens-before between them may not conflict   (40.4)
//! a frame is used only inside the closure that presents it           (45.3)
//! ```
//!
//! # Which check happens when
//!
//! Section 40.5 lists ten checks that must be complete *before any native
//! submission*, and section 41.3 puts the same list in `Device::submit`'s Phase A.
//! They are not all decidable in the same place, and saying which is which is the
//! point of this file:
//!
//! ```text
//! insertion time   batch rules, point ownership, lane route, frame ownership
//! build()          the dependency graph, the hazard analysis, the frame closure
//! submit() Phase A identity, liveness, and the cross-plan hazard of section 41.4
//! the backend      whether an external dependency's route can actually be proven
//! ```
//!
//! The last line is the one gap that is not an omission: section 40.3 asks whether
//! the *source* and destination execution domains can be ordered, and a
//! [`CompletionPoint`] carries no lane — only a device and a serial (section 39.2).
//! The source lane of a plan that already ran is accepted-work history the backend
//! keeps, so `add_external_dependency` records the edge and its device identity
//! check is the part this layer can prove. See its documentation.
//!
//! # Two constructors
//!
//! [`SubmissionPlanBuilder::new`] is the public one and reads two facts off the
//! device handle rather than taking them from the caller: the device's lane table
//! and a device-scoped plan serial. Both are device state — section 40.1 decides
//! against the first and section 39.1 mints from the second — so a caller that
//! supplied either could construct a plan the device would never have issued.
//! The crate-private `with_facts` is the one that actually takes them, and it is
//! what `new` calls and what the contract tests drive.
//!
//! Notably absent from `new`: the plan serial. It is *drawn* from the device, not
//! taken even implicitly, because section 39.1's uniqueness rule is per device and
//! can only hold if one counter serves every builder over it.

use crate::api::command::RecordedWork;
use crate::api::command::{AccessMask, ResourceUse};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceIdentity;
use crate::api::platform::Device;
use crate::api::presentation::{AcquiredFrame, AcquiredFrameId, PresentPlanId};
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::subresource::{TextureAspects, TextureSubresourceRange};
use crate::api::resource::transient::{TransientAllocator, TransientLifetime};
use crate::api::submission::plan::{
    CompletionPoint, PlanBatch, PlanBody, PlanPoint, PlanPresent, SubmissionPlan, SubmissionPlanId,
};
use crate::api::submission::{
    LaneDependencyRoute, SubmissionBatchId, SubmissionCapabilities, SubmissionLaneId,
    TransientLifetimeRegistry,
};

/// Builds one submission plan.
///
/// Section 40's surface, and the only way a [`SubmissionPlan`] comes into being.
/// A builder is bound to one device for its whole life (section 40's own
/// description: "opaque, bound to `DeviceIdentity + SubmissionPlanId`"), so every
/// point it hands out names a batch of one plan on one device and every frame it
/// consumes must belong to that device.
///
/// # Ownership of consumed frames
///
/// Section 41.9 makes this the owner of every [`AcquiredFrame`] that
/// `present_after` consumed, until ownership transfers into a successfully built
/// plan. There is deliberately **no `Drop` impl here**: `build(self)` has to be
/// able to move the frames out into the plan, and the abandonment bookkeeping of
/// section 41.9 is already the frame's own `Drop`, which every path out of this
/// builder reaches — an error from `build`, a dropped builder, or a dropped plan.
/// Adding a second implementation here would be a second place for a frame to end.
pub struct SubmissionPlanBuilder {
    /// The identity of the plan being built. Handed in by the device's plan-serial
    /// source; see [`SubmissionPlanId::new`].
    plan: SubmissionPlanId,
    /// The device this plan will be submitted to.
    device: DeviceIdentity,
    /// The live device used solely to realize Dedicated transient resources.
    /// Keeping this one handle also keeps its native allocation factory alive
    /// while a caller is assembling the plan.
    device_handle: Option<Device>,
    /// The device's lanes and their cross-lane routes, which section 40.1 and
    /// section 40.2 are both decided against.
    ///
    /// A snapshot rather than a `&Device`: section 7.2 makes enabled capabilities
    /// immutable device data, so there is nothing for a snapshot to go stale
    /// against, and holding one keeps this builder free of a lifetime.
    lanes: SubmissionCapabilities,
    /// The batches, in insertion order.
    batches: Vec<PlanBatch>,
    /// Explicit happens-before edges between batches of this plan.
    dependencies: Vec<(PlanPoint, PlanPoint)>,
    /// Edges from already-submitted work into this plan.
    external_dependencies: Vec<(CompletionPoint, PlanPoint)>,
    /// The presentations this plan carries, in the order they were added.
    presents: Vec<PlanPresent>,
    /// The frames `present_after` consumed, in the same order.
    frames: Vec<AcquiredFrame>,
    /// Every transient lifetime allocated for this plan, including resources
    /// that never enter recorded work.
    transient_lifetimes: TransientLifetimeRegistry,
}

impl SubmissionPlanBuilder {
    /// Opens a builder for one plan on `device`.
    ///
    /// Section 39.1 mints the plan identity here, and section 40.1 needs the
    /// device's lane table here, so this is the one constructor that reads both
    /// off the device handle. It returns `Self` rather than a `RhiResult` for the
    /// reason the module note gives: a device that is already lost is refused by
    /// [`Device::submit`], which is the verb that has an error channel, and a
    /// builder that refused early would take that decision away from it.
    ///
    /// Neither fact is copied from the caller. The serial comes from the device's
    /// own source, which is what makes section 39.1's "a `PlanPoint` from another
    /// plan is `InvalidUsage`" checkable at all — a caller-chosen serial would let
    /// two builders collide, and the private field would then be protecting
    /// nothing.
    pub fn new(device: &Device) -> Self {
        Self::with_device(
            SubmissionPlanId::new(device.identity(), device.serials().next_plan()),
            device,
            device.capabilities().submission().clone(),
        )
    }

    fn with_device(plan: SubmissionPlanId, device: &Device, lanes: SubmissionCapabilities) -> Self {
        Self {
            plan,
            device: device.identity(),
            device_handle: Some(device.clone()),
            lanes,
            batches: Vec::new(),
            dependencies: Vec::new(),
            external_dependencies: Vec::new(),
            presents: Vec::new(),
            frames: Vec::new(),
            transient_lifetimes: TransientLifetimeRegistry::default(),
        }
    }

    /// Opens a builder over facts the caller supplies.
    ///
    /// Crate-private, and the honest constructor: it takes everything this type
    /// stores, so it is what the port's `new` will call, and it is what the tests
    /// drive. It checks nothing, because every fact it is given is one the device
    /// enumeration already decided — a lane the device did not offer cannot be
    /// added here, and a plan identity from another device's serial source is
    /// `add_batch`'s problem to notice, not this constructor's.
    pub(crate) fn with_facts(
        plan: SubmissionPlanId,
        device: DeviceIdentity,
        lanes: SubmissionCapabilities,
    ) -> Self {
        Self {
            plan,
            device,
            device_handle: None,
            lanes,
            batches: Vec::new(),
            dependencies: Vec::new(),
            external_dependencies: Vec::new(),
            presents: Vec::new(),
            frames: Vec::new(),
            transient_lifetimes: TransientLifetimeRegistry::default(),
        }
    }

    /// Reserves an empty logical batch point before its recorded work exists.
    ///
    /// A reserved point is deliberately not a submit-able empty batch: it must be
    /// filled exactly once through [`Self::set_batch`] before [`Self::build`].
    /// Reserving points first lets transient lifetimes be expressed in the same
    /// `PlanPoint` vocabulary that submission already uses.
    pub fn reserve_batch(&mut self, lane: SubmissionLaneId) -> RhiResult<PlanPoint> {
        if self.lanes.lane(lane).is_none() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{lane:?} is not a lane this device offers"),
            )
            .at("SubmissionPlanBuilder::reserve_batch"));
        }
        let index = u32::try_from(self.batches.len()).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "this plan already holds the maximum number of batches",
            )
            .at("SubmissionPlanBuilder::reserve_batch")
        })?;
        let point = PlanPoint::new(self.plan, SubmissionBatchId::new(index));
        self.batches.push(PlanBatch {
            point,
            lane,
            work: Vec::new(),
        });
        Ok(point)
    }

    /// Fills one previously reserved batch point exactly once.
    pub fn set_batch(&mut self, point: PlanPoint, work: Vec<RecordedWork>) -> RhiResult<()> {
        let lane = self.lane_of(point, "SubmissionPlanBuilder::set_batch")?;
        if self
            .batches
            .iter()
            .find(|batch| batch.point == point)
            .is_some_and(|batch| !batch.work.is_empty())
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a reserved batch may only be filled once",
            )
            .at("SubmissionPlanBuilder::set_batch"));
        }
        self.validate_batch_work(lane, &work, "SubmissionPlanBuilder::set_batch")?;
        self.batches
            .iter_mut()
            .find(|batch| batch.point == point)
            .expect("lane_of proved the point is a batch of this plan")
            .work = work;
        Ok(())
    }

    /// Adds a batch of recorded work to one lane.
    ///
    /// Section 40.1's rules, in its own order, with the kind each one produces:
    ///
    /// ```text
    /// work is non-empty                                  else InvalidUsage
    /// the lane is one this device offers                 else WrongDevice
    /// every work belongs to this device                  else WrongDevice
    /// the lane accepts every work's domains              else InvalidUsage
    /// ```
    ///
    /// The lane rule is [`WrongDevice`](RhiErrorKind::WrongDevice) rather than
    /// `Unsupported` because a lane identity is device-scoped: a `SubmissionLaneId`
    /// is only ever obtained from a device's own enumeration, so a lane this device
    /// does not offer came from another device. The domain rule is
    /// [`InvalidUsage`](RhiErrorKind::InvalidUsage) because it is the caller's own
    /// mismatch — the lane exists and would take this work if the work were the
    /// kind it accepts — and section 40.1 states it as a rule about the *batch*.
    ///
    /// The returned point is what every later verb names this batch by. Insertion
    /// order on one lane is that lane's logical submission order (section 40.1):
    /// the builder does not reorder, and a dependency edge is never needed to
    /// order two batches of one lane.
    pub fn add_batch(
        &mut self,
        lane: SubmissionLaneId,
        work: Vec<RecordedWork>,
    ) -> RhiResult<PlanPoint> {
        self.validate_batch_work(lane, &work, "SubmissionPlanBuilder::add_batch")?;
        let point = self.reserve_batch(lane)?;
        // Validation happened before reserving, so this cannot leave a failed
        // convenience call behind as an unfilled point.
        self.set_batch(point, work)?;
        Ok(point)
    }

    /// Returns an allocator scoped to this builder's Device and plan identity.
    pub fn transient_allocator(&self) -> TransientAllocator<'_> {
        TransientAllocator::new_with_registry(
            self.device,
            self.plan,
            &self.transient_lifetimes,
            self.device_handle.as_ref(),
        )
    }

    fn validate_batch_work(
        &self,
        lane: SubmissionLaneId,
        work: &[RecordedWork],
        operation: &'static str,
    ) -> RhiResult<()> {
        if work.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a batch must contain at least one recorded work; an empty batch has \
                 nothing to submit and no reason to exist",
            )
            .at(operation));
        }
        let accepted = match self.lanes.lane(lane) {
            Some(info) => info.domains(),
            None => {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    format!(
                        "{lane:?} is not a lane this device offers; a lane identity is \
                         device-scoped, so this one belongs to another device"
                    ),
                )
                .at(operation));
            }
        };
        for item in work {
            if item.device_identity() != self.device {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    format!(
                        "recorded work {:?} was recorded on another device; section 3.3 gives \
                         P0 no path that moves work between devices",
                        item.id()
                    ),
                )
                .at(operation));
            }
        }
        for item in work {
            let domains = item.work_domains();
            if !accepted.contains(domains) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "work {:?} contains {domains}, and this lane accepts {accepted}; add it \
                         to a lane whose domains cover it",
                        item.id()
                    ),
                )
                .at(operation));
            }
        }
        Ok(())
    }

    /// Adds happens-before from `before` to `after`.
    ///
    /// Section 40.2 is the whole rule: two different lanes are unordered by
    /// default, and this call is the only thing that creates a portable
    /// happens-before between them. Whether it *can* be created is the device's
    /// answer, read here through
    /// [`SubmissionCapabilities::dependency_route`](crate::api::submission::SubmissionCapabilities::dependency_route):
    ///
    /// ```text
    /// Ordered     accepted; the two batches are already in one ordered domain
    /// Gpu         accepted; the backend lowers a GPU-side dependency
    /// Collapse    accepted; the backend collapses the lanes, keeping the order
    /// Unsupported refused with Unsupported — restructure, or observe completion
    /// ```
    ///
    /// `Unsupported` is the device's honest answer and not a defect to work around:
    /// a caller that needs the order and cannot get it either puts both batches on
    /// one lane (the base guarantee of section 10 gives every device at least one
    /// lane that takes raster and copy work) or observes the first batch's
    /// completion on the host before submitting the next plan.
    ///
    /// A point from another plan is [`InvalidUsage`](RhiErrorKind::InvalidUsage)
    /// rather than a cross-device error, per section 39.1, and a self-loop is
    /// refused for the same reason: an edge that says a batch runs before itself
    /// states no order at all.
    pub fn add_dependency(&mut self, before: PlanPoint, after: PlanPoint) -> RhiResult<()> {
        let from = self.lane_of(before, "SubmissionPlanBuilder::add_dependency")?;
        let to = self.lane_of(after, "SubmissionPlanBuilder::add_dependency")?;
        if before == after {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a batch cannot depend on itself",
            )
            .at("SubmissionPlanBuilder::add_dependency"));
        }
        if self.lanes.dependency_route(from, to) == LaneDependencyRoute::Unsupported {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device cannot establish happens-before from lane {from:?} to lane \
                     {to:?}; put both batches on one lane, or observe the earlier batch's \
                     completion before submitting it"
                ),
            )
            .at("SubmissionPlanBuilder::add_dependency"));
        }
        self.dependencies.push((before, after));
        Ok(())
    }

    /// Establishes happens-before from already-submitted work into a batch.
    ///
    /// Section 40.3 is why this verb exists: one logical lane stays lane-ordered
    /// across `Device::submit` calls, but two *different* lanes across two plans
    /// have no implicit order at all, so a frame that used one lane last plan and
    /// another this plan needs an edge.
    ///
    /// What this layer decides is the portable half:
    ///
    /// ```text
    /// the completion token's device is this builder's   else WrongDevice
    /// the batch belongs to this plan                    else InvalidUsage
    /// ```
    ///
    /// What it deliberately does not decide is the route, and that is a gap with a
    /// reason rather than an omission. Section 40.3 accepts the edge when the
    /// backend can lower a GPU-side wait *or* when the source and destination
    /// execution domains are already ordered, including a proven `Collapse` — but
    /// the source domain is the lane of a plan that already ran, and a
    /// [`CompletionPoint`] carries a device and a serial, not a lane (section 39.2).
    /// Which lane that completion came from is accepted-work history the backend
    /// keeps for section 41.4's hazard check, so the route is proven during
    /// `Device::submit`'s Phase A and a device that can prove no route answers
    /// `Unsupported` there — where section 41.3 still guarantees that nothing was
    /// submitted. A caller must not read an `Ok` here as a promise that the edge
    /// will hold; it is a promise that the request is well formed.
    pub fn add_external_dependency(
        &mut self,
        before: CompletionPoint,
        after: PlanPoint,
    ) -> RhiResult<()> {
        self.lane_of(after, "SubmissionPlanBuilder::add_external_dependency")?;
        if before.device_identity() != self.device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "this completion token belongs to another device; work cannot be ordered \
                 across devices",
            )
            .at("SubmissionPlanBuilder::add_external_dependency"));
        }
        self.external_dependencies.push((before, after));
        Ok(())
    }

    /// Adds the presentation of `frame` after `after`, consuming the frame.
    ///
    /// Section 45.1 makes this where a frame changes hands: `Acquired` becomes
    /// `PlannedForPresent`, the builder owns the frame from here (section 41.9),
    /// and the returned [`PresentPlanId`] is the plan-local name of this
    /// presentation — what a later [`PresentReceipt`](crate::api::presentation::PresentReceipt)
    /// reports about.
    ///
    /// ```text
    /// the after-point belongs to this plan        else InvalidUsage   (40.5)
    /// the frame belongs to this device            else WrongDevice
    /// the frame is in the Acquired state          else InvalidUsage   (44.6)
    /// ```
    ///
    /// The after-point is validated rather than defaulted: "present at the end" and
    /// "present after this batch" are different plans whenever the plan keeps
    /// submitting after the present, and section 45.3 refuses a plan whose work uses
    /// the frame after the point it is presented at.
    ///
    /// That two presentations of one frame cannot both exist is *not* checked here
    /// — it is checked in `build`, because section 40.5 puts "each `AcquiredFrame`
    /// consumed only once" in build's checklist, and because it is a property of the
    /// plan rather than of this call.
    pub fn present_after(
        &mut self,
        frame: AcquiredFrame,
        after: PlanPoint,
    ) -> RhiResult<PresentPlanId> {
        self.lane_of(after, "SubmissionPlanBuilder::present_after")?;
        if frame.device_identity() != self.device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "this frame was acquired on another device, so this plan cannot present it",
            )
            .at("SubmissionPlanBuilder::present_after"));
        }
        let frame_id = frame.id();
        let index = u32::try_from(self.presents.len()).map_err(|_| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "this plan already presents the maximum number of frames",
            )
            .at("SubmissionPlanBuilder::present_after")
        })?;
        let mut frame = frame;
        frame.mark_planned_for_present()?;
        let id = PresentPlanId::new(self.plan, index);
        self.presents.push(PlanPresent {
            id,
            frame: frame_id,
            after,
        });
        self.frames.push(frame);
        Ok(id)
    }

    /// Validates the plan and closes the builder.
    ///
    /// Section 40.5's checklist, with the parts that are decidable here. The
    /// insertion-time checks on that list are not repeated: a batch's lane and
    /// domains, a point's plan, and a frame's device were each refused at the call
    /// that supplied them, and no verb in this type can change them afterwards.
    ///
    /// What runs here, in order:
    ///
    /// ```text
    /// every dependency names a batch of this plan          else InvalidUsage
    /// the dependency graph has no cycle                    else InvalidUsage
    /// no unordered pair of batches conflicts               else MissingDependency
    /// each frame is presented at most once                 else InvalidUsage
    /// every frame use is inside its present closure        else InvalidUsage
    /// ```
    ///
    /// On `Err` the builder is dropped, and with it every frame it consumed — which
    /// is section 41.9's no-submit abandonment path, performed by each frame's own
    /// `Drop`. No submission happens on that path, by construction: nothing here
    /// touches a device.
    ///
    /// Sections 40.5's liveness and object-identity rows are *not* here: this type
    /// holds a `DeviceIdentity` and not a `Device`, and `Device::submit`'s Phase A
    /// is where a live device answers them.
    pub fn build(self) -> RhiResult<SubmissionPlan> {
        if self.batches.iter().any(|batch| batch.work.is_empty()) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "every reserved batch must be filled exactly once before building a plan",
            )
            .at("SubmissionPlanBuilder::build"));
        }
        validate_plan_graph(&self.batches, &self.dependencies, &self.presents)?;
        let transient_lifetimes = self.transient_lifetimes.snapshot();
        validate_transient_lifetimes(
            &self.batches,
            &self.dependencies,
            self.plan,
            &transient_lifetimes,
        )?;
        validate_transient_uses(&self.batches, &self.dependencies, self.plan)?;
        Ok(SubmissionPlan::new(
            self.plan,
            self.device,
            PlanBody {
                batches: self.batches,
                dependencies: self.dependencies,
                external_dependencies: self.external_dependencies,
                presents: self.presents,
                frames: self.frames,
            },
        ))
    }

    /// Finds the batch a point names, refusing a point from another plan.
    ///
    /// Section 39.1's rule, in one place: a `PlanPoint` from builder A passed to
    /// builder B is [`RhiErrorKind::InvalidUsage`], and so is a point that names a
    /// batch this plan never handed out. Both are the same check, because both are
    /// "this point is not one of mine".
    fn lane_of(&self, point: PlanPoint, operation: &'static str) -> RhiResult<SubmissionLaneId> {
        if point.plan() != self.plan {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "this plan point belongs to a different plan",
            )
            .at(operation));
        }
        self.batches
            .iter()
            .find(|batch| batch.point == point)
            .map(|batch| batch.lane)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this plan point does not name a batch of this plan",
                )
                .at(operation)
            })
    }
}

fn validate_transient_lifetimes(
    batches: &[PlanBatch],
    dependencies: &[(PlanPoint, PlanPoint)],
    plan: SubmissionPlanId,
    lifetimes: &[TransientLifetime],
) -> RhiResult<()> {
    let successors = plan_successors(batches, dependencies)?;
    for lifetime in lifetimes {
        if lifetime.acquire().plan() != plan {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a transient lifetime belongs to a different submission plan",
            )
            .at("SubmissionPlanBuilder::build"));
        }
        let acquire = index_of(batches, lifetime.acquire()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a transient lifetime acquire point is not a batch of this plan",
            )
            .at("SubmissionPlanBuilder::build")
        })?;
        if lifetime.release_frontier().is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a transient lifetime requires at least one release frontier point",
            )
            .at("SubmissionPlanBuilder::build"));
        }
        for release in lifetime.release_frontier() {
            let release = index_of(batches, *release).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a transient lifetime release point is not a batch of this plan",
                )
                .at("SubmissionPlanBuilder::build")
            })?;
            if !reaches(&successors, acquire, release) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a transient acquire point must happen before every release frontier point",
                )
                .at("SubmissionPlanBuilder::build"));
            }
        }
    }
    Ok(())
}

/// Validates the plan-scoped lifetime carried by every transient resource use.
///
/// This is intentionally submission validation, rather than a resource-creation
/// check: only the completed batch DAG knows whether an actual use lies between
/// the acquire point and at least one release frontier point.
fn validate_transient_uses(
    batches: &[PlanBatch],
    dependencies: &[(PlanPoint, PlanPoint)],
    plan: SubmissionPlanId,
) -> RhiResult<()> {
    let successors = plan_successors(batches, dependencies)?;

    for (use_index, batch) in batches.iter().enumerate() {
        for use_record in batch.work.iter().flat_map(RecordedWork::resource_uses) {
            let lifetime = match use_record {
                ResourceUse::Buffer(use_record) => use_record.buffer.transient_lifetime(),
                ResourceUse::Texture(use_record) => use_record.texture.transient_lifetime(),
                ResourceUse::Frame(_) => None,
            };
            let Some(lifetime) = lifetime else {
                continue;
            };

            if lifetime.acquire().plan() != plan {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a transient resource belongs to a different submission plan",
                )
                .at("SubmissionPlanBuilder::build"));
            }
            let acquire = index_of(batches, lifetime.acquire()).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a transient lifetime acquire point is not a batch of this plan",
                )
                .at("SubmissionPlanBuilder::build")
            })?;
            if lifetime.release_frontier().is_empty() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a transient lifetime requires at least one release frontier point",
                )
                .at("SubmissionPlanBuilder::build"));
            }

            let mut use_reaches_release = false;
            for release in lifetime.release_frontier() {
                let release = index_of(batches, *release).ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a transient lifetime release point is not a batch of this plan",
                    )
                    .at("SubmissionPlanBuilder::build")
                })?;
                if !reaches(&successors, acquire, release) {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a transient acquire point must happen before every release frontier point",
                    )
                    .at("SubmissionPlanBuilder::build"));
                }
                use_reaches_release |=
                    use_index == release || reaches(&successors, use_index, release);
            }
            if use_index != acquire && !reaches(&successors, acquire, use_index) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a transient resource use occurs before or unordered with its acquire point",
                )
                .at("SubmissionPlanBuilder::build"));
            }
            if !use_reaches_release {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a transient resource use cannot reach any release frontier point",
                )
                .at("SubmissionPlanBuilder::build"));
            }
        }
    }
    Ok(())
}

fn plan_successors(
    batches: &[PlanBatch],
    dependencies: &[(PlanPoint, PlanPoint)],
) -> RhiResult<Vec<Vec<usize>>> {
    let mut successors = vec![Vec::new(); batches.len()];
    for (index, batch) in batches.iter().enumerate() {
        for earlier in (0..index).rev() {
            if batches[earlier].lane == batch.lane {
                successors[earlier].push(index);
                break;
            }
        }
    }
    for (before, after) in dependencies {
        let from = index_of(batches, *before).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a dependency names a plan point that is not a batch of this plan",
            )
            .at("SubmissionPlanBuilder::build")
        })?;
        let to = index_of(batches, *after).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a dependency names a plan point that is not a batch of this plan",
            )
            .at("SubmissionPlanBuilder::build")
        })?;
        successors[from].push(to);
    }
    Ok(successors)
}

impl core::fmt::Debug for SubmissionPlanBuilder {
    /// Prints portable identity and the size of the plan being built.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan: the backend port adds the native submission machinery
    /// this builder will hand work to, and printing that into a log is a leak.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmissionPlanBuilder")
            .field("plan", &self.plan)
            .field("device", &self.device)
            .field("batches", &self.batches.len())
            .field("dependencies", &self.dependencies.len())
            .field("external_dependencies", &self.external_dependencies.len())
            .field("presents", &self.presents.len())
            .finish_non_exhaustive()
    }
}

/// Checks the plan graph a builder assembled.
///
/// The three rules that need the whole plan rather than one call, in section 40.5's
/// order. Crate-private because it is the *rule* and not a verb: section 40 gives a
/// caller exactly one way to reach a plan, and `build` is it.
///
/// ```text
/// every edge names two batches of this plan                        InvalidUsage
/// the graph, implicit lane order included, has no cycle            InvalidUsage
/// no unordered pair of batches has a conflicting use               MissingDependency
/// each frame is presented at most once                             InvalidUsage
/// every frame use is ordered before its frame's present point      InvalidUsage
/// ```
///
/// The kinds are section 4's mapping applied to what each rule is about. A cycle,
/// a foreign point, and a frame used outside its closure are all *plan structure*
/// mistakes, which section 4 makes `InvalidUsage`. A conflicting pair is the one
/// case the specification names outright — section 40.4 says
/// `Err(RhiErrorKind::MissingDependency)` in as many words, because the plan is
/// executable and merely unsynchronized: the caller's fix is to add the edge, not
/// to change the work.
///
/// Section 45.3 names no kind for its two rows, and the inventory records both as
/// an inferred `InvalidUsage` (04-05 S37-S41). That is the reading taken here: a
/// frame that is drawn into and never presented, or drawn into after the point it
/// is presented at, is not a missing edge between two batches — it is a plan whose
/// frame closure does not close.
///
/// It takes facts rather than a builder, so that the rules are reviewable and
/// exercisable without a device: every input is portable data.
pub(crate) fn validate_plan_graph(
    batches: &[PlanBatch],
    dependencies: &[(PlanPoint, PlanPoint)],
    presents: &[PlanPresent],
) -> RhiResult<()> {
    let count = batches.len();
    // `successors[i]` are the batches that happen after batch `i`.
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); count];

    // Implicit order: on one lane, insertion order is logical submission order
    // (section 40.1). Recorded as consecutive edges, whose transitive closure is
    // the same as connecting every earlier batch to every later one.
    for (index, batch) in batches.iter().enumerate() {
        for earlier in (0..index).rev() {
            if batches[earlier].lane == batch.lane {
                successors[earlier].push(index);
                break;
            }
        }
    }

    for (before, after) in dependencies {
        let from = index_of(batches, *before).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a dependency names a plan point that is not a batch of this plan",
            )
            .at("SubmissionPlanBuilder::build")
        })?;
        let to = index_of(batches, *after).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a dependency names a plan point that is not a batch of this plan",
            )
            .at("SubmissionPlanBuilder::build")
        })?;
        successors[from].push(to);
    }

    check_acyclic(&successors)?;

    for left in 0..count {
        for right in (left + 1)..count {
            if reaches(&successors, left, right) || reaches(&successors, right, left) {
                continue;
            }
            if let Some(conflict) = conflicting_use(&batches[left], &batches[right]) {
                return Err(RhiError::new(
                    RhiErrorKind::MissingDependency,
                    format!(
                        "batches {:?} and {:?} both touch {conflict}, at least one of them \
                         writing, and nothing orders them: add a dependency between them or \
                         put them on one lane",
                        batches[left].point, batches[right].point
                    ),
                )
                .at("SubmissionPlanBuilder::build"));
            }
        }
    }

    check_frame_closure(batches, presents, &successors)
}

/// Checks section 45.3's closure rules for every frame the plan presents.
///
/// ```text
/// at most one present plan per frame
/// a present point names a batch of this plan
/// a batch that uses a frame is ordered before that frame's present point
/// ```
///
/// The last rule is also the third one the section states, in its other
/// direction: if every frame use happens before the present point, then no work
/// uses the frame after it. The first rule is what makes a frame's present point
/// single-valued, without which neither direction could be stated; the second is
/// what makes it *ordered* at all, and it is checked before any frame use is
/// looked at rather than while walking one — a plan whose present point names no
/// batch of the plan is malformed whether or not anything draws into the frame.
fn check_frame_closure(
    batches: &[PlanBatch],
    presents: &[PlanPresent],
    successors: &[Vec<usize>],
) -> RhiResult<()> {
    // The batch each presentation is ordered after, resolved once: both the
    // single-present rule and the ordering rule below read it.
    let mut present_after: Vec<(AcquiredFrameId, usize)> = Vec::with_capacity(presents.len());

    for (index, present) in presents.iter().enumerate() {
        if presents[..index]
            .iter()
            .any(|earlier| earlier.frame == present.frame)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "frame {:?} is presented twice in one plan; a frame has one present point",
                    present.frame
                ),
            )
            .at("SubmissionPlanBuilder::build"));
        }
        let after = index_of(batches, present.after).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a present point does not name a batch of this plan",
            )
            .at("SubmissionPlanBuilder::build")
        })?;
        present_after.push((present.frame, after));
    }

    for (index, batch) in batches.iter().enumerate() {
        for use_record in batch.work.iter().flat_map(RecordedWork::resource_uses) {
            let ResourceUse::Frame(frame_use) = use_record else {
                continue;
            };
            let Some((_, after)) = present_after
                .iter()
                .find(|(frame, _)| *frame == frame_use.frame)
            else {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "batch {:?} draws into frame {:?}, and this plan never presents that \
                         frame; a plan that uses a frame is the plan that presents it",
                        batch.point, frame_use.frame
                    ),
                )
                .at("SubmissionPlanBuilder::build"));
            };
            if index != *after && !reaches(successors, index, *after) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "batch {:?} draws into frame {:?} but does not happen before the point \
                         that presents it; order it before that point, or present after it",
                        batch.point, frame_use.frame
                    ),
                )
                .at("SubmissionPlanBuilder::build"));
            }
        }
    }
    Ok(())
}

/// Describes the first resource two batches both touch with a write on one side.
///
/// Section 40.4's overlap test, per resource kind:
///
/// ```text
/// same buffer, overlapping byte range
/// same texture, overlapping mips, layers, and aspects
/// same frame
/// ```
///
/// `None` means the pair is safe to leave unordered — either they touch different
/// resources or every shared use is a read, which section 40.4 states needs no
/// dependency.
///
/// The returned text names the resource so that the refusal is actionable; it is
/// assembled only on the error path, which is why it is a `String` and not a
/// reference into the plan.
fn conflicting_use(left: &PlanBatch, right: &PlanBatch) -> Option<String> {
    for one in left.work.iter().flat_map(RecordedWork::resource_uses) {
        for other in right.work.iter().flat_map(RecordedWork::resource_uses) {
            if !writes(one) && !writes(other) {
                continue;
            }
            if let Some(described) = shared_resource(one, other) {
                return Some(described);
            }
        }
    }
    None
}

/// Names the one object both uses touch, when they touch the same one.
///
/// The per-kind overlap rules of section 40.4, kept apart from the read/write
/// question [`conflicting_use`] asks first. A pair from two different kinds is
/// never a shared object: buffer, texture, and frame identities are three separate
/// spaces, which is what section 44.3's "a `FrameAttachment` is not a `Texture`"
/// keeps true.
fn shared_resource(one: &ResourceUse, other: &ResourceUse) -> Option<String> {
    match (one, other) {
        (ResourceUse::Buffer(one), ResourceUse::Buffer(other)) => {
            if one.buffer.id() == other.buffer.id()
                && buffer_ranges_overlap(&one.range, &other.range)
            {
                return Some(format!(
                    "buffer {} over bytes {}..{} and {}..{}",
                    one.buffer.id().as_u64(),
                    one.range.offset,
                    one.range.offset.saturating_add(one.range.size),
                    other.range.offset,
                    other.range.offset.saturating_add(other.range.size),
                ));
            }
            None
        }
        (ResourceUse::Texture(one), ResourceUse::Texture(other)) => {
            if one.texture.id() == other.texture.id()
                && subresources_overlap(&one.subresources, &other.subresources)
            {
                return Some(format!(
                    "texture {} over mips {}..{} layers {}..{}",
                    one.texture.id().as_u64(),
                    one.subresources.base_mip,
                    one.subresources
                        .base_mip
                        .saturating_add(one.subresources.mip_count),
                    one.subresources.base_layer,
                    one.subresources
                        .base_layer
                        .saturating_add(one.subresources.layer_count),
                ));
            }
            None
        }
        (ResourceUse::Frame(one), ResourceUse::Frame(other)) => {
            if one.frame == other.frame {
                return Some(format!("frame {:?}", one.frame));
            }
            None
        }
        (ResourceUse::Buffer(_), ResourceUse::Texture(_))
        | (ResourceUse::Texture(_), ResourceUse::Buffer(_))
        | (ResourceUse::Buffer(_), ResourceUse::Frame(_))
        | (ResourceUse::Frame(_), ResourceUse::Buffer(_))
        | (ResourceUse::Texture(_), ResourceUse::Frame(_))
        | (ResourceUse::Frame(_), ResourceUse::Texture(_)) => None,
    }
}

/// Whether a use writes what it names.
///
/// Section 40.4's three hazardous combinations are `READ/WRITE`, `WRITE/READ`, and
/// `WRITE/WRITE`, so the whole test is "does either side write".
///
/// Presentation is not an access-mask bit. A frame's render use is expressed as
/// `COLOR_WRITE` on [`ResourceUse::Frame`]; presentation itself is a plan action.
fn writes(use_record: &ResourceUse) -> bool {
    let access = match use_record {
        ResourceUse::Buffer(use_record) => use_record.access,
        ResourceUse::Texture(use_record) => use_record.access,
        ResourceUse::Frame(use_record) => use_record.access,
    };
    [
        AccessMask::SHADER_WRITE,
        AccessMask::COLOR_WRITE,
        AccessMask::DEPTH_WRITE,
        AccessMask::STENCIL_WRITE,
        AccessMask::COPY_WRITE,
    ]
    .into_iter()
    .any(|bit| access.contains(bit))
}

/// Whether two byte ranges of one buffer share a byte.
///
/// A range whose end does not survive `offset + size` is treated as conflicting
/// with everything on that buffer. That range is not legal — section 12.4 makes
/// "no integer overflow" a rule every point of use checks — so this only decides
/// which refusal a caller sees first, and the conservative answer keeps the
/// hazard analysis from clearing a pair it cannot reason about.
fn buffer_ranges_overlap(one: &BufferRange, other: &BufferRange) -> bool {
    match (one.end(), other.end()) {
        (Some(one_end), Some(other_end)) => one.offset < other_end && other.offset < one_end,
        _ => true,
    }
}

/// Whether two subresource ranges of one texture share a texel.
///
/// All three axes must overlap for the two to conflict: they must share an aspect,
/// a mip level, and an array layer. The arithmetic is done in `u64` because a
/// `base + count` on `u32` fields can overflow, and a wrapped end would make two
/// disjoint ranges look adjacent.
fn subresources_overlap(one: &TextureSubresourceRange, other: &TextureSubresourceRange) -> bool {
    let spans = |base: u32, count: u32| (u64::from(base), u64::from(base) + u64::from(count));
    let (one_mip, one_mip_end) = spans(one.base_mip, one.mip_count);
    let (other_mip, other_mip_end) = spans(other.base_mip, other.mip_count);
    let (one_layer, one_layer_end) = spans(one.base_layer, one.layer_count);
    let (other_layer, other_layer_end) = spans(other.base_layer, other.layer_count);
    aspects_overlap(one.aspects, other.aspects)
        && one_mip < other_mip_end
        && other_mip < one_mip_end
        && one_layer < other_layer_end
        && other_layer < one_layer_end
}

/// Whether two aspect sets share an aspect.
///
/// A set intersection is not expressible through
/// [`TextureAspects::contains`](crate::api::resource::subresource::TextureAspects::contains)
/// and `union`, so the aspects the type declares are enumerated. A new aspect bit
/// has to be added here as well as there — which is the intended pressure, and the
/// same one this crate's match arms on its own enums apply.
fn aspects_overlap(one: TextureAspects, other: TextureAspects) -> bool {
    [
        TextureAspects::COLOR,
        TextureAspects::DEPTH,
        TextureAspects::STENCIL,
    ]
    .into_iter()
    .any(|aspect| one.contains(aspect) && other.contains(aspect))
}

/// Finds the batch a point names.
///
/// A linear scan rather than an index computed from the batch number: the plan's
/// batch numbering is an implementation detail of insertion order, and reading it
/// back here would make the graph analysis depend on that detail staying
/// sequential.
fn index_of(batches: &[PlanBatch], point: PlanPoint) -> Option<usize> {
    batches.iter().position(|batch| batch.point == point)
}

/// Checks that the happens-before graph has no cycle.
///
/// Section 40.5 requires this before any submission, and section 40.5's last line
/// says implicit same-lane order participates: a batch on lane A that depends on a
/// later batch of the same lane is a cycle even though no explicit edge closes it.
///
/// Kahn's algorithm rather than a recursive depth-first search, because the depth
/// of a search is the length of the plan's longest dependency chain and that is
/// caller-controlled. A plan that processes fewer nodes than it has is one whose
/// remaining nodes all have an unmet predecessor — which is exactly a cycle.
fn check_acyclic(successors: &[Vec<usize>]) -> RhiResult<()> {
    let mut waiting_on: Vec<usize> = vec![0; successors.len()];
    for edges in successors {
        for successor in edges {
            waiting_on[*successor] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..successors.len())
        .filter(|node| waiting_on[*node] == 0)
        .collect();
    let mut settled = 0;
    while let Some(node) = ready.pop() {
        settled += 1;
        for successor in &successors[node] {
            waiting_on[*successor] -= 1;
            if waiting_on[*successor] == 0 {
                ready.push(*successor);
            }
        }
    }
    if settled == successors.len() {
        return Ok(());
    }
    Err(RhiError::new(
        RhiErrorKind::InvalidUsage,
        "the plan's dependencies form a cycle, counting the implicit order of batches on \
         one lane; no submission order satisfies them",
    )
    .at("SubmissionPlanBuilder::build"))
}

/// Whether `from` happens before `to` through any chain of edges.
///
/// Section 40.4 asks this of every batch pair, and both the implicit same-lane
/// edges and the explicit dependencies are in the graph. The search is iterative
/// for the reason [`check_acyclic`] is: its depth is caller-controlled.
fn reaches(successors: &[Vec<usize>], from: usize, to: usize) -> bool {
    let mut seen = vec![false; successors.len()];
    let mut pending = vec![from];
    seen[from] = true;
    while let Some(node) = pending.pop() {
        for successor in &successors[node] {
            if *successor == to {
                return true;
            }
            if !seen[*successor] {
                seen[*successor] = true;
                pending.push(*successor);
            }
        }
    }
    false
}
