//! Submission, completion, and presentation (specification sections 39 through 45).
//!
//! This module owns the plan and its acceptance: what a caller builds before it
//! submits, what the RHI guarantees when it accepts, what completion means once
//! work is running, and when the RHI may reclaim a resource. The presentation
//! chapter is the sibling that runs before and after it —
//! [`crate::api::presentation`] owns the target, the lease, and the frame, this
//! module owns the plan the frame enters and the receipt that reports what became
//! of it.
//!
//! ```text
//! submission/plan.rs        plan identity, the validated plan body              (39, 40)
//! submission/builder.rs     the plan builder, hazard and closure validation     (40)
//! submission/completion.rs  completion state, receipt, submit acceptance         (41)
//! ```
//!
//! # What this module does not own
//!
//! - **Recorded work.** A batch carries `Vec<RecordedWork>`, which the recording
//!   chapter owns and this module only validates against a lane. The plan
//!   builder never records a command and never inspects a command's internals;
//!   it reads the two things a recording publishes for exactly this purpose —
//!   its device identity and its execution domains — plus the merged actual-use
//!   summary that sections 40.4 and 45.3 need.
//! - **Native submission machinery.** Section 39 opens by listing what is
//!   deliberately absent from this surface: no queue, fence, semaphore, event, or
//!   timeline value is public, and a logical completion point is not a native
//!   fence value (section 41.7). `SubmissionLaneId` plus explicit dependency
//!   edges are the complete portable input for a backend that later maps lanes to
//!   multiple native queues; a backend that has one queue may serialize them.
//!   Native queue selection, fence waits, descriptor reuse, resource retirement,
//!   and state tracking remain backend-private implementation choices.
//! - **Retirement bookkeeping.** Section 41.6 defines retirement as a *property*
//!   rather than a verb: native backing may be reclaimed once the last batch that
//!   actually referenced an object is terminal and no CPU logical owner remains.
//!   There is deliberately no `retire()` a caller could call, and this module adds
//!   none. This is also the seam for backend-private descriptor retirement: a
//!   descriptor is not reusable merely because its logical handle was dropped.
//! - **Presentation.** Section 45.1 puts `present_after` on this module's builder,
//!   but the frame it consumes, the lease it leases, and the present outcome types
//!   are [`crate::api::presentation`]'s.
//!
//! # Invariants this module enforces
//!
//! ```text
//! a batch's work is non-empty and belongs to the plan's device      (40.1)
//! a lane accepts every execution domain its work contains           (40.1)
//! different lanes have no implicit order                            (40.2)
//! an unordered overlapping write is MissingDependency               (40.4, 41.4)
//! accepted != complete, and no Err is returned after acceptance     (41.3)
//! a dropped plan or builder submits nothing and abandons its frames (41.9, 45.6)
//! ```
//!
//! # The submission-lane vocabulary lives here
//!
//! `SubmissionLaneId`, `SubmissionLaneClass`, `LaneWorkDomains`,
//! `SubmissionLaneInfo`, `SubmissionCapabilities`, and `LaneDependencyRoute` are
//! stated by module 02's section 10, but they are submission concepts: a lane is
//! what a batch is added to, and a dependency route is what `add_dependency`
//! consumes. They are therefore defined in this file, and their *use* in resource
//! queries stays in [`crate::api::resource`]. That split is adjudication A18 in
//! the 0.16 plan: the definition site is the owner, and a file follows the meaning
//! of a type rather than the chapter number that happens to state it.

use crate::api::capability::{encode_entry, write_section};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::resource::transient::TransientLifetime;
use std::sync::Mutex;

pub(crate) mod backend;
pub mod builder;
pub mod completion;
pub mod plan;

pub use builder::SubmissionPlanBuilder;
pub use completion::{CompletionFailure, CompletionState, SubmissionReceipt};
pub use plan::{
    CompletionPoint, PlanPoint, SubmissionBatchId, SubmissionPlan, SubmissionPlanId,
    SubmissionPoint,
};

/// Submission-owned registry shared with the plan-scoped transient allocator.
/// It is synchronization state for one builder, not an independent lifetime
/// authority: the builder snapshots and validates it against its DAG at build.
#[derive(Default)]
pub(crate) struct TransientLifetimeRegistry(Mutex<Vec<TransientLifetime>>);

impl TransientLifetimeRegistry {
    pub(crate) fn record(&self, lifetime: TransientLifetime) {
        self.0
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(lifetime);
    }

    pub(crate) fn snapshot(&self) -> Vec<TransientLifetime> {
        self.0
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }
}

/// Opaque identity of one logical submission lane on one device.
///
/// A lane is *a logically ordered submission domain* (section 10), not a native
/// queue family, command queue, or GPU engine. Section 10.3 forbids the portable
/// surface from exposing `queue_family_index` or `native_queue_count` for exactly
/// that reason: several logical lanes do not promise hardware overlap, and one
/// native queue does not prevent several logical scheduling lanes.
///
/// There is no public constructor and no public numeric accessor, so the value is
/// evidence of equality rather than an ordinal a caller could count with — the
/// same rule [`crate::api::capability::CapabilityCompatibilityId`] follows. A
/// caller learns a lane ID from [`SubmissionLaneInfo::id`] and uses it to add a
/// batch; it never names one by number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionLaneId(u16);

impl SubmissionLaneId {
    /// Mints a lane identity during device enumeration.
    ///
    /// Crate-private: a lane exists because a device reported it, so only the
    /// enumeration path may name one. A caller that could mint a lane ID could
    /// ask for work on a lane the device never offered.
    ///
    /// The expectation is gated on `all(not(test), not(feature = "dx12"))` rather
    /// than on either alone, for the reason `api::capability` records on
    /// `AvailableCapabilities::from_facts`: with the DX12 backend compiled out, the
    /// only remaining enumeration is the mock's, which lives in the test build —
    /// so a `not(test)` expectation would sit unfulfilled as soon as `dx12` is on,
    /// and an ungated constructor is a hard error in the lib when both are off.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "minted by the two backends that enumerate lanes: the DX12 provider and the test-build mock"
        )
    )]
    pub(crate) fn new(value: u16) -> Self {
        Self(value)
    }

    /// Writes this identity into a canonical encoding.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
}

/// How a lane is classified for scheduling and diagnostics.
///
/// A label, not a legality rule. Section 10.1 is explicit that what decides
/// whether a command may go to a lane is [`SubmissionLaneInfo::domains`], not this
/// class: a backend is free to report a `Graphics` lane that also accepts compute
/// and copy work, and a caller must not infer the legal command set from the name.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmissionLaneClass {
    /// Carries every domain the device supports.
    General,
    /// Classified for raster/draw work.
    Graphics,
    /// Classified for compute work.
    Compute,
    /// Classified for copy work.
    Transfer,
}

impl SubmissionLaneClass {
    /// Writes this class into a canonical encoding.
    ///
    /// A tag byte rather than a discriminant read back out of the type: the enum
    /// is `#[non_exhaustive]`, so a reader that cast the value to an integer would
    /// be encoding a number this crate does not promise. The wildcard-free match
    /// makes a new variant a compile error here, which is what keeps the encoding
    /// injective as the vocabulary grows.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(match self {
            Self::General => 0,
            Self::Graphics => 1,
            Self::Compute => 2,
            Self::Transfer => 3,
        });
    }
}

/// The execution domains a unit of recorded work contains or a lane accepts.
///
/// A hand-rolled bitset over `u8`, following
/// [`crate::api::resource::buffer::BufferUsage`]: section 10.1 fixes the three
/// constants, and the portable surface does not depend on the `bitflags` crate.
///
/// This type is what makes "the lane must accept this work" a checkable rule
/// instead of a naming convention. `RecordedWork::work_domains()` reports what a
/// recording actually contains, and section 40.1 requires
/// `lane.domains().contains(work.work_domains())` for every work in a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneWorkDomains(u8);

impl LaneWorkDomains {
    /// Raster work: draw and attachment commands.
    pub const RASTER: Self = Self(1 << 0);
    /// Compute work: dispatch commands.
    pub const COMPUTE: Self = Self(1 << 1);
    /// Copy work: copy commands, uploads, and readbacks.
    pub const COPY: Self = Self(1 << 2);

    /// Whether every bit set in `other` is set in `self`.
    ///
    /// An empty `other` is contained in everything, which is what makes the
    /// "a recording must contain some domain" rule a rule about the recording
    /// rather than about this comparison.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two domain sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no domain bit is set.
    ///
    /// A lane that accepts no domain can accept no work, and a recording that
    /// contains no domain records nothing executable. Both are refused where they
    /// are reported rather than being treated as a valid empty set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Writes this domain set into a canonical encoding.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(self.0);
    }
}

impl core::fmt::Display for LaneWorkDomains {
    /// Renders the set as `RASTER|COPY`, or `<none>` when empty.
    ///
    /// Diagnostic text for errors and logs. Section 10.1 does not declare this
    /// impl; it is added because a refused batch must be able to say *which*
    /// domains the chosen lane cannot take, and `LaneWorkDomains(5)` cannot.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let names = [
            (Self::RASTER, "RASTER"),
            (Self::COMPUTE, "COMPUTE"),
            (Self::COPY, "COPY"),
        ];
        let mut written = false;
        for (bit, name) in names {
            if self.contains(bit) {
                if written {
                    formatter.write_str("|")?;
                }
                formatter.write_str(name)?;
                written = true;
            }
        }
        if !written {
            formatter.write_str("<none>")?;
        }
        Ok(())
    }
}

/// One lane a device offers, with the domains it accepts.
///
/// The pair is the whole fact: a lane ID to add batches to, and the set of
/// execution domains a batch submitted to it may contain.
#[derive(Clone, Debug)]
pub struct SubmissionLaneInfo {
    id: SubmissionLaneId,
    class: SubmissionLaneClass,
    domains: LaneWorkDomains,
}

impl SubmissionLaneInfo {
    /// Assembles one lane's facts during enumeration.
    ///
    /// Crate-private: a lane's domains are a device answer, and a caller-built
    /// one would be a claim about hardware nobody asked.
    ///
    /// Gated on the same pair as [`SubmissionLaneId::new`], for the same reason.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "assembled by the two backends that enumerate lanes: the DX12 provider and the test-build mock"
        )
    )]
    pub(crate) fn new(
        id: SubmissionLaneId,
        class: SubmissionLaneClass,
        domains: LaneWorkDomains,
    ) -> Self {
        Self { id, class, domains }
    }

    /// This lane's identity.
    pub fn id(&self) -> SubmissionLaneId {
        self.id
    }

    /// This lane's scheduling/diagnostic classification.
    ///
    /// Not a legality answer; see [`SubmissionLaneClass`].
    pub fn class(&self) -> SubmissionLaneClass {
        self.class
    }

    /// Which recorded-work domains may be submitted to this lane.
    ///
    /// This is the correctness fact section 10.1 keeps apart from the class, and
    /// the one section 40.1 requires a batch to respect.
    pub fn domains(&self) -> LaneWorkDomains {
        self.domains
    }

    /// Writes this lane's whole fact set into a canonical encoding.
    ///
    /// All three fields, including the class. The class is explicitly *not* a
    /// legality answer (see [`SubmissionLaneClass`]), so it cannot change what a
    /// batch is allowed to do — and encoding it anyway is deliberate. Two devices
    /// that agree on every domain set but classify their lanes differently are two
    /// different device models, and a fingerprint that compared them equal would be
    /// describing a contract it does not hold.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        self.id.encode_into(out);
        self.class.encode_into(out);
        self.domains.encode_into(out);
    }
}

/// How happens-before can be established between two lanes.
///
/// Section 40.2 makes this the whole of the portable cross-lane story: two
/// different lanes are *unordered* by default, and only an explicit
/// `add_dependency` creates an order between them. Each variant says which
/// mechanism can carry that order.
///
/// Note what section 40.2 removed: a host wait is no longer a route. A CPU
/// orchestration step between two GPU works belongs to a higher-level
/// continuation, not to a lane dependency, so `HostWait` is not a variant here and
/// may not be added as one.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneDependencyRoute {
    /// Producer and consumer are already in one ordered execution domain, so
    /// lane order itself satisfies happens-before and no separate native wait is
    /// needed.
    ///
    /// A backend must answer this rather than refusing an already-ordered edge
    /// merely because it has no separate wait primitive for it (section 40.3).
    Ordered,

    /// Different lanes, and the backend can lower a GPU-side dependency between
    /// them.
    Gpu,

    /// The two logical lanes are collapsed into one ordered execution domain
    /// during lowering, which retains the required logical order.
    Collapse,

    /// No route proves the required order. `add_dependency` and `build` report
    /// [`crate::api::error::RhiErrorKind::Unsupported`], and a caller that needs
    /// the order restructures — a single lane always exists (the base guarantee
    /// below) — or observes completion on the host first.
    Unsupported,
}

impl LaneDependencyRoute {
    /// Writes this route into a canonical encoding.
    ///
    /// Tagged by variant rather than by a stored integer, for the reason given on
    /// [`SubmissionLaneClass::encode_into`].
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(match self {
            Self::Ordered => 0,
            Self::Gpu => 1,
            Self::Collapse => 2,
            Self::Unsupported => 3,
        });
    }
}

/// The lanes one device offers, and how they can be ordered against each other.
///
/// Section 7.2 gives an enabled device this snapshot through
/// `EnabledCapabilities::submission`, and section 40.2 has the plan builder read
/// [`Self::dependency_route`] from it. The snapshot is immutable device data, like
/// every other capability fact: there is no verb that adds a lane after creation.
///
/// # Base guarantee
///
/// Every device has at least one lane whose domains include `RASTER | COPY`, and
/// if [`crate::api::platform::requirements::OptionalFeature::Compute`] is enabled,
/// at least one lane also includes `COMPUTE` — the same lane may satisfy both.
/// `Self::validate_base_guarantee` is that rule, and it is an enumeration-time
/// obligation rather than a constructor precondition: the empty snapshot below is
/// what a device carries before its request has filled the facts in.
#[derive(Clone, Debug)]
pub struct SubmissionCapabilities {
    lanes: Vec<SubmissionLaneInfo>,
    /// Cross-lane routes the backend reported, keyed by lane pair.
    ///
    /// The specification writes this type's body as `lanes: Vec<SubmissionLaneInfo>`
    /// and gives `dependency_route` no derivation, which means the answer is a
    /// device fact rather than something the RHI can compute from the lane list: a
    /// lane list says which domains a lane accepts, not whether a native
    /// dependency primitive exists between two of them. The table is where that
    /// fact is carried. A pair with no entry answers
    /// [`LaneDependencyRoute::Unsupported`], which is the conservative reading —
    /// a backend that *can* order two lanes must report it, and under-reporting
    /// costs a refusal the caller can always restructure around, never an accepted
    /// edge that is not actually ordered.
    routes: Vec<(SubmissionLaneId, SubmissionLaneId, LaneDependencyRoute)>,
}

impl SubmissionCapabilities {
    /// Assembles a snapshot from an enumerated lane set.
    ///
    /// Crate-private, and the constructor `api::capability` calls: an
    /// [`crate::api::capability::EnabledCapabilities`] must be able to hold one of
    /// these to answer `submission()`, and an empty lane set is what a device
    /// carries in the window between its creation and the enumeration that fills
    /// its facts in. A caller may not fabricate either, because a fabricated lane
    /// set would let a caller name a lane the device never offered.
    ///
    /// Its callers are the backends, each describing the lanes it actually offers:
    /// the mock backend's default and the DX12 direct queue. A backend that
    /// assembled the lane set itself and then reported a feature the lanes do not
    /// match is refused at [`crate::api::platform::Device::new`], which is the one
    /// place both halves are in hand.
    ///
    /// Gated on the same pair as [`SubmissionLaneId::new`], for the same reason.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "assembled by the two backends that enumerate lanes: the DX12 provider and the test-build mock"
        )
    )]
    pub(crate) fn new(lanes: Vec<SubmissionLaneInfo>) -> Self {
        Self {
            lanes,
            routes: Vec::new(),
        }
    }

    /// Records that the backend can establish `route` from `from` to `to`.
    ///
    /// Crate-private: this is an enumeration-time device fact, and section 40.2
    /// makes it the answer `add_dependency` branches on. Recording a self-pair is
    /// ignored rather than refused, because `from == to` is answered by definition
    /// (a lane is one ordered domain) and a backend that reports it as a fact has
    /// said nothing this type does not already know.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "filled by device enumeration when the backend port lands"
        )
    )]
    pub(crate) fn record_dependency_route(
        &mut self,
        from: SubmissionLaneId,
        to: SubmissionLaneId,
        route: LaneDependencyRoute,
    ) {
        if from == to {
            return;
        }
        self.routes.retain(|entry| entry.0 != from || entry.1 != to);
        self.routes.push((from, to, route));
    }

    /// The lanes this device offers, in the order it reported them.
    pub fn lanes(&self) -> &[SubmissionLaneInfo] {
        &self.lanes
    }

    /// The lane with this identity, or `None` when the device offers no such lane.
    ///
    /// `None` is the answer section 40.1's "lane belongs to current Device" rule
    /// becomes: a batch added to a lane the device never reported is refused
    /// rather than handed to a backend, which is the rule that stops one device's
    /// lane ID from being accepted by another.
    pub fn lane(&self, id: SubmissionLaneId) -> Option<&SubmissionLaneInfo> {
        self.lanes.iter().find(|lane| lane.id == id)
    }

    /// How happens-before from `from` to `to` can be established.
    ///
    /// ```text
    /// from == to            Ordered, by definition
    /// a reported route      that route
    /// anything else         Unsupported
    /// ```
    ///
    /// The last line covers both "the backend reported no route for this pair"
    /// and "the device offers no such lane": in neither case does the RHI have a
    /// fact proving the order, and section 40.2 resolves an unprovable order as
    /// `Unsupported` rather than as a silent guess.
    pub fn dependency_route(
        &self,
        from: SubmissionLaneId,
        to: SubmissionLaneId,
    ) -> LaneDependencyRoute {
        if from == to {
            return LaneDependencyRoute::Ordered;
        }
        self.routes
            .iter()
            .find(|entry| entry.0 == from && entry.1 == to)
            .map_or(LaneDependencyRoute::Unsupported, |entry| entry.2)
    }

    /// Checks the base guarantee section 10 states.
    ///
    /// ```text
    /// at least one lane accepts RASTER | COPY
    /// at least one lane accepts COMPUTE when the Compute feature is enabled
    /// ```
    ///
    /// The result is [`RhiErrorKind::BackendFailure`], not
    /// [`RhiErrorKind::Unsupported`]: every device is required to have such a
    /// lane, so a snapshot without one is a defect in the enumeration that
    /// produced it rather than a capability a caller may not use. Reporting it at
    /// enumeration time is what keeps root section 4's rule — a portable defect
    /// may not be handed down for a driver to discover — true for lanes.
    pub(crate) fn validate_base_guarantee(&self, compute_enabled: bool) -> RhiResult<()> {
        let base = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
        if !self.lanes.iter().any(|lane| lane.domains.contains(base)) {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "this device offers no lane that accepts both raster and copy work, \
                 which every device must have",
            ));
        }
        if compute_enabled
            && !self
                .lanes
                .iter()
                .any(|lane| lane.domains.contains(LaneWorkDomains::COMPUTE))
        {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "the compute feature was enabled but no lane accepts compute work",
            ));
        }
        Ok(())
    }

    /// Writes this snapshot into a canonical encoding.
    ///
    /// Two sections, both sorted by their encoded bytes, because neither carries an
    /// order the contract depends on: [`Self::lanes`] reports the order the backend
    /// enumerated in, and [`Self::record_dependency_route`] appends in the order
    /// routes were discovered. A lane's *identity* is [`SubmissionLaneId`], so two
    /// devices reporting the same lanes in a different order offer the same
    /// contract and must encode to the same bytes.
    ///
    /// The routes are written as one entry per `(from, to, route)` rather than as a
    /// matrix, because that is the shape the facts are held in and a matrix would
    /// write an entry for every pair the backend never answered — turning "no route
    /// was reported" into a recorded [`LaneDependencyRoute::Unsupported`], which is
    /// a different and stronger statement.
    ///
    /// See [`crate::api::capability::CapabilityFacts::canonical_bytes`] for the rules
    /// this contributes to.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        write_section(
            out,
            self.lanes
                .iter()
                .map(|lane| encode_entry(|out| lane.encode_into(out), |_| {}))
                .collect(),
        );
        write_section(
            out,
            self.routes
                .iter()
                .map(|(from, to, route)| {
                    encode_entry(
                        |out| {
                            from.encode_into(out);
                            to.encode_into(out);
                        },
                        |out| route.encode_into(out),
                    )
                })
                .collect(),
        );
    }
}

// `submission/builder.rs` implements section 40's builder against the real
// `crate::api::command::{RecordedWork, ResourceUse}`,
// so it is red until module 04 declares `mod record;` in `command.rs`. The lane
// vocabulary at the top of this file is separated out first because module 04's
// `RecordedWork::work_domains` names `LaneWorkDomains`, so the type has to exist
// before that module can compile.
