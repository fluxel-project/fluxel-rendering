//! The capture and diagnostic tooling SPI (specification module 07).
//!
//! This module owns exactly one thing: **the seam a capture tool reaches the RHI
//! through** — the SPI version it negotiates against, the device-scoped access
//! object, the observer contract and the ordering rule its callbacks obey, and
//! the identity of the events it is handed. What those events *contain* is the
//! business of the sibling files ([`definition`](crate::api::tooling::definition),
//! [`mutation`](crate::api::tooling::mutation),
//! [`work`](crate::api::tooling::work), [`plan`](crate::api::tooling::plan),
//! [`event`](crate::api::tooling::event)); this file only states how a tool gets
//! to them.
//!
//! Those paths are written fully-qualified rather than shortened. That is no
//! longer *required* — the outer doc comment `api/mod.rs` used to place on
//! `pub mod tooling` is gone, and with it the merge that made rustdoc resolve
//! this block's links against `api` instead of against `tooling`. They are kept
//! because a fully-qualified path cannot be broken by where the module is
//! declared, and this is not a trap worth re-arming for the sake of four deleted
//! path segments.
//!
//! It does not own a capture *format*, a capture *policy*, or a replayer. See
//! "What is deliberately absent" below — that list is load-bearing, and the
//! reason this chapter is short is that most of what a reader expects from a
//! "tooling" module belongs to a layer that is not the RHI.
//!
//! # Why this is `#[doc(hidden)]`
//!
//! Declared `#[doc(hidden)]` by `api/mod.rs` because it is an audience
//! statement, not a stability one. Everything here is public, versioned, and
//! semver-governed; it is simply not what a *rendering* caller learns. A caller
//! that never opens a capture tool never names a type in this module.
//!
//! # The two invariants
//!
//! 1. **Nothing in a captured record is a live handle.** Every captured type
//!    names an object by [`crate::api::identity::ObjectId`], by
//!    [`crate::api::presentation::AcquiredFrameId`], or by its own capture-local
//!    definition. Section 54 states this as the list of what a tooling definition
//!    may contain — value types, `ObjectId`, `AcquiredFrameId` — and section 58.8
//!    states the security half of it: no native pointer, OS handle, GPU virtual
//!    address, descriptor heap index, absolute host-memory address, or credential.
//!    The two rules together are why a captured record can outlive the device
//!    that produced it.
//! 2. **[`SemanticEventId`](crate::api::tooling::SemanticEventId) is CPU
//!    observation order, not GPU execution order**
//!    (§53.3). Two events in increasing id order say nothing about which GPU work
//!    ran first. Real GPU order is stated only by the order within a
//!    `PortableCommand` list, by `SubmissionPlan` lane order, by explicit
//!    dependencies, and by the present relation — see
//!    [`work`](crate::api::tooling::work) and
//!    [`plan`](crate::api::tooling::plan). A
//!    reader that treats the event stream as a trace has misread the chapter.
//!
//! # Section 52's prerequisites, audited
//!
//! Section 52 is not a list of types for this module to add. It is the list of
//! capabilities RHI must *already* have for portable capture/replay to be
//! possible later, and every one of them belongs to the chapter that owns the
//! thing being made observable. Auditing them here is the point of the section:
//! a gap is discovered as a missing accessor rather than as a Replay bug two
//! years later. Each item below was traced to its owner.
//!
//! ```text
//! §52.1  Buffer / Texture / TextureView / Sampler
//!        api::resource::{buffer, texture, view, sampler}
//!        -- each handle answers id(), device_identity(), descriptor()
//!
//! §52.1  ShaderModule
//!        api::shader::ShaderModule -- id(), device_identity(), artifact();
//!        the reconstructable definition is the artifact itself, which is why
//!        artifact() is a real accessor rather than a creation-time-only input
//!
//! §52.1  BindGroupLayout / BindGroup
//!        api::binding::layout::BindGroupLayout -- descriptor() returns the
//!        *canonicalized* descriptor, so capture records what the layout is
//!        rather than what the caller typed
//!        api::binding::group::BindGroup -- layout(), descriptor()
//!
//! §52.1  PipelineInterface / RasterPipeline / ComputePipeline
//!        api::pipeline::{interface, raster, compute} -- descriptor() on each,
//!        plus interface() on the two pipelines
//!
//! §52.1  UploadJob
//!        api::resource::transfer::upload::UploadJob -- id(),
//!        device_identity(), descriptor()
//!
//! §52.1  ReadbackTicket
//!        api::resource::transfer::readback::ReadbackTicket -- id(),
//!        device_identity(), request()
//!
//! §52.1  RecordedWork
//!        api::command::RecordedWork -- id(), device_identity(),
//!        work_domains(), resource_uses(), and a crate-private commands()
//!        that the describe_work path reads
//!
//! §52.1  Submission / Completion, AcquiredFrame / Present
//!        Named by their own device-scoped tokens -- SubmissionPlanId,
//!        SubmissionPoint, CompletionPoint, PresentPlanId, PresentReceiptId,
//!        AcquiredFrameId -- and NOT by ObjectId. Section 52.1's list says
//!        "ObjectId" for every entry; sections 57 and 58.1 use these tokens for
//!        exactly these rows. Resolution: the specific sections win, because a
//!        plan point and a present receipt are evidence of an event rather than
//!        an object in the inventory, and ObjectId is defined as the identity of
//!        an object the device created. Recorded in the 0.16 audit.
//!
//! §52.2  Live RecordedWork rebuildable
//!        api::command retains the ordered commands with their per-command use
//!        lists, the merged use summary, the work domains, and the debug push /
//!        pop / marker payloads with their labels. The commands are rebuilt by
//!        *normalization* rather than echoed -- a draw records the state that was
//!        current when it was issued, not a preceding run of state-setting
//!        commands -- which work::PortableCommand documents in full.
//!
//! §52.3  Object reconstruction graph
//!        definition::CapturedObjectDefinition, fed by the accessors listed under
//!        §52.1 above: a TextureView names its texture, a BindGroup names its
//!        layout and its resources, a PipelineInterface names its layouts, and a
//!        RasterPipeline names its modules, its interface, and its fixed state.
//!
//! §52.4  Upload mutation
//!        api::resource::transfer::upload::UploadDescriptor retains the
//!        destination, the offset/subresource, the HostTexelLayout, and the
//!        source bytes as an Arc<[u8]>, so capture never re-reads the CPU data it
//!        just uploaded.
//!
//! §52.5  Readback primitive
//!        api::resource::transfer::readback -- ReadbackRequest carries the
//!        buffer range or the textured region, ReadbackTexelLayout carries the
//!        row/image layout, and ReadbackStatus plus ReadbackTicket::try_read
//!        carry completion-aware readiness. RHI does not decide when to snapshot.
//!
//! §52.6  Submission / present visibility
//!        api::submission::{SubmissionPlanBuilder, SubmissionPlan,
//!        SubmissionReceipt, PlanPoint, SubmissionLaneId, SubmissionBatchId,
//!        SubmissionPoint, CompletionPoint} and
//!        api::presentation::{PresentPlanId, PresentReceiptId, PresentState}.
//!        The builder is what makes a plan observable at all; it was still
//!        unwritten when this audit was first taken and landed in module 05
//!        during the same series, so the item is closed.
//!        plan::CapturedSubmissionPlan carries the relation rather than the
//!        counts.
//!
//! §52.7  Shader provenance
//!        api::shader::ShaderArtifact carries code, abi_version, interface,
//!        requirements, and provenance; definition::CapturedObjectDefinition::
//!        Shader carries the whole artifact rather than a lowered form, so the
//!        ReplayRuntime can judge direct acceptance / recompile / unsupported
//!        without RHI having a cross-backend compiler.
//!
//! §52.8  Device loss / terminal events
//!        api::platform::{DeviceStatus, DeviceLossInfo} plus the terminal states
//!        CompletionState::DeviceLost, ReadbackStatus::DeviceLost,
//!        AcquiredFrameState::DeviceLost, PresentState::DeviceLost, and
//!        event::SemanticEvent::DeviceLost. The tooling subscription's own half
//!        of the rule is enforced here: subscribe() refuses on an already-lost
//!        device, because section 52.8 forbids handing a capture tool a
//!        subscription that can never reach a terminal observation.
//! ```
//!
//! ## Known gaps found by that audit
//!
//! One item is open, and it is a manifest decision rather than a source one.
//!
//! * **The tooling SPI is not behind its feature.** Section 53 recommends an
//!   independent `rhi-tooling` feature so that an ordinary release build does not
//!   carry the observer surface, and `crates/rhi/Cargo.toml` declares none, so
//!   there is nothing for this module to be gated on. A manifest change is the
//!   caller's, not this module's. The detail is in the Cargo section below.
//!
//! # What is deliberately absent
//!
//! Sections 58.3 through 58.6 and 58.8 are absences, and the honest output for
//! an absence is a statement rather than a type. There is no `ArtifactLayer`,
//! no `ReplayRuntime`, no `SnapshotPolicy`, and no capture-file writer in this
//! crate, and adding one "for completeness" would put a debug product inside the
//! execution API:
//!
//! * **The Artifact Layer (§58.3, §58.7).** RHI does not own the magic, the
//!   schema major/minor, chunks, a manifest, compression, dedup, blob hashing,
//!   signatures, encryption, or artifact migration.
//!   [`TOOLING_SPI_VERSION`](crate::api::tooling::TOOLING_SPI_VERSION) is
//!   the tooling SPI's version and is *not* an artifact schema version; the two
//!   version independently, and conflating them is the specific mistake section
//!   58.3 names. Section 56 adds the encoding half: a Rust enum discriminant or
//!   memory layout must never be written directly as an artifact opcode, so the
//!   artifact layer must supply its own tagged, versioned, bounds-checked,
//!   canonical encoding.
//! * **Capture dependency closure (§58.4).** RHI does not answer "which
//!   producers must be added before passes 30 to 40", "must this persistent
//!   texture written in an earlier frame be snapshotted", or "how many frames of
//!   a history resource are needed". That is RenderGraph's and the Capture
//!   Coordinator's question.
//! * **Snapshot policy (§58.5).** RHI provides readback, resource descriptors,
//!   and actual use. The capture side decides the snapshot point, the
//!   subresource, full bytes versus hash-only, the size budget, redaction, and
//!   external fixtures.
//! * **`ReplayRuntime` (§58.6).** Reading and safely parsing an artifact, schema
//!   migration, target selection, capability negotiation, shader rebuild,
//!   object-graph rebuild, command rebuild, submission rebuild, external
//!   fixtures, observation/diff, and the step debugger all belong to it. It
//!   ultimately calls only the normal RHI API, which is why nothing in this
//!   module is a replay entry point. Section 58.7 fixes the source of truth it
//!   reads: the
//!   [`definition::CapturedObjectDefinition`](crate::api::tooling::definition::CapturedObjectDefinition)
//!   graph, initial snapshots and upload mutations,
//!   [`work::PortableCommand`](crate::api::tooling::work::PortableCommand), and
//!   [`plan::CapturedSubmissionPlan`](crate::api::tooling::plan::CapturedSubmissionPlan).
//!   `FrozenGraphIR` belongs to RenderGraph and
//!   is provenance and validation material, not the replay input.
//! * **The security boundary (§58.8).** Redaction, omission, and encryption of
//!   shader source, labels, and upload bytes are capture policy. RHI keeps the
//!   *semantics* free of native identifiers; it does not decide what a tool is
//!   allowed to write down. Note the one type that carries content rather than
//!   structure:
//!   [`mutation::CapturedUploadDefinition`](crate::api::tooling::mutation::CapturedUploadDefinition)
//!   carries the uploaded
//!   bytes, and section 55.3 states plainly that the artifact layer, not RHI,
//!   decides whether they are redacted or encrypted.
//!
//! # Section 53's Cargo feature is not declared here
//!
//! Section 53 recommends an independent `rhi-tooling` feature so that an ordinary
//! release build does not carry the observer surface. `api/mod.rs` declares
//! `pub mod tooling` unconditionally today, and the feature is a manifest
//! decision rather than a source decision, so it is recorded as a gap instead of
//! being spelled here as a `#[cfg(feature = ...)]` that no manifest backs. What
//! section 53 does fix regardless of the feature is that the *reconstructable
//! semantic contract of `RecordedWork`* exists in every build.

use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::{Device, DeviceStatus};

pub mod definition;
pub mod event;
pub mod mutation;
pub mod plan;
pub mod work;

pub use definition::CapturedObjectDefinition;
pub use event::SemanticEvent;
pub use plan::{
    CapturedDependencySource, CapturedPlanDependency, CapturedPresentPlan, CapturedSubmissionBatch,
    CapturedSubmissionPlan, CapturedSubmissionReceipt,
};
pub use work::{CapturedCommand, CapturedRecordedWork, CapturedResourceUse, PortableCommand};

// `mutation` and `work`'s full surface are re-exported through their own modules
// rather than flattened here: section 55's captured copy values are fourteen
// names that only a capture tool ever writes, and flattening them into the
// module root would bury the three verbs a caller actually calls.

/// The version of the tooling SPI.
///
/// Section 53.1 fixes this as an ordinary value with public fields rather than an
/// opaque token, so that a tool can compare it and branch without a constructor.
/// There is deliberately no `compatible_with`: the rule is "bump major when
/// changing an incompatible semantic", and a caller that has the two fields can
/// state that rule itself, which keeps the compatibility policy out of the API.
///
/// This is **not** a Capture Artifact schema version (§53.1, §58.3). The two
/// version separately on purpose: the SPI changes when the in-memory seam
/// changes, and the artifact schema changes when the on-disk encoding does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToolingSpiVersion {
    /// Bumped when a change is not backward compatible.
    pub major: u16,
    /// Bumped when a change is backward compatible.
    pub minor: u16,
}

/// The tooling SPI version this build implements.
///
/// A tool reads this once and decides whether it understands the events it is
/// about to be handed. It is a `const` rather than a function of the device
/// because the SPI's shape is a property of the compiled library, not of any one
/// backend or adapter.
pub const TOOLING_SPI_VERSION: ToolingSpiVersion = ToolingSpiVersion { major: 1, minor: 0 };

/// The device-scoped entry point to the tooling SPI.
///
/// Obtained from [`Device::tooling`]. Cloneable, because holding one is holding a
/// reference to the device rather than a registration in it — the registration is
/// a [`ToolingSubscription`], and a clone of an access grants no subscription.
///
/// # What it deliberately does not expose
///
/// The device behind it is private. Section 58.2 forbids an observer from
/// re-entering a mutating API on the same device, and while that rule binds the
/// observer rather than the RHI, an access object that also handed out the
/// `Device` would make the rule one line harder to keep than it needs to be.
/// A tool that wants to create objects holds its own `Device` clone for that, on
/// purpose and in its own code.
///
/// # Refusal order
///
/// Every verb here checks the device's status first, in O(1), and answers
/// [`RhiErrorKind::DeviceLost`] rather than reaching a registry that a lost
/// device no longer has. That is section 52.8's rule — "capture does not allow
/// waiting forever for an event that the RHI already knows will not complete" —
/// applied to the query verbs as well as to the subscription, and it is also
/// root section 3.1's: the check is portable and happens before anything native.
#[derive(Clone)]
pub struct ToolingAccess {
    /// The device this access is scoped to.
    ///
    /// Held as the shared logical owner rather than as a bare
    /// [`DeviceIdentity`] because the status checks above are the device's own
    /// answer, and a snapshot of the identity would not answer whether it is
    /// still alive.
    device: Device,
}

impl ToolingAccess {
    /// Assembles a device's tooling access.
    ///
    /// Crate-private: an access is device-scoped, so only the device may mint
    /// one — the same rule every other created handle in this crate follows.
    pub(crate) fn new(device: Device) -> Self {
        Self { device }
    }

    /// The device this access is scoped to.
    ///
    /// Public because a capture tool's own bookkeeping is keyed by it: an event
    /// stream is ordered per `DeviceIdentity` (§53.3), and a tool that multiplexes
    /// several devices has to say which one an event came from. Added rather than
    /// transcribed from section 53.2, which lists three verbs and no accessor.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device.identity()
    }

    /// Registers an observer and returns the subscription that unregisters it.
    ///
    /// # The start linearization point
    ///
    /// There is exactly one: the successful insertion of the observer into the
    /// device's observer set. This call returns only after that point. The
    /// returned subscription then receives **every** event whose
    /// [`SemanticEventId`] was assigned after the point and before the
    /// subscription's Drop unregistration point — exactly once each, in
    /// increasing id order — and receives nothing assigned before the point.
    ///
    /// ```text
    /// assigned before subscribe() returns     -> not delivered
    /// assigned after  subscribe() returns,
    ///   before Drop begins                    -> delivered exactly once
    /// assigned after Drop returns             -> not delivered
    /// ```
    ///
    /// That boundary is the reason [`Self::describe_object`] exists: an object
    /// created before a tool started watching produces no `ObjectCreated` event
    /// for it, and the lazy query is how the tool learns what it missed. The
    /// boundary is sharp rather than eventual on purpose — a "mostly once"
    /// boundary would make a capture record's header depend on scheduler timing.
    ///
    /// The callback then runs synchronously on the RHI's thread, and the
    /// references in the event are valid only for the duration of that call
    /// (§58.2). An observer that keeps data must copy it before returning; RHI
    /// does not wait for later observer processing, and an observer that blocks
    /// is blocking the RHI.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::DeviceLost`] if the device is already lost. Registering
    /// into a device that will emit no further events would hand the caller a
    /// subscription that can never be observed to end, which section 52.8
    /// forbids; the terminal observation is the error itself.
    ///
    /// # Contract on the observer
    ///
    /// The observer must not re-enter a mutating API on the same device or
    /// identity, must not wait for GPU completion or otherwise block, must not
    /// alter command or submission semantics, and must not drop its own
    /// subscription from inside the callback — dropping it waits for the callback
    /// currently executing, which is the observer waiting on itself. It may add
    /// capture and debug CPU overhead; it may not change GPU results.
    pub fn subscribe(&self, observer: Arc<dyn SemanticObserver>) -> RhiResult<ToolingSubscription> {
        self.refuse_if_lost()?;
        let registration = self.device.insert_observer(observer);
        Ok(ToolingSubscription::new(self.device.clone(), registration))
    }

    /// Asks what an object that is still live actually is.
    ///
    /// The lazy half of the seam, and the reason it is needed is the start
    /// linearization point of [`Self::subscribe`]: objects created before a tool
    /// began watching have no `ObjectCreated` event to replay, so the tool pulls
    /// their definition by [`ObjectId`] instead.
    ///
    /// Answers only for objects that are currently live, which is what makes the
    /// answer a plain value rather than a borrow tied to a handle: the returned
    /// definition owns everything it states.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::DeviceLost`] if the device is lost — the inventory a lost
    /// device had is not answerable, and pretending otherwise would fabricate a
    /// description. An id that names no live object on this device is refused by
    /// the registry when it lands; that refusal is not portable and is therefore
    /// not decided here.
    pub fn describe_object(&self, id: ObjectId) -> RhiResult<CapturedObjectDefinition> {
        self.refuse_if_lost()?;
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "runtime object descriptions are not retained by this RHI build",
        )
        .with_object(id))
    }

    /// Asks for the complete portable semantics of a live `RecordedWork`.
    ///
    /// Keyed by [`ObjectId`] rather than by a `RecordedWork` handle on purpose
    /// (§58.1): work recorded before a tool began watching, but still retained by
    /// plan or submission ownership, must be describable by a tool that never
    /// held the handle. A verb that took the handle would make the definition
    /// available exactly to the callers that no longer need it.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::DeviceLost`] if the device is lost, for the reason
    /// [`Self::describe_object`] gives.
    pub fn describe_work(&self, work: ObjectId) -> RhiResult<CapturedRecordedWork> {
        self.refuse_if_lost()?;
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "recorded-work descriptions are not retained by this RHI build",
        )
        .with_object(work))
    }

    /// Refuses when the device is gone.
    ///
    /// The one portable check every verb in this module shares, written once so
    /// that the three of them cannot drift into three different refusals.
    fn refuse_if_lost(&self) -> RhiResult<()> {
        match self.device.status() {
            DeviceStatus::Active => Ok(()),
            DeviceStatus::Lost => {
                let message = match self.device.loss_info() {
                    Some(info) => format!(
                        "the device was lost, so tooling cannot reach it: {}",
                        info.message()
                    ),
                    None => String::from("the device was lost, so tooling cannot reach it"),
                };
                Err(RhiError::new(RhiErrorKind::DeviceLost, message))
            }
        }
    }
}

impl Device {
    /// Opens this device's tooling access.
    ///
    /// Section 53.2 puts this verb on [`Device`] rather than making
    /// [`ToolingAccess`] constructible, because an access is device-scoped: there
    /// is no tooling surface that is not somebody's device's.
    ///
    /// Works on a lost device too, and deliberately: the access object is a
    /// capability to *ask*, and the asking is what refuses. Handing back an error
    /// here would make "my device died, let me look" unreachable, which is close
    /// to the opposite of what a capture tool needs.
    pub fn tooling(&self) -> ToolingAccess {
        ToolingAccess::new(self.clone())
    }
}

/// The registration of one observer in one device's observer set.
///
/// Its whole contract is its `Drop`, and the shape of that contract is why this
/// is a distinct type rather than a token returned by value:
///
/// ```text
/// subscribe() returned   -> the observer is registered and is being called
/// Drop begins            -> unregistration; every callback that began before
///                           this point is waited for
/// Drop returns           -> no callback for this subscription can begin, and
///                           none remains active
/// ```
///
/// So a tool that drops its subscription and then touches shared observer state
/// is guaranteed not to race with a callback it already got. The reverse is a
/// contract the type cannot enforce: an observer must not drop its own
/// subscription from inside `on_event()`, because that waits for the callback
/// currently executing — it would be the observer waiting on itself, and if it
/// could be detected it would be a deadlock rather than an error.
///
/// Not `Clone`: there is one registration and one unregistration point. A second
/// handle that unregistered on its own drop would give the subscription two end
/// points, and "the events before Drop" would stop naming a single boundary.
pub struct ToolingSubscription {
    /// The device whose observer set this registration lives in.
    ///
    /// Also the key the unregistration path needs, which is the observer
    /// registry the backend port adds alongside it.
    device: Device,
    registration: u64,
}

impl ToolingSubscription {
    /// Records a registration.
    ///
    /// Crate-private, and only `subscribe` may call it: a subscription is
    /// evidence that a registration happened, so it cannot be minted by a
    /// caller. Not yet called by `subscribe`, because that verb's insertion step
    /// is the backend port's; the contract tests reach it so that the type's
    /// identity half is exercised rather than asserted.
    pub(crate) fn new(device: Device, registration: u64) -> Self {
        Self {
            device,
            registration,
        }
    }

    /// The device this subscription observes.
    ///
    /// Added rather than transcribed from section 53.2, which declares the type
    /// with no surface at all: a tool that holds subscriptions for several
    /// devices has to say which one a subscription is for, and the alternative —
    /// holding the `Device` alongside every subscription — is a second field
    /// whose pairing with this one has nothing checking it.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device.identity()
    }
}

impl Drop for ToolingSubscription {
    /// Performs the unregistration described on the type.
    fn drop(&mut self) {
        self.device.remove_observer(self.registration);
    }
}

/// Receives semantic events from one device.
///
/// Implemented by a capture tool, a RenderGraph diagnostics sink, or a debugger.
/// The three bounds are section 53.2's and each is load-bearing:
///
/// * `Send + Sync` — events are dispatched from whichever thread produced them,
///   so the observer is reached from more than one.
/// * `'static` — the device's observer set outlives any borrow a caller could
///   hold, so the observer cannot borrow from the caller's stack.
///
/// The callback is synchronous, and section 58.2 fixes what that means: the
/// references inside the event are valid only for the duration of the call, RHI
/// does not wait for later observer processing, and an observer that needs to
/// keep something clones or copies it into its own queue before returning.
pub trait SemanticObserver: Send + Sync + 'static {
    /// Observes one event.
    ///
    /// Called on the RHI's thread, synchronously, once per event, in increasing
    /// [`SemanticEventId`] order per device. It must not re-enter a mutating API
    /// on the same device or identity, must not wait for GPU completion or
    /// otherwise block, must not alter command or submission semantics, and must
    /// not drop its own [`ToolingSubscription`]. It may add CPU overhead; it may
    /// not change GPU results.
    fn on_event(&self, event: SemanticEvent<'_>);
}

/// The identity of one semantic event.
///
/// Unique and monotonic within one [`DeviceIdentity`] (§53.3). It is an opaque
/// token for the same reason every other identity in this crate is: a caller
/// that could write `SemanticEventId(7)` could forge an ordering it did not
/// observe.
///
/// # It orders CPU observation, not GPU execution
///
/// ```text
/// Event 100 < Event 101
/// ```
///
/// does **not** mean that GPU work 100 happens-before GPU work 101. Under a
/// multi-threaded recorder two threads' events interleave in the order the
/// dispatcher saw them. Real GPU order is stated only by the order within a
/// `PortableCommand` list, by `SubmissionPlan` lane order, by explicit
/// dependencies, and by the present relation.
///
/// # Why it carries `Ord`
///
/// Section 53.2's subscription contract is stated in terms of this type —
/// "exactly once and in increasing `SemanticEventId` order" — and a caller cannot
/// check a rule it cannot express. Section 53.3's own derive list omits `Ord`;
/// adding it is the smallest change that makes the section's other half usable,
/// and it is safe to add to a token whose whole meaning is a total order. The
/// series audit records it as an addition rather than a transcription.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticEventId(u64);

impl SemanticEventId {
    /// Mints the next event identity for a device.
    ///
    /// Crate-private: the monotonic assignment is the device's, and a caller that
    /// could mint one could place itself in the order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the semantic event dispatcher mints these when the backend port lands"
        )
    )]
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw counter.
    ///
    /// Added rather than transcribed, on the same reasoning as `Ord`: a capture
    /// record that says "events 4, 5 and 9 from this device" is evidence, and
    /// evidence has to be writable down. It is a counter within one device's
    /// life, not a timestamp and not a GPU serial — see the type's ordering note,
    /// and note the deliberate contrast with
    /// [`crate::api::submission::SubmissionLaneId`], which has no numeric
    /// accessor precisely because nothing should order by a lane's number.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}
