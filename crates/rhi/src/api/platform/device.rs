//! The device: a shared logical execution domain (specification section 6).
//!
//! Everything a caller creates, records, submits, or presents is owned by a
//! device, and every one of those objects must be traceable to the same
//! [`DeviceIdentity`]. This module owns the device's own surface — who it is,
//! what it came from, what it can do, and whether it is still alive.
//!
//! The verbs that *create* things are not here. `Device::create_buffer` is
//! written in the resource module, `Device::create_pipeline` in the pipeline
//! module, and so on, each next to the types it produces. Rust allows the
//! inherent impl to live in another module of the same crate, and doing so keeps
//! one rule in one place instead of collecting every creation verb into a file
//! that would have to know about every resource in the crate.
//!
//! This module deliberately does not own the capability vocabulary
//! ([`crate::api::capability`]) or presentation
//! ([`crate::api::presentation`]); it only hands out handles to them.
//!
//! # What is decided here and what is asked elsewhere
//!
//! The device holds an [`DeviceIdentity`] and a backend, and nothing else. Every
//! verb that needs a native fact — provenance, identity, liveness, progress —
//! asks the backend through the crate-private lowering seam, which section 59 of
//! `08-governance-freeze-checklist.md` keeps off the public surface and which
//! this documentation therefore cannot link to. What stays here is the part a
//! backend must not decide: the identity comparison section 3.1 puts first, the
//! order in which ownership and liveness are judged, and the structured error a
//! caller sees.
//!
//! Liveness in particular has one home and it is not this one. The backend
//! observes the loss and holds the fact; this module holds the rules about it —
//! that it is terminal, that the summary is stable (section 6.5), and that a lost
//! device answers `DeviceLost` only once ownership has been settled. Caching the
//! fact here as well would create a second authority for it, which is the thing
//! section 65.3 rules out.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::capability::EnabledCapabilities;
use crate::api::diagnostics::DiagnosticEvent;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::backend::DeviceBackend;
use crate::api::platform::provider::{AdapterInfo, BackendKind};
use crate::api::platform::requirements::OptionalFeature;
use crate::api::statistics::{CumulativeStatistics, StatisticsConfig};
use crate::api::tooling::CapturedRecordedWork;
use crate::api::tooling::{SemanticEvent, SemanticEventId, SemanticObserver};

/// Whether a device is still usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceStatus {
    /// The device is usable.
    Active,
    /// The device is gone, and permanently so.
    Lost,
}

/// A stable summary of why a device was lost.
///
/// Section 6.5 requires the summary to be stable rather than a one-shot
/// notification: a caller that asks twice, or asks long after the loss, must get
/// the same answer.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DeviceLossInfo {
    message: String,
}

impl DeviceLossInfo {
    /// Describes a loss.
    ///
    /// Crate-private: only the code that observed the loss may summarize it.
    ///
    /// The DX12 backend's allocation path and Vulkan's native boundaries are
    /// such observers, so the expectation below is gated on their features as well as on the test
    /// build. A module-scope expectation in that backend makes references *out*
    /// of it count as live for the items they point at, so with `dx12` on this
    /// constructor is reached in a non-test build too and a `not(test)`-only
    /// expectation would sit unfulfilled — the same trap the provider's module
    /// documentation records for its own callees.
    #[cfg_attr(
        all(not(test), not(any(feature = "dx12", feature = "vulkan"))),
        expect(
            dead_code,
            reason = "called by the contract tests and DX12/Vulkan native-loss observers; with both backends compiled out, the code that observes native loss is not written"
        )
    )]
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }

    /// The human-readable loss summary.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// A shared logical execution domain.
///
/// Cloneable, and cloning is not a new device:
///
/// ```text
/// Device::clone()        the same domain, the same DeviceIdentity
/// request_device() again a new domain and a new identity
/// ```
///
/// Section 6.1 makes that a portable contract rather than an implementation
/// detail: even where a backend reuses one native device internally, two
/// successful requests still produce two isolated logical domains that must not
/// accept each other's objects.
///
/// Loss is terminal for the whole identity. There is no transparent replacement
/// of an existing public device; a retry is a new
/// `PlatformProvider::request_device()` and therefore a new identity.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    identity: DeviceIdentity,
    /// The native execution domain this handle lowers through.
    ///
    /// Directly owned by this inner execution domain. `Device` clones share this
    /// one `DeviceInner`, so they reach the same native device without separately
    /// reference-counting its backend. It is never replaced after construction — loss does not
    /// re-point it, because section 6.5 makes loss terminal and recovery a new
    /// request with a new identity.
    ///
    /// Liveness lives on the backend rather than beside this field, because the
    /// backend is what observes a native loss. Keeping one copy is section 65.3's
    /// rule; keeping it on the side that can see the event is what makes the copy
    /// authoritative.
    native: Box<dyn DeviceBackend>,
    /// What this device can actually do, built once from the backend's
    /// enumeration and never rebuilt.
    ///
    /// Interned here and nowhere else. Section 7.1 makes the compatibility id the
    /// interning of the contract's semantics, so the id has to be minted at the one
    /// moment the whole contract is in hand — which is construction. A device whose
    /// id were computed per call could answer two different ids for one contract.
    ///
    /// Shared rather than owned, because [`Device`] is cloned liberally and a clone
    /// is the same domain under the same identity (section 6.1): cloning the maps
    /// would copy a few hundred entries per clone to re-derive an answer that
    /// cannot have changed. `Arc` makes a clone cheap and keeps the guarantee that
    /// every clone reports the *same* id, which is what section 7.1's reuse rule
    /// depends on.
    ///
    /// Not a second authority for anything: it is immutable by contract (section
    /// 7.2), the backend owns no copy of it, and it is never replaced — loss does
    /// not re-point it, because section 6.5 makes recovery a new request with a new
    /// identity.
    capabilities: EnabledCapabilities,
    /// The serials this domain hands out.
    ///
    /// Shared rather than owned, and for the same reason `native` is: a clone is
    /// the same domain under the same identity (section 6.1), so two clones that
    /// each kept their own counter would mint the same plan serial twice. Section
    /// 39.1 makes a plan serial unique *within the device*, and this is what makes
    /// "the device" mean the identity rather than the handle.
    serials: DomainSerials,
    /// The exact compatibility tokens this domain has minted.
    ///
    /// Shared rather than owned, for the reason `serials` is: a clone is the same
    /// domain under the same identity (section 6.1), so two clones that each kept
    /// their own table would mint two different
    /// [`BindGroupLayoutCompatibilityId`](crate::api::binding::BindGroupLayoutCompatibilityId)s
    /// for one canonical layout. Section 21.1's rule is about the *device*, so the
    /// table has to hang off the identity rather than off the handle.
    interning: DomainInterning,
    /// Small portable runtime services belong to the device execution domain;
    /// they are not a second resource manager.
    runtime: RuntimeServices,
}

pub(crate) struct RuntimeServices {
    diagnostics: Mutex<Vec<DiagnosticEvent>>,
    statistics: Mutex<StatisticsState>,
    observers: Mutex<ObserverState>,
    pub(crate) captured_work: Mutex<HashMap<ObjectId, CapturedRecordedWork>>,
    /// Portable state machine around an optional backend debugger capture.
    pub(crate) native_capture_active: Mutex<bool>,
}

struct ObserverState {
    next: u64,
    next_event: u64,
    entries: Vec<(u64, Arc<dyn SemanticObserver>)>,
}

pub(crate) struct StatisticsState {
    pub(crate) config: StatisticsConfig,
    pub(crate) epoch: u64,
    pub(crate) sequence: u64,
    pub(crate) started: std::time::Instant,
    pub(crate) cumulative: CumulativeStatistics,
}

impl RuntimeServices {
    fn new() -> Self {
        Self {
            diagnostics: Mutex::new(Vec::new()),
            statistics: Mutex::new(StatisticsState {
                config: StatisticsConfig::default(),
                epoch: 1,
                sequence: 1,
                started: std::time::Instant::now(),
                cumulative: CumulativeStatistics::default(),
            }),
            observers: Mutex::new(ObserverState {
                next: 1,
                next_event: 1,
                entries: Vec::new(),
            }),
            captured_work: Mutex::new(HashMap::new()),
            native_capture_active: Mutex::new(false),
        }
    }
}

/// The per-domain serial sources.
///
/// One counter per kind rather than one shared, because the two answer different
/// questions and have different consumers: a plan serial is evidence that two
/// points came from one builder (section 39.1), and a submission serial is the
/// order in which this device accepted work (section 41.7). Collapsing them would
/// make a plan's identity depend on how many plans had been *submitted*, which
/// section 39.1's "an empty plan still has an identity" would then make visible.
///
/// Neither counter is process-global, unlike
/// [`crate::api::identity::ObjectId`]'s. That one has to be, because "globally
/// unique within the process" is its own contract; these two are scoped to one
/// device by section 39.1 and section 41.7 respectively, and widening them would
/// claim a uniqueness nothing needs.
pub(crate) struct DomainSerials {
    plans: AtomicU64,
    submissions: AtomicU64,
    presents: AtomicU64,
}

impl DomainSerials {
    fn new() -> Self {
        Self {
            // Both start at 1 for the reason the mock's completion counter does:
            // a zero would make a zero-initialized token indistinguishable from a
            // minted one, and neither type has a "never minted" spelling.
            plans: AtomicU64::new(1),
            submissions: AtomicU64::new(1),
            presents: AtomicU64::new(1),
        }
    }

    /// The next plan serial of this domain.
    pub(crate) fn next_plan(&self) -> u64 {
        self.plans.fetch_add(1, Ordering::Relaxed)
    }

    /// The next acceptance serial of this domain.
    pub(crate) fn next_submission(&self) -> u64 {
        self.submissions.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn next_present(&self) -> u64 {
        self.presents.fetch_add(1, Ordering::Relaxed)
    }
}

/// The per-domain interning tables behind module 03's exact compatibility tokens.
///
/// Section 21.1 says a `BindGroupLayoutCompatibilityId` is "produced by Device
/// interning canonical layout descriptors", and section 21.2 says the same of
/// `PipelineInterfaceCompatibilityId`. Both rules have the same two halves, and
/// both halves are why this table is here rather than in either chapter:
///
/// - *On one device, two equal canonical descriptors intern to one id.* A chapter
///   could not do this alone — it would have to keep a table, and two chapters
///   keeping two tables is two places the rule can drift.
/// - *A caller cannot construct the token.* The id is an opaque `u64` wrapped by
///   each chapter's own newtype, so the value has to come from somewhere the
///   caller does not reach. Returning a bare `u64` from here keeps this module
///   from having to name a type from module 03, which is the layering the
///   chapter-per-module split exists to preserve: the numbering is minted here,
///   and what it is *called* is decided next to the type it identifies.
///
/// The chapter that interns supplies the canonical bytes. This module never
/// encodes a descriptor — it does not know what one is, and giving it the
/// knowledge would make it the second authority on what "canonical" means.
///
/// # Two tables rather than one
///
/// The two tokens answer different questions and section 25's cache reuse keys on
/// them separately, so a layout and an interface that happened to encode to the
/// same bytes must not collide. The keys are the canonical bytes themselves, not a
/// digest of them, following
/// [`crate::api::capability`]'s table: a digest would trade a definite comparison
/// for a probabilistic one and buy nothing, because the entry has to be compared
/// on hit anyway to return the id.
///
/// Ids start at 1, as every other minted id in this crate does, so that a
/// zero-initialized field is distinguishable from a minted token.
pub(crate) struct DomainInterning {
    layouts: Mutex<InternTable>,
    interfaces: Mutex<InternTable>,
}

/// One keyed table of canonical bytes to minted ids.
struct InternTable {
    ids: HashMap<Vec<u8>, u64>,
    next: u64,
}

impl InternTable {
    fn new() -> Self {
        Self {
            ids: HashMap::new(),
            next: 1,
        }
    }

    /// The id already minted for `canonical`, or a fresh one.
    fn intern(&mut self, canonical: &[u8]) -> u64 {
        if let Some(existing) = self.ids.get(canonical) {
            return *existing;
        }
        let id = self.next;
        self.next += 1;
        self.ids.insert(canonical.to_vec(), id);
        id
    }
}

impl DomainInterning {
    /// A domain's tables, both empty.
    pub(crate) fn new() -> Self {
        Self {
            layouts: Mutex::new(InternTable::new()),
            interfaces: Mutex::new(InternTable::new()),
        }
    }

    /// Interns one canonical bind-group-layout descriptor (section 21.1).
    ///
    /// # Panics
    ///
    /// Never in practice: a poisoned lock is recovered rather than propagated, as
    /// [`crate::api::capability`]'s table does, because the mutex guards a map and
    /// not an invariant. A panic while holding it cannot have left a half-written
    /// id behind — every mutation below is a single insert of a complete entry —
    /// so resuming on the recovered guard is the correct behaviour and refusing to
    /// intern ever again is not.
    pub(crate) fn intern_layout(&self, canonical: &[u8]) -> u64 {
        let mut table = self
            .layouts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        table.intern(canonical)
    }

    /// Interns one canonical pipeline-interface descriptor (section 21.2).
    ///
    /// The same contract and the same recovery as [`Self::intern_layout`].
    pub(crate) fn intern_interface(&self, canonical: &[u8]) -> u64 {
        let mut table = self
            .interfaces
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        table.intern(canonical)
    }
}

impl Device {
    /// Opens a logical execution domain under a freshly minted identity.
    ///
    /// Crate-private: section 6.1 ties identity minting to a completed device
    /// request, and section 6.1's other half is that a backend does not mint its
    /// own — so only the request path that did both may call this.
    ///
    /// # Why this can fail
    ///
    /// It reads the backend's enumeration, interns it, and checks section 7.2's
    /// base guarantee in one place. That check is a portable rule about a device
    /// fact, and putting it here rather than in the request path means it holds for
    /// *every* way a device comes into existence — including the mock backend's
    /// direct construction, where a test device would otherwise be exempt from the
    /// contract the tests exist to check.
    ///
    /// A failure here is [`RhiErrorKind::BackendFailure`] rather than
    /// [`RhiErrorKind::Unsupported`]: every device is required to have a lane
    /// accepting `RASTER | COPY`, so a snapshot without one is a defect in the
    /// enumeration that produced it, not a capability the caller may not use. The
    /// device is not published, which is the point — section 6.9 forbids handing a
    /// portable defect down for a driver or a validation layer to discover later.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::BackendFailure`] when the enumeration violates the base
    /// guarantee, naming which half of it was violated.
    pub(crate) fn new(identity: DeviceIdentity, native: Box<dyn DeviceBackend>) -> RhiResult<Self> {
        let capabilities = EnabledCapabilities::from_facts(
            native.capability_facts(),
            native.submission_capabilities(),
        );
        // The `Compute` feature is read back out of the contract that was just
        // interned rather than asked of the backend separately: section 7.2's rule
        // is about the *enabled* feature set, and reading it from anywhere else
        // would let the two answers disagree at exactly the moment the rule is
        // being checked.
        capabilities
            .submission()
            .validate_base_guarantee(capabilities.supports_feature(OptionalFeature::Compute))?;
        Ok(Self {
            inner: Arc::new(DeviceInner {
                identity,
                native,
                capabilities,
                serials: DomainSerials::new(),
                interning: DomainInterning::new(),
                runtime: RuntimeServices::new(),
            }),
        })
    }

    /// The plan and acceptance serials of this domain.
    ///
    /// Crate-private, and reached by the two places that mint from it:
    /// [`crate::api::submission::SubmissionPlanBuilder`] takes a plan serial at
    /// construction, and [`Self::submit`] takes an acceptance serial once the
    /// backend has accepted. Nothing else may mint one, which is what section
    /// 39.1's "a plan identity is minted by the builder that owns it" requires.
    pub(crate) fn serials(&self) -> &DomainSerials {
        &self.inner.serials
    }

    /// The portable state cell for native debugger capture nesting.
    pub(crate) fn native_capture_active(&self) -> &Mutex<bool> {
        &self.inner.runtime.native_capture_active
    }

    /// The exact compatibility tokens this domain has minted.
    ///
    /// Crate-private, and reached by the two chapters that mint from it:
    /// [`crate::api::binding`] interns canonical layout descriptors (section 21.1)
    /// and [`crate::api::pipeline`] interns canonical interface descriptors
    /// (section 21.2). Nothing else may mint one, which is what section 21.1's "a
    /// caller cannot construct it" requires.
    pub(crate) fn interning(&self) -> &DomainInterning {
        &self.inner.interning
    }

    /// Runtime service state is owned by the one device inner, so every public
    /// device clone observes exactly the same queues and collection epoch.
    pub(crate) fn diagnostics_queue(&self) -> &Mutex<Vec<DiagnosticEvent>> {
        &self.inner.runtime.diagnostics
    }

    pub(crate) fn statistics_state(&self) -> &Mutex<StatisticsState> {
        &self.inner.runtime.statistics
    }

    pub(crate) fn retain_captured_work(&self, work: CapturedRecordedWork) {
        self.inner
            .runtime
            .captured_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(work.work, work);
    }

    pub(crate) fn captured_work(&self, id: ObjectId) -> Option<CapturedRecordedWork> {
        self.inner
            .runtime
            .captured_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&id)
            .cloned()
    }

    pub(crate) fn insert_observer(&self, observer: Arc<dyn SemanticObserver>) -> u64 {
        let mut state = self
            .inner
            .runtime
            .observers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = state.next;
        state.next = state.next.saturating_add(1);
        state.entries.push((id, observer));
        id
    }

    pub(crate) fn remove_observer(&self, id: u64) {
        let mut state = self
            .inner
            .runtime
            .observers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.entries.retain(|(entry, _)| *entry != id);
    }

    pub(crate) fn dispatch_diagnostic(&self, diagnostic: &DiagnosticEvent) {
        let (event, observers) = {
            let mut state = self
                .inner
                .runtime
                .observers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let event = SemanticEventId::new(state.next_event);
            state.next_event = state.next_event.saturating_add(1);
            let observers = state
                .entries
                .iter()
                .map(|(_, observer)| Arc::clone(observer))
                .collect::<Vec<_>>();
            (event, observers)
        };
        for observer in observers {
            observer.on_event(SemanticEvent::Diagnostic { event, diagnostic });
        }
    }

    /// A shared handle to this device's capability snapshot.
    ///
    /// Distinct from [`Self::capabilities`], which lends the same value: a caller
    /// that must *store* it needs an owned handle, and copying the snapshot would
    /// be both wasteful and a second answer to questions section 7.2 makes
    /// immutable. The one caller is `Device::create_recorder`, whose recorder holds
    /// the snapshot rather than the device — see that type's documentation for why
    /// the narrow shape is the one that keeps a recorder unable to lower.
    /// This device's identity.
    ///
    /// Every object the device owns carries the same identity, and section 3.1
    /// makes the comparison against it the first thing any public operation does
    /// — in O(1), before a backend is touched.
    pub fn identity(&self) -> DeviceIdentity {
        self.inner.identity
    }

    /// The native domain this handle lowers through.
    ///
    /// Crate-private because section 59 keeps native lowering out of the exported
    /// surface and because no caller outside this crate may name a backend — the
    /// traits it returns are `pub(crate)` for the same reason, so this is not a
    /// leak with a narrow door but the seam's ordinary inside face.
    ///
    /// The callers are the creation verbs of the later chapters, which live
    /// beside the types they produce (adjudication A28) rather than here, and
    /// therefore need to reach the backend through the handle they were given.
    /// Reaching it *through* the handle rather than storing a copy is what keeps
    /// one device from having two authoritative backends.
    pub(crate) fn native(&self) -> &dyn DeviceBackend {
        &*self.inner.native
    }

    /// The backend family this device came from.
    ///
    /// For diagnostics, UI, capture provenance, and benchmark reports only. It
    /// is not a capability oracle: the same family exposes different
    /// capabilities on different drivers, so asking the backend what it is
    /// instead of asking the device what it can do is the mistake section 6.3
    /// names.
    pub fn backend(&self) -> BackendKind {
        self.inner.native.backend_kind()
    }

    /// A snapshot of the adapter that was actually selected.
    ///
    /// Available even when the provider does not support adapter enumeration:
    /// section 6.3 asks a device to report what it actually got, which is a
    /// weaker and always-answerable question than listing the candidates.
    pub fn adapter_info(&self) -> &AdapterInfo {
        self.inner.native.adapter_info()
    }

    /// What this device can actually do.
    ///
    /// The only correct source of capability answers. The relationship to the
    /// adapter's snapshot is one-way:
    ///
    /// ```text
    /// EnabledOnDevice subset-of AvailableOnAdapter
    /// ```
    ///
    /// so a feature the adapter reported as available may still be absent here,
    /// and a caller that planned against the adapter would be wrong.
    ///
    /// The contract was interned once, at construction, so this is a borrow of
    /// storage the device already owns rather than a query: two calls, and two
    /// calls on two clones of one device, return the same
    /// [`crate::api::capability::CapabilityCompatibilityId`] — which is what
    /// section 7.1's `CompiledGraph` reuse rule reads.
    pub fn capabilities(&self) -> &EnabledCapabilities {
        &self.inner.capabilities
    }

    /// Whether the device is still usable.
    ///
    /// The answer comes from the backend, which is what observes a loss, and it
    /// is asked every time rather than cached here: caching it would give the
    /// crate two places that know whether this device is alive, and section 65.3
    /// allows exactly one authority per concern.
    pub fn status(&self) -> DeviceStatus {
        self.inner.native.status()
    }

    /// Why the device was lost, or `None` while it is active.
    pub fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.inner.native.loss_info()
    }

    /// Non-blockingly advances completion, loss, and callback bookkeeping.
    ///
    /// This is RHI-owned bookkeeping, not the host's event loop. Section 6.6
    /// requires the host to keep pumping its own loop; a host that polled this
    /// instead would starve the browser or window messages the RHI depends on.
    ///
    /// Pending work must not stay pending forever (section 6.5), which is what
    /// makes polling the device a way to make progress rather than only to
    /// observe it.
    pub fn poll(&self) -> RhiResult<()> {
        self.require_active()
            .map_err(|error| error.at("Device::poll"))?;
        self.inner.native.poll()
    }

    /// Waits until the device is idle.
    ///
    /// For shutdown and diagnostics only. Section 6.7 forbids it as a per-frame
    /// retirement mechanism and as the correctness mechanism of a render loop:
    /// a caller that needs to know when work finished is asking about a
    /// completion point, and a caller that needs resources back is asking about
    /// retirement. On a restricted host or backend this returns
    /// [`crate::api::error::RhiErrorKind::Unsupported`] rather than pretending to
    /// have waited.
    pub async fn wait_idle(&self) -> RhiResult<()> {
        self.require_active()
            .map_err(|error| error.at("Device::wait_idle"))?;
        self.inner.native.wait_idle()
    }

    /// This device's process-local object ID.
    ///
    /// Section 3 gives every RHI object a process-local [`ObjectId`] distinct
    /// from any native handle, and section 7.1 requires tooling to be able to
    /// *describe* what it observes by that ID rather than by a pointer.
    ///
    /// Sections 3 through 7 declare no accessor that yields an `ObjectId`, so
    /// this verb is an addition rather than a transcription. It is added because
    /// the alternative is worse: `RhiError::object` returns an `ObjectId` and
    /// section 7.1 requires tooling to describe objects by one, which is
    /// unreachable if no object can name its own ID.
    pub fn object_id(&self) -> ObjectId {
        self.inner.native.object_id()
    }

    /// Refuses an operation that would use this device while it is lost.
    ///
    /// Section 6.5 states the rule for exactly this case, so it is quoted rather
    /// than paraphrased: after a loss the handles listed there "must return
    /// `WrongDevice` when passed to that new Device, and return **`DeviceLost`
    /// when used through their lost original Device**". Every creation verb is
    /// such a use, which is why each one calls this.
    ///
    /// It is a portable verdict and not something the backend is left to notice.
    /// Section 6.9 names this case beside the wrong-device one and draws the line
    /// in the same place for both:
    ///
    /// ```text
    /// wrong device / device lost
    ///     -> Fluxel RHI structured validation -> Err(WrongDevice | DeviceLost)
    /// ```
    ///
    /// rather than passing a stale handle down and letting a driver, a validation
    /// layer, or a browser "handle it unpredictably". The reason section 6.9
    /// gives is the reason this belongs here: native validation may not even be
    /// enabled in a release environment.
    ///
    /// # Why this runs *after* the ownership comparison
    ///
    /// Section 3.1 puts the O(1) identity comparison first, and section 6.5 gives
    /// the two questions different answers. An object handed to a device that is
    /// not its own is `WrongDevice` even when that device is also lost; if
    /// liveness were checked first, such a caller would be told `DeviceLost` when
    /// the actual mistake was the object it passed. So this check sits after
    /// every portable ownership verdict and before the first device-fact read.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::DeviceLost`], carrying section 6.5's stable loss summary
    /// when one was recorded. The summary is in the message rather than beside it
    /// because [`DeviceLossInfo`]'s own accessor is on the device, and an error
    /// that says only "lost" would send every caller back to ask the device a
    /// question this call site already had the answer to.
    pub(crate) fn require_active(&self) -> RhiResult<()> {
        match self.inner.native.status() {
            DeviceStatus::Active => Ok(()),
            DeviceStatus::Lost => {
                let message = match &self.inner.native.loss_info() {
                    Some(loss) => format!(
                        "this device is lost and section 6.5 makes loss terminal, so this \
                         operation cannot be performed through it: {}",
                        loss.message()
                    ),
                    None => "this device is lost and section 6.5 makes loss terminal, so this \
                             operation cannot be performed through it"
                        .to_string(),
                };
                Err(RhiError::new(RhiErrorKind::DeviceLost, message))
            }
        }
    }
}

impl core::fmt::Debug for Device {
    /// Prints portable identity, status, and loss summary.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan. The trait is required rather than optional: section
    /// an awaited device request may yield this value to a caller that logs it,
    /// so the payload must be printable. Printing the execution domain instead would be wrong on two
    /// counts — the backend port will add a native field that has no reason to be
    /// `Debug`, and a device's native state is not something a log should
    /// describe.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Device")
            .field("identity", &self.inner.identity)
            .field("status", &self.inner.native.status())
            .field("loss", &self.inner.native.loss_info())
            .finish_non_exhaustive()
    }
}
