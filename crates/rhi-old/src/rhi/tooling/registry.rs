//! The observer registry: the device-scoped delivery gate, the observer set,
//! and the event-identity mint (rhi-design sections 53.2 and 58.2).
//!
//! This is the mechanism behind the contract [`super::access`] states. It is a
//! separate module because it changes for a different reason: the consumer
//! surface changes when the SPI vocabulary or the promised contract changes,
//! and this changes when the *enforcement* of that contract changes.
//!
//! # The lock layout
//!
//! Three pieces of state implement the linearization contract, and nothing else
//! does:
//!
//! ```text
//! gate     Mutex<()>                  held for the whole of one emit, across every callback
//! set      Mutex<ObserverSet>         the observer slots; held only to insert, remove, snapshot
//! side     Mutex<Side>                who holds the gate right now, and the violation counters
//! ```
//!
//! *The gate is what makes delivery ordered and `Drop` a wait.* One `emit_with`
//! holds it from before the event identity is assigned until after the last
//! callback returns, so two emitting threads cannot interleave their callbacks
//! and an observer sees events in the order their ids were assigned. `Drop`
//! takes the same gate, so it cannot return until every callback that began
//! before it has returned, and it removes the slot while still holding the gate,
//! so no later emitter can snapshot the observer. That is the whole of "receives
//! every event assigned before its Drop point, and none after".
//!
//! *One `set` acquisition for the identity and the snapshot is what makes the
//! start point exact.* `emit_with` assigns the event identity and clones the
//! observer list under the same `set` lock, then releases `set` before calling
//! anything. So an observer inserted by a concurrent `subscribe` is either in
//! the snapshot — and then the identity did not exist when it was inserted, so
//! the event really is assigned after its start point — or it is not, and it
//! never sees an event that was assigned before it arrived. Releasing `set`
//! before the callbacks is what additionally lets `subscribe` run while a
//! callback is in progress without deadlocking on the set.
//!
//! *`side` is what makes the prohibited cases refusable instead of deadlocking.*
//! A callback runs while its thread holds the gate, so a `Drop` that would wait
//! on the gate from inside a callback would wait on itself forever. The registry
//! therefore records the gate holder's `ThreadId` in `side` — readable without
//! the gate — and `Drop` checks it first:
//!
//! ```text
//! self-drop from inside on_event  -> unregister without waiting, count the violation
//! subscribe from inside on_event  -> refuse with InvalidUsage, count the violation
//! emit from inside a callback     -> refuse the nested event, count the violation
//! ```
//!
//! The `ToolingViolation::ReentrantEmit` refusal consumes no event identity, and
//! `emit_with` reports it as `None`, so a nested emitter cannot silently advance
//! the device's event numbering.
//!
//! Lock order is `gate -> set` and `gate -> side`. `side` is never held while
//! waiting for anything, so the checks above cannot deadlock against the gate.
//!
//! # What a backend owns
//!
//! A backend implements [`ToolingBackend`] and owns exactly one
//! [`ToolingRegistry`] per device identity, alongside the objects it describes.
//! Keeping the registry device-scoped is what makes `SemanticEventId` monotonic
//! within one `DeviceIdentity` without a global counter, and what makes an event
//! from a lost identity impossible to deliver to a replacement identity's
//! observers. The registry lives here rather than in the backend so that every
//! backend gets one implementation of the rules above instead of one each.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};

use super::super::platform::{ObjectId, RhiResult};
use super::access::{SemanticObserver, ToolingViolationCounts};
use super::events::{SemanticEvent, SemanticEventId};
use super::objects::CapturedObjectDefinition;
use super::work::CapturedRecordedWork;

/// The definitions a device backend supplies to the tooling SPI.
///
/// This is the seam the SPI is written against: the tooling module owns the
/// vocabulary, the subscription linearization, and the observer registry, and a
/// backend owns the objects. A backend supplies definitions lazily, so a
/// capture pays for a definition only when a consumer asks for the object it
/// describes, and never pays for a second debug IR while nobody is observing
/// (section 52.2).
///
/// It is crate-private on purpose: an implementation is part of the backend,
/// not part of the SPI a consumer codes against.
pub(crate) trait ToolingBackend: Send + Sync + 'static {
    /// This device's observer registry, when this backend exposes the tooling
    /// SPI.
    ///
    /// `None` means the backend implements no observation, and the SPI answers
    /// `Unsupported` rather than accepting observers it would never call.
    fn observers(&self) -> Option<Arc<ToolingRegistry>>;

    /// Describes a live object of this device.
    fn describe_object(&self, id: ObjectId) -> RhiResult<CapturedObjectDefinition>;

    /// Describes live recorded work of this device.
    fn describe_work(&self, work: ObjectId) -> RhiResult<CapturedRecordedWork>;
}

/// One device identity's observer set and event-delivery gate.
pub(crate) struct ToolingRegistry {
    /// The observer slots. Held only for allocation, removal, and snapshot.
    set: Mutex<ObserverSet>,
    /// The delivery gate. Held for the whole of one `emit`.
    gate: Mutex<()>,
    /// The gate holder and the violation counters.
    side: Mutex<Side>,
    /// The next event id to mint.
    next_event: AtomicU64,
}

/// The observer slots and the free list used to reuse them.
#[derive(Default)]
struct ObserverSet {
    slots: Vec<Option<Arc<dyn SemanticObserver>>>,
    free: Vec<usize>,
}

/// Readable-without-the-gate state.
#[derive(Default)]
struct Side {
    /// The thread holding the gate, when one does.
    emitter: Option<ThreadId>,
    /// The cumulative violation counts.
    violations: ToolingViolationCounts,
}

/// A violation an observer committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolingViolation {
    /// An observer dropped its own subscription from `on_event`.
    SelfDrop,
    /// A callback re-entered a tooling entry point on its own device.
    ReentrantEntry,
    /// A nested `emit` was attempted from inside a callback.
    ReentrantEmit,
    /// An observer callback panicked.
    CallbackPanicked,
}

/// Holds the gate for one `emit` and records who holds it.
struct DispatchGuard<'a> {
    registry: &'a ToolingRegistry,
    /// Released by the field's own drop, after this guard's `Drop` body has run,
    /// so the emitter field is always cleared while the gate is still held.
    #[allow(
        dead_code,
        reason = "the gate is held for its lifetime, not for its value"
    )]
    gate: MutexGuard<'a, ()>,
}

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        let mut side = self.registry.lock_side();
        side.emitter = None;
    }
}

impl ToolingRegistry {
    /// A registry with no observers and no minted event ids.
    pub(crate) fn new() -> Self {
        Self {
            set: Mutex::new(ObserverSet::default()),
            gate: Mutex::new(()),
            side: Mutex::new(Side::default()),
            next_event: AtomicU64::new(1),
        }
    }

    /// Assigns one event identity and delivers the event it belongs to.
    ///
    /// This is the only way to publish an event, and the two halves are
    /// deliberately one operation. Assignment and the observer snapshot happen
    /// under a single acquisition of the observer set, so an observer inserted
    /// after this event's id was assigned can never be in the snapshot the id is
    /// delivered from, and every observer in that snapshot was inserted before
    /// the id existed. Splitting the mint from the delivery — `next_event_id()`
    /// followed later by an `emit()` — would leave a window in which a
    /// subscription receives an event assigned before its start point, which the
    /// linearization contract forbids.
    ///
    /// `build` is called with the assigned identity and returns the event that
    /// carries it, borrowing whatever definition the caller owns for the length
    /// of the call.
    ///
    /// The whole delivery holds the gate, so two emitters cannot interleave and
    /// an observer sees events in increasing `SemanticEventId` order.
    ///
    /// Returns `None` when the emitter is inside one of this registry's own
    /// callbacks: a nested event is not delivered, no identity is consumed, and
    /// the violation is counted.
    pub(crate) fn emit_with<'a>(
        &self,
        build: impl FnOnce(SemanticEventId) -> SemanticEvent<'a>,
    ) -> Option<SemanticEventId> {
        let Some(_gate) = self.begin_dispatch() else {
            self.note_violation(ToolingViolation::ReentrantEmit);
            return None;
        };
        let (event, targets) = {
            let set = self.lock_set();
            let event = SemanticEventId::from_raw(self.next_event.fetch_add(1, Ordering::Relaxed));
            let targets: Vec<Arc<dyn SemanticObserver>> =
                set.slots.iter().flatten().cloned().collect();
            (event, targets)
        };
        let semantic = build(event);
        for observer in &targets {
            // An observer is diagnostics: a panic in one must not abort the
            // device operation, nor silence the observers behind it.
            let outcome = catch_unwind(AssertUnwindSafe(|| observer.on_event(semantic)));
            if outcome.is_err() {
                self.note_violation(ToolingViolation::CallbackPanicked);
            }
        }
        Some(event)
    }

    /// How many observers are currently registered.
    ///
    /// A backend checks this to skip building a definition nobody will read.
    /// Skipping is legal because an unobserved event has no consumer to order
    /// against; it is not legal to skip the materialization of a recording's
    /// own portable semantics, which must exist for its whole live lifetime
    /// regardless of observers (section 52.2).
    pub(crate) fn observer_count(&self) -> usize {
        let set = self.lock_set();
        set.slots.iter().filter(|slot| slot.is_some()).count()
    }

    /// The cumulative violation counts.
    pub(crate) fn violations(&self) -> ToolingViolationCounts {
        self.lock_side().violations
    }

    /// Adds one observer and returns its slot.
    ///
    /// Insertion happens under `set` only, and `emit_with` takes both its
    /// snapshot and its event identity under that same lock, which is what makes
    /// the start point exact: an observer inserted here cannot appear in a
    /// snapshot whose event identity already existed when it was inserted.
    pub(crate) fn allocate(&self, observer: Arc<dyn SemanticObserver>) -> usize {
        let mut set = self.lock_set();
        match set.free.pop() {
            Some(slot) => {
                set.slots[slot] = Some(observer);
                slot
            }
            None => {
                set.slots.push(Some(observer));
                set.slots.len() - 1
            }
        }
    }

    /// Removes one observer from the delivery set.
    ///
    /// It is idempotent in effect: a slot that was already removed, or was
    /// never allocated, is simply already absent. Removing an entry that a
    /// running snapshot has cloned does not invalidate that snapshot's `Arc`,
    /// so the in-flight callback finishes with the observer it started with.
    pub(crate) fn unregister(&self, slot: usize) {
        let mut set = self.lock_set();
        if let Some(entry) = set.slots.get_mut(slot) {
            if entry.is_some() {
                *entry = None;
                set.free.push(slot);
            }
        }
    }

    /// Takes the gate and records this thread as its holder.
    ///
    /// Returns `None` when this thread already holds it, which means the call
    /// came from inside a callback of this registry.
    fn begin_dispatch(&self) -> Option<DispatchGuard<'_>> {
        if self.is_emitting_on_current_thread() {
            return None;
        }
        let gate = self.lock_gate();
        let mut side = self.lock_side();
        side.emitter = Some(thread::current().id());
        Some(DispatchGuard {
            registry: self,
            gate,
        })
    }

    /// Whether this thread is currently inside one of this registry's callbacks.
    ///
    /// Only the gate holder can be, so this is exactly "would waiting on the
    /// gate wait on myself".
    pub(crate) fn is_emitting_on_current_thread(&self) -> bool {
        self.lock_side().emitter == Some(thread::current().id())
    }

    /// Adds one to a violation counter.
    pub(crate) fn note_violation(&self, violation: ToolingViolation) {
        let mut side = self.lock_side();
        let counts = &mut side.violations;
        match violation {
            ToolingViolation::SelfDrop => {
                counts.self_drop_attempts = counts.self_drop_attempts.saturating_add(1);
            }
            ToolingViolation::ReentrantEntry | ToolingViolation::ReentrantEmit => {
                counts.reentrant_attempts = counts.reentrant_attempts.saturating_add(1);
            }
            ToolingViolation::CallbackPanicked => {
                counts.callback_panics = counts.callback_panics.saturating_add(1);
            }
        }
    }

    /// Locks the gate, recovering from a poisoned lock.
    ///
    /// A panic inside a callback is contained, so poisoning can only come from a
    /// panic in this module's own bookkeeping. Recovering keeps one bad observer
    /// from permanently disabling observation for the whole device.
    pub(crate) fn lock_gate(&self) -> MutexGuard<'_, ()> {
        self.gate.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the observer set, recovering from a poisoned lock.
    fn lock_set(&self) -> MutexGuard<'_, ObserverSet> {
        self.set.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the side state, recovering from a poisoned lock.
    fn lock_side(&self) -> MutexGuard<'_, Side> {
        self.side.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl core::fmt::Debug for ToolingRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ToolingRegistry")
            .field("observers", &self.observer_count())
            .finish_non_exhaustive()
    }
}
