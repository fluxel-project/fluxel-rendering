//! The subscription linearization contract (rhi-design section 53.2), and the
//! two prohibited callback cases of section 58.2.
//!
//! Every test here is about an ordering or a wait: which events a subscription
//! is entitled to, in what order, how long `Drop` waits, and which prohibited
//! re-entries are refused instead of deadlocked.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use super::super::registry::ToolingRegistry;
use super::{FakeDevice, Recorder, definition_id, small_buffer, subscribe};
use crate::rhi::platform::RhiErrorKind;
use crate::rhi::tooling::{
    CapturedObjectDefinition, SemanticEvent, SemanticObserver, ToolingAccess, ToolingSubscription,
};

#[test]
fn an_event_assigned_before_subscribe_is_not_delivered_and_a_later_one_is() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    // Assigned before the start point: the observer must never see it.
    let before = device.emit_created(id);

    let access = device.access();
    let recorder = Arc::new(Recorder::default());
    let _subscription = subscribe(&access, Arc::clone(&recorder) as Arc<dyn SemanticObserver>);

    // Assigned after the start point: the observer must see it.
    let after = device.emit_created(id);

    assert!(
        after > before,
        "event ids are monotonic within one device identity"
    );
    assert_eq!(
        recorder.event_ids(),
        vec![after.as_u64()],
        "only the event assigned after the subscription point is delivered"
    );
}

#[test]
fn events_are_delivered_once_and_in_increasing_id_order() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let recorder = Arc::new(Recorder::default());
    let _subscription = subscribe(&access, Arc::clone(&recorder) as Arc<dyn SemanticObserver>);

    let minted: Vec<u64> = (0..4)
        .map(|_| device.emit_created(id).as_u64())
        .collect();

    assert_eq!(
        recorder.event_ids(),
        minted,
        "each observer receives each event exactly once, in id order"
    );
    assert!(
        minted.windows(2).all(|pair| pair[0] < pair[1]),
        "the mint is strictly monotonic"
    );
}

#[test]
fn every_observer_receives_every_event_exactly_once() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let first = Arc::new(Recorder::default());
    let second = Arc::new(Recorder::default());
    let _a = subscribe(&access, Arc::clone(&first) as Arc<dyn SemanticObserver>);
    let _b = subscribe(&access, Arc::clone(&second) as Arc<dyn SemanticObserver>);

    let minted = device.emit_created(id).as_u64();

    assert_eq!(first.event_ids(), vec![minted]);
    assert_eq!(second.event_ids(), vec![minted]);
}

#[test]
fn a_dropped_subscription_stops_receiving_events() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let recorder = Arc::new(Recorder::default());
    let subscription = subscribe(&access, Arc::clone(&recorder) as Arc<dyn SemanticObserver>);

    let first = device.emit_created(id).as_u64();
    drop(subscription);
    let _second = device.emit_created(id);

    assert_eq!(
        recorder.event_ids(),
        vec![first],
        "Drop is the unregistration point"
    );
}

#[test]
fn dropping_a_subscription_waits_for_an_in_flight_callback() {
    /// An observer that parks inside `on_event` until the test releases it.
    struct Parking {
        entered: Sender<()>,
        release: Mutex<Option<Receiver<()>>>,
    }

    impl SemanticObserver for Parking {
        fn on_event(&self, _event: SemanticEvent<'_>) {
            self.entered.send(()).expect("the test is still listening");
            // Blocking inside a callback is forbidden by section 58.2, and this
            // is the only way to prove that Drop waits for one.
            let release = self
                .release
                .lock()
                .expect("release channel")
                .take()
                .expect("exactly one event");
            release.recv().expect("the test releases the callback");
        }
    }

    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let (entered_tx, entered_rx) = channel();
    let (release_tx, release_rx) = channel();
    let (done_tx, done_rx) = channel();
    let parking = Arc::new(Parking {
        entered: entered_tx,
        release: Mutex::new(Some(release_rx)),
    });
    let subscription = subscribe(&access, Arc::clone(&parking) as Arc<dyn SemanticObserver>);

    let emitting = Arc::clone(&device);
    let emitter = thread::spawn(move || {
        emitting.emit_created(id);
    });

    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the callback began");

    let dropper = thread::spawn(move || {
        drop(subscription);
        done_tx.send(()).expect("the test is still listening");
    });

    // Give the drop every chance to return while the callback is still parked.
    assert!(
        done_rx.recv_timeout(Duration::from_millis(200)).is_err(),
        "Drop must wait for a callback that began before it"
    );

    release_tx.send(()).expect("the callback is still parked");
    done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("Drop returns once the callback has returned");

    emitter.join().expect("the emitting thread finished");
    dropper.join().expect("the dropping thread finished");
}

/// An observer that keeps its own subscription and drops it from `on_event`.
struct SelfDropping {
    subscription: Mutex<Option<ToolingSubscription>>,
    seen: Mutex<usize>,
}

impl SemanticObserver for SelfDropping {
    fn on_event(&self, _event: SemanticEvent<'_>) {
        *self.seen.lock().expect("count") += 1;
        let subscription = self
            .subscription
            .lock()
            .expect("subscription")
            .take()
            .expect("exactly one event");
        // The prohibited drop. It must return, not deadlock.
        drop(subscription);
    }
}

#[test]
fn dropping_its_own_subscription_from_on_event_is_refused_rather_than_deadlocking() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let observer = Arc::new(SelfDropping {
        subscription: Mutex::new(None),
        seen: Mutex::new(0),
    });

    let subscription = subscribe(&access, Arc::clone(&observer) as Arc<dyn SemanticObserver>);
    *observer.subscription.lock().expect("subscription") = Some(subscription);

    // If the refusal were a wait, this call would never return.
    device.emit_created(id);

    assert_eq!(*observer.seen.lock().expect("count"), 1);
    assert_eq!(
        access.violations().self_drop_attempts,
        1,
        "the refused drop is recorded rather than silent"
    );

    // The refusal still honoured the unregistration at that instant.
    device.emit_created(id);
    assert_eq!(*observer.seen.lock().expect("count"), 1);
}

/// An observer that tries to subscribe while it is being called.
struct Reentrant {
    access: ToolingAccess,
    outcome: Mutex<Option<RhiErrorKind>>,
}

impl SemanticObserver for Reentrant {
    fn on_event(&self, _event: SemanticEvent<'_>) {
        let outcome = self
            .access
            .subscribe(Arc::new(Recorder::default()) as Arc<dyn SemanticObserver>)
            .err()
            .map(|error| error.kind());
        *self.outcome.lock().expect("outcome") = outcome;
    }
}

#[test]
fn subscribing_from_on_event_is_refused_with_a_structured_error() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let observer = Arc::new(Reentrant {
        access: access.clone(),
        outcome: Mutex::new(None),
    });
    let _subscription = subscribe(&access, Arc::clone(&observer) as Arc<dyn SemanticObserver>);

    device.emit_created(id);

    assert_eq!(
        *observer.outcome.lock().expect("outcome"),
        Some(RhiErrorKind::InvalidUsage)
    );
    assert_eq!(access.violations().reentrant_attempts, 1);
}

/// An observer that tries to emit a nested event while it is being called.
struct ReentrantEmitter {
    registry: Arc<ToolingRegistry>,
    definition: CapturedObjectDefinition,
    /// Whether the nested emission handed back an event identity.
    nested_minted: Mutex<Option<bool>>,
}

impl SemanticObserver for ReentrantEmitter {
    fn on_event(&self, _event: SemanticEvent<'_>) {
        let nested = self.registry.emit_with(|event| SemanticEvent::ObjectCreated {
            event,
            definition: &self.definition,
        });
        *self.nested_minted.lock().expect("nested outcome") = Some(nested.is_some());
    }
}

#[test]
fn emitting_from_on_event_is_refused_and_consumes_no_event_identity() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let observer = Arc::new(ReentrantEmitter {
        registry: Arc::clone(&device.registry),
        definition: small_buffer(),
        nested_minted: Mutex::new(None),
    });
    let _subscription = subscribe(&access, Arc::clone(&observer) as Arc<dyn SemanticObserver>);

    let delivered = device.emit_created(id);

    assert_eq!(
        *observer.nested_minted.lock().expect("nested outcome"),
        Some(false),
        "a nested emission must not be delivered"
    );
    assert_eq!(access.violations().reentrant_attempts, 1);

    // A refused emission must not advance the device's event numbering, or a
    // consumer ordering two events would see a gap it cannot explain.
    let next = device.emit_created(id);
    assert_eq!(next.as_u64(), delivered.as_u64() + 1);
}

/// An observer that panics on every event.
struct Panicking;

impl SemanticObserver for Panicking {
    fn on_event(&self, _event: SemanticEvent<'_>) {
        panic!("an observer is diagnostics, not a correctness channel");
    }
}

#[test]
fn a_panicking_observer_does_not_silence_the_others() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    device.with_object(id, object);

    let access = device.access();
    let _panicking = subscribe(&access, Arc::new(Panicking) as Arc<dyn SemanticObserver>);
    let recorder = Arc::new(Recorder::default());
    let _recorder = subscribe(&access, Arc::clone(&recorder) as Arc<dyn SemanticObserver>);

    let minted = device.emit_created(id).as_u64();

    assert_eq!(
        recorder.event_ids(),
        vec![minted],
        "the observers behind a panicking one still receive the event"
    );
    assert_eq!(access.violations().callback_panics, 1);

    // The registry is still usable, so a panic is not terminal for the device.
    let second = device.emit_created(id).as_u64();
    assert!(second > minted);
}
