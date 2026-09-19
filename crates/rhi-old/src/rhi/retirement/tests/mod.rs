//! Contract tests for rhi-design 02 section 18.6 and 05 section 41.6.
//!
//! The subject is the one property the contract exists for: a dropped public
//! handle must not be a release, and the release must wait for the last batch
//! that referenced the object to be terminal. Several tests are written against
//! a `Weak`, because "the backing is still alive" is the actual claim and a
//! strong count alone would not distinguish it from a registry that merely
//! remembers an id.

use std::sync::{Arc, Weak};

use super::super::platform::{DeviceIdentity, ObjectId, next_identity, next_object_id};
use super::super::resource::{BufferBackend, BufferDescriptor, BufferUsage};
use super::super::statistics::ObjectKind;
use super::super::submission::CompletionPoint;
use super::{Reclaimed, Retained, RetirementRegistry};

/// A backing that exists only to be held and counted.
struct TestBuffer {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: BufferDescriptor,
}

impl BufferBackend for TestBuffer {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}

/// A fresh backing, its weak handle, and the device it belongs to.
fn backing() -> (Arc<dyn BufferBackend>, Weak<dyn BufferBackend>, DeviceIdentity) {
    let device = next_identity();
    let arc: Arc<dyn BufferBackend> = Arc::new(TestBuffer {
        id: next_object_id(),
        device,
        descriptor: BufferDescriptor::new(64, BufferUsage::COPY_SRC),
    });
    let weak = Arc::downgrade(&arc);
    (arc, weak, device)
}

/// A completion point on `device` at completion serial `serial`.
fn point(device: DeviceIdentity, serial: u64) -> CompletionPoint {
    CompletionPoint::new(device, serial)
}

#[test]
fn a_registered_backing_survives_the_callers_last_handle() {
    let registry = RetirementRegistry::new();
    let (arc, weak, _device) = backing();
    let caller_handle = arc.clone();

    registry.register(Retained::Buffer(arc));
    drop(caller_handle);

    assert!(
        weak.upgrade().is_some(),
        "the registry must hold the backing past the caller's last clone; \
         without it, dropping the last handle would be the release and the GPU \
         could still be reading the object"
    );
    assert_eq!(registry.len(), 1, "and it is still inventoried");
}

#[test]
fn a_sweep_is_what_releases_the_backing() {
    let registry = RetirementRegistry::new();
    let (arc, weak, _device) = backing();
    let id = arc.id();

    registry.register(Retained::Buffer(arc));
    let reclaimed = registry.sweep();

    assert_eq!(
        reclaimed,
        vec![Reclaimed {
            id,
            kind: ObjectKind::Buffer,
        }]
    );
    assert!(
        weak.upgrade().is_none(),
        "the sweep is the release, and it is the only release"
    );
    assert_eq!(registry.len(), 0);
}

#[test]
fn an_object_with_no_accepted_use_is_reclaimed_without_waiting() {
    let registry = RetirementRegistry::new();
    let (arc, weak, _device) = backing();
    registry.register(Retained::Buffer(arc));

    // No completion point references it, so the second condition is vacuous:
    // there is no GPU reader to wait for.
    assert_eq!(registry.sweep().len(), 1);
    assert!(weak.upgrade().is_none());
}

#[test]
fn an_object_still_held_by_a_live_handle_is_not_reclaimed() {
    let registry = RetirementRegistry::new();
    let (arc, weak, _device) = backing();
    let live = arc.clone();

    registry.register(Retained::Buffer(arc));
    let reclaimed = registry.sweep();

    assert!(
        reclaimed.is_empty(),
        "a caller that still holds the object has not stopped referencing it"
    );
    assert!(weak.upgrade().is_some());

    drop(live);
    assert_eq!(
        registry.sweep().len(),
        1,
        "and once it lets go, the sweep frees it"
    );
}

#[test]
fn an_object_whose_only_use_is_pending_is_not_reclaimed() {
    let registry = RetirementRegistry::new();
    let (arc, weak, device) = backing();
    let id = arc.id();
    registry.register(Retained::Buffer(arc));

    registry.note_use(id, point(device, 1));
    assert!(
        registry.sweep().is_empty(),
        "an accepted, unfinished batch still reads this object"
    );
    assert!(weak.upgrade().is_some());

    registry.note_terminal(point(device, 1));
    assert_eq!(registry.sweep().len(), 1);
    assert!(weak.upgrade().is_none());
}

#[test]
fn reclamation_waits_for_the_last_point_not_the_first() {
    let registry = RetirementRegistry::new();
    let (arc, _weak, device) = backing();
    let id = arc.id();
    registry.register(Retained::Buffer(arc));

    let earlier = point(device, 1);
    let later = point(device, 2);
    registry.note_use(id, earlier);
    registry.note_use(id, later);

    registry.note_terminal(earlier);
    assert!(
        registry.sweep().is_empty(),
        "the later batch is still pending, so the object is still referenced"
    );

    registry.note_terminal(later);
    assert_eq!(registry.sweep().len(), 1);
}

#[test]
fn a_use_reported_out_of_order_does_not_lower_the_bar() {
    let registry = RetirementRegistry::new();
    let (arc, _weak, device) = backing();
    let id = arc.id();
    registry.register(Retained::Buffer(arc));

    let earlier = point(device, 1);
    let later = point(device, 7);

    // The later point arrives first. "Last use" is a maximum, not "most
    // recently reported": a submission landing out of order must not replace a
    // pending reader with an already-terminal one.
    registry.note_use(id, later);
    registry.note_use(id, earlier);

    registry.note_terminal(earlier);
    assert!(
        registry.sweep().is_empty(),
        "the higher point is still pending and must still hold the object"
    );

    registry.note_terminal(later);
    assert_eq!(registry.sweep().len(), 1);
}

#[test]
fn a_terminal_point_that_never_referenced_the_object_does_not_release_it() {
    let registry = RetirementRegistry::new();
    let (arc, weak, device) = backing();
    let id = arc.id();
    registry.register(Retained::Buffer(arc));
    registry.note_use(id, point(device, 9));

    // A terminal report for unrelated work says nothing about this object's
    // reader. Matching by "something completed" rather than by the point would
    // release a buffer the GPU is still reading.
    registry.note_terminal(point(device, 3));
    assert!(registry.sweep().is_empty());
    assert!(weak.upgrade().is_some());
}

#[test]
fn device_loss_reclaims_pending_work_too() {
    let registry = RetirementRegistry::new();
    let (arc, weak, device) = backing();
    let id = arc.id();
    registry.register(Retained::Buffer(arc));
    registry.note_use(id, point(device, 1));

    registry.note_device_lost();

    assert_eq!(
        registry.sweep().len(),
        1,
        "no later observation can report a point from a lost device, so waiting \
         for per-point reports would hold every object from it forever"
    );
    assert!(weak.upgrade().is_none());
}

#[test]
fn a_reclaimed_object_is_not_reported_twice() {
    let registry = RetirementRegistry::new();
    let (arc, _weak, _device) = backing();
    registry.register(Retained::Buffer(arc));

    assert_eq!(registry.sweep().len(), 1);
    assert!(
        registry.sweep().is_empty(),
        "the registry does not hold it any more, so it cannot release it again"
    );
}

#[test]
fn a_use_of_an_object_this_registry_never_inventoried_is_ignored() {
    let registry = RetirementRegistry::new();
    let (arc, _weak, device) = backing();
    let foreign = arc.id();

    // A foreign device's object: a portable check should already have refused
    // it. The registry must not panic on the bookkeeping miss, because the
    // missing refusal is the bug and it is not this module's to report.
    registry.note_use(foreign, point(device, 1));

    assert_eq!(registry.len(), 0);
    assert!(registry.sweep().is_empty());
}
