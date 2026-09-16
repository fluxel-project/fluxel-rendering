//! Tests for the context-generation mapping.
//!
//! The mapping is four lines of code and the tests are the argument for the four
//! lines: the last one below is the reason the identity is allocated rather than
//! derived from the stamp, and it is the only test here that a plausible
//! alternative implementation would fail.

use super::*;
use crate::webgl2::api::{ContextEpoch, DeviceIdentity as GlDeviceIdentity};

/// The stamp of a context on GL device `device` at generation `epoch`.
///
/// Generations are counted from one, matching [`ContextEpoch::INITIAL`], so the
/// call sites below read as generation numbers rather than as offsets.
fn stamp(device: u64, epoch: u64) -> ContextStamp {
    assert_ne!(epoch, 0, "generations start at one");
    let mut generation = ContextEpoch::INITIAL;
    for _ in 1..epoch {
        generation = generation.checked_next().expect("a representable epoch");
    }
    ContextStamp::new(
        GlDeviceIdentity::new(device).expect("a nonzero GL device identity"),
        generation,
    )
}

#[test]
fn adopting_the_stamp_already_held_is_not_a_generation_change() {
    let mut map = DeviceIdentityMap::new(stamp(1, 1));
    let before = map.identity();

    assert!(
        !map.adopt(stamp(1, 1)),
        "the generation is the one already held, so nothing was recreated"
    );
    assert_eq!(
        map.identity(),
        before,
        "a context that did not change generation keeps the identity its bound objects carry"
    );
}

#[test]
fn a_restored_context_is_a_new_generation_and_a_new_common_identity() {
    let mut map = DeviceIdentityMap::new(stamp(1, 1));
    let before = map.identity();

    assert!(
        map.adopt(stamp(1, 2)),
        "the epoch advanced, so the generation did"
    );
    assert_ne!(
        map.identity(),
        before,
        "a changed generation is what 0.14 residency sees instead of the resources \
         the new context never held"
    );
    assert_eq!(map.stamp(), stamp(1, 2));
}

#[test]
fn a_different_gl_device_is_a_new_common_identity() {
    let mut map = DeviceIdentityMap::new(stamp(1, 1));
    let before = map.identity();

    assert!(map.adopt(stamp(2, 1)));
    assert_ne!(map.identity(), before);
}

/// The property the allocation exists for, and the one derivation would fail.
///
/// A stamp is not unique to a live context: two contexts created on one GL
/// device can be handed the same allocator metadata, so the two genuinely are
/// indistinguishable by stamp.  The common identity is not a restatement of that
/// metadata -- it is the value the graph rejects a foreign bound resource with
/// -- so two live contexts must not share it, or a resource bound to one would
/// be accepted by the other.
#[test]
fn two_contexts_of_one_gl_device_never_share_a_common_identity() {
    let mut first = DeviceIdentityMap::new(stamp(7, 1));
    let second = DeviceIdentityMap::new(stamp(7, 1));
    let before = first.identity();

    assert_eq!(
        first.stamp(),
        second.stamp(),
        "the two contexts really are indistinguishable by stamp"
    );
    assert_ne!(first.identity(), second.identity());
    assert!(
        !first.adopt(second.stamp()),
        "one context's stamp is not a generation change for the other"
    );
    assert_eq!(
        first.identity(),
        before,
        "so the second context's existence left the first one's identity alone"
    );
}
