//! Section 20.1-20.2: indices and counts.
//!
//! The bind-group index and the slot id round-trip, and the one-or-array count
//! invariant. `use super::*` brings in the fixtures.

use super::*;

// ---------------------------------------------------------------------------
// Section 20.1 / 20.2: indices and counts.
// ---------------------------------------------------------------------------

#[test]
fn group_index_and_slot_id_round_trip() {
    assert_eq!(BindGroupIndex::new(3).get(), 3);
    assert_eq!(BindingSlotId::new(7).get(), 7);
    assert_ne!(BindGroupIndex::new(0), BindGroupIndex::new(1));
    assert_ne!(BindingSlotId::new(0), BindingSlotId::new(1));
}

#[test]
fn a_binding_count_is_one_or_a_fixed_array_of_at_least_two() {
    assert_eq!(BindingCount::One.elements(), 1);
    assert_eq!(BindingCount::Fixed(2).elements(), 2);
    assert_eq!(BindingCount::Fixed(9).elements(), 9);

    // Section 20.5: `Fixed(n)` means `n >= 2`, refused rather than repaired,
    // because section 22.1 makes `One` the only spelling of a single element.
    let layout = layout_from(vec![uniform_slot(0, 64).with_count(BindingCount::Fixed(1))]);
    assert_kind(
        validate_bind_group_layout_descriptor(layout.descriptor(), 8, permissive),
        RhiErrorKind::InvalidUsage,
    );

    let legal = layout_from(vec![uniform_slot(0, 64).with_count(BindingCount::Fixed(2))]);
    assert!(validate_bind_group_layout_descriptor(legal.descriptor(), 8, permissive).is_ok());
}
