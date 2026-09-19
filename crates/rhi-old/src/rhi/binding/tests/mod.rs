//! Contract tests for bind group descriptors and their entry validation.
//!
//! The subject here is rhi-design section 48.1 for the set-like vectors on the
//! binding surface: a layout's slots and a bind group's entries are sets, so
//! the order a caller filled them in is not part of either object's identity.
//! Canonicalization is also what makes the duplicate check below a neighbour
//! comparison rather than a quadratic scan.

use super::super::shader::ShaderStages;
use super::{
    BindGroupEntry, BindGroupLayoutDescriptor, BindingCount, BindingKind, BindingResource,
    BindingSlot, BindingSlotId, SamplerKind, validate_bind_group_entries, validate_layout_entries,
};

/// A uniform buffer slot with room for a `vec4`.
fn uniform_slot(slot: u32) -> BindingSlot {
    BindingSlot::new(
        BindingSlotId::new(slot),
        ShaderStages::FRAGMENT,
        BindingKind::UniformBuffer { min_size: 16 },
    )
}

/// An entry whose resource class matches [`uniform_slot`].
///
/// An empty array is deliberate: it carries the buffer class the layout wants
/// without needing a device, so these tests can exercise the entry rules as
/// pure data. The count mismatch it also carries is what the dedicated count
/// test below asserts on.
fn uniform_class_entry(slot: u32) -> BindGroupEntry {
    BindGroupEntry::new(
        BindingSlotId::new(slot),
        BindingResource::BufferArray(Vec::new()),
    )
}

/// An entry whose resource class does not match a uniform buffer slot.
fn sampler_class_entry(slot: u32) -> BindGroupEntry {
    BindGroupEntry::new(
        BindingSlotId::new(slot),
        BindingResource::SamplerArray(Vec::new()),
    )
}

/// Sorts entries by ascending slot id.
///
/// Spelled out here rather than calling the descriptor method, so a test that
/// depends on the ordering is not testing the implementation against itself.
fn sorted(mut entries: Vec<BindGroupEntry>) -> Vec<BindGroupEntry> {
    entries.sort_by_key(|entry| entry.slot);
    entries
}

/// Sorts slots by ascending slot id.
fn sorted_slots(mut slots: Vec<BindingSlot>) -> Vec<BindingSlot> {
    slots.sort_by_key(|slot| slot.slot);
    slots
}

/// The raw slot ids of a descriptor, in declaration order.
fn slot_ids(slots: &[BindingSlot]) -> Vec<u32> {
    slots.iter().map(|slot| slot.slot.get()).collect()
}

/// The raw slot ids of an entry list, in declaration order.
fn entry_slot_ids(entries: &[BindGroupEntry]) -> Vec<u32> {
    entries.iter().map(|entry| entry.slot.get()).collect()
}

#[test]
fn layout_slots_canonicalize_to_ascending_slot_order() {
    let descriptor =
        BindGroupLayoutDescriptor::new(vec![uniform_slot(7), uniform_slot(0), uniform_slot(3)]);

    // The raw descriptor keeps the order the caller declared.
    assert_eq!(slot_ids(&descriptor.entries), vec![7, 0, 3]);

    let canonical = descriptor.canonicalized();

    assert_eq!(slot_ids(&canonical.entries), vec![0, 3, 7]);
    assert_eq!(canonical.entries, sorted_slots(descriptor.entries.clone()));
}

#[test]
fn layout_declaration_order_does_not_change_the_canonical_form() {
    let ascending = BindGroupLayoutDescriptor::new(vec![uniform_slot(0), uniform_slot(1)]);
    let descending = BindGroupLayoutDescriptor::new(vec![uniform_slot(1), uniform_slot(0)]);

    assert_ne!(ascending, descending, "the raw descriptors do differ");
    assert_eq!(
        ascending.canonicalized(),
        descending.canonicalized(),
        "two spellings of one layout must canonicalize to one descriptor"
    );
}

#[test]
fn a_repeated_layout_slot_is_rejected() {
    let entries = BindGroupLayoutDescriptor::new(vec![uniform_slot(2), uniform_slot(2)])
        .canonicalized()
        .entries;

    let error =
        validate_layout_entries(&entries, 16).expect_err("a layout declaring one slot twice");
    assert!(error.message().contains("declares slot 2 twice"), "{error}");
}

#[test]
fn a_layout_with_no_slots_is_rejected() {
    let error = validate_layout_entries(&[], 16).expect_err("a layout with no slots is rejected");
    assert!(error.message().contains("at least one entry"), "{error}");
}

#[test]
fn bind_group_entries_canonicalize_to_ascending_slot_order() {
    let entries = vec![
        uniform_class_entry(9),
        uniform_class_entry(1),
        uniform_class_entry(4),
    ];

    assert_eq!(entry_slot_ids(&entries), vec![9, 1, 4]);
    assert_eq!(entry_slot_ids(&sorted(entries.clone())), vec![1, 4, 9]);

    // Two fillings of the same slots in different orders must reach the same
    // canonical order. The comparison is on slot ids rather than on the
    // descriptors: a bind group's entries name live resources, and two entries
    // holding structurally equal but distinct buffers are not the same entry.
    let other_order = vec![
        uniform_class_entry(4),
        uniform_class_entry(9),
        uniform_class_entry(1),
    ];
    assert_ne!(entry_slot_ids(&entries), entry_slot_ids(&other_order));
    assert_eq!(
        entry_slot_ids(&sorted(entries)),
        entry_slot_ids(&sorted(other_order)),
        "the order a caller filled entries in must not reach the canonical form"
    );
}

#[test]
fn a_bind_group_filling_one_slot_twice_is_rejected() {
    let layout = vec![uniform_slot(3)];
    let entries = sorted(vec![uniform_class_entry(3), uniform_class_entry(3)]);

    let error = validate_bind_group_entries(&entries, &layout)
        .expect_err("two entries claiming one slot is a contradiction");
    assert!(error.message().contains("fills slot 3 twice"), "{error}");
}

#[test]
fn a_bind_group_may_not_fill_a_slot_outside_its_layout() {
    let layout = vec![uniform_slot(3)];
    let entries = sorted(vec![uniform_class_entry(4)]);

    let error = validate_bind_group_entries(&entries, &layout)
        .expect_err("an undeclared slot is rejected before any backend sees it");
    assert!(error.message().contains("does not declare"), "{error}");
}

#[test]
fn a_bind_group_may_not_fill_a_slot_with_the_wrong_resource_class() {
    let layout = vec![uniform_slot(3)];
    let entries = sorted(vec![sampler_class_entry(3)]);

    let error = validate_bind_group_entries(&entries, &layout)
        .expect_err("binding a sampler where the layout declares a buffer is rejected");
    assert!(error.message().contains("wrong class"), "{error}");
}

#[test]
fn a_scalar_slot_requires_the_scalar_form_and_the_declared_count() {
    let layout = vec![uniform_slot(3)];
    let entries = sorted(vec![uniform_class_entry(3)]);

    // The class matches and the form is an array, but the slot is scalar.
    let error =
        validate_bind_group_entries(&entries, &layout).expect_err("an array is not a scalar slot");
    assert!(error.message().contains("declares 1"), "{error}");
}

#[test]
fn an_array_slot_requires_the_declared_count() {
    let array_slot = BindingSlot::new(
        BindingSlotId::new(3),
        ShaderStages::FRAGMENT,
        BindingKind::Sampler {
            kind: SamplerKind::Filtering,
        },
    )
    .with_count(BindingCount::Fixed(2));
    let layout = vec![array_slot];
    let entries = sorted(vec![sampler_class_entry(3)]);

    let error =
        validate_bind_group_entries(&entries, &layout).expect_err("an empty array is too short");
    assert!(error.message().contains("declares 2"), "{error}");
}

#[test]
fn a_hole_in_a_layout_is_not_itself_an_error() {
    // Two slots, one filled. The rejection below is about the filler's shape,
    // never about slot 1 having no entry: whether a hole is legal is a property
    // of the consuming pipeline, not of the packet.
    let hole = BindingSlot::new(
        BindingSlotId::new(1),
        ShaderStages::FRAGMENT,
        BindingKind::Sampler {
            kind: SamplerKind::Filtering,
        },
    );
    let layout = vec![uniform_slot(0), hole];
    let entries = sorted(vec![uniform_class_entry(0)]);

    let error = validate_bind_group_entries(&entries, &layout)
        .expect_err("the filler is an array on a scalar slot");
    assert!(error.message().contains("fills slot 0"), "{error}");
    assert!(
        !error.message().contains("slot 1"),
        "the empty slot must not be what the error is about: {error}"
    );
}
