//! Sections 20.5 and 21.2: layout validation and canonicalization.
//!
//! The per-slot rule (a slot declared once, by what it is visible to, with a
//! dynamic offset only on a buffer) and the aggregate rules the layout validator
//! states. `use super::*` brings in the fixtures.

use super::*;

// ---------------------------------------------------------------------------
// Section 20.5 / 21.2: layout validation and canonicalization.
// ---------------------------------------------------------------------------

#[test]
fn a_canonical_layout_is_accepted() {
    let layout = layout_from(vec![uniform_slot(0, 64), uniform_slot(1, 128)]);
    assert!(validate_bind_group_layout_descriptor(layout.descriptor(), 8, permissive).is_ok());
}

#[test]
fn a_binding_visible_to_several_stages_is_one_binding() {
    // Section 20.2's reason for a stage *set*: a vertex-and-fragment binding is one
    // slot with two stage bits, not two slots with the same data.
    let shared = BindingSlot::new(
        slot(0),
        ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
        BindingKind::UniformBuffer { min_size: 64 },
    );
    let layout = layout_from(vec![shared]);
    assert!(validate_bind_group_layout_descriptor(layout.descriptor(), 8, permissive).is_ok());
    assert!(
        layout
            .slot(slot(0))
            .expect("the slot is declared")
            .visibility
            .contains(ShaderStages::VERTEX.union(ShaderStages::FRAGMENT))
    );
}

#[test]
fn a_layout_may_not_declare_one_slot_twice() {
    let layout = layout_from(vec![uniform_slot(0, 64), uniform_slot(0, 128)]);
    assert_kind(
        validate_bind_group_layout_descriptor(layout.descriptor(), 8, permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_dynamic_offset_is_refused_on_a_non_buffer_binding() {
    // Section 20.5: there is nothing in a texture or sampler binding to offset, so
    // the layout refuses it rather than leaving a backend to decide.
    let texture =
        texture_slot(TextureViewDimension::D2, TextureSampleType::Float).with_dynamic_offset(true);
    let layout = layout_from(vec![texture]);
    assert_kind(
        validate_bind_group_layout_descriptor(layout.descriptor(), 8, permissive),
        RhiErrorKind::InvalidUsage,
    );

    let buffer = uniform_slot(0, 64).with_dynamic_offset(true);
    let layout = layout_from(vec![buffer]);
    assert!(validate_bind_group_layout_descriptor(layout.descriptor(), 8, permissive).is_ok());
}

#[test]
fn a_layout_over_the_device_binding_ceiling_is_refused() {
    let layout = layout_from(vec![uniform_slot(0, 64), uniform_slot(1, 64)]);
    assert_kind(
        validate_bind_group_layout_descriptor(layout.descriptor(), 1, permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_binding_the_device_cannot_express_refuses_the_layout() {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    assert_kind(
        validate_bind_group_layout_descriptor(layout.descriptor(), 8, |_| {
            BindingSupport::Unsupported
        }),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn layout_entries_are_canonical_and_dynamic_offsets_count_elements() {
    // Section 21.2's canonical form is ascending slot order whatever order the
    // caller typed, and section 21.3 counts a `Fixed(n)` binding with a dynamic
    // offset as `n` offsets rather than one.
    let layout = BindGroupLayout::new(
        object(11),
        device(),
        BindGroupLayoutDescriptor::new(vec![
            uniform_slot(2, 64)
                .with_count(BindingCount::Fixed(3))
                .with_dynamic_offset(true),
            uniform_slot(0, 64),
        ])
        .canonicalized(),
        BindGroupLayoutCompatibilityId::new(2),
        LayoutFingerprint([9; 32]),
    );

    let declared: Vec<u32> = layout
        .descriptor()
        .entries
        .iter()
        .map(|entry| entry.slot.get())
        .collect();
    assert_eq!(
        declared,
        vec![0, 2],
        "entries must be stored in ascending slot order"
    );
    assert_eq!(layout.dynamic_offset_count(), 3);
    assert_eq!(layout.fingerprint(), LayoutFingerprint([9; 32]));
    assert_eq!(layout.compatibility_id().get(), 2);
    assert_eq!(layout.device_identity(), device());
    assert_eq!(layout.id(), object(11));
}
