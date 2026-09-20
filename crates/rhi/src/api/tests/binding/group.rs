//! Section 22: the packet.
//!
//! Filling a layout exactly: the right count, the right kind, the right usage,
//! the right device, and the right sample type. `use super::*` brings in the
//! fixtures.

use super::*;
use crate::api::tests::fixture;
use crate::api::tests::mock::bind_group_backend_for_test;

// ---------------------------------------------------------------------------
// Section 22: the packet.
// ---------------------------------------------------------------------------

#[test]
fn a_matching_packet_is_accepted() {
    let (_, group) = one_uniform_slot();
    assert!(check(&group).is_ok());
}

#[test]
fn a_packet_that_fills_one_slot_twice_is_refused() {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    let group = BindGroupDescriptor::new(layout)
        .with_entry(BindGroupEntry::new(
            slot(0),
            range_of(1, 0, 64, BufferUsage::UNIFORM),
        ))
        .with_entry(BindGroupEntry::new(
            slot(0),
            range_of(2, 0, 64, BufferUsage::UNIFORM),
        ));
    assert_kind(check(&group), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_packet_may_not_fill_a_slot_the_layout_does_not_declare() {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    let group = BindGroupDescriptor::new(layout).with_entry(BindGroupEntry::new(
        slot(4),
        range_of(1, 0, 64, BufferUsage::UNIFORM),
    ));
    assert_kind(check(&group), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_resource_from_another_device_is_refused() {
    // Section 3.1: identity is compared in O(1) before any other rule, and there is
    // no implicit migration between devices.
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    let foreign = fixture::buffer(
        object(1),
        identity(2),
        BufferDescriptor::new(256, BufferUsage::UNIFORM),
    );
    let group = group_with(
        layout,
        BindingResource::Buffer(BufferBinding::new(
            foreign,
            BufferRange {
                offset: 0,
                size: 64,
            },
        )),
    );
    assert_kind(check(&group), RhiErrorKind::WrongDevice);
}

#[test]
fn a_scalar_slot_takes_a_scalar_and_an_array_slot_takes_exactly_its_length() {
    // Section 22.1: an array of length 1 cannot stand in for `BindingCount::One`,
    // and a fixed array must be filled exactly.
    let scalar = layout_from(vec![uniform_slot(0, 64)]);
    let one_element_array = group_with(
        scalar,
        BindingResource::BufferArray(vec![BufferBinding::new(
            buffer(1, 256, BufferUsage::UNIFORM),
            BufferRange {
                offset: 0,
                size: 64,
            },
        )]),
    );
    assert_kind(check(&one_element_array), RhiErrorKind::InvalidUsage);

    let two = || {
        let array = layout_from(vec![
            uniform_slot(0, 64)
                .with_count(BindingCount::Fixed(2))
                .with_dynamic_offset(true),
        ]);
        (
            array,
            BindingResource::BufferArray(vec![
                BufferBinding::new(
                    buffer(1, 256, BufferUsage::UNIFORM),
                    BufferRange {
                        offset: 0,
                        size: 64,
                    },
                ),
                BufferBinding::new(
                    buffer(2, 256, BufferUsage::UNIFORM),
                    BufferRange {
                        offset: 64,
                        size: 64,
                    },
                ),
            ]),
        )
    };

    let (array, exact) = two();
    assert!(check(&group_with(array, exact)).is_ok());

    let (array, _) = two();
    let short = group_with(
        array,
        BindingResource::BufferArray(vec![BufferBinding::new(
            buffer(1, 256, BufferUsage::UNIFORM),
            BufferRange {
                offset: 0,
                size: 64,
            },
        )]),
    );
    assert_kind(check(&short), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_buffer_binding_needs_the_matching_usage_bit() {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    let group = group_with(layout, range_of(1, 0, 64, BufferUsage::VERTEX));
    assert_kind(check(&group), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_buffer_binding_must_cover_the_layout_minimum() {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    let group = group_with(layout, range_of(1, 0, 32, BufferUsage::UNIFORM));
    assert_kind(check(&group), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_buffer_binding_is_measured_against_the_device_maximum_and_alignment() {
    let (layout, group) = one_uniform_slot();

    assert_kind(
        validate_bind_group_descriptor(
            &group,
            BindGroupLimits::new(32, 32, 0, 0),
            sample_type_float,
            storage_access_available,
        ),
        RhiErrorKind::InvalidUsage,
    );

    let misaligned = group_with(layout, range_of(1, 8, 64, BufferUsage::UNIFORM));
    assert_kind(
        validate_bind_group_descriptor(
            &misaligned,
            BindGroupLimits::new(1 << 16, 1 << 20, 16, 0),
            sample_type_float,
            storage_access_available,
        ),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_buffer_binding_must_stay_inside_the_buffer() {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    // A 256-byte buffer with a 64-byte range at offset 64 and a declared size of
    // 256: the range itself is legal, but the buffer is only 96 bytes in the
    // second case.
    let oversize = group_with(layout.clone(), range_of(1, 64, 64, BufferUsage::UNIFORM));
    assert!(check(&oversize).is_ok());

    let outside = group_with(
        layout,
        BindingResource::Buffer(BufferBinding::new(
            buffer(1, 96, BufferUsage::UNIFORM),
            BufferRange {
                offset: 64,
                size: 64,
            },
        )),
    );
    assert_kind(check(&outside), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_sampled_texture_must_match_the_declared_dimension() {
    let layout = layout_from(vec![texture_slot(
        TextureViewDimension::Cube,
        TextureSampleType::Float,
    )]);
    let group = group_with(
        layout,
        BindingResource::Texture(sampled_view(
            2,
            TextureUsage::SAMPLED,
            TextureFormat::Rgba8Unorm,
        )),
    );
    assert_kind(check(&group), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_sampled_texture_must_match_the_declared_sample_type() {
    // The rule that keeps reflection and layout from disagreeing about what the
    // shader reads: the view's format must sample as the declared type.
    let layout = layout_from(vec![texture_slot(
        TextureViewDimension::D2,
        TextureSampleType::Float,
    )]);
    let group = group_with(
        layout.clone(),
        BindingResource::Texture(sampled_view(
            2,
            TextureUsage::SAMPLED,
            TextureFormat::Rgba8Unorm,
        )),
    );
    assert!(check(&group).is_ok());

    assert_kind(
        validate_bind_group_descriptor(
            &group,
            generous_limits(),
            |_| Some(TextureSampleType::Uint),
            storage_access_available,
        ),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_sampled_texture_binding_needs_the_sampled_usage_bit() {
    let layout = layout_from(vec![texture_slot(
        TextureViewDimension::D2,
        TextureSampleType::Float,
    )]);
    let group = group_with(
        layout,
        BindingResource::Texture(sampled_view(
            2,
            TextureUsage::COPY_SRC,
            TextureFormat::Rgba8Unorm,
        )),
    );
    assert_kind(check(&group), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_storage_texture_the_device_cannot_access_is_unsupported() {
    // A layout problem is the caller's mistake; a device that cannot express the
    // access is not, which is why the two kinds differ here.
    let layout = layout_from(vec![BindingSlot::new(
        slot(0),
        ShaderStages::COMPUTE,
        BindingKind::StorageTexture {
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            access: StorageAccess::ReadWrite,
        },
    )]);
    let group = group_with(
        layout,
        BindingResource::Texture(sampled_view(
            2,
            TextureUsage::STORAGE,
            TextureFormat::Rgba8Unorm,
        )),
    );
    assert_kind(
        validate_bind_group_descriptor(&group, generous_limits(), sample_type_float, |_, _| false),
        RhiErrorKind::Unsupported,
    );
    assert!(check(&group).is_ok());
}

#[test]
fn a_comparison_sampler_declaration_must_agree_with_the_descriptor() {
    // Section 22.3's comparison half: the shader's expectation and the descriptor's
    // comparison function must agree, because a sampler that compares cannot be
    // bound where the shader samples it.
    let with_kind = |kind| {
        layout_from(vec![BindingSlot::new(
            slot(0),
            ShaderStages::FRAGMENT,
            BindingKind::Sampler { kind },
        )])
    };

    let missing_comparison = group_with(
        with_kind(SamplerKind::Comparison),
        BindingResource::Sampler(sampler(3, None)),
    );
    assert_kind(check(&missing_comparison), RhiErrorKind::InvalidUsage);

    let unexpected_comparison = group_with(
        with_kind(SamplerKind::Filtering),
        BindingResource::Sampler(sampler(3, Some(CompareFunction::Less))),
    );
    assert_kind(check(&unexpected_comparison), RhiErrorKind::InvalidUsage);

    let matching = group_with(
        with_kind(SamplerKind::Comparison),
        BindingResource::Sampler(sampler(3, Some(CompareFunction::Less))),
    );
    assert!(check(&matching).is_ok());
}

#[test]
fn a_resource_whose_variant_does_not_match_the_slot_kind_is_refused() {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    let group = group_with(layout, BindingResource::Sampler(sampler(3, None)));
    assert_kind(check(&group), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_storage_buffer_binding_is_measured_against_the_storage_limits() {
    let layout = layout_from(vec![BindingSlot::new(
        slot(0),
        ShaderStages::COMPUTE,
        BindingKind::StorageBuffer {
            access: BufferBindingAccess::ReadWrite,
            min_size: 64,
        },
    )]);
    let group = group_with(layout, range_of(1, 0, 64, BufferUsage::STORAGE));
    assert!(check(&group).is_ok());
}

// ---------------------------------------------------------------------------
// Section 22.2: the created group.
// ---------------------------------------------------------------------------

#[test]
fn a_group_reports_its_layout_and_canonical_entries() {
    let (layout, descriptor) = one_uniform_slot();
    // Canonicalized once and handed to both halves, which is what the real
    // creation verb does: the packet the backend lowers is the packet the handle
    // describes, and a test that canonicalized twice would be testing two
    // packets that happen to be equal.
    let canonical = descriptor.canonicalized();
    let group = BindGroup::new(
        object(20),
        device(),
        canonical.clone(),
        bind_group_backend_for_test(canonical),
    );

    assert_eq!(group.id(), object(20));
    assert_eq!(group.device_identity(), device());
    assert_eq!(group.layout().id(), layout.id());
    assert_eq!(group.descriptor().entries.len(), 1);
    assert_eq!(group.descriptor().entries[0].slot, slot(0));
}

#[test]
fn a_group_debug_prints_portable_identity_only() {
    // Defect D6 of the 0.16 series: the specification declares `#[derive(Clone)]`
    // and no `Debug` on the handle, while descriptors that contain a layout or a
    // group do derive it. The handle implements `Debug` by hand and prints identity
    // only, so the native field the backend port adds is never printed into a log.
    let (_, descriptor) = one_uniform_slot();
    let canonical = descriptor.canonicalized();
    let group = BindGroup::new(
        object(21),
        device(),
        canonical.clone(),
        bind_group_backend_for_test(canonical),
    );
    let text = format!("{group:?}");
    assert!(text.contains("BindGroup"), "{text}");
    assert!(text.contains("id"), "{text}");
}
