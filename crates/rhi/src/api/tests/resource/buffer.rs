//! Sections 11.1, 11.2, 12.2, 12.3 and 12.4: usage flags, the buffer descriptor and its builders, buffer creation and range rules.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::identity::Label;
use crate::api::resource::buffer::{
    Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferSupport, BufferSupportLimits,
    BufferUsage, ResourceMemoryPreference, validate_buffer_descriptor, validate_buffer_ownership,
    validate_buffer_range,
};
use crate::api::resource::texture::{
    Extent3d, TextureDescriptor, TextureDimension, TextureUsage, TextureViewCompatibility,
    mip_ceiling,
};

#[test]
fn buffer_usage_bits_are_distinct_and_compose() {
    // The six constants must be six different bits: a bitset whose members
    // overlapped would make `contains` answer questions about the wrong
    // operation.
    let all = [
        BufferUsage::COPY_SRC,
        BufferUsage::COPY_DST,
        BufferUsage::VERTEX,
        BufferUsage::INDEX,
        BufferUsage::UNIFORM,
        BufferUsage::STORAGE,
    ];
    for (index, first) in all.iter().enumerate() {
        for second in all.iter().skip(index + 1) {
            assert_ne!(first, second, "{first} and {second} are the same set");
            assert!(
                !first.contains(*second),
                "{first} must not contain {second}"
            );
        }
        assert!(!first.is_empty());
    }

    let vertex_index = BufferUsage::VERTEX.union(BufferUsage::INDEX);
    assert!(vertex_index.contains(BufferUsage::VERTEX));
    assert!(vertex_index.contains(BufferUsage::INDEX));
    assert!(!vertex_index.contains(BufferUsage::STORAGE));
    assert!(!vertex_index.is_empty());

    // `contains` is a subset test, so the empty set — if it were reachable —
    // would be contained in everything. That is why "usage must not be empty" is
    // a descriptor rule rather than something `contains` can express.
    assert!(vertex_index.contains(BufferUsage::VERTEX.union(BufferUsage::VERTEX)));
}

#[test]
fn texture_usage_bits_are_distinct_and_compose() {
    let all = [
        TextureUsage::COPY_SRC,
        TextureUsage::COPY_DST,
        TextureUsage::SAMPLED,
        TextureUsage::STORAGE,
        TextureUsage::COLOR_ATTACHMENT,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ];
    for (index, first) in all.iter().enumerate() {
        for second in all.iter().skip(index + 1) {
            assert!(
                !first.contains(*second),
                "{first} must not contain {second}"
            );
        }
        assert!(!first.is_empty());
    }

    let sampled_copy = TextureUsage::SAMPLED.union(TextureUsage::COPY_DST);
    assert!(sampled_copy.contains(TextureUsage::SAMPLED));
    assert!(sampled_copy.contains(TextureUsage::COPY_DST));
    assert!(!sampled_copy.contains(TextureUsage::STORAGE));
}

#[test]
fn a_texture_usage_set_is_never_empty_either() {
    // The texture counterpart of the buffer algebra test above, and for the same
    // reason: section 13.4 refuses an empty usage set, and section 11.1's bitset
    // gives a caller no way to build one.
    let union = TextureUsage::SAMPLED.union(TextureUsage::SAMPLED);
    assert!(union.contains(TextureUsage::SAMPLED));
    assert!(!union.is_empty());
    assert!(
        !TextureUsage::COPY_SRC
            .union(TextureUsage::COPY_DST)
            .is_empty()
    );
}

// ---------------------------------------------------------------------------
// Section 11.2 and 12.2: the descriptor and its builders.
// ---------------------------------------------------------------------------

#[test]
fn a_buffer_descriptor_round_trips_through_its_own_accessors() {
    let descriptor = BufferDescriptor::new(256, BufferUsage::VERTEX.union(BufferUsage::COPY_DST))
        .with_label("vertices")
        .with_memory_preference(ResourceMemoryPreference::DeviceLocalPreferred);

    assert_eq!(descriptor.size, 256);
    assert_eq!(
        descriptor.usage,
        BufferUsage::VERTEX.union(BufferUsage::COPY_DST)
    );
    assert_eq!(descriptor.label.as_deref(), Some("vertices"));
    assert_eq!(
        descriptor.memory,
        ResourceMemoryPreference::DeviceLocalPreferred
    );
}

#[test]
fn a_buffer_descriptor_starts_automatic_and_unlabelled() {
    let descriptor = BufferDescriptor::new(64, BufferUsage::UNIFORM);
    assert_eq!(descriptor.label, Label::default());
    assert_eq!(descriptor.memory, ResourceMemoryPreference::Automatic);
}

#[test]
fn a_texture_descriptor_round_trips_through_its_own_accessors() {
    let descriptor =
        TextureDescriptor::new_2d(256, 128, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
            .with_label("gbuffer")
            .with_mip_levels(8)
            .with_array_layers(4)
            .with_sample_count(1)
            .with_view_format(TextureFormat::Rgba8UnormSrgb)
            .with_view_compatibility(TextureViewCompatibility::CUBE)
            .with_memory_preference(ResourceMemoryPreference::DeviceLocalPreferred);

    assert_eq!(descriptor.dimension, TextureDimension::D2);
    assert_eq!(descriptor.extent, Extent3d::d2(256, 128));
    assert_eq!(descriptor.mip_levels, 8);
    assert_eq!(descriptor.array_layers, 4);
    assert_eq!(descriptor.sample_count, 1);
    assert_eq!(descriptor.format, TextureFormat::Rgba8Unorm);
    assert_eq!(descriptor.usage, TextureUsage::SAMPLED);
    assert_eq!(descriptor.view_formats, vec![TextureFormat::Rgba8UnormSrgb]);
    assert_eq!(
        descriptor.view_compatibility,
        TextureViewCompatibility::CUBE
    );
    assert_eq!(
        descriptor.memory,
        ResourceMemoryPreference::DeviceLocalPreferred
    );
}

#[test]
fn the_three_constructors_produce_the_three_dimensions() {
    let one = TextureDescriptor::new_1d(64, TextureFormat::R8Unorm, TextureUsage::SAMPLED);
    assert_eq!(one.dimension, TextureDimension::D1);
    assert_eq!(one.extent, Extent3d::d1(64));
    assert_eq!(one.extent.height, 1);
    assert_eq!(one.extent.depth, 1);

    let two = TextureDescriptor::new_2d(64, 32, TextureFormat::R8Unorm, TextureUsage::SAMPLED);
    assert_eq!(two.dimension, TextureDimension::D2);
    assert_eq!(two.extent.depth, 1);

    let three =
        TextureDescriptor::new_3d(64, 32, 16, TextureFormat::R8Unorm, TextureUsage::SAMPLED);
    assert_eq!(three.dimension, TextureDimension::D3);
    assert_eq!(three.extent, Extent3d::d3(64, 32, 16));

    // All three start at the minimum the invariants permit, so a descriptor
    // built by a constructor validates as soon as its extent does.
    for descriptor in [one, two, three] {
        assert_eq!(descriptor.mip_levels, 1);
        assert_eq!(descriptor.array_layers, 1);
        assert_eq!(descriptor.sample_count, 1);
        assert!(descriptor.view_formats.is_empty());
        assert_eq!(
            descriptor.view_compatibility,
            TextureViewCompatibility::NONE
        );
    }
}

#[test]
fn the_view_format_set_is_canonical_at_every_step() {
    // Section 13.1 makes `view_formats` a *set*: sorted, without duplicates. If
    // the invariant only held at creation, a descriptor would show a caller one
    // list and later feed a different one to the capability query — and two
    // spellings of one question would compare unequal as cache keys.
    let descriptor = simple_texture_descriptor(TextureUsage::SAMPLED)
        .with_view_format(TextureFormat::Bgra8Unorm)
        .with_view_format(TextureFormat::Rgba8UnormSrgb)
        .with_view_format(TextureFormat::Bgra8Unorm);

    assert_eq!(descriptor.view_formats.len(), 2);
    let mut sorted = descriptor.view_formats.clone();
    sorted.sort_unstable_by_key(|format| *format as u32);
    assert_eq!(descriptor.view_formats, sorted);
}

#[test]
fn the_mip_ceiling_is_floor_log2_plus_one() {
    // Section 13.1's arithmetic, at the boundaries where an off-by-one lives:
    // exactly at each power of two, and one past it.
    assert_eq!(mip_ceiling(Extent3d::d2(1, 1)), 1);
    assert_eq!(mip_ceiling(Extent3d::d2(2, 1)), 2);
    assert_eq!(mip_ceiling(Extent3d::d2(3, 1)), 2);
    assert_eq!(mip_ceiling(Extent3d::d2(4, 1)), 3);
    assert_eq!(mip_ceiling(Extent3d::d2(7, 1)), 3);
    assert_eq!(mip_ceiling(Extent3d::d2(8, 1)), 4);
    assert_eq!(mip_ceiling(Extent3d::d2(1024, 1024)), 11);
    // The ceiling is over the largest component, not the width.
    assert_eq!(mip_ceiling(Extent3d::d3(1, 1, 8)), 4);
}

// ---------------------------------------------------------------------------
// Sections 12.3 and 12.4: buffer creation and range rules.
// ---------------------------------------------------------------------------

#[test]
fn a_legal_buffer_descriptor_is_accepted() {
    let descriptor = BufferDescriptor::new(1024, BufferUsage::VERTEX);
    assert!(validate_buffer_descriptor(&descriptor, &generous_buffer_support()).is_ok());

    // A labelled descriptor with a placement preference is equally legal: the
    // label and the preference are not validation inputs.
    let preferred = BufferDescriptor::new(1024, BufferUsage::VERTEX)
        .with_label("vertices")
        .with_memory_preference(ResourceMemoryPreference::DeviceLocalPreferred);
    assert!(validate_buffer_descriptor(&preferred, &generous_buffer_support()).is_ok());
}

#[test]
fn a_buffer_of_zero_bytes_is_refused() {
    // Section 12.3's first rule, and the one it calls out explicitly: "If the
    // Buffer size is 0, the P0 portable contract will not be entered."
    let descriptor = BufferDescriptor::new(0, BufferUsage::VERTEX);
    assert_kind(
        validate_buffer_descriptor(&descriptor, &generous_buffer_support()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_usage_set_is_never_empty_and_union_is_idempotent() {
    // Section 12.3 refuses an empty usage set, and section 11.1's hand-rolled
    // bitset has a private field, no `Default`, and no `EMPTY` constant — so no
    // expression a caller can write produces the empty set, and the refusing
    // branch of `validate_buffer_descriptor` is unreachable from outside this
    // module. That is root section 4's preference (make the illegal state
    // unconstructible) working as intended, and it is reported in the series
    // audit as a rule that has no downloadable test.
    //
    // What *is* testable is the algebra the rule rests on: a union never clears a
    // bit, so the result of every reachable union is non-empty.
    let union = BufferUsage::VERTEX.union(BufferUsage::VERTEX);
    assert!(union.contains(BufferUsage::VERTEX));
    assert!(!union.is_empty());

    let mixed = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);
    assert!(!mixed.is_empty());
    assert!(mixed.contains(BufferUsage::COPY_SRC));
    assert!(mixed.contains(BufferUsage::COPY_DST));
    assert!(!mixed.contains(BufferUsage::STORAGE));

    // The descriptor built from such a set validates, which is the assertion
    // that keeps the two facts above connected to creation.
    let descriptor = BufferDescriptor {
        label: Label::default(),
        size: 16,
        usage: union,
        memory: ResourceMemoryPreference::Automatic,
    };
    assert!(validate_buffer_descriptor(&descriptor, &generous_buffer_support()).is_ok());
}

#[test]
fn a_buffer_usage_the_device_cannot_express_is_unsupported() {
    // Section 12.1's own example: WebGL2 has vertex, index, uniform, and copy
    // buffers but no storage buffers, and the answer must arrive here rather
    // than at a bind group later.
    let descriptor = BufferDescriptor::new(64, BufferUsage::STORAGE);
    assert_kind(
        validate_buffer_descriptor(&descriptor, &BufferSupport::Unsupported),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn a_buffer_past_the_devices_ceiling_is_refused() {
    // Section 12.3's fourth rule, on both sides of the boundary.
    let support = BufferSupport::Supported(BufferSupportLimits::new(4096));

    assert!(
        validate_buffer_descriptor(&BufferDescriptor::new(4096, BufferUsage::VERTEX), &support)
            .is_ok()
    );
    assert_kind(
        validate_buffer_descriptor(&BufferDescriptor::new(4097, BufferUsage::VERTEX), &support),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_buffer_range_end_reports_overflow_instead_of_wrapping() {
    // Section 12.4 requires every point of use to check that `offset + size`
    // does not overflow. An accessor that wrapped would let a caller compare a
    // wrapped end against a buffer size and conclude the range was fine, so the
    // arithmetic is surfaced as an `Option` and tested here at the exact
    // boundary.
    assert_eq!(BufferRange::new(0, 0).end(), Some(0));
    assert_eq!(BufferRange::new(10, 5).end(), Some(15));
    assert_eq!(BufferRange::new(u64::MAX, 0).end(), Some(u64::MAX));
    assert_eq!(BufferRange::new(u64::MAX, 1).end(), None);
    assert_eq!(BufferRange::new(u64::MAX - 1, 2).end(), None);
    assert_eq!(BufferRange::new(u64::MAX - 1, 1).end(), Some(u64::MAX));
}

#[test]
fn a_buffer_range_is_checked_against_the_buffer_it_resolves_against() {
    // Both sides of the boundary: ending exactly at the end is legal, ending one
    // byte later is not.
    assert!(validate_buffer_range(BufferRange::new(0, 64), 64).is_ok());
    assert!(validate_buffer_range(BufferRange::new(32, 32), 64).is_ok());
    assert_kind(
        validate_buffer_range(BufferRange::new(32, 33), 64),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_range_of_zero_bytes_is_refused_wherever_it_sits() {
    // Including at exactly the buffer's end, where the end comparison alone
    // would accept it: that is what proves the empty-range rule is checked
    // first rather than falling out of the range arithmetic.
    assert_kind(
        validate_buffer_range(BufferRange::new(0, 0), 64),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_buffer_range(BufferRange::new(64, 0), 64),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_buffer_range(BufferRange::new(0, 0), 0),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn an_overlapping_range_is_refused_before_it_can_wrap() {
    // The explicit overflow case, reaching the rule rather than the end
    // comparison: the end cannot be computed at all.
    assert_kind(
        validate_buffer_range(BufferRange::new(u64::MAX, 8), u64::MAX),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_buffer_from_another_device_is_wrong_device() {
    // Section 3.3: the only answer for cross-device use in P0, because there is
    // no implicit copy, binding, handle unwrap, staging bridge, or peer transfer
    // to fall back on.
    let buffer = buffer_with(BufferUsage::COPY_SRC, 64);
    assert!(validate_buffer_ownership(&buffer, device()).is_ok());
    assert_kind(
        validate_buffer_ownership(&buffer, identity(1, 2)),
        RhiErrorKind::WrongDevice,
    );
    assert_kind(
        validate_buffer_ownership(&buffer, identity(2, 1)),
        RhiErrorKind::WrongDevice,
    );
}

#[test]
fn a_buffer_reports_its_own_id_device_and_descriptor() {
    let descriptor = BufferDescriptor::new(128, BufferUsage::INDEX).with_label("indices");
    let buffer = Buffer::new(object(7), identity(3, 4), descriptor.clone());

    assert_eq!(buffer.id(), object(7));
    assert_eq!(buffer.device_identity(), identity(3, 4));
    assert_eq!(buffer.descriptor().size, 128);
    assert_eq!(buffer.descriptor().label.as_deref(), Some("indices"));

    // A clone is the same logical object, not a second one.
    let clone = buffer.clone();
    assert_eq!(clone.id(), buffer.id());
    assert_eq!(clone.device_identity(), buffer.device_identity());
}

#[test]
fn a_binding_carries_the_buffer_and_the_range_together() {
    let buffer = buffer_with(BufferUsage::UNIFORM, 256);
    let range = BufferRange::new(16, 64);
    let binding = BufferBinding::new(buffer.clone(), range);

    assert_eq!(binding.range, range);
    assert_eq!(binding.buffer.id(), buffer.id());
    assert_eq!(binding.buffer.descriptor().size, 256);
}
