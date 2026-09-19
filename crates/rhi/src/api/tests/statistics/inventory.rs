//! Live-inventory and memory-estimate tests (specification 47.14–47.18).
//!
//! This is the one part of section 47 whose arithmetic is fully specified and
//! fully testable without a device, so it is tested exhaustively rather than
//! reviewed as a shape. Section 47.17 gives a closed formula over a descriptor
//! and a format's facts, and the facts are a parameter rather than a device
//! query — which is exactly the seam that lets a test drive every branch,
//! including the two that produce [`MemoryEstimate::unknown`].
//!
//! The test names record the expected byte counts as literals rather than as
//! recomputed expressions. That is deliberate: a test that re-derived the number
//! the implementation derives would pass for any implementation, and the point of
//! these is to be an independent statement of what the specification says the
//! number is.

use crate::api::format::{FormatFacts, StorageAccessSupport, TextureFormat};
use crate::api::resource::texture::{TextureDescriptor, TextureUsage};
use crate::api::statistics::inventory::estimate_texture_bytes;
use crate::api::statistics::{
    DeviceStatistics, InventoryStatistics, LiveObjectCounts, MemoryEstimate, MemoryEstimateQuality,
    ResourceMemoryStatistics,
};

use super::{device, other_device};

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

/// Facts with every storage access available.
///
/// Storage access is irrelevant to the estimate — section 47.17 reads only
/// `logical_bytes_per_block`, `block_width`, and `block_height` — but a
/// `FormatFacts` cannot be assembled without it, because it is a probed device
/// fact and the type has no partial constructor.
fn facts(format: TextureFormat) -> FormatFacts {
    FormatFacts::new(format, StorageAccessSupport::new(true, true, true))
}

/// A descriptor with every dimension given explicitly.
fn texture(
    width: u32,
    height: u32,
    depth: u32,
    format: TextureFormat,
    mip_levels: u32,
    array_layers: u32,
    sample_count: u32,
) -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::new_3d(width, height, depth, format, TextureUsage::COLOR_ATTACHMENT);
    descriptor.mip_levels = mip_levels;
    descriptor.array_layers = array_layers;
    descriptor.sample_count = sample_count;
    descriptor
}

/// The byte count the estimate gives, or `None` when it has none.
fn bytes(descriptor: &TextureDescriptor, format: TextureFormat) -> Option<u64> {
    estimate_texture_bytes(descriptor, &facts(format)).logical_estimated_bytes
}

// ---------------------------------------------------------------------------
// Section 47.15 — the estimate's two states.
// ---------------------------------------------------------------------------

/// `Unknown` is not a failure and not an error: it is an answer, and the field
/// that follows it is `None` rather than zero.
#[test]
fn an_unknown_estimate_carries_no_number_and_a_different_quality() {
    let unknown = MemoryEstimate::unknown();

    assert_eq!(unknown.logical_estimated_bytes, None);
    assert_eq!(unknown.quality, MemoryEstimateQuality::Unknown);
    assert_ne!(unknown.quality, MemoryEstimateQuality::LogicalEstimate);
    assert_ne!(unknown.quality, MemoryEstimate::logical(0).quality);
}

/// Section 47.15 declares `Default` without saying which value it is, and
/// `Unknown` is the only honest one.
///
/// `ResourceMemoryStatistics` derives `Default`, so the other candidate would be
/// a false statement baked into a total: `logical(0)` claims a resource of
/// exactly zero bytes *with the quality of a real estimate*, so a defaulted
/// inventory would assert that an unmeasured frame occupies nothing.
#[test]
fn the_default_estimate_claims_nothing_was_estimated() {
    assert_eq!(
        MemoryEstimate::default().quality,
        MemoryEstimateQuality::Unknown
    );
    assert_eq!(MemoryEstimate::default().logical_estimated_bytes, None);

    // Which is what makes a defaulted inventory honest rather than merely empty.
    // The comparison is field by field because section 47.15 gives
    // `MemoryEstimate` no `PartialEq` — a caller compares the two fields it cares
    // about, and the quality asymmetry is visible while it does.
    let defaulted = ResourceMemoryStatistics::default();
    assert_eq!(
        defaulted.buffers.logical_estimated_bytes,
        MemoryEstimate::default().logical_estimated_bytes
    );
    assert_eq!(defaulted.buffers.quality, MemoryEstimate::default().quality);
    assert_eq!(defaulted.total_resources.logical_estimated_bytes, None);
}

/// The estimate is a named field carrying a number and its quality side by side,
/// because a caller deciding whether to warn about memory wants both and a caller
/// summing five estimates has to know whether one was `Unknown` before it can say
/// anything about the total.
#[test]
fn a_known_estimate_pairs_its_number_with_its_quality() {
    let known = MemoryEstimate::logical(1 << 20);

    assert_eq!(known.logical_estimated_bytes, Some(1 << 20));
    assert_eq!(known.quality, MemoryEstimateQuality::LogicalEstimate);

    // Zero is a real answer and is spelled as one, which is why it cannot double
    // as the unknown state.
    let zero = MemoryEstimate::logical(0);
    assert_eq!(zero.logical_estimated_bytes, Some(0));
    assert_ne!(zero.quality, MemoryEstimateQuality::Unknown);
}

// ---------------------------------------------------------------------------
// Section 47.17 — the texture estimate.
// ---------------------------------------------------------------------------

/// One texel per block in P0, so the estimate of a single-mip texture is its
/// texel count times the format's bytes per texel.
#[test]
fn a_single_mip_texture_estimates_its_texel_bytes() {
    // 64 * 64 * 4 bytes.
    let rgba = texture(64, 64, 1, TextureFormat::Rgba8Unorm, 1, 1, 1);
    assert_eq!(bytes(&rgba, TextureFormat::Rgba8Unorm), Some(16_384));

    // 4 * 4 * 1 byte.
    let r8 = texture(4, 4, 1, TextureFormat::R8Unorm, 1, 1, 1);
    assert_eq!(bytes(&r8, TextureFormat::R8Unorm), Some(16));

    // A 16-byte format, to show the table is read and not assumed.
    let rgba32 = texture(2, 2, 1, TextureFormat::Rgba32Float, 1, 1, 1);
    assert_eq!(bytes(&rgba32, TextureFormat::Rgba32Float), Some(64));
}

/// Every mip level is summed, and `max(1, dim >> mip)` is what makes the chain
/// terminate at one texel instead of at zero.
#[test]
fn a_mip_chain_sums_every_level_and_stops_at_one_texel() {
    // 4x4, 2x2, 1x1 as R8: 16 + 4 + 1.
    let power_of_two = texture(4, 4, 1, TextureFormat::R8Unorm, 3, 1, 1);
    assert_eq!(bytes(&power_of_two, TextureFormat::R8Unorm), Some(21));

    // The same chain on a non-power-of-two extent, which is where a naive
    // `width >> mip` would produce a zero-sized level: 5x5, max(1, 2)xmax(1, 2),
    // 1x1  ->  25 + 4 + 1.
    let odd = texture(5, 5, 1, TextureFormat::R8Unorm, 3, 1, 1);
    assert_eq!(bytes(&odd, TextureFormat::R8Unorm), Some(30));

    // One mip is one level, not a chain: the formula must not add levels the
    // descriptor did not ask for.
    let single = texture(4, 4, 1, TextureFormat::R8Unorm, 1, 1, 1);
    assert_eq!(bytes(&single, TextureFormat::R8Unorm), Some(16));
}

/// A `mip_levels` larger than the shift width is not a panic and not a zero: the
/// specification's arithmetic shifts to zero, and its rule then raises that to
/// one.
///
/// This is reachable because `mip_levels` is a `u32` on a public descriptor and
/// the estimate is a device-free function: a caller can hand it any number, and a
/// shift past the width of a `u32` is where a naive implementation panics in a
/// debug build. The count here is 40 levels on a 2x2 texture — 4 + 1, then
/// thirty-eight more levels of one texel each.
#[test]
fn a_mip_chain_longer_than_the_extent_terminates_at_one_texel_per_level() {
    let overlong = texture(2, 2, 1, TextureFormat::R8Unorm, 40, 1, 1);

    assert_eq!(bytes(&overlong, TextureFormat::R8Unorm), Some(4 + 1 + 38));
}

/// Array layers and samples are both multipliers, and they multiply the whole
/// chain rather than one level of it.
#[test]
fn array_layers_and_samples_multiply_the_whole_chain() {
    let layered = texture(4, 4, 1, TextureFormat::R8Unorm, 1, 2, 1);
    assert_eq!(bytes(&layered, TextureFormat::R8Unorm), Some(32));

    let multisampled = texture(4, 4, 1, TextureFormat::R8Unorm, 1, 1, 4);
    assert_eq!(bytes(&multisampled, TextureFormat::R8Unorm), Some(64));

    // Both at once, on a three-level chain: (16 + 4 + 1) * 2 * 4.
    let both = texture(4, 4, 1, TextureFormat::R8Unorm, 3, 2, 4);
    assert_eq!(bytes(&both, TextureFormat::R8Unorm), Some(21 * 8));
}

/// Depth is a third dimension and appears once per level, not once per layer:
/// section 14.2 keeps a 3D texture's Z range out of its array layers, and the
/// estimate follows the same shape.
#[test]
fn a_threed_texture_multiplies_by_its_depth() {
    let volume = texture(4, 4, 4, TextureFormat::R8Unorm, 1, 1, 1);
    assert_eq!(bytes(&volume, TextureFormat::R8Unorm), Some(64));

    // A mip chain halves the depth as well: 4x4x4, 2x2x2, 1x1x1.
    let chained = texture(4, 4, 4, TextureFormat::R8Unorm, 3, 1, 1);
    assert_eq!(bytes(&chained, TextureFormat::R8Unorm), Some(64 + 8 + 1));
}

/// The formats whose backing the format name does not fix have no estimate, and
/// that is an answer rather than a failure to compute one.
///
/// Section 47.17 names `Depth24Plus`: the backend chooses the backing, and a
/// driver may store `Depth32FloatStencil8` as one packed 5-byte texel or as a
/// depth plane plus a separate stencil plane. Returning a number for them would
/// turn a settled estimate into a guess a caller cannot tell apart from a
/// measurement.
#[test]
fn a_format_whose_backing_is_implementation_defined_has_no_estimate() {
    for format in [
        TextureFormat::Depth24Plus,
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ] {
        let descriptor = texture(64, 64, 1, format, 1, 1, 1);

        let estimate = estimate_texture_bytes(&descriptor, &facts(format));
        assert_eq!(
            estimate.quality,
            MemoryEstimateQuality::Unknown,
            "{format:?} should not have a fixed backing"
        );
        assert_eq!(estimate.logical_estimated_bytes, None, "{format:?}");
    }

    // The neighbouring formats that *do* name a layout still estimate, which is
    // what makes the rule about the format name rather than about depth formats.
    let depth16 = texture(64, 64, 1, TextureFormat::Depth16Unorm, 1, 1, 1);
    assert_eq!(
        bytes(&depth16, TextureFormat::Depth16Unorm),
        Some(64 * 64 * 2)
    );

    let depth32 = texture(16, 16, 1, TextureFormat::Depth32Float, 1, 1, 1);
    assert_eq!(
        bytes(&depth32, TextureFormat::Depth32Float),
        Some(16 * 16 * 4)
    );
}

/// Overflow anywhere in the chain yields `Unknown`, never a wrapped total.
///
/// A wrapped number would be a plausible-looking byte count — the worst possible
/// answer, because nothing downstream could detect it. The extremes here are
/// reachable from public fields: `Extent3d` is three `u32`s, so a cube of
/// `u32::MAX` is a descriptor a caller can write.
#[test]
fn an_overflowing_estimate_is_unknown_rather_than_wrapped() {
    let enormous = texture(
        u32::MAX,
        u32::MAX,
        u32::MAX,
        TextureFormat::R8Unorm,
        1,
        1,
        1,
    );
    assert_eq!(bytes(&enormous, TextureFormat::R8Unorm), None);

    // Layers overflow as readily as extent does: 65536 * 65536 * 16 bytes is
    // 2^36, and multiplying by `u32::MAX` layers takes it past 2^64.
    let many_layers = texture(
        65_536,
        65_536,
        1,
        TextureFormat::Rgba32Float,
        1,
        u32::MAX,
        1,
    );
    assert_eq!(bytes(&many_layers, TextureFormat::Rgba32Float), None);

    // And the overflow is caught by the arithmetic rather than by a range check
    // on any one field: a large-but-legal extent with a large-but-legal layer
    // count still estimates, so the refusal above is about the product and not
    // about a field being too big to accept.
    let large_but_legal = texture(4096, 4096, 1, TextureFormat::Rgba32Float, 1, 4096, 1);
    assert_eq!(
        bytes(&large_but_legal, TextureFormat::Rgba32Float),
        Some(4096_u64 * 4096 * 4096 * 16)
    );
}

/// The estimate grows with every dimension that grew, which is the only property
/// a caller could reasonably build a budget on.
#[test]
fn the_estimate_is_monotone_in_every_dimension() {
    let base = texture(8, 8, 1, TextureFormat::R8Unorm, 1, 1, 1);
    let wider = texture(16, 8, 1, TextureFormat::R8Unorm, 1, 1, 1);
    let taller = texture(8, 16, 1, TextureFormat::R8Unorm, 1, 1, 1);
    let deeper = texture(8, 8, 2, TextureFormat::R8Unorm, 1, 1, 1);
    let layered = texture(8, 8, 1, TextureFormat::R8Unorm, 1, 2, 1);
    let multisampled = texture(8, 8, 1, TextureFormat::R8Unorm, 1, 1, 2);
    let chained = texture(8, 8, 1, TextureFormat::R8Unorm, 2, 1, 1);

    let base_bytes = bytes(&base, TextureFormat::R8Unorm).expect("a fixed format");

    for (name, larger) in [
        ("wider", wider),
        ("taller", taller),
        ("deeper", deeper),
        ("layered", layered),
        ("multisampled", multisampled),
        ("chained", chained),
    ] {
        let estimate = bytes(&larger, TextureFormat::R8Unorm).expect("a fixed format");
        assert!(
            estimate > base_bytes,
            "growing {name} did not grow the estimate"
        );
    }
}

// ---------------------------------------------------------------------------
// Section 47.14 and 47.15 — the inventory records.
// ---------------------------------------------------------------------------

/// Plain counts, defaulting to nothing, with the four texture fields explicitly
/// overlapping.
///
/// `textures` counts every texture and each attachment field counts a subset, so
/// a render target that is also a color attachment is counted in both subsets and
/// once in the total. Adding the subsets to `textures` would double-count, and the
/// records are shaped so that a reader can see the overlap rather than having to
/// know it.
#[test]
fn the_live_object_counts_default_to_nothing_and_keep_the_subsets_apart() {
    let empty = LiveObjectCounts::default();
    assert_eq!(empty.buffers, 0);
    assert_eq!(empty.textures, 0);
    assert_eq!(empty.render_target_textures, 0);
    assert_eq!(empty.color_attachment_textures, 0);
    assert_eq!(empty.depth_stencil_textures, 0);
    assert_eq!(empty.texture_views, 0);
    assert_eq!(empty.samplers, 0);
    assert_eq!(empty.shader_modules, 0);
    assert_eq!(empty.bind_group_layouts, 0);
    assert_eq!(empty.bind_groups, 0);
    assert_eq!(empty.pipeline_interfaces, 0);
    assert_eq!(empty.raster_pipelines, 0);
    assert_eq!(empty.compute_pipelines, 0);
    assert_eq!(empty.outstanding_frames, 0);

    // One texture that is both a render target and a color attachment is two
    // subset memberships and one texture, which is the arithmetic the chapter
    // warns about.
    let overlapped = LiveObjectCounts {
        buffers: 2,
        textures: 1,
        render_target_textures: 1,
        color_attachment_textures: 1,
        outstanding_frames: 1,
        ..LiveObjectCounts::default()
    };
    assert_eq!(overlapped.textures, 1);
    assert_eq!(overlapped.render_target_textures, 1);
    assert_ne!(
        overlapped.textures,
        overlapped.textures + overlapped.render_target_textures
    );
}

/// The byte estimate has the same top-level/subset split as the counts, and the
/// three top-level fields are the ones that add up.
#[test]
fn the_memory_statistics_separate_the_total_from_the_overlapping_subsets() {
    let measured = ResourceMemoryStatistics {
        buffers: MemoryEstimate::logical(1024),
        textures: MemoryEstimate::logical(2048),
        total_resources: MemoryEstimate::logical(3072),
        render_target_textures: MemoryEstimate::logical(2048),
        color_attachment_textures: MemoryEstimate::logical(2048),
        depth_stencil_textures: MemoryEstimate::unknown(),
    };

    // The top-level relation, which is the one a caller may sum.
    assert_eq!(
        measured.buffers.logical_estimated_bytes.unwrap()
            + measured.textures.logical_estimated_bytes.unwrap(),
        measured.total_resources.logical_estimated_bytes.unwrap()
    );

    // The subsets are already inside `textures`, so adding them would double it.
    assert_ne!(
        measured.textures.logical_estimated_bytes,
        Some(
            measured.textures.logical_estimated_bytes.unwrap()
                + measured
                    .render_target_textures
                    .logical_estimated_bytes
                    .unwrap()
        )
    );

    // And a subset with no fixed backing is `Unknown` without making the total
    // unknown, because the top level counts the texture once, not per subset.
    assert_eq!(
        measured.depth_stencil_textures.quality,
        MemoryEstimateQuality::Unknown
    );
    assert_eq!(
        measured.total_resources.quality,
        MemoryEstimateQuality::LogicalEstimate
    );
}

/// The inventory reports the device it observed, so that a caller aggregating two
/// devices can tell the two reports apart.
///
/// Compiled, never called: the live inventory reads the object lifecycle table the
/// RHI maintains as it creates and reclaims objects, and that table does not exist
/// yet. What this reviews is the shape of the answer — a device identity, the
/// counts, and the estimate travelling together, rather than three calls that
/// could observe three different moments.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_an_inventory_report_names_its_device(service: &DeviceStatistics) -> InventoryStatistics {
    let report = service
        .inventory()
        .expect("the lifecycle table is readable");

    assert_ne!(report.device, other_device());
    report
}

/// The refusal path of the inventory verb is not testable today and the estimate's
/// is, which is worth stating where the two sit side by side: `inventory()` names
/// no object, so it has no portable refusal path and panics unconditionally,
/// while `estimate_buffer_memory` refuses before it does anything else.
#[test]
fn the_device_free_estimate_is_the_half_that_can_be_driven_today() {
    let service = DeviceStatistics::new(device());

    let buffer = crate::api::tests::fixture::buffer(
        crate::api::identity::ObjectId::new(9),
        device(),
        crate::api::resource::buffer::BufferDescriptor::new(
            4096,
            crate::api::resource::buffer::BufferUsage::STORAGE,
        ),
    );

    assert_eq!(
        service
            .estimate_buffer_memory(&buffer)
            .expect("the owning device")
            .logical_estimated_bytes,
        Some(4096)
    );
}
