//! Tests for the capability lowering.
//!
//! Every test runs against a snapshot the Layer 1 fixtures build, not against a
//! hand-built `DeviceCapabilities`, because what is being tested is the *reading*
//! of a discovery result -- and a value constructed directly in the test would
//! assert itself.  Two fixtures carry most of the file: the WebGL2 snapshot, which
//! proves nothing optional, and the desktop snapshot that proved compute and
//! storage.  The interesting assertions read those two against the same field.

use super::*;
use crate::webgl2::api::tests::{
    builder, compute_storage_snapshot, desktop_limits, formats, indirect_snapshot, limits,
    snapshot, snapshot_with_fact, snapshot_with_formats, snapshot_with_surface,
};
use crate::webgl2::api::{
    GlExtensionSet, GlFamilyProfile, GlFormatEvidence, GlFormatResourceKind, GlOperationProbe,
};

/// The one queue a GL-family context has.
///
/// A single context is one ordered queue, reported as queue zero rather than as a
/// list, because `QueueId` is documented as a logical identifier and not a native
/// queue-family index.
fn queue(capabilities: &DeviceCapabilities) -> QueueDescriptor {
    assert_eq!(capabilities.queues.len(), 1, "one context is one queue");
    capabilities.queues[0]
}

/// One format's row, which every format test asserts against.
fn format_row(
    capabilities: &DeviceCapabilities,
    format: TextureFormat,
) -> TextureFormatCapabilities {
    capabilities
        .texture_formats
        .iter()
        .find(|entry| entry.format == format)
        .unwrap_or_else(|| panic!("{format:?} is reported"))
        .clone()
}

/// The formats the lowering reported, in the order it reported them.
fn reported(capabilities: &DeviceCapabilities) -> Vec<TextureFormat> {
    capabilities
        .texture_formats
        .iter()
        .map(|entry| entry.format)
        .collect()
}

// ---------------------------------------------------------------------------
// The optional domains: reported only where they were proved.
// ---------------------------------------------------------------------------

#[test]
fn a_webgl2_context_reports_no_optional_domain() {
    let capabilities = capabilities(&snapshot(GlFamilyProfile::WebGl2));

    let queue = queue(&capabilities);
    assert!(
        queue.capabilities.raster && queue.capabilities.copy,
        "raster and copy are core on every accepted profile"
    );
    assert!(
        !queue.capabilities.compute,
        "a WebGL2 context proved no compute, so its queue cannot execute any"
    );
    assert!(
        !queue.capabilities.present,
        "no surface is attached in this slice, so no queue presents"
    );
    assert_eq!(
        capabilities.buffers,
        BufferCapabilities::new(false, false, false),
        "no storage and no indirect domain was proved"
    );
}

#[test]
fn a_context_that_proved_compute_and_storage_reports_both() {
    let capabilities = capabilities(&compute_storage_snapshot(true));

    assert!(queue(&capabilities).capabilities.compute);
    assert_eq!(
        capabilities.buffers,
        // Both storage directions from the one proved fact: Layer 1 carries
        // read-only and read-write as a per-binding usage, and the discovery set
        // has no fact that could separate them, so reporting one without the
        // other would describe a distinction nothing observed.
        BufferCapabilities::new(true, true, false),
        "storage buffers reported, indirect still not proved"
    );
}

/// Both indirect rows the shipped providers can pass reach the buffer flag.
///
/// The flag is the *union* of three ledger rows and nothing in the suite would
/// have noticed if it collapsed to one term or to `false`: before this test the
/// field was asserted `false` twice and never `true`.  Two rows are exercised
/// rather than one because they are the two a shipped provider can enable --
/// the draw half is core one desktop version before the dispatch half, so a
/// lowering wired to either row alone is a real possibility rather than a
/// hypothetical one.
#[test]
fn a_proved_indirect_command_row_reports_indirect_buffer_reads() {
    for (row, name) in [
        (GlCapability::IndirectDraw, "draw"),
        (GlCapability::IndirectDispatch, "dispatch"),
    ] {
        let capabilities = capabilities(&indirect_snapshot(row, GlOperationProbe::Passed));
        assert!(
            capabilities.buffers.indirect_read,
            "a proved indirect {name} reads its parameters from a buffer"
        );
        assert!(
            !capabilities.buffers.storage_read && !capabilities.buffers.storage_write,
            "and proves nothing about the storage rows, which are different rows"
        );
    }
}

/// The fail-closed direction, on the one indirect row no provider can pass.
///
/// `ProbeReport::multi_draw_indirect` is `ProbeAnswer::Unavailable` at its only
/// construction site (`api/native/probes/mod.rs`, with the reason recorded
/// there: glow 0.18 binds no entry point for it), and
/// `passed_probes_enable_core_proved_capabilities_on_desktop_46` asserts the row
/// stays disabled even when every probe passes.  So this fixture is the row's
/// reachable state, and the assertion is what the compiler reads from it: a
/// graph may not declare an indirect buffer read here, because nothing in this
/// release could honour one.
#[test]
fn an_indirect_row_without_a_passed_probe_reports_no_indirect_buffer_read() {
    let capabilities = capabilities(&indirect_snapshot(
        GlCapability::MultiDrawIndirect,
        GlOperationProbe::NotRun,
    ));
    assert_eq!(
        capabilities.buffers,
        BufferCapabilities::new(false, false, false),
        "the route and the count limit are both satisfied, and the row still does not enable"
    );
}

#[test]
fn the_workgroup_count_is_reported_only_for_a_context_that_proved_dispatch() {
    // The discriminating pair: both fixtures queried the same non-zero workgroup
    // counts, and only the one that proved compute may report them.  A lowering
    // that copied the queried limits would pass the second assertion and fail the
    // first.
    let webgl2 = capabilities(&snapshot(GlFamilyProfile::WebGl2));
    assert_eq!(
        webgl2.limits.max_compute_workgroups_per_dimension, [0; 3],
        "the limit was queried and is non-zero, but no dispatch was ever proved"
    );

    let compute = capabilities(&compute_storage_snapshot(false));
    assert_eq!(
        compute.limits.max_compute_workgroups_per_dimension,
        desktop_limits().max_compute_work_group_count,
        "a proved dispatch reports the queried dimension"
    );
}

#[test]
fn the_colour_attachment_limit_is_the_smaller_of_the_two_queried_counts() {
    // `glGet` answers these two counts separately and a real context answers them
    // differently, so the fixture makes them differ: the pass limit is the
    // smaller, and reporting the framebuffer's own count instead would let the
    // compiler accept a pass the draw state then rejects.  The framebuffer's is
    // the one raised, because both counts have a profile floor below them.
    let mut queried = limits();
    queried.max_color_attachments = 8;
    let framebuffer_limit = queried.max_color_attachments;
    let alignment = queried.uniform_buffer_offset_alignment;
    let snapshot = builder(GlFamilyProfile::WebGl2, GlExtensionSet::default(), queried).build();

    let capabilities = capabilities(&snapshot);
    assert_eq!(
        capabilities.limits.max_color_attachments, 4,
        "the smaller of the two is the legal attachment count"
    );
    assert_ne!(
        capabilities.limits.max_color_attachments, framebuffer_limit,
        "the framebuffer's own count is not on its own a legal pass"
    );
    assert_eq!(
        capabilities.limits.min_uniform_buffer_offset_alignment, alignment,
        "and an unrelated limit is still a copy of what was queried"
    );
}

#[test]
fn the_shape_of_the_context_is_reported_rather_than_inherited() {
    // These fields all equal the builder's fail-closed default.  The test pins
    // them because "equal to the default" is not the same claim as "read from
    // this context", and the reading is what the lowering is for.
    let capabilities = capabilities(&snapshot(GlFamilyProfile::WebGl2));
    assert_eq!(
        capabilities.recording,
        RecordingCapabilities::new(RecordingModel::ImmediateContext, false),
        "Layer 1's verbs are immediate and the owner-thread rule forbids a second recorder"
    );
    assert_eq!(
        capabilities.transitions,
        TransitionCapabilities::BackendManaged
    );
    assert_eq!(
        capabilities.synchronization,
        SynchronizationCapabilities::SingleQueueOrdering
    );
    assert_eq!(
        capabilities.transient_resources,
        TransientResourceCapabilities::new(false, false, false),
        "no transient-creation path exists yet, so nothing may be assumed about reuse"
    );
    assert_eq!(
        capabilities.timestamps,
        TimestampCapabilities::Unsupported,
        "a timer query's counter width is not a pass-boundary timestamp"
    );
    assert_eq!(capabilities.surface, None, "presentation is a later slice");
}

// ---------------------------------------------------------------------------
// The surface: a claim made only where the drawable was read.
// ---------------------------------------------------------------------------

#[test]
fn a_drawable_read_as_four_eight_bit_channels_is_the_one_format_it_names() {
    let capabilities = capabilities(&snapshot_with_surface([8, 8, 8, 8]));

    assert_eq!(
        capabilities.surface,
        Some(SurfaceCapabilities::new(
            vec![TextureFormat::Rgba8Unorm],
            true,
            true
        )),
        "the widths name one format, and both operations follow the resource path"
    );
    assert!(
        queue(&capabilities).capabilities.present,
        "a queue that cannot present frames the device can describe is the same fact told twice"
    );
}

#[test]
fn a_surface_is_reported_only_where_the_drawable_was_read() {
    // Every one of these is a drawable this contract has no format for, and the
    // answer to each is the fail-closed one: no surface, and a queue that does
    // not present, which is what makes a present root fail in the compiler with
    // a reason about the device rather than one about the graph.
    for (case, facts) in [
        ("never observed", snapshot(GlFamilyProfile::WebGl2)),
        (
            "sixteen-bit channels",
            snapshot_with_surface([16, 16, 16, 16]),
        ),
        ("ten-bit channels", snapshot_with_surface([10, 10, 10, 2])),
        // A context created without alpha really does report a zero width, and
        // the claim has to follow the observation rather than the common shape.
        ("no alpha channel", snapshot_with_surface([8, 8, 8, 0])),
    ] {
        let capabilities = capabilities(&facts);
        assert_eq!(capabilities.surface, None, "{case} claims no surface");
        assert!(
            !queue(&capabilities).capabilities.present,
            "{case} presents nothing"
        );
    }
}

// ---------------------------------------------------------------------------
// The format table: what folds, what is dropped, and which side it lands on.
// ---------------------------------------------------------------------------

#[test]
fn a_colour_format_and_a_depth_format_land_on_their_own_sides_of_an_attachment() {
    let capabilities = capabilities(&snapshot(GlFamilyProfile::WebGl2));

    let colour = format_row(&capabilities, TextureFormat::Rgba8Unorm);
    assert!(colour.color_attachment);
    assert!(
        !colour.depth_stencil_attachment,
        "one format is never both sides: GL's renderable fact does not say which, so the format does"
    );

    let depth = format_row(&capabilities, TextureFormat::Depth32Float);
    assert!(depth.depth_stencil_attachment);
    assert!(!depth.color_attachment, "and the same the other way round");
}

#[test]
fn a_format_the_common_contract_has_no_name_for_is_not_reported() {
    let capabilities = capabilities(&snapshot(GlFamilyProfile::WebGl2));

    assert_eq!(
        reported(&capabilities),
        vec![
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Depth32Float,
        ],
        "the three the fixture recorded, in the GL table's own order"
    );
    assert!(
        !reported(&capabilities).contains(&TextureFormat::Bgra8Unorm),
        "no accepted GL-family profile has a BGRA texture format, so nothing observed it"
    );
    assert!(
        !reported(&capabilities).contains(&TextureFormat::Rgba16Float),
        "a format the table never recorded has no row, proved or otherwise"
    );
}

#[test]
fn a_format_is_reported_without_its_storage_facts_until_one_was_probed() {
    // The distinction the compiler acts on: a row with storage false is a format
    // that may be used and not written from a shader, while an absent row is one
    // the backend cannot name at all.
    let plain = capabilities(&snapshot(GlFamilyProfile::WebGl2));
    let probed = capabilities(&compute_storage_snapshot(true));

    let plain_storage = format_row(&plain, TextureFormat::Rgba8Unorm);
    assert!(
        !plain_storage.storage_read && !plain_storage.storage_write,
        "the WebGL2 fixture recorded no operation evidence for storage"
    );
    let probed_storage = format_row(&probed, TextureFormat::Rgba8Unorm);
    assert!(probed_storage.storage_read && probed_storage.storage_write);
    assert!(
        probed_storage.copy_source && probed_storage.copy_destination,
        "storage being proved did not replace the core-guaranteed copy facts"
    );
}

#[test]
fn the_attachment_sample_counts_are_the_renderable_texture_counts() {
    // A texture row at a second sample count folds into the same format, and a
    // renderbuffer row at any count does not: the common contract's format row is
    // about textures, and the GL table splits the two because WebGL2 exposes one
    // without the other.
    let texture = capabilities(&snapshot_with_fact(GlFormatResourceKind::Texture, 4));
    assert_eq!(
        format_row(&texture, TextureFormat::Rgba8Unorm).attachment_sample_counts,
        vec![1, 4],
        "both counts are legal attachments, ascending"
    );

    let renderbuffer = capabilities(&snapshot_with_fact(GlFormatResourceKind::Renderbuffer, 4));
    assert_eq!(
        format_row(&renderbuffer, TextureFormat::Rgba8Unorm).attachment_sample_counts,
        vec![1],
        "the renderbuffer's count is a fact about a resource this row does not describe"
    );
}

#[test]
fn a_recorded_format_with_no_renderable_count_is_reported_with_none() {
    // "Recorded, samplable and copyable, but not an attachment at any count" is a
    // row that exists with an empty list -- which is not the same reading as the
    // absent row the previous test produces, and the compiler rejects different
    // work for each.  The case is reached with an extended float format, which a
    // WebGL2 context samples only with the linear-filter extension and cannot
    // render to without a second one.
    let mut table = formats(false);
    table
        .record(GlFormatCapabilities {
            format: GlFormat::Rgba16Float,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::OperationProbed,
            sampled: true,
            filterable: false,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: true,
            copy_destination: true,
        })
        .expect("the baseline table does not record it");

    let entry = format_row(
        &capabilities(&snapshot_with_formats(table)),
        TextureFormat::Rgba16Float,
    );
    assert!(
        entry.sampled,
        "the row exists because sampling was observed"
    );
    assert!(
        !entry.filterable,
        "and it does not claim the linear filtering no extension supplied"
    );
    assert!(!entry.color_attachment && !entry.depth_stencil_attachment);
    assert!(
        entry.attachment_sample_counts.is_empty(),
        "recorded, usable, and not an attachment at any count"
    );
}
