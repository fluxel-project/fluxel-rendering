//! The two verbs that reach the driver, and the two that reach nothing.
//!
//! F2's copy slice is the first part of this adapter that issues a command, and
//! the tests here are split along the line that makes it worth reviewing: a
//! *copy* has to reach Layer 1 with the region the caller named, and a
//! *transition* has to reach nothing at all while still refusing what it must.
//!
//! # What a copy test can and cannot observe
//!
//! The recorder keeps the identities a copy names and not its region
//! (`MockCall::CopyTexture`, `MockCall::CopyBuffer`), so "the lowered region was
//! correct" cannot be read off the trace.  It is established from two sides
//! instead, which is what makes either mean something.  The lowering is read
//! field by field where it is called directly, in the same way `transient`'s is
//! and for the same reason; and it is read again through the adapter, where a
//! region inside the descriptor reaches the driver and a region *outside* it is
//! rejected by Layer 1 with the operation's own name.  Drop the origin in the
//! lowering and the direct test fails on the origin while the in-bounds
//! behavioural test still passes -- which is exactly why both are here.
//!
//! # Why a transition's test asserts an empty trace
//!
//! "This verb is accepted and issues nothing" is the strongest claim in the
//! slice, and the only evidence for it that does not restate the implementation
//! is the trace: every mock call the adapter made, which for a transition must be
//! none at all -- no command, and no destruction either, since the adapter also
//! destroys deferred objects at some of its entry points.  What that cannot show
//! is that the ordering is genuinely carried elsewhere; that argument is made in
//! `accept_transition`'s documentation, and what is measured here is that the
//! verb does not quietly emit a command instead.

use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferCopyRegion, BufferDesc, BufferRange, BufferUsage,
    BufferUsageKind, ExecutionBackend, QueueId, ResourceAccessState, TextureCopyRegion,
    TextureDesc, TextureDimension, TextureRange, TextureUsage, TextureUsageKind,
};

use super::super::retention::GlRetentionLease;
use super::{Adapter, adapter, calls, invalid, plain_texture, texture, trace_from_here};
use crate::webgl2::api::{
    BufferId, GlError, GlExtent3d, GlFamilyApi, GlTextureAspect, MockCall, TextureId,
};

/// A transient texture created for exactly one side of a texture copy.
fn copyable(
    adapter: &mut Adapter,
    descriptor: TextureDesc,
    side: TextureUsageKind,
) -> BoundTexture<TextureId, GlRetentionLease> {
    adapter
        .create_transient_texture(descriptor, TextureUsage::from_kinds([side]))
        .expect("a transient texture")
}

/// A transient buffer created for both sides of a buffer copy.
fn copyable_buffer(adapter: &mut Adapter, size: u64) -> BoundBuffer<BufferId, GlRetentionLease> {
    adapter
        .create_transient_buffer(
            BufferDesc { size },
            BufferUsage::from_kinds([
                BufferUsageKind::CopySource,
                BufferUsageKind::CopyDestination,
            ]),
        )
        .expect("a transient buffer")
}

/// A copy region of `extent` texels at `origin`, on mip zero of both sides.
fn copy_region(source_origin: [u32; 3], extent: [u32; 3]) -> TextureCopyRegion {
    TextureCopyRegion {
        source_origin,
        destination_origin: [0, 0, 0],
        extent,
        source_mip_level: 0,
        destination_mip_level: 0,
    }
}

/// The operation name of the stale-object failure `result` carries.
fn stale<T>(result: Result<T, GlError>) -> &'static str {
    match result {
        Ok(_) => panic!("the adapter was expected to reject this as stale"),
        Err(GlError::StaleObject { operation, .. }) => operation,
        Err(other) => panic!("expected a stale-object failure, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The lowering, called directly because the trace cannot show a region.
// ---------------------------------------------------------------------------

#[test]
fn the_lowering_writes_out_every_number_the_contracts_region_carries() {
    let mut adapter = adapter();
    let texture = copyable(&mut adapter, plain_texture(), TextureUsageKind::CopySource).physical;

    // Every field of the contract's region has one place in Layer 1's, and the
    // two it has *none* for -- the aspect and the layers -- are the decisions
    // the module documents.  Read field by field rather than compared as a
    // whole, so a failure says which number went wrong.
    let region = super::super::region::texture_region(texture, 2, [1, 2, 3], [4, 5, 6]);

    assert_eq!(region.subresource.texture, texture);
    assert_eq!(
        region.subresource.mip_level, 2,
        "the mip level is the caller's and is not defaulted"
    );
    assert_eq!(
        region.subresource.aspect,
        GlTextureAspect::All,
        "the contract's region names no aspect, and All is the value that asserts nothing about the resource"
    );
    assert_eq!(
        (
            region.subresource.base_layer,
            region.subresource.layer_count
        ),
        (0, 1),
        "one selected layer, which is the whole of what the contract's region can express"
    );
    assert_eq!(region.origin, [1, 2, 3], "and the origin travels whole");
    assert_eq!(
        region.extent,
        GlExtent3d {
            width: 4,
            height: 5,
            depth_or_layers: 6,
        },
        "the third extent is carried as the contract wrote it; what it means is the descriptor's business"
    );
}

#[test]
fn the_buffer_lowering_keeps_the_two_offsets_apart_and_shares_one_size() {
    let mut adapter = adapter();
    let buffer = copyable_buffer(&mut adapter, 64).physical;

    let source = super::super::region::buffer_range(buffer, 8, 16);
    assert_eq!(source.buffer, buffer);
    assert_eq!(
        (source.offset, source.size),
        (8, 16),
        "the offset is the caller's, and the size is the one number the contract states for both sides"
    );
}

// ---------------------------------------------------------------------------
// The copies, which are the one thing here that reaches the driver.
// ---------------------------------------------------------------------------

#[test]
fn a_texture_copy_reaches_the_driver_as_one_copy_command() {
    let mut adapter = adapter();
    let source = copyable(&mut adapter, plain_texture(), TextureUsageKind::CopySource);
    let destination = copyable(
        &mut adapter,
        plain_texture(),
        TextureUsageKind::CopyDestination,
    );
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    assert!(
        adapter
            .copy_texture(
                &mut encoder,
                &source.physical,
                &destination.physical,
                copy_region([1, 1, 0], [2, 2, 1]),
            )
            .is_ok(),
        "a sub-rectangle inside both 4x4 mips is a copy this family can make"
    );

    // One call, and it is the driver's copy verb rather than a bracketed
    // sequence: this family has no copy scope to open, which is the same fact
    // `begin_copy` is a no-op for.
    assert!(
        matches!(
            calls(&mut adapter).as_slice(),
            [MockCall::CopyTexture { .. }]
        ),
        "one copy is one command: {:?}",
        calls(&mut adapter)
    );
    assert_eq!(
        calls(&mut adapter),
        vec![MockCall::CopyTexture {
            source: source.physical,
            destination: destination.physical,
        }],
        "and it names the two resources the caller named, in the caller's order"
    );
}

#[test]
fn the_region_the_caller_named_is_the_region_layer_one_validates() {
    let mut adapter = adapter();
    let source = copyable(&mut adapter, plain_texture(), TextureUsageKind::CopySource);
    let destination = copyable(
        &mut adapter,
        plain_texture(),
        TextureUsageKind::CopyDestination,
    );
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    // One texel past the right edge of a 4x4 mip.  The refusal is Layer 1's own
    // bound check and not the adapter's, which is the point: the adapter passes
    // the origin through and the provider is what knows the texture is 4 wide.
    assert_eq!(
        invalid(adapter.copy_texture(
            &mut encoder,
            &source.physical,
            &destination.physical,
            copy_region([4, 0, 0], [1, 1, 1]),
        )),
        "copy-texture",
        "an origin outside the mip is rejected against the real descriptor"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and a rejected copy issues nothing: {:?}",
        calls(&mut adapter)
    );

    // The same for the destination side, which is validated separately.
    assert_eq!(
        invalid(adapter.copy_texture(
            &mut encoder,
            &source.physical,
            &destination.physical,
            TextureCopyRegion {
                source_origin: [0, 0, 0],
                destination_origin: [0, 4, 0],
                extent: [1, 1, 1],
                source_mip_level: 0,
                destination_mip_level: 0,
            },
        )),
        "copy-texture"
    );

    // And the mip level is the caller's rather than a default: a texture with one
    // mip has no mip one.
    let single_mip = copyable(&mut adapter, plain_texture(), TextureUsageKind::CopySource);
    let mut region = copy_region([0, 0, 0], [1, 1, 1]);
    region.source_mip_level = 1;
    assert_eq!(
        invalid(adapter.copy_texture(
            &mut encoder,
            &single_mip.physical,
            &destination.physical,
            region,
        )),
        "copy-texture",
        "a mip the source does not have is rejected, so the mip level travelled"
    );
}

#[test]
fn the_array_layers_the_contract_cannot_name_stay_at_zero_and_a_volume_is_untouched() {
    let mut adapter = adapter();

    // A three-layer array: the contract's copy region names one extent and no
    // layer, so the lowering selects layer zero, and Layer 1 reads the third
    // extent against that one selected layer.
    let arrayed = texture(TextureDimension::D2, 3, 1);
    let source = copyable(&mut adapter, arrayed, TextureUsageKind::CopySource);
    let destination = copyable(&mut adapter, arrayed, TextureUsageKind::CopyDestination);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    assert!(
        adapter
            .copy_texture(
                &mut encoder,
                &source.physical,
                &destination.physical,
                copy_region([0, 0, 0], [4, 4, 1]),
            )
            .is_ok(),
        "one layer at layer zero is exactly what the region can ask for"
    );
    assert_eq!(
        invalid(adapter.copy_texture(
            &mut encoder,
            &source.physical,
            &destination.physical,
            copy_region([0, 0, 0], [4, 4, 3]),
        )),
        "copy-texture",
        "a region spanning the array's layers is refused rather than silently reinterpreted"
    );

    // A volume is the case the third extent means something else for: Layer 1
    // reads it as depth there, so the same shape of region goes through.  That
    // the third number survived the lowering at all is the direct test's
    // evidence and not this one's; what is shown here is that Layer 1 accepts a
    // region this deep against a volume and would not against an array.
    let volume = texture(TextureDimension::D3, 1, 6);
    let volume_source = copyable(&mut adapter, volume, TextureUsageKind::CopySource);
    let volume_destination = copyable(&mut adapter, volume, TextureUsageKind::CopyDestination);
    trace_from_here(&mut adapter);
    assert!(
        adapter
            .copy_texture(
                &mut encoder,
                &volume_source.physical,
                &volume_destination.physical,
                copy_region([0, 0, 0], [4, 4, 6]),
            )
            .is_ok(),
        "a volume's third extent is texels, and the whole of it is one copy"
    );
    assert_eq!(
        calls(&mut adapter),
        vec![MockCall::CopyTexture {
            source: volume_source.physical,
            destination: volume_destination.physical,
        }]
    );
}

#[test]
fn a_buffer_copy_reaches_the_driver_with_both_offsets_and_one_size() {
    let mut adapter = adapter();
    let buffer = copyable_buffer(&mut adapter, 64);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    assert!(
        adapter
            .copy_buffer(
                &mut encoder,
                &buffer.physical,
                &buffer.physical,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 32,
                    size: 32,
                },
            )
            .is_ok(),
        "one buffer, two disjoint halves of it, is a copy this family can make"
    );
    assert_eq!(
        calls(&mut adapter),
        vec![MockCall::CopyBuffer {
            source: buffer.physical,
            destination: buffer.physical,
            size: 32,
        }],
        "the size is one number for both sides, which is what the contract's region carries"
    );

    // The offsets are the caller's, and the provider is what checks them against
    // the real buffer: a range past the end is refused there.
    assert_eq!(
        invalid(adapter.copy_buffer(
            &mut encoder,
            &buffer.physical,
            &buffer.physical,
            BufferCopyRegion {
                source_offset: 48,
                destination_offset: 0,
                size: 32,
            },
        )),
        "copy-buffer"
    );
}

// ---------------------------------------------------------------------------
// The transitions, which are accepted and issue nothing.
// ---------------------------------------------------------------------------

#[test]
fn a_transition_is_accepted_whatever_it_names_and_issues_nothing() {
    use fluxel_rendergraph::TextureAspect;

    let mut adapter = adapter();
    let source = copyable(&mut adapter, plain_texture(), TextureUsageKind::CopySource);
    let buffer = copyable_buffer(&mut adapter, 64);
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    trace_from_here(&mut adapter);

    // A state change, a same-state pair -- the contract's memory-barrier case,
    // which it forbids treating as a no-op -- and a narrowed range.  All three
    // are accepted, and all three emit nothing, because in this family there is
    // no command either half of them would have been.
    assert!(
        adapter
            .transition_texture(
                &mut encoder,
                &source.physical,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .is_ok()
    );
    assert!(
        adapter
            .transition_texture(
                &mut encoder,
                &source.physical,
                TextureRange::Subresources {
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: TextureAspect::All,
                },
                ResourceAccessState::ColorAttachmentWrite,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .is_ok(),
        "the same-state case is the one the contract calls a barrier, and this family carries it elsewhere"
    );
    assert!(
        adapter
            .transition_buffer(
                &mut encoder,
                &buffer.physical,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ShaderStorageWrite,
            )
            .is_ok()
    );
    assert!(
        adapter
            .transition_buffer(
                &mut encoder,
                &buffer.physical,
                BufferRange::Bytes {
                    offset: 0,
                    size: 32,
                },
                ResourceAccessState::ShaderStorageWrite,
                ResourceAccessState::ShaderStorageWrite,
            )
            .is_ok()
    );

    // The trace is the evidence, and it has to be empty of *everything*: a
    // transition that issued no command but destroyed a deferred object would
    // still be doing work at a call that is documented as doing none.
    assert!(
        calls(&mut adapter).is_empty(),
        "a transition in this family is a check and nothing else: {:?}",
        calls(&mut adapter)
    );

    // And it did not close the encoder: a copy after the transitions still
    // reaches the driver.
    assert!(
        adapter
            .copy_buffer(
                &mut encoder,
                &buffer.physical,
                &buffer.physical,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 32,
                    size: 32,
                },
            )
            .is_ok()
    );
    assert_eq!(calls(&mut adapter).len(), 1);
}

// ---------------------------------------------------------------------------
// What the new verbs still refuse, and why that is not a fail-closed arm.
// ---------------------------------------------------------------------------

#[test]
fn a_copy_or_transition_naming_a_superseded_generation_is_refused() {
    let mut adapter = adapter();
    let stale_texture = copyable(&mut adapter, plain_texture(), TextureUsageKind::CopySource);
    let stale_buffer = copyable_buffer(&mut adapter, 64);
    let mut stale_encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");

    adapter
        .machine
        .backend()
        .context_lost()
        .expect("context loss");
    adapter
        .machine
        .backend()
        .context_restored()
        .expect("context restoration");

    // The encoder opened against the previous generation is the first thing
    // rejected, which is the whole job of the stamp it carries.
    assert_eq!(
        stale(adapter.transition_texture(
            &mut stale_encoder,
            &stale_texture.physical,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )),
        "transition-texture"
    );

    // A fresh encoder isolates the resource: the same generation question, asked
    // about an identity instead of about the recording.
    let mut encoder = adapter.begin_encoder(QueueId::new(0)).expect("an encoder");
    let current = copyable(
        &mut adapter,
        plain_texture(),
        TextureUsageKind::CopyDestination,
    );
    trace_from_here(&mut adapter);

    assert_eq!(
        stale(adapter.transition_buffer(
            &mut encoder,
            &stale_buffer.physical,
            BufferRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ShaderStorageWrite,
        )),
        "transition-buffer"
    );
    assert_eq!(
        stale(adapter.copy_texture(
            &mut encoder,
            &stale_texture.physical,
            &current.physical,
            copy_region([0, 0, 0], [4, 4, 1]),
        )),
        "copy-texture",
        "the stale object is rejected even though the recording is current"
    );
    assert_eq!(
        stale(adapter.copy_buffer(
            &mut encoder,
            &stale_buffer.physical,
            &stale_buffer.physical,
            BufferCopyRegion {
                source_offset: 0,
                destination_offset: 32,
                size: 32,
            },
        )),
        "copy-buffer"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and none of the four reached the driver: {:?}",
        calls(&mut adapter)
    );
}
