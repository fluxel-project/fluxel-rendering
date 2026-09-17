//! Reading host bytes back out of an object this device created.
//!
//! The mirror of [`super::upload`], and the same shape of suite for the same
//! reason: these are the two verbs that move host bytes across the adapter's
//! boundary, so they are the only place a shape or a format is checked against
//! something other than a graph's compiled requirement.
//!
//! # What is asserted here, and what is not
//!
//! The wiring -- which call the verb lowers to, which requests it refuses, and
//! what it does about the lifecycle it adopts first -- is assertable over the
//! recorder, because those are facts about the adapter.  Whether the bytes are
//! the right picture is not: the recorder has no rasterizer, so a readback here
//! returns the recorder's own zeroes.  That claim belongs to a real context, and
//! its vehicle is the hardware run (`CLAUDE.md` §4.5).
//!
//! # The one place this suite is shorter than `upload`'s
//!
//! An upload's caller hands in bytes, so its length is a fact that can disagree
//! with the level it fills and the suite has a case for that disagreement.  A
//! readback's caller hands in nothing, so there is no such case to write: the
//! extent, the shape and the format all come from the creation record, and the
//! test that would have been about a wrong descriptor instead asserts that the
//! record is what decides -- an eight-by-four object reads back eight by four
//! because nothing else could have said so.

use fluxel_rendergraph::{
    BoundTexture, ExecutionBackend, Extent3d, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage, TextureUsageKind,
};

use super::super::retention::GlRetentionLease;
use super::{
    Adapter, adapter, calls, invalid, plain_texture, refusal_reason, refused, texture,
    trace_from_here,
};
use crate::webgl2::api::{GlFamilyApi, MockCall, TextureId};

/// A transient texture created from `descriptor`, sampled and nothing else.
///
/// Built through the adapter rather than by hand, for the reason every fixture in
/// this module is: the identity has to be one a real creation produced, because
/// the adapter's record of it is what the readback is resolved against.
fn sampled_texture(
    adapter: &mut Adapter,
    descriptor: TextureDesc,
) -> BoundTexture<TextureId, GlRetentionLease> {
    adapter
        .create_transient_texture(
            descriptor,
            TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        )
        .unwrap_or_else(|error| panic!("a transient sampled texture: {error:?}"))
}

#[test]
fn a_readback_is_one_call_over_the_extent_the_creation_record_holds() {
    let mut adapter = adapter();
    // Not the square every other fixture here uses, and deliberately: a
    // transposed or defaulted extent would read back as four by four and this is
    // the assertion that would catch it.
    let descriptor = TextureDesc {
        extent: Extent3d {
            width: 8,
            height: 4,
            depth: 1,
        },
        ..plain_texture()
    };
    let texture = sampled_texture(&mut adapter, descriptor);
    trace_from_here(&mut adapter);

    let pixels = adapter
        .read_texture(texture.physical)
        .expect("one tightly packed RGBA8 mip of an eight by four texture");

    assert_eq!(
        calls(&mut adapter),
        vec![MockCall::ReadTexture(texture.physical)],
        "one call, and no state traffic around it: Layer 1 owns the pixel-store save and restore, \
         so a recorder that saw a bind here would be recording a command this family never issues"
    );
    assert_eq!(
        pixels.extent,
        [8, 4],
        "the extent is the record's, which is the only place it could have come from"
    );
    assert_eq!(
        pixels.bytes.len(),
        8 * 4 * 4,
        "and the length follows from the pitch the shared transfer derived, so a padded or \
         transposed layout would not have produced this many bytes"
    );
}

/// The lifecycle is adopted first, exactly as on the upload arm.
///
/// A context generation change invalidates every record of the previous epoch, so
/// an identity resolved without noticing one would be resolved against a record
/// that no longer describes this device's objects.  What the verb must do about
/// that is refuse rather than read, and the assertion is that the refusal happens
/// where the records are dropped rather than by luck at the provider.
#[test]
fn a_readback_after_a_context_generation_change_is_refused() {
    let mut adapter = adapter();
    let descriptor = plain_texture();
    let texture = sampled_texture(&mut adapter, descriptor);

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
    trace_from_here(&mut adapter);

    assert_eq!(
        invalid(adapter.read_texture(texture.physical)),
        "read-texture",
        "the identity outlived the generation that minted it, so the record is gone and the verb \
         refuses rather than reading whatever now answers to that slot"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and the refusal is decided before any driver call"
    );
}

#[test]
fn a_readback_naming_a_texture_this_device_has_destroyed_is_refused() {
    let mut adapter = adapter();
    let texture = sampled_texture(&mut adapter, plain_texture());
    let physical = texture.physical;
    drop(texture);
    adapter
        .release_pending()
        .expect("a live context destroys what it was asked to");
    trace_from_here(&mut adapter);

    assert_eq!(
        invalid(adapter.read_texture(physical)),
        "read-texture",
        "the refusal names the verb the caller called"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and it is decided against the adapter's own record, before any driver call"
    );
}

#[test]
fn a_readback_of_a_shape_this_verb_cannot_address_is_refused() {
    let mut adapter = adapter();
    // Four layers rather than one: the common contract spells arrayed storage as
    // a layer count on a two-dimensional descriptor, and the extent is still four
    // by four -- so a verb that read only the extent would read one layer of four
    // and report it as the whole level.  Nothing about the request is wrong; the
    // shape simply is not one rectangle can describe the whole of.
    let layered = texture(TextureDimension::D2, 4, 1);
    let layered_texture = sampled_texture(&mut adapter, layered);
    // And the third dimension, which is the other way a rectangle stops being the
    // whole of an object: a depth that is not one.
    let volume = texture(TextureDimension::D3, 1, 4);
    let volume_texture = sampled_texture(&mut adapter, volume);
    trace_from_here(&mut adapter);

    assert_eq!(
        refused(adapter.read_texture(layered_texture.physical)),
        "read-texture",
        "the refusal names the verb the caller called, not the binding point that would have refused \
         the same object"
    );
    assert_eq!(
        refusal_reason(adapter.read_texture(volume_texture.physical)),
        "this verb transfers a whole two-dimensional level, which is not the shape this attachment was created with",
        "and it is the fail-closed kind, because the shape is a fact about the family rather than about the request"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and both are decided before any driver call"
    );
}

#[test]
fn a_readback_of_a_format_this_family_cannot_transfer_is_refused() {
    let mut adapter = adapter();
    // A depth format, and the fixture is chosen rather than convenient: this
    // family both creates and samples it, so the object reaching the verb is a
    // legitimate one in every other respect and the only thing wrong with the
    // call is the one under test.  It is also the case that shows why the format
    // policy is shared with the upload rather than restated here: the depth format
    // *does* have a GL format, and the refusal is about the client encoding rather
    // than about the object existing.
    let depth = TextureDesc {
        format: TextureFormat::Depth32Float,
        ..plain_texture()
    };
    let texture = sampled_texture(&mut adapter, depth);
    trace_from_here(&mut adapter);

    assert_eq!(
        refused(adapter.read_texture(texture.physical)),
        "read-texture",
        "the refusal names the verb the caller called"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and it is decided against the record rather than by a provider"
    );
}
