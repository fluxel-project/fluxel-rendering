//! Writing host bytes into an object this device created.
//!
//! The frame suites drive these verbs as the thing that makes an import
//! possible.  What these tests ask is the narrower question those suites cannot:
//! what each verb lowers to on its own, and which requests it refuses before any
//! driver call.  Both matter because these are the only verbs on the adapter that
//! take host bytes at all -- the execution contract has none, on purpose -- so
//! they are the only place a length, an offset or an extent is checked against
//! anything other than a graph's compiled requirement.
//!
//! # Two arms, and the asymmetry that shapes both suites
//!
//! A buffer's extent is the caller's slice, so its verb derives the range from
//! `bytes.len()` and a length that disagrees with something the caller also had
//! in mind cannot be expressed.  A texture's extent is a fact about the object, so
//! its verb derives the region from the creation record and the caller's slice has
//! to agree with it -- which puts one more refusal in this suite's reach, and it
//! is the one worth reading closely.
//!
//! # The refusals are two error kinds, and telling them apart is the point
//!
//! Every refusal below names the verb the caller called, so the operation string
//! alone cannot say *why*.  What separates them is the kind and the moment.  An
//! identity this device never created is a validation failure decided *here*,
//! against the adapter's own creation record.  A range past a buffer's end, or a
//! slice that does not fill a texture's level, is a validation failure decided by
//! the layer holding the real descriptor -- after this adapter has committed to
//! the call and before any driver command.  A dimension or a format this family
//! cannot transfer is `Unsupported` rather than a validation failure, because it
//! is a fact about the family and not about the request.  So a suite has to check
//! which happened rather than only that something did, and the driver-call count
//! is what says which.

use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferDesc, BufferUsage, BufferUsageKind, ExecutionBackend,
    TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUsageKind,
};

use super::super::retention::GlRetentionLease;
use super::{
    Adapter, adapter, calls, invalid, plain_texture, refusal_reason, refused, texture,
    trace_from_here,
};
use crate::webgl2::api::{BufferId, MockCall, TextureId};

/// A transient buffer of `size` bytes, in the role the tests here upload for.
///
/// Built through the adapter rather than by hand for the reason every fixture in
/// this module is: the identity has to be one a real creation produced, because
/// the adapter's record of it is exactly what the refusal below is about.
fn buffer(adapter: &mut Adapter, size: u64) -> BoundBuffer<BufferId, GlRetentionLease> {
    adapter
        .create_transient_buffer(
            BufferDesc { size },
            BufferUsage::from_kinds([BufferUsageKind::Vertex]),
        )
        .unwrap_or_else(|error| panic!("a transient vertex buffer: {error:?}"))
}

/// A transient texture created from `descriptor`, sampled and nothing else.
///
/// Sampled because that is the role every texture these verbs fill is created
/// for, and it is also the usage that proves the ordering the frame suites depend
/// on: a texture may be created for a *read* it has not been given contents for
/// yet, since the record this adapter keeps is about existence rather than about
/// what the pixels currently are.
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

/// The tightly packed RGBA8 bytes for one descriptor's mip zero.
///
/// Built from the extent rather than written out as a byte literal, so the length
/// is something a reader can check against the descriptor instead of trusting --
/// and so that a test which needs the length to be *wrong* can say by how much
/// rather than restating the whole array.  Each texel carries its own index in
/// red, which makes the array visibly a grid of pixels and not a block of fill.
fn pixels(descriptor: TextureDesc) -> Vec<u8> {
    let texels = descriptor.extent.width as usize * descriptor.extent.height as usize;
    (0..texels)
        .flat_map(|texel| [texel as u8, 0, 0, 0xff])
        .collect()
}

#[test]
fn an_upload_lowers_to_one_call_over_the_callers_offset_and_the_slices_length() {
    let mut adapter = adapter();
    let buffer = buffer(&mut adapter, 64);
    trace_from_here(&mut adapter);

    adapter
        .upload_buffer(buffer.physical, 8, &[1, 2, 3, 4])
        .expect("four bytes at offset eight of a sixty-four byte allocation");

    assert_eq!(
        calls(&mut adapter),
        vec![MockCall::UploadBuffer {
            buffer: buffer.physical,
            offset: 8,
            size: 4,
        }],
        "one call, at the caller's offset, covering the caller's bytes rather than the allocation"
    );
}

#[test]
fn an_upload_naming_a_buffer_this_device_has_destroyed_is_refused() {
    let mut adapter = adapter();
    let buffer = buffer(&mut adapter, 64);
    let physical = buffer.physical;
    drop(buffer);
    // The lease's drop is what queues the object; draining the queue is what
    // forgets the size record, so the identity stops resolving exactly here and
    // not when the last handle went away.
    adapter
        .release_pending()
        .expect("a live context destroys what it was asked to");
    trace_from_here(&mut adapter);

    assert_eq!(
        invalid(adapter.upload_buffer(physical, 0, &[0; 4])),
        "upload-buffer",
        "the refusal names the verb the caller called"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and it is decided against the adapter's record, before any driver call"
    );
}

#[test]
fn an_upload_past_the_allocations_end_is_refused_by_the_layer_holding_the_descriptor() {
    let mut adapter = adapter();
    let buffer = buffer(&mut adapter, 16);
    trace_from_here(&mut adapter);

    // Sixteen bytes at offset eight of a sixteen byte allocation: the slice's
    // own length is what made the range, so this is the case the adapter cannot
    // see -- only the layer holding the descriptor knows where the buffer ends.
    assert_eq!(
        invalid(adapter.upload_buffer(buffer.physical, 8, &[0; 16])),
        "upload-buffer",
        "the refusal reaches the caller under the same operation name"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and no driver command was issued for it"
    );

    // The control: the same buffer, the same sixteen bytes, at the offset where
    // they fit.  Without it the refusals above would be evidence that this verb
    // refuses, not that it checks.
    adapter
        .upload_buffer(buffer.physical, 0, &[0; 16])
        .expect("the whole allocation is a range the caller may fill");
}

#[test]
fn a_texture_upload_covers_the_extent_the_creation_record_holds() {
    let mut adapter = adapter();
    let descriptor = plain_texture();
    let texture = sampled_texture(&mut adapter, descriptor);
    let bytes = pixels(descriptor);
    trace_from_here(&mut adapter);

    adapter
        .upload_texture(texture.physical, &bytes)
        .expect("one tightly packed RGBA8 mip of a four by four attachment");

    assert_eq!(
        calls(&mut adapter),
        vec![MockCall::UploadTexture(texture.physical)],
        "one call, and no state traffic around it: Layer 1 owns the pixel-store save and restore, \
         so a recorder that saw a bind here would be recording a command this family never issues"
    );
    assert_eq!(
        bytes.len(),
        4 * 4 * 4,
        "the caller's slice is the whole level, which is what makes its length a fact the verb can be wrong about"
    );
}

#[test]
fn a_texture_upload_naming_a_texture_this_device_has_destroyed_is_refused() {
    let mut adapter = adapter();
    let texture = sampled_texture(&mut adapter, plain_texture());
    let physical = texture.physical;
    drop(texture);
    adapter
        .release_pending()
        .expect("a live context destroys what it was asked to");
    trace_from_here(&mut adapter);

    assert_eq!(
        invalid(adapter.upload_texture(physical, &[0; 64])),
        "upload-texture",
        "the refusal names the verb the caller called"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and it is decided against the attachment record, before any driver call"
    );
}

#[test]
fn a_texture_upload_that_does_not_fill_the_level_is_refused_by_the_layer_holding_the_descriptor() {
    let mut adapter = adapter();
    let descriptor = plain_texture();
    let texture = sampled_texture(&mut adapter, descriptor);
    let bytes = pixels(descriptor);
    trace_from_here(&mut adapter);

    // The asymmetry this file's own doc names, in executable form: the caller's
    // slice is the *only* thing that could have said how big a buffer upload is,
    // while a texture's extent is already a fact and the slice has to agree with
    // it.  So this is the refusal the buffer arm has no counterpart for.
    let short = &bytes[..bytes.len() - 1];
    assert_eq!(
        invalid(adapter.upload_texture(texture.physical, short)),
        "upload-texture",
        "the refusal reaches the caller under the same operation name"
    );
    // Not `is_empty`, and the difference is Layer 1's rather than this adapter's:
    // the mock's upload path decides a length mismatch through the helper that
    // records the refusal as a `MockCall::Error`, while the buffer path's range
    // check builds the same error without recording it.  So the trace here holds
    // the recorder's own note and no command, and the assertion has to say which
    // it is checking for -- a bare count would pass for the wrong reason on one
    // arm and fail on the other.
    assert!(
        calls(&mut adapter)
            .iter()
            .all(|call| matches!(call, MockCall::Error(_))),
        "and the only entry is the recorder's note of the refusal, not a driver command"
    );

    // The control: the same texture and the same bytes, at the length that fills
    // the level.  Without it the refusal above would be evidence that this verb
    // refuses, not that it checks.
    adapter
        .upload_texture(texture.physical, &bytes)
        .expect("the whole level is the extent the record holds");
}

#[test]
fn a_texture_upload_of_a_shape_this_verb_cannot_address_is_refused() {
    let mut adapter = adapter();
    // Four layers rather than one: the common contract spells arrayed storage as
    // a layer count on a two-dimensional descriptor, and the adapter's lowering is
    // what turns that into the arrayed GL target.  The extent is still four by
    // four, so a verb that read only the extent would upload one layer of four and
    // call it the whole level.  Nothing about the request is wrong -- the shape
    // simply is not one a rectangle can describe the whole of.
    let layered = texture(TextureDimension::D2, 4, 1);
    let texture = sampled_texture(&mut adapter, layered);
    trace_from_here(&mut adapter);

    assert_eq!(
        refused(adapter.upload_texture(texture.physical, &[0; 64])),
        "upload-texture",
        "the refusal names the verb the caller called, not the binding point that would have refused the same attachment"
    );
    assert_eq!(
        refusal_reason(adapter.upload_texture(texture.physical, &[0; 64])),
        "this verb transfers a whole two-dimensional level, which is not the shape this attachment was created with",
        "and it is the fail-closed kind, because the shape is a fact about the family rather than about the request"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and it is decided before any driver call"
    );
}

#[test]
fn a_texture_upload_of_a_format_this_family_cannot_transfer_is_refused() {
    let mut adapter = adapter();
    // A depth format, and the fixture is chosen rather than convenient: this
    // family both creates and samples it, so the object reaching the verb is a
    // legitimate one in every other respect, and the only thing wrong with the
    // call is the one under test.  The half-float format would have made a weaker
    // witness and does not exist on this snapshot's table at all -- so a test
    // written with it would have been asserting on a creation refusal instead.
    let depth = TextureDesc {
        format: TextureFormat::Depth32Float,
        ..plain_texture()
    };
    let texture = sampled_texture(&mut adapter, depth);
    trace_from_here(&mut adapter);

    assert_eq!(
        refused(adapter.upload_texture(texture.physical, &[0; 64])),
        "upload-texture",
        "the refusal names the verb the caller called"
    );
    assert!(
        calls(&mut adapter).is_empty(),
        "and it is decided before any driver call, against the record rather than by a provider"
    );
}
