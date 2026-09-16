//! Writing host bytes into a buffer this device created.
//!
//! The frame suite drives this verb once, as the thing that makes an import
//! possible.  What these tests ask is the narrower question that suite cannot:
//! what the verb lowers to on its own, and which requests it refuses before any
//! driver call.  Both matter because this is the one verb on the adapter that
//! takes host bytes at all -- the execution contract has none, on purpose -- so
//! it is the only place a length, an offset or an allocation is checked against
//! anything other than a graph's compiled requirement.
//!
//! # The two refusals are the same error kind, and that is the point
//!
//! A buffer this device never created and a range past the allocation's end both
//! come back as a validation failure at `upload-buffer`; they are told apart by
//! the sentence and by *when* they are decided.  The first is refused here,
//! against the adapter's own creation record, and the second by the layer that
//! holds the real descriptor -- so a suite has to check which happened rather
//! than only that something did, and the driver-call count is what says which.

use fluxel_rendergraph::{BoundBuffer, BufferDesc, BufferUsage, BufferUsageKind, ExecutionBackend};

use super::super::retention::GlRetentionLease;
use super::{Adapter, adapter, calls, invalid, trace_from_here};
use crate::webgl2::api::{BufferId, MockCall};

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
