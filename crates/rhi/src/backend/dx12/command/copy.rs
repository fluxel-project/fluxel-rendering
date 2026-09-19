//! Lowering a buffer-to-buffer copy.
//!
//! The chapter's simplest lowering, and the one that establishes the shape the
//! others follow: bracket the command with the transitions that put each resource
//! in the state Direct3D 12 requires, then return both to `COMMON` so the
//! module-level invariant holds for whatever records next.
//!
//! A free function rather than a method on
//! [`Dx12CommandSpine`](crate::backend::dx12::command::Dx12CommandSpine),
//! because a copy needs nothing from the spine: not the queue, not the fence, not
//! the device. Saying so in the signature is what keeps a reader from having to
//! check whether this step quietly touches submission state.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    ID3D12GraphicsCommandList,
};

use crate::api::command::copy::BufferCopy;

use super::dx12_buffer;
use super::failure::SpineFailure;
use super::transition::Transitions;

/// Lowers a buffer-to-buffer copy.
///
/// Both resources are named in a barrier out of `COMMON` and back into it, which
/// is the invariant the chapter documentation states. `CopyBufferRegion` itself
/// does not transition anything: Direct3D 12 requires a copy's source to be in
/// `COPY_SOURCE` and its destination in `COPY_DEST` when the list executes, and
/// the two barriers are how that becomes true.
///
/// # Errors
///
/// [`SpineFailure::Unsupported`] when either buffer's native allocation belongs
/// to another backend, which is unreachable for a plan this device accepted.
pub(super) fn lower_buffer_copy(
    list: &ID3D12GraphicsCommandList,
    copy: &BufferCopy,
) -> Result<(), SpineFailure> {
    let source = dx12_buffer(&copy.src)?;
    let destination = dx12_buffer(&copy.dst)?;

    let mut entering = Transitions::default();
    entering.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);

    // SAFETY: both resources are alive for at least as long as this call, the
    // two offsets and the size were validated against them at record time
    // (section 34), and the barriers immediately above and below put each
    // resource in the state the copy requires.
    unsafe {
        list.CopyBufferRegion(
            destination.resource(),
            copy.dst_offset,
            source.resource(),
            copy.src_offset,
            copy.size,
        );
    }

    let mut leaving = Transitions::default();
    leaving.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    Ok(())
}
