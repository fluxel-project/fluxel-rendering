//! Lowering the two transfer directions: host bytes into a buffer, and a buffer
//! back out to the host.
//!
//! The two are mirror images that share a staging heap choice and almost nothing
//! else, which is why they live together and why the retention type below carries
//! both — the difference between them is the difference between a `Vec<Dx12Buffer>`
//! and a ticket, and a reader comparing the pair should see that in one place.
//!
//! # Why the staging outlives the recording
//!
//! `ExecuteCommandLists` is asynchronous, so a staging allocation whose last
//! reference is dropped when recording returns would be freed while the GPU is
//! still copying through it. [`CommittedBatch`] is what holds it until the fence
//! reports the batch finished, and it is the only thing in this chapter that
//! exists for that reason.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    ID3D12Device, ID3D12GraphicsCommandList,
};

use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTicket, UploadDescriptor, UploadJob,
};
use crate::backend::dx12::ffi;
use crate::backend::dx12::resource::{Dx12Buffer, StagingHeap, create_staging, readback_bytes};

use super::dx12_buffer;
use super::transition::Transitions;
use crate::backend::dx12::failure::{Dx12Failure, ref_native};

/// A batch that has been committed, and the host-visible memory its command list
/// reads or writes.
///
/// Both halves are needed for the same reason: `ExecuteCommandLists` is
/// asynchronous, so a staging allocation whose last reference is dropped when
/// recording returns would be freed while the GPU is still copying out of it.
/// Retaining it here until the fence reports the batch finished is what keeps the
/// list's resource references valid for the list's whole life.
pub(super) struct CommittedBatch {
    /// The serial that reports this batch's completion.
    pub(super) serial: u64,
    /// Upload staging: written by the CPU before the commit, read by the GPU
    /// during it, and of no use afterwards.
    pub(super) staging: Vec<Dx12Buffer>,
    /// Readback staging: written by the GPU, read by the CPU once the serial is
    /// reached, and then published to the ticket that asked for it.
    pub(super) readbacks: Vec<ReadbackRetention>,
}

/// A readback's staging buffer and the ticket waiting on it.
pub(super) struct ReadbackRetention {
    /// The `READBACK` heap allocation the GPU copies into.
    pub(super) staging: Dx12Buffer,
    /// The ticket whose bytes these are.
    pub(super) ticket: ReadbackTicket,
    /// How many bytes were copied, which is the range's size rather than the
    /// buffer's.
    pub(super) size: u64,
}

/// Lowers a buffer upload: staging copy from caller bytes, then GPU copy.
///
/// # Why the CPU write happens here and not at `create_buffer_upload`
///
/// The upload heap allocation could have been made and filled when the job
/// was created, since the bytes are already retained and immutable
/// (section 17.2). It is made here instead because the allocation must
/// outlive the *commit*, not the job: a staging buffer created at job
/// creation would be held by the job, which the caller may keep for as long
/// as it likes, and a job encoded into several plans would need one staging
/// buffer per plan anyway. Creating it inside the recording ties its life to
/// the batch that reads it, which is exactly the lifetime the fence reports.
pub(super) fn lower_upload(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    job: &UploadJob,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let UploadDescriptor::Buffer(descriptor) = job.descriptor() else {
        return Err(Dx12Failure::Unsupported {
            what: "a texture upload",
            why: "this spine has no texture lowering at all, so there is no \
                  destination state, no region copy, and no host-layout repacking \
                  to write through",
        });
    };
    let destination = dx12_buffer(&descriptor.dst)?;

    let length = descriptor.bytes.len();
    let staging =
        create_staging(device, length as u64, StagingHeap::Upload).map_err(Dx12Failure::Native)?;

    let mut pointer: *mut core::ffi::c_void = std::ptr::null_mut();
    // SAFETY: `Map` on an `UPLOAD` heap resource makes the whole allocation
    // CPU-writable and writes the address into `pointer`; the null read range
    // is what Direct3D 12 requires for a write-only heap. The mapping stays
    // live until the `Unmap` below.
    unsafe {
        staging
            .resource()
            .Map(0, None, Some(&mut pointer))
            .map_err(|error| ref_native(&error))?;
    }
    let Some(pointer) = std::ptr::NonNull::new(pointer.cast::<u8>()) else {
        return Err(Dx12Failure::Native(
            ffi::NativeError::driver_contract_violation(
                "Map reported success without producing a pointer",
                "Dx12Device::submit",
            ),
        ));
    };
    // SAFETY: the mapping covers `length` bytes because that is the resource's
    // own width, which is what `create_staging` was asked for. The source is
    // the job's retained `Arc<[u8]>`, alive for the whole recording, and host
    // memory and a GPU allocation cannot overlap. `Unmap` follows the copy and
    // is the single matching call for the single mapping above.
    unsafe {
        std::ptr::copy_nonoverlapping(descriptor.bytes.as_ptr(), pointer.as_ptr(), length);
        staging.resource().Unmap(0, None);
    }

    let mut entering = Transitions::default();
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);
    // SAFETY: the staging buffer was created in `GENERIC_READ` and stays
    // there — Direct3D 12 permits no transition in an upload heap — which is
    // a state the copy's source half may be in. The destination was put in
    // `COPY_DEST` by the barrier above, and `dst_offset` plus `length` was
    // validated against the destination's size at job creation (section 17.3).
    unsafe {
        list.CopyBufferRegion(
            destination.resource(),
            descriptor.dst_offset,
            staging.resource(),
            0,
            length as u64,
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);

    committed.staging.push(staging);
    Ok(())
}

/// Lowers a buffer readback: GPU copy into staging, then a ticket the drain
/// publishes from.
///
/// The reverse of [`lower_upload`] in every respect, including which
/// way the staging is retained: upload staging is dead the moment the batch
/// finishes, while readback staging is the thing the batch's completion is
/// *for*.
pub(super) fn lower_readback(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    ticket: &ReadbackTicket,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let ReadbackRequest::Buffer { src, range, .. } = ticket.request() else {
        return Err(Dx12Failure::Unsupported {
            what: "a texture readback",
            why: "this spine has no texture lowering at all, so there is no source \
                  state and no footprint to copy through",
        });
    };
    let source = dx12_buffer(src)?;

    let staging =
        create_staging(device, range.size, StagingHeap::Readback).map_err(Dx12Failure::Native)?;

    let mut entering = Transitions::default();
    entering.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    entering.record(list);
    // SAFETY: the staging buffer was created in `COPY_DEST` and stays there —
    // Direct3D 12 permits no transition in a readback heap — which is a state
    // the copy's destination half may be in. The source was put in
    // `COPY_SOURCE` by the barrier above, and the range was validated against
    // the source's size at record time (section 18.1).
    unsafe {
        list.CopyBufferRegion(
            staging.resource(),
            0,
            source.resource(),
            range.offset,
            range.size,
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);

    committed.readbacks.push(ReadbackRetention {
        staging,
        ticket: ticket.clone(),
        size: range.size,
    });
    Ok(())
}

/// Copies one finished readback's bytes off the GPU and hands them to its ticket.
///
/// A failure here is reported as [`ReadbackStatus::Failed`] rather than as a
/// device loss: mapping a readback heap can fail for reasons that say nothing
/// about the device, and section 18.2 makes `Failed` exactly the terminal state
/// for a backend failure. The ticket carries no message, so the state is the
/// whole report.
pub(super) fn publish_readback(retention: &ReadbackRetention) {
    match readback_bytes(&retention.staging, retention.size) {
        // A buffer range is tightly packed by definition, so there is no texel
        // layout to publish beside its bytes.
        Ok(bytes) => retention.ticket.publish(bytes, None),
        Err(_) => retention.ticket.set_status(ReadbackStatus::Failed),
    }
}
