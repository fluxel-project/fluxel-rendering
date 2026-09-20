//! Vulkan buffer-transfer lowering.
//!
//! This is the first command vertical slice.  It intentionally owns only
//! buffer-to-buffer copy, buffer upload and buffer readback; every other
//! recorded payload remains a Phase-A `Unsupported` refusal.  Keeping that
//! boundary explicit is what prevents capability facts from getting ahead of
//! actual lowering.

use std::sync::Arc;

use ash::vk;

use crate::api::command::copy::BufferCopy;
use crate::api::resource::buffer::Buffer;
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTicket, UploadDescriptor, UploadJob,
};
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::resource::{VulkanBuffer, VulkanStagingBuffer, create_staging_buffer};

/// Resources whose portable handles have to outlive accepted GPU work.
///
/// Vulkan command buffers retain native handles, not Fluxel handles.  These
/// clones therefore remain in `PendingBatch` until its fence is terminal.
#[derive(Default)]
pub(super) struct TransferRetention {
    pub(super) resources: Vec<Buffer>,
    pub(super) staging: Vec<VulkanStagingBuffer>,
    pub(super) readbacks: Vec<ReadbackRetention>,
}

impl TransferRetention {
    pub(super) fn readback_tickets(&self) -> Vec<ReadbackTicket> {
        self.readbacks
            .iter()
            .map(|retention| retention.ticket.clone())
            .collect()
    }
}

pub(super) struct ReadbackRetention {
    pub(super) staging: VulkanStagingBuffer,
    pub(super) ticket: ReadbackTicket,
}

/// Lowers a direct buffer copy with conservative transfer-domain memory
/// dependencies. Future graphics/compute lowering must replace this local
/// transfer-only state model with a unified resource-state tracker; it must not
/// silently assume this barrier covers shader or attachment accesses.
pub(super) fn lower_buffer_copy(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &BufferCopy,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let source = native_buffer(&copy.src)?;
    let destination = native_buffer(&copy.dst)?;
    transfer_dependency(
        shared,
        command_buffer,
        source.buffer(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
    );
    transfer_dependency(
        shared,
        command_buffer,
        destination.buffer(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::TRANSFER_WRITE,
    );
    let region = vk::BufferCopy::default()
        .src_offset(copy.src_offset)
        .dst_offset(copy.dst_offset)
        .size(copy.size);
    // SAFETY: portable recording validated ownership, COPY usage, ranges, and
    // non-overlap. Both native handles belong to `shared` and the barriers above
    // make preceding transfer writes available to this operation.
    unsafe {
        shared.device.cmd_copy_buffer(
            command_buffer,
            source.buffer(),
            destination.buffer(),
            &[region],
        );
    }
    retention.resources.push(copy.src.clone());
    retention.resources.push(copy.dst.clone());
    Ok(())
}

pub(super) fn lower_upload(
    shared: &Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    job: &UploadJob,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let UploadDescriptor::Buffer(descriptor) = job.descriptor() else {
        return Err(VulkanFailure::Unsupported {
            what: "a texture upload",
            why: "the Vulkan buffer-transfer slice does not lower texture uploads",
        });
    };
    let destination = native_buffer(&descriptor.dst)?;
    let staging = create_staging_buffer(
        Arc::clone(shared),
        descriptor.bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )
    .map_err(native("Vulkan buffer-upload staging allocation"))?;
    staging
        .write(&descriptor.bytes)
        .map_err(native("Vulkan buffer-upload staging map/flush"))?;
    host_write_to_transfer_read(shared, command_buffer, staging.buffer());
    transfer_dependency(
        shared,
        command_buffer,
        destination.buffer(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::TRANSFER_WRITE,
    );
    let region = vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(descriptor.dst_offset)
        .size(descriptor.bytes.len() as u64);
    // SAFETY: the staging allocation remains retained through the batch fence;
    // portable validation checked destination range and COPY_DST usage.
    unsafe {
        shared.device.cmd_copy_buffer(
            command_buffer,
            staging.buffer(),
            destination.buffer(),
            &[region],
        );
    }
    retention.resources.push(descriptor.dst.clone());
    retention.staging.push(staging);
    Ok(())
}

pub(super) fn lower_readback(
    shared: &Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    ticket: &ReadbackTicket,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let ReadbackRequest::Buffer { src, range, .. } = ticket.request() else {
        return Err(VulkanFailure::Unsupported {
            what: "a texture readback",
            why: "the Vulkan buffer-transfer slice does not lower texture readbacks",
        });
    };
    let source = native_buffer(src)?;
    let staging = create_staging_buffer(
        Arc::clone(shared),
        range.size,
        vk::BufferUsageFlags::TRANSFER_DST,
    )
    .map_err(native("Vulkan buffer-readback staging allocation"))?;
    transfer_dependency(
        shared,
        command_buffer,
        source.buffer(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
    );
    let region = vk::BufferCopy::default()
        .src_offset(range.offset)
        .dst_offset(0)
        .size(range.size);
    // SAFETY: portable validation checked the source range and COPY_SRC usage;
    // staging remains retained until this batch's fence completes.
    unsafe {
        shared
            .device
            .cmd_copy_buffer(command_buffer, source.buffer(), staging.buffer(), &[region]);
    }
    transfer_write_to_host_read(shared, command_buffer, staging.buffer());
    retention.resources.push(src.clone());
    retention.readbacks.push(ReadbackRetention {
        staging,
        ticket: ticket.clone(),
    });
    Ok(())
}

/// Turns a fence-complete staging allocation into the ticket's RAII-readable
/// bytes. A terminal map/invalidate failure is returned to the device loss
/// authority; a non-terminal mapping failure terminates only this ticket.
pub(super) fn publish_readback(retention: &ReadbackRetention) -> Result<(), vk::Result> {
    match retention.staging.read() {
        Ok(bytes) => {
            retention.ticket.publish(bytes, None);
            Ok(())
        }
        Err(error) => {
            retention
                .ticket
                .set_status(if error == vk::Result::ERROR_DEVICE_LOST {
                    ReadbackStatus::DeviceLost
                } else {
                    ReadbackStatus::Failed
                });
            Err(error)
        }
    }
}

fn native(operation: &'static str) -> impl FnOnce(vk::Result) -> VulkanFailure {
    move |result| {
        VulkanFailure::Native(crate::backend::vulkan::ffi::NativeError::new(
            result, operation,
        ))
    }
}

fn native_buffer(buffer: &Buffer) -> Result<&VulkanBuffer, VulkanFailure> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<VulkanBuffer>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a non-Vulkan buffer",
            why: "a Vulkan submission may only lower buffers created by the same Vulkan device",
        })
}

fn transfer_dependency(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    // SAFETY: command buffer is recording; all buffers use the selected queue
    // family exclusively, so no queue-family ownership transfer is requested.
    unsafe {
        shared.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

fn host_write_to_transfer_read(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::HOST_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    unsafe {
        shared.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

fn transfer_write_to_host_read(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    unsafe {
        shared.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}
