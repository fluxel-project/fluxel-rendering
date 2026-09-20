use super::memory::memory_type;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::DeviceLossInfo;
use crate::api::resource::{
    BufferRange, MapMode,
    backend::{BufferBackend, MappedBufferBackend, MappingRequestBackend},
    buffer::{BufferDescriptor, BufferUsage},
};
use crate::backend::vulkan::platform::device::VulkanShared;
use ash::vk;
use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
};

pub(crate) struct VulkanBuffer {
    shared: Arc<VulkanShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    allocation_size: u64,
    coherent: bool,
    last_accepted_serial: AtomicU64,
}
impl VulkanBuffer {
    pub(crate) fn mark_accepted(&self, serial: u64) {
        self.last_accepted_serial
            .fetch_max(serial, Ordering::Release);
    }

    fn last_accepted_serial(&self) -> u64 {
        self.last_accepted_serial.load(Ordering::Acquire)
    }
    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
    #[expect(
        dead_code,
        reason = "reserved for binding and copy range validation in Vulkan lowering"
    )]
    pub(crate) fn size(&self) -> u64 {
        self.size
    }
}

/// A one-submit, host-visible transfer allocation.
///
/// This is deliberately separate from a portable `Buffer`: staging is a
/// backend implementation detail and its lifetime is the accepted batch, not a
/// caller-visible resource lifetime.  `VulkanCommandSpine` retains it until its
/// fence becomes terminal, so a GPU transfer can never observe freed host
/// memory.
pub(crate) struct VulkanStagingBuffer {
    shared: Arc<VulkanShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    allocation_size: u64,
    coherent: bool,
}

impl VulkanStagingBuffer {
    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Copies CPU bytes into this upload allocation and makes non-coherent
    /// writes visible to the device before command submission.
    pub(crate) fn write(&self, bytes: &[u8]) -> Result<(), vk::Result> {
        debug_assert_eq!(bytes.len() as u64, self.size);
        let pointer = unsafe {
            self.shared.device.map_memory(
                self.memory,
                0,
                self.allocation_size,
                vk::MemoryMapFlags::empty(),
            )
        }?;
        // SAFETY: the allocation is host visible, mapped over its complete
        // allocation, and `bytes` is exactly the requested staging payload.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.cast(), bytes.len()) };
        let result = if self.coherent {
            Ok(())
        } else {
            // Mapping and flushing the complete dedicated allocation avoids a
            // guessed alignment. VK_WHOLE_SIZE is explicitly valid here and
            // covers the non_coherent_atom_size tail required by Vulkan.
            debug_assert!(self.shared.non_coherent_atom_size > 0);
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe { self.shared.device.flush_mapped_memory_ranges(&[range]) }
        };
        unsafe { self.shared.device.unmap_memory(self.memory) };
        result
    }

    /// Reads a completed download allocation after invalidating any
    /// non-coherent device writes.  The caller must have observed the batch
    /// fence: mapping itself is not GPU synchronization.
    pub(crate) fn read(&self) -> Result<Vec<u8>, vk::Result> {
        let pointer = unsafe {
            self.shared.device.map_memory(
                self.memory,
                0,
                self.allocation_size,
                vk::MemoryMapFlags::empty(),
            )
        }?;
        let result = (|| {
            if !self.coherent {
                debug_assert!(self.shared.non_coherent_atom_size > 0);
                let range = vk::MappedMemoryRange::default()
                    .memory(self.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                unsafe { self.shared.device.invalidate_mapped_memory_ranges(&[range]) }?;
            }
            let mut bytes = vec![
                0;
                usize::try_from(self.size)
                    .map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?
            ];
            // SAFETY: successful mapping covers `allocation_size`, which is at
            // least `size`; the destination Vec owns exactly `size` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(pointer.cast::<u8>(), bytes.as_mut_ptr(), bytes.len())
            };
            Ok(bytes)
        })();
        unsafe { self.shared.device.unmap_memory(self.memory) };
        result
    }
}

impl Drop for VulkanStagingBuffer {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}
impl BufferBackend for VulkanBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Maps a primary host-visible allocation. Vulkan mapping is synchronous once
/// ownership is safe, but the portable future still owns the eventual RAII
/// lease and provides the cross-backend asynchronous vocabulary.
pub(crate) fn map_buffer(
    buffer: &VulkanBuffer,
    mode: MapMode,
    range: BufferRange,
) -> Result<Box<dyn MappingRequestBackend>, vk::Result> {
    Ok(Box::new(VulkanMappingRequest {
        shared: Arc::clone(&buffer.shared),
        wait_serial: buffer.last_accepted_serial(),
        waiter_slot: buffer.shared.mapping_waiter_slot(),
        memory: buffer.memory,
        allocation_size: buffer.allocation_size,
        range,
        writable: matches!(mode, MapMode::Write),
        coherent: buffer.coherent,
        lease: None,
    }))
}

struct VulkanMappingRequest {
    shared: Arc<VulkanShared>,
    wait_serial: u64,
    waiter_slot: u64,
    memory: vk::DeviceMemory,
    allocation_size: u64,
    range: BufferRange,
    writable: bool,
    coherent: bool,
    lease: Option<VulkanMappedBuffer>,
}

impl MappingRequestBackend for VulkanMappingRequest {
    fn poll(&mut self, context: &mut Context<'_>) -> Poll<RhiResult<Box<dyn MappedBufferBackend>>> {
        if self.wait_serial != 0 {
            match self
                .shared
                .map_completion(self.wait_serial, self.waiter_slot, context.waker())
            {
                Ok(true) => self
                    .shared
                    .unregister_mapping_waiter(self.wait_serial, self.waiter_slot),
                Ok(false) => return Poll::Pending,
                Err(info) => {
                    self.shared
                        .unregister_mapping_waiter(self.wait_serial, self.waiter_slot);
                    return Poll::Ready(Err(RhiError::new(
                        RhiErrorKind::DeviceLost,
                        info.message().to_owned(),
                    )));
                }
            }
        }
        if self.lease.is_none() {
            let pointer = unsafe {
                self.shared.device.map_memory(
                    self.memory,
                    0,
                    self.allocation_size,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|result| {
                if result == vk::Result::ERROR_DEVICE_LOST {
                    self.shared.mark_lost(DeviceLossInfo::new(
                        "Vulkan reported VK_ERROR_DEVICE_LOST from vkMapMemory".to_string(),
                    ));
                    return RhiError::new(
                        RhiErrorKind::DeviceLost,
                        "the Vulkan device was lost while mapping a buffer",
                    );
                }
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    format!("vkMapMemory failed: {result:?}"),
                )
            })?;
            let pointer = std::ptr::NonNull::new(pointer.cast::<u8>()).ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "vkMapMemory succeeded without a pointer",
                )
            })?;
            let offset = usize::try_from(self.range.offset).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "mapped offset exceeds host address space",
                )
            })?;
            let length = usize::try_from(self.range.size).map_err(|_| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "mapped length exceeds host address space",
                )
            })?;
            self.lease = Some(VulkanMappedBuffer {
                shared: Arc::clone(&self.shared),
                memory: self.memory,
                pointer: unsafe { pointer.as_ptr().add(offset) },
                length,
                writable: self.writable,
                coherent: self.coherent,
            });
        }
        match self.lease.take() {
            Some(lease) => Poll::Ready(Ok(Box::new(lease))),
            None => Poll::Ready(Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a Vulkan mapping request was polled after producing its lease",
            ))),
        }
    }
}

impl Drop for VulkanMappingRequest {
    fn drop(&mut self) {
        if self.wait_serial != 0 {
            self.shared
                .unregister_mapping_waiter(self.wait_serial, self.waiter_slot);
        }
    }
}

/// A native map lease. Its destructor is the unique matching `vkUnmapMemory`.
struct VulkanMappedBuffer {
    shared: Arc<VulkanShared>,
    memory: vk::DeviceMemory,
    pointer: *mut u8,
    length: usize,
    writable: bool,
    coherent: bool,
}

impl MappedBufferBackend for VulkanMappedBuffer {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer, self.length) }
    }

    fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.writable
            .then(|| unsafe { std::slice::from_raw_parts_mut(self.pointer, self.length) })
    }

    fn flush(&mut self) -> RhiResult<()> {
        if !self.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe { self.shared.device.flush_mapped_memory_ranges(&[range]) }.map_err(
                |result| {
                    if result == vk::Result::ERROR_DEVICE_LOST {
                        self.shared.mark_lost(DeviceLossInfo::new(
                            "Vulkan reported VK_ERROR_DEVICE_LOST from vkFlushMappedMemoryRanges"
                                .to_string(),
                        ));
                        return RhiError::new(
                            RhiErrorKind::DeviceLost,
                            "the Vulkan device was lost while flushing a mapped buffer",
                        );
                    }
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        format!("vkFlushMappedMemoryRanges failed: {result:?}"),
                    )
                },
            )?;
        }
        Ok(())
    }

    fn invalidate(&mut self) -> RhiResult<()> {
        if !self.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe { self.shared.device.invalidate_mapped_memory_ranges(&[range]) }.map_err(
                |result| {
                    if result == vk::Result::ERROR_DEVICE_LOST {
                        self.shared.mark_lost(DeviceLossInfo::new(
                            "Vulkan reported VK_ERROR_DEVICE_LOST from vkInvalidateMappedMemoryRanges"
                                .to_string(),
                        ));
                        return RhiError::new(
                            RhiErrorKind::DeviceLost,
                            "the Vulkan device was lost while invalidating a mapped buffer",
                        );
                    }
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        format!("vkInvalidateMappedMemoryRanges failed: {result:?}"),
                    )
                },
            )?;
        }
        Ok(())
    }
}

impl Drop for VulkanMappedBuffer {
    fn drop(&mut self) {
        unsafe { self.shared.device.unmap_memory(self.memory) };
    }
}
impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

pub(crate) fn create_buffer(
    shared: Arc<VulkanShared>,
    desc: &BufferDescriptor,
) -> Result<VulkanBuffer, vk::Result> {
    let info = vk::BufferCreateInfo::default()
        .size(desc.size)
        .usage(usage(desc.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { shared.device.create_buffer(&info, None) }?;
    let requirements = unsafe { shared.device.get_buffer_memory_requirements(buffer) };
    let mapped =
        desc.usage.contains(BufferUsage::MAP_READ) || desc.usage.contains(BufferUsage::MAP_WRITE);
    let memory_type_index = if mapped {
        host_visible_memory_type(&shared, requirements.memory_type_bits)
    } else {
        memory_type(&shared, requirements.memory_type_bits, desc.memory)
    };
    let Some(memory_type_index) = memory_type_index else {
        unsafe { shared.device.destroy_buffer(buffer, None) };
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    };
    let coherent = shared.memory_properties.memory_types[memory_type_index as usize]
        .property_flags
        .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = match unsafe { shared.device.allocate_memory(&allocation, None) } {
        Ok(value) => value,
        Err(error) => {
            unsafe { shared.device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    if let Err(error) = unsafe { shared.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            shared.device.free_memory(memory, None);
            self::destroy(&shared, buffer);
        }
        return Err(error);
    }
    Ok(VulkanBuffer {
        shared,
        buffer,
        memory,
        size: desc.size,
        allocation_size: requirements.size,
        coherent,
        last_accepted_serial: AtomicU64::new(0),
    })
}

/// Creates a dedicated host-visible transfer buffer.  Requiring HOST_VISIBLE
/// is the correctness floor; HOST_COHERENT is only an optimization and is never
/// assumed to exist on an otherwise conforming Vulkan implementation.
pub(crate) fn create_staging_buffer(
    shared: Arc<VulkanShared>,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<VulkanStagingBuffer, vk::Result> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { shared.device.create_buffer(&info, None) }?;
    let requirements = unsafe { shared.device.get_buffer_memory_requirements(buffer) };
    let Some(memory_type_index) = host_visible_memory_type(&shared, requirements.memory_type_bits)
    else {
        unsafe { shared.device.destroy_buffer(buffer, None) };
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    };
    let flags = shared.memory_properties.memory_types[memory_type_index as usize].property_flags;
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = match unsafe { shared.device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { shared.device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    if let Err(error) = unsafe { shared.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            shared.device.free_memory(memory, None);
            shared.device.destroy_buffer(buffer, None);
        }
        return Err(error);
    }
    Ok(VulkanStagingBuffer {
        shared,
        buffer,
        memory,
        size,
        allocation_size: requirements.size,
        coherent: flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
    })
}

fn host_visible_memory_type(shared: &VulkanShared, bits: u32) -> Option<u32> {
    let properties = &shared.memory_properties;
    // Prefer coherent memory, but correctness includes non-coherent memory with
    // the explicit flush/invalidate path above.
    for coherent in [true, false] {
        for index in 0..properties.memory_type_count {
            if bits & (1 << index) == 0 {
                continue;
            }
            let flags = properties.memory_types[index as usize].property_flags;
            if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && (!coherent || flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT))
            {
                return Some(index);
            }
        }
    }
    None
}
fn destroy(shared: &VulkanShared, buffer: vk::Buffer) {
    unsafe { shared.device.destroy_buffer(buffer, None) }
}
fn usage(value: BufferUsage) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::empty();
    if value.contains(BufferUsage::COPY_SRC) {
        flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if value.contains(BufferUsage::COPY_DST) || value.contains(BufferUsage::QUERY_RESOLVE) {
        flags |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if value.contains(BufferUsage::VERTEX) {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if value.contains(BufferUsage::INDEX) {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if value.contains(BufferUsage::UNIFORM) {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if value.contains(BufferUsage::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    // MAP_* selects host-visible memory rather than a Vulkan buffer-usage
    // flag. Keep it out of this native mask; Vulkan exposes no MAP usage bit.
    flags
}
