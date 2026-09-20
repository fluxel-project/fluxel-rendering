use super::memory::memory_type;
use crate::api::resource::{
    backend::BufferBackend,
    buffer::{BufferDescriptor, BufferUsage},
};
use crate::backend::vulkan::platform::device::VulkanShared;
use ash::vk;
use std::{any::Any, sync::Arc};

pub(crate) struct VulkanBuffer {
    shared: Arc<VulkanShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}
impl VulkanBuffer {
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
    let Some(memory_type_index) = memory_type(&shared, requirements.memory_type_bits, desc.memory)
    else {
        unsafe { shared.device.destroy_buffer(buffer, None) };
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    };
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
    if value.contains(BufferUsage::COPY_DST) {
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
    flags
}
