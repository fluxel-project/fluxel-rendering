use crate::api::resource::buffer::ResourceMemoryPreference;
use crate::backend::vulkan::platform::device::VulkanShared;
use ash::vk;

/// The portable preference never makes DEVICE_LOCAL a correctness requirement.
pub(super) fn memory_type(
    shared: &VulkanShared,
    bits: u32,
    preference: ResourceMemoryPreference,
) -> Option<u32> {
    let properties = &shared.memory_properties;
    if matches!(preference, ResourceMemoryPreference::DeviceLocalPreferred) {
        for index in 0..properties.memory_type_count {
            if bits & (1 << index) != 0
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            {
                return Some(index);
            }
        }
    }
    (0..properties.memory_type_count).find(|&index| bits & (1 << index) != 0)
}
