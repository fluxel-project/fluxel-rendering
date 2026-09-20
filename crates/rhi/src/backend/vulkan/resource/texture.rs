use super::memory::memory_type;
use crate::api::resource::{
    backend::TextureBackend,
    texture::{TextureDescriptor, TextureDimension, TextureUsage, TextureViewCompatibility},
};
use crate::backend::vulkan::format::vk_format;
use crate::backend::vulkan::platform::device::VulkanShared;
use ash::vk;
use std::{any::Any, sync::Arc};

pub(crate) struct VulkanTexture {
    shared: Arc<VulkanShared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    format: vk::Format,
}
impl VulkanTexture {
    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }
    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }
}
impl TextureBackend for VulkanTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Drop for VulkanTexture {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_image(self.image, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

pub(crate) fn create_texture(
    shared: Arc<VulkanShared>,
    desc: &TextureDescriptor,
) -> Result<VulkanTexture, vk::Result> {
    let Some(format) = vk_format(desc.format) else {
        return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
    };
    let mut flags = vk::ImageCreateFlags::empty();
    if desc
        .view_compatibility
        .contains(TextureViewCompatibility::CUBE)
    {
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    // Vulkan requires MUTABLE_FORMAT at image creation before any view may use
    // a format other than the image's own. Portable capability validation has
    // already proved every declared pair view-compatible; this flag grants the
    // native permission and does not broaden that public set.
    if !desc.view_formats.is_empty() {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    let info = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(match desc.dimension {
            TextureDimension::D1 => vk::ImageType::TYPE_1D,
            TextureDimension::D2 => vk::ImageType::TYPE_2D,
            TextureDimension::D3 => vk::ImageType::TYPE_3D,
        })
        .format(format)
        .extent(vk::Extent3D {
            width: desc.extent.width,
            height: desc.extent.height,
            depth: desc.extent.depth,
        })
        .mip_levels(desc.mip_levels)
        .array_layers(desc.array_layers)
        .samples(vk::SampleCountFlags::from_raw(desc.sample_count))
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage(desc.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { shared.device.create_image(&info, None) }?;
    let requirements = unsafe { shared.device.get_image_memory_requirements(image) };
    let Some(memory_type_index) = memory_type(&shared, requirements.memory_type_bits, desc.memory)
    else {
        unsafe { shared.device.destroy_image(image, None) };
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    };
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = match unsafe { shared.device.allocate_memory(&allocation, None) } {
        Ok(value) => value,
        Err(error) => {
            unsafe { shared.device.destroy_image(image, None) };
            return Err(error);
        }
    };
    if let Err(error) = unsafe { shared.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            shared.device.free_memory(memory, None);
            shared.device.destroy_image(image, None);
        }
        return Err(error);
    }
    Ok(VulkanTexture {
        shared,
        image,
        memory,
        format,
    })
}
fn usage(value: TextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    if value.contains(TextureUsage::COPY_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if value.contains(TextureUsage::COPY_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if value.contains(TextureUsage::SAMPLED) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if value.contains(TextureUsage::STORAGE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if value.contains(TextureUsage::COLOR_ATTACHMENT) {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if value.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    flags
}
