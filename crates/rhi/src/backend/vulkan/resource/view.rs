use super::texture::VulkanTexture;
use crate::api::resource::{
    backend::TextureViewBackend,
    subresource::TextureAspects,
    texture::TextureDescriptor,
    view::{TextureViewDescriptor, TextureViewDimension},
};
use crate::backend::vulkan::format::vk_format;
use crate::backend::vulkan::platform::device::VulkanShared;
use ash::vk;
use std::{any::Any, sync::Arc};
pub(crate) struct VulkanTextureView {
    shared: Arc<VulkanShared>,
    view: vk::ImageView,
}
impl VulkanTextureView {
    #[expect(dead_code, reason = "reserved for Vulkan descriptor-set lowering")]
    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }
}
impl TextureViewBackend for VulkanTextureView {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Drop for VulkanTextureView {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_image_view(self.view, None) }
    }
}
pub(crate) fn create_texture_view(
    shared: Arc<VulkanShared>,
    texture: &VulkanTexture,
    _base: &TextureDescriptor,
    desc: &TextureViewDescriptor,
) -> Result<VulkanTextureView, vk::Result> {
    let native_format = match desc.format {
        Some(value) => vk_format(value).ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?,
        None => texture.format(),
    };
    let info = vk::ImageViewCreateInfo::default()
        .image(texture.image())
        .view_type(match desc.dimension {
            TextureViewDimension::D1 => vk::ImageViewType::TYPE_1D,
            TextureViewDimension::D2 => vk::ImageViewType::TYPE_2D,
            TextureViewDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            TextureViewDimension::Cube => vk::ImageViewType::CUBE,
            TextureViewDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
            TextureViewDimension::D3 => vk::ImageViewType::TYPE_3D,
        })
        .format(native_format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspects(desc.aspects),
            base_mip_level: desc.base_mip,
            level_count: desc.mip_count,
            base_array_layer: desc.base_layer,
            layer_count: desc.layer_count,
        });
    let view = unsafe { shared.device.create_image_view(&info, None) }?;
    Ok(VulkanTextureView { shared, view })
}
fn aspects(value: TextureAspects) -> vk::ImageAspectFlags {
    let mut flags = vk::ImageAspectFlags::empty();
    if value.contains(TextureAspects::COLOR) {
        flags |= vk::ImageAspectFlags::COLOR;
    }
    if value.contains(TextureAspects::DEPTH) {
        flags |= vk::ImageAspectFlags::DEPTH;
    }
    if value.contains(TextureAspects::STENCIL) {
        flags |= vk::ImageAspectFlags::STENCIL;
    }
    flags
}
