//! Vulkan names for the finite v13 texture-format vocabulary.
//!
//! Capability probing and resource creation share this table. Keeping one
//! mapping authority is correctness-critical: a probe performed with one
//! `VkFormat` and an image created with another would publish a capability the
//! backend did not actually test.

use ash::vk;

use crate::api::format::TextureFormat;

/// Every portable format the current dedicated image creator can name.
pub(super) const FORMATS: &[TextureFormat] = &[
    TextureFormat::R8Unorm,
    TextureFormat::R8Snorm,
    TextureFormat::R8Uint,
    TextureFormat::R8Sint,
    TextureFormat::Rg8Unorm,
    TextureFormat::Rg8Snorm,
    TextureFormat::Rg8Uint,
    TextureFormat::Rg8Sint,
    TextureFormat::Rgba8Unorm,
    TextureFormat::Rgba8UnormSrgb,
    TextureFormat::Rgba8Snorm,
    TextureFormat::Rgba8Uint,
    TextureFormat::Rgba8Sint,
    TextureFormat::Bgra8Unorm,
    TextureFormat::Bgra8UnormSrgb,
    TextureFormat::R16Uint,
    TextureFormat::R16Sint,
    TextureFormat::R16Float,
    TextureFormat::Rg16Uint,
    TextureFormat::Rg16Sint,
    TextureFormat::Rg16Float,
    TextureFormat::Rgba16Uint,
    TextureFormat::Rgba16Sint,
    TextureFormat::Rgba16Float,
    TextureFormat::R32Uint,
    TextureFormat::R32Sint,
    TextureFormat::R32Float,
    TextureFormat::Rg32Uint,
    TextureFormat::Rg32Sint,
    TextureFormat::Rg32Float,
    TextureFormat::Rgba32Uint,
    TextureFormat::Rgba32Sint,
    TextureFormat::Rgba32Float,
    TextureFormat::Depth16Unorm,
    TextureFormat::Depth32Float,
    TextureFormat::Depth32FloatStencil8,
];

pub(super) fn vk_format(value: TextureFormat) -> Option<vk::Format> {
    Some(match value {
        TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        TextureFormat::R8Snorm => vk::Format::R8_SNORM,
        TextureFormat::R8Uint => vk::Format::R8_UINT,
        TextureFormat::R8Sint => vk::Format::R8_SINT,
        TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        TextureFormat::Rg8Snorm => vk::Format::R8G8_SNORM,
        TextureFormat::Rg8Uint => vk::Format::R8G8_UINT,
        TextureFormat::Rg8Sint => vk::Format::R8G8_SINT,
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Rgba8Snorm => vk::Format::R8G8B8A8_SNORM,
        TextureFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
        TextureFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::R16Uint => vk::Format::R16_UINT,
        TextureFormat::R16Sint => vk::Format::R16_SINT,
        TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        TextureFormat::Rg16Uint => vk::Format::R16G16_UINT,
        TextureFormat::Rg16Sint => vk::Format::R16G16_SINT,
        TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        TextureFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
        TextureFormat::Rgba16Sint => vk::Format::R16G16B16A16_SINT,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::R32Uint => vk::Format::R32_UINT,
        TextureFormat::R32Sint => vk::Format::R32_SINT,
        TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        TextureFormat::Rg32Uint => vk::Format::R32G32_UINT,
        TextureFormat::Rg32Sint => vk::Format::R32G32_SINT,
        TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        TextureFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        TextureFormat::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        TextureFormat::Depth16Unorm => vk::Format::D16_UNORM,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        TextureFormat::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
        TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => return None,
    })
}
