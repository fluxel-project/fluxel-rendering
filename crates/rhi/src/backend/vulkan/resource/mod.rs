//! Dedicated Vulkan resource backing. Each allocation owns one `VkDeviceMemory`.
//! Aliasing is deliberately not advertised until submission can lower its barriers.
mod buffer;
mod memory;
mod query;
mod sampler;
mod texture;
mod view;
pub(crate) use buffer::{
    VulkanBuffer, VulkanStagingBuffer, create_buffer, create_staging_buffer, map_buffer,
};
pub(in crate::backend::vulkan) use query::{VulkanQuerySet, create_query_set};
pub(crate) use sampler::{VulkanSampler, create_sampler};
pub(crate) use texture::{VulkanTexture, create_texture};
pub(crate) use view::{VulkanTextureView, create_texture_view};
