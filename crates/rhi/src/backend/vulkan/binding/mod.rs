//! Backend-private Vulkan descriptor-set packets.
//!
//! The portable `BindGroup` remains the logical owner of its resources.  This
//! module only materializes that immutable packet as one descriptor-set layout,
//! one descriptor pool, and one descriptor set.  Keeping the mapping here gives
//! both packet creation and the future pipeline-layout lowering one authority
//! for the `BindingSlot -> VkDescriptorSetLayoutBinding` translation.

mod group;
mod layout;

pub(in crate::backend::vulkan) use group::{
    VulkanBindGroup, create_bind_group, descriptor_image_layout,
};
pub(in crate::backend::vulkan) use layout::layout_bindings;
