//! The single Vulkan mapping authority for a portable bind-group layout.
//!
//! A Vulkan descriptor-set layout has a native object identity, while the
//! portable `BindGroupLayout` intentionally does not.  The first baseline gives
//! every immutable group its own `VkDescriptorSetLayout`; a future pipeline
//! layout must call this function again (or cache its result) rather than grow a
//! second, subtly different mapping.

use ash::vk;

use crate::api::binding::{BindGroupLayoutDescriptor, BindingKind};
use crate::api::shader::ShaderStages;
use crate::backend::vulkan::failure::VulkanFailure;

/// Maps one validated portable layout to Vulkan layout bindings in canonical
/// slot order.  Dynamic offsets are deliberately closed for this vertical slice:
/// accepting them here would make later command lowering silently bind wrong
/// buffer ranges.
pub(in crate::backend::vulkan) fn layout_bindings(
    descriptor: &BindGroupLayoutDescriptor,
) -> Result<Vec<vk::DescriptorSetLayoutBinding<'static>>, VulkanFailure> {
    let mut bindings = Vec::with_capacity(descriptor.entries.len());
    for entry in &descriptor.entries {
        if entry.dynamic_offset {
            return Err(VulkanFailure::Unsupported {
                what: "Vulkan bind-group dynamic offsets",
                why: "this baseline has no dynamic-offset command lowering",
            });
        }
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(entry.slot.get())
                .descriptor_type(descriptor_type(&entry.kind))
                .descriptor_count(entry.count.elements())
                .stage_flags(shader_stages(entry.visibility)),
        );
    }
    Ok(bindings)
}

pub(crate) fn descriptor_type(kind: &BindingKind) -> vk::DescriptorType {
    match kind {
        BindingKind::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER,
        BindingKind::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
        BindingKind::SampledTexture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
        BindingKind::Sampler { .. } => vk::DescriptorType::SAMPLER,
    }
}

fn shader_stages(stages: ShaderStages) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if stages.contains(ShaderStages::VERTEX) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if stages.contains(ShaderStages::FRAGMENT) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if stages.contains(ShaderStages::COMPUTE) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    flags
}
