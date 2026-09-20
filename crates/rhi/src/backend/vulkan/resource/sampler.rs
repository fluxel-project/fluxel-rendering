use crate::api::resource::{
    backend::SamplerBackend,
    sampler::{AddressMode, CompareFunction, FilterMode, SamplerDescriptor},
};
use crate::backend::vulkan::platform::device::VulkanShared;
use ash::vk;
use std::{any::Any, sync::Arc};
pub(crate) struct VulkanSampler {
    shared: Arc<VulkanShared>,
    sampler: vk::Sampler,
}
impl VulkanSampler {
    pub(crate) fn sampler(&self) -> vk::Sampler {
        self.sampler
    }
}
impl SamplerBackend for VulkanSampler {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Drop for VulkanSampler {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_sampler(self.sampler, None) }
    }
}
pub(crate) fn create_sampler(
    shared: Arc<VulkanShared>,
    desc: &SamplerDescriptor,
) -> Result<VulkanSampler, vk::Result> {
    // The logical device does not enable VkPhysicalDeviceFeatures::samplerAnisotropy
    // and its capability facts therefore reject values above one before this
    // seam. Keep the native guard as a closure tripwire: enabling the descriptor
    // bit without enabling the feature would be invalid Vulkan.
    if desc.max_anisotropy > 1 {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(filter(desc.mag_filter))
        .min_filter(filter(desc.min_filter))
        .mipmap_mode(mipmap(desc.mip_filter))
        .address_mode_u(address(desc.address_u))
        .address_mode_v(address(desc.address_v))
        .address_mode_w(address(desc.address_w))
        .mip_lod_bias(0.0)
        .anisotropy_enable(false)
        .max_anisotropy(desc.max_anisotropy as f32)
        .compare_enable(desc.compare.is_some())
        .compare_op(compare(desc.compare))
        .min_lod(desc.lod_min)
        .max_lod(desc.lod_max)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .unnormalized_coordinates(false);
    let sampler = unsafe { shared.device.create_sampler(&info, None) }?;
    Ok(VulkanSampler { shared, sampler })
}
fn filter(value: FilterMode) -> vk::Filter {
    match value {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}
fn mipmap(value: FilterMode) -> vk::SamplerMipmapMode {
    match value {
        FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}
fn address(value: AddressMode) -> vk::SamplerAddressMode {
    match value {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}
fn compare(value: Option<CompareFunction>) -> vk::CompareOp {
    match value {
        None | Some(CompareFunction::Never) => vk::CompareOp::NEVER,
        Some(CompareFunction::Less) => vk::CompareOp::LESS,
        Some(CompareFunction::Equal) => vk::CompareOp::EQUAL,
        Some(CompareFunction::LessEqual) => vk::CompareOp::LESS_OR_EQUAL,
        Some(CompareFunction::Greater) => vk::CompareOp::GREATER,
        Some(CompareFunction::NotEqual) => vk::CompareOp::NOT_EQUAL,
        Some(CompareFunction::GreaterEqual) => vk::CompareOp::GREATER_OR_EQUAL,
        Some(CompareFunction::Always) => vk::CompareOp::ALWAYS,
    }
}
