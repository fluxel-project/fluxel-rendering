//! One immutable portable bind group lowered to Vulkan descriptors.

use std::any::Any;
use std::sync::Arc;

use ash::vk;

use crate::api::binding::backend::BindGroupBackend;
use crate::api::binding::{BindGroupDescriptor, BindingKind, BindingResource};
use crate::api::resource::buffer::BufferBinding;
use crate::api::resource::subresource::TextureAspects;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::resource::{VulkanBuffer, VulkanSampler, VulkanTextureView};

use super::layout::{descriptor_type, layout_bindings};

/// Native immutable descriptor packet behind one portable bind group.
///
/// The pool owns the set and must be destroyed before the layout.  `shared` is
/// the sole native-device ownership domain; it makes that destruction order safe
/// even when the portable handle is retired after `VulkanDevice` itself.
pub(crate) struct VulkanBindGroup {
    shared: Arc<VulkanShared>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
}

impl VulkanBindGroup {
    /// Descriptor set needed by future compute/raster bind commands.
    pub(crate) fn set(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl BindGroupBackend for VulkanBindGroup {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for VulkanBindGroup {
    fn drop(&mut self) {
        unsafe {
            // A descriptor set is pool-owned.  Destroy it first, then release
            // the layout it was allocated against; this remains valid without
            // FREE_DESCRIPTOR_SET because individual set destruction is absent.
            self.shared.device.destroy_descriptor_pool(self.pool, None);
            self.shared
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

/// Creates one dedicated descriptor pool/layout/set and writes its immutable
/// resource references.  Any native failure retains its `VkResult` until the
/// owning device's loss authority observes it.
pub(in crate::backend::vulkan) fn create_bind_group(
    shared: Arc<VulkanShared>,
    descriptor: &BindGroupDescriptor,
) -> Result<VulkanBindGroup, VulkanFailure> {
    let bindings = layout_bindings(descriptor.layout.descriptor())?;
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let layout = unsafe {
        shared
            .device
            .create_descriptor_set_layout(&layout_info, None)
    }
    .map_err(|result| native(result, "vkCreateDescriptorSetLayout"))?;

    let pool_sizes = pool_sizes(descriptor);
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = match unsafe { shared.device.create_descriptor_pool(&pool_info, None) } {
        Ok(pool) => pool,
        Err(result) => {
            unsafe { shared.device.destroy_descriptor_set_layout(layout, None) };
            return Err(native(result, "vkCreateDescriptorPool"));
        }
    };

    let layouts = [layout];
    let allocate = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = match unsafe { shared.device.allocate_descriptor_sets(&allocate) } {
        Ok(mut sets) => match sets.pop() {
            Some(set) => set,
            None => {
                unsafe {
                    shared.device.destroy_descriptor_pool(pool, None);
                    shared.device.destroy_descriptor_set_layout(layout, None);
                }
                return Err(VulkanFailure::Unsupported {
                    what: "Vulkan descriptor-set allocation",
                    why: "the driver returned no descriptor set for one requested layout",
                });
            }
        },
        Err(result) => {
            unsafe {
                shared.device.destroy_descriptor_pool(pool, None);
                shared.device.destroy_descriptor_set_layout(layout, None);
            }
            return Err(native(result, "vkAllocateDescriptorSets"));
        }
    };

    if let Err(error) = write_descriptor_set(&shared, set, descriptor) {
        unsafe {
            shared.device.destroy_descriptor_pool(pool, None);
            shared.device.destroy_descriptor_set_layout(layout, None);
        }
        return Err(error);
    }

    Ok(VulkanBindGroup {
        shared,
        pool,
        layout,
        set,
    })
}

fn pool_sizes(descriptor: &BindGroupDescriptor) -> Vec<vk::DescriptorPoolSize> {
    // Five portable kinds map one-to-one to five descriptor types. Keeping the
    // accumulation explicit avoids relying on enum integer values as map keys.
    let mut uniform = 0;
    let mut storage_buffer = 0;
    let mut sampled_image = 0;
    let mut storage_image = 0;
    let mut sampler = 0;
    for entry in &descriptor.layout.descriptor().entries {
        let count = entry.count.elements();
        match descriptor_type(&entry.kind) {
            vk::DescriptorType::UNIFORM_BUFFER => uniform += count,
            vk::DescriptorType::STORAGE_BUFFER => storage_buffer += count,
            vk::DescriptorType::SAMPLED_IMAGE => sampled_image += count,
            vk::DescriptorType::STORAGE_IMAGE => storage_image += count,
            vk::DescriptorType::SAMPLER => sampler += count,
            _ => unreachable!("portable binding kinds have closed Vulkan descriptor mapping"),
        }
    }
    [
        (vk::DescriptorType::UNIFORM_BUFFER, uniform),
        (vk::DescriptorType::STORAGE_BUFFER, storage_buffer),
        (vk::DescriptorType::SAMPLED_IMAGE, sampled_image),
        (vk::DescriptorType::STORAGE_IMAGE, storage_image),
        (vk::DescriptorType::SAMPLER, sampler),
    ]
    .into_iter()
    .filter(|(_, count)| *count != 0)
    .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
        ty,
        descriptor_count,
    })
    .collect()
}

fn write_descriptor_set(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    descriptor: &BindGroupDescriptor,
) -> Result<(), VulkanFailure> {
    // `VkWriteDescriptorSet` borrows its info arrays, so construct and submit
    // each entry independently. A group is immutable, and update calls are
    // void in Vulkan; all fallible work occurred before this point.
    for entry in &descriptor.entries {
        let Some(layout_entry) = descriptor.layout.slot(entry.slot) else {
            return Err(VulkanFailure::Unsupported {
                what: "Vulkan bind-group packet",
                why: "portable validation did not leave a layout slot for an entry",
            });
        };
        write_entry(
            shared,
            set,
            entry.slot.get(),
            &layout_entry.kind,
            &entry.resource,
        )?;
    }
    Ok(())
}

fn write_entry(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    kind: &BindingKind,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    match kind {
        BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. } => {
            write_buffers(shared, set, binding, descriptor_type(kind), resource)
        }
        BindingKind::SampledTexture { .. } | BindingKind::StorageTexture { .. } => {
            write_images(shared, set, binding, descriptor_type(kind), resource)
        }
        BindingKind::Sampler { .. } => write_samplers(shared, set, binding, resource),
        BindingKind::AccelerationStructure | BindingKind::ExternalTexture => {
            Err(VulkanFailure::Unsupported {
                what: "a Vulkan acceleration-structure or external-texture descriptor",
                why: "the descriptor layout gate must reject this unavailable Vulkan extension path",
            })
        }
    }
}

fn write_buffers(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    descriptor_type: vk::DescriptorType,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    let one;
    let bindings: &[BufferBinding] = match resource {
        BindingResource::Buffer(value) => {
            one = [value.clone()];
            &one
        }
        BindingResource::BufferArray(values) => values,
        _ => return packet_mismatch("buffer", "a non-buffer resource"),
    };
    let mut infos = Vec::with_capacity(bindings.len());
    for value in bindings {
        let Some(buffer) = value
            .buffer
            .native()
            .as_any()
            .downcast_ref::<VulkanBuffer>()
        else {
            return packet_mismatch("Vulkan buffer", "a resource from another backend");
        };
        infos.push(
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer())
                .offset(value.range.offset)
                .range(value.range.size),
        );
    }
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(descriptor_type)
        .buffer_info(&infos);
    unsafe { shared.device.update_descriptor_sets(&[write], &[]) };
    Ok(())
}

fn write_images(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    descriptor_type: vk::DescriptorType,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    let one;
    let views: &[crate::api::resource::TextureView] = match resource {
        BindingResource::Texture(value) => {
            one = [value.clone()];
            &one
        }
        BindingResource::TextureArray(values) => values,
        _ => return packet_mismatch("texture", "a non-texture resource"),
    };
    let mut infos = Vec::with_capacity(views.len());
    for value in views {
        let Some(view) = value.native().as_any().downcast_ref::<VulkanTextureView>() else {
            return packet_mismatch("Vulkan texture view", "a resource from another backend");
        };
        infos.push(
            vk::DescriptorImageInfo::default()
                .image_view(view.view())
                .image_layout(descriptor_image_layout(
                    descriptor_type,
                    value.descriptor().aspects,
                )),
        );
    }
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(descriptor_type)
        .image_info(&infos);
    unsafe { shared.device.update_descriptor_sets(&[write], &[]) };
    Ok(())
}

/// The layout recorded in an immutable image descriptor.
///
/// Command lowering must use this exact rule before binding the descriptor.
/// In particular, depth/stencil sampled views need Vulkan's depth/stencil
/// read-only layout rather than the color-image shader-read layout.  Keeping
/// the mapping here prevents descriptor metadata and image barriers from
/// quietly drifting apart as more command paths are added.
pub(crate) fn descriptor_image_layout(
    descriptor_type: vk::DescriptorType,
    aspects: TextureAspects,
) -> vk::ImageLayout {
    match descriptor_type {
        vk::DescriptorType::SAMPLED_IMAGE => {
            if aspects.contains(TextureAspects::DEPTH) || aspects.contains(TextureAspects::STENCIL)
            {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            }
        }
        vk::DescriptorType::STORAGE_IMAGE => vk::ImageLayout::GENERAL,
        _ => unreachable!("image-layout mapping only receives portable image descriptor types"),
    }
}

fn write_samplers(
    shared: &VulkanShared,
    set: vk::DescriptorSet,
    binding: u32,
    resource: &BindingResource,
) -> Result<(), VulkanFailure> {
    let one;
    let samplers: &[crate::api::resource::Sampler] = match resource {
        BindingResource::Sampler(value) => {
            one = [value.clone()];
            &one
        }
        BindingResource::SamplerArray(values) => values,
        _ => return packet_mismatch("sampler", "a non-sampler resource"),
    };
    let mut infos = Vec::with_capacity(samplers.len());
    for value in samplers {
        let Some(sampler) = value.native().as_any().downcast_ref::<VulkanSampler>() else {
            return packet_mismatch("Vulkan sampler", "a resource from another backend");
        };
        infos.push(vk::DescriptorImageInfo::default().sampler(sampler.sampler()));
    }
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .image_info(&infos);
    unsafe { shared.device.update_descriptor_sets(&[write], &[]) };
    Ok(())
}

fn packet_mismatch<T>(expected: &'static str, actual: &'static str) -> Result<T, VulkanFailure> {
    Err(VulkanFailure::Unsupported {
        what: expected,
        why: actual,
    })
}

fn native(result: vk::Result, operation: &'static str) -> VulkanFailure {
    VulkanFailure::Native(ffi::NativeError::new(result, operation))
}
