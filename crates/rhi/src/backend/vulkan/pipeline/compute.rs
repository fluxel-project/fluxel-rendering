//! Vulkan compute-pipeline lowering.
//!
//! The portable layer has already proved that the shader, interface and device
//! identities agree.  This module supplies only the native ABI: one descriptor
//! set layout for every ordered portable interface group, one pipeline layout,
//! and one compute pipeline.  Descriptor-set layouts intentionally use the
//! same `layout_bindings` authority as immutable bind groups, so the two Vulkan
//! objects cannot drift into different slot/type/stage mappings.

use std::any::Any;
use std::ffi::CString;
use std::sync::Arc;

use ash::vk;

use crate::api::pipeline::ComputePipelineDescriptor;
use crate::api::pipeline::backend::ComputePipelineBackend;
use crate::backend::vulkan::binding::layout_bindings;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::shader::VulkanShaderModule;

/// Native state behind one portable compute pipeline.
///
/// `shared` is the sole native-device ownership domain.  The pipeline owns the
/// layout and its set-layout children, so it is valid for a portable shader or
/// bind-group handle to disappear after pipeline construction.
pub(crate) struct VulkanComputePipeline {
    shared: Arc<VulkanShared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl VulkanComputePipeline {
    pub(crate) fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl ComputePipelineBackend for VulkanComputePipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for VulkanComputePipeline {
    fn drop(&mut self) {
        unsafe {
            // Vulkan children must be released before their parents.  The
            // shared device outlives all three native object classes.
            self.shared.device.destroy_pipeline(self.pipeline, None);
            self.shared
                .device
                .destroy_pipeline_layout(self.layout, None);
            for layout in self.set_layouts.drain(..) {
                self.shared
                    .device
                    .destroy_descriptor_set_layout(layout, None);
            }
        }
    }
}

/// Builds a real `VkPipeline` from a portable, already-validated compute
/// descriptor.  No fallback shader representation is accepted: Vulkan's first
/// slice consumes the SPIR-V module created by this same backend.
pub(in crate::backend::vulkan) fn create_compute_pipeline(
    shared: Arc<VulkanShared>,
    descriptor: &ComputePipelineDescriptor,
) -> Result<VulkanComputePipeline, VulkanFailure> {
    let shader = descriptor
        .shader
        .native()
        .as_any()
        .downcast_ref::<VulkanShaderModule>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a compute shader from another backend",
            why: "a Vulkan compute pipeline requires a Vulkan SPIR-V shader module",
        })?;
    let entry = CString::new(descriptor.shader.artifact().entry_point.as_str()).map_err(|_| {
        VulkanFailure::Unsupported {
            what: "a compute shader entry-point name containing a NUL byte",
            why: "Vulkan entry-point names are NUL-terminated C strings",
        }
    })?;

    let mut set_layouts = Vec::with_capacity(descriptor.interface.descriptor().groups.len());
    for group in &descriptor.interface.descriptor().groups {
        let bindings = layout_bindings(group.descriptor())?;
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        match unsafe { shared.device.create_descriptor_set_layout(&info, None) } {
            Ok(layout) => set_layouts.push(layout),
            Err(result) => {
                destroy_set_layouts(&shared, &mut set_layouts);
                return Err(native(
                    result,
                    "vkCreateDescriptorSetLayout for compute pipeline",
                ));
            }
        }
    }

    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let layout = match unsafe { shared.device.create_pipeline_layout(&layout_info, None) } {
        Ok(layout) => layout,
        Err(result) => {
            destroy_set_layouts(&shared, &mut set_layouts);
            return Err(native(result, "vkCreatePipelineLayout"));
        }
    };

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader.module())
        .name(&entry);
    let create = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = match unsafe {
        shared
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create], None)
    } {
        Ok(mut pipelines) => match pipelines.pop() {
            Some(pipeline) => pipeline,
            None => {
                unsafe {
                    shared.device.destroy_pipeline_layout(layout, None);
                }
                destroy_set_layouts(&shared, &mut set_layouts);
                return Err(VulkanFailure::Unsupported {
                    what: "Vulkan compute-pipeline creation",
                    why: "the driver returned no pipeline for one requested create-info",
                });
            }
        },
        Err((_, result)) => {
            unsafe {
                shared.device.destroy_pipeline_layout(layout, None);
            }
            destroy_set_layouts(&shared, &mut set_layouts);
            return Err(native(result, "vkCreateComputePipelines"));
        }
    };

    Ok(VulkanComputePipeline {
        shared,
        pipeline,
        layout,
        set_layouts,
    })
}

fn destroy_set_layouts(shared: &VulkanShared, set_layouts: &mut Vec<vk::DescriptorSetLayout>) {
    unsafe {
        for layout in set_layouts.drain(..) {
            shared.device.destroy_descriptor_set_layout(layout, None);
        }
    }
}

fn native(result: vk::Result, operation: &'static str) -> VulkanFailure {
    VulkanFailure::Native(ffi::NativeError::new(result, operation))
}
