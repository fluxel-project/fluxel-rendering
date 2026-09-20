//! Vulkan compute-dispatch lowering.
//!
//! This deliberately accepts buffer-only dispatches.  Binding an image without
//! transitioning it to the layout advertised in its descriptor is incorrect,
//! and the transfer-only tracker is not a substitute for a general Vulkan
//! access/stage/layout tracker.  Texture uses therefore remain a Phase-A
//! refusal until that tracker exists; capability facts must not promise this
//! path ahead of its lowering.

use ash::vk;

use crate::api::binding::BindGroup;
use crate::api::command::record::ComputeDispatch;
use crate::api::command::{AccessMask, ResourceUse};
use crate::api::pipeline::ComputePipeline;
use crate::backend::vulkan::binding::VulkanBindGroup;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::pipeline::VulkanComputePipeline;
use crate::backend::vulkan::platform::device::VulkanShared;

/// Portable handles retained by an accepted dispatch until its batch fence is
/// terminal.  Vulkan command buffers only retain native handles; without these
/// clones a caller could drop a pipeline or descriptor packet while queued GPU
/// work still refers to it.
#[derive(Default)]
pub(super) struct ComputeRetention {
    pub(super) pipelines: Vec<ComputePipeline>,
    pub(super) bind_groups: Vec<BindGroup>,
}

/// Records one validated compute dispatch into a command buffer.
///
/// The caller must merge the returned retention into its accepted-batch
/// retention before queue submission.  Keeping that ownership explicit makes
/// Phase A allocation/recording failure leave no accepted-work side effects.
pub(super) fn lower_compute_dispatch(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    dispatch: &ComputeDispatch,
    uses: &[ResourceUse],
) -> Result<ComputeRetention, VulkanFailure> {
    let pipeline = dispatch
        .pipeline
        .native()
        .as_any()
        .downcast_ref::<VulkanComputePipeline>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a compute pipeline this Vulkan device did not create",
            why: "its native pipeline belongs to another backend",
        })?;

    let mut sets = Vec::with_capacity(dispatch.groups.len());
    let mut first_sets = Vec::with_capacity(dispatch.groups.len());
    for bound in &dispatch.groups {
        if !bound.dynamic_offsets.is_empty() {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan compute bind group with dynamic offsets",
                why: "this slice has no dynamic-offset command lowering",
            });
        }
        let native = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<VulkanBindGroup>()
            .ok_or(VulkanFailure::Unsupported {
                what: "a bind group this Vulkan device did not create",
                why: "its descriptor set belongs to another backend",
            })?;
        first_sets.push(bound.index.get());
        sets.push(native.set());
    }

    // A buffer barrier is intentionally conservative.  It gives a preceding
    // transfer or shader write visibility to this dispatch without claiming to
    // be the persistent resource-state system needed by textures/raster.
    for use_ in uses {
        let ResourceUse::Buffer(buffer) = use_ else {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan compute dispatch that touches a texture or presentation frame",
                why: "image layout/access tracking is not implemented for compute dispatch",
            });
        };
        let destination_access = if buffer.access.contains(AccessMask::SHADER_WRITE) {
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
        } else {
            vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_READ
        };
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(destination_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(native_buffer(buffer.buffer.native())?)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe {
            shared.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }

    unsafe {
        shared.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline(),
        );
        // The recorder stores groups by their logical index.  Vulkan permits
        // sparse binding here, but its API binds contiguous ranges, so emit one
        // one-set bind per group and preserve that index exactly.
        for (first_set, set) in first_sets.into_iter().zip(sets) {
            shared.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                first_set,
                &[set],
                &[],
            );
        }
        let (x, y, z) = dispatch.workgroups;
        shared.device.cmd_dispatch(command_buffer, x, y, z);
    }

    Ok(ComputeRetention {
        pipelines: vec![dispatch.pipeline.clone()],
        bind_groups: dispatch
            .groups
            .iter()
            .map(|bound| bound.group.clone())
            .collect(),
    })
}

fn native_buffer(
    native: &dyn crate::api::resource::backend::BufferBackend,
) -> Result<vk::Buffer, VulkanFailure> {
    native
        .as_any()
        .downcast_ref::<crate::backend::vulkan::resource::VulkanBuffer>()
        .map(crate::backend::vulkan::resource::VulkanBuffer::buffer)
        .ok_or(VulkanFailure::Unsupported {
            what: "a buffer this Vulkan device did not create",
            why: "its native allocation belongs to another backend",
        })
}
