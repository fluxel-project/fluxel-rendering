//! Vulkan compute-dispatch lowering.
//!
//! Image descriptors name a layout, so accepting an image binding without a
//! matching transition would be invalid Vulkan.  Dispatch lowering therefore
//! routes every actual shader image use through the same queue-domain,
//! per-subresource tracker used by transfers.  That tracker is intentionally
//! shared rather than duplicated here: a texture copied in one submission and
//! consumed by compute in the next must retain its real prior layout.

use ash::vk;

use crate::api::binding::BindGroup;
use crate::api::command::record::{BoundGroup, ComputeDispatch, ComputeIndirect};
use crate::api::command::{AccessMask, ResourceUse};
use crate::api::pipeline::ComputePipeline;
use crate::backend::vulkan::binding::VulkanBindGroup;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::pipeline::VulkanComputePipeline;
use crate::backend::vulkan::platform::device::VulkanShared;

use super::transfer::{self, TransferRetention};

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
    transfer_retention: &mut TransferRetention,
) -> Result<ComputeRetention, VulkanFailure> {
    lower_compute(
        shared,
        command_buffer,
        &dispatch.pipeline,
        &dispatch.groups,
        uses,
        transfer_retention,
        |device, command_buffer, _pipeline| unsafe {
            let (x, y, z) = dispatch.workgroups;
            device.cmd_dispatch(command_buffer, x, y, z);
        },
        &dispatch.immediates,
    )
}

/// Records core `vkCmdDispatchIndirect` using the same binding/state path as a
/// direct dispatch. The argument buffer is already present in `uses`, so the
/// common barrier path retains it and establishes INDIRECT_COMMAND_READ.
pub(super) fn lower_compute_indirect(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    dispatch: &ComputeIndirect,
    uses: &[ResourceUse],
    transfer_retention: &mut TransferRetention,
) -> Result<ComputeRetention, VulkanFailure> {
    let arguments = native_buffer(dispatch.arguments.native())?;
    lower_compute(
        shared,
        command_buffer,
        &dispatch.pipeline,
        &dispatch.groups,
        uses,
        transfer_retention,
        |device, command_buffer, _| unsafe {
            device.cmd_dispatch_indirect(command_buffer, arguments, dispatch.arguments_offset);
        },
        &[],
    )
}

fn lower_compute<F>(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    pipeline_handle: &ComputePipeline,
    groups: &[BoundGroup],
    uses: &[ResourceUse],
    transfer_retention: &mut TransferRetention,
    emit: F,
    immediates: &[crate::api::command::record::ImmediateWrite],
) -> Result<ComputeRetention, VulkanFailure>
where
    F: FnOnce(&ash::Device, vk::CommandBuffer, &VulkanComputePipeline),
{
    let pipeline = pipeline_handle
        .native()
        .as_any()
        .downcast_ref::<VulkanComputePipeline>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a compute pipeline this Vulkan device did not create",
            why: "its native pipeline belongs to another backend",
        })?;

    let mut sets = Vec::with_capacity(groups.len());
    let mut first_sets = Vec::with_capacity(groups.len());
    for bound in groups {
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

    // Buffer barriers remain conservative until buffer-range state tracking is
    // added.  Images are more constrained: descriptor layouts are part of the
    // native command contract, and so use the persistent image tracker below.
    for use_ in uses {
        match use_ {
            ResourceUse::Buffer(buffer) => {
                let (destination_stage, destination_access) =
                    if buffer.access.contains(AccessMask::INDIRECT_READ) {
                        (
                            vk::PipelineStageFlags::DRAW_INDIRECT,
                            vk::AccessFlags::INDIRECT_COMMAND_READ,
                        )
                    } else if buffer.access.contains(AccessMask::SHADER_WRITE) {
                        (
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                    } else {
                        (
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_READ,
                        )
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
                        destination_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                }
            }
            ResourceUse::Texture(texture) => transfer::transition_shader_texture(
                shared,
                command_buffer,
                texture,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                transfer_retention,
            )?,
            ResourceUse::Frame(_) => {
                return Err(VulkanFailure::Unsupported {
                    what: "a presentation frame used by a Vulkan compute dispatch",
                    why: "presentation images are not compute-bindable in this Vulkan slice",
                });
            }
            ResourceUse::AccelerationStructure(_) => {
                return Err(VulkanFailure::Unsupported {
                    what: "an acceleration structure in a Vulkan compute dispatch",
                    why: "the Vulkan ray-query descriptor and synchronization extension path is not enabled",
                });
            }
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
        for immediate in immediates {
            shared.device.cmd_push_constants(
                command_buffer,
                pipeline.layout(),
                shader_stage_flags(immediate.visibility)?,
                immediate.offset,
                &immediate.bytes,
            );
        }
        emit(&shared.device, command_buffer, pipeline);
    }

    Ok(ComputeRetention {
        pipelines: vec![pipeline_handle.clone()],
        bind_groups: groups.iter().map(|bound| bound.group.clone()).collect(),
    })
}

fn shader_stage_flags(
    stages: crate::api::shader::ShaderStages,
) -> Result<vk::ShaderStageFlags, VulkanFailure> {
    if !stages.contains(crate::api::shader::ShaderStages::COMPUTE) {
        return Err(VulkanFailure::Unsupported {
            what: "a Vulkan compute immediate write without compute visibility",
            why: "the portable pipeline interface must declare the consuming compute stage",
        });
    }
    Ok(vk::ShaderStageFlags::COMPUTE)
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
