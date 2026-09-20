//! Vulkan buffer-transfer lowering.
//!
//! This is the first command vertical slice.  It intentionally owns only
//! buffer-to-buffer copy, buffer upload and buffer readback; every other
//! recorded payload remains a Phase-A `Unsupported` refusal.  Keeping that
//! boundary explicit is what prevents capability facts from getting ahead of
//! actual lowering.

use std::sync::Arc;

use ash::vk;

use crate::api::binding::BindGroup;
use crate::api::command::copy::{BufferCopy, BufferTextureCopy, TextureCopy};
use crate::api::command::{AccessMask, TextureUse, TextureUseIntent};
use crate::api::format::{block_extent, logical_bytes_per_block};
use crate::api::identity::ObjectId;
use crate::api::pipeline::ComputePipeline;
use crate::api::query::QuerySet;
use crate::api::resource::buffer::Buffer;
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::subresource::{HostTexelLayout, TextureAspect, TextureSubresourceLayers};
use crate::api::resource::texture::{Texture, TextureDimension};
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTexelLayout, ReadbackTicket, UploadDescriptor,
    UploadJob,
};
use crate::api::resource::view::TextureView;
use crate::backend::vulkan::binding::descriptor_image_layout;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::resource::{
    VulkanBuffer, VulkanStagingBuffer, VulkanTexture, create_staging_buffer,
};

/// Resources whose portable handles have to outlive accepted GPU work.
///
/// Vulkan command buffers retain native handles, not Fluxel handles.  These
/// clones therefore remain in `PendingBatch` until its fence is terminal.
#[derive(Default)]
pub(super) struct TransferRetention {
    pub(super) buffers: Vec<Buffer>,
    pub(super) textures: Vec<Texture>,
    pub(super) staging: Vec<VulkanStagingBuffer>,
    pub(super) readbacks: Vec<ReadbackRetention>,
    pub(super) compute_pipelines: Vec<ComputePipeline>,
    pub(super) bind_groups: Vec<BindGroup>,
    /// Query pools are native objects too; retain their portable owners until
    /// the enclosing fence retires rather than relying on recorded work to
    /// survive submission.
    pub(super) query_sets: Vec<QuerySet>,
    pub(super) raster: Vec<super::raster::RasterRetention>,
    /// Per-subresource layout knowledge seeded from the queue-domain tracker.
    /// Fluxel object identity, rather than a recyclable `VkImage` handle, makes
    /// retained entries safe after a texture is destroyed.
    image_layouts: Vec<ImageLayoutState>,
}

#[derive(Clone, Copy)]
pub(super) struct ImageLayoutState {
    texture: ObjectId,
    aspect: TextureAspect,
    mip_level: u32,
    array_layer: u32,
    layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags,
    access: vk::AccessFlags,
}

impl TransferRetention {
    pub(super) fn readback_tickets(&self) -> Vec<ReadbackTicket> {
        self.readbacks
            .iter()
            .map(|retention| retention.ticket.clone())
            .collect()
    }

    pub(super) fn seed_image_layouts(&mut self, layouts: &[ImageLayoutState]) {
        self.image_layouts.extend_from_slice(layouts);
    }

    pub(super) fn image_layouts(&self) -> Vec<ImageLayoutState> {
        self.image_layouts.clone()
    }

    pub(super) fn retain_compute(&mut self, compute: super::compute::ComputeRetention) {
        self.compute_pipelines.extend(compute.pipelines);
        self.bind_groups.extend(compute.bind_groups);
    }

    pub(super) fn retain_raster(&mut self, raster: super::raster::RasterRetention) {
        self.raster.push(raster);
    }
}

pub(super) struct ReadbackRetention {
    pub(super) staging: VulkanStagingBuffer,
    pub(super) ticket: ReadbackTicket,
    pub(super) layout: Option<ReadbackTexelLayout>,
}

/// Lowers a direct buffer copy with conservative whole-command-buffer memory
/// dependencies. This baseline deliberately covers preceding shader writes as
/// well as transfers; narrowing it requires the future unified buffer access
/// tracker to prove the previous stage/access for each range.
pub(super) fn lower_buffer_copy(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &BufferCopy,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let source = native_buffer(&copy.src)?;
    let destination = native_buffer(&copy.dst)?;
    transfer_dependency(
        shared,
        command_buffer,
        source.buffer(),
        vk::AccessFlags::TRANSFER_READ,
    );
    transfer_dependency(
        shared,
        command_buffer,
        destination.buffer(),
        vk::AccessFlags::TRANSFER_WRITE,
    );
    let region = vk::BufferCopy::default()
        .src_offset(copy.src_offset)
        .dst_offset(copy.dst_offset)
        .size(copy.size);
    // SAFETY: portable recording validated ownership, COPY usage, ranges, and
    // non-overlap. Both native handles belong to `shared` and the barriers above
    // make preceding transfer writes available to this operation.
    unsafe {
        shared.device.cmd_copy_buffer(
            command_buffer,
            source.buffer(),
            destination.buffer(),
            &[region],
        );
    }
    retention.buffers.push(copy.src.clone());
    retention.buffers.push(copy.dst.clone());
    Ok(())
}

/// Zeroes one 4-byte-aligned portable buffer range with Vulkan's native fill
/// command. Public validation owns the alignment rule; this lowering keeps the
/// native state transition and lifetime retention beside the other transfer
/// writes so `ClearBuffer` cannot be advertised as a no-op.
pub(super) fn lower_clear_buffer(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: &Buffer,
    range: BufferRange,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let destination = native_buffer(buffer)?;
    transfer_dependency(
        shared,
        command_buffer,
        destination.buffer(),
        vk::AccessFlags::TRANSFER_WRITE,
    );
    unsafe {
        shared.device.cmd_fill_buffer(
            command_buffer,
            destination.buffer(),
            range.offset,
            range.size,
            0,
        );
    }
    retention.buffers.push(buffer.clone());
    Ok(())
}

/// Clears validated texture subresources to the portable zero value using
/// Vulkan's native image-clear commands. No render/compute emulation is used.
pub(super) fn lower_clear_texture(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    texture: &Texture,
    range: crate::api::resource::TextureSubresourceRange,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let native = native_texture(texture)?;
    let native_range = vk::ImageSubresourceRange::default()
        .base_mip_level(range.base_mip)
        .level_count(range.mip_count)
        .base_array_layer(range.base_layer)
        .layer_count(range.layer_count);
    for aspect in shader_texture_aspects(range.aspects) {
        for mip in range.base_mip..range.base_mip + range.mip_count {
            transition_image(
                shared,
                command_buffer,
                native.image(),
                texture,
                TextureSubresourceLayers {
                    aspect,
                    mip_level: mip,
                    base_layer: range.base_layer,
                    layer_count: range.layer_count,
                },
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                retention,
            );
        }
    }
    let color = range
        .aspects
        .contains(crate::api::resource::TextureAspects::COLOR);
    if color {
        unsafe {
            shared.device.cmd_clear_color_image(
                command_buffer,
                native.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { uint32: [0; 4] },
                &[native_range.aspect_mask(vk::ImageAspectFlags::COLOR)],
            );
        }
    }
    let depth_stencil = range
        .aspects
        .contains(crate::api::resource::TextureAspects::DEPTH)
        || range
            .aspects
            .contains(crate::api::resource::TextureAspects::STENCIL);
    if depth_stencil {
        let mut aspects = vk::ImageAspectFlags::empty();
        if range
            .aspects
            .contains(crate::api::resource::TextureAspects::DEPTH)
        {
            aspects |= vk::ImageAspectFlags::DEPTH;
        }
        if range
            .aspects
            .contains(crate::api::resource::TextureAspects::STENCIL)
        {
            aspects |= vk::ImageAspectFlags::STENCIL;
        }
        unsafe {
            shared.device.cmd_clear_depth_stencil_image(
                command_buffer,
                native.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
                &[native_range.aspect_mask(aspects)],
            );
        }
    }
    retention.textures.push(texture.clone());
    Ok(())
}

pub(super) fn lower_upload(
    shared: &Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    job: &UploadJob,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let UploadDescriptor::Buffer(descriptor) = job.descriptor() else {
        return Err(VulkanFailure::Unsupported {
            what: "a texture upload",
            why: "the Vulkan buffer-transfer slice does not lower texture uploads",
        });
    };
    let destination = native_buffer(&descriptor.dst)?;
    let staging = create_staging_buffer(
        Arc::clone(shared),
        descriptor.bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )
    .map_err(native("Vulkan buffer-upload staging allocation"))?;
    staging
        .write(&descriptor.bytes)
        .map_err(native("Vulkan buffer-upload staging map/flush"))?;
    host_write_to_transfer_read(shared, command_buffer, staging.buffer());
    transfer_dependency(
        shared,
        command_buffer,
        destination.buffer(),
        vk::AccessFlags::TRANSFER_WRITE,
    );
    let region = vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(descriptor.dst_offset)
        .size(descriptor.bytes.len() as u64);
    // SAFETY: the staging allocation remains retained through the batch fence;
    // portable validation checked destination range and COPY_DST usage.
    unsafe {
        shared.device.cmd_copy_buffer(
            command_buffer,
            staging.buffer(),
            destination.buffer(),
            &[region],
        );
    }
    retention.buffers.push(descriptor.dst.clone());
    retention.staging.push(staging);
    Ok(())
}

pub(super) fn lower_readback(
    shared: &Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    ticket: &ReadbackTicket,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let ReadbackRequest::Buffer { src, range, .. } = ticket.request() else {
        return Err(VulkanFailure::Unsupported {
            what: "a texture readback",
            why: "the Vulkan buffer-transfer slice does not lower texture readbacks",
        });
    };
    let source = native_buffer(src)?;
    let staging = create_staging_buffer(
        Arc::clone(shared),
        range.size,
        vk::BufferUsageFlags::TRANSFER_DST,
    )
    .map_err(native("Vulkan buffer-readback staging allocation"))?;
    transfer_dependency(
        shared,
        command_buffer,
        source.buffer(),
        vk::AccessFlags::TRANSFER_READ,
    );
    let region = vk::BufferCopy::default()
        .src_offset(range.offset)
        .dst_offset(0)
        .size(range.size);
    // SAFETY: portable validation checked the source range and COPY_SRC usage;
    // staging remains retained until this batch's fence completes.
    unsafe {
        shared
            .device
            .cmd_copy_buffer(command_buffer, source.buffer(), staging.buffer(), &[region]);
    }
    transfer_write_to_host_read(shared, command_buffer, staging.buffer());
    retention.buffers.push(src.clone());
    retention.readbacks.push(ReadbackRetention {
        staging,
        ticket: ticket.clone(),
        layout: None,
    });
    Ok(())
}

/// Lowers direct image transfer commands. Layouts are keyed by the portable
/// texture `ObjectId` plus subresource and persist across accepted submissions;
/// using a recycled `VkImage` handle as identity, or restarting every submit at
/// `UNDEFINED`, would allow Vulkan to discard live texture contents.
///
/// This remains a transfer-only correctness tracker. Before raster or
/// texture-backed compute is advertised it must become one backend-private
/// access/stage/layout authority covering all image uses. Merely adding another
/// local table would create contradictory layout histories. Stale `ObjectId`
/// retirement is a bounded-memory TODO and must be completion-safe.
pub(super) fn lower_buffer_texture_copy(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &BufferTextureCopy,
    to_texture: bool,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let buffer = native_buffer(&copy.buffer)?;
    let texture = native_texture(&copy.texture)?;
    let layout = if to_texture {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL
    } else {
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
    };
    transition_image(
        shared,
        command_buffer,
        texture.image(),
        &copy.texture,
        copy.texture_subresource,
        layout,
        vk::PipelineStageFlags::TRANSFER,
        if to_texture {
            vk::AccessFlags::TRANSFER_WRITE
        } else {
            vk::AccessFlags::TRANSFER_READ
        },
        retention,
    );
    let bytes = logical_bytes_per_block(copy.texture.descriptor().format).ok_or(
        VulkanFailure::Unsupported {
            what: "a texture transfer format",
            why: "the format has no Vulkan byte-copy block size",
        },
    )?;
    let (block_width, block_height) = block_extent(copy.texture.descriptor().format);
    // Vulkan expresses these two fields in *texels*, whereas Fluxel host
    // layouts express row pitch and rows-per-image in compressed blocks. A
    // 16-byte ASTC block is not one texel: passing the block count directly
    // would make Vulkan interpret the source as a 1x1 format.
    let row_length = (copy.bytes_per_row / bytes)
        .checked_mul(block_width)
        .ok_or(VulkanFailure::Unsupported {
            what: "a texture transfer row pitch",
            why: "the Vulkan texel row length overflows u32",
        })?;
    let image_height =
        copy.rows_per_image
            .checked_mul(block_height)
            .ok_or(VulkanFailure::Unsupported {
                what: "a texture transfer image pitch",
                why: "the Vulkan texel image height overflows u32",
            })?;
    let region = vk::BufferImageCopy::default()
        .buffer_offset(copy.buffer_offset)
        .buffer_row_length(row_length)
        .buffer_image_height(image_height)
        .image_subresource(image_layers(copy.texture_subresource))
        .image_offset(image_offset(copy.texture_origin))
        .image_extent(image_extent(copy.extent));
    unsafe {
        if to_texture {
            shared.device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer.buffer(),
                texture.image(),
                layout,
                &[region],
            );
        } else {
            shared.device.cmd_copy_image_to_buffer(
                command_buffer,
                texture.image(),
                layout,
                buffer.buffer(),
                &[region],
            );
        }
    }
    retention.buffers.push(copy.buffer.clone());
    retention.textures.push(copy.texture.clone());
    Ok(())
}

pub(super) fn lower_texture_copy(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    copy: &TextureCopy,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let source = native_texture(&copy.src)?;
    let destination = native_texture(&copy.dst)?;
    transition_image(
        shared,
        command_buffer,
        source.image(),
        &copy.src,
        copy.src_subresource,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_READ,
        retention,
    );
    transition_image(
        shared,
        command_buffer,
        destination.image(),
        &copy.dst,
        copy.dst_subresource,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_WRITE,
        retention,
    );
    let region = vk::ImageCopy::default()
        .src_subresource(image_layers(copy.src_subresource))
        .src_offset(image_offset(copy.src_origin))
        .dst_subresource(image_layers(copy.dst_subresource))
        .dst_offset(image_offset(copy.dst_origin))
        .extent(image_extent(copy.extent));
    unsafe {
        shared.device.cmd_copy_image(
            command_buffer,
            source.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination.image(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    retention.textures.push(copy.src.clone());
    retention.textures.push(copy.dst.clone());
    Ok(())
}

pub(super) fn lower_texture_upload(
    shared: &Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    job: &UploadJob,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let UploadDescriptor::Texture(desc) = job.descriptor() else {
        return Err(VulkanFailure::Unsupported {
            what: "a buffer upload",
            why: "not a texture upload",
        });
    };
    let block =
        logical_bytes_per_block(desc.dst.descriptor().format).ok_or(VulkanFailure::Unsupported {
            what: "a texture upload format",
            why: "the format has no Vulkan byte-copy block size",
        })? as usize;
    let (block_width, block_height) = block_extent(desc.dst.descriptor().format);
    let tight_row = desc.extent.width.div_ceil(block_width) as usize * block;
    let rows = desc.extent.height.div_ceil(block_height) as usize;
    let images = if desc.dst.descriptor().dimension == TextureDimension::D3 {
        desc.extent.depth as usize
    } else {
        desc.subresource.layer_count as usize
    };
    let tight_size = tight_row
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(images))
        .ok_or(VulkanFailure::Unsupported {
            what: "a texture upload",
            why: "the packed staging size overflows usize",
        })?;
    let mut packed = vec![0u8; tight_size];
    repack_rows(
        &desc.bytes,
        desc.source_layout,
        tight_row,
        rows,
        images,
        &mut packed,
    )?;
    let staging = create_staging_buffer(
        Arc::clone(shared),
        packed.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )
    .map_err(native("Vulkan texture-upload staging allocation"))?;
    staging
        .write(&packed)
        .map_err(native("Vulkan texture-upload staging map/flush"))?;
    host_write_to_transfer_read(shared, command_buffer, staging.buffer());
    let texture = native_texture(&desc.dst)?;
    transition_image(
        shared,
        command_buffer,
        texture.image(),
        &desc.dst,
        desc.subresource,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_WRITE,
        retention,
    );
    let region = vk::BufferImageCopy::default()
        .image_subresource(image_layers(desc.subresource))
        .image_offset(image_offset(desc.origin))
        .image_extent(image_extent(desc.extent));
    unsafe {
        shared.device.cmd_copy_buffer_to_image(
            command_buffer,
            staging.buffer(),
            texture.image(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    retention.textures.push(desc.dst.clone());
    retention.staging.push(staging);
    Ok(())
}

pub(super) fn lower_texture_readback(
    shared: &Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    ticket: &ReadbackTicket,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let ReadbackRequest::Texture {
        src,
        subresource,
        origin,
        extent,
        ..
    } = ticket.request()
    else {
        return Err(VulkanFailure::Unsupported {
            what: "a buffer readback",
            why: "not a texture readback",
        });
    };
    let block =
        logical_bytes_per_block(src.descriptor().format).ok_or(VulkanFailure::Unsupported {
            what: "a texture readback format",
            why: "the format has no Vulkan byte-copy block size",
        })?;
    let (block_width, block_height) = block_extent(src.descriptor().format);
    let row = extent
        .width
        .div_ceil(block_width)
        .checked_mul(block)
        .ok_or(VulkanFailure::Unsupported {
            what: "a texture readback",
            why: "row pitch overflows u32",
        })?;
    let images = if src.descriptor().dimension == TextureDimension::D3 {
        extent.depth
    } else {
        subresource.layer_count
    };
    let size = u64::from(row)
        .checked_mul(u64::from(extent.height.div_ceil(block_height)))
        .and_then(|v| v.checked_mul(u64::from(images)))
        .ok_or(VulkanFailure::Unsupported {
            what: "a texture readback",
            why: "staging size overflows u64",
        })?;
    let staging =
        create_staging_buffer(Arc::clone(shared), size, vk::BufferUsageFlags::TRANSFER_DST)
            .map_err(native("Vulkan texture-readback staging allocation"))?;
    let texture = native_texture(src)?;
    transition_image(
        shared,
        command_buffer,
        texture.image(),
        src,
        *subresource,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_READ,
        retention,
    );
    let region = vk::BufferImageCopy::default()
        .image_subresource(image_layers(*subresource))
        .image_offset(image_offset(*origin))
        .image_extent(image_extent(*extent));
    unsafe {
        shared.device.cmd_copy_image_to_buffer(
            command_buffer,
            texture.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging.buffer(),
            &[region],
        );
    }
    transfer_write_to_host_read(shared, command_buffer, staging.buffer());
    retention.textures.push(src.clone());
    retention.readbacks.push(ReadbackRetention {
        staging,
        ticket: ticket.clone(),
        layout: Some(ReadbackTexelLayout {
            bytes_per_row: row,
            rows_per_image: extent.height.div_ceil(block_height),
            total_size: size,
        }),
    });
    Ok(())
}

/// Turns a fence-complete staging allocation into the ticket's RAII-readable
/// bytes. A terminal map/invalidate failure is returned to the device loss
/// authority; a non-terminal mapping failure terminates only this ticket.
pub(super) fn publish_readback(retention: &ReadbackRetention) -> Result<(), vk::Result> {
    match retention.staging.read() {
        Ok(bytes) => {
            retention.ticket.publish(bytes, retention.layout);
            Ok(())
        }
        Err(error) => {
            retention
                .ticket
                .set_status(if error == vk::Result::ERROR_DEVICE_LOST {
                    ReadbackStatus::DeviceLost
                } else {
                    ReadbackStatus::Failed
                });
            Err(error)
        }
    }
}

fn native(operation: &'static str) -> impl FnOnce(vk::Result) -> VulkanFailure {
    move |result| {
        VulkanFailure::Native(crate::backend::vulkan::ffi::NativeError::new(
            result, operation,
        ))
    }
}

fn native_buffer(buffer: &Buffer) -> Result<&VulkanBuffer, VulkanFailure> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<VulkanBuffer>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a non-Vulkan buffer",
            why: "a Vulkan submission may only lower buffers created by the same Vulkan device",
        })
}

fn native_texture(texture: &Texture) -> Result<&VulkanTexture, VulkanFailure> {
    texture
        .native()
        .as_any()
        .downcast_ref::<VulkanTexture>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a non-Vulkan texture",
            why: "a Vulkan submission may only lower textures created by the same Vulkan device",
        })
}

fn image_layers(value: TextureSubresourceLayers) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(match value.aspect {
            TextureAspect::Color => vk::ImageAspectFlags::COLOR,
            TextureAspect::Depth => vk::ImageAspectFlags::DEPTH,
            TextureAspect::Stencil => vk::ImageAspectFlags::STENCIL,
            TextureAspect::Plane0 => vk::ImageAspectFlags::PLANE_0,
            TextureAspect::Plane1 => vk::ImageAspectFlags::PLANE_1,
            TextureAspect::Plane2 => vk::ImageAspectFlags::PLANE_2,
        })
        .mip_level(value.mip_level)
        .base_array_layer(value.base_layer)
        .layer_count(value.layer_count)
}
fn image_offset(value: crate::api::resource::subresource::Origin3d) -> vk::Offset3D {
    vk::Offset3D {
        x: value.x as i32,
        y: value.y as i32,
        z: value.z as i32,
    }
}
fn image_extent(value: crate::api::resource::texture::Extent3d) -> vk::Extent3D {
    vk::Extent3D {
        width: value.width,
        height: value.height,
        depth: value.depth,
    }
}

pub(super) fn transition_image(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    texture: &Texture,
    layers: TextureSubresourceLayers,
    new_layout: vk::ImageLayout,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
    retention: &mut TransferRetention,
) {
    for array_layer in layers.base_layer..layers.base_layer + layers.layer_count {
        let existing = retention.image_layouts.iter_mut().find(|known| {
            known.texture == texture.id()
                && known.aspect == layers.aspect
                && known.mip_level == layers.mip_level
                && known.array_layer == array_layer
        });
        let old_layout = existing
            .as_ref()
            .map_or(vk::ImageLayout::UNDEFINED, |known| known.layout);
        let src_stage = existing
            .as_ref()
            .map_or(vk::PipelineStageFlags::TOP_OF_PIPE, |known| known.stage);
        let src_access = existing
            .as_ref()
            .map_or(vk::AccessFlags::empty(), |known| known.access);
        if old_layout == new_layout && src_stage == dst_stage && src_access == dst_access {
            continue;
        }
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_layers(layers).aspect_mask)
                    .base_mip_level(layers.mip_level)
                    .level_count(1)
                    .base_array_layer(array_layer)
                    .layer_count(1),
            );
        unsafe {
            shared.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        if let Some(entry) = existing {
            entry.layout = new_layout;
            entry.stage = dst_stage;
            entry.access = dst_access;
        } else {
            retention.image_layouts.push(ImageLayoutState {
                texture: texture.id(),
                aspect: layers.aspect,
                mip_level: layers.mip_level,
                array_layer,
                layout: new_layout,
                stage: dst_stage,
                access: dst_access,
            });
        }
    }
}

/// Transitions shader-visible image subresources to the exact layout written
/// into their immutable Vulkan descriptors.
///
/// This is deliberately owned by the queue-domain image-state authority rather
/// than by compute or raster lowering.  A copy in one submission followed by a
/// sampled image use in another must observe the same `ObjectId`/aspect/mip/
/// layer state. `stage` lets a caller select compute or graphics without
/// introducing a second tracker for either execution domain.
pub(super) fn transition_shader_texture(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    use_: &TextureUse,
    stage: vk::PipelineStageFlags,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let (layout, access) = match use_.intent {
        TextureUseIntent::ShaderRead => (
            descriptor_image_layout(vk::DescriptorType::SAMPLED_IMAGE, use_.subresources.aspects),
            vk::AccessFlags::SHADER_READ,
        ),
        TextureUseIntent::ShaderReadWrite => {
            let mut access = vk::AccessFlags::empty();
            if use_.access.contains(AccessMask::SHADER_READ) {
                access |= vk::AccessFlags::SHADER_READ;
            }
            if use_.access.contains(AccessMask::SHADER_WRITE) {
                access |= vk::AccessFlags::SHADER_WRITE;
            }
            if access.is_empty() {
                return Err(VulkanFailure::Unsupported {
                    what: "a Vulkan storage-image shader use without shader access",
                    why: "the portable ResourceUse must declare the storage image read and/or write",
                });
            }
            (vk::ImageLayout::GENERAL, access)
        }
        _ => {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan texture use outside a shader binding",
                why: "copy, resolve, and attachment intents have separate lowering paths",
            });
        }
    };
    let native = native_texture(&use_.texture)?;
    let range = use_.subresources;
    for aspect in shader_texture_aspects(range.aspects) {
        for mip_level in range.base_mip..range.base_mip + range.mip_count {
            transition_image(
                shared,
                command_buffer,
                native.image(),
                &use_.texture,
                TextureSubresourceLayers {
                    aspect,
                    mip_level,
                    base_layer: range.base_layer,
                    layer_count: range.layer_count,
                },
                layout,
                stage,
                access,
                retention,
            );
        }
    }
    Ok(())
}

fn shader_texture_aspects(
    aspects: crate::api::resource::subresource::TextureAspects,
) -> Vec<TextureAspect> {
    let mut result = Vec::with_capacity(3);
    if aspects.contains(crate::api::resource::subresource::TextureAspects::COLOR) {
        result.push(TextureAspect::Color);
    }
    if aspects.contains(crate::api::resource::subresource::TextureAspects::DEPTH) {
        result.push(TextureAspect::Depth);
    }
    if aspects.contains(crate::api::resource::subresource::TextureAspects::STENCIL) {
        result.push(TextureAspect::Stencil);
    }
    result
}

/// Routes a raster attachment transition through the queue-domain image-state
/// authority used by transfer commands. A view may cover several layers or
/// depth/stencil aspects; each physical subresource gets an independent state.
pub(super) fn transition_raster_attachment(
    shared: &Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    view: &TextureView,
    new_layout: vk::ImageLayout,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let texture = view.texture();
    let native = native_texture(texture)?;
    let descriptor = view.descriptor();
    let mut aspects = Vec::with_capacity(2);
    if descriptor
        .aspects
        .contains(crate::api::resource::TextureAspects::COLOR)
    {
        aspects.push(TextureAspect::Color);
    }
    if descriptor
        .aspects
        .contains(crate::api::resource::TextureAspects::DEPTH)
    {
        aspects.push(TextureAspect::Depth);
    }
    if descriptor
        .aspects
        .contains(crate::api::resource::TextureAspects::STENCIL)
    {
        aspects.push(TextureAspect::Stencil);
    }
    for aspect in aspects {
        for mip_level in descriptor.base_mip..descriptor.base_mip + descriptor.mip_count {
            transition_image(
                shared,
                command_buffer,
                native.image(),
                texture,
                TextureSubresourceLayers {
                    aspect,
                    mip_level,
                    base_layer: descriptor.base_layer,
                    layer_count: descriptor.layer_count,
                },
                new_layout,
                dst_stage,
                dst_access,
                retention,
            );
        }
    }
    Ok(())
}

/// Conservative raster buffer dependency until the range-aware state tracker
/// can narrow stage/access masks. The destination mask is still derived from
/// the actual portable use so validation and native synchronization cannot
/// silently disagree about vertex/index/uniform/storage roles.
pub(super) fn barrier_raster_buffer(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: &Buffer,
    access: AccessMask,
    retention: &mut TransferRetention,
) -> Result<(), VulkanFailure> {
    let native = native_buffer(buffer)?;
    let mut destination = vk::AccessFlags::empty();
    if access.contains(AccessMask::INDIRECT_READ) {
        destination |= vk::AccessFlags::INDIRECT_COMMAND_READ;
    }
    if access.contains(AccessMask::VERTEX_READ) {
        destination |= vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    if access.contains(AccessMask::INDEX_READ) {
        destination |= vk::AccessFlags::INDEX_READ;
    }
    if access.contains(AccessMask::UNIFORM_READ) {
        destination |= vk::AccessFlags::UNIFORM_READ;
    }
    if access.contains(AccessMask::SHADER_READ) {
        destination |= vk::AccessFlags::SHADER_READ;
    }
    if access.contains(AccessMask::SHADER_WRITE) {
        destination |= vk::AccessFlags::SHADER_WRITE;
    }
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(destination)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(native.buffer())
        .offset(0)
        .size(vk::WHOLE_SIZE);
    unsafe {
        shared.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            // DRAW_INDIRECT covers indirect argument/count reads; graphics
            // stages cover the remaining vertex/index/shader accesses.
            vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::ALL_GRAPHICS,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
    retention.buffers.push(buffer.clone());
    Ok(())
}

fn repack_rows(
    source: &[u8],
    layout: HostTexelLayout,
    tight_row: usize,
    rows: usize,
    images: usize,
    destination: &mut [u8],
) -> Result<(), VulkanFailure> {
    let source_row = layout.bytes_per_row as usize;
    let source_image = layout.rows_per_image as usize;
    for image in 0..images {
        for row in 0..rows {
            let src = image
                .checked_mul(source_image)
                .and_then(|v| v.checked_add(row))
                .and_then(|v| v.checked_mul(source_row))
                .ok_or(VulkanFailure::Unsupported {
                    what: "a texture upload",
                    why: "source layout offset overflows usize",
                })?;
            let dst = (image * rows + row) * tight_row;
            let end = src
                .checked_add(tight_row)
                .ok_or(VulkanFailure::Unsupported {
                    what: "a texture upload",
                    why: "source layout end overflows usize",
                })?;
            let target_end = dst + tight_row;
            let bytes = source.get(src..end).ok_or(VulkanFailure::Unsupported {
                what: "a texture upload",
                why: "validated source layout was not readable during lowering",
            })?;
            destination[dst..target_end].copy_from_slice(bytes);
        }
    }
    Ok(())
}

fn transfer_dependency(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        // A dispatch may have been the preceding writer. Until a persistent
        // range tracker exists, ALL_COMMANDS/MEMORY_* is the conservative
        // correctness bridge from any earlier buffer use to this transfer.
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    // SAFETY: command buffer is recording; all buffers use the selected queue
    // family exclusively, so no queue-family ownership transfer is requested.
    unsafe {
        shared.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

fn host_write_to_transfer_read(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::HOST_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    unsafe {
        shared.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

fn transfer_write_to_host_read(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    unsafe {
        shared.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}
