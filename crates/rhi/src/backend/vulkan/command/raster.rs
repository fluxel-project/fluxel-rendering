//! Vulkan raster-scope lowering.
//!
//! This module deliberately consumes the portable `TextureView`/`FrameAttachment`
//! distinction at its boundary.  An ordinary view becomes a Vulkan image view;
//! an acquired frame is refused until the presentation slice supplies the same
//! native attachment shape.  It must not leak a fake `TextureView` for a frame
//! into the public API merely because Vulkan swapchain images happen to be
//! `VkImage`s.
//!
//! Vulkan 1.0 render passes are used instead of dynamic rendering.  This keeps
//! the baseline valid on the advertised Vulkan 1.0 floor.  Pipelines therefore
//! have to be created against an attachment-compatible render pass; compatibility
//! is by attachment format/sample/reference shape, not by handle identity.

use std::sync::Arc;

use ash::vk;

use crate::api::binding::BindGroup;
use crate::api::command::attachment::{
    ColorAttachmentView, DepthAttachmentMode, StencilAttachmentMode,
};
use crate::api::command::geometry::{ColorClearValue, LoadOp, StoreOp};
use crate::api::command::record::{RasterBegin, RasterDraw};
use crate::api::command::{IndexFormat, ResourceUse, TextureUse, TextureUseIntent};
use crate::api::pipeline::RasterPipeline;
use crate::api::resource::buffer::Buffer;
use crate::api::resource::view::TextureView;
use crate::backend::vulkan::binding::VulkanBindGroup;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::format::vk_format;
use crate::backend::vulkan::pipeline::VulkanRasterPipeline;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::resource::{VulkanBuffer, VulkanTextureView};

use super::transfer;

/// Native/portable ownership retained for a raster command buffer until its
/// fence completes.  The spine merges this with its batch retention; native
/// command buffers do not retain Fluxel handles by themselves.
#[derive(Default)]
pub(super) struct RasterRetention {
    #[allow(
        dead_code,
        reason = "ownership retains native pipelines until the batch fence"
    )]
    pub(super) pipelines: Vec<RasterPipeline>,
    pub(super) bind_groups: Vec<BindGroup>,
    pub(super) buffers: Vec<Buffer>,
    pub(super) views: Vec<TextureView>,
    objects: Vec<RasterObjects>,
}

/// One open Vulkan render pass.  It is intentionally linear: a malformed
/// recording cannot issue a second begin or end the same pass twice.
pub(super) struct RasterScopeState {
    objects: RasterObjects,
    extent: vk::Extent2D,
    colors: Vec<TextureView>,
    depth: Option<TextureView>,
}

/// Render-pass/framebuffer handles are command-buffer references, so they may
/// be destroyed only after the accepted batch fence.  Keeping their destruction
/// private makes that lifetime rule difficult to accidentally violate.
struct RasterObjects {
    shared: Arc<VulkanShared>,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
}

impl Drop for RasterObjects {
    fn drop(&mut self) {
        unsafe {
            self.shared
                .device
                .destroy_framebuffer(self.framebuffer, None);
            self.shared
                .device
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

/// Begins one Vulkan 1.0 render pass and clears attachments where requested.
///
/// `transition_attachment` is the single image-layout authority owned by the
/// transfer/state tracker.  Raster must call it rather than invent a second
/// per-command layout map: upload -> raster -> readback can cross submissions.
pub(super) fn lower_raster_begin(
    shared: Arc<VulkanShared>,
    command_buffer: vk::CommandBuffer,
    begin: &RasterBegin,
    shader_texture_uses: &[TextureUse],
    transfer_retention: &mut transfer::TransferRetention,
) -> Result<RasterScopeState, VulkanFailure> {
    reject_raster_feedback(begin, shader_texture_uses)?;
    // Vulkan forbids vkCmdPipelineBarrier inside a render pass. The spine
    // therefore gathers this scope's draw uses before calling us, allowing the
    // shared image-state authority to establish every descriptor layout before
    // vkCmdBeginRenderPass. Do not move this into `lower_raster_draw`.
    for use_ in shader_texture_uses {
        transfer::transition_shader_texture(
            &shared,
            command_buffer,
            use_,
            vk::PipelineStageFlags::ALL_GRAPHICS,
            transfer_retention,
        )?;
    }
    let mut attachments = Vec::new();
    let mut attachment_views = Vec::new();
    let mut color_refs = Vec::with_capacity(begin.colors.len());
    let mut clears = Vec::new();
    let mut colors = Vec::with_capacity(begin.colors.len());
    let mut width = None;
    let mut height = None;

    // Vulkan permits sparse attachment locations via ATTACHMENT_UNUSED.  Keep
    // the indices rather than compacting them: fragment output location N must
    // remain color attachment N.
    for (location, color) in &begin.colors {
        if color.resolve.is_some() {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan raster attachment resolve",
                why: "resolve lowering is not implemented by the Vulkan raster slice",
            });
        }
        while color_refs.len() < *location as usize {
            color_refs.push(vk::AttachmentReference {
                attachment: vk::ATTACHMENT_UNUSED,
                layout: vk::ImageLayout::UNDEFINED,
            });
        }
        let view = texture_color_view(&color.view)?;
        check_extent(&mut width, &mut height, view.extent())?;
        transfer::transition_raster_attachment(
            &shared,
            command_buffer,
            &view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            transfer_retention,
        )?;
        let index = attachments.len() as u32;
        attachments.push(vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: vk_format(view.format()).ok_or(VulkanFailure::Unsupported {
                what: "a Vulkan color attachment format",
                why: "the view format has no Vulkan mapping",
            })?,
            samples: vk::SampleCountFlags::from_raw(view.sample_count()),
            load_op: color_load(color.load),
            store_op: color_store(color.store),
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        });
        color_refs.push(vk::AttachmentReference {
            attachment: index,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        });
        attachment_views.push(native_view(&view)?.view());
        clears.push(clear_color(color.load));
        colors.push(view);
    }

    let mut depth_ref = None;
    let mut depth = None;
    if let Some(attachment) = &begin.depth_stencil {
        let view = attachment.view.clone();
        check_extent(&mut width, &mut height, view.extent())?;
        let write = depth_writes(attachment.depth) || stencil_writes(attachment.stencil);
        let layout = if write {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        };
        let access = if write {
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        } else {
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
        };
        transfer::transition_raster_attachment(
            &shared,
            command_buffer,
            &view,
            layout,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access,
            transfer_retention,
        )?;
        let index = attachments.len() as u32;
        attachments.push(vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: vk_format(view.format()).ok_or(VulkanFailure::Unsupported {
                what: "a Vulkan depth/stencil attachment format",
                why: "the view format has no Vulkan mapping",
            })?,
            samples: vk::SampleCountFlags::from_raw(view.sample_count()),
            load_op: depth_load(attachment.depth),
            store_op: depth_store(attachment.depth),
            stencil_load_op: stencil_load(attachment.stencil),
            stencil_store_op: stencil_store(attachment.stencil),
            initial_layout: layout,
            final_layout: layout,
        });
        depth_ref = Some(vk::AttachmentReference {
            attachment: index,
            layout,
        });
        attachment_views.push(native_view(&view)?.view());
        clears.push(clear_depth_stencil(attachment.depth, attachment.stencil));
        depth = Some(view);
    }

    let (width, height) = width.zip(height).ok_or(VulkanFailure::Unsupported {
        what: "a Vulkan raster scope without attachments",
        why: "portable validation should reject an empty attachment set",
    })?;
    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    if let Some(reference) = depth_ref.as_ref() {
        subpass = subpass.depth_stencil_attachment(reference);
    }
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .dst_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
    ];
    let pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);
    let render_pass = unsafe { shared.device.create_render_pass(&pass_info, None) }
        .map_err(native("vkCreateRenderPass for raster scope"))?;
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachment_views)
        .width(width)
        .height(height)
        .layers(1);
    let framebuffer = match unsafe { shared.device.create_framebuffer(&framebuffer_info, None) } {
        Ok(value) => value,
        Err(result) => {
            unsafe { shared.device.destroy_render_pass(render_pass, None) };
            return Err(native("vkCreateFramebuffer for raster scope")(result));
        }
    };
    let objects = RasterObjects {
        shared,
        render_pass,
        framebuffer,
    };
    let info = vk::RenderPassBeginInfo::default()
        .render_pass(objects.render_pass)
        .framebuffer(objects.framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        })
        .clear_values(&clears);
    unsafe {
        objects.shared.device.cmd_begin_render_pass(
            command_buffer,
            &info,
            vk::SubpassContents::INLINE,
        );
    }
    Ok(RasterScopeState {
        objects,
        extent: vk::Extent2D { width, height },
        colors,
        depth,
    })
}

/// Lowers one draw with its recorded state.
pub(super) fn lower_raster_draw(
    shared: &VulkanShared,
    command_buffer: vk::CommandBuffer,
    draw: &RasterDraw,
    uses: &[ResourceUse],
    scope: &RasterScopeState,
    transfer_retention: &mut transfer::TransferRetention,
) -> Result<RasterRetention, VulkanFailure> {
    let pipeline = draw
        .pipeline
        .native()
        .as_any()
        .downcast_ref::<VulkanRasterPipeline>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a raster pipeline this Vulkan device did not create",
            why: "its native pipeline belongs to another backend",
        })?;
    let mut retention = RasterRetention {
        pipelines: vec![draw.pipeline.clone()],
        bind_groups: Vec::new(),
        buffers: Vec::new(),
        views: Vec::new(),
        objects: Vec::new(),
    };
    for resource_use in uses {
        match resource_use {
            ResourceUse::Buffer(use_) => {
                transfer::barrier_raster_buffer(
                    shared,
                    command_buffer,
                    &use_.buffer,
                    use_.access,
                    transfer_retention,
                )?;
                retention.buffers.push(use_.buffer.clone());
            }
            ResourceUse::Texture(use_) => match use_.intent {
                TextureUseIntent::ColorAttachment
                | TextureUseIntent::DepthStencilRead
                | TextureUseIntent::DepthStencilWrite => {}
                // Shader image layouts were transitioned before render-pass
                // begin after the spine pre-scanned this complete scope. A
                // barrier here would be invalid Vulkan.
                TextureUseIntent::ShaderRead | TextureUseIntent::ShaderReadWrite => {}
                _ => {
                    return Err(VulkanFailure::Unsupported {
                        what: "a raster draw texture use outside an attachment or shader binding",
                        why: "copy and resolve uses have separate lowering paths",
                    });
                }
            },
            ResourceUse::Frame(_) => {
                return Err(VulkanFailure::Unsupported {
                    what: "a presentation frame used outside the raster attachment",
                    why: "Vulkan presentation lowering is not implemented yet",
                });
            }
        }
    }
    let mut sets = Vec::with_capacity(draw.groups.len());
    for bound in &draw.groups {
        if !bound.dynamic_offsets.is_empty() {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan raster bind group with dynamic offsets",
                why: "this baseline has no dynamic-offset lowering",
            });
        }
        let group = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<VulkanBindGroup>()
            .ok_or(VulkanFailure::Unsupported {
                what: "a bind group this Vulkan device did not create",
                why: "its descriptor set belongs to another backend",
            })?;
        sets.push((bound.index.get(), group.set()));
        retention.bind_groups.push(bound.group.clone());
    }
    let mut vertex_buffers = Vec::with_capacity(draw.vertex_buffers.len());
    for (slot, binding) in &draw.vertex_buffers {
        vertex_buffers.push((
            *slot,
            native_buffer(&binding.buffer)?.buffer(),
            binding.range.offset,
        ));
        retention.buffers.push(binding.buffer.clone());
    }
    unsafe {
        shared.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline(),
        );
        for (index, set) in sets {
            shared.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout(),
                index,
                &[set],
                &[],
            );
        }
        // Portable vertex slots may be sparse. Binding each one explicitly
        // preserves slot identity instead of compacting slot 3 into binding 0.
        for (slot, buffer, offset) in vertex_buffers {
            shared
                .device
                .cmd_bind_vertex_buffers(command_buffer, slot, &[buffer], &[offset]);
        }
        let viewport = draw.viewport.unwrap_or_else(|| {
            crate::api::command::Viewport::new(
                0.0,
                0.0,
                scope.extent.width as f32,
                scope.extent.height as f32,
                0.0,
                1.0,
            )
        });
        shared.device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport {
                x: viewport.x,
                y: viewport.y,
                width: viewport.width,
                height: viewport.height,
                min_depth: viewport.min_depth,
                max_depth: viewport.max_depth,
            }],
        );
        let scissor = draw.scissor.unwrap_or_else(|| {
            crate::api::command::Rect::new(0, 0, scope.extent.width, scope.extent.height)
        });
        shared.device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D {
                    x: scissor.x as i32,
                    y: scissor.y as i32,
                },
                extent: vk::Extent2D {
                    width: scissor.width,
                    height: scissor.height,
                },
            }],
        );
        if pipeline.uses_blend_constant() {
            shared.device.cmd_set_blend_constants(
                command_buffer,
                &[
                    draw.blend_constant.r,
                    draw.blend_constant.g,
                    draw.blend_constant.b,
                    draw.blend_constant.a,
                ],
            );
        }
        shared.device.cmd_set_stencil_reference(
            command_buffer,
            vk::StencilFaceFlags::FRONT_AND_BACK,
            draw.stencil_reference,
        );
        if let Some(index) = &draw.index {
            shared.device.cmd_bind_index_buffer(
                command_buffer,
                native_buffer(&index.binding.buffer)?.buffer(),
                index.binding.range.offset,
                match index.format {
                    IndexFormat::Uint16 => vk::IndexType::UINT16,
                    IndexFormat::Uint32 => vk::IndexType::UINT32,
                },
            );
            retention.buffers.push(index.binding.buffer.clone());
            shared.device.cmd_draw_indexed(
                command_buffer,
                draw.range.end - draw.range.start,
                draw.instances.end - draw.instances.start,
                draw.range.start,
                draw.base_vertex,
                draw.instances.start,
            );
        } else {
            shared.device.cmd_draw(
                command_buffer,
                draw.range.end - draw.range.start,
                draw.instances.end - draw.instances.start,
                draw.range.start,
                draw.instances.start,
            );
        }
    }
    Ok(retention)
}

/// Rejects the unsupported feedback-loop shape before any native command is
/// recorded. This baseline has no `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` path,
/// and its render pass describes attachments independently from shader image
/// descriptors. Conservatively rejecting the whole texture (rather than trying
/// to prove disjoint mip/layer ranges) keeps the descriptor and attachment
/// layouts unambiguous.
fn reject_raster_feedback(
    begin: &RasterBegin,
    shader_texture_uses: &[TextureUse],
) -> Result<(), VulkanFailure> {
    let mut attachments =
        Vec::with_capacity(begin.colors.len() + usize::from(begin.depth_stencil.is_some()));
    for (_, color) in &begin.colors {
        if let ColorAttachmentView::Texture(view) = &color.view {
            attachments.push(view.texture().id());
        }
    }
    if let Some(depth_stencil) = &begin.depth_stencil {
        attachments.push(depth_stencil.view.texture().id());
    }
    if shader_texture_uses
        .iter()
        .any(|use_| attachments.contains(&use_.texture.id()))
    {
        return Err(VulkanFailure::Unsupported {
            what: "a Vulkan raster attachment also used as a shader image",
            why: "this baseline does not implement Vulkan attachment feedback-loop layouts",
        });
    }
    Ok(())
}

/// Ends the native pass, returns its retained native objects, and transitions
/// attachments to GENERAL so the unified tracker has an explicit, portable
/// cross-scope state.  A later state-diff tracker may retain a more specific
/// layout without changing any RHI contract.
pub(super) fn lower_raster_end(
    command_buffer: vk::CommandBuffer,
    mut scope: RasterScopeState,
    retention: &mut RasterRetention,
    transfer_retention: &mut transfer::TransferRetention,
) -> Result<(), VulkanFailure> {
    unsafe {
        scope
            .objects
            .shared
            .device
            .cmd_end_render_pass(command_buffer)
    };
    for view in scope.colors.drain(..) {
        transfer::transition_raster_attachment(
            &scope.objects.shared,
            command_buffer,
            &view,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            transfer_retention,
        )?;
        retention.views.push(view);
    }
    if let Some(view) = scope.depth.take() {
        transfer::transition_raster_attachment(
            &scope.objects.shared,
            command_buffer,
            &view,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            transfer_retention,
        )?;
        retention.views.push(view);
    }
    retention.objects.push(scope.objects);
    Ok(())
}

fn texture_color_view(view: &ColorAttachmentView) -> Result<TextureView, VulkanFailure> {
    match view {
        ColorAttachmentView::Texture(view) => Ok(view.clone()),
        ColorAttachmentView::Frame(_) => Err(VulkanFailure::Unsupported {
            what: "a Vulkan frame attachment",
            why: "Vulkan swapchain/presentation lowering has not been implemented",
        }),
    }
}
fn native_view(view: &TextureView) -> Result<&VulkanTextureView, VulkanFailure> {
    view.native()
        .as_any()
        .downcast_ref::<VulkanTextureView>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a texture view this Vulkan device did not create",
            why: "its native image view belongs to another backend",
        })
}
fn native_buffer(buffer: &Buffer) -> Result<&VulkanBuffer, VulkanFailure> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<VulkanBuffer>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a buffer this Vulkan device did not create",
            why: "its native allocation belongs to another backend",
        })
}
fn check_extent(
    width: &mut Option<u32>,
    height: &mut Option<u32>,
    extent: crate::api::resource::texture::Extent3d,
) -> Result<(), VulkanFailure> {
    match (*width).zip(*height) {
        Some((known_width, known_height))
            if (known_width, known_height) != (extent.width, extent.height) =>
        {
            Err(VulkanFailure::Unsupported {
                what: "a Vulkan raster scope with mismatched attachment extents",
                why: "portable validation should have made the attachment set uniform",
            })
        }
        Some(_) => Ok(()),
        None => {
            *width = Some(extent.width);
            *height = Some(extent.height);
            Ok(())
        }
    }
}
fn color_load(value: LoadOp<ColorClearValue>) -> vk::AttachmentLoadOp {
    match value {
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
    }
}
fn color_store(value: StoreOp) -> vk::AttachmentStoreOp {
    match value {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
    }
}
fn depth_load(value: Option<DepthAttachmentMode>) -> vk::AttachmentLoadOp {
    match value {
        Some(DepthAttachmentMode::ReadOnly) => vk::AttachmentLoadOp::LOAD,
        Some(DepthAttachmentMode::ReadWrite { load, .. }) => match load {
            LoadOp::Load => vk::AttachmentLoadOp::LOAD,
            LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
        },
        _ => vk::AttachmentLoadOp::DONT_CARE,
    }
}
fn depth_store(value: Option<DepthAttachmentMode>) -> vk::AttachmentStoreOp {
    match value {
        Some(DepthAttachmentMode::ReadOnly) => vk::AttachmentStoreOp::STORE,
        Some(DepthAttachmentMode::ReadWrite { store, .. }) => color_store(store),
        _ => vk::AttachmentStoreOp::DONT_CARE,
    }
}
fn stencil_load(value: Option<StencilAttachmentMode>) -> vk::AttachmentLoadOp {
    match value {
        Some(StencilAttachmentMode::ReadOnly) => vk::AttachmentLoadOp::LOAD,
        Some(StencilAttachmentMode::ReadWrite { load, .. }) => match load {
            LoadOp::Load => vk::AttachmentLoadOp::LOAD,
            LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
        },
        _ => vk::AttachmentLoadOp::DONT_CARE,
    }
}
fn stencil_store(value: Option<StencilAttachmentMode>) -> vk::AttachmentStoreOp {
    match value {
        Some(StencilAttachmentMode::ReadOnly) => vk::AttachmentStoreOp::STORE,
        Some(StencilAttachmentMode::ReadWrite { store, .. }) => color_store(store),
        _ => vk::AttachmentStoreOp::DONT_CARE,
    }
}
fn depth_writes(value: Option<DepthAttachmentMode>) -> bool {
    matches!(value, Some(DepthAttachmentMode::ReadWrite { .. }))
}
fn stencil_writes(value: Option<StencilAttachmentMode>) -> bool {
    matches!(value, Some(StencilAttachmentMode::ReadWrite { .. }))
}
fn clear_color(value: LoadOp<ColorClearValue>) -> vk::ClearValue {
    let color = match value {
        LoadOp::Clear(ColorClearValue::Float(value)) => vk::ClearColorValue { float32: value },
        LoadOp::Clear(ColorClearValue::Sint(value)) => vk::ClearColorValue { int32: value },
        LoadOp::Clear(ColorClearValue::Uint(value)) => vk::ClearColorValue { uint32: value },
        LoadOp::Load => vk::ClearColorValue { float32: [0.0; 4] },
    };
    vk::ClearValue { color }
}
fn clear_depth_stencil(
    depth: Option<DepthAttachmentMode>,
    stencil: Option<StencilAttachmentMode>,
) -> vk::ClearValue {
    let depth = match depth {
        Some(DepthAttachmentMode::ReadWrite {
            load: LoadOp::Clear(value),
            ..
        }) => value,
        _ => 1.0,
    };
    let stencil = match stencil {
        Some(StencilAttachmentMode::ReadWrite {
            load: LoadOp::Clear(value),
            ..
        }) => value,
        _ => 0,
    };
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
    }
}
fn native(operation: &'static str) -> impl FnOnce(vk::Result) -> VulkanFailure {
    move |result| {
        VulkanFailure::Native(crate::backend::vulkan::ffi::NativeError::new(
            result, operation,
        ))
    }
}
