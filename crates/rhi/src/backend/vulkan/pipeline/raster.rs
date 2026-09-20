//! Vulkan graphics-pipeline lowering.
//!
//! This is deliberately a private translation layer.  The portable descriptor
//! has already established device identity, shader-stage compatibility, format
//! support and fixed-state validity; this module turns that validated vocabulary
//! into the Vulkan objects a render-pass command encoder consumes.  It owns the
//! compatible render pass as well as the pipeline: the former is not a public
//! Fluxel object, but Vulkan requires a concrete compatible render pass when
//! dynamic rendering has not been enabled on the device.

use std::any::Any;
use std::ffi::CString;
use std::sync::Arc;

use ash::vk;

use crate::api::format::TextureFormat;
use crate::api::pipeline::backend::RasterPipelineBackend;
use crate::api::pipeline::{
    BlendFactor, BlendOperation, ColorWriteMask, CullMode, FrontFace, PrimitiveTopology,
    RasterPipelineDescriptor, StencilFaceState, StencilOperation, VertexFormat, VertexStepMode,
};
use crate::api::resource::sampler::CompareFunction;
use crate::backend::vulkan::binding::layout_bindings;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::format::vk_format;
use crate::backend::vulkan::pipeline::cache::native_cache;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::shader::VulkanShaderModule;

/// Native state behind one portable raster pipeline.
///
/// `render_pass` is an implementation detail needed by the Vulkan 1.0 baseline.
/// The raster command lowerer must begin this exact compatible pass (or a pass
/// Vulkan considers compatible) before binding `pipeline`.  This avoids making a
/// public `RenderPass` object merely to fit one backend's native ABI.
pub(crate) struct VulkanRasterPipeline {
    shared: Arc<VulkanShared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    render_pass: vk::RenderPass,
    uses_blend_constant: bool,
}

impl VulkanRasterPipeline {
    pub(crate) fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// Vulkan takes blend constants as command state. The raster lowerer maps the
    /// recorded v13 `Color` through `vkCmdSetBlendConstants`; this flag lets its
    /// state cache avoid the call for pipelines that cannot observe it.
    pub(crate) fn uses_blend_constant(&self) -> bool {
        self.uses_blend_constant
    }
}

impl RasterPipelineBackend for VulkanRasterPipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for VulkanRasterPipeline {
    fn drop(&mut self) {
        unsafe {
            // Destruction follows Vulkan parent/child ordering. `shared` is the
            // sole device ownership domain, and therefore outlives every handle.
            self.shared.device.destroy_pipeline(self.pipeline, None);
            self.shared
                .device
                .destroy_render_pass(self.render_pass, None);
            self.shared
                .device
                .destroy_pipeline_layout(self.layout, None);
            for set_layout in self.set_layouts.drain(..) {
                self.shared
                    .device
                    .destroy_descriptor_set_layout(set_layout, None);
            }
        }
    }
}

/// Builds a Vulkan graphics pipeline from a portable descriptor which has
/// already passed the public v13 validator.
pub(in crate::backend::vulkan) fn create_raster_pipeline(
    shared: Arc<VulkanShared>,
    descriptor: &RasterPipelineDescriptor,
) -> Result<VulkanRasterPipeline, VulkanFailure> {
    let vertex = shader_module(&descriptor.vertex, "vertex")?;
    let vertex_entry = entry_point(&descriptor.vertex, "vertex")?;
    let fragment = descriptor
        .fragment
        .as_ref()
        .map(|shader| {
            Ok((
                shader_module(shader, "fragment")?,
                entry_point(shader, "fragment")?,
            ))
        })
        .transpose()?;

    let mut set_layouts = create_set_layouts(&shared, descriptor)?;
    let push_ranges = push_constant_ranges(descriptor.interface.descriptor())?;
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    let layout = match unsafe { shared.device.create_pipeline_layout(&layout_info, None) } {
        Ok(layout) => layout,
        Err(result) => {
            destroy_set_layouts(&shared, &mut set_layouts);
            return Err(native(result, "vkCreatePipelineLayout for raster pipeline"));
        }
    };

    let render_pass = match create_render_pass(&shared, descriptor) {
        Ok(pass) => pass,
        Err(error) => {
            unsafe { shared.device.destroy_pipeline_layout(layout, None) };
            destroy_set_layouts(&shared, &mut set_layouts);
            return Err(error);
        }
    };

    let result = create_graphics_pipeline(
        &shared,
        descriptor,
        vertex,
        &vertex_entry,
        fragment.as_ref().map(|(module, entry)| (*module, entry)),
        layout,
        render_pass,
    );
    let (pipeline, uses_blend_constant) = match result {
        Ok(result) => result,
        Err(error) => {
            unsafe {
                shared.device.destroy_render_pass(render_pass, None);
                shared.device.destroy_pipeline_layout(layout, None);
            }
            destroy_set_layouts(&shared, &mut set_layouts);
            return Err(error);
        }
    };

    Ok(VulkanRasterPipeline {
        shared,
        pipeline,
        layout,
        set_layouts,
        render_pass,
        uses_blend_constant,
    })
}

fn create_set_layouts(
    shared: &VulkanShared,
    descriptor: &RasterPipelineDescriptor,
) -> Result<Vec<vk::DescriptorSetLayout>, VulkanFailure> {
    let mut layouts = Vec::with_capacity(descriptor.interface.descriptor().groups.len());
    for group in &descriptor.interface.descriptor().groups {
        let bindings = layout_bindings(group.descriptor())?;
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        match unsafe { shared.device.create_descriptor_set_layout(&info, None) } {
            Ok(layout) => layouts.push(layout),
            Err(result) => {
                destroy_set_layouts(shared, &mut layouts);
                return Err(native(
                    result,
                    "vkCreateDescriptorSetLayout for raster pipeline",
                ));
            }
        }
    }
    Ok(layouts)
}

fn create_render_pass(
    shared: &VulkanShared,
    descriptor: &RasterPipelineDescriptor,
) -> Result<vk::RenderPass, VulkanFailure> {
    // Vulkan's attachment-reference indices name a dense attachment array; the
    // portable color-target vector may contain holes.  Keep a location->native
    // mapping so fragment output location N still lands on its intended target.
    let mut attachments = Vec::new();
    let mut color_references = Vec::with_capacity(descriptor.color_targets.len());
    for target in &descriptor.color_targets {
        match target {
            Some(target) => {
                let format = texture_format(target.format)?;
                let index = attachments.len() as u32;
                attachments.push(color_attachment(
                    format,
                    samples(descriptor.multisample.count)?,
                ));
                color_references.push(vk::AttachmentReference {
                    attachment: index,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                });
            }
            None => color_references.push(vk::AttachmentReference {
                attachment: vk::ATTACHMENT_UNUSED,
                layout: vk::ImageLayout::UNDEFINED,
            }),
        }
    }

    let depth_reference = descriptor
        .depth_stencil
        .as_ref()
        .map(|state| {
            let index = attachments.len() as u32;
            let format = texture_format(state.format)?;
            attachments.push(depth_attachment(
                format,
                samples(descriptor.multisample.count)?,
            ));
            Ok(vk::AttachmentReference {
                attachment: index,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            })
        })
        .transpose()?;

    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_references);
    if let Some(reference) = depth_reference.as_ref() {
        subpass = subpass.depth_stencil_attachment(reference);
    }
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass));
    unsafe { shared.device.create_render_pass(&info, None) }
        .map_err(|result| native(result, "vkCreateRenderPass for raster pipeline"))
}

fn create_graphics_pipeline(
    shared: &VulkanShared,
    descriptor: &RasterPipelineDescriptor,
    vertex: &VulkanShaderModule,
    vertex_entry: &CString,
    fragment: Option<(&VulkanShaderModule, &CString)>,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<(vk::Pipeline, bool), VulkanFailure> {
    let mut stages = vec![
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex.module())
            .name(vertex_entry),
    ];
    if let Some((fragment, entry)) = fragment {
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment.module())
                .name(entry),
        );
    }

    let (bindings, attributes) = vertex_input(descriptor)?;
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(topology(descriptor.primitive.topology))
        .primitive_restart_enable(descriptor.primitive.strip_index_format.is_some());
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(polygon_mode(descriptor.primitive.polygon_mode))
        .cull_mode(cull_mode(descriptor.primitive.cull_mode))
        .front_face(front_face(descriptor.primitive.front_face))
        .depth_bias_enable(descriptor.primitive.depth_bias.is_some())
        .depth_bias_constant_factor(
            descriptor
                .primitive
                .depth_bias
                .map_or(0.0, |bias| bias.constant as f32),
        )
        .depth_bias_slope_factor(
            descriptor
                .primitive
                .depth_bias
                .map_or(0.0, |bias| bias.slope_scale),
        )
        .depth_bias_clamp(
            descriptor
                .primitive
                .depth_bias
                .map_or(0.0, |bias| bias.clamp),
        )
        .depth_clamp_enable(descriptor.primitive.unclipped_depth)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples(descriptor.multisample.count)?)
        .sample_shading_enable(false)
        .min_sample_shading(0.0)
        .sample_mask(std::slice::from_ref(&descriptor.multisample.mask))
        .alpha_to_coverage_enable(descriptor.multisample.alpha_to_coverage_enabled)
        .alpha_to_one_enable(false);
    let depth_stencil = depth_stencil(descriptor);
    let (attachments, uses_blend_constant) = blend_attachments(descriptor);
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&attachments);
    let dynamic_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_REFERENCE,
    ];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let create = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let mut pipelines = match unsafe {
        shared.device.create_graphics_pipelines(
            native_cache(descriptor.cache.as_ref())?,
            std::slice::from_ref(&create),
            None,
        )
    } {
        Ok(pipelines) => pipelines,
        Err((pipelines, result)) => {
            // Vulkan may return successfully created objects alongside a later
            // create failure when more than one create-info was supplied.  This
            // call has one create-info, but releasing the returned vector keeps
            // the error path correct if a driver returns a partial result.
            unsafe {
                for pipeline in pipelines {
                    shared.device.destroy_pipeline(pipeline, None);
                }
            }
            return Err(native(result, "vkCreateGraphicsPipelines"));
        }
    };
    let pipeline = pipelines.pop().ok_or(VulkanFailure::Unsupported {
        what: "Vulkan graphics-pipeline creation",
        why: "the driver returned no pipeline for one requested create-info",
    })?;
    Ok((pipeline, uses_blend_constant))
}

fn vertex_input(
    descriptor: &RasterPipelineDescriptor,
) -> Result<
    (
        Vec<vk::VertexInputBindingDescription>,
        Vec<vk::VertexInputAttributeDescription>,
    ),
    VulkanFailure,
> {
    let mut bindings = Vec::with_capacity(descriptor.vertex_input.buffers.len());
    let mut attributes = Vec::new();
    for (binding, buffer) in descriptor.vertex_input.buffers.iter().enumerate() {
        bindings.push(
            vk::VertexInputBindingDescription::default()
                .binding(binding as u32)
                .stride(
                    u32::try_from(buffer.stride)
                        .map_err(|_| unsupported("a vertex stride larger than u32"))?,
                )
                .input_rate(match buffer.step_mode {
                    VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                }),
        );
        for attribute in &buffer.attributes {
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .location(attribute.location.get())
                    .binding(binding as u32)
                    .format(vertex_format(attribute.format))
                    .offset(
                        u32::try_from(attribute.offset).map_err(|_| {
                            unsupported("a vertex attribute offset larger than u32")
                        })?,
                    ),
            );
        }
    }
    Ok((bindings, attributes))
}

fn blend_attachments(
    descriptor: &RasterPipelineDescriptor,
) -> (Vec<vk::PipelineColorBlendAttachmentState>, bool) {
    let mut uses_constant = false;
    let attachments = descriptor
        .color_targets
        .iter()
        .map(|target| match target {
            None => vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::empty()),
            Some(target) => {
                let blend = target.blend;
                if let Some(blend) = blend {
                    uses_constant |= uses_blend_constant(blend.color.src_factor)
                        || uses_blend_constant(blend.color.dst_factor)
                        || uses_blend_constant(blend.alpha.src_factor)
                        || uses_blend_constant(blend.alpha.dst_factor);
                }
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(blend.is_some())
                    .src_color_blend_factor(blend.map_or(vk::BlendFactor::ONE, |state| {
                        blend_factor(state.color.src_factor)
                    }))
                    .dst_color_blend_factor(blend.map_or(vk::BlendFactor::ZERO, |state| {
                        blend_factor(state.color.dst_factor)
                    }))
                    .color_blend_op(
                        blend.map_or(vk::BlendOp::ADD, |state| blend_op(state.color.operation)),
                    )
                    .src_alpha_blend_factor(blend.map_or(vk::BlendFactor::ONE, |state| {
                        blend_factor(state.alpha.src_factor)
                    }))
                    .dst_alpha_blend_factor(blend.map_or(vk::BlendFactor::ZERO, |state| {
                        blend_factor(state.alpha.dst_factor)
                    }))
                    .alpha_blend_op(
                        blend.map_or(vk::BlendOp::ADD, |state| blend_op(state.alpha.operation)),
                    )
                    .color_write_mask(color_write_mask(target.write_mask))
            }
        })
        .collect();
    (attachments, uses_constant)
}

fn depth_stencil(
    descriptor: &RasterPipelineDescriptor,
) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    let depth = descriptor
        .depth_stencil
        .as_ref()
        .and_then(|state| state.depth);
    let stencil = descriptor
        .depth_stencil
        .as_ref()
        .and_then(|state| state.stencil);
    vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth.is_some())
        .depth_write_enable(depth.is_some_and(|state| state.write_enabled))
        .depth_compare_op(depth.map_or(vk::CompareOp::ALWAYS, |state| compare(state.compare)))
        .depth_bounds_test_enable(false)
        .stencil_test_enable(stencil.is_some())
        .front(stencil.map_or(default_stencil_face(), |state| {
            stencil_face(state.front, state.read_mask, state.write_mask)
        }))
        .back(stencil.map_or(default_stencil_face(), |state| {
            stencil_face(state.back, state.read_mask, state.write_mask)
        }))
}

fn color_attachment(
    format: vk::Format,
    samples: vk::SampleCountFlags,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription::default()
        .format(format)
        .samples(samples)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
}

fn depth_attachment(
    format: vk::Format,
    samples: vk::SampleCountFlags,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription::default()
        .format(format)
        .samples(samples)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::LOAD)
        .stencil_store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
}

fn shader_module<'a>(
    shader: &'a crate::api::shader::ShaderModule,
    stage: &'static str,
) -> Result<&'a VulkanShaderModule, VulkanFailure> {
    shader
        .native()
        .as_any()
        .downcast_ref::<VulkanShaderModule>()
        .ok_or(VulkanFailure::Unsupported {
            what: "a raster shader from another backend",
            why: stage,
        })
}

fn entry_point(
    shader: &crate::api::shader::ShaderModule,
    stage: &'static str,
) -> Result<CString, VulkanFailure> {
    CString::new(shader.artifact().entry_point.as_str()).map_err(|_| VulkanFailure::Unsupported {
        what: "a raster shader entry-point name containing a NUL byte",
        why: stage,
    })
}

fn push_constant_ranges(
    interface: &crate::api::pipeline::PipelineInterfaceDescriptor,
) -> Result<Vec<vk::PushConstantRange>, VulkanFailure> {
    interface
        .immediate_ranges
        .iter()
        .map(|range| {
            let mut stages = vk::ShaderStageFlags::empty();
            if range.visibility.contains(crate::api::shader::ShaderStages::VERTEX) {
                stages |= vk::ShaderStageFlags::VERTEX;
            }
            if range.visibility.contains(crate::api::shader::ShaderStages::FRAGMENT) {
                stages |= vk::ShaderStageFlags::FRAGMENT;
            }
            if stages.is_empty() {
                return Err(VulkanFailure::Unsupported {
                    what: "a raster immediate range with unsupported stage visibility",
                    why: "this Vulkan raster pipeline only lowers vertex and fragment push constants",
                });
            }
            Ok(vk::PushConstantRange { stage_flags: stages, offset: range.offset, size: range.size })
        })
        .collect()
}

fn texture_format(format: TextureFormat) -> Result<vk::Format, VulkanFailure> {
    vk_format(format).ok_or(VulkanFailure::Unsupported {
        what: "a raster attachment format without a Vulkan mapping",
        why: "the dedicated Vulkan image layer does not expose this portable format",
    })
}

fn samples(count: u32) -> Result<vk::SampleCountFlags, VulkanFailure> {
    Ok(match count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => return Err(unsupported("a non-Vulkan raster sample count")),
    })
}

fn vertex_format(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Uint8 => vk::Format::R8_UINT,
        VertexFormat::Uint8x2 => vk::Format::R8G8_UINT,
        VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
        VertexFormat::Sint8 => vk::Format::R8_SINT,
        VertexFormat::Sint8x2 => vk::Format::R8G8_SINT,
        VertexFormat::Sint8x4 => vk::Format::R8G8B8A8_SINT,
        VertexFormat::Unorm8 => vk::Format::R8_UNORM,
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
        VertexFormat::Uint32x2 => vk::Format::R32G32_UINT,
        VertexFormat::Uint32x3 => vk::Format::R32G32B32_UINT,
        VertexFormat::Uint32x4 => vk::Format::R32G32B32A32_UINT,
        VertexFormat::Sint32 => vk::Format::R32_SINT,
        VertexFormat::Sint32x2 => vk::Format::R32G32_SINT,
        VertexFormat::Sint32x3 => vk::Format::R32G32B32_SINT,
        VertexFormat::Sint32x4 => vk::Format::R32G32B32A32_SINT,
        VertexFormat::Unorm8x2 => vk::Format::R8G8_UNORM,
        VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
        VertexFormat::Unorm8x4Bgra => vk::Format::B8G8R8A8_UNORM,
        VertexFormat::Snorm8 => vk::Format::R8_SNORM,
        VertexFormat::Snorm8x2 => vk::Format::R8G8_SNORM,
        VertexFormat::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
        VertexFormat::Uint16 => vk::Format::R16_UINT,
        VertexFormat::Uint16x2 => vk::Format::R16G16_UINT,
        VertexFormat::Uint16x4 => vk::Format::R16G16B16A16_UINT,
        VertexFormat::Sint16 => vk::Format::R16_SINT,
        VertexFormat::Sint16x2 => vk::Format::R16G16_SINT,
        VertexFormat::Sint16x4 => vk::Format::R16G16B16A16_SINT,
        VertexFormat::Unorm16 => vk::Format::R16_UNORM,
        VertexFormat::Unorm16x2 => vk::Format::R16G16_UNORM,
        VertexFormat::Unorm16x4 => vk::Format::R16G16B16A16_UNORM,
        VertexFormat::Snorm16 => vk::Format::R16_SNORM,
        VertexFormat::Snorm16x2 => vk::Format::R16G16_SNORM,
        VertexFormat::Snorm16x4 => vk::Format::R16G16B16A16_SNORM,
        VertexFormat::Float16 => vk::Format::R16_SFLOAT,
        VertexFormat::Float16x2 => vk::Format::R16G16_SFLOAT,
        VertexFormat::Float16x4 => vk::Format::R16G16B16A16_SFLOAT,
        VertexFormat::Float64 => vk::Format::R64_SFLOAT,
        VertexFormat::Float64x2 => vk::Format::R64G64_SFLOAT,
        VertexFormat::Float64x3 => vk::Format::R64G64B64_SFLOAT,
        VertexFormat::Float64x4 => vk::Format::R64G64B64A64_SFLOAT,
        VertexFormat::Unorm10_10_10_2 => vk::Format::A2B10G10R10_UNORM_PACK32,
    }
}

fn polygon_mode(mode: crate::api::pipeline::PolygonMode) -> vk::PolygonMode {
    match mode {
        crate::api::pipeline::PolygonMode::Fill => vk::PolygonMode::FILL,
        crate::api::pipeline::PolygonMode::Line => vk::PolygonMode::LINE,
        crate::api::pipeline::PolygonMode::Point => vk::PolygonMode::POINT,
    }
}

fn topology(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

fn cull_mode(mode: CullMode) -> vk::CullModeFlags {
    match mode {
        CullMode::None => vk::CullModeFlags::NONE,
        CullMode::Front => vk::CullModeFlags::FRONT,
        CullMode::Back => vk::CullModeFlags::BACK,
    }
}

fn front_face(face: FrontFace) -> vk::FrontFace {
    match face {
        FrontFace::Ccw => vk::FrontFace::COUNTER_CLOCKWISE,
        FrontFace::Cw => vk::FrontFace::CLOCKWISE,
    }
}

fn color_write_mask(mask: ColorWriteMask) -> vk::ColorComponentFlags {
    let mut flags = vk::ColorComponentFlags::empty();
    if mask.contains(ColorWriteMask::RED) {
        flags |= vk::ColorComponentFlags::R;
    }
    if mask.contains(ColorWriteMask::GREEN) {
        flags |= vk::ColorComponentFlags::G;
    }
    if mask.contains(ColorWriteMask::BLUE) {
        flags |= vk::ColorComponentFlags::B;
    }
    if mask.contains(ColorWriteMask::ALPHA) {
        flags |= vk::ColorComponentFlags::A;
    }
    flags
}

fn blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::Src => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::Src1 => vk::BlendFactor::SRC1_COLOR,
        BlendFactor::OneMinusSrc1 => vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        BlendFactor::Src1Alpha => vk::BlendFactor::SRC1_ALPHA,
        BlendFactor::OneMinusSrc1Alpha => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
        BlendFactor::Dst => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        BlendFactor::Constant => vk::BlendFactor::CONSTANT_COLOR,
        BlendFactor::OneMinusConstant => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
    }
}

fn uses_blend_constant(factor: BlendFactor) -> bool {
    matches!(
        factor,
        BlendFactor::Constant | BlendFactor::OneMinusConstant
    )
}

fn blend_op(operation: BlendOperation) -> vk::BlendOp {
    match operation {
        BlendOperation::Add => vk::BlendOp::ADD,
        BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOperation::Min => vk::BlendOp::MIN,
        BlendOperation::Max => vk::BlendOp::MAX,
    }
}

fn compare(compare: CompareFunction) -> vk::CompareOp {
    match compare {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

fn default_stencil_face() -> vk::StencilOpState {
    vk::StencilOpState::default()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .depth_fail_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS)
        .compare_mask(u32::MAX)
        .write_mask(u32::MAX)
        .reference(0)
}

fn stencil_face(face: StencilFaceState, read_mask: u32, write_mask: u32) -> vk::StencilOpState {
    vk::StencilOpState::default()
        .fail_op(stencil_op(face.fail_op))
        .pass_op(stencil_op(face.pass_op))
        .depth_fail_op(stencil_op(face.depth_fail_op))
        .compare_op(compare(face.compare))
        .compare_mask(read_mask)
        .write_mask(write_mask)
        .reference(0)
}

fn stencil_op(operation: StencilOperation) -> vk::StencilOp {
    match operation {
        StencilOperation::Keep => vk::StencilOp::KEEP,
        StencilOperation::Zero => vk::StencilOp::ZERO,
        StencilOperation::Replace => vk::StencilOp::REPLACE,
        StencilOperation::Invert => vk::StencilOp::INVERT,
        StencilOperation::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOperation::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOperation::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOperation::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

fn destroy_set_layouts(shared: &VulkanShared, layouts: &mut Vec<vk::DescriptorSetLayout>) {
    unsafe {
        for layout in layouts.drain(..) {
            shared.device.destroy_descriptor_set_layout(layout, None);
        }
    }
}

fn unsupported(what: &'static str) -> VulkanFailure {
    VulkanFailure::Unsupported {
        what,
        why: "the Vulkan raster-pipeline lowerer cannot represent it",
    }
}

fn native(result: vk::Result, operation: &'static str) -> VulkanFailure {
    VulkanFailure::Native(ffi::NativeError::new(result, operation))
}
