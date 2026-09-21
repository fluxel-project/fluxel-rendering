//! Pure CPU raster packet construction shared by native GL and WebGL2.
//!
//! Object-name lookup and GL calls deliberately stay outside this module.  A
//! The packet builder derives the GL binding layout from the immutable shader
//! artifacts and consumes the packet atomically after every conversion below
//! has succeeded.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::raster_state::{
    BlendComponent, BlendFactor, BlendOperation, ColorWriteMask, CullMode, FrontFace,
    PrimitiveTopology, StencilFaceState, StencilOperation,
};
use crate::api::pipeline::{RasterPipelineDescriptor, VertexStepMode};

use super::api::*;
use super::translate::{
    pipeline_layout_from_artifacts, shader_source, texture_format, vertex_format,
};

/// Whether a raster scope targets an allocated framebuffer or the one default
/// framebuffer supplied by the surface owner.  The latter has no invented
/// Texture/View handle: that distinction is essential for WebGL2.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlFramebufferCarrier {
    Offscreen(GlFramebufferDescriptor),
    Default { color_locations: Vec<u32> },
}

/// Converts the attachment shape of a recorded raster begin. `resolve_view`
/// returns `None` only for a FrameAttachment, which becomes the default
/// framebuffer carrier. Mixing frame and texture attachments is refused before
/// a framebuffer is allocated or bound.
pub(crate) fn raster_begin_carrier(
    begin: &crate::api::command::record::RasterBegin,
    mut resolve_view: impl FnMut(
        &crate::api::command::ColorAttachmentView,
    ) -> RhiResult<Option<GlTextureView>>,
) -> RhiResult<GlFramebufferCarrier> {
    let mut views = Vec::new();
    let mut locations = Vec::new();
    let mut has_default = false;
    for (location, attachment) in &begin.colors {
        match resolve_view(&attachment.view)? {
            Some(view) => {
                views.push(view);
                locations.push(*location);
            }
            None => {
                has_default = true;
                locations.push(*location);
            }
        }
    }
    if has_default {
        if !views.is_empty() || begin.depth_stencil.is_some() || locations.len() != 1 {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "a default framebuffer cannot be mixed with texture/depth attachments",
            )
            .at("GL::begin_raster"));
        }
        return Ok(GlFramebufferCarrier::Default {
            color_locations: locations,
        });
    }
    let depth_stencil_attachment = match &begin.depth_stencil {
        Some(attachment) => resolve_view(&crate::api::command::ColorAttachmentView::Texture(
            attachment.view.clone(),
        ))?,
        None => None,
    };
    Ok(GlFramebufferCarrier::Offscreen(GlFramebufferDescriptor {
        color_attachments: views,
        depth_stencil_attachment,
        draw_buffers: locations,
    }))
}

/// Scalar state from one recorded draw after object-backed pipeline/bind-group
/// lowering has been selected.  Buffer object lookup stays with the owner
/// driver, but ranges, offsets and dynamic state cannot be reconstructed there
/// and are therefore preserved in this CPU packet.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GlRasterDrawScalars {
    pub viewport: Option<GlViewport>,
    pub scissor: Option<GlScissorRect>,
    pub blend_constant: [u32; 4],
    pub stencil_reference: u32,
    pub draw: GlAdvancedDrawCommand,
    pub bind_groups: Vec<(u32, crate::api::identity::ObjectId, Vec<u32>)>,
}

pub(crate) fn raster_draw_scalars(
    draw: &crate::api::command::record::RasterDraw,
) -> RhiResult<GlRasterDrawScalars> {
    let count = draw
        .range
        .end
        .checked_sub(draw.range.start)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "recorded draw range is inverted",
            )
            .at("GL::submit")
        })?;
    let instances = draw
        .instances
        .end
        .checked_sub(draw.instances.start)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "recorded instance range is inverted",
            )
            .at("GL::submit")
        })?;
    let command = match draw.index {
        Some(_) => GlDrawCommand::Indexed(GlIndexedDraw {
            first_index: draw.range.start,
            index_count: count,
            instance_count: instances,
        }),
        None => GlDrawCommand::NonIndexed(GlNonIndexedDraw {
            first_vertex: draw.range.start,
            vertex_count: count,
            instance_count: instances,
        }),
    };
    let viewport = draw
        .viewport
        .map(|v| {
            if v.x < 0.0
                || v.y < 0.0
                || v.width < 0.0
                || v.height < 0.0
                || !v.x.is_finite()
                || !v.y.is_finite()
                || !v.width.is_finite()
                || !v.height.is_finite()
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "GL viewport needs finite non-negative integer coordinates",
                )
                .at("GL::submit"));
            }
            Ok(GlViewport {
                x: v.x as u32,
                y: v.y as u32,
                width: v.width as u32,
                height: v.height as u32,
                min_depth: v.min_depth.to_bits(),
                max_depth: v.max_depth.to_bits(),
            })
        })
        .transpose()?;
    let scissor = draw.scissor.map(|s| GlScissorRect {
        x: s.x,
        y: s.y,
        width: s.width,
        height: s.height,
    });
    Ok(GlRasterDrawScalars {
        viewport,
        scissor,
        blend_constant: [
            draw.blend_constant.r.to_bits(),
            draw.blend_constant.g.to_bits(),
            draw.blend_constant.b.to_bits(),
            draw.blend_constant.a.to_bits(),
        ],
        stencil_reference: draw.stencil_reference,
        draw: GlAdvancedDrawCommand {
            draw: command,
            base_vertex: draw.base_vertex,
            first_instance: draw.instances.start,
        },
        bind_groups: draw
            .groups
            .iter()
            .map(|group| {
                (
                    group.index.get(),
                    group.group.id(),
                    group.dynamic_offsets.clone(),
                )
            })
            .collect(),
    })
}

fn unsupported(detail: impl Into<String>) -> RhiError {
    RhiError::new(RhiErrorKind::Unsupported, detail).at("GL::create_raster_pipeline")
}

/// A native-independent raster construction result.  Program linking, VAO
/// allocation and state application consume the three values independently.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlRasterPipelinePacket {
    pub program: GlProgramDescriptor,
    pub vertex_layout: GlVertexLayout,
    pub state: GlRasterState,
}

/// Builds the GL program/state packet from a fully portable descriptor.
pub(crate) fn raster_pipeline_packet(
    desc: &RasterPipelineDescriptor,
    default_viewport: GlViewport,
) -> RhiResult<GlRasterPipelinePacket> {
    let Some(fragment_module) = desc.fragment.as_ref() else {
        return Err(unsupported(
            "GL-family raster program requires an explicit fragment artifact",
        ));
    };
    let vertex = shader_source(desc.vertex.artifact())?;
    let fragment = shader_source(fragment_module.artifact())?;
    if vertex.stage != GlShaderStage::Vertex || fragment.stage != GlShaderStage::Fragment {
        return Err(unsupported(
            "raster descriptor did not contain vertex + fragment stages",
        ));
    }
    // The layout must be reconstructed from the two immutable artifacts and
    // the pipeline interface.  A driver-supplied empty layout used to make a
    // resource-bearing descriptor appear bindable; the shared lowering owns the
    // contract instead, so both browser and native routes get the same refusal.
    let layout = pipeline_layout_from_artifacts(
        &desc.interface,
        &[desc.vertex.artifact(), fragment_module.artifact()],
    )?;
    let program = GlProgramDescriptor {
        kind: GlProgramKind::Raster { vertex, fragment },
        layout,
        debug_name: desc.label.0.clone(),
    };
    let mut buffers = Vec::with_capacity(desc.vertex_input.buffers.len());
    let mut attributes = Vec::new();
    for (slot, buffer) in desc.vertex_input.buffers.iter().enumerate() {
        let slot = u32::try_from(slot)
            .map_err(|_| unsupported("vertex buffer slot exceeds GL u32 namespace"))?;
        let stride = u32::try_from(buffer.stride)
            .map_err(|_| unsupported("vertex stride exceeds GLsizei"))?;
        buffers.push(GlVertexBufferLayout {
            slot,
            stride,
            step_mode: match buffer.step_mode {
                VertexStepMode::Vertex => GlVertexStepMode::Vertex,
                VertexStepMode::Instance => GlVertexStepMode::Instance,
            },
        });
        for attribute in &buffer.attributes {
            attributes.push(GlVertexAttribute {
                location: attribute.location.get(),
                buffer_slot: slot,
                format: vertex_format(attribute.format)?,
                offset: u32::try_from(attribute.offset)
                    .map_err(|_| unsupported("vertex attribute offset exceeds GLsizei"))?,
            });
        }
    }
    let color_targets = desc
        .color_targets
        .iter()
        .map(|target| match target {
            None => Ok(GlColorTargetState {
                write_mask: 0,
                blend: None,
            }),
            Some(target) => Ok(GlColorTargetState {
                write_mask: color_mask(target.write_mask),
                blend: target.blend.map(blend_state).transpose()?,
            }),
        })
        .collect::<RhiResult<Vec<_>>>()?;
    let depth_stencil = desc
        .depth_stencil
        .as_ref()
        .map(depth_stencil_state)
        .transpose()?;
    let topology = match desc.primitive.topology {
        PrimitiveTopology::PointList => GlPrimitiveTopology::Points,
        PrimitiveTopology::LineList => GlPrimitiveTopology::Lines,
        PrimitiveTopology::LineStrip => GlPrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => GlPrimitiveTopology::Triangles,
        PrimitiveTopology::TriangleStrip => GlPrimitiveTopology::TriangleStrip,
    };
    // Polygon mode, unclipped depth and conservative coverage have no baseline
    // GL-family contract here.  They must have been capability-refused before
    // this packet is requested; keeping a non-default value fail-closed also
    // protects an embedding that calls this helper directly.
    if !matches!(
        desc.primitive.polygon_mode,
        crate::api::pipeline::PolygonMode::Fill
    ) || desc.primitive.unclipped_depth
        || desc.primitive.conservative
    {
        return Err(unsupported(
            "requested raster state needs an optional GL route",
        ));
    }
    Ok(GlRasterPipelinePacket {
        program,
        vertex_layout: GlVertexLayout {
            buffers,
            attributes,
        },
        state: GlRasterState {
            topology,
            cull_mode: match desc.primitive.cull_mode {
                CullMode::None => GlCullMode::None,
                CullMode::Front => GlCullMode::Front,
                CullMode::Back => GlCullMode::Back,
            },
            front_face: match desc.primitive.front_face {
                FrontFace::Ccw => GlFrontFace::CounterClockwise,
                FrontFace::Cw => GlFrontFace::Clockwise,
            },
            depth_stencil,
            color_targets,
            multisample: GlMultisampleState {
                sample_count: desc.multisample.count,
                alpha_to_coverage_enabled: desc.multisample.alpha_to_coverage_enabled,
                sample_mask: desc.multisample.mask,
            },
            viewport: default_viewport,
            scissor: None,
            blend_constant: [0.0f32.to_bits(); 4],
        },
    })
}

fn color_mask(mask: ColorWriteMask) -> u8 {
    [
        (ColorWriteMask::RED, 1),
        (ColorWriteMask::GREEN, 2),
        (ColorWriteMask::BLUE, 4),
        (ColorWriteMask::ALPHA, 8),
    ]
    .into_iter()
    .fold(0, |bits, (flag, bit)| {
        if mask.contains(flag) {
            bits | bit
        } else {
            bits
        }
    })
}
fn blend_state(value: crate::api::pipeline::BlendState) -> RhiResult<GlBlendState> {
    Ok(GlBlendState {
        color: blend_component(value.color)?,
        alpha: blend_component(value.alpha)?,
    })
}
fn blend_component(value: BlendComponent) -> RhiResult<GlBlendComponent> {
    Ok(GlBlendComponent {
        src_factor: blend_factor(value.src_factor)?,
        dst_factor: blend_factor(value.dst_factor)?,
        operation: match value.operation {
            BlendOperation::Add => GlBlendOperation::Add,
            BlendOperation::Subtract => GlBlendOperation::Subtract,
            BlendOperation::ReverseSubtract => GlBlendOperation::ReverseSubtract,
            BlendOperation::Min => GlBlendOperation::Min,
            BlendOperation::Max => GlBlendOperation::Max,
        },
    })
}
fn blend_factor(value: BlendFactor) -> RhiResult<GlBlendFactor> {
    Ok(match value {
        BlendFactor::Zero => GlBlendFactor::Zero,
        BlendFactor::One => GlBlendFactor::One,
        BlendFactor::Src => GlBlendFactor::Src,
        BlendFactor::OneMinusSrc => GlBlendFactor::OneMinusSrc,
        BlendFactor::SrcAlpha => GlBlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => GlBlendFactor::OneMinusSrcAlpha,
        BlendFactor::Dst => GlBlendFactor::Dst,
        BlendFactor::OneMinusDst => GlBlendFactor::OneMinusDst,
        BlendFactor::DstAlpha => GlBlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => GlBlendFactor::OneMinusDstAlpha,
        BlendFactor::Constant => GlBlendFactor::Constant,
        BlendFactor::OneMinusConstant => GlBlendFactor::OneMinusConstant,
        BlendFactor::SrcAlphaSaturated => GlBlendFactor::SrcAlphaSaturated,
        BlendFactor::Src1
        | BlendFactor::OneMinusSrc1
        | BlendFactor::Src1Alpha
        | BlendFactor::OneMinusSrc1Alpha => {
            return Err(unsupported("dual-source blend needs an optional GL route"));
        }
    })
}
fn compare(value: crate::api::resource::CompareFunction) -> GlDepthCompareFunction {
    match value {
        crate::api::resource::CompareFunction::Never => GlDepthCompareFunction::Never,
        crate::api::resource::CompareFunction::Less => GlDepthCompareFunction::Less,
        crate::api::resource::CompareFunction::Equal => GlDepthCompareFunction::Equal,
        crate::api::resource::CompareFunction::LessEqual => GlDepthCompareFunction::LessEqual,
        crate::api::resource::CompareFunction::Greater => GlDepthCompareFunction::Greater,
        crate::api::resource::CompareFunction::NotEqual => GlDepthCompareFunction::NotEqual,
        crate::api::resource::CompareFunction::GreaterEqual => GlDepthCompareFunction::GreaterEqual,
        crate::api::resource::CompareFunction::Always => GlDepthCompareFunction::Always,
    }
}
fn stencil_face(value: StencilFaceState, read_mask: u32, write_mask: u32) -> GlStencilFaceState {
    GlStencilFaceState {
        compare: compare(value.compare),
        fail_op: stencil_op(value.fail_op),
        depth_fail_op: stencil_op(value.depth_fail_op),
        pass_op: stencil_op(value.pass_op),
        read_mask,
        write_mask,
    }
}
fn stencil_op(value: StencilOperation) -> GlStencilOperation {
    match value {
        StencilOperation::Keep => GlStencilOperation::Keep,
        StencilOperation::Zero => GlStencilOperation::Zero,
        StencilOperation::Replace => GlStencilOperation::Replace,
        StencilOperation::IncrementClamp => GlStencilOperation::IncrementClamp,
        StencilOperation::DecrementClamp => GlStencilOperation::DecrementClamp,
        StencilOperation::Invert => GlStencilOperation::Invert,
        StencilOperation::IncrementWrap => GlStencilOperation::IncrementWrap,
        StencilOperation::DecrementWrap => GlStencilOperation::DecrementWrap,
    }
}
fn depth_stencil_state(
    value: &crate::api::pipeline::DepthStencilState,
) -> RhiResult<GlDepthStencilState> {
    texture_format(value.format)?;
    let depth = value.depth.unwrap_or(crate::api::pipeline::DepthState::new(
        crate::api::resource::CompareFunction::Always,
    ));
    let stencil = value
        .stencil
        .unwrap_or(crate::api::pipeline::StencilState::new(
            StencilFaceState::new(crate::api::resource::CompareFunction::Always),
            StencilFaceState::new(crate::api::resource::CompareFunction::Always),
        ));
    Ok(GlDepthStencilState {
        depth_write_enabled: depth.write_enabled,
        depth_compare: compare(depth.compare),
        depth_bias: [0; 3],
        stencil_front: stencil_face(stencil.front, stencil.read_mask, stencil.write_mask),
        stencil_back: stencil_face(stencil.back, stencil.read_mask, stencil.write_mask),
        stencil_reference: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn color_write_mask_preserves_each_channel() {
        assert_eq!(
            color_mask(ColorWriteMask::RED.union(ColorWriteMask::BLUE)),
            0b0101
        );
        assert_eq!(color_mask(ColorWriteMask::NONE), 0);
    }
    #[test]
    fn dual_source_blending_is_never_silently_rewritten() {
        assert_eq!(
            blend_factor(BlendFactor::Src1).unwrap_err().kind(),
            RhiErrorKind::Unsupported
        );
    }
    #[test]
    fn stencil_operation_and_comparison_are_lossless() {
        assert_eq!(
            stencil_op(StencilOperation::IncrementWrap),
            GlStencilOperation::IncrementWrap
        );
        assert_eq!(
            compare(crate::api::resource::CompareFunction::GreaterEqual),
            GlDepthCompareFunction::GreaterEqual
        );
    }
}
