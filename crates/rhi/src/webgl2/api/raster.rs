//! Complete fixed raster state and explicit draw payloads.

use super::{GlError, GlFamilyApi, ProgramId, VertexArrayId};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlPrimitiveTopology {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlCullMode {
    None,
    Front,
    Back,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlFrontFace {
    Clockwise,
    CounterClockwise,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlDepthCompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlStencilOperation {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlBlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    Constant,
    OneMinusConstant,
    SrcAlphaSaturated,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlBlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlStencilFaceState {
    pub compare: GlDepthCompareFunction,
    pub fail_op: GlStencilOperation,
    pub depth_fail_op: GlStencilOperation,
    pub pass_op: GlStencilOperation,
    pub read_mask: u32,
    pub write_mask: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlDepthStencilState {
    pub depth_write_enabled: bool,
    pub depth_compare: GlDepthCompareFunction,
    pub depth_bias: [u32; 3],
    pub stencil_front: GlStencilFaceState,
    pub stencil_back: GlStencilFaceState,
    pub stencil_reference: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlBlendComponent {
    pub src_factor: GlBlendFactor,
    pub dst_factor: GlBlendFactor,
    pub operation: GlBlendOperation,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlBlendState {
    pub color: GlBlendComponent,
    pub alpha: GlBlendComponent,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlColorTargetState {
    pub write_mask: u8,
    pub blend: Option<GlBlendState>,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlViewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub min_depth: u32,
    pub max_depth: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlMultisampleState {
    pub sample_count: u32,
    pub alpha_to_coverage_enabled: bool,
    pub sample_mask: u32,
}

/// Entire fixed pipeline state, with all floating values represented as
/// IEEE-754 bits. This is a valid Layer 2 structural cache key.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlRasterState {
    pub topology: GlPrimitiveTopology,
    pub cull_mode: GlCullMode,
    pub front_face: GlFrontFace,
    pub depth_stencil: Option<GlDepthStencilState>,
    pub color_targets: Vec<GlColorTargetState>,
    pub multisample: GlMultisampleState,
    pub viewport: GlViewport,
    pub scissor: Option<GlScissorRect>,
    pub blend_constant: [u32; 4],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlNonIndexedDraw {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlIndexedDraw {
    pub first_index: u32,
    pub index_count: u32,
    pub instance_count: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlDrawCommand {
    NonIndexed(GlNonIndexedDraw),
    Indexed(GlIndexedDraw),
}
/// Opt-in offsets are separate from the portable WebGL2/GL4/GLES3 core draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlAdvancedDrawCommand {
    pub draw: GlDrawCommand,
    pub base_vertex: i32,
    pub first_instance: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlAdvancedRasterCapabilities {
    pub base_vertex: bool,
    pub first_instance: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlRasterValidationError {
    ZeroViewport,
    ViewportOutOfBounds,
    InvalidDepthRange,
    ScissorOutOfBounds,
    InvalidWriteMask,
    ZeroSampleCount,
    SampleCountExceedsLimit,
    AlphaToCoverageWithoutMultisample,
    NonFiniteFloat,
    MissingAdvancedCapability,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlRasterValidationInfo {
    pub width: u32,
    pub height: u32,
    pub max_samples: u32,
    pub max_color_targets: u32,
}

impl GlRasterState {
    pub(crate) fn validate(
        &self,
        info: GlRasterValidationInfo,
    ) -> Result<(), GlRasterValidationError> {
        if self.color_targets.len() > info.max_color_targets as usize {
            return Err(GlRasterValidationError::InvalidWriteMask);
        }
        if self
            .color_targets
            .iter()
            .any(|target| target.write_mask & !0x0f != 0)
        {
            return Err(GlRasterValidationError::InvalidWriteMask);
        }
        if self.multisample.sample_count == 0 {
            return Err(GlRasterValidationError::ZeroSampleCount);
        }
        if self.multisample.sample_count > info.max_samples {
            return Err(GlRasterValidationError::SampleCountExceedsLimit);
        }
        if self.multisample.alpha_to_coverage_enabled && self.multisample.sample_count == 1 {
            return Err(GlRasterValidationError::AlphaToCoverageWithoutMultisample);
        }
        let end_x = self.viewport.x.checked_add(self.viewport.width);
        let end_y = self.viewport.y.checked_add(self.viewport.height);
        if self.viewport.width == 0 || self.viewport.height == 0 {
            return Err(GlRasterValidationError::ZeroViewport);
        }
        if end_x.is_none_or(|end| end > info.width) || end_y.is_none_or(|end| end > info.height) {
            return Err(GlRasterValidationError::ViewportOutOfBounds);
        }
        let min = f32::from_bits(self.viewport.min_depth);
        let max = f32::from_bits(self.viewport.max_depth);
        if !min.is_finite()
            || !max.is_finite()
            || !(0.0..=1.0).contains(&min)
            || !(0.0..=1.0).contains(&max)
            || min > max
        {
            return Err(GlRasterValidationError::InvalidDepthRange);
        }
        if let Some(scissor) = self.scissor {
            if scissor
                .x
                .checked_add(scissor.width)
                .is_none_or(|end| end > info.width)
                || scissor
                    .y
                    .checked_add(scissor.height)
                    .is_none_or(|end| end > info.height)
            {
                return Err(GlRasterValidationError::ScissorOutOfBounds);
            }
        }
        if self
            .blend_constant
            .iter()
            .any(|bits| !f32::from_bits(*bits).is_finite())
            || self.depth_stencil.is_some_and(|state| {
                state
                    .depth_bias
                    .iter()
                    .any(|bits| !f32::from_bits(*bits).is_finite())
            })
        {
            return Err(GlRasterValidationError::NonFiniteFloat);
        }
        Ok(())
    }
}
impl GlAdvancedDrawCommand {
    pub(crate) fn validate(
        self,
        capabilities: GlAdvancedRasterCapabilities,
    ) -> Result<(), GlRasterValidationError> {
        if self.base_vertex != 0 && !capabilities.base_vertex
            || self.first_instance != 0 && !capabilities.first_instance
        {
            Err(GlRasterValidationError::MissingAdvancedCapability)
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlRasterPipeline {
    pub program: ProgramId,
    pub vertex_array: VertexArrayId,
    pub state: GlRasterState,
}

/// Executable raster domain. A provider must have an active render pass before
/// accepting `set_raster_pipeline` or `draw_raster`.
pub(crate) trait GlRasterCommandApi: GlFamilyApi {
    fn set_raster_pipeline(&mut self, pipeline: &GlRasterPipeline) -> Result<(), GlError>;
    fn draw_raster(&mut self, draw: GlDrawCommand) -> Result<(), GlError>;
}
/// Providers expose this only after their discovery snapshot proves both
/// optional commands; the core trait never implies these offsets.
pub(crate) trait GlAdvancedRasterApi: GlRasterCommandApi {
    fn draw_advanced_raster(&mut self, draw: GlAdvancedDrawCommand) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexed_draw_keeps_base_and_instance_offsets() {
        let draw = GlDrawCommand::Indexed(GlIndexedDraw {
            first_index: 3,
            index_count: 6,
            instance_count: 2,
        });
        assert!(matches!(
            draw,
            GlDrawCommand::Indexed(GlIndexedDraw {
                instance_count: 2,
                ..
            })
        ));
    }
    #[test]
    fn advanced_offset_is_capability_gated() {
        let draw = GlAdvancedDrawCommand {
            draw: GlDrawCommand::NonIndexed(GlNonIndexedDraw {
                first_vertex: 0,
                vertex_count: 3,
                instance_count: 1,
            }),
            base_vertex: 0,
            first_instance: 2,
        };
        assert_eq!(
            draw.validate(GlAdvancedRasterCapabilities {
                base_vertex: true,
                first_instance: false
            }),
            Err(GlRasterValidationError::MissingAdvancedCapability)
        );
    }
}
