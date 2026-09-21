//! Browser fixed-raster and draw execution.
//!
//! WebGL2 blends and color masks globally: independent per-attachment state
//! needs `EXT_draw_buffers_indexed`, so mismatched per-target declarations
//! reject with a structured error instead of being silently collapsed to
//! global state (plan: per-attachment rejection rule).

use web_sys::WebGl2RenderingContext as Gl;

use super::super::api::{
    GlBlendFactor, GlBlendOperation, GlCullMode, GlDepthCompareFunction, GlDepthStencilState,
    GlDrawCommand, GlError, GlFamilyApi as _, GlFrontFace, GlPrimitiveTopology, GlRasterCommandApi,
    GlRasterPipeline, GlRasterState, GlRasterValidationInfo, GlStencilFaceState,
    GlStencilOperation,
};
use super::discovery::WebGl2BrowserDiscovery;
use super::exec_vertex::{indexed_draw_offset, indexed_draw_span, indexed_draw_type};

pub(super) const fn topology_mode(topology: GlPrimitiveTopology) -> u32 {
    match topology {
        GlPrimitiveTopology::Points => Gl::POINTS,
        GlPrimitiveTopology::Lines => Gl::LINES,
        GlPrimitiveTopology::LineStrip => Gl::LINE_STRIP,
        GlPrimitiveTopology::Triangles => Gl::TRIANGLES,
        GlPrimitiveTopology::TriangleStrip => Gl::TRIANGLE_STRIP,
    }
}

const fn compare_function(compare: GlDepthCompareFunction) -> u32 {
    match compare {
        GlDepthCompareFunction::Never => Gl::NEVER,
        GlDepthCompareFunction::Less => Gl::LESS,
        GlDepthCompareFunction::Equal => Gl::EQUAL,
        GlDepthCompareFunction::LessEqual => Gl::LEQUAL,
        GlDepthCompareFunction::Greater => Gl::GREATER,
        GlDepthCompareFunction::NotEqual => Gl::NOTEQUAL,
        GlDepthCompareFunction::GreaterEqual => Gl::GEQUAL,
        GlDepthCompareFunction::Always => Gl::ALWAYS,
    }
}

const fn stencil_operation(operation: GlStencilOperation) -> u32 {
    match operation {
        GlStencilOperation::Keep => Gl::KEEP,
        GlStencilOperation::Zero => Gl::ZERO,
        GlStencilOperation::Replace => Gl::REPLACE,
        GlStencilOperation::IncrementClamp => Gl::INCR,
        GlStencilOperation::DecrementClamp => Gl::DECR,
        GlStencilOperation::Invert => Gl::INVERT,
        GlStencilOperation::IncrementWrap => Gl::INCR_WRAP,
        GlStencilOperation::DecrementWrap => Gl::DECR_WRAP,
    }
}

const fn blend_factor(factor: GlBlendFactor) -> u32 {
    match factor {
        GlBlendFactor::Zero => Gl::ZERO,
        GlBlendFactor::One => Gl::ONE,
        GlBlendFactor::Src => Gl::SRC_COLOR,
        GlBlendFactor::OneMinusSrc => Gl::ONE_MINUS_SRC_COLOR,
        GlBlendFactor::SrcAlpha => Gl::SRC_ALPHA,
        GlBlendFactor::OneMinusSrcAlpha => Gl::ONE_MINUS_SRC_ALPHA,
        GlBlendFactor::Dst => Gl::DST_COLOR,
        GlBlendFactor::OneMinusDst => Gl::ONE_MINUS_DST_COLOR,
        GlBlendFactor::DstAlpha => Gl::DST_ALPHA,
        GlBlendFactor::OneMinusDstAlpha => Gl::ONE_MINUS_DST_ALPHA,
        GlBlendFactor::Constant => Gl::CONSTANT_COLOR,
        GlBlendFactor::OneMinusConstant => Gl::ONE_MINUS_CONSTANT_COLOR,
        GlBlendFactor::SrcAlphaSaturated => Gl::SRC_ALPHA_SATURATE,
    }
}

const fn blend_operation(operation: GlBlendOperation) -> u32 {
    match operation {
        GlBlendOperation::Add => Gl::FUNC_ADD,
        GlBlendOperation::Subtract => Gl::FUNC_SUBTRACT,
        GlBlendOperation::ReverseSubtract => Gl::FUNC_REVERSE_SUBTRACT,
        GlBlendOperation::Min => Gl::MIN,
        GlBlendOperation::Max => Gl::MAX,
    }
}

/// One draw with every value a browser draw call needs already converted and
/// every bound the single-draw path checks already checked.
///
/// A batch readies all of its draws into this form before it issues the first
/// command, so the checks live in exactly one place and a batch cannot submit
/// half of itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PreparedDraw {
    /// `drawArrays`-shaped values.
    NonIndexed {
        first: i32,
        count: i32,
        instances: i32,
    },
    /// `drawElements`-shaped values; `offset` is a byte offset.
    Indexed {
        count: i32,
        index_type: u32,
        offset: i32,
        instances: i32,
    },
}

impl GlRasterCommandApi for WebGl2BrowserDiscovery {
    fn set_raster_pipeline(&mut self, pipeline: &GlRasterPipeline) -> Result<(), GlError> {
        const OP: &str = "set-raster-pipeline";
        self.assert_provider_ready(OP)?;
        let pass = self
            .pass
            .as_ref()
            .ok_or_else(|| Self::validation(OP, "no active render pass"))?;
        let program_raw = self.program(OP, pipeline.program)?.raw.clone();
        let vertex_array_raw = self.vertex_array(OP, pipeline.vertex_array)?.raw.clone();
        pipeline
            .state
            .validate(GlRasterValidationInfo {
                width: pass.width,
                height: pass.height,
                max_samples: self.discovery().limits().max_samples,
                max_color_targets: self.discovery().limits().max_color_attachments,
            })
            .map_err(|_| Self::validation(OP, "invalid raster state"))?;
        if pipeline.state.multisample.sample_count != pass.samples {
            return Err(Self::validation(
                OP,
                "pipeline sample count does not match the active framebuffer",
            ));
        }
        self.raw.use_program(Some(&program_raw));
        self.raw.bind_vertex_array(Some(&vertex_array_raw));
        // What the driver now holds, recorded where the binding happens, for the
        // same reason the input domain records it there: the draws read this slot
        // and not the identity passed to this install, because the geometry domain
        // replaces the installed array on every request under the uncached
        // execution mode.
        self.bound_vertex_array = Some(pipeline.vertex_array);
        self.apply_raster_state(OP, &pipeline.state)?;
        if let Err(error) = self.driver_error(OP) {
            self.raster = None;
            return Err(error);
        }
        self.raster = Some(super::objects::ActiveRaster {
            program: pipeline.program,
            topology: pipeline.state.topology,
        });
        Ok(())
    }

    fn draw_raster(&mut self, draw: GlDrawCommand) -> Result<(), GlError> {
        const OP: &str = "draw-raster";
        self.assert_provider_ready(OP)?;
        if self.pass.is_none() {
            return Err(Self::validation(OP, "no active render pass"));
        }
        let raster = self
            .raster
            .as_ref()
            .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
        let mode = topology_mode(raster.topology);
        match self.prepare_draw(OP, draw)? {
            PreparedDraw::NonIndexed {
                first,
                count,
                instances,
            } => {
                if instances == 1 {
                    self.raw.draw_arrays(mode, first, count);
                } else {
                    self.raw
                        .draw_arrays_instanced(mode, first, count, instances);
                }
            }
            PreparedDraw::Indexed {
                count,
                index_type,
                offset,
                instances,
            } => {
                if instances == 1 {
                    self.raw
                        .draw_elements_with_i32(mode, count, index_type, offset);
                } else {
                    self.raw.draw_elements_instanced_with_i32(
                        mode, count, index_type, offset, instances,
                    );
                }
            }
        }
        self.driver_error(OP)
    }
}

impl WebGl2BrowserDiscovery {
    /// Validates one draw against the active pass, the installed pipeline, and
    /// the index allocation it reads, returning the values the browser call
    /// needs.
    ///
    /// It issues no GL command, so a batch can ready every draw before its
    /// first submission. Bounds are checked against the allocation table, so an
    /// out-of-range draw rejects before the browser can silently ignore it.
    pub(super) fn prepare_draw(
        &self,
        operation: &'static str,
        draw: GlDrawCommand,
    ) -> Result<PreparedDraw, GlError> {
        if self.raster.is_none() {
            return Err(Self::validation(
                operation,
                "no raster pipeline is installed",
            ));
        }
        // The array comes from the binding record rather than from the pipeline,
        // the same way `draw_raster` reads it in the native provider: the geometry
        // domain reconciles inputs between the install and the draw and replaces
        // the installed array under the uncached execution mode.
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(operation, "no vertex array is bound"))?;
        let vertex_array = self.vertex_array(operation, bound)?;
        let index = vertex_array.index;
        match draw {
            GlDrawCommand::NonIndexed(draw) => {
                if draw.vertex_count == 0 || draw.instance_count == 0 {
                    return Err(Self::validation(
                        operation,
                        "draw count and instances must be nonzero",
                    ));
                }
                let first = i32::try_from(draw.first_vertex)
                    .map_err(|_| Self::validation(operation, "first vertex exceeds i32"))?;
                let count = i32::try_from(draw.vertex_count)
                    .map_err(|_| Self::validation(operation, "vertex count exceeds i32"))?;
                let instances = i32::try_from(draw.instance_count)
                    .map_err(|_| Self::validation(operation, "instance count exceeds i32"))?;
                Ok(PreparedDraw::NonIndexed {
                    first,
                    count,
                    instances,
                })
            }
            GlDrawCommand::Indexed(draw) => {
                if draw.index_count == 0 || draw.instance_count == 0 {
                    return Err(Self::validation(
                        operation,
                        "draw count and instances must be nonzero",
                    ));
                }
                let Some(index) = index else {
                    return Err(Self::validation(
                        operation,
                        "indexed draw requires a bound index buffer",
                    ));
                };
                let byte_length = self.buffer(operation, index.buffer)?.desc.size;
                let span = indexed_draw_span(index, draw.first_index, draw.index_count)
                    .ok_or_else(|| Self::validation(operation, "indexed draw span overflows"))?;
                if span > byte_length {
                    return Err(Self::validation(
                        operation,
                        "indexed draw leaves the index buffer",
                    ));
                }
                let offset = indexed_draw_offset(index, draw.first_index)
                    .and_then(|value| i32::try_from(value).ok())
                    .ok_or_else(|| Self::validation(operation, "index offset exceeds i32"))?;
                let count = i32::try_from(draw.index_count)
                    .map_err(|_| Self::validation(operation, "index count exceeds i32"))?;
                let instances = i32::try_from(draw.instance_count)
                    .map_err(|_| Self::validation(operation, "instance count exceeds i32"))?;
                Ok(PreparedDraw::Indexed {
                    count,
                    index_type: indexed_draw_type(index),
                    offset,
                    instances,
                })
            }
        }
    }
}

impl WebGl2BrowserDiscovery {
    /// Applies one complete validated raster state in a fixed order.
    fn apply_raster_state(
        &mut self,
        op: &'static str,
        state: &GlRasterState,
    ) -> Result<(), GlError> {
        let viewport = &state.viewport;
        self.raw.viewport(
            viewport.x as i32,
            viewport.y as i32,
            viewport.width as i32,
            viewport.height as i32,
        );
        self.raw.depth_range(
            f32::from_bits(viewport.min_depth),
            f32::from_bits(viewport.max_depth),
        );
        match state.scissor {
            Some(scissor) => {
                self.raw.enable(Gl::SCISSOR_TEST);
                self.raw.scissor(
                    scissor.x as i32,
                    scissor.y as i32,
                    scissor.width as i32,
                    scissor.height as i32,
                );
            }
            None => self.raw.disable(Gl::SCISSOR_TEST),
        }
        match state.cull_mode {
            GlCullMode::None => self.raw.disable(Gl::CULL_FACE),
            GlCullMode::Front => {
                self.raw.enable(Gl::CULL_FACE);
                self.raw.cull_face(Gl::FRONT);
            }
            GlCullMode::Back => {
                self.raw.enable(Gl::CULL_FACE);
                self.raw.cull_face(Gl::BACK);
            }
        }
        self.raw.front_face(match state.front_face {
            GlFrontFace::Clockwise => Gl::CW,
            GlFrontFace::CounterClockwise => Gl::CCW,
        });
        match &state.depth_stencil {
            Some(depth_stencil) => self.apply_depth_stencil(op, depth_stencil)?,
            None => {
                self.raw.disable(Gl::DEPTH_TEST);
                self.raw.disable(Gl::STENCIL_TEST);
                self.raw.disable(Gl::POLYGON_OFFSET_FILL);
            }
        }
        self.apply_color_state(op, state)?;
        Ok(())
    }

    fn apply_depth_stencil(
        &mut self,
        op: &'static str,
        state: &GlDepthStencilState,
    ) -> Result<(), GlError> {
        self.raw.enable(Gl::DEPTH_TEST);
        self.raw.depth_mask(state.depth_write_enabled);
        self.raw.depth_func(compare_function(state.depth_compare));
        // Depth bias travels as exact bits: [slope factor, units, clamp].
        // WebGL2 exposes no polygon-offset clamp, so a nonzero clamp rejects.
        let (factor, units, clamp) = (
            f32::from_bits(state.depth_bias[0]),
            f32::from_bits(state.depth_bias[1]),
            f32::from_bits(state.depth_bias[2]),
        );
        if clamp != 0.0 {
            return Err(Self::validation(
                op,
                "depth-bias clamp has no WebGL2 expression",
            ));
        }
        if factor == 0.0 && units == 0.0 {
            self.raw.disable(Gl::POLYGON_OFFSET_FILL);
        } else {
            self.raw.enable(Gl::POLYGON_OFFSET_FILL);
            self.raw.polygon_offset(factor, units);
        }
        self.raw.enable(Gl::STENCIL_TEST);
        self.apply_stencil_face(Gl::FRONT, &state.stencil_front, state.stencil_reference);
        self.apply_stencil_face(Gl::BACK, &state.stencil_back, state.stencil_reference);
        Ok(())
    }

    fn apply_stencil_face(&mut self, face: u32, state: &GlStencilFaceState, reference: u32) {
        self.raw.stencil_func_separate(
            face,
            compare_function(state.compare),
            reference as i32,
            state.read_mask,
        );
        self.raw.stencil_mask_separate(face, state.write_mask);
        self.raw.stencil_op_separate(
            face,
            stencil_operation(state.fail_op),
            stencil_operation(state.depth_fail_op),
            stencil_operation(state.pass_op),
        );
    }

    fn apply_color_state(
        &mut self,
        op: &'static str,
        state: &GlRasterState,
    ) -> Result<(), GlError> {
        let Some(first) = state.color_targets.first() else {
            return Err(Self::validation(
                op,
                "raster state declares no color target for the active pass",
            ));
        };
        // Independent per-attachment blend/write-mask is not expressible in
        // core WebGL2; refuse to collapse differing declarations silently.
        if state
            .color_targets
            .iter()
            .any(|target| target.write_mask != first.write_mask || target.blend != first.blend)
        {
            return Err(Self::validation(
                op,
                "independent per-attachment blend/write-mask is unavailable on WebGL2",
            ));
        }
        self.raw.color_mask(
            first.write_mask & 0b0001 != 0,
            first.write_mask & 0b0010 != 0,
            first.write_mask & 0b0100 != 0,
            first.write_mask & 0b1000 != 0,
        );
        match first.blend {
            Some(blend) => {
                self.raw.enable(Gl::BLEND);
                self.raw.blend_func_separate(
                    blend_factor(blend.color.src_factor),
                    blend_factor(blend.color.dst_factor),
                    blend_factor(blend.alpha.src_factor),
                    blend_factor(blend.alpha.dst_factor),
                );
                self.raw.blend_equation_separate(
                    blend_operation(blend.color.operation),
                    blend_operation(blend.alpha.operation),
                );
                self.raw.blend_color(
                    f32::from_bits(state.blend_constant[0]),
                    f32::from_bits(state.blend_constant[1]),
                    f32::from_bits(state.blend_constant[2]),
                    f32::from_bits(state.blend_constant[3]),
                );
            }
            None => self.raw.disable(Gl::BLEND),
        }
        // WebGL2 multisampling is framebuffer state; only the coverage knobs
        // are pipeline state. A sample mask other than "all" has no typed
        // WebGL2 command, so it rejects instead of being ignored.
        if state.multisample.sample_mask != u32::MAX {
            return Err(Self::validation(
                op,
                "explicit sample masks have no typed WebGL2 command",
            ));
        }
        if state.multisample.alpha_to_coverage_enabled {
            self.raw.enable(Gl::SAMPLE_ALPHA_TO_COVERAGE);
        } else {
            self.raw.disable(Gl::SAMPLE_ALPHA_TO_COVERAGE);
        }
        Ok(())
    }
}
