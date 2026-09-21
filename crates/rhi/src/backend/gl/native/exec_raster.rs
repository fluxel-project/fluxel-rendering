//! Native fixed-raster and draw execution.
//!
//! The core GL 4.x/GLES 3.x families blend and mask color globally like
//! WebGL2: independent per-attachment state needs `EXT_draw_buffers_indexed`,
//! so mismatched per-target declarations reject with a structured error
//! instead of being silently collapsed to global state.

use super::exec_vertex::{indexed_draw_offset, indexed_draw_span, indexed_draw_type};
use super::provider::{ActiveRaster, NativeGlProvider};
use crate::backend::gl::api::{
    GlBlendFactor, GlBlendOperation, GlCullMode, GlDepthCompareFunction, GlDepthStencilState,
    GlDrawCommand, GlError, GlFamilyApi as _, GlFrontFace, GlMultiDraw, GlMultiDrawApi,
    GlPrimitiveTopology, GlRasterCommandApi, GlRasterPipeline, GlRasterState,
    GlRasterValidationInfo, GlStencilFaceState, GlStencilOperation, issue_single_draws,
};

pub(super) const fn topology_mode(topology: GlPrimitiveTopology) -> u32 {
    match topology {
        GlPrimitiveTopology::Points => glow::POINTS,
        GlPrimitiveTopology::Lines => glow::LINES,
        GlPrimitiveTopology::LineStrip => glow::LINE_STRIP,
        GlPrimitiveTopology::Triangles => glow::TRIANGLES,
        GlPrimitiveTopology::TriangleStrip => glow::TRIANGLE_STRIP,
    }
}

const fn compare_function(compare: GlDepthCompareFunction) -> u32 {
    match compare {
        GlDepthCompareFunction::Never => glow::NEVER,
        GlDepthCompareFunction::Less => glow::LESS,
        GlDepthCompareFunction::Equal => glow::EQUAL,
        GlDepthCompareFunction::LessEqual => glow::LEQUAL,
        GlDepthCompareFunction::Greater => glow::GREATER,
        GlDepthCompareFunction::NotEqual => glow::NOTEQUAL,
        GlDepthCompareFunction::GreaterEqual => glow::GEQUAL,
        GlDepthCompareFunction::Always => glow::ALWAYS,
    }
}

const fn stencil_operation(operation: GlStencilOperation) -> u32 {
    match operation {
        GlStencilOperation::Keep => glow::KEEP,
        GlStencilOperation::Zero => glow::ZERO,
        GlStencilOperation::Replace => glow::REPLACE,
        GlStencilOperation::IncrementClamp => glow::INCR,
        GlStencilOperation::DecrementClamp => glow::DECR,
        GlStencilOperation::Invert => glow::INVERT,
        GlStencilOperation::IncrementWrap => glow::INCR_WRAP,
        GlStencilOperation::DecrementWrap => glow::DECR_WRAP,
    }
}

const fn blend_factor(factor: GlBlendFactor) -> u32 {
    match factor {
        GlBlendFactor::Zero => glow::ZERO,
        GlBlendFactor::One => glow::ONE,
        GlBlendFactor::Src => glow::SRC_COLOR,
        GlBlendFactor::OneMinusSrc => glow::ONE_MINUS_SRC_COLOR,
        GlBlendFactor::SrcAlpha => glow::SRC_ALPHA,
        GlBlendFactor::OneMinusSrcAlpha => glow::ONE_MINUS_SRC_ALPHA,
        GlBlendFactor::Dst => glow::DST_COLOR,
        GlBlendFactor::OneMinusDst => glow::ONE_MINUS_DST_COLOR,
        GlBlendFactor::DstAlpha => glow::DST_ALPHA,
        GlBlendFactor::OneMinusDstAlpha => glow::ONE_MINUS_DST_ALPHA,
        GlBlendFactor::Constant => glow::CONSTANT_COLOR,
        GlBlendFactor::OneMinusConstant => glow::ONE_MINUS_CONSTANT_COLOR,
        GlBlendFactor::SrcAlphaSaturated => glow::SRC_ALPHA_SATURATE,
    }
}

const fn blend_operation(operation: GlBlendOperation) -> u32 {
    match operation {
        GlBlendOperation::Add => glow::FUNC_ADD,
        GlBlendOperation::Subtract => glow::FUNC_SUBTRACT,
        GlBlendOperation::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
        GlBlendOperation::Min => glow::MIN,
        GlBlendOperation::Max => glow::MAX,
    }
}

impl GlRasterCommandApi for NativeGlProvider {
    fn set_raster_pipeline(&mut self, pipeline: &GlRasterPipeline) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "set-raster-pipeline";
        self.assert_ready(OP)?;
        let pass = self
            .pass
            .as_ref()
            .ok_or_else(|| Self::validation(OP, "no active render pass"))?;
        // The program is resolved first even though it is selected last, so that
        // a dead program is reported before anything about the state is.
        self.program(OP, pipeline.program)?;
        pipeline
            .state
            .validate(GlRasterValidationInfo {
                width: pass.width,
                height: pass.height,
                max_samples: self.discovery.limits().max_samples,
                max_color_targets: self.discovery.limits().max_color_attachments,
            })
            .map_err(|_| Self::validation(OP, "invalid raster state"))?;
        if pipeline.state.multisample.sample_count != pass.samples {
            return Err(Self::validation(
                OP,
                "pipeline sample count does not match the active framebuffer",
            ));
        }
        // The program selection goes through the one owner of that fact, so a
        // compute install that ran in between cannot leave this pipeline's
        // program unselected without this verb noticing.  The vertex array goes
        // through the owner of the other binding slot for the same reason: this
        // install states which array it wants, but what the driver holds is the
        // provider's record to keep, and both draws read that record rather than
        // this call's argument.
        self.ensure_program(OP, pipeline.program)?;
        self.ensure_vertex_array(OP, pipeline.vertex_array)?;
        // SAFETY: current-context contract; the full validated state is
        // applied in one fixed order and a failure clears the installation.
        let applied = unsafe { self.apply_raster_state(OP, &pipeline.state) };
        if let Err(error) = applied.and_then(|()| self.driver_error(OP)) {
            self.raster = None;
            return Err(error);
        }
        self.raster = Some(ActiveRaster {
            program: pipeline.program,
            topology: pipeline.state.topology,
        });
        Ok(())
    }

    fn draw_raster(&mut self, draw: GlDrawCommand) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "draw-raster";
        self.assert_ready(OP)?;
        if self.pass.is_none() {
            return Err(Self::validation(OP, "no active render pass"));
        }
        let (program, topology) = {
            let raster = self
                .raster
                .as_ref()
                .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
            (raster.program, raster.topology)
        };
        // A compute install or a link may have left another program current since
        // this pipeline was installed, so re-asserting the program is what makes
        // this draw the one the caller asked for rather than the one that happens
        // to be selected.
        self.ensure_program(OP, program)?;
        // The vertex array is *not* re-asserted from the pipeline: the geometry
        // domain reconciles inputs between the install and this draw and, under
        // the uncached execution mode, replaces the array on every request, so
        // the install-time identity is a dead object by now.  The provider's
        // record of what the driver holds is the fact this draw needs, and it is
        // also the array whose index binding and layout the draw must read.
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(OP, "no vertex array is bound"))?;
        let mode = topology_mode(topology);
        let vertex_array = self.vertex_array(OP, bound)?;
        let index = vertex_array.index;
        // SAFETY: current-context contract; counts, instances, and index spans
        // are checked against the recorded allocations before any draw call.
        unsafe {
            match draw {
                GlDrawCommand::NonIndexed(draw) => {
                    if draw.vertex_count == 0 || draw.instance_count == 0 {
                        return Err(Self::validation(
                            OP,
                            "draw count and instances must be nonzero",
                        ));
                    }
                    let first = i32::try_from(draw.first_vertex)
                        .map_err(|_| Self::validation(OP, "first vertex exceeds i32"))?;
                    let count = i32::try_from(draw.vertex_count)
                        .map_err(|_| Self::validation(OP, "vertex count exceeds i32"))?;
                    let instances = i32::try_from(draw.instance_count)
                        .map_err(|_| Self::validation(OP, "instance count exceeds i32"))?;
                    if instances == 1 {
                        self.gl.draw_arrays(mode, first, count);
                    } else {
                        self.gl.draw_arrays_instanced(mode, first, count, instances);
                    }
                }
                GlDrawCommand::Indexed(draw) => {
                    if draw.index_count == 0 || draw.instance_count == 0 {
                        return Err(Self::validation(
                            OP,
                            "draw count and instances must be nonzero",
                        ));
                    }
                    let Some(index) = index else {
                        return Err(Self::validation(
                            OP,
                            "indexed draw requires a bound index buffer",
                        ));
                    };
                    // Bound checks use the allocation table, so an out-of-range
                    // draw rejects before the driver can silently ignore it.
                    let byte_length = self.buffer(OP, index.buffer)?.1.size;
                    let span = indexed_draw_span(index, draw.first_index, draw.index_count)
                        .ok_or_else(|| Self::validation(OP, "indexed draw span overflows"))?;
                    if span > byte_length {
                        return Err(Self::validation(OP, "indexed draw leaves the index buffer"));
                    }
                    let offset = indexed_draw_offset(index, draw.first_index)
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or_else(|| Self::validation(OP, "index offset exceeds i32"))?;
                    let count = i32::try_from(draw.index_count)
                        .map_err(|_| Self::validation(OP, "index count exceeds i32"))?;
                    let instances = i32::try_from(draw.instance_count)
                        .map_err(|_| Self::validation(OP, "instance count exceeds i32"))?;
                    if instances == 1 {
                        self.gl
                            .draw_elements(mode, count, indexed_draw_type(index), offset);
                    } else {
                        self.gl.draw_elements_instanced(
                            mode,
                            count,
                            indexed_draw_type(index),
                            offset,
                            instances,
                        );
                    }
                }
            }
        }
        self.driver_error(OP)
    }
}

/// A batch always takes the single-draw route on the native families.
///
/// The native contract proves no combined batch command, so rather than
/// branching on a capability the provider decomposes every batch into the
/// validated single-draw path.
impl GlMultiDrawApi for NativeGlProvider {
    fn multi_draw(&mut self, command: &GlMultiDraw) -> Result<(), GlError> {
        const OP: &str = "multi-draw";
        self.assert_ready(OP)?;
        issue_single_draws(self, command)
    }
}

impl NativeGlProvider {
    /// Applies one complete validated raster state in a fixed order.
    ///
    /// # Safety
    ///
    /// Current-context contract; the state must already be validated.
    unsafe fn apply_raster_state(
        &self,
        op: &'static str,
        state: &GlRasterState,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        unsafe {
            let viewport = &state.viewport;
            self.gl.viewport(
                viewport.x as i32,
                viewport.y as i32,
                viewport.width as i32,
                viewport.height as i32,
            );
            self.gl.depth_range_f32(
                f32::from_bits(viewport.min_depth),
                f32::from_bits(viewport.max_depth),
            );
            match state.scissor {
                Some(scissor) => {
                    self.gl.enable(glow::SCISSOR_TEST);
                    self.gl.scissor(
                        scissor.x as i32,
                        scissor.y as i32,
                        scissor.width as i32,
                        scissor.height as i32,
                    );
                }
                None => self.gl.disable(glow::SCISSOR_TEST),
            }
            match state.cull_mode {
                GlCullMode::None => self.gl.disable(glow::CULL_FACE),
                GlCullMode::Front => {
                    self.gl.enable(glow::CULL_FACE);
                    self.gl.cull_face(glow::FRONT);
                }
                GlCullMode::Back => {
                    self.gl.enable(glow::CULL_FACE);
                    self.gl.cull_face(glow::BACK);
                }
            }
            self.gl.front_face(match state.front_face {
                GlFrontFace::Clockwise => glow::CW,
                GlFrontFace::CounterClockwise => glow::CCW,
            });
            match &state.depth_stencil {
                Some(depth_stencil) => self.apply_depth_stencil(op, depth_stencil)?,
                None => {
                    self.gl.disable(glow::DEPTH_TEST);
                    self.gl.disable(glow::STENCIL_TEST);
                    self.gl.disable(glow::POLYGON_OFFSET_FILL);
                }
            }
            self.apply_color_state(op, state)?;
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Current-context contract; the state must already be validated.
    unsafe fn apply_depth_stencil(
        &self,
        op: &'static str,
        state: &GlDepthStencilState,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        unsafe {
            self.gl.enable(glow::DEPTH_TEST);
            self.gl.depth_mask(state.depth_write_enabled);
            self.gl.depth_func(compare_function(state.depth_compare));
            // Depth bias travels as exact bits: [slope factor, units, clamp].
            // The shared semantic exposes no polygon-offset clamp; desktop
            // expressiveness would require EXT_polygon_offset_clamp, which is
            // not a typed route, so a nonzero clamp rejects everywhere.
            let (factor, units, clamp) = (
                f32::from_bits(state.depth_bias[0]),
                f32::from_bits(state.depth_bias[1]),
                f32::from_bits(state.depth_bias[2]),
            );
            if clamp != 0.0 {
                return Err(Self::validation(
                    op,
                    "depth-bias clamp has no typed expression in this contract",
                ));
            }
            if factor == 0.0 && units == 0.0 {
                self.gl.disable(glow::POLYGON_OFFSET_FILL);
            } else {
                self.gl.enable(glow::POLYGON_OFFSET_FILL);
                self.gl.polygon_offset(factor, units);
            }
            self.gl.enable(glow::STENCIL_TEST);
            self.apply_stencil_face(glow::FRONT, &state.stencil_front, state.stencil_reference);
            self.apply_stencil_face(glow::BACK, &state.stencil_back, state.stencil_reference);
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Current-context contract.
    unsafe fn apply_stencil_face(&self, face: u32, state: &GlStencilFaceState, reference: u32) {
        use glow::HasContext as _;
        unsafe {
            self.gl.stencil_func_separate(
                face,
                compare_function(state.compare),
                reference as i32,
                state.read_mask,
            );
            self.gl.stencil_mask_separate(face, state.write_mask);
            self.gl.stencil_op_separate(
                face,
                stencil_operation(state.fail_op),
                stencil_operation(state.depth_fail_op),
                stencil_operation(state.pass_op),
            );
        }
    }

    /// # Safety
    ///
    /// Current-context contract; the state must already be validated.
    unsafe fn apply_color_state(
        &self,
        op: &'static str,
        state: &GlRasterState,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        unsafe {
            let Some(first) = state.color_targets.first() else {
                return Err(Self::validation(
                    op,
                    "raster state declares no color target for the active pass",
                ));
            };
            // Independent per-attachment blend/write-mask is not expressible
            // in the common GL core semantic; refuse to collapse differing
            // declarations silently (EXT_draw_buffers_indexed is not a typed
            // route in this contract).
            if state
                .color_targets
                .iter()
                .any(|target| target.write_mask != first.write_mask || target.blend != first.blend)
            {
                return Err(Self::validation(
                    op,
                    "independent per-attachment blend/write-mask is unavailable in the common semantic",
                ));
            }
            self.gl.color_mask(
                first.write_mask & 0b0001 != 0,
                first.write_mask & 0b0010 != 0,
                first.write_mask & 0b0100 != 0,
                first.write_mask & 0b1000 != 0,
            );
            match first.blend {
                Some(blend) => {
                    self.gl.enable(glow::BLEND);
                    self.gl.blend_func_separate(
                        blend_factor(blend.color.src_factor),
                        blend_factor(blend.color.dst_factor),
                        blend_factor(blend.alpha.src_factor),
                        blend_factor(blend.alpha.dst_factor),
                    );
                    self.gl.blend_equation_separate(
                        blend_operation(blend.color.operation),
                        blend_operation(blend.alpha.operation),
                    );
                    self.gl.blend_color(
                        f32::from_bits(state.blend_constant[0]),
                        f32::from_bits(state.blend_constant[1]),
                        f32::from_bits(state.blend_constant[2]),
                        f32::from_bits(state.blend_constant[3]),
                    );
                }
                None => self.gl.disable(glow::BLEND),
            }
            // Multisampling is framebuffer state; only the coverage knobs are
            // pipeline state. glow 0.18 binds no `glSampleMaski` and the
            // embedded families have no typed command at all, so an explicit
            // partial sample mask rejects on every native profile instead of
            // being silently ignored (browser parity).
            if state.multisample.sample_mask != u32::MAX {
                return Err(Self::validation(
                    op,
                    "explicit sample masks have no typed command in this provider",
                ));
            }
            self.gl.enable(glow::MULTISAMPLE);
            if state.multisample.alpha_to_coverage_enabled {
                self.gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            } else {
                self.gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            }
        }
        Ok(())
    }
}
