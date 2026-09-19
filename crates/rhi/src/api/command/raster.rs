//! `RasterScope` and the raster command set (specification section 32).
//!
//! This module owns the render-pass-shaped half of a recording: the attachment
//! set is fixed when a scope opens, every portable state value starts unset, and
//! the draw verbs are the only places where a bound resource becomes an *actual*
//! use.
//!
//! # What this module does not own
//!
//! - Whether an attachment set is legal. That is
//!   [`validate_raster_scope`](crate::api::command::attachment::validate_raster_scope),
//!   which [`CommandRecorder::begin_raster`] calls before a scope exists.
//! - What a use record looks like. The access masks are
//!   [`crate::api::command::uses`]'s, so that a storage buffer costs the same
//!   here as it does in a compute scope.
//! - Whether the scope is lowerable. A backend lowers the recorded sequence; this
//!   module only records what the specification calls semantic.
//!
//! # The invariant this module enforces
//!
//! **A scope is entered and left exactly once, and the borrow checker is what
//! says so.** `RasterScope` holds the recorder mutably, so a caller cannot begin
//! a second scope, issue a copy, or finish the recording while one is open —
//! section 29.2's rule becomes a compile error rather than a runtime check. The
//! one thing a borrow cannot state is a scope dropped without `end`, which is a
//! runtime fact; that is what [`RasterScope`]'s `Drop` is for, and it poisons the
//! recording because the recording it was writing into has a hole in it that no
//! later command can close.

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::attachment::{
    ColorAttachment, ColorAttachmentView, DepthAttachmentMode, DepthStencilAttachment,
    RasterScopeDescriptor, StencilAttachmentMode,
};
use crate::api::command::geometry::{
    Color, LoadOp, Rect, Viewport, validate_rect, validate_viewport,
};
use crate::api::command::record::{
    BoundGroup, BoundIndexBuffer, RasterBegin, RasterDraw, RecordedPayload,
};
use crate::api::command::uses::{
    bound_group_uses, buffer_use, frame_use, require_valid_dynamic_offsets, texture_use_of_view,
    validate_bound_groups,
};
use crate::api::command::{CommandRecorder, IndexFormat, RecorderPhase, require_device};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::graph_bridge::{AccessMask, PipelineScope, ResourceUse, TextureUseIntent};
use crate::api::identity::Label;
use crate::api::pipeline::{RasterPipeline, RenderTargetSignature, VertexStepMode};
use crate::api::resource::buffer::{BufferBinding, BufferUsage, validate_buffer_range};
use crate::api::resource::texture::Extent3d;

/// The domain bit every raster command contributes.
const RASTER_DOMAIN: crate::api::submission::LaneWorkDomains =
    crate::api::submission::LaneWorkDomains::RASTER;

impl CommandRecorder {
    /// Opens a raster scope over a validated attachment set.
    ///
    /// The descriptor is canonicalized (trailing unattached color locations
    /// removed) and then checked by
    /// `validate_raster_scope`.
    /// Everything that can make an attachment set illegal is decided here, before
    /// the scope exists, so a `RasterScope` always describes a set that one native
    /// render pass could lower.
    ///
    /// Nothing is poisoned by a refusal: a refused attachment set never became a
    /// scope, and section 29.3 puts only *scope finalization* failure and
    /// backend-internal failure in the poisoning category, never parameter
    /// refusal.
    pub fn begin_raster<'a>(
        &'a mut self,
        desc: &RasterScopeDescriptor,
    ) -> RhiResult<RasterScope<'a>> {
        self.require_open("begin_raster")?;

        let desc = desc.clone().canonicalized();
        crate::api::command::attachment::validate_raster_scope(&desc)?;

        let colors: Vec<(u32, ColorAttachment)> = desc
            .attached_colors()
            .map(|(location, attachment)| (location, attachment.clone()))
            .collect();
        let signature = desc.target_signature();
        let extent = primary_extent(&colors, desc.depth_stencil.as_ref());

        let uses = begin_uses(&colors, desc.depth_stencil.as_ref());
        let begin = RasterBegin {
            label: desc.label.clone(),
            colors: colors.clone(),
            depth_stencil: desc.depth_stencil.clone(),
        };
        self.record_command(RecordedPayload::RasterBegin(begin), uses, RASTER_DOMAIN);
        self.set_phase(RecorderPhase::RasterScopeOpen);

        Ok(RasterScope {
            recorder: self,
            signature,
            extent,
            colors,
            depth_stencil: desc.depth_stencil,
            pipeline: None,
            groups: Vec::new(),
            vertex_buffers: Vec::new(),
            index: None,
            viewport: None,
            scissor: None,
            blend_constant: Color::new(0.0, 0.0, 0.0, 0.0),
            stencil_reference: 0,
            debug_stack: Vec::new(),
            ended: false,
        })
    }
}

/// A raster scope, open until it is ended.
///
/// The lifetime is the recorder's, which is section 29.2's "Rust mutable borrow
/// prevents the scope from directly issuing another type of command to the
/// Recorder while it is alive". The borrow is a real `&mut CommandRecorder`
/// rather than the `PhantomData` the specification's sketch writes, because a
/// `PhantomData` body has nothing to record into: the sketch shows the *shape* of
/// an opaque scope, and the borrow it documents is the same borrow this field
/// holds. A caller sees no difference — the type is opaque either way — and the
/// recorder is now able to write down what the scope did.
pub struct RasterScope<'a> {
    /// The recorder this scope is writing into.
    recorder: &'a mut CommandRecorder,

    /// The attachment set's pipeline-facing signature, fixed at `begin_raster`.
    signature: RenderTargetSignature,
    /// The primary attachment's extent, for the portable dynamic-state defaults.
    extent: Extent3d,
    /// The color attachments, by location, fixed at `begin_raster`.
    colors: Vec<(u32, ColorAttachment)>,
    /// The depth/stencil attachment, fixed at `begin_raster`.
    depth_stencil: Option<DepthStencilAttachment>,

    /// The bound pipeline, if any.
    pipeline: Option<RasterPipeline>,
    /// The bound bind groups, by index.
    groups: Vec<BoundGroup>,
    /// The bound vertex buffers, by slot.
    vertex_buffers: Vec<(u32, BufferBinding)>,
    /// The bound index buffer, if any.
    index: Option<BoundIndexBuffer>,

    /// The viewport, or `None` while the portable default applies.
    viewport: Option<Viewport>,
    /// The scissor rect, or `None` while the portable default applies.
    scissor: Option<Rect>,
    /// The blend constant. Portable default: all four components zero.
    blend_constant: Color,
    /// The stencil reference. Portable default: zero.
    stencil_reference: u32,

    /// This scope's own debug-group stack, independent of the recorder's.
    debug_stack: Vec<String>,
    /// Whether `end` completed, which is what decides whether `Drop` poisons.
    ended: bool,
}

impl RasterScope<'_> {
    /// Binds a pipeline, if it matches this scope's attachment set.
    ///
    /// Section 32.2's two checks, in its order: same device first, then the target
    /// signature. The signature comparison is structural and covers sparse color
    /// formats, the depth/stencil format, and the sample count. A resolve target is
    /// *not* part of it, because the pipeline renders into the multisampled source
    /// and never sees the resolve target.
    pub fn set_pipeline(&mut self, pipeline: &RasterPipeline) -> RhiResult<()> {
        require_device(
            pipeline.device_identity(),
            self.recorder.device_identity(),
            "the pipeline",
        )?;
        if pipeline.target_signature() != &self.signature {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                "the pipeline's target signature does not match this raster scope's attachments",
            ));
        }
        self.pipeline = Some(pipeline.clone());
        Ok(())
    }

    /// Binds a bind group at an index.
    ///
    /// The dynamic offsets are validated here against the group's own layout,
    /// because that is the only point at which the caller can be told *which*
    /// binding an offset moved past the end of. Section 32.3 lists "dynamic offsets
    /// valid" among the draw checks as well, and [`RasterScope::draw`] re-checks
    /// it, so a group that reached a draw through some other path cannot be
    /// lowered unvalidated.
    pub fn set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        dynamic_offsets: &[u32],
    ) -> RhiResult<()> {
        require_device(
            group.device_identity(),
            self.recorder.device_identity(),
            "the bind group",
        )?;
        require_device(
            group.layout().device_identity(),
            self.recorder.device_identity(),
            "the bind group's layout",
        )?;
        require_valid_dynamic_offsets(index, group, dynamic_offsets)?;

        let bound = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: dynamic_offsets.to_vec(),
        };
        match self
            .groups
            .iter_mut()
            .find(|existing| existing.index == index)
        {
            Some(existing) => *existing = bound,
            None => self.groups.push(bound),
        }
        Ok(())
    }

    /// Binds a vertex buffer at a slot.
    ///
    /// The `VERTEX` usage bit is checked here rather than at a draw because it is a
    /// property of the buffer, not of the draw: a buffer that can never be a vertex
    /// source should be refused at the verb that tries to make it one. The *other*
    /// half of section 32.3's vertex-buffer rule — that the draw stays in bounds —
    /// needs the vertex range and belongs to [`RasterScope::draw`].
    pub fn set_vertex_buffer(&mut self, slot: u32, binding: &BufferBinding) -> RhiResult<()> {
        require_device(
            binding.buffer.device_identity(),
            self.recorder.device_identity(),
            "the vertex buffer",
        )?;
        if !binding
            .buffer
            .descriptor()
            .usage
            .contains(BufferUsage::VERTEX)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("the buffer bound at vertex slot {slot} was not created with VERTEX usage"),
            ));
        }
        validate_buffer_range(binding.range, binding.buffer.descriptor().size)?;

        match self
            .vertex_buffers
            .iter_mut()
            .find(|(bound, _)| *bound == slot)
        {
            Some(existing) => existing.1 = binding.clone(),
            None => self.vertex_buffers.push((slot, binding.clone())),
        }
        Ok(())
    }

    /// Binds the index buffer and the format it is cut with.
    ///
    /// The format is a property of the *binding* rather than of the pipeline,
    /// because one pipeline may be drawn with either index width. What the strip
    /// topology constrains is the relationship between the two, and that is checked
    /// in [`RasterScope::draw_indexed`].
    pub fn set_index_buffer(
        &mut self,
        binding: &BufferBinding,
        format: IndexFormat,
    ) -> RhiResult<()> {
        require_device(
            binding.buffer.device_identity(),
            self.recorder.device_identity(),
            "the index buffer",
        )?;
        if !binding
            .buffer
            .descriptor()
            .usage
            .contains(BufferUsage::INDEX)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the index buffer was not created with INDEX usage",
            ));
        }
        validate_buffer_range(binding.range, binding.buffer.descriptor().size)?;

        self.index = Some(BoundIndexBuffer {
            binding: binding.clone(),
            format,
        });
        Ok(())
    }

    /// Sets the viewport, or leaves the portable default in force.
    pub fn set_viewport(&mut self, viewport: Viewport) -> RhiResult<()> {
        validate_viewport(viewport)?;
        self.viewport = Some(viewport);
        Ok(())
    }

    /// Sets the scissor rect, or leaves the portable default in force.
    pub fn set_scissor(&mut self, rect: Rect) -> RhiResult<()> {
        validate_rect(rect, "the scissor rect")?;
        self.scissor = Some(rect);
        Ok(())
    }

    /// Sets the blend constant.
    ///
    /// Not validated, deliberately: section 30 states no rule for a color, and the
    /// blend constant's components are weights rather than a bit pattern, so a
    /// value outside `0.0..=1.0` is a caller's choice and not a portable error.
    /// Refusing it here would make a legal native command illegal.
    pub fn set_blend_constant(&mut self, color: Color) -> RhiResult<()> {
        self.blend_constant = color;
        Ok(())
    }

    /// Sets the stencil reference value.
    pub fn set_stencil_reference(&mut self, value: u32) -> RhiResult<()> {
        self.stencil_reference = value;
        Ok(())
    }

    /// Draws a range of vertices.
    ///
    /// Section 32.3's list, and section 32.4's rule that a draw is where a bound
    /// resource becomes an *actual* use: the attachments, the shader-referenced
    /// binding resources, and the vertex buffers all produce uses here and nowhere
    /// else.
    pub fn draw(
        &mut self,
        vertices: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
    ) -> RhiResult<()> {
        let pipeline = self.bound_pipeline()?;
        self.validate_groups(&pipeline)?;
        self.validate_vertex_buffers(&pipeline, &vertices, &instances)?;

        let uses = self.draw_uses(false)?;
        let draw = RasterDraw {
            pipeline,
            groups: self.groups.clone(),
            vertex_buffers: self.vertex_buffers.clone(),
            index: None,
            viewport: self.viewport,
            scissor: self.scissor,
            blend_constant: self.blend_constant,
            stencil_reference: self.stencil_reference,
            range: vertices,
            instances,
            base_vertex: 0,
        };
        self.recorder.record_command(
            RecordedPayload::RasterDraw(Box::new(draw)),
            uses,
            RASTER_DOMAIN,
        );
        Ok(())
    }

    /// Draws a range of indices.
    ///
    /// `base_vertex` is recorded but not range-checked. An indexed draw's vertex
    /// fetch address depends on index *values*, which the RHI does not read, so the
    /// honest portable bound is the index range and the buffer ranges it addresses.
    /// Section 32.3 asks for exactly those two.
    pub fn draw_indexed(
        &mut self,
        indices: core::ops::Range<u32>,
        base_vertex: i32,
        instances: core::ops::Range<u32>,
    ) -> RhiResult<()> {
        let pipeline = self.bound_pipeline()?;
        self.validate_groups(&pipeline)?;

        let Some(index) = self.index.clone() else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "an indexed draw needs an index buffer, and this raster scope has none bound",
            ));
        };
        let element_size = match index.format {
            IndexFormat::Uint16 => 2u64,
            IndexFormat::Uint32 => 4u64,
        };
        let needed = u64::from(indices.end) * element_size;
        if needed > index.binding.range.size {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "an indexed draw reaches byte {} of an index buffer range of {} bytes",
                    needed, index.binding.range.size
                ),
            ));
        }

        // Section 32.3's strip rule, read as a *requirement* rather than as three
        // independent facts: a strip topology needs a primitive-restart index
        // format, and an indexed draw must use the format the pipeline declared. A
        // non-strip topology places no requirement on the index format — section
        // 25.1 makes `strip_index_format` meaningful only for a strip — so an
        // indexed triangle-list draw is not refused for having none.
        if pipeline.descriptor().primitive.topology.is_strip() {
            match pipeline.descriptor().primitive.strip_index_format {
                Some(declared) if declared == index.format => {}
                Some(_) => {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "this indexed draw's index format does not match the pipeline's strip \
                         index format",
                    ));
                }
                None => {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "an indexed strip draw needs a pipeline that declares a strip index \
                         format",
                    ));
                }
            }
        }

        self.validate_vertex_buffers(&pipeline, &indices, &instances)?;

        let uses = self.draw_uses(true)?;
        let draw = RasterDraw {
            pipeline,
            groups: self.groups.clone(),
            vertex_buffers: self.vertex_buffers.clone(),
            index: Some(index),
            viewport: self.viewport,
            scissor: self.scissor,
            blend_constant: self.blend_constant,
            stencil_reference: self.stencil_reference,
            range: indices,
            instances,
            base_vertex,
        };
        self.recorder.record_command(
            RecordedPayload::RasterDraw(Box::new(draw)),
            uses,
            RASTER_DOMAIN,
        );
        Ok(())
    }

    /// Pushes a label onto this scope's own debug-group stack.
    ///
    /// Section 36 gives a scope its own stack rather than one shared with the
    /// recorder, because the recorder's stack describes commands outside any pass
    /// and this one describes the pass's interior; the two nest in the recording
    /// but never in each other.
    pub fn push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.debug_stack.push(label.to_owned());
        self.recorder.record_command(
            RecordedPayload::DebugPush(Label(Some(label.to_owned()))),
            Vec::new(),
            RASTER_DOMAIN,
        );
        Ok(())
    }

    /// Pops this scope's own debug-group stack.
    ///
    /// An unmatched pop is a parameter error and poisons nothing: section 29.3
    /// keeps parameter refusal out of the poisoning category, and the recording is
    /// still describable after a caller is told it made a mistake.
    pub fn pop_debug_group(&mut self) -> RhiResult<()> {
        if self.debug_stack.pop().is_none() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "pop_debug_group has no matching push_debug_group on this raster scope",
            ));
        }
        self.recorder
            .record_command(RecordedPayload::DebugPop, Vec::new(), RASTER_DOMAIN);
        Ok(())
    }

    /// Inserts a marker without changing the stack.
    pub fn insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.recorder.record_command(
            RecordedPayload::DebugMarker(Label(Some(label.to_owned()))),
            Vec::new(),
            RASTER_DOMAIN,
        );
        Ok(())
    }

    /// Ends the scope and returns the recorder to the open state.
    ///
    /// A non-empty debug-group stack refuses the end, and the scope is deliberately
    /// left unterminated: section 29.3 puts a failed scope finalization in the
    /// *poisoning* category, and the `Drop` that follows this refusal is what
    /// carries that out. A caller that reaches here has left a debug group open
    /// inside the recording, and no later command can close it.
    pub fn end(mut self) -> RhiResult<()> {
        if !self.debug_stack.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "this raster scope still has {} debug group(s) open at end()",
                    self.debug_stack.len()
                ),
            ));
        }

        let uses = self.end_uses();
        self.recorder
            .record_command(RecordedPayload::RasterEnd, uses, RASTER_DOMAIN);
        self.recorder.set_phase(RecorderPhase::Open);
        self.ended = true;
        Ok(())
    }

    /// The viewport that will be lowered, with the portable default applied.
    ///
    /// Section 32.3 fixes the default as the full attachment extent "and not
    /// backend-selected behavior", which is only checkable if the resolved value is
    /// readable rather than inferred from whatever a native API defaults to.
    pub fn effective_viewport(&self) -> Viewport {
        self.viewport.unwrap_or(Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })
    }

    /// The scissor rect that will be lowered, with the portable default applied.
    pub fn effective_scissor(&self) -> Rect {
        self.scissor.unwrap_or(Rect {
            x: 0,
            y: 0,
            width: self.extent.width,
            height: self.extent.height,
        })
    }

    /// The blend constant currently in force.
    pub fn blend_constant(&self) -> Color {
        self.blend_constant
    }

    /// The stencil reference currently in force.
    pub fn stencil_reference(&self) -> u32 {
        self.stencil_reference
    }

    /// The pipeline bound at this point, or a refusal.
    ///
    /// Section 32.3's first draw check. Returned by value because the draw records
    /// its own copy of the state that was current when it was issued: a later
    /// `set_pipeline` must not retroactively change an earlier draw.
    fn bound_pipeline(&self) -> RhiResult<RasterPipeline> {
        match &self.pipeline {
            Some(pipeline) => Ok(pipeline.clone()),
            None => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a draw needs a pipeline, and none is bound in this raster scope",
            )),
        }
    }

    /// Checks the bind groups the pipeline's interface actually uses.
    ///
    /// The rule itself is [`validate_bound_groups`]'s, because a compute scope's
    /// dispatch obeys the same one; what is raster-specific is only that a *draw*
    /// is where it is checked.
    fn validate_groups(&self, pipeline: &RasterPipeline) -> RhiResult<()> {
        validate_bound_groups(pipeline.interface(), &self.groups)
    }

    /// Checks the vertex buffers the input state actually reads.
    ///
    /// Two of section 32.3's three vertex items. The `VERTEX` usage bit was checked
    /// when the buffer was bound, because it is a property of the buffer; what is
    /// checked here is the *access bound*, which needs the vertex or index range. A
    /// slot the input state declares with no attributes reads nothing and needs no
    /// binding.
    ///
    /// The element count comes from the step mode: a per-vertex binding advances
    /// once per vertex or index, a per-instance binding once per instance. That is
    /// section 32.3's "per-vertex/instance access does not exceed the buffer range",
    /// and it is why the bound is a stride-times-count product rather than a count.
    fn validate_vertex_buffers(
        &self,
        pipeline: &RasterPipeline,
        elements: &core::ops::Range<u32>,
        instances: &core::ops::Range<u32>,
    ) -> RhiResult<()> {
        for (slot, layout) in pipeline
            .descriptor()
            .vertex_input
            .buffers
            .iter()
            .enumerate()
        {
            if layout.attributes.is_empty() {
                continue;
            }
            let slot = slot as u32;
            let Some((_, bound)) = self
                .vertex_buffers
                .iter()
                .find(|(bound_slot, _)| *bound_slot == slot)
            else {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "the vertex input state reads slot {} and this raster scope has no buffer \
                         bound there",
                        slot
                    ),
                ));
            };

            for attribute in &layout.attributes {
                let end = attribute.offset + u64::from(attribute.format.byte_size());
                if end > layout.stride {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        format!(
                            "vertex attribute {} at slot {} reaches byte {} of a {}-byte stride",
                            attribute.location.get(),
                            slot,
                            end,
                            layout.stride
                        ),
                    ));
                }
            }

            let count = match layout.step_mode {
                VertexStepMode::Vertex => u64::from(elements.end),
                VertexStepMode::Instance => u64::from(instances.end),
            };
            let needed = count.saturating_mul(layout.stride);
            if needed > bound.range.size {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "the draw reaches byte {} of the vertex buffer bound at slot {}, which \
                         covers {} bytes",
                        needed, slot, bound.range.size
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The uses a draw produces.
    ///
    /// Section 32.4's list, in its own order: active attachments, shader-referenced
    /// binding resources, vertex buffers, and the index buffer of an indexed draw.
    /// `indexed` is an argument rather than a read of `self.index` so that each call
    /// site states which verb it is, instead of one of them trusting a field to be
    /// `None`.
    ///
    /// Fallible, because a bound group can be one whose slots disagree with the
    /// layout it was built against — a state creation refuses, and therefore one
    /// that only a path which skipped creation can produce. Reporting it here is
    /// what keeps that path from lowering a resource set nobody checked.
    fn draw_uses(&self, indexed: bool) -> RhiResult<Vec<ResourceUse>> {
        let mut uses = Vec::new();

        // Section 37.3: a draw reads and writes its raster attachments.
        for (_, attachment) in &self.colors {
            uses.extend(color_use(
                &attachment.view,
                AccessMask::COLOR_READ.union(AccessMask::COLOR_WRITE),
                TextureUseIntent::ColorAttachment,
            ));
        }
        if let Some(depth) = &self.depth_stencil {
            uses.push(depth_draw_use(depth));
        }

        // Section 37.2: not every resource in the group, only the slots the
        // pipeline's interface can reach.
        for group in &self.groups {
            uses.extend(bound_group_uses(&group.group)?);
        }

        for (_, binding) in &self.vertex_buffers {
            uses.push(buffer_use(
                &binding.buffer,
                binding.range,
                PipelineScope::VERTEX,
                AccessMask::VERTEX_READ,
            ));
        }

        if indexed {
            if let Some(index) = &self.index {
                uses.push(buffer_use(
                    &index.binding.buffer,
                    index.binding.range,
                    PipelineScope::VERTEX,
                    AccessMask::INDEX_READ,
                ));
            }
        }

        Ok(uses)
    }

    /// The uses a scope's *end* produces.
    ///
    /// A resolve reads the multisampled source and writes the single-sampled target
    /// when the pass ends, which is why it is recorded here rather than at a draw:
    /// the ordering inside a recording is what hazard lowering reads (section 37.1),
    /// and a resolve that no draw fed is still a write at the end of the pass.
    fn end_uses(&self) -> Vec<ResourceUse> {
        let mut uses = Vec::new();
        for (_, attachment) in &self.colors {
            if let Some(resolve) = &attachment.resolve {
                uses.extend(color_use(
                    resolve,
                    AccessMask::COLOR_WRITE,
                    TextureUseIntent::ResolveDst,
                ));
            }
        }
        uses
    }
}

impl Drop for RasterScope<'_> {
    /// Poisons the recorder when a scope did not reach `end`.
    ///
    /// Section 29.2's "drop without end -> Poisoned". Drop performs no failing
    /// native finalize — the recorder holds no native encoder — so this is the one
    /// thing it may do: mark the recording unusable and say why. It is also what
    /// carries out section 29.3's "scope finalization failure poisons" when
    /// [`RasterScope::end`] refuses.
    fn drop(&mut self) {
        if !self.ended {
            self.recorder
                .poison("a raster scope was dropped without a successful end()");
        }
    }
}

/// What one depth or stencil plane does over the course of a scope.
///
/// The two plane types are separate enums in the attachment module, so flattening
/// them here is what lets the use mapping be written once instead of twice with a
/// type substituted. The three attached shapes are the three cases section 37.3
/// distinguishes: `ReadOnly` never writes, and a read-write plane's *load*
/// operation decides whether the scope begins with a read or a write.
#[derive(Clone, Copy)]
enum PlaneEffect {
    /// No plane is attached at this position.
    Absent,
    /// Attached read-only.
    ReadOnly,
    /// Attached read-write, loaded.
    Loaded,
    /// Attached read-write, cleared.
    Cleared,
}

/// What the depth plane does.
fn depth_plane(depth: &DepthStencilAttachment) -> PlaneEffect {
    match &depth.depth {
        None => PlaneEffect::Absent,
        Some(DepthAttachmentMode::ReadOnly) => PlaneEffect::ReadOnly,
        Some(DepthAttachmentMode::ReadWrite { load, .. }) => plane_effect(load.is_clear()),
    }
}

/// What the stencil plane does.
fn stencil_plane(depth: &DepthStencilAttachment) -> PlaneEffect {
    match &depth.stencil {
        None => PlaneEffect::Absent,
        Some(StencilAttachmentMode::ReadOnly) => PlaneEffect::ReadOnly,
        Some(StencilAttachmentMode::ReadWrite { load, .. }) => plane_effect(load.is_clear()),
    }
}

/// Turns "the scope clears this plane" into the attached shape.
fn plane_effect(cleared: bool) -> PlaneEffect {
    if cleared {
        PlaneEffect::Cleared
    } else {
        PlaneEffect::Loaded
    }
}

/// The use a depth/stencil attachment produces when a scope *begins*.
///
/// Section 37.3's beginning rows applied per plane: a read-only plane and a loaded
/// plane are reads of the existing contents, and a cleared plane is a write. Both
/// planes of one view share one use record, because they are one allocation —
/// a hazard lookup that saw the same range twice would learn nothing.
fn depth_begin_use(depth: &DepthStencilAttachment) -> Option<ResourceUse> {
    let mut access: Option<AccessMask> = None;
    let mut writes = false;
    for (effect, read_bit, write_bit) in [
        (
            depth_plane(depth),
            AccessMask::DEPTH_READ,
            AccessMask::DEPTH_WRITE,
        ),
        (
            stencil_plane(depth),
            AccessMask::STENCIL_READ,
            AccessMask::STENCIL_WRITE,
        ),
    ] {
        let mask = match effect {
            PlaneEffect::Absent => continue,
            PlaneEffect::ReadOnly | PlaneEffect::Loaded => read_bit,
            PlaneEffect::Cleared => {
                writes = true;
                write_bit
            }
        };
        access = Some(match access {
            Some(existing) => existing.union(mask),
            None => mask,
        });
    }

    access.map(|access| {
        texture_use_of_view(
            &depth.view,
            PipelineScope::FRAGMENT,
            access,
            if writes {
                TextureUseIntent::DepthStencilWrite
            } else {
                TextureUseIntent::DepthStencilRead
            },
        )
    })
}

/// The use a depth/stencil attachment produces at a draw.
///
/// Section 31.2's rule that a read-only plane produces no attachment *write* use is
/// why the two read-write shapes take their read and write bits while `ReadOnly`
/// takes only the read bit: a depth test that only reads is a hazard input, not an
/// output a later pass must wait on. The load operation is irrelevant here — it was
/// consumed at scope begin — so `Loaded` and `Cleared` differ only there.
fn depth_draw_use(depth: &DepthStencilAttachment) -> ResourceUse {
    let mut access: Option<AccessMask> = None;
    let mut writes = false;
    for (effect, read_bit, write_bit) in [
        (
            depth_plane(depth),
            AccessMask::DEPTH_READ,
            AccessMask::DEPTH_WRITE,
        ),
        (
            stencil_plane(depth),
            AccessMask::STENCIL_READ,
            AccessMask::STENCIL_WRITE,
        ),
    ] {
        let mask = match effect {
            PlaneEffect::Absent => continue,
            PlaneEffect::ReadOnly => read_bit,
            PlaneEffect::Loaded | PlaneEffect::Cleared => {
                writes = true;
                read_bit.union(write_bit)
            }
        };
        access = Some(match access {
            Some(existing) => existing.union(mask),
            None => mask,
        });
    }

    texture_use_of_view(
        &depth.view,
        PipelineScope::FRAGMENT,
        access.unwrap_or(AccessMask::DEPTH_READ),
        if writes {
            TextureUseIntent::DepthStencilWrite
        } else {
            TextureUseIntent::DepthStencilRead
        },
    )
}

/// The uses an attachment set produces when a scope *begins*.
///
/// Section 37.3's two rows that happen at the beginning: `Load` is a read of the
/// existing contents and `Clear` is a write, so a scope that clears and stores with
/// no draw at all still declares a write. That case is exactly why these uses exist
/// separately from a draw's, and it is what makes a `RasterScope` with no draw a
/// meaningful recording rather than an empty one.
///
/// A `Discard` store adds nothing: section 37.3 makes the result undefined, so
/// there is no later pass with anything to depend on. The attachment is still in the
/// recorded command's payload, so the recording says what happened even though the
/// hazard summary has nothing to add.
fn begin_uses(
    colors: &[(u32, ColorAttachment)],
    depth_stencil: Option<&DepthStencilAttachment>,
) -> Vec<ResourceUse> {
    let mut uses = Vec::new();

    for (_, attachment) in colors {
        let access = match &attachment.load {
            LoadOp::Load => AccessMask::COLOR_READ,
            LoadOp::Clear(_) => AccessMask::COLOR_WRITE,
        };
        uses.extend(color_use(
            &attachment.view,
            access,
            TextureUseIntent::ColorAttachment,
        ));
    }

    if let Some(depth) = depth_stencil {
        uses.extend(depth_begin_use(depth));
    }

    uses
}

/// The extent that fixes a scope's portable dynamic-state defaults.
///
/// Section 31.4 makes every main attachment share one extent, so the first
/// attachment answers for the set. A validated scope always has one; the fallback
/// exists so that this function does not panic on a set that validation is about to
/// refuse.
fn primary_extent(
    colors: &[(u32, ColorAttachment)],
    depth_stencil: Option<&DepthStencilAttachment>,
) -> Extent3d {
    if let Some((_, attachment)) = colors.first() {
        return attachment.view.extent();
    }
    if let Some(depth) = depth_stencil {
        return depth.view.extent();
    }
    Extent3d {
        width: 0,
        height: 0,
        depth: 1,
    }
}

/// The uses one color attachment view produces with a given access.
///
/// A frame records through its own variant and its own identity, because it is not
/// a texture: there is no subresource range to name, since a caller does not choose
/// which layer of an acquired image it renders into.
fn color_use(
    view: &ColorAttachmentView,
    access: AccessMask,
    intent: TextureUseIntent,
) -> Vec<ResourceUse> {
    match view {
        ColorAttachmentView::Texture(texture_view) => vec![texture_use_of_view(
            texture_view,
            PipelineScope::FRAGMENT,
            access,
            intent,
        )],
        ColorAttachmentView::Frame(frame) => {
            vec![frame_use(frame, PipelineScope::FRAGMENT, access)]
        }
    }
}
