//! The raster family's verbs: the pass bracket, its recipe, its vertex input
//! and its two draws.
//!
//! This is [`super::compute`]'s counterpart, and it exists for the same reason
//! that one does: a family's command verbs are one subject -- what the family
//! admits, what it records into the admission, and what it issues out of it --
//! and the trait impl in [`super::backend`] is a delegation surface that may
//! hold no more than the trait's own declaration order wants to show.  The
//! compute half already lived here; the raster half did not, which is the
//! asymmetry this module removes rather than a line count it reduces.
//!
//! Responsibility: drive Layer 2's raster verbs in the order the contract
//! requires, and own the raster pass's ownership bookkeeping.  Not owned here:
//! the recording machinery these verbs drive ([`super::encoder`], which holds
//! `open_pass`, `raster_pass`, `commit`, `bind_slots` and `destroy_owned`), the
//! lowering they feed it ([`super::pass`], which is pure), what a recipe or a
//! binding slot holds ([`super::object`]), and which verbs the common contract
//! declares or in what order ([`super::backend`], which is the trait impl and
//! keeps the trait's declaration order as it is written there).

use std::ops::Range;

use fluxel_rendergraph::{IndexFormat, RasterPassDescriptor};

use crate::webgl2::api::{
    GlDrawCommand, GlError, GlIndexBinding, GlIndexedDraw, GlNonIndexedDraw,
    GlRenderPassDescriptor, GlVertexBufferBinding,
};
use crate::webgl2::state::GlStateBackend;

use super::GlCompatibilityDevice;
use super::compute::ComputeDomain;
use super::encoder::{GlEncoder, InstalledPipeline, OpenPass, PassShape};
use super::failure::{self, malformed, pass_open, unsupported};
use super::object;
use super::pass;

/// The instance count of a draw that asks for exactly one instance.
///
/// This family's instanced draw is `draw_advanced_raster`, an optional verb with
/// per-instance offsets this adapter has no lowering for, so a range naming more
/// than one instance is refused by name rather than silently drawn once.
///
/// A range naming *none* is refused as well, and by this verb rather than by the
/// backend.  Zero instances is a legal command that rasterizes nothing, so the
/// check belongs last -- but "last" here means inside the provider, and a zero
/// that reached it would be reported against `draw-raster`, an operation name
/// the frame never issued.  So the refusal is made where the name is still the
/// caller's.  It is also a request the executor cannot produce: it refuses an
/// empty instance range before any backend sees one, which leaves a caller that
/// bypassed the executor as the only way to arrive here.
fn single_instance(operation: &'static str, instances: &Range<u32>) -> Result<u32, GlError> {
    match instances.len() {
        0 => Err(malformed(
            operation,
            "a draw that runs no instance has nothing to rasterize",
        )),
        1 => Ok(1),
        _ => Err(unsupported(
            operation,
            "this family's instanced draw is an optional verb this adapter has no lowering for, so a draw asks for at most one instance",
        )),
    }
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// The body of [`super::backend`]'s `begin_raster`, whose doc is the
    /// contract-facing half of the same explanation.
    pub(super) fn open_raster_pass(
        &mut self,
        encoder: &mut GlEncoder,
        descriptor: &RasterPassDescriptor<'_, crate::webgl2::api::TextureId>,
    ) -> Result<(), GlError> {
        self.refresh();
        if encoder.pass.is_some() {
            return Err(pass_open("begin-raster"));
        }
        self.machine
            .backend()
            .validate_object_context("begin-raster", encoder.context)?;
        let attachment = pass::admit(descriptor.colors, descriptor.depth_stencil.as_ref())?;
        // The adapter's own record of what it created.  A texture this device did
        // not create has no shape to lower, and the refusal names that rather
        // than reporting a missing attachment.
        let facts = *self.attachments.get(attachment.texture).ok_or_else(|| {
            malformed(
                "begin-raster",
                "the pass attaches a texture this device did not create",
            )
        })?;
        self.machine
            .backend()
            .validate_object_context("begin-raster", attachment.texture.context)?;
        let views = vec![pass::view(*attachment.texture, facts, attachment.range)];
        let colors = pass::attachments(descriptor.colors, &views)?;
        let requested = pass::framebuffer(views, None);
        let (framebuffer, owned) = self
            .machine
            .framebuffer_for(&requested)
            .map_err(failure::into_gl_error)?;

        let mut open = OpenPass::raster(facts);
        if owned {
            // Narrowed by hand rather than through `OpenPass::raster`, which
            // returns a `Result`: the pass was built as a raster pass one line
            // above, so the narrowing cannot refuse, and this is the one place in
            // this module where an error path would have to be invented rather
            // than reported.  The framebuffer is already derived and has no
            // second name, so a `?` here would leak it.
            if let PassShape::Raster(shape) = &mut open.shape {
                shape.owned_framebuffers.push(framebuffer);
            }
        }
        encoder.pass = Some(open);

        let render_pass = GlRenderPassDescriptor {
            framebuffer,
            color_attachments: colors,
            depth_stencil_attachment: None,
        };
        if let Err(error) = self
            .machine
            .begin_pass(render_pass)
            .map_err(failure::into_gl_error)
        {
            // A pass that never opened has no `end_raster` coming: the executor
            // returns on this error rather than bracketing the callback, so
            // whatever this encoder came to own is destroyed here instead of by
            // a close that will not happen.
            if let Some(abandoned) = encoder.pass.take() {
                let _ = self.destroy_owned(abandoned);
            }
            return Err(error);
        }
        Ok(())
    }

    /// The body of [`super::backend`]'s `end_raster`.
    pub(super) fn close_raster_pass(&mut self, encoder: &mut GlEncoder) -> Result<(), GlError> {
        self.refresh();
        let pass = encoder.take_raster("end-raster")?;
        let closed = self.machine.end_pass().map_err(failure::into_gl_error);
        // Destroyed whether or not the boundary closed: these objects are
        // unreachable either way, and a failure to end the pass is not a reason
        // to leak them as well.  Not destroyed, though, when the context was
        // replaced while the pass was open -- their identities belong to an epoch
        // the backend no longer accepts, the context's own teardown already
        // released them, and asking would turn a context loss into a second,
        // unrelated failure on the frame's error path.
        let released = if encoder.context == self.machine.backend().context_stamp() {
            self.destroy_owned(pass)
        } else {
            Ok(())
        };
        closed.and(released)
    }

    /// The body of [`super::backend`]'s `set_raster_pipeline`.
    pub(super) fn record_raster_pipeline(
        &mut self,
        encoder: &mut GlEncoder,
        pipeline: &object::RasterPipeline,
    ) -> Result<(), GlError> {
        let pass = self.open_pass(encoder, "set-raster-pipeline")?;
        // Refused when the open pass is a compute pass, which is what the
        // narrowing here is for: the two kinds do not share a pipeline slot, and a
        // raster recipe recorded in a compute pass would be installed by the next
        // dispatch.
        pass.raster_shape("set-raster-pipeline")?;
        // The recorded bindings are *kept*, and checked against this recipe
        // rather than dropped with the previous one: they are facts about what
        // the frame resolved, and a recipe that does not read them is a mistake
        // worth reporting at the bind that made it rather than a silent discard.
        pass.pipeline = Some(InstalledPipeline::Raster(pipeline.clone()));
        Ok(())
    }

    /// The body of [`super::backend`]'s `set_vertex_buffer`.
    pub(super) fn record_vertex_buffer(
        &mut self,
        encoder: &mut GlEncoder,
        slot: u32,
        buffer: &crate::webgl2::api::BufferId,
        offset: u64,
    ) -> Result<(), GlError> {
        let shape = self.raster_pass(encoder, "set-vertex-buffer")?;
        shape.vertex.retain(|bound| bound.slot != slot);
        shape.vertex.push(GlVertexBufferBinding {
            slot,
            buffer: *buffer,
            offset,
        });
        Ok(())
    }

    /// The body of [`super::backend`]'s `set_index_buffer`.
    pub(super) fn record_index_buffer(
        &mut self,
        encoder: &mut GlEncoder,
        buffer: &crate::webgl2::api::BufferId,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), GlError> {
        let shape = self.raster_pass(encoder, "set-index-buffer")?;
        shape.index = Some(GlIndexBinding {
            buffer: *buffer,
            format: pass::index_format(format),
            offset,
        });
        Ok(())
    }

    /// The body of [`super::backend`]'s `draw`.
    pub(super) fn issue_draw(
        &mut self,
        encoder: &mut GlEncoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<(), GlError> {
        self.refresh();
        let kernel = encoder.kernel("draw")?;
        if pass::indexed(kernel) {
            return Err(malformed(
                "draw",
                "this artifact takes its vertices from an index buffer, so it is drawn with draw-indexed",
            ));
        }
        let instance_count = single_instance("draw", &instances)?;
        if vertices.is_empty() {
            return Err(malformed(
                "draw",
                "a draw with no vertices has nothing to rasterize",
            ));
        }
        self.commit("draw", encoder)?;
        self.machine
            .backend()
            .draw_raster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
                first_vertex: vertices.start,
                vertex_count: vertices.len() as u32,
                instance_count,
            }))
    }

    /// The body of [`super::backend`]'s `draw_indexed`.
    pub(super) fn issue_indexed_draw(
        &mut self,
        encoder: &mut GlEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<(), GlError> {
        self.refresh();
        let kernel = encoder.kernel("draw-indexed")?;
        if !pass::indexed(kernel) {
            return Err(malformed(
                "draw-indexed",
                "this artifact draws its vertices without an index buffer, so it is drawn with draw",
            ));
        }
        let instance_count = single_instance("draw-indexed", &instances)?;
        if indices.is_empty() {
            return Err(malformed(
                "draw-indexed",
                "an indexed draw with no indices has nothing to rasterize",
            ));
        }
        if base_vertex != 0 {
            return Err(unsupported(
                "draw-indexed",
                "this family adds the base vertex to each index inside the shader pipeline, which is an optional verb this adapter has no lowering for",
            ));
        }
        self.commit("draw-indexed", encoder)?;
        self.machine
            .backend()
            .draw_raster(GlDrawCommand::Indexed(GlIndexedDraw {
                first_index: indices.start,
                index_count: indices.len() as u32,
                instance_count,
            }))
    }
}
