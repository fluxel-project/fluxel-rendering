//! One recording in progress, and what the device does with it.
//!
//! [`GlEncoder`] is what a frame records into, [`OpenPass`] is what it has
//! recorded inside the pass it has open, and [`GlCommandBuffer`] is what is left
//! once the recording finishes.  The methods below are the device's half of that:
//! admitting a pass and deriving its framebuffer, the commit point a draw
//! reaches, applying a set one binding at a time, and destroying what a pass came
//! to own.
//!
//! The recording verbs themselves are one file over, in [`super::backend`],
//! because they are the common contract's methods and a trait has one impl block
//! per type.  That is why the record's fields are `pub(super)`: the verbs that
//! write them cannot live here, and a record only a reader inside this file could
//! update would be a record nothing updates.  Nothing outside this module tree
//! reaches them, and the visibility is what says so rather than a comment.
//!
//! Not owned here: what a pass *becomes* ([`super::pass`], which is pure and
//! checkable without a context), what a binding slot holds ([`super::object`]),
//! and when the adapter refreshes or retires ([`super`]).  Why the pass is
//! recorded rather than issued verb by verb is argued on [`GlEncoder`] itself,
//! where the record it needs is defined.

use fluxel_rendergraph::{BufferRange, TextureRange};

use crate::resource::RasterKernel;
use crate::webgl2::api::{
    ContextStamp, FramebufferId, GlError, GlIndexBinding, GlRasterPipeline, GlScissorRect,
    GlVertexBufferBinding, GlViewport, ProgramId,
};
use crate::webgl2::state::GlStateBackend;

use super::GlCompatibilityDevice;
use super::failure::{self, malformed, no_pass, unsupported};
use super::object::{self, BindingSlot};
use super::pass;

/// One recording in progress.
///
/// It carries the context generation it was opened against and, while a pass is
/// open, what the frame has recorded inside it.  The context stamp is what lets
/// submission reject a command buffer finished against a generation that has
/// since been replaced.
///
/// # Why a pass is recorded and not issued as its verbs arrive
///
/// This family's context is immediate, so a verb *could* issue -- and the copy
/// verbs do.  A raster pass cannot, and the reason is a shape difference rather
/// than a policy: Layer 2 installs rasterization state as one value that names
/// its vertex array (`GlRasterPipeline { program, vertex_array, state }`), and
/// deriving that array needs the vertex input, while the contract supplies the
/// pipeline first and the vertex buffers after it.  So the verbs record, and the
/// draw drives Layer 2 once everything is in hand.
///
/// That is not deferred execution: nothing here is queued for a later frame, and
/// a pass that records a draw issues it before `draw` returns.  What the record
/// buys is that the *order* Layer 2 needs is the order the commit uses, instead
/// of an order this vocabulary cannot express.
pub(crate) struct GlEncoder {
    pub(super) context: ContextStamp,
    pub(super) pass: Option<OpenPass>,
}

/// What the frame has recorded inside the pass this encoder has open.
///
/// Every field is per-pass and cleared where the pass ends, which is the same
/// lifetime Layer 1's own pass scope has: a binding recorded in one pass is not
/// in force in the next, and neither is a viewport.
pub(super) struct OpenPass {
    /// The colour attachment's shape, which the viewport defaults to and the
    /// pipeline's sample count is taken from.
    pub(super) target: pass::Attachment,
    /// The recipe the frame selected, if it has selected one.
    pub(super) pipeline: Option<object::RasterPipeline>,
    /// The binding set the frame resolved, with the artifact it was resolved
    /// *for*: the kernel is kept beside the slots because the slots alone cannot
    /// answer whether they belong to the recipe installed at the draw, and a
    /// pipeline installed after a set is the one way the two can disagree.
    /// Replaced rather than accumulated: a fixed artifact declares one set, so a
    /// second one is a different recipe's.
    pub(super) bindings: Option<(RasterKernel, Vec<BindingSlot>)>,
    /// The vertex buffers the frame bound, by slot.
    pub(super) vertex: Vec<GlVertexBufferBinding>,
    /// The index buffer the frame bound.
    pub(super) index: Option<GlIndexBinding>,
    /// The viewport, defaulting to the whole attachment.
    pub(super) viewport: GlViewport,
    /// The scissor, absent until the frame sets one.
    pub(super) scissor: Option<GlScissorRect>,
    /// Framebuffers and programs whose ownership Layer 2 handed to this encoder.
    /// Destroyed when the pass ends, because a caller-owned object is one no
    /// cache will ever free.
    pub(super) owned_framebuffers: Vec<FramebufferId>,
    pub(super) owned_programs: Vec<ProgramId>,
}

impl GlEncoder {
    /// The recipe this pass has selected, refusing when it has none.
    ///
    /// Asked before the commit rather than after it, so that a draw issued
    /// through the wrong verb of the pair -- a non-indexed draw of an indexed
    /// artifact, say -- is refused before anything is installed for it.
    pub(super) fn kernel(&self, operation: &'static str) -> Result<RasterKernel, GlError> {
        self.pass
            .as_ref()
            .and_then(|pass| pass.pipeline.as_ref())
            .map(object::RasterPipeline::kernel)
            .ok_or_else(|| {
                malformed(
                    operation,
                    "no raster pipeline is installed in this pass, so there is nothing to draw with",
                )
            })
    }
}

/// One finished recording, ready to submit.
pub(crate) struct GlCommandBuffer {
    pub(super) context: ContextStamp,
}

impl<B: GlStateBackend> GlCompatibilityDevice<B> {
    /// The pass an encoder has open, or the refusal that it has none.
    ///
    /// Every verb that records into a pass reads it through here, so that the
    /// four of them report the same mistake the same way and none of them can
    /// reach the state without asking.
    pub(super) fn open_pass<'e>(
        &mut self,
        encoder: &'e mut GlEncoder,
        operation: &'static str,
    ) -> Result<&'e mut OpenPass, GlError> {
        encoder.pass.as_mut().ok_or_else(|| no_pass(operation))
    }

    /// Drives Layer 2 so the backend holds everything the pass's draw needs.
    ///
    /// Called at the draw and not at each verb, because a GL-family pipeline
    /// carries its program, its vertex array and its rasterization state as one
    /// value while the contract supplies those three in the order a
    /// pipeline-first API supplies them -- so the pieces can only be assembled
    /// once all of them have arrived.  See the module documentation.
    ///
    /// The order inside is the one Layer 2 documents, and each step is here for a
    /// reason rather than by convention:
    ///
    /// 1. **The program**, because `program_for` links when its cache misses and a
    ///    link invalidates the mirror's installed pipeline -- so it has to happen
    ///    before the install below rather than after it, or the install would be
    ///    recorded against a pipeline the link already forgot.
    /// 2. **The vertex input**, because the pipeline has to *name* the array, and
    ///    the identity comes from the geometry domain's derivation.
    /// 3. **The pipeline**, which installs that array -- and binds it *without*
    ///    re-emitting its attribute description, which is why the geometry domain
    ///    drops its claim here.
    /// 4. **The vertex input again**, which is the only thing that re-enables the
    ///    attributes after that install.
    /// 5. **The bindings**, one at a time, because the texture domain's request is
    ///    single-valued: `active_texture`, `bind_texture` and `bind_sampler` each
    ///    *overwrite* the previous request, so a whole set recorded and applied
    ///    once would bind only its last unit.
    pub(super) fn commit(
        &mut self,
        operation: &'static str,
        encoder: &mut GlEncoder,
    ) -> Result<(), GlError> {
        let Some(pass) = encoder.pass.as_mut() else {
            return Err(no_pass(operation));
        };
        let recipe = pass.pipeline.clone().ok_or_else(|| {
            malformed(
                operation,
                "no raster pipeline is installed in this pass, so there is nothing to draw with",
            )
        })?;
        if let Some((resolved_for, _)) = pass.bindings.as_ref() {
            // A pipeline installed *after* the set is the one way the two can
            // disagree: `set_bindings` checks the set against the pipeline that
            // was installed when it was called, and this is the check that holds
            // for the pipeline installed now.
            if *resolved_for != recipe.kernel() {
                return Err(malformed(
                    operation,
                    "the binding set in this pass was resolved for a different artifact than the one installed at the draw",
                ));
            }
        }
        let input = pass::vertex_input(&recipe, &pass.vertex, pass.index, operation)?;

        let (program, _reflection, owned) = self
            .machine
            .program_for(recipe.descriptor())
            .map_err(failure::into_gl_error)?;
        if owned {
            pass.owned_programs.push(program);
        }

        self.machine.set_vertex_input(input);
        let _ = self
            .machine
            .apply_geometry()
            .map_err(failure::into_gl_error)?;
        let vertex_array = self
            .machine
            .geometry()
            .applied_vertex_array()
            .ok_or_else(|| {
                malformed(
                    operation,
                    "the vertex input was bound without naming a vertex array",
                )
            })?;

        self.machine.set_pipeline(&GlRasterPipeline {
            program,
            vertex_array,
            state: pass::raster_state(
                recipe.kernel(),
                pass.viewport,
                pass.scissor,
                pass.target.sample_count(),
            ),
        });
        let _ = self
            .machine
            .apply_pipeline()
            .map_err(failure::into_gl_error)?;

        let _ = self
            .machine
            .apply_geometry()
            .map_err(failure::into_gl_error)?;
        match pass.bindings.as_ref() {
            Some((_, slots)) => self.bind_slots(operation, slots),
            // No set was recorded, and that is refused rather than read as an
            // empty one.  The set is also what names the artifact its resources
            // were resolved for -- [`Self::set_bindings`] is where that recipe
            // check is made -- so a draw without one is a draw whose bindings were
            // never checked against the pipeline installed after them.  It is not
            // a shape a frame can produce: a renderer resolves a set per draw
            // whatever the artifact reads, and the bare triangle's is empty and is
            // still recorded.
            None => Err(malformed(
                operation,
                "no binding set is recorded in this pass, and the set is what names the artifact its resources were resolved for",
            )),
        }
    }

    /// Binds one registered set, one logical binding at a time.
    ///
    /// Each binding is applied as soon as it is recorded, for the reason step 5
    /// of [`Self::commit`] gives: the texture domain's request is single-valued,
    /// so applying per binding is the only order in which a set of more than one
    /// arrives intact.
    pub(super) fn bind_slots(
        &mut self,
        operation: &'static str,
        slots: &[BindingSlot],
    ) -> Result<(), GlError> {
        for slot in slots {
            match *slot {
                BindingSlot::Uniform {
                    binding,
                    buffer,
                    range,
                } => {
                    let (offset, size) = uniform_range(operation, range)?;
                    self.machine
                        .bind_uniform_buffer(binding, Some(buffer), offset, size);
                    self.machine
                        .apply_buffers()
                        .map_err(failure::into_gl_error)?;
                }
                BindingSlot::Texture {
                    binding,
                    texture,
                    range,
                } => {
                    // This family has no texture view, so a sampled binding can
                    // only name a whole texture.  Lowering a subresource range
                    // would mean sampling the wrong mips, which is a wrong image
                    // rather than a missing feature.
                    if !matches!(range, TextureRange::Whole) {
                        return Err(unsupported(
                            operation,
                            "this family has no texture view, so a sampled binding names a whole texture",
                        ));
                    }
                    let facts = *self.attachments.get(&texture).ok_or_else(|| {
                        malformed(
                            operation,
                            "a binding names a texture this device did not create",
                        )
                    })?;
                    let target = facts.target()?;
                    self.machine.bind_texture(binding, target, Some(texture));
                    self.machine
                        .apply_textures()
                        .map_err(failure::into_gl_error)?;
                }
            }
        }
        Ok(())
    }

    /// Destroys the objects a pass came to own, reporting the first failure.
    ///
    /// Every object is attempted even after one fails: they are already
    /// unreachable -- the pass that derived them is the last thing that named
    /// them -- so stopping at the first would leave the rest with no name at all.
    /// The failure is reported rather than counted, unlike the domains' own
    /// cleanup, because this is a verb's error path and not a silent pass over a
    /// cache.
    pub(super) fn destroy_owned(&mut self, pass: OpenPass) -> Result<(), GlError> {
        let mut first: Option<GlError> = None;
        for framebuffer in pass.owned_framebuffers {
            if let Err(error) = self.machine.backend().destroy_framebuffer(framebuffer) {
                first.get_or_insert(error);
            }
        }
        for program in pass.owned_programs {
            if let Err(error) = self.machine.backend().destroy_program(program) {
                first.get_or_insert(error);
            }
        }
        match first {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// The byte range a uniform binding point is given, from the range the graph
/// authorized.
///
/// `Whole` becomes `(0, 0)`, which is Layer 1's own spelling of "from the offset
/// through the end of the allocation" and not a range of no bytes.  An explicit
/// range wider than an indexed binding point can express is refused: this
/// family's offset and size are 32-bit, and truncating a 64-bit one would bind a
/// range the graph never authorized.
fn uniform_range(operation: &'static str, range: BufferRange) -> Result<(u32, u32), GlError> {
    let (offset, size) = match range {
        BufferRange::Whole => (0, 0),
        BufferRange::Bytes { offset, size } => (
            u32::try_from(offset).map_err(|_| {
                malformed(
                    operation,
                    "the authorized offset is beyond what an indexed binding point can address",
                )
            })?,
            u32::try_from(size).map_err(|_| {
                malformed(
                    operation,
                    "the authorized size is beyond what an indexed binding point can address",
                )
            })?,
        ),
    };
    Ok((offset, size))
}
