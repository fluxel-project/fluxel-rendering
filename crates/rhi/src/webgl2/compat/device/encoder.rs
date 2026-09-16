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
use super::compute::ComputeDomain;
use super::failure::{self, malformed, no_compute_pass, no_pass, unsupported};
use super::object::{self, BindingSlot, Recipe};
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

/// What kind of pass this is, and the state only that kind has.
///
/// The raster state is a *variant* rather than a set of fields that happen to be
/// empty for a compute pass, and that is the point: a compute pass has no
/// attachment, no vertex buffers, no viewport and no framebuffer of its own, and
/// a struct holding them anyway would make "which of these mean anything" a
/// question every reader answers from the pass kind rather than from the type.
pub(super) enum PassShape {
    /// A raster pass, over the attachment it renders into.
    Raster(RasterShape),
    /// A compute pass, which renders into nothing: a dispatch's storage bindings
    /// are its inputs and its outputs, and Layer 1 declares no pass boundary for
    /// this family's compute commands at all.
    Compute,
}

/// The per-pass state only a raster pass has.
///
/// Every field is per-pass and cleared where the pass ends, which is the same
/// lifetime Layer 1's own pass scope has: a binding recorded in one pass is not
/// in force in the next, and neither is a viewport.
pub(super) struct RasterShape {
    /// The colour attachment's shape, which the viewport defaults to and the
    /// pipeline's sample count is taken from.
    pub(super) target: pass::Attachment,
    /// The vertex buffers the frame bound, by slot.
    pub(super) vertex: Vec<GlVertexBufferBinding>,
    /// The index buffer the frame bound.
    pub(super) index: Option<GlIndexBinding>,
    /// The viewport, defaulting to the whole attachment.
    pub(super) viewport: GlViewport,
    /// The scissor, absent until the frame sets one.
    pub(super) scissor: Option<GlScissorRect>,
    /// Framebuffers whose ownership Layer 2 handed to this pass.  Destroyed when
    /// the pass ends, because a caller-owned object is one no cache will ever
    /// free.  Programs are deliberately not here: both kinds of pass link one and
    /// can come to own it, so they live on [`OpenPass`] instead.
    pub(super) owned_framebuffers: Vec<FramebufferId>,
}

impl RasterShape {
    /// A fresh pass over `target`, with the two defaults a pass starts from.
    pub(super) fn new(target: pass::Attachment) -> Self {
        Self {
            target,
            vertex: Vec::new(),
            index: None,
            // The attachment's own extent, so that a pass whose frame sets no
            // viewport renders into the whole of what it attached rather than
            // into whatever the previous pass left selected.
            viewport: pass::whole_extent(target),
            scissor: None,
            owned_framebuffers: Vec::new(),
        }
    }
}

/// What the frame has recorded inside the pass this encoder has open.
pub(super) struct OpenPass {
    /// Which kind of pass this is, with the state only that kind has.
    pub(super) shape: PassShape,
    /// The recipe the frame selected, if it has selected one.
    ///
    /// One slot holds whichever kind this pass is, so the variant says which
    /// family installed it without a second field that would have to be kept in
    /// step with [`Self::shape`].
    pub(super) pipeline: Option<InstalledPipeline>,
    /// The binding set the frame resolved, with the artifact it was resolved
    /// *for*: the recipe is kept beside the slots because the slots alone cannot
    /// answer whether they belong to the recipe installed at the commit, and a
    /// pipeline installed after a set is the one way the two can disagree.
    /// Replaced rather than accumulated: a fixed artifact declares one set, so a
    /// second one is a different recipe's.
    pub(super) bindings: Option<(Recipe, Vec<BindingSlot>)>,
    /// Programs whose ownership Layer 2 handed to this pass.  Destroyed when the
    /// pass ends, because a caller-owned object is one no cache will ever free.
    /// Shared by both kinds because both link a program at their commit point.
    pub(super) owned_programs: Vec<ProgramId>,
}

impl OpenPass {
    /// A pass that renders into `target`.
    pub(super) fn raster(target: pass::Attachment) -> Self {
        Self {
            shape: PassShape::Raster(RasterShape::new(target)),
            pipeline: None,
            bindings: None,
            owned_programs: Vec::new(),
        }
    }

    /// A pass that dispatches and renders into nothing.
    pub(super) fn compute() -> Self {
        Self {
            shape: PassShape::Compute,
            pipeline: None,
            bindings: None,
            owned_programs: Vec::new(),
        }
    }

    /// Whether this is a compute pass.
    pub(super) fn is_compute(&self) -> bool {
        matches!(self.shape, PassShape::Compute)
    }

    /// The raster half of this pass, or the refusal that it is not a raster pass.
    ///
    /// Every raster-only verb reads the pass through here, so that a verb reached
    /// inside a compute pass reports the same mistake the same way.  It is a
    /// refusal and not a silent no-op because a viewport recorded in a compute
    /// pass is a frame that believes it is rendering: the state would be
    /// recorded, nothing would read it, and the draw that later wanted it would
    /// find whatever the compute pass left rather than what it set.
    ///
    /// Named for what it hands back rather than `raster`, because that name is the
    /// pass *constructor* beside it: `OpenPass::raster(target)` and a narrowing
    /// accessor cannot share a name, and the constructor is the one both families
    /// read as a pair (`raster` and `compute`).
    pub(super) fn raster_shape(
        &mut self,
        operation: &'static str,
    ) -> Result<&mut RasterShape, GlError> {
        match &mut self.shape {
            PassShape::Raster(shape) => Ok(shape),
            PassShape::Compute => Err(malformed(
                operation,
                "a compute pass is open on this encoder, and a compute pass records no rasterization state",
            )),
        }
    }
}

/// The recipe a pass has installed, in whichever family it belongs to.
///
/// One enum rather than two optional fields, so that "a pass has one pipeline" is
/// a fact of the type and a compute recipe cannot sit beside a raster one.  It is
/// `Clone` because both commit points take it out before driving the machine: a
/// link can invalidate the mirror's installed pipeline, so the recipe has to be
/// held across calls that need the pass mutably.
#[derive(Clone, Debug)]
pub(super) enum InstalledPipeline {
    /// One of the ten closed raster artifacts, lowered.
    Raster(object::RasterPipeline),
    /// One of the five closed compute artifacts, lowered.
    Compute(object::ComputePipeline),
}

impl InstalledPipeline {
    /// Which fixed artifact this is.
    ///
    /// The one question both kinds answer the same way, and the reason
    /// `set_bindings` is a shared verb rather than one per family: what a binding
    /// set is checked against is the artifact it was resolved for, and that is a
    /// [`Recipe`] whichever family installed it.
    pub(super) fn recipe(&self) -> Recipe {
        match self {
            Self::Raster(pipeline) => Recipe::Raster(pipeline.kernel()),
            Self::Compute(pipeline) => Recipe::Compute(pipeline.kernel()),
        }
    }
}

impl GlEncoder {
    /// The recipe this pass has selected, refusing when it has none.
    ///
    /// Asked before the commit rather than after it, so that a draw issued
    /// through the wrong verb of the pair -- a non-indexed draw of an indexed
    /// artifact, say -- is refused before anything is installed for it.
    pub(super) fn kernel(&self, operation: &'static str) -> Result<RasterKernel, GlError> {
        match self.pass.as_ref().and_then(|pass| pass.pipeline.as_ref()) {
            Some(InstalledPipeline::Raster(pipeline)) => Ok(pipeline.kernel()),
            _ => Err(malformed(
                operation,
                "no raster pipeline is installed in this pass, so there is nothing to draw with",
            )),
        }
    }

    /// The pass an encoder has open, or the refusal that it has none.
    ///
    /// The shared verbs read it through here -- a binding set is recorded in
    /// either kind of pass -- while the raster-only ones go through the device's
    /// `raster_pass`, which narrows this to the kind they need.
    pub(super) fn open(&mut self, operation: &'static str) -> Result<&mut OpenPass, GlError> {
        self.pass.as_mut().ok_or_else(|| no_pass(operation))
    }

    /// The compute pass an encoder has open, or the refusal that it is not one.
    ///
    /// The two arms are separate sentences rather than one, because they are two
    /// different mistakes: a frame that dispatched outside any pass and a frame
    /// that dispatched inside a raster pass both need to be told what they did,
    /// and "no compute pass is open" would be false about the second.
    pub(super) fn compute(&mut self, operation: &'static str) -> Result<&mut OpenPass, GlError> {
        match self.pass.as_mut() {
            Some(pass) if pass.is_compute() => Ok(pass),
            Some(_) => Err(malformed(
                operation,
                "a raster pass is open on this encoder, and a raster pass records no compute command",
            )),
            None => Err(malformed(
                operation,
                "no compute pass is open on this encoder, so a compute command has no pass to be recorded in",
            )),
        }
    }

    /// The compute pass an encoder has open, taken out for its close.
    ///
    /// `end_compute` reads the pass out rather than through [`Self::compute`],
    /// because a refusal here has to put the pass *back*: a raster pass found at
    /// this verb belongs to a frame that still has to close it with `end_raster`,
    /// and taking it would leave that close with nothing to find.  So this is the
    /// one reader that both narrows to the compute kind and restores what it
    /// found, which is why the close does not repeat the check.
    ///
    /// The two refusals are separate sentences because they are two different
    /// mistakes: a close with no pass open, and a close of the wrong kind.  The
    /// latter names the verb that does close it, because the frame's next step is
    /// the whole of what it needs to know.
    pub(super) fn take_compute(&mut self, operation: &'static str) -> Result<OpenPass, GlError> {
        match self.pass.take() {
            Some(pass) if pass.is_compute() => Ok(pass),
            Some(pass) => {
                self.pass = Some(pass);
                Err(malformed(
                    operation,
                    "a raster pass is open on this encoder, and a raster pass is closed by end-raster",
                ))
            }
            None => Err(no_compute_pass(operation)),
        }
    }

    /// The raster pass an encoder has open, taken out for its close.
    ///
    /// [`Self::take_compute`]'s mirror, on its terms and for its reason: the
    /// close reads the pass out because a refusal has to put back what it found,
    /// and a compute pass found at `end_raster` belongs to a frame that still has
    /// to close it with `end_compute`.  The wrong-kind message names that verb,
    /// because the frame's next step is the whole of what it needs to know.
    pub(super) fn take_raster(&mut self, operation: &'static str) -> Result<OpenPass, GlError> {
        match self.pass.take() {
            Some(pass) if !pass.is_compute() => Ok(pass),
            Some(pass) => {
                self.pass = Some(pass);
                Err(malformed(
                    operation,
                    "a compute pass is open on this encoder, and a compute pass is closed by end-compute",
                ))
            }
            None => Err(no_pass(operation)),
        }
    }
}

/// One finished recording, ready to submit.
pub(crate) struct GlCommandBuffer {
    pub(super) context: ContextStamp,
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
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
        encoder.open(operation)
    }

    /// The raster half of the pass an encoder has open.
    ///
    /// The raster-only verbs read through here, so that each of them is one line
    /// and none of them decides for itself what to do when the open pass is a
    /// compute pass.  A verb that recorded rasterization state into one would
    /// record it where nothing reads it, and the draw that later wanted it would
    /// find whatever the previous pass left.
    pub(super) fn raster_pass<'e>(
        &mut self,
        encoder: &'e mut GlEncoder,
        operation: &'static str,
    ) -> Result<&'e mut RasterShape, GlError> {
        encoder.open(operation)?.raster_shape(operation)
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
        let recipe = match pass.pipeline.clone() {
            Some(InstalledPipeline::Raster(pipeline)) => pipeline,
            _ => {
                return Err(malformed(
                    operation,
                    "no raster pipeline is installed in this pass, so there is nothing to draw with",
                ));
            }
        };
        if let Some((resolved_for, _)) = pass.bindings.as_ref() {
            // A pipeline installed *after* the set is the one way the two can
            // disagree: `set_bindings` checks the set against the pipeline that
            // was installed when it was called, and this is the check that holds
            // for the pipeline installed now.
            if *resolved_for != Recipe::Raster(recipe.kernel()) {
                return Err(malformed(
                    operation,
                    "the binding set in this pass was resolved for a different artifact than the one installed at the draw",
                ));
            }
        }
        // Everything the raster shape is asked for is read here, in one scope, so
        // that the rest of the commit holds no borrow of the pass beyond the two
        // fields it writes.  The three scalars are copied rather than borrowed
        // because they are needed after the machine has been driven, and a borrow
        // of a field cannot outlive a call that takes the pass mutably.
        let (input, viewport, scissor, sample_count) = {
            let shape = pass.raster_shape(operation)?;
            (
                pass::vertex_input(&recipe, &shape.vertex, shape.index, operation)?,
                shape.viewport,
                shape.scissor,
                shape.target.sample_count(),
            )
        };

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
            state: pass::raster_state(recipe.kernel(), viewport, scissor, sample_count),
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
    ///
    /// One loop serves both families, and the two storage arms are why it can.
    /// They lower their range through [`Self::storage_range`] and
    /// [`Self::storage_image`], which read the device's own record of what it
    /// created, and then reach the machine through the witness -- which is what
    /// makes this method `C`-generic.  The alternative, a second loop beside this
    /// one for compute sets, would have to repeat the texture arm and its three
    /// calls, and the compute family really does use it: `TexturePackRgba8` reads
    /// a sampled texture through exactly this path.  Two copies of that arm would
    /// be two places for its refusal and its fold to drift.
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
                // The two storage arms keep the range and the access rather than
                // lowering them here, because the lowering needs the device's
                // record of the object and this loop is a method on the device
                // but not the object.  Both refusals they can raise name the
                // operation, so a frame is told which verb asked.
                BindingSlot::StorageBuffer {
                    binding,
                    buffer,
                    range,
                    usage,
                } => {
                    let range = self.storage_range(operation, buffer, range, usage)?;
                    C::bind_storage_buffer(&mut self.machine, operation, binding, range)?;
                }
                BindingSlot::StorageImage {
                    binding,
                    texture,
                    range,
                    access,
                } => {
                    let image = self.storage_image(operation, texture, range, access)?;
                    C::bind_storage_image(&mut self.machine, operation, binding, image)?;
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
        // A compute pass owns no framebuffer, which is a fact of its shape rather
        // than a list that happens to be empty -- so the destructuring is where
        // that is read, and there is no second field to keep in step.
        if let PassShape::Raster(shape) = pass.shape {
            for framebuffer in shape.owned_framebuffers {
                if let Err(error) = self.machine.backend().destroy_framebuffer(framebuffer) {
                    first.get_or_insert(error);
                }
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
