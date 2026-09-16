//! The optional compute domain, and the witness that says whether this adapter
//! has one.
//!
//! Responsibility: answer the contract's four compute verbs -- the two pass
//! brackets, the pipeline selection, the binding set and the dispatch -- through
//! one indirection that is a *type* rather than a run-time flag, and lower the
//! three facts an image unit needs that no other layer holds.
//!
//! Not owned here: the recipes themselves ([`super::super::shader::compute`]),
//! what a compute pipeline *is* ([`super::object`]), and the recording the verbs
//! drive ([`super::encoder`]).
//!
//! # Why the domain is a type parameter
//!
//! Two language facts leave no other shape, and both are worth stating because
//! each rules out an alternative a reader would otherwise expect:
//!
//! - A trait has one impl block per type (E0119), so
//!   `ExecutionBackend for GlCompatibilityDevice<B>` cannot be written once for
//!   every `B` and once more for a `B` that also dispatches: the two would
//!   overlap and neither is more specific.
//! - A method of that impl cannot be *stricter* than the impl's own bounds
//!   (E0276).  So a `where B: GlOptionalComputeBackend` on `dispatch` alone is
//!   rejected, even though only the optional backends could serve it.
//!
//! The optional bound therefore moves onto a type parameter of the adapter --
//! [`GlCompatibilityDevice<B, C>`](super::GlCompatibilityDevice), whose `C`
//! defaults to [`NoCompute`] -- and `C`'s own trait impls are where the bound
//! lives.  The two impls do not overlap, and the reason is the one coherence
//! asks for and this design is built on: `Self` differs.
//!
//! The alternative Layer 2 records as not compiling
//! (`state/machine/mod.rs`, on `apply_storage_buffers`) is a helper trait with a
//! no-op impl for every backend beside a real one for the optional backends:
//! those two blanket impls overlap, and Rust has no specialization to pick.
//!
//! # The two refusals are different facts
//!
//! A verb here can be refused twice over, and the sentences are kept apart
//! because they answer different questions:
//!
//! - **The adapter has no such domain.**  `C = NoCompute` means this machine is
//!   over a backend type that cannot dispatch at all -- the browser provider,
//!   whose shading language has no compute stage.  [`NoCompute::REFUSAL`] is that
//!   fact, and it is a `&'static str` because it is about the backend rather than
//!   about the request, so no caller can change it.
//! - **The context did not prove it.**  `C = WithCompute` means the backend
//!   *could* dispatch, and whether this context may is the discovery snapshot's
//!   answer.  A backend that implements the optional traits on a context whose
//!   capability row was never proved must still refuse *before any side effect*,
//!   which is Layer 2's rule (`state/backend.rs`) and the reason
//!   `compat::capabilities` reports nothing the snapshot did not prove.
//!
//! Both are `GlError::Unsupported`, and the difference is the sentence rather
//! than the variant: in one case the vocabulary is absent, in the other the
//! vocabulary exists and this context is not one it may be used on.

use fluxel_rendergraph::{BufferRange, TextureRange};

use crate::webgl2::api::{
    BufferId, GlCapability, GlDispatchGroups, GlError, GlStorageBufferRange, GlStorageBufferUsage,
    GlStorageImageAccess, GlStorageImageBinding, ProgramId, TextureId,
};
use crate::webgl2::state::{GlOptionalComputeBackend, GlStateBackend, GlStateMachine};

use super::GlCompatibilityDevice;
use super::encoder::{GlEncoder, InstalledPipeline, OpenPass};
use super::failure::{malformed, pass_open, unsupported};
use super::object::{ComputePipeline, Recipe};
use super::pass;

/// What a compute-capable adapter over `B` can do, as a witness type.
///
/// Every method is an associated function rather than a `self` method, because
/// the witness is a *marker* and not a value: what it carries is its own impls,
/// and a value of it would be a field of the adapter that nothing reads.
///
/// Reach matches the adapter's, and it has to: this trait is a bound on a type
/// parameter of a `pub(crate)` struct and the default of that parameter, and
/// Rust reports a narrower reach here as a private bound on a public item.  The
/// module this sits in is private either way, so nothing outside the adapter is
/// reached by it.
pub(crate) trait ComputeDomain<B: GlStateBackend> {
    /// Refuses when this domain could not run `capability` on this context.
    ///
    /// The capability is a parameter rather than a fixed one because the three
    /// this domain asks for are separated by the discovery snapshot even when
    /// they are all present in the trait bound: a context can prove compute and
    /// not storage.  Checking at the verb rather than only at the pass bracket is
    /// what keeps the answer the same whichever verb is reached first.
    fn admit(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        capability: GlCapability,
    ) -> Result<(), GlError>;

    /// Installs the linked program that later dispatches execute.
    fn install_program(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        program: ProgramId,
    ) -> Result<(), GlError>;

    /// Issues one dispatch of `groups` workgroups.
    fn dispatch(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        groups: GlDispatchGroups,
    ) -> Result<(), GlError>;

    /// Binds one storage buffer range at a storage binding point.
    fn bind_storage_buffer(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        binding: u32,
        range: GlStorageBufferRange,
    ) -> Result<(), GlError>;

    /// Binds one storage image at an image unit.
    fn bind_storage_image(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        unit: u32,
        image: GlStorageImageBinding,
    ) -> Result<(), GlError>;
}

/// The witness for an adapter whose backend has no compute command domain.
///
/// It is the adapter's default, and that default is the fail-closed one: a
/// caller that names no witness gets the adapter that refuses every compute verb
/// rather than one that reaches an optional entry point on a backend that has
/// none.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NoCompute;

impl NoCompute {
    /// The reason every verb of this domain gives.
    ///
    /// It names the backend rather than a slice number or a version, because it
    /// is a claim about the adapter's type and not about a schedule: the same
    /// sentence is true of every caller, and no caller can make it false.
    pub(crate) const REFUSAL: &'static str = "this adapter's state machine is over a backend with no compute command domain, so a compute command cannot be recorded for it";
}

impl<B: GlStateBackend> ComputeDomain<B> for NoCompute {
    fn admit(
        _machine: &mut GlStateMachine<B>,
        operation: &'static str,
        _capability: GlCapability,
    ) -> Result<(), GlError> {
        Err(unsupported(operation, Self::REFUSAL))
    }

    /// Unreachable, and still a refusal rather than a panic.
    ///
    /// `admit` is called at the pass bracket before this can be reached, so no
    /// frame gets here -- but the alternative to a refusal is an `unreachable!()`
    /// or a silent `Ok(())`, and both would turn "this adapter has no compute
    /// domain" into something a future edit could quietly change.
    fn install_program(
        _machine: &mut GlStateMachine<B>,
        operation: &'static str,
        _program: ProgramId,
    ) -> Result<(), GlError> {
        Err(unsupported(operation, Self::REFUSAL))
    }

    /// Unreachable, on [`Self::install_program`]'s terms.
    fn dispatch(
        _machine: &mut GlStateMachine<B>,
        operation: &'static str,
        _groups: GlDispatchGroups,
    ) -> Result<(), GlError> {
        Err(unsupported(operation, Self::REFUSAL))
    }

    /// Unreachable, on [`Self::install_program`]'s terms.
    fn bind_storage_buffer(
        _machine: &mut GlStateMachine<B>,
        operation: &'static str,
        _binding: u32,
        _range: GlStorageBufferRange,
    ) -> Result<(), GlError> {
        Err(unsupported(operation, Self::REFUSAL))
    }

    /// Unreachable, on [`Self::install_program`]'s terms.
    fn bind_storage_image(
        _machine: &mut GlStateMachine<B>,
        operation: &'static str,
        _unit: u32,
        _image: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        Err(unsupported(operation, Self::REFUSAL))
    }
}

/// The witness for an adapter whose backend has the optional compute domains.
///
/// Implemented for a backend that declares all three of the optional traits and
/// for no other type, which is what makes "this adapter can dispatch" a fact
/// about the adapter's type rather than a flag a caller could set.  The context
/// it actually runs on is still the snapshot's question, and that is what
/// [`ComputeDomain::admit`] asks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WithCompute;

impl<B: GlOptionalComputeBackend> ComputeDomain<B> for WithCompute {
    fn admit(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        capability: GlCapability,
    ) -> Result<(), GlError> {
        if machine
            .backend()
            .discovery()
            .capabilities()
            .supports(capability)
        {
            return Ok(());
        }
        Err(unsupported(operation, unproved(capability)))
    }

    /// Installs the linked program as the one dispatch work executes.
    ///
    /// The operation name is unused here, and that is not an oversight: this is
    /// the one method of the domain whose Layer 1 call takes no operation, because
    /// the provider's own refusal already names its own verb (`set-compute-program`
    /// on both executable providers) and a second name for it could only disagree
    /// with the first.  The parameter stays on the trait because the *refusing*
    /// impl has nothing else to name the adapter's verb with.
    fn install_program(
        machine: &mut GlStateMachine<B>,
        _operation: &'static str,
        program: ProgramId,
    ) -> Result<(), GlError> {
        machine.backend().set_compute_program(program)
    }

    /// Installs nothing: the program is already installed, and Layer 1's own
    /// documentation says a dispatch rejects while none is.
    ///
    /// The lifecycle preflight is repeated here because this is the one command
    /// of the domain that Layer 2 does not mediate -- the compute program is
    /// deliberately not mirrored there (`state/knowledge.rs`), so the two calls
    /// either side of it go straight to the backend and neither domain's
    /// `assert_ready` runs for them.
    fn dispatch(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        groups: GlDispatchGroups,
    ) -> Result<(), GlError> {
        machine.backend().assert_ready(operation)?;
        machine.backend().dispatch(groups)
    }

    fn bind_storage_buffer(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        binding: u32,
        range: GlStorageBufferRange,
    ) -> Result<(), GlError> {
        Self::admit(machine, operation, GlCapability::StorageBuffer)?;
        machine.bind_storage_buffer(binding, range);
        machine
            .apply_storage_buffers()
            .map_err(super::failure::into_gl_error)
    }

    fn bind_storage_image(
        machine: &mut GlStateMachine<B>,
        operation: &'static str,
        unit: u32,
        image: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        Self::admit(machine, operation, GlCapability::StorageImage)?;
        machine.bind_storage_image(unit, image);
        machine
            .apply_storage_images()
            .map_err(super::failure::into_gl_error)
    }
}

/// The reason a verb refuses on a context whose snapshot did not prove what it
/// needs.
///
/// The three arms are the capabilities this domain asks for.  A fourth arriving
/// here would be a change to this file, and the fallback sentence is still true
/// of it -- the snapshot did not prove what the verb needs -- which is why it is
/// a reason rather than a panic.
fn unproved(capability: GlCapability) -> &'static str {
    match capability {
        GlCapability::Compute => {
            "the context's discovery snapshot did not prove the compute capability, so a dispatch cannot be issued on it"
        }
        GlCapability::StorageBuffer => {
            "the context's discovery snapshot did not prove the storage-buffer capability, so no shader storage block can be bound on it"
        }
        GlCapability::StorageImage => {
            "the context's discovery snapshot did not prove the storage-image capability, so no image unit can be bound on it"
        }
        _ => {
            "the context's discovery snapshot did not prove the capability this command domain needs"
        }
    }
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// Opens the compute pass this encoder will record a dispatch in.
    ///
    /// The pass is recorded and issued at the dispatch, on [`super::encoder`]'s
    /// terms for a raster pass and for a smaller reason here: this family's
    /// compute commands are immediate too, and what the record buys is that a
    /// binding set can be checked against the recipe *installed at the dispatch*
    /// rather than against whichever one was installed when the set was.
    ///
    /// Nothing is opened in Layer 1, and that is not an omission: this family's
    /// pass boundary is a framebuffer's, so `begin_pass` and `end_pass` are about
    /// attachments and a dispatch has none.  The refusal on an unproved profile
    /// happens here rather than at the dispatch so that a frame learns it at the
    /// bracket the executor opened rather than in the middle of its work.
    ///
    /// The admission is asked before the bracket is even looked at, so that an
    /// adapter with no compute domain is told that rather than told about a pass
    /// it could never have opened -- the same order, for the same reason, as every
    /// other verb of this domain.
    pub(super) fn open_compute_pass(
        &mut self,
        encoder: &mut GlEncoder,
        operation: &'static str,
    ) -> Result<(), GlError> {
        self.refresh();
        C::admit(&mut self.machine, operation, GlCapability::Compute)?;
        if encoder.pass.is_some() {
            return Err(pass_open(operation));
        }
        encoder.pass = Some(OpenPass::compute());
        Ok(())
    }

    /// Closes the compute pass this encoder has open.
    ///
    /// Unlike `end_raster` there is no Layer 1 boundary to close, because this
    /// family's pass boundary is a framebuffer's and a dispatch has none -- so the
    /// whole of this verb is the ownership bookkeeping.  Which pass is open, and
    /// whether it is the right kind, is [`GlEncoder::take_compute`]'s question,
    /// and it puts a raster pass back rather than dropping it: the frame still has
    /// to close that one, and taking it here would leave `end_raster` with nothing
    /// to find.
    ///
    /// The admission is asked first, as it is at every verb of this domain, and
    /// the order is load-bearing rather than stylistic: an adapter with no compute
    /// domain can never have a compute pass open, so a frame that closed one would
    /// otherwise be told "no compute pass is open" -- a diagnosis about the
    /// bracket for a mistake that is about the adapter.
    pub(super) fn close_compute_pass(
        &mut self,
        encoder: &mut GlEncoder,
        operation: &'static str,
    ) -> Result<(), GlError> {
        self.refresh();
        C::admit(&mut self.machine, operation, GlCapability::Compute)?;
        let pass = encoder.take_compute(operation)?;
        // Nothing is destroyed when the context was replaced while the pass was
        // open, for `end_raster`'s reason: those identities belong to an epoch the
        // backend no longer accepts, and the context's own teardown released them.
        if encoder.context == self.machine.backend().context_stamp() {
            self.destroy_owned(pass)
        } else {
            Ok(())
        }
    }

    /// Records the compute recipe this pass will dispatch with.
    pub(super) fn record_compute_pipeline(
        &mut self,
        encoder: &mut GlEncoder,
        operation: &'static str,
        pipeline: &ComputePipeline,
    ) -> Result<(), GlError> {
        C::admit(&mut self.machine, operation, GlCapability::Compute)?;
        let pass = encoder.compute(operation)?;
        pass.pipeline = Some(InstalledPipeline::Compute(pipeline.clone()));
        Ok(())
    }

    /// Links the recorded recipe's program, binds its set, and dispatches.
    ///
    /// This is the compute half's commit point and it is the dispatch for the
    /// same reason a draw is the raster half's: the program has to be linked
    /// before anything is bound against it, and a link invalidates the mirror's
    /// installed pipeline -- so the order inside is program, then bindings, then
    /// the command, which is also the order Layer 1 requires.
    ///
    /// The binding set is checked here as well as at `set_bindings`, and the two
    /// are not the same check: this one catches a pipeline installed *after* the
    /// set, which would otherwise bind resources at the wrong numbers.
    ///
    /// The admission is asked first, as at every verb of this domain, and for the
    /// reason [`Self::close_compute_pass`] gives: an adapter with no compute
    /// domain must be told that, and not be told about a pass it could never have
    /// opened.
    pub(super) fn issue_dispatch(
        &mut self,
        encoder: &mut GlEncoder,
        operation: &'static str,
        groups: [u32; 3],
    ) -> Result<(), GlError> {
        self.refresh();
        C::admit(&mut self.machine, operation, GlCapability::Compute)?;
        let pass = encoder.compute(operation)?;
        let Some(InstalledPipeline::Compute(pipeline)) = pass.pipeline.clone() else {
            return Err(malformed(
                operation,
                "no compute pipeline is installed in this pass, so there is nothing to dispatch with",
            ));
        };
        let Some((resolved_for, slots)) = pass.bindings.as_ref() else {
            return Err(malformed(
                operation,
                "no binding set is recorded in this pass, and the set is what names the artifact its resources were resolved for",
            ));
        };
        if *resolved_for != Recipe::Compute(pipeline.kernel()) {
            return Err(malformed(
                operation,
                "the binding set in this pass was resolved for a different artifact than the one installed at the dispatch",
            ));
        }
        let (program, _reflection, owned) = self
            .machine
            .program_for(pipeline.descriptor())
            .map_err(super::failure::into_gl_error)?;
        // The install is re-asserted per dispatch rather than once per pass,
        // because this family has one current program shared with the pipeline
        // domain and any link between two dispatches moves that slot.
        C::install_program(&mut self.machine, operation, program)?;
        self.bind_slots(operation, slots)?;
        if owned {
            pass.owned_programs.push(program);
        }
        C::dispatch(&mut self.machine, operation, GlDispatchGroups(groups))
    }
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// One storage image, lowered onto the image-unit binding Layer 1 takes.
    ///
    /// This is why the slot holds a range and an access rather than a ready
    /// [`GlStorageImageBinding`]: three of the seven facts an image unit needs
    /// come from the device's own record of what it created, and that record is a
    /// field of the adapter -- a [`BindingSlot`](super::object::BindingSlot) is
    /// built from the frame's resolution and cannot reach it.
    ///
    /// A texture this device did not create is refused on the same terms the
    /// sampled path already refuses one: the id would name another context's
    /// object, and this adapter holds no shape or format for it.
    ///
    /// An image unit names **one** mip level, so a range selecting several has no
    /// lowering at all and is refused by name: binding the base level and
    /// dispatching anyway would run the kernel over a different subresource than
    /// the graph authorized, which is a wrong image rather than a missing feature.
    pub(super) fn storage_image(
        &self,
        operation: &'static str,
        texture: TextureId,
        range: TextureRange,
        access: GlStorageImageAccess,
    ) -> Result<GlStorageImageBinding, GlError> {
        let Some(facts) = self.attachments.get(&texture).copied() else {
            return Err(malformed(
                operation,
                "a binding names a texture this device did not create",
            ));
        };
        let level = match range {
            TextureRange::Whole => 0,
            TextureRange::Subresources {
                base_mip_level,
                mip_level_count: 1,
                ..
            } => base_mip_level,
            TextureRange::Subresources { .. } => {
                return Err(unsupported(
                    operation,
                    "an image unit names one mip level, so a range selecting several has no lowering in this family",
                ));
            }
        };
        let (layered, layer) = image_layers(operation, facts, range)?;
        Ok(GlStorageImageBinding {
            texture,
            level,
            sample_count: facts.sample_count(),
            layered,
            layer,
            format: facts.format(),
            access,
        })
    }

    /// One storage buffer range, lowered onto the binding point Layer 1 takes.
    ///
    /// `Whole` becomes the whole allocation, and that is the one place this
    /// differs from the uniform path next door: an indexed uniform binding point
    /// spells "to the end" as an offset of zero with a size of zero, while a
    /// storage binding is validated against the real allocation and so has to be
    /// given the real size.  That is why this adapter records each buffer's extent
    /// as it creates it, the way it records each texture's shape.
    ///
    /// A buffer this device did not create is refused on
    /// [`Self::storage_image`]'s terms, and for the same reason: without the
    /// allocation there is no size to spell `Whole` with.
    ///
    /// The range stays 64-bit where the uniform one is narrowed to 32, because a
    /// storage binding point addresses bytes and not a binding point's window:
    /// truncating here would bind a range the graph never authorized.
    pub(super) fn storage_range(
        &self,
        operation: &'static str,
        buffer: BufferId,
        range: BufferRange,
        usage: GlStorageBufferUsage,
    ) -> Result<GlStorageBufferRange, GlError> {
        let Some(allocated) = self.buffers.get(&buffer).copied() else {
            return Err(malformed(
                operation,
                "a binding names a buffer this device did not create",
            ));
        };
        let (offset, size) = match range {
            BufferRange::Whole => (0, allocated),
            BufferRange::Bytes { offset, size } => (offset, size),
        };
        Ok(GlStorageBufferRange {
            buffer,
            offset,
            size,
            usage,
        })
    }
}

/// Whether the image unit covers every layer, and which layer it names when it
/// does not.
///
/// The pair is not free: `GlStorageImageBinding` requires the layer for a
/// non-layered view and forbids it for a layered one, so one of the two is always
/// a fact this function had to decide rather than a field it copied.
///
/// The whole-allocation case asks the attachment, because the answer is a
/// property of the texture and not of the range: a texture whose dimension can
/// address its layers as a set is bound layerless over all of them, and one with
/// a single layer still has to name it.
fn image_layers(
    operation: &'static str,
    facts: pass::Attachment,
    range: TextureRange,
) -> Result<(bool, Option<u32>), GlError> {
    match range {
        TextureRange::Whole if facts.layered() => Ok((true, None)),
        TextureRange::Whole => Ok((false, Some(0))),
        TextureRange::Subresources {
            base_array_layer,
            array_layer_count,
            ..
        } => {
            if array_layer_count == facts.layers() && base_array_layer == 0 && facts.layered() {
                return Ok((true, None));
            }
            if array_layer_count == 1 {
                return Ok((false, Some(base_array_layer)));
            }
            Err(unsupported(
                operation,
                "an image unit names one layer or every layer, so a range selecting another count has no lowering in this family",
            ))
        }
    }
}
