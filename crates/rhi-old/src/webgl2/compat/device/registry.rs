//! The registry the common executor resolves this adapter's objects through.
//!
//! Responsibility: answer the contract's three `RenderObjectProvider` questions
//! from what a renderer registered before the frame, and hold the two facts that
//! answer them -- which recipe a [`RasterPipelineId`] or a [`ComputePipelineId`]
//! is, and which recipe a [`BindingSetId`] belongs to.  It resolves nothing
//! native: a pipeline is lowered at registration and a binding set is validated
//! against the frame's own resources, so the only work a frame does here is a
//! lookup and a check.
//!
//! Not owned here: the frame's resources (the executor's
//! `FrameResourceProvider`), the machine that links and binds (that is
//! [`super`]'s, at the verbs that install), and the graph's own validation that
//! a pass declared every access it binds.
//!
//! # Why a binding set holds a recipe and not a pipeline
//!
//! A set id means *which bindings a layout declares*, and that arrangement is a
//! property of the artifact -- so the registry holds the artifact's identity and
//! not a lowered pipeline.  The arrangement itself is read from the lowering
//! ([`super::super::shader::compute_layout`] for the compute family), because a
//! second table here would be a third statement about the same kernel beside the
//! body and the layout.
//!
//! # Why this is a value and not a field of the adapter
//!
//! The contract hands the executor a *provider* beside the backend
//! (`FrameExecutor::execute(&self, graph, execution, resources, objects)`), and
//! the two are separate values for a reason this adapter would otherwise have to
//! work around: the recorder calls `objects` while it holds `&mut` on the
//! backend, so a provider that reached back into the adapter would be a second
//! borrow of the machine at the moment it is already borrowed.
//!
//! That is also what keeps F1's self-deadlock finding satisfied here.  This
//! registry holds no lock, no `RefCell` and no reference to the adapter: the
//! renderer fills it at its own `&mut` entry point, before any frame, and a
//! frame only reads it.  The `Rc<ReleaseQueue>` it shares with the adapter is a
//! queue of *released* objects and not of live ones, and nothing on the
//! resolution path pushes to it.
//!
//! # Failures are the contract's own
//!
//! Every refusal here is `RecordingErrorKind::IncompatibleBindingRecipe`, which
//! the contract describes as "renderer/RHI binding metadata is incompatible with
//! the selected pipeline" -- exactly the fact being reported, and the reason
//! that kind exists rather than a backend-specific one.  The diagnostic context
//! carries no pass and no resource: a provider is not inside the recorder and
//! does not know which pass asked, and inventing one would be a worse diagnostic
//! than an empty list.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use fluxel_rendergraph::{
    BindingSetId, BoundBindings, BoundComputePipeline, BoundRasterPipeline, ComputePipelineId,
    DeviceIdentity, DiagnosticContext, RasterPipelineId, RecordingError, RecordingErrorKind,
    RenderObjectProvider, ResolvedBindingResource,
};

use crate::resource::{ComputeKernel, RasterKernel};
use crate::webgl2::api::{BufferId, GlError, GlFamilyProfile, TextureId};
use crate::webgl2::state::GlStateBackend;

use super::super::shader;
use super::GlCompatibilityDevice;
use super::compute::{ComputeDomain, NoCompute};
use super::object::{Bindings, ComputePipeline, RasterPipeline, Recipe};
use super::retention::{GlRetentionLease, ReleaseQueue};

/// The objects one adapter's renderer has registered.
///
/// Not `Sync` and not shared: it is filled on the context's owner thread and
/// read by frames recorded there, which is the same single-threaded reach the
/// rest of this adapter has.
///
/// # Why both of the adapter's parameters are here
///
/// This type's only purpose is to implement a trait parameterized by the adapter
/// it answers for, so "which adapter this answers for" is its identity and it
/// carries all of that identity rather than part of it.  Both halves are needed,
/// and the second is needed for a reason that is not visible in the trait bound:
/// **none of the three answers mentions the witness in its return type.**  So a
/// registry that left `C` free would leave it *undetermined at every call* --
/// `objects.raster_pipeline(id)` has nothing to infer it from -- and a renderer
/// holding the registry beside its adapter would have to name the adapter type at
/// each question.  Carrying it makes the value's own type decide, which is what
/// the caller already knows.
///
/// That is also why the register methods below are *inherent* and not gated on
/// the witness: a compute pipeline can be registered for an adapter that will
/// refuse to dispatch it, which is exactly what lets a frame be built once and
/// run against a provider that turns out not to support it.  The refusal belongs
/// at the verb that records the command, where the ledger can name it, and not at
/// the call that described the artifact.
///
/// Both are `fn() -> _` phantoms because nothing is ever built from either: the
/// registry must not become a place a second adapter could be reached through.
pub(crate) struct GlObjectRegistry<B: GlStateBackend, C: ComputeDomain<B> = NoCompute> {
    device: DeviceIdentity,
    profile: GlFamilyProfile,
    releases: Rc<ReleaseQueue>,
    raster: HashMap<RasterPipelineId, RasterPipeline>,
    compute: HashMap<ComputePipelineId, ComputePipeline>,
    bindings: HashMap<BindingSetId, Recipe>,
    backend: PhantomData<fn() -> (B, C)>,
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlObjectRegistry<B, C> {
    /// A registry that lowers for `profile` and retains through `releases`.
    ///
    /// Both facts come from the adapter that minted it rather than from the
    /// frames it will serve, and both are captured rather than read per call for
    /// the reason [`super::GlCompatibilityDevice::object_registry`] states: a
    /// provider is reached through `&self` while the recorder holds `&mut` on the
    /// adapter, so it cannot ask the adapter anything.
    pub(super) fn new(
        device: DeviceIdentity,
        profile: GlFamilyProfile,
        releases: Rc<ReleaseQueue>,
    ) -> Self {
        Self {
            device,
            profile,
            releases,
            raster: HashMap::new(),
            compute: HashMap::new(),
            bindings: HashMap::new(),
            backend: PhantomData,
        }
    }

    /// Registers one raster recipe under an identity the graph names it by.
    ///
    /// Registration lowers the artifact, and a profile this family cannot lower
    /// for is refused here rather than at the first draw -- the same fail-closed
    /// ordering the rest of this adapter follows, where a fact that makes every
    /// later call wrong is reported at the call that established it.
    pub(crate) fn register_raster_pipeline(
        &mut self,
        id: RasterPipelineId,
        kernel: RasterKernel,
    ) -> Result<(), GlError> {
        let pipeline = RasterPipeline::new(kernel, self.profile).map_err(|_unsupported| {
            GlError::Unsupported {
                operation: "register-raster-pipeline",
                reason: shader::UnsupportedProfile::REASON,
            }
        })?;
        self.raster.insert(id, pipeline);
        Ok(())
    }

    /// Registers one compute recipe under an identity the graph names it by.
    ///
    /// On [`Self::register_raster_pipeline`]'s terms, with one difference that is
    /// the whole of why the compute family needed its own unit error: the refusal
    /// here is not "this profile has no dialect" but "this profile's shading
    /// language has no compute stage", because the dialect rule answers for
    /// profiles the compute lowering must still narrow away.  Registration
    /// succeeds on an adapter whose witness has no compute domain -- see the
    /// type's documentation for why that is deliberate rather than an oversight.
    pub(crate) fn register_compute_pipeline(
        &mut self,
        id: ComputePipelineId,
        kernel: ComputeKernel,
    ) -> Result<(), GlError> {
        let pipeline = ComputePipeline::new(kernel, self.profile).map_err(|_unsupported| {
            GlError::Unsupported {
                operation: "register-compute-pipeline",
                reason: shader::UnsupportedComputeProfile::REASON,
            }
        })?;
        self.compute.insert(id, pipeline);
        Ok(())
    }

    /// Records which artifact a binding set id belongs to.
    ///
    /// The recipe and not a resolved set: what a set id means is which bindings
    /// a pipeline layout declares, and the resources those bindings name arrive
    /// with the frame.  A set that was never registered is what `bindings`
    /// refuses; an id registered twice keeps the last recipe, which is the same
    /// last-write-wins a re-registration of a pipeline has.
    ///
    /// The kind is carried rather than inferred, and a set registered with the
    /// wrong one is caught where the two meet: a compute set in a raster pass is
    /// refused at `set_bindings`, and the reverse at the dispatch.  Inferring it
    /// from the two maps here would be a check made against whichever pipeline
    /// happened to be registered, which is not the pipeline the frame will
    /// install.
    pub(crate) fn register_bindings(&mut self, id: BindingSetId, recipe: Recipe) {
        self.bindings.insert(id, recipe);
    }

    /// The lease a resolved object of this adapter is handed out with.
    ///
    /// Empty, and the module documentation of [`super::retention`] shows why for
    /// each of the two objects.  Minted per resolution rather than shared, so
    /// that no reader has to ask whether two bound values share a retention.
    fn retention(&self) -> GlRetentionLease {
        GlRetentionLease::new([], Rc::clone(&self.releases))
    }

    /// The contract's refusal for a recipe that is not registered or is wrong.
    fn recipe_error(detail: impl Into<String>) -> RecordingError {
        RecordingError {
            kind: RecordingErrorKind::IncompatibleBindingRecipe,
            context: DiagnosticContext {
                passes: Vec::new(),
                resource: None,
                texture_slot: None,
                buffer_slot: None,
                detail: detail.into(),
                unsupported: None,
            },
        }
    }
}

impl<B: GlStateBackend, C: ComputeDomain<B>> RenderObjectProvider<GlCompatibilityDevice<B, C>>
    for GlObjectRegistry<B, C>
{
    fn raster_pipeline(
        &self,
        id: RasterPipelineId,
    ) -> Result<BoundRasterPipeline<RasterPipeline, GlRetentionLease>, RecordingError> {
        let pipeline = self.raster.get(&id).cloned().ok_or_else(|| {
            Self::recipe_error("no raster pipeline is registered for this identity")
        })?;
        Ok(BoundRasterPipeline {
            device: self.device,
            physical: pipeline,
            lease: self.retention(),
        })
    }

    fn compute_pipeline(
        &self,
        id: ComputePipelineId,
    ) -> Result<BoundComputePipeline<ComputePipeline, GlRetentionLease>, RecordingError> {
        let pipeline = self.compute.get(&id).cloned().ok_or_else(|| {
            Self::recipe_error("no compute pipeline is registered for this identity")
        })?;
        Ok(BoundComputePipeline {
            device: self.device,
            physical: pipeline,
            lease: self.retention(),
        })
    }

    fn bindings(
        &self,
        id: BindingSetId,
        resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
        dynamic_offsets: &[u32],
    ) -> Result<BoundBindings<Bindings, GlRetentionLease>, RecordingError> {
        let recipe = *self.bindings.get(&id).ok_or_else(|| {
            Self::recipe_error("no binding recipe is registered for this identity")
        })?;
        // Dynamic offsets belong to storage and uniform *arrays*, which both
        // families' fixed artifacts have none of: every logical binding is a
        // single non-arrayed resource (`shader::layout` and
        // `shader::compute_layout` each set `array_count: 1`, and both suites
        // hold that for every kernel of their family).  A frame that sent one is
        // describing a recipe this adapter is not, and ignoring it silently would
        // apply a caller's offset to nothing.
        if !dynamic_offsets.is_empty() {
            return Err(Self::recipe_error(
                "the fixed bindings declare no arrays, so a dynamic offset has nothing to apply to",
            ));
        }
        let physical = Bindings::new(recipe, resources).map_err(Self::recipe_error)?;
        Ok(BoundBindings {
            device: self.device,
            physical,
            lease: self.retention(),
        })
    }
}
