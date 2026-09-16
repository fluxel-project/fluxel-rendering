//! The registry the common executor resolves this adapter's objects through.
//!
//! Responsibility: answer the contract's three `RenderObjectProvider` questions
//! from what a renderer registered before the frame, and hold the two facts that
//! answer them -- which raster recipe a [`RasterPipelineId`] is, and which recipe
//! a [`BindingSetId`] belongs to.  It resolves nothing native: a pipeline is
//! lowered at registration and a binding set is validated against the frame's
//! own resources, so the only work a frame does here is a lookup and a check.
//!
//! Not owned here: the frame's resources (the executor's
//! `FrameResourceProvider`), the machine that links and binds (that is
//! [`super`]'s, at the verbs that install), and the graph's own validation that
//! a pass declared every access it binds.
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

use crate::resource::RasterKernel;
use crate::webgl2::api::{BufferId, GlError, GlFamilyProfile, TextureId};
use crate::webgl2::state::GlStateBackend;

use super::super::shader;
use super::object::{Bindings, RasterPipeline};
use super::retention::{GlRetentionLease, ReleaseQueue};
use super::{GlCompatibilityDevice, UnsupportedComputePipeline};

/// The objects one adapter's renderer has registered.
///
/// Not `Sync` and not shared: it is filled on the context's owner thread and
/// read by frames recorded there, which is the same single-threaded reach the
/// rest of this adapter has.
///
/// # Why the backend is a type parameter
///
/// This type's only purpose is to implement a trait parameterized by the adapter
/// it answers for, and the backend is not otherwise a fact it holds -- so without
/// the parameter, `RenderObjectProvider<GlCompatibilityDevice<B>>` would leave
/// `B` unconstrained at every call through a registry value, and a caller who
/// already knows its backend (a test, or a renderer holding the registry beside
/// its adapter) would have to name it by hand.  The parameter makes "which
/// adapter this answers for" part of the registry's identity rather than a fact
/// only its impl knows, and it is a `fn() -> B` phantom because nothing is ever
/// built from it: the registry must not become a place a second adapter could be
/// reached through.
pub(crate) struct GlObjectRegistry<B: GlStateBackend> {
    device: DeviceIdentity,
    profile: GlFamilyProfile,
    releases: Rc<ReleaseQueue>,
    raster: HashMap<RasterPipelineId, RasterPipeline>,
    bindings: HashMap<BindingSetId, RasterKernel>,
    backend: PhantomData<fn() -> B>,
}

impl<B: GlStateBackend> GlObjectRegistry<B> {
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

    /// Records which artifact a binding set id belongs to.
    ///
    /// The recipe and not a resolved set: what a set id means is which bindings
    /// a pipeline layout declares, and the resources those bindings name arrive
    /// with the frame.  A set that was never registered is what `bindings`
    /// refuses; an id registered twice keeps the last recipe, which is the same
    /// last-write-wins a re-registration of a pipeline has.
    pub(crate) fn register_bindings(&mut self, id: BindingSetId, kernel: RasterKernel) {
        self.bindings.insert(id, kernel);
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

impl<B: GlStateBackend> RenderObjectProvider<GlCompatibilityDevice<B>> for GlObjectRegistry<B> {
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
        _id: ComputePipelineId,
    ) -> Result<BoundComputePipeline<UnsupportedComputePipeline, GlRetentionLease>, RecordingError>
    {
        // Unreachable rather than unimplemented: this adapter's `ComputePipeline`
        // is uninhabited, so no compute pipeline value can exist for a pass to
        // select.  Returning the contract's error rather than panicking keeps
        // that a refusal if it ever does become reachable, and naming the fact
        // here is what makes the uninhabited type a claim instead of a comment.
        Err(Self::recipe_error(
            "this adapter has no compute pipeline object, so none can be registered",
        ))
    }

    fn bindings(
        &self,
        id: BindingSetId,
        resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
        dynamic_offsets: &[u32],
    ) -> Result<BoundBindings<Bindings, GlRetentionLease>, RecordingError> {
        let kernel = *self.bindings.get(&id).ok_or_else(|| {
            Self::recipe_error("no binding recipe is registered for this identity")
        })?;
        // Dynamic offsets belong to storage and uniform *arrays*, which this
        // family's fixed artifacts have none of: every logical binding is a
        // single non-arrayed resource (`shader::layout` sets `array_count: 1`,
        // and the lowering's suite holds that for all ten).  A frame that sent
        // one is describing a recipe this adapter is not, and ignoring it
        // silently would apply a caller's offset to nothing.
        if !dynamic_offsets.is_empty() {
            return Err(Self::recipe_error(
                "the fixed raster bindings declare no arrays, so a dynamic offset has nothing to apply to",
            ));
        }
        let physical = Bindings::new(kernel, resources).map_err(Self::recipe_error)?;
        Ok(BoundBindings {
            device: self.device,
            physical,
            lease: self.retention(),
        })
    }
}
