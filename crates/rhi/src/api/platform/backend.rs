//! Crate-private backend contract for the platform API (sections 5 through 7).
//!
//! Three handles in [`crate::api::platform`] reach a native implementation:
//! [`crate::api::platform::PlatformProvider`] wraps a native instance,
//! [`crate::api::platform::DeviceRequest`] tracks an asynchronous creation, and
//! [`crate::api::platform::Device`] is the logical execution domain everything
//! else is created from. Each has one trait here.
//!
//! Portable validation and legality decisions remain in the public API façade;
//! these methods only report native facts or lower already-validated requests.
//!
//! # What this seam deliberately does not carry
//!
//! No type here names a native handle. The provider's `IDXGIFactory`,
//! `VkInstance`, `MTLDevice`, GPU object, or rendering context stays inside the
//! backend that owns it, and this trait is how the portable layer asks questions
//! about it without ever holding one. That is what section 5.1 is protecting when
//! it keeps the provider constructor off the portable surface.

use crate::api::error::RhiResult;
use crate::api::identity::ObjectId;
use crate::api::platform::DeviceLossInfo;
use crate::api::platform::DeviceStatus;
use crate::api::platform::provider::{AdapterId, AdapterInfo, BackendKind};
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::presentation::PresentationTarget;

/// The native instance behind a [`crate::api::platform::PlatformProvider`].
///
/// One provider is one backend family (section 5), and it may create more than
/// one device: section 3.1 permits several independent logical devices of the
/// same backend, and each request that succeeds receives its own
/// [`crate::api::identity::DeviceIdentity`].
pub(crate) trait ProviderBackend: Send + Sync + 'static {
    /// Enumerates the adapters this provider can expose explicitly.
    ///
    /// The two success shapes are distinct and both are legal: `Ok(None)` means
    /// the provider does not expose portable enumeration at all, which WebGPU and
    /// adopted-context providers may legitimately be in, and `Ok(Some(vec![]))`
    /// means it can enumerate and currently has no candidate. Collapsing them
    /// would tell a caller that a provider with no enumeration "has no adapters",
    /// which is a different and wrong statement.
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>>;

    /// Whether `adapter` has a portable presentation route to `target`.
    ///
    /// The portable layer has already established that `adapter` belongs to this
    /// provider (section 5.4, section 3.1). This answers only the hardware
    /// question, and its answer is never a substitute for carrying the target in
    /// the device request: section 5.8 says the device that is finally created is
    /// what has to present.
    fn supports_presentation(
        &self,
        adapter: AdapterId,
        target: &PresentationTarget,
    ) -> RhiResult<bool>;

    /// Starts lowering a device request.
    ///
    /// Returning the request's backend rather than a
    /// [`crate::api::platform::DeviceRequest`] keeps the portable layer in charge
    /// of the handle: the request's single-shot rule, and the identity the
    /// resulting device is minted under, are portable contracts and not
    /// something a backend gets to shape.
    fn request_device(
        &self,
        descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn DeviceRequestBackend>>;
}

/// An in-flight device request's native side (section 5.9).
///
/// Single-shot by contract: `poll` is called until it reports
/// [`RequestProgress::Ready`] or an error, and the portable layer retires the
/// request at that point. A backend therefore does not need to defend against
/// being polled after it has answered.
pub(crate) trait DeviceRequestBackend: Send + Sync + 'static {
    /// Advances the request one step without blocking.
    ///
    /// This is not a place to spin on a native fence. Section 5.9 requires the
    /// host to keep pumping its own event loop; a backend that blocked here would
    /// deadlock the very browser or window messages the request depends on.
    fn poll(&mut self) -> RhiResult<RequestProgress>;
}

/// What a device request's native side has produced so far.
///
/// Only a backend constructs one, and the only backend in the tree today is
/// compiled for the test build, so in a non-test build both variants are
/// unconstructed. That expectation expires on its own the moment a native
/// backend lands: an `expect` that is no longer fulfilled is an error, so the
/// attribute cannot quietly outlive the reason written on it.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "constructed by a backend, and the only backend in the tree is the test-build mock"
    )
)]
pub(crate) enum RequestProgress {
    /// Still in flight; poll again later.
    Pending,
    /// The native device exists.
    ///
    /// The portable layer mints the identity and composes the
    /// [`crate::api::platform::Device`], because section 6.1 ties identity
    /// minting to a completed request and a backend that chose its own identity
    /// could hand two domains the same one.
    Ready(Box<dyn DeviceBackend>),
}

/// The native device behind a [`crate::api::platform::Device`].
///
/// This trait owns the device's liveness, which is why there is no `mark_lost`
/// on the portable handle: the backend is what observes a native loss, and
/// section 65.3's rule that each concern has exactly one authority makes the
/// backend that authority. The portable layer owns the *rules* about loss — that
/// it is terminal, that the summary is stable, and that a lost device answers
/// `DeviceLost` only after ownership has been decided — and reads the fact from
/// here.
pub(crate) trait DeviceBackend: Send + Sync + 'static {
    /// The backend family this device came from.
    ///
    /// Diagnostics, selection provenance, and tooling UI only. Section 6.3 is
    /// explicit that this is not a capability oracle: the same family exposes
    /// different capabilities on different drivers.
    fn backend_kind(&self) -> BackendKind;

    /// A snapshot of the adapter that was actually selected.
    ///
    /// Section 6.3 asks a device to report what it actually got, which is a
    /// weaker and always-answerable question than listing the candidates — which
    /// is why this must answer even on a provider that does not enumerate.
    fn adapter_info(&self) -> &AdapterInfo;

    /// What this device's enumeration observed it can do.
    ///
    /// Returned by value rather than borrowed, because the portable layer takes
    /// ownership of these facts and interns them: section 7.1 makes the
    /// compatibility id a function of the facts, and a backend that minted its own
    /// id could hand two domains the same one. The backend's job is to describe
    /// what it saw; deciding what that description is *called* is not its job.
    ///
    /// Section 7.2's completeness rule makes this a demanding method rather than a
    /// courtesy, and it is worth being precise about how demanding, because the
    /// answer is not "record everything".
    /// [`crate::api::capability::CapabilityFacts`] has two lookup rules and the
    /// difference between them is whether the query's key space is one a backend
    /// can walk in full. Where it is — [`crate::api::resource::buffer::BufferUsage`]'s
    /// sixty-four masks — an absent entry is a hole in enumeration and the query
    /// panics, so this must record all of them. Where it is not — a texture's
    /// sample count, a binding's element count, a route's sample count — no
    /// enumeration could have been complete, an absent entry answers the negative,
    /// and what this owes is a faithful table rather than an exhaustive one.
    ///
    /// The distinction matters in both directions. A backend that reads "record
    /// everything" as licence to skip the enumerable families will panic the first
    /// time a caller asks about a buffer; a backend that reads it as licence to
    /// skip the others will silently refuse textures, and the symptom will look
    /// like a driver limitation rather than like a gap.
    ///
    /// Like [`Self::adapter_info`], this must answer even on a device whose
    /// provider does not enumerate: it describes the device that exists, not the
    /// candidates that might have been chosen.
    fn capability_facts(&self) -> crate::api::capability::CapabilityFacts;

    /// The logical submission lanes this device offers.
    ///
    /// Separate from [`Self::capability_facts`] because they are a different kind
    /// of fact — lanes are what a batch is added to, not what a query is asked
    /// against — and because section 7.2's base guarantee relates the two: every
    /// device has a lane accepting `RASTER | COPY`, and one accepting `COMPUTE`
    /// when that feature is enabled. The portable layer is what checks that
    /// relation, at the one moment both halves are in hand
    /// ([`crate::api::submission::SubmissionCapabilities::validate_base_guarantee`]),
    /// so a backend that reported a compute lane and no `Compute` feature is
    /// refused rather than published.
    fn submission_capabilities(&self) -> crate::api::submission::SubmissionCapabilities;

    /// This device's process-local object ID.
    ///
    /// Section 3 gives every RHI object an [`ObjectId`] distinct from any native
    /// handle, and section 7.1 requires tooling to describe what it observes by
    /// that ID rather than by a pointer.
    fn object_id(&self) -> ObjectId;

    /// Whether the device is still usable.
    fn status(&self) -> DeviceStatus;

    /// Why the device was lost, or `None` while it is active.
    ///
    /// The summary must stay available rather than being delivered once, which is
    /// what section 6.5 requires of it: a caller that asks twice, or asks long
    /// after the loss, gets the same answer.
    fn loss_info(&self) -> Option<DeviceLossInfo>;

    /// Non-blockingly advances RHI-owned completion, loss, and callback
    /// bookkeeping.
    ///
    /// Section 6.6 requires the host to keep pumping its own loop; a host that
    /// polled this instead would starve the browser or window messages the RHI
    /// depends on.
    fn poll(&self) -> RhiResult<()>;

    /// Blocks until the device is idle.
    ///
    /// Shutdown and diagnostics only. Section 6.7 forbids it as a per-frame
    /// retirement mechanism and as the correctness mechanism of a render loop. A
    /// backend that cannot wait must return
    /// [`crate::api::RhiErrorKind::Unsupported`] rather than pretend to have
    /// waited — the no-silent-fallback rule, applied to a verb where the
    /// substitute would be invisible.
    fn wait_idle(&self) -> RhiResult<()>;

    /// Allocates the native memory behind one buffer.
    ///
    /// The division of labour is the one section 3 draws and the reason this
    /// returns a backend object rather than a [`crate::api::resource::Buffer`]:
    /// the *portable* layer mints the identity, retains the descriptor, and runs
    /// every refusal in [`crate::api::resource::buffer::validate_buffer_descriptor`] —
    /// so by the time this is reached the request is legal and the backend's only
    /// remaining question is whether the driver will satisfy it. A backend that
    /// minted its own identity could hand two domains the same one, and a backend
    /// that returned the handle would be the second place that decides whether a
    /// size of zero is acceptable.
    ///
    /// Nothing about the descriptor is rewritten on the way through: validation is
    /// a check and not a normalization, so what section 18.8 recovers from the
    /// handle is what the caller wrote.
    ///
    /// `descriptor.memory` is a preference and never a correctness guarantee
    /// (section 11.2), so a backend is free to place the allocation wherever its
    /// own API puts it. What it may not do is substitute a different *kind* of
    /// allocation to make the request succeed — discipline 3, and the reason a
    /// buffer that could not be placed device-locally is still not silently
    /// turned into a host-visible one.
    fn create_buffer(
        &self,
        descriptor: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::BufferBackend>>;

    /// Prepares the native entry point behind one shader module.
    ///
    /// The same division of labour as [`Self::create_buffer`], with one difference
    /// that is the whole point of this method's documentation: the portable layer
    /// decides whether the device may use the artifact at all —
    /// [`crate::api::capability::EnabledCapabilities::shader_acceptance`], section
    /// 19.10, which consults the device's recorded facts and never a backend kind —
    /// so by the time this is reached the *portable* question is answered and the
    /// backend's only remaining question is how its own API wants the bytes.
    ///
    /// **What success here does not mean.** For a backend whose native API has no
    /// shader-module object, preparing an entry point is copying bytes and cannot
    /// fail; the driver's verdict on whether those bytes are a legal program
    /// arrives later, at pipeline creation, and a backend must not report it here.
    /// Section 19.10 puts a real compile error in
    /// [`crate::api::RhiError`] plus a `DiagnosticEvent`, which is a thing that
    /// happens when there is something to compile — a source form on a runtime
    /// compiler. A backend that could not do the work must report
    /// [`crate::api::RhiErrorKind::Unsupported`] rather than return an object that
    /// will fail at first use (discipline 3: never a silent substitute).
    fn create_shader(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<Box<dyn crate::api::shader::backend::ShaderModuleBackend>>;

    /// Assembles the native descriptor packet behind one bind group.
    ///
    /// The same division of labour as [`Self::create_buffer`]. By the time this is
    /// reached the layout match, every resource's range and usage, the device's
    /// `max_*_binding_size` and offset-alignment limits, and the storage-access
    /// question have all been answered portably in
    /// [`crate::api::binding::validate_bind_group_descriptor`] — so a backend has
    /// no legality question left, only a native one: which descriptors to write,
    /// where, and what to keep alive so that the addresses in them stay valid.
    ///
    /// **What the returned object must own.** A native descriptor is a *pointer*
    /// into an allocation, not an owner of it. A backend that wrote an address and
    /// kept no reference to the resource behind it would hand the GPU a
    /// use-after-free the first time the caller dropped its buffer, and nothing
    /// portable would catch it: section 22.2 makes a bind group a logical owner of
    /// everything it binds, so the ownership has to be established here, on the
    /// native side, where the address is taken.
    ///
    /// A backend whose native API has no descriptor-packet object — one that
    /// resolves bindings at draw time from the portable handle — must still answer
    /// this method, because the portable handle's shape does not depend on it.
    /// What it may not do is return an object that fails at first use
    /// (discipline 3: never a silent substitute).
    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>>;

    /// Builds the driver's pipeline state object behind one compute pipeline.
    ///
    /// This is the first creation verb in the crate whose native call can fail for
    /// a reason about the *program* rather than about the descriptor. The bytes a
    /// producer supplied have passed the portable acceptance rule, which is a
    /// statement about recorded device facts; whether they are a legal program,
    /// whether they match the interface the pipeline declares, and whether the
    /// driver can build state for them are questions only the native compiler and
    /// driver answer, and this is where they are asked.
    ///
    /// A native failure here is therefore reported as it arrives, with its own
    /// kind, and never folded into a portable one: a driver refusing a shader is
    /// [`crate::api::RhiErrorKind::BackendFailure`], not `InvalidUsage`, because
    /// the caller's descriptor had already passed every check this crate can make
    /// (discipline 4).
    fn create_compute_pipeline(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>>;

    /// Lowers and submits one plan that has passed the portable preflight.
    ///
    /// This is Phase B of section 41.3 and the two phases are not symmetric. By
    /// the time this is reached the portable layer has decided identity, lane and
    /// work-domain legality, the dependency graph, and the in-flight hazard check
    /// — everything section 40.5 lists — so a backend has no legality question
    /// left to ask. What it has is a lowering to perform, and the one thing it
    /// may not do is fail after accepting: an `Err` here would tell the caller
    /// "nothing happened" while a native queue had already been fed, which is the
    /// outcome section 41.3 exists to forbid.
    ///
    /// So the division is: `Err` means *nothing was committed*, and it is
    /// reserved for the cases where that is true — a command this backend cannot
    /// lower ([`crate::api::RhiErrorKind::Unsupported`], discipline 3 in
    /// the backend contract), a device that ended before the commit
    /// ([`crate::api::RhiErrorKind::DeviceLost`]). A problem discovered *after*
    /// the commit is reported through [`Self::completion`] as a terminal
    /// [`crate::api::submission::CompletionState::Failed`].
    ///
    /// The serials in the returned [`crate::api::submission::backend::SubmissionOutcome`]
    /// are this backend's own numbers. The portable layer wraps each into a
    /// completion token whose device half only it can mint, which is what keeps
    /// section 3.1's identity rule on the portable side of the seam.
    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome>;

    /// The state of one completion serial this backend reported.
    ///
    /// Non-blocking (section 41.10): a frame loop polls this alongside
    /// [`Self::poll`], and nothing here may wait on the GPU. [`Self::wait_idle`]
    /// is the only blocking verb in this crate and section 6.7 confines it to
    /// shutdown, recovery and diagnostics.
    ///
    /// Section 41.8's liveness rule is the demanding half. After a device loss
    /// every serial this backend ever reported must answer
    /// [`crate::api::submission::CompletionState::DeviceLost`] — not `Pending`,
    /// and not forever. A backend that answered `Pending` for work whose device
    /// has ended would hang the caller's loop on work that can never finish.
    fn completion(&self, serial: u64) -> crate::api::submission::CompletionState;
}
