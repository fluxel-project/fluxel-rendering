//! The CPU/mock backend for portable API conformance tests.
//!
//! `version-plan.md` section 4 requires a "CPU/mock conformance suite shared by
//! all three backends" as proof for 0.16, and this module is the backend half of
//! it. It exists for a specific reason rather than for convenience: most of the
//! portable contract is decidable without hardware, and a contract rule that can
//! only be checked on a machine with a discrete GPU is a rule that will be broken
//! by the next change. A backend that answers from memory makes those rules
//! testable everywhere, on every platform, in the same run as everything else.
//!
//! It is a **mock and not a reference implementation**, and the difference
//! matters for what it may be used to prove. It performs no lowering, holds no
//! native object, and has no opinion about GPUs; it cannot show that anything is
//! correct on hardware, and `CLAUDE.md` section 4.8 forbids the reverse reading —
//! a mock or a `TestRhi` must never stand in for real-GPU correctness. What it
//! can show, and what it is here to show, is that the portable layer's decisions
//! are made in the portable layer.
//!
//! # Scope limits, recorded rather than implied
//!
//! Where its capability table answers completely and where it does not, so that
//! neither reading is left to be guessed:
//!
//! - **Complete where an answer is a finite decision.** `supports_feature`,
//!   `limit`, `limits`, `format`, `submission`, `compatibility_id`, and
//!   `fingerprint` all answer from a table that a test fills in whole. A test that
//!   wants a device with facts asks for them by name
//!   ([`MockDevice::with_capabilities`]), so the contract under examination is
//!   visible in the test rather than implied by a default.
//! - **It records no support answers at all, and what that costs is not uniform.**
//!   This backend probes nothing, so its support tables are empty, and the four
//!   accessors answer an empty table in the two ways
//!   [`crate::api::capability::CapabilityFacts`] documents. `buffer_support` —
//!   keyed on the sixty-four usage masks, a space enumeration could have covered —
//!   panics, which is the correct reading: a mock that was asked to be a device
//!   and recorded nothing is a broken mock. `texture_support`, `binding_support`,
//!   and `route` carry an unbounded component in their keys and answer
//!   `Unsupported`. A test that needs any of the four wants
//!   [`MockDevice::with_capabilities`] and a table it states itself, which is the
//!   same thing that makes the contract under test visible.
//! - **It lowers buffer creation and nothing else.** A
//!   [`crate::api::resource::backend::BufferBackend`] it hands back is a token holding
//!   the size and usage it was asked for — no memory, no address, no operation —
//!   which is enough to prove the portable verb reached the backend and not
//!   enough to prove anything about a GPU. Texture, view, sampler, shader,
//!   binding, pipeline, recorder, submission, and presentation seams do not exist
//!   yet; when they do, this backend grows the same way the native ones will.
//!
//! # Why it is `cfg(test)` and not behind `test-support`
//!
//! The `test-support` feature exists in `Cargo.toml`, and the mock will move
//! behind it as soon as something outside this crate's own test build needs it —
//! a sibling crate's integration test, an example, or a conformance binary. It is
//! not there yet because the promise would be premature: moving it now would mean
//! this backend's use in a `--features test-support` (non-`test`) build
//! un-fulfilling the `#[expect(dead_code)]` attributes that guard the
//! constructors it calls, and trading a real diagnostic for a feature name.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::platform::backend::{
    DeviceBackend, DeviceRequestBackend, ProviderBackend, RequestProgress,
};
use crate::api::platform::{
    AdapterId, AdapterInfo, BackendKind, Device, DeviceLossInfo, DeviceRequestDescriptor,
    DeviceStatus,
};
use crate::api::presentation::backend::{ConfiguredPresentationBackend, PresentationBackend};
use crate::api::presentation::{
    Extent2d, PresentReceiptId, PresentState, PresentationConfiguration, PresentationExtentControl,
    PresentationTarget, PresentationTargetCapabilities, PresentationTimestamp,
    PresentationTimingCapabilities,
};
use crate::api::resource::backend::{
    BufferBackend, MappedBufferBackend, MappingRequestBackend, QuerySetBackend,
};
use crate::api::resource::buffer::{
    BufferDescriptor, BufferRange, BufferSupport, BufferSupportLimits, BufferUsage,
};
use crate::api::resource::{Buffer, MapMode};
use crate::api::shader::vocabulary::AcceptedCodeForm;
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};

/// What a mock device request eventually reports.
///
/// A failure carries a message rather than a built [`RhiError`] because the
/// request is single-shot and the error is produced at the moment it is
/// reported, which is the only moment at which "the backend failed" is still
/// true of the request.
enum MockOutcome {
    /// A device is available.
    Succeeds,
    /// The request fails with [`RhiErrorKind::Unsupported`] and this message.
    Fails(String),
}

/// How a mock provider answers `enumerate_adapters`.
///
/// The three shapes are the three the contract distinguishes, and they are
/// distinct on purpose: see [`ProviderBackend::enumerate_adapters`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MockEnumeration {
    /// The provider exposes no portable enumeration at all (`Ok(None)`).
    NotExposed,
    /// The provider can enumerate and currently has no candidate.
    NoCandidate,
    /// The provider can enumerate its one adapter.
    Available,
}

/// A provider that answers from memory.
pub(crate) struct MockProvider {
    backend: BackendKind,
    instance: DeviceInstanceId,
    enumeration: MockEnumeration,
    presents: bool,
    /// How many polls report `Pending` before the request resolves.
    pending_steps: u32,
    outcome: MockOutcome,
    /// The contract the device this provider produces will report.
    facts: CapabilityFacts,
    /// The lanes that device will offer.
    submission: SubmissionCapabilities,
}

impl MockProvider {
    /// A provider for `backend` under `instance` that enumerates one adapter and
    /// resolves a device request on the first poll.
    ///
    /// The default is the boring one deliberately: a test that wants the
    /// interesting shapes — no enumeration, a multi-step request, a failure —
    /// asks for them by name, so a reader of the test learns which contract shape
    /// is under examination.
    pub(crate) fn new(backend: BackendKind, instance: DeviceInstanceId) -> Self {
        Self {
            backend,
            instance,
            enumeration: MockEnumeration::Available,
            presents: true,
            pending_steps: 0,
            outcome: MockOutcome::Succeeds,
            facts: CapabilityFacts::empty(),
            submission: default_lanes(),
        }
    }

    /// Makes the device this provider produces report `facts`.
    ///
    /// The adapter snapshot this provider hands out keeps its own, separate facts;
    /// that asymmetry is section 7.2's, not an oversight — `Available` is what the
    /// adapter offers and `Enabled` is what the device got, and the two are
    /// deliberately not the same object here.
    pub(crate) fn with_capability_facts(mut self, facts: CapabilityFacts) -> Self {
        self.facts = facts;
        self
    }

    /// Makes the device this provider produces offer `submission`.
    pub(crate) fn with_submission_capabilities(
        mut self,
        submission: SubmissionCapabilities,
    ) -> Self {
        self.submission = submission;
        self
    }

    /// The adapter this provider reports, under this provider's identity.
    ///
    /// Visible so a test can hand the same snapshot to a [`MockDevice`] directly
    /// instead of going through a device request, which is what the tests that
    /// need to observe a loss have to do.
    pub(crate) fn adapter(&self) -> AdapterInfo {
        AdapterInfo::new(
            AdapterId::new(self.instance.as_u64(), 0),
            format!("mock {:?} adapter", self.backend),
            self.backend,
            None,
            None,
            AvailableCapabilities::from_facts(CapabilityFacts::empty()),
        )
    }

    /// Changes what `enumerate_adapters` reports.
    pub(crate) fn enumerating(mut self, enumeration: MockEnumeration) -> Self {
        self.enumeration = enumeration;
        self
    }

    /// Changes whether `supports_presentation` answers yes.
    pub(crate) fn presenting(mut self, presents: bool) -> Self {
        self.presents = presents;
        self
    }

    /// Makes a device request stay `Pending` for `steps` polls before resolving.
    pub(crate) fn pending_steps(mut self, steps: u32) -> Self {
        self.pending_steps = steps;
        self
    }

    /// Makes a device request fail instead of producing a device.
    pub(crate) fn failing(mut self, message: &str) -> Self {
        self.outcome = MockOutcome::Fails(message.to_string());
        self
    }

    /// Wraps this provider for a [`crate::api::platform::PlatformProvider`].
    pub(crate) fn boxed(self) -> Box<dyn ProviderBackend> {
        Box::new(self)
    }
}

impl ProviderBackend for MockProvider {
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        match self.enumeration {
            MockEnumeration::NotExposed => Ok(None),
            MockEnumeration::NoCandidate => Ok(Some(Vec::new())),
            MockEnumeration::Available => Ok(Some(vec![self.adapter()])),
        }
    }

    fn supports_presentation(
        &self,
        _adapter: AdapterId,
        _target: &PresentationTarget,
    ) -> RhiResult<bool> {
        Ok(self.presents)
    }

    fn request_device(
        &self,
        _descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn DeviceRequestBackend>> {
        Ok(Box::new(MockRequest {
            backend: self.backend,
            adapter: self.adapter(),
            remaining: self.pending_steps,
            outcome: match &self.outcome {
                MockOutcome::Succeeds => MockOutcome::Succeeds,
                MockOutcome::Fails(message) => MockOutcome::Fails(message.clone()),
            },
            facts: self.facts.clone(),
            submission: self.submission.clone(),
        }))
    }
}

/// A device request that resolves after a fixed number of polls.
struct MockRequest {
    backend: BackendKind,
    adapter: AdapterInfo,
    remaining: u32,
    outcome: MockOutcome,
    facts: CapabilityFacts,
    submission: SubmissionCapabilities,
}

impl DeviceRequestBackend for MockRequest {
    fn poll_or_register_waker(&mut self, waker: &Waker) -> RhiResult<RequestProgress> {
        if self.remaining > 0 {
            self.remaining -= 1;
            // The mock models immediately schedulable progress: it never relies
            // on the provider self-waking a pending request.
            waker.wake_by_ref();
            return Ok(RequestProgress::Pending);
        }
        match &self.outcome {
            // The contract is cloned into the device rather than moved, because
            // this borrows `self` and the request is single-shot by contract: the
            // portable layer retires a request the moment it reports `Ready`, so
            // this arm runs at most once and the tables are copied once.
            MockOutcome::Succeeds => Ok(RequestProgress::Ready(Box::new(MockDevice {
                backend: self.backend,
                adapter: self.adapter.clone(),
                object: ObjectId::next(),
                liveness: Mutex::new(Liveness {
                    status: DeviceStatus::Active,
                    loss: None,
                }),
                facts: self.facts.clone(),
                submission: self.submission.clone(),
                allocations: AtomicUsize::new(0),
                shader_modules: AtomicUsize::new(0),
                bind_groups: AtomicUsize::new(0),
                compute_pipelines: AtomicUsize::new(0),
                submissions: AtomicUsize::new(0),
                next_completion: AtomicU64::new(1),
                holding: AtomicBool::new(false),
                completed_at_loss: AtomicBool::new(false),
                completion_waiters: Mutex::new(Vec::new()),
                mapping: Arc::new(MockMappingControl {
                    held: AtomicBool::new(false),
                    waiters: Mutex::new(Vec::new()),
                }),
                presentation_timing: AtomicBool::new(false),
                presentation_available: AtomicBool::new(true),
            }))),
            MockOutcome::Fails(message) => {
                Err(RhiError::new(RhiErrorKind::Unsupported, message.clone()))
            }
        }
    }
}

/// A device's liveness, as the backend observes it.
struct Liveness {
    status: DeviceStatus,
    loss: Option<DeviceLossInfo>,
}

/// The lane set a mock device offers unless a test says otherwise.
///
/// One lane accepting `RASTER | COPY`, which is the smallest set section 10's base
/// guarantee accepts. Deliberately not including `COMPUTE`: the guarantee's compute
/// clause is conditional on the `Compute` feature being enabled, and the default
/// facts enable no feature — so a default that also accepted compute work would
/// describe a device more capable than its own fact table says, which is exactly
/// the kind of half-consistent enumeration
/// [`crate::api::submission::SubmissionCapabilities::validate_base_guarantee`]
/// exists to catch.
fn default_lanes() -> SubmissionCapabilities {
    SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
        SubmissionLaneId::new(0),
        SubmissionLaneClass::General,
        LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
    )])
}

/// Query fixtures enable compute/timestamp-in-compute, so their lane contract
/// must advertise the matching execution domain as well.
fn query_lanes() -> SubmissionCapabilities {
    SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
        SubmissionLaneId::new(0),
        SubmissionLaneClass::General,
        LaneWorkDomains::RASTER
            .union(LaneWorkDomains::COMPUTE)
            .union(LaneWorkDomains::COPY),
    )])
}

/// A buffer this backend allocated.
///
/// It holds the two facts the portable layer handed over and nothing else, which
/// is honestly all a backend that allocates no memory has. What it is *for* is
/// telling "the creation verb reached the backend" apart from "the verb returned
/// before lowering" — the one thing a mock of this seam can prove, and the thing
/// that the `unimplemented!` this replaces made unknowable.
///
/// Note what it does not hold: an [`ObjectId`]. Section 3 gives the id to the
/// object that created the resource, and the mutation this struct makes to the
/// portable layer's design is exactly that the backend is not asked for one.
pub(crate) struct MockBuffer {
    size: u64,
    usage: BufferUsage,
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl MockBuffer {
    /// The size the portable layer asked for.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// The usage mask the portable layer asked for.
    ///
    /// Recorded rather than acted on, because usage is a creation-time
    /// *correctness* contract (section 11.1) and this backend has no operation
    /// that could consult it. A test reads it to show that what arrived is what
    /// the caller stated.
    pub(crate) fn usage(&self) -> BufferUsage {
        self.usage
    }
}

impl BufferBackend for MockBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// In-memory mapping lease used only by portable mapping contract tests.
struct MockMappedBuffer {
    destination: Arc<Mutex<Vec<u8>>>,
    offset: usize,
    bytes: Vec<u8>,
    writable: bool,
}

impl Drop for MockMappedBuffer {
    fn drop(&mut self) {
        if self.writable {
            let mut destination = self
                .destination
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            destination[self.offset..self.offset + self.bytes.len()].copy_from_slice(&self.bytes);
        }
    }
}

impl MappedBufferBackend for MockMappedBuffer {
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.writable.then_some(self.bytes.as_mut_slice())
    }
    fn flush(&mut self) -> RhiResult<()> {
        Ok(())
    }
    fn invalidate(&mut self) -> RhiResult<()> {
        Ok(())
    }
}

/// A controllable asynchronous mock mapping request.
///
/// It is deliberately separate from [`MockMappedBuffer`]: a pending request has
/// no host lease and must not copy or publish bytes until its waker is released.
struct MockMappingRequest {
    control: Arc<MockMappingControl>,
    destination: Arc<Mutex<Vec<u8>>>,
    offset: usize,
    len: usize,
    writable: bool,
}

impl MappingRequestBackend for MockMappingRequest {
    fn poll(&mut self, context: &mut Context<'_>) -> Poll<RhiResult<Box<dyn MappedBufferBackend>>> {
        if self.control.held.load(Ordering::Relaxed) {
            self.control.register_waker(context.waker());
            if self.control.held.load(Ordering::Relaxed) {
                return Poll::Pending;
            }
        }
        let bytes = self
            .destination
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            [self.offset..self.offset + self.len]
            .to_vec();
        Poll::Ready(Ok(Box::new(MockMappedBuffer {
            destination: Arc::clone(&self.destination),
            offset: self.offset,
            bytes,
            writable: self.writable,
        })))
    }
}

/// Shared test-only native progress state for mapping requests.
struct MockMappingControl {
    held: AtomicBool,
    waiters: Mutex<Vec<Waker>>,
}

impl MockMappingControl {
    fn register_waker(&self, waker: &Waker) {
        self.waiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(waker.clone());
    }

    fn wake_waiters(&self) {
        let mut waiters = self
            .waiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let registered = core::mem::take(&mut *waiters);
        drop(waiters);
        for waker in registered {
            waker.wake();
        }
    }
}

/// A shader entry point that was never compiled, because there is no compiler.
///
/// The mock has no GPU and no runtime shader compiler, so this holds the artifact's
/// own bytes and nothing else. That is the honest model of a device in this
/// position rather than a shortcut: section 19.10 makes a compile *error* a thing a
/// runtime compiler produces, and this backend does not have one, so it has no
/// error to report and no compiled form to hand back. What it does have is
/// [`Self::artifact`], which is what lets a test show that the module's stage,
/// entry point and code form arrived unchanged.
pub(crate) struct MockShaderModule {
    artifact: crate::api::shader::ShaderArtifact,
}

impl MockShaderModule {
    /// Keeps `artifact` as this module's entry point.
    pub(crate) fn new(artifact: crate::api::shader::ShaderArtifact) -> Self {
        Self { artifact }
    }

    /// The artifact this module was created from.
    pub(crate) fn artifact(&self) -> &crate::api::shader::ShaderArtifact {
        &self.artifact
    }
}

impl crate::api::shader::backend::ShaderModuleBackend for MockShaderModule {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A descriptor packet this backend assembled, in the sense that it kept a copy.
///
/// There is no native descriptor and no descriptor heap here, so what this proves
/// is the same thing [`MockBuffer`] proves and no more: that the creation verb
/// reached the lowerer, and that what arrived is the canonical packet.
/// [`Self::descriptor`] is that packet, and a test reads it to show that the
/// entries were in ascending slot order rather than in the order the caller wrote
/// them — the half of section 22.2 that is otherwise invisible from the outside.
///
/// It holds no [`ObjectId`], for the reason [`MockBuffer`]'s note gives.
pub(crate) struct MockBindGroup {
    descriptor: crate::api::binding::BindGroupDescriptor,
}

impl MockBindGroup {
    /// Keeps `descriptor` as this packet.
    pub(crate) fn new(descriptor: crate::api::binding::BindGroupDescriptor) -> Self {
        Self { descriptor }
    }

    /// The canonical packet the portable layer handed over.
    pub(crate) fn descriptor(&self) -> &crate::api::binding::BindGroupDescriptor {
        &self.descriptor
    }
}

impl crate::api::binding::backend::BindGroupBackend for MockBindGroup {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A pipeline state that was never built, because there is no driver to build it.
///
/// The counterpart of [`MockShaderModule`], and it stops one step later: this
/// backend has no native compiler, so it has no verdict on whether the bytes are a
/// legal program, and it says nothing rather than inventing a refusal. It keeps the
/// descriptor, which is what lets a test show that the pipeline that arrived is the
/// one the caller described — in particular that the shader and the interface were
/// the *same* two objects, since a mock cannot discover a mismatch the way a driver
/// would.
pub(crate) struct MockComputePipeline {
    descriptor: crate::api::pipeline::ComputePipelineDescriptor,
}

impl MockComputePipeline {
    /// Keeps `descriptor` as this pipeline's description.
    pub(crate) fn new(descriptor: crate::api::pipeline::ComputePipelineDescriptor) -> Self {
        Self { descriptor }
    }

    /// The descriptor the portable layer handed over.
    pub(crate) fn descriptor(&self) -> &crate::api::pipeline::ComputePipelineDescriptor {
        &self.descriptor
    }
}

impl crate::api::pipeline::backend::ComputePipelineBackend for MockComputePipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A deliberately inert graphics pipeline for portable tests that only need a
/// valid created-handle backing. Native command lowering is tested by backend
/// suites with that backend's concrete state instead.
pub(crate) struct MockRasterPipeline;

impl crate::api::pipeline::backend::RasterPipelineBackend for MockRasterPipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Inert mesh pipeline backing for portable command-contract tests.
pub(crate) struct MockMeshPipeline;
impl crate::api::pipeline::backend::MeshPipelineBackend for MockMeshPipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Inert ray-tracing pipeline backing for portable command-contract tests.
pub(crate) struct MockRayTracingPipeline;
impl crate::api::pipeline::backend::RayTracingPipelineBackend for MockRayTracingPipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) struct MockTexture;

impl crate::api::resource::backend::TextureBackend for MockTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) struct MockPipelineCache;
impl crate::api::pipeline::backend::PipelineCacheBackend for MockPipelineCache {
    fn serialized_data(&self) -> RhiResult<Vec<u8>> {
        Ok(vec![0xca, 0xce])
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(crate) struct MockTextureView;

impl crate::api::resource::backend::TextureViewBackend for MockTextureView {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) struct MockSampler;

impl crate::api::resource::backend::SamplerBackend for MockSampler {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct MockQuerySet;

impl QuerySetBackend for MockQuerySet {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Native-free AS backing used solely to prove portable sizing/ownership rules.
struct MockAccelerationStructure;

impl crate::api::resource::backend::AccelerationStructureBackend for MockAccelerationStructure {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Native graphics state for a portable test that is not about lowering.
pub(crate) fn raster_pipeline_backend_for_test()
-> Box<dyn crate::api::pipeline::backend::RasterPipelineBackend> {
    Box::new(MockRasterPipeline)
}

/// Native-free mesh backing for public recording tests.
pub(crate) fn mesh_pipeline_backend_for_test()
-> Box<dyn crate::api::pipeline::backend::MeshPipelineBackend> {
    Box::new(MockMeshPipeline)
}
/// Native-free ray-tracing backing for public recording tests.
pub(crate) fn ray_tracing_pipeline_backend_for_test()
-> Box<dyn crate::api::pipeline::backend::RayTracingPipelineBackend> {
    Box::new(MockRayTracingPipeline)
}

/// A bind group backend holding the canonical packet it was handed.
///
/// The same shape as [`module_backend_for_test`], one chapter later: a test that
/// needs a portable `BindGroup` has to supply the backend half, and the honest
/// backend for a test that is not about lowering is one that holds what it was
/// given and claims nothing. It is a function rather than a call to
/// `Arc::new(MockBindGroup::new(..))` at each site so that the *type* the seam
/// wants — `Arc<dyn BindGroupBackend>` — is named once, where a change to the seam
/// will be seen.
pub(crate) fn bind_group_backend_for_test(
    descriptor: crate::api::binding::BindGroupDescriptor,
) -> Box<dyn crate::api::binding::backend::BindGroupBackend> {
    Box::new(MockBindGroup::new(descriptor))
}

/// A compute pipeline backend holding the descriptor it was handed.
///
/// [`bind_group_backend_for_test`]'s counterpart, with one difference worth
/// stating: a pipeline descriptor is *not* canonicalized on the way down — section
/// 28 has no canonical form to hand over — so the descriptor a test passes here is
/// the caller's own, and the mock holding it is what lets a test show that the
/// shader and the interface reaching the backend were the same two objects the
/// caller named.
pub(crate) fn compute_pipeline_backend_for_test(
    descriptor: crate::api::pipeline::ComputePipelineDescriptor,
) -> Box<dyn crate::api::pipeline::backend::ComputePipelineBackend> {
    Box::new(MockComputePipeline::new(descriptor))
}

/// A device that answers from memory.
pub(crate) struct MockDevice {
    backend: BackendKind,
    adapter: AdapterInfo,
    object: ObjectId,
    liveness: Mutex<Liveness>,
    /// What this device reports it can do.
    facts: CapabilityFacts,
    /// The lanes it reports offering.
    submission: SubmissionCapabilities,
    /// How many allocations have reached this backend.
    ///
    /// A counter rather than nothing, because the ordering discipline 1 states —
    /// portable validation first, the backend second — is not observable from the
    /// *result* of a refused creation: a backend that was handed an illegal
    /// descriptor and returned `OutOfMemory` would look the same to a caller as
    /// one that was never called. Counting is what tells those apart, and the
    /// count is read by the tests that assert a refusal never arrives here.
    allocations: AtomicUsize,
    /// How many shader entry points have reached this backend.
    ///
    /// The same observable as `allocations`, one chapter later, and for the same
    /// reason: section 19.10 puts the acceptance verdict *before* the backend call,
    /// and "before" is invisible in the return value — a refused `create_shader`
    /// looks identical whether the backend was reached and declined or was never
    /// called. Counting is what separates them, and it is what pins the verdict's
    /// placement against a later edit that moves it past the port.
    shader_modules: AtomicUsize,
    /// How many descriptor packets have reached this backend.
    ///
    /// The same observable, for the half of section 22 that has one: the layout
    /// match, the range rules and the four binding limits all run inside
    /// `Device::create_bind_group`, so a packet they refuse must leave this at its
    /// previous value.
    bind_groups: AtomicUsize,
    /// How many compute pipelines have reached this backend.
    ///
    /// The same observable again, and here it carries a second meaning: this
    /// counts the times the *native* pipeline builder was asked, which is the only
    /// place a driver's verdict on the program can come from. A pipeline the
    /// portable gate refused must leave this unchanged.
    compute_pipelines: AtomicUsize,
    /// How many plans have reached this backend's submit.
    ///
    /// The same observable as `allocations`, one chapter later. Section 41.3's
    /// Phase A is supposed to have refused a bad plan *before* any native submit,
    /// and "before" is invisible in the return value: `Device::submit` returning
    /// `Err` looks identical whether the backend was reached and declined or was
    /// never called at all. Counting is what separates them, and it is also what
    /// pins the half of section 41.3 that is easiest to get wrong in the
    /// direction nobody notices — a preflight that runs *after* the commit.
    submissions: AtomicUsize,
    /// The next completion serial this device will report.
    ///
    /// Per device rather than process-global, unlike
    /// [`crate::api::identity::ObjectId`]'s counter, and the difference is the
    /// point: these serials are never minted into public identity here — the
    /// portable layer wraps them into a `CompletionPoint` whose device half it
    /// supplies — so two devices sharing a number is not a collision. It is also
    /// what makes `completion` checkable: a serial at or above this value was
    /// never reported, and that is exactly the query a stale token produces.
    next_completion: AtomicU64,
    /// Whether reported work is held short of completion.
    ///
    /// The one thing this backend cannot otherwise model, and the reason it exists:
    /// every other answer here is immediate, so an unheld mock reports `Complete`
    /// the instant a plan is accepted and section 41.8's "it may not remain
    /// `Pending` forever" is unfalsifiable against it. Holding separates the two
    /// moments the chapter is built on — acceptance, which stays immediate
    /// (section 41.7), from completion, which a test can now keep open and then
    /// close.
    ///
    /// Defaults to false so that a test which does not care about the distinction
    /// sees the old immediate answer.
    holding: AtomicBool,
    /// Whether the mock's already-issued work had completed at loss time.
    completed_at_loss: AtomicBool,
    /// Async waiters registered while `holding` made completion pending.
    completion_waiters: Mutex<Vec<Waker>>,
    /// Native progress/waker state for test mapping requests.
    mapping: Arc<MockMappingControl>,
    /// Whether this test device reports a presentation clock for its targets.
    presentation_timing: AtomicBool,
    /// Whether this mock exposes a presentation lowering at all.
    presentation_available: AtomicBool,
}

impl MockDevice {
    /// A live device under the given backend, reporting `adapter`.
    ///
    /// Returned as an `Arc` rather than by value because a test that wants to
    /// observe a loss has to keep a handle of its own: the portable
    /// [`crate::api::platform::Device`] owns the backend, and the backend — not
    /// the handle — is what observes a native loss.
    ///
    /// The capabilities are empty facts over [`default_lanes`], which is the
    /// honest minimum: this backend performs no lowering and probes no hardware, so
    /// a device built here genuinely has no optional feature and no limit to
    /// report. A test that wants a device *with* facts asks for them through
    /// [`Self::with_capabilities`].
    pub(crate) fn new(backend: BackendKind, adapter: AdapterInfo) -> Arc<Self> {
        Self::with_capabilities(backend, adapter, CapabilityFacts::empty(), default_lanes())
    }

    /// The same, with the capability contract a test wants to examine.
    ///
    /// A separate constructor rather than a setter, so that a mock device's
    /// contract is fixed before the portable [`crate::api::platform::Device`] wraps
    /// it and interns it. Section 7.2 makes an enabled contract immutable; a
    /// backend that could still change its facts after the id was minted would make
    /// the id describe something other than the device holds.
    pub(crate) fn with_capabilities(
        backend: BackendKind,
        adapter: AdapterInfo,
        facts: CapabilityFacts,
        submission: SubmissionCapabilities,
    ) -> Arc<Self> {
        Arc::new(Self {
            backend,
            adapter,
            object: ObjectId::next(),
            liveness: Mutex::new(Liveness {
                status: DeviceStatus::Active,
                loss: None,
            }),
            facts,
            submission,
            allocations: AtomicUsize::new(0),
            shader_modules: AtomicUsize::new(0),
            bind_groups: AtomicUsize::new(0),
            compute_pipelines: AtomicUsize::new(0),
            submissions: AtomicUsize::new(0),
            // Starts at 1 so that serial 0 is never reported. A zero would make
            // the "never reported" check below depend on which side of the
            // counter's first draw a caller landed, and a sentinel that is also
            // the first real value is not a sentinel.
            next_completion: AtomicU64::new(1),
            holding: AtomicBool::new(false),
            completed_at_loss: AtomicBool::new(false),
            completion_waiters: Mutex::new(Vec::new()),
            mapping: Arc::new(MockMappingControl {
                held: AtomicBool::new(false),
                waiters: Mutex::new(Vec::new()),
            }),
            presentation_timing: AtomicBool::new(false),
            presentation_available: AtomicBool::new(true),
        })
    }

    /// How many allocations have reached this backend.
    ///
    /// The observable half of discipline 1. A test asserts this is unchanged
    /// after a refused creation, which is the only way to tell "the portable
    /// layer refused before lowering" apart from "the backend was asked and
    /// happened to refuse too".
    pub(crate) fn allocations(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    /// How many shader entry points have reached this backend.
    ///
    /// The same observable one chapter later. Section 19.10 puts the acceptance
    /// verdict before the port, so an artifact this device refuses must leave this
    /// at its previous value.
    pub(crate) fn shader_modules(&self) -> usize {
        self.shader_modules.load(Ordering::Relaxed)
    }

    /// How many descriptor packets have reached this backend.
    ///
    /// Section 22.2's canonicality rule and section 22.3's per-resource lists run
    /// before the port, so a packet they refuse must leave this unchanged.
    pub(crate) fn bind_groups(&self) -> usize {
        self.bind_groups.load(Ordering::Relaxed)
    }

    /// How many compute pipelines have reached this backend.
    ///
    /// Section 28's whole creation list runs before the port, so a pipeline it
    /// refuses must leave this unchanged.
    pub(crate) fn compute_pipelines(&self) -> usize {
        self.compute_pipelines.load(Ordering::Relaxed)
    }

    /// How many plans have reached this backend's `submit`.
    ///
    /// The same observable one chapter later. Section 41.3 puts the whole
    /// preflight before any native submit, so a plan the portable layer refuses
    /// must leave this at its previous value — and a test that only checked the
    /// returned `Err` could not tell that from a backend that was called and
    /// declined.
    pub(crate) fn submissions(&self) -> usize {
        self.submissions.load(Ordering::Relaxed)
    }

    /// Makes this mock represent a backend without presentation lowering.
    pub(crate) fn disable_presentation(&self) {
        self.presentation_available.store(false, Ordering::Relaxed);
    }

    /// Holds every reported completion short, so it answers `Pending`.
    ///
    /// Models a device that has accepted work and has not finished it — the state
    /// section 41.8 forbids a caller's polling loop from being stuck in forever,
    /// and the state this backend is otherwise unable to produce. Acceptance is
    /// deliberately *not* affected: a submit while holding still succeeds, because
    /// section 41.7 makes those two different facts and a mock that conflated them
    /// would hide the distinction the chapter is about.
    pub(crate) fn hold_completion(&self) {
        self.holding.store(true, Ordering::Relaxed);
    }

    /// Lets held work complete.
    ///
    /// One-way, like a fence being signalled. In a real backend the progress would
    /// arrive from the driver through `poll`; here it is a switch, which is honest
    /// because this device has no work to actually run and no driver to run it.
    pub(crate) fn release_completion(&self) {
        self.holding.store(false, Ordering::Relaxed);
        self.wake_completion_waiters();
    }

    /// Holds later mapping requests pending until [`Self::release_mapping`].
    pub(crate) fn hold_mapping(&self) {
        self.mapping.held.store(true, Ordering::Relaxed);
    }

    /// Completes mapping requests currently held by [`Self::hold_mapping`].
    pub(crate) fn release_mapping(&self) {
        self.mapping.held.store(false, Ordering::Relaxed);
        self.mapping.wake_waiters();
    }

    /// Changes the surface-specific presentation-clock fact for contract tests.
    pub(crate) fn set_presentation_timing(&self, enabled: bool) {
        self.presentation_timing.store(enabled, Ordering::Relaxed);
    }

    /// Records that this device is gone, with the reason.
    ///
    /// One-way, like the loss it records: section 6.5 makes device loss terminal
    /// for the whole identity, so there is no matching `mark_active`.
    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        self.completed_at_loss
            .store(!self.holding.load(Ordering::Relaxed), Ordering::Relaxed);
        let mut liveness = self.liveness();
        liveness.status = DeviceStatus::Lost;
        liveness.loss = Some(info);
        drop(liveness);
        self.wake_completion_waiters();
        self.mapping.wake_waiters();
    }

    /// Borrows the liveness cell, surviving a poisoned lock.
    ///
    /// Recovering from poisoning rather than propagating it is correct here and
    /// only here: the guarded value is two plain fields with no invariant that a
    /// panicking holder could have left half-written, so a panic elsewhere must
    /// not turn a later `status()` into a second panic and hide the first one.
    fn liveness(&self) -> MutexGuard<'_, Liveness> {
        self.liveness
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wake_completion_waiters(&self) {
        let mut waiters = self
            .completion_waiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let registered = core::mem::take(&mut *waiters);
        drop(waiters);
        for waker in registered {
            waker.wake();
        }
    }
}

impl DeviceBackend for MockDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }

    fn capability_facts(&self) -> CapabilityFacts {
        self.facts.clone()
    }

    fn submission_capabilities(&self) -> SubmissionCapabilities {
        self.submission.clone()
    }

    fn object_id(&self) -> ObjectId {
        self.object
    }

    fn status(&self) -> DeviceStatus {
        self.liveness().status
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.liveness().loss.clone()
    }

    fn poll(&self) -> RhiResult<()> {
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        Ok(())
    }

    fn create_pipeline_cache(
        &self,
        _descriptor: &crate::api::pipeline::PipelineCacheDescriptor,
    ) -> RhiResult<(
        Box<dyn crate::api::pipeline::backend::PipelineCacheBackend>,
        crate::api::pipeline::PipelineCacheValidationKey,
    )> {
        Ok((
            Box::new(MockPipelineCache),
            crate::api::pipeline::PipelineCacheValidationKey::from_bytes([9; 32]),
        ))
    }

    fn external_memory_capabilities(
        &self,
    ) -> RhiResult<crate::api::external::ExternalMemoryCapabilities> {
        Ok(crate::api::external::ExternalMemoryCapabilities {
            supported_handle_types: vec![crate::api::external::ExternalMemoryHandleType::DmaBuf],
        })
    }

    fn import_external_memory_texture(
        &self,
        _descriptor: &crate::api::external::ExternalTextureImportDescriptor,
        _accepted: &crate::api::resource::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        Ok(Box::new(MockTexture))
    }

    fn allocator_report(&self) -> RhiResult<crate::api::diagnostics::AllocatorReport> {
        Ok(crate::api::diagnostics::AllocatorReport { heaps: Vec::new() })
    }

    fn begin_native_graphics_capture(&self) -> RhiResult<()> {
        Ok(())
    }

    fn end_native_graphics_capture(&self) -> RhiResult<()> {
        Ok(())
    }

    fn presentation(&self) -> Option<&dyn PresentationBackend> {
        self.presentation_available
            .load(Ordering::Relaxed)
            .then_some(self)
    }

    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Box<dyn BufferBackend>> {
        // No refusal here, and the absence is a decision rather than an
        // unfinished arm. Every portable rule about this descriptor has already
        // run in `Device::create_buffer`, and a second opinion here would be
        // either a duplicate of one of those rules or a new one invented by a
        // backend — disciplines 2 and 4. A mock that refused nothing is therefore
        // the correct mock: it has nothing left to refuse.
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(MockBuffer {
            size: descriptor.size,
            usage: descriptor.usage,
            bytes: Arc::new(Mutex::new(vec![
                0;
                usize::try_from(descriptor.size).map_err(
                    |_| RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "mock buffer exceeds host address space"
                    )
                )?
            ])),
        }))
    }

    fn map_buffer(
        &self,
        buffer: &Buffer,
        mode: MapMode,
        range: BufferRange,
    ) -> RhiResult<Box<dyn MappingRequestBackend>> {
        let native = buffer
            .native()
            .as_any()
            .downcast_ref::<MockBuffer>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "mock cannot map a foreign native buffer",
                )
            })?;
        let offset = usize::try_from(range.offset).map_err(|_| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "mapped offset exceeds host address space",
            )
        })?;
        let len = usize::try_from(range.size).map_err(|_| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "mapped length exceeds host address space",
            )
        })?;
        Ok(Box::new(MockMappingRequest {
            control: Arc::clone(&self.mapping),
            destination: Arc::clone(&native.bytes),
            offset,
            len,
            writable: matches!(mode, MapMode::Write),
        }))
    }

    fn create_query_set(
        &self,
        _descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<Box<dyn QuerySetBackend>> {
        Ok(Box::new(MockQuerySet))
    }

    fn acceleration_structure_build_sizes(
        &self,
        _descriptor: &crate::api::resource::AccelerationStructureDescriptor,
    ) -> RhiResult<crate::api::resource::AccelerationStructureBuildSizes> {
        Ok(crate::api::resource::AccelerationStructureBuildSizes {
            acceleration_structure_size: 256,
            build_scratch_size: 256,
            update_scratch_size: 256,
        })
    }

    fn create_acceleration_structure(
        &self,
        _descriptor: &crate::api::resource::AccelerationStructureDescriptor,
        _sizes: crate::api::resource::AccelerationStructureBuildSizes,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::AccelerationStructureBackend>> {
        Ok(Box::new(MockAccelerationStructure))
    }

    fn create_texture(
        &self,
        _descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(MockTexture))
    }

    fn create_texture_view(
        &self,
        _texture: &crate::api::resource::Texture,
        _descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        Ok(Box::new(MockTextureView))
    }

    fn create_sampler(
        &self,
        _descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        Ok(Box::new(MockSampler))
    }

    fn create_shader_request(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::shader::backend::ShaderModuleBackend,
                >,
        >,
    > {
        // No refusal here, and the absence is the same decision `create_buffer`
        // records rather than an unfinished arm: section 19.10's acceptance verdict
        // and every canonicality rule about this artifact have already run in
        // `Device::create_shader`, so there is nothing left for this backend to
        // refuse — and a backend that invented a rule here would be discipline 2's
        // violation.
        //
        // `MockShaderModule` deliberately does *not* claim to have compiled
        // anything. The mock has no runtime compiler, and a fabricated failure
        // would be worse than the absence: it would tell a caller its artifact was
        // wrong when the truth is that nothing ever looked at it.
        self.shader_modules.fetch_add(1, Ordering::Relaxed);
        Ok(crate::api::platform::backend::ready_creation_request(
            Box::new(MockShaderModule::new(artifact.clone())),
        ))
    }

    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        // No refusal here, and the absence is the same decision `create_buffer`
        // records rather than an unfinished arm: the layout match, every range and
        // usage rule, the device's four binding limits and the storage-access
        // question have all been answered portably in `Device::create_bind_group`.
        //
        // What this backend cannot model is the part section 22.2 makes
        // *native* — that a descriptor holds addresses and therefore that the
        // object returned here must outlive them. There is no address here to
        // dangle, so keeping the packet is the honest whole of it, and a test that
        // wants to observe that obligation has to read it off the DX12 backend,
        // where the `Arc`s are real.
        self.bind_groups.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(MockBindGroup::new(descriptor.clone())))
    }

    fn create_compute_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::ComputePipelineBackend,
                >,
        >,
    > {
        // No refusal here, and the absence matters more here than anywhere else in
        // this file: this is the one port in the crate whose native call a real
        // driver *can* refuse for a reason no portable check covers — whether the
        // bytes are a legal program, and whether they match the interface. A mock
        // with no compiler has no such verdict to give, and inventing one would be
        // worse than silence (disciplines 2 and 3).
        //
        // The consequence for a test is that a rejected program is only observable
        // on a backend with a driver behind it. That is not a gap in the mock; it
        // is the shape of the question.
        self.compute_pipelines.fetch_add(1, Ordering::Relaxed);
        Ok(crate::api::platform::backend::ready_creation_request(
            Box::new(MockComputePipeline::new(descriptor.clone())),
        ))
    }

    fn create_raster_pipeline_request(
        &self,
        _descriptor: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::RasterPipelineBackend,
                >,
        >,
    > {
        Ok(crate::api::platform::backend::ready_creation_request(
            Box::new(MockRasterPipeline),
        ))
    }

    /// Accepts the plan, and executes none of it.
    ///
    /// The mock has no GPU, so the honest model of it is a device whose work is
    /// already finished: the serials it reports here are answered
    /// [`CompletionState::Complete`] by `completion` from the moment they exist,
    /// unless a test calls [`Self::hold_completion`] to keep them open. That is not
    /// a shortcut around completion — it is what lets the *portable* half of
    /// section 41 be tested at all. A caller's polling loop, the receipt's two
    /// completion levels, and the fallback in `completion_for` are all portable
    /// logic, and a backend whose only setting were "finished instantly" would
    /// leave the polling half of it unobservable.
    ///
    /// The device's liveness is still consulted on the way out, because the one
    /// thing this backend must model faithfully is section 41.8: a device that has
    /// ended accepts nothing, and every serial it ever reported answers
    /// `DeviceLost`.
    ///
    /// No refusal beyond that, and the absence is a decision rather than an
    /// unfinished arm — the same one `create_buffer` records: every rule about
    /// this plan has already run in `Device::submit`, so a mock with nothing left
    /// to refuse refuses nothing.
    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        if let DeviceStatus::Lost = self.status() {
            return Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "this device was lost; the plan was not submitted",
            )
            .at("MockDevice::submit"));
        }

        self.submissions.fetch_add(1, Ordering::Relaxed);

        // One serial for the whole plan and one per batch, drawn from the same
        // counter so that they are distinguishable. The per-batch serials are
        // reported rather than omitted because this backend *can* be finer, and
        // section 41.2 makes finer the better answer when it is available: the
        // point of a per-batch token is that a readback does not have to await the
        // slowest unrelated batch, and a mock that always fell back to the overall
        // token would leave that path untested.
        let mut serial = self.next_completion.fetch_add(1, Ordering::Relaxed);
        let overall = serial;
        let mut points = Vec::with_capacity(request.batches.len());
        for batch in request.batches {
            serial = self.next_completion.fetch_add(1, Ordering::Relaxed);
            points.push((batch.point, serial));
        }

        Ok(crate::api::submission::backend::SubmissionOutcome {
            completion: overall,
            points,
        })
    }

    /// Answers a serial this backend reported.
    ///
    /// A loss preserves work that was already complete and changes only work the
    /// mock was holding into `DeviceLost`, matching section 41.8's per-point
    /// distinction.
    ///
    /// A serial this backend never reported is a portable-layer bug rather than a
    /// caller error. It answers `Failed` naming the serial instead of panicking,
    /// because the alternative would abort a caller that merely raced a loss, and
    /// because a panic in a query a frame loop polls is worse than a terminal
    /// state it can branch on.
    fn completion(&self, serial: u64) -> crate::api::submission::CompletionState {
        use crate::api::submission::{CompletionFailure, CompletionState};

        if serial >= self.next_completion.load(Ordering::Relaxed) {
            return CompletionState::Failed(CompletionFailure::new(format!(
                "completion serial {serial} was never reported by this device"
            )));
        }

        if let Some(info) = self.loss_info() {
            return if self.completed_at_loss.load(Ordering::Relaxed) {
                CompletionState::Complete
            } else {
                CompletionState::DeviceLost(info)
            };
        }

        // Held work, after the two terminal answers above and never before them.
        // The order is the contract: a lost device answers `DeviceLost` for a held
        // serial, and a serial this device never minted is a bug in the caller's
        // token rather than work in flight — reporting either as `Pending` would
        // be a polling loop that never ends, which is what section 41.8 forbids.
        if self.holding.load(Ordering::Relaxed) {
            return CompletionState::Pending;
        }

        CompletionState::Complete
    }

    fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> crate::api::submission::CompletionState {
        let state = self.completion(serial);
        if matches!(state, crate::api::submission::CompletionState::Pending) {
            self.completion_waiters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(waker.clone());
            // `release_completion` can race the sample above. Re-sampling after
            // publication closes the otherwise lost-wake window without making
            // the portable future depend on an executor-specific primitive.
            let after_registration = self.completion(serial);
            if !matches!(
                after_registration,
                crate::api::submission::CompletionState::Pending
            ) {
                self.wake_completion_waiters();
            }
            return after_registration;
        }
        state
    }
}

impl PresentationBackend for MockDevice {
    fn capabilities(&self, _target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        Ok(PresentationTargetCapabilities::new(
            vec![crate::api::format::TextureFormat::Bgra8Unorm],
            vec![crate::api::presentation::PresentMode::Fifo],
            PresentationExtentControl::HostManaged {
                current: Some(Extent2d {
                    width: 640,
                    height: 480,
                }),
            },
        )
        .with_timing_and_hdr(
            PresentationTimingCapabilities {
                timestamps: self.presentation_timing.load(Ordering::Relaxed),
            },
            None,
        ))
    }

    fn configure(
        &self,
        _device: DeviceIdentity,
        _target: ObjectId,
        _config: &PresentationConfiguration,
    ) -> RhiResult<Box<dyn ConfiguredPresentationBackend>> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the presentation mock only implements capability and clock queries",
        ))
    }

    fn present_state(&self, _receipt: PresentReceiptId) -> RhiResult<PresentState> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the presentation mock has no present receipts",
        ))
    }

    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        _waker: &Waker,
    ) -> RhiResult<PresentState> {
        self.present_state(receipt)
    }

    fn presentation_timestamp(&self, _target: ObjectId) -> RhiResult<PresentationTimestamp> {
        if !self.presentation_timing.load(Ordering::Relaxed) {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "the presentation mock clock is disabled",
            ));
        }
        Ok(PresentationTimestamp {
            value: 42,
            period_nanos: 0.5,
        })
    }
}

/// Test-only adapter which lets assertions retain a view of a mock backend while
/// the portable `Device` still directly owns one boxed backend. Production
/// backends never take this extra reference-counted path.
struct ObservedMockDevice(Arc<MockDevice>);

pub(crate) fn observed_backend(device: Arc<MockDevice>) -> Box<dyn DeviceBackend> {
    Box::new(ObservedMockDevice(device))
}

impl DeviceBackend for ObservedMockDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn backend_kind(&self) -> BackendKind {
        self.0.backend_kind()
    }
    fn adapter_info(&self) -> &AdapterInfo {
        self.0.adapter_info()
    }
    fn capability_facts(&self) -> CapabilityFacts {
        self.0.capability_facts()
    }
    fn submission_capabilities(&self) -> SubmissionCapabilities {
        self.0.submission_capabilities()
    }
    fn object_id(&self) -> ObjectId {
        self.0.object_id()
    }
    fn status(&self) -> DeviceStatus {
        self.0.status()
    }
    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.0.loss_info()
    }
    fn poll(&self) -> RhiResult<()> {
        self.0.poll()
    }
    fn wait_idle(&self) -> RhiResult<()> {
        self.0.wait_idle()
    }
    fn create_pipeline_cache(
        &self,
        descriptor: &crate::api::pipeline::PipelineCacheDescriptor,
    ) -> RhiResult<(
        Box<dyn crate::api::pipeline::backend::PipelineCacheBackend>,
        crate::api::pipeline::PipelineCacheValidationKey,
    )> {
        self.0.create_pipeline_cache(descriptor)
    }
    fn external_memory_capabilities(
        &self,
    ) -> RhiResult<crate::api::external::ExternalMemoryCapabilities> {
        self.0.external_memory_capabilities()
    }
    fn import_external_memory_texture(
        &self,
        descriptor: &crate::api::external::ExternalTextureImportDescriptor,
        accepted: &crate::api::resource::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        self.0.import_external_memory_texture(descriptor, accepted)
    }
    fn allocator_report(&self) -> RhiResult<crate::api::diagnostics::AllocatorReport> {
        self.0.allocator_report()
    }
    fn begin_native_graphics_capture(&self) -> RhiResult<()> {
        self.0.begin_native_graphics_capture()
    }
    fn end_native_graphics_capture(&self) -> RhiResult<()> {
        self.0.end_native_graphics_capture()
    }
    fn presentation(&self) -> Option<&dyn PresentationBackend> {
        self.0.presentation()
    }
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Box<dyn BufferBackend>> {
        self.0.create_buffer(descriptor)
    }
    fn map_buffer(
        &self,
        buffer: &Buffer,
        mode: MapMode,
        range: BufferRange,
    ) -> RhiResult<Box<dyn MappingRequestBackend>> {
        self.0.map_buffer(buffer, mode, range)
    }
    fn create_query_set(
        &self,
        descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<Box<dyn QuerySetBackend>> {
        self.0.create_query_set(descriptor)
    }
    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        self.0.create_texture(descriptor)
    }
    fn create_texture_view(
        &self,
        texture: &crate::api::resource::Texture,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        self.0.create_texture_view(texture, descriptor)
    }
    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        self.0.create_sampler(descriptor)
    }
    fn create_shader_request(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::shader::backend::ShaderModuleBackend,
                >,
        >,
    > {
        self.0.create_shader_request(artifact)
    }
    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        self.0.create_bind_group(descriptor)
    }
    fn create_compute_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::ComputePipelineBackend,
                >,
        >,
    > {
        self.0.create_compute_pipeline_request(descriptor)
    }
    fn create_raster_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::RasterPipelineBackend,
                >,
        >,
    > {
        self.0.create_raster_pipeline_request(descriptor)
    }
    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        self.0.submit(request)
    }
    fn completion(&self, serial: u64) -> crate::api::submission::CompletionState {
        self.0.completion(serial)
    }
    fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> crate::api::submission::CompletionState {
        self.0.completion_or_register_waker(serial, waker)
    }
}

/// A portable device handle over a fresh mock backend.
///
/// For a test that needs a device and has no use for the backend behind it. A
/// test that needs to observe a loss wants [`paired_device_for_test`] instead,
/// because the handle owns the backend and cannot hand it back.
pub(crate) fn device_for_test(identity: DeviceIdentity) -> Device {
    Device::new(identity, observed_backend(mock_native(BackendKind::Dx12)))
        .expect("the mock backend offers a lane accepting raster and copy work")
}

/// A mock device exposing exactly the sampler features supplied by a test.
/// This lets sampler validation assert capability refusal before any native
/// descriptor allocation is attempted.
pub(crate) fn sampler_device_for_test(
    identity: DeviceIdentity,
    features: &[crate::api::platform::OptionalFeature],
    max_anisotropy: Option<u64>,
) -> Device {
    let mut facts = CapabilityFacts::empty();
    for feature in features {
        facts.record_feature(*feature);
    }
    if let Some(max_anisotropy) = max_anisotropy {
        facts.record_limit(
            crate::api::platform::LimitKey::MaxSamplerAnisotropy,
            max_anisotropy,
        );
    }
    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        default_lanes(),
    );
    Device::new(identity, observed_backend(native))
        .expect("sampler mock exposes the base submission lane")
}

/// A portable device handle paired with the backend that owns its liveness.
pub(crate) fn paired_device_for_test(identity: DeviceIdentity) -> (Device, Arc<MockDevice>) {
    let native = mock_native(BackendKind::Dx12);
    (
        Device::new(identity, observed_backend(native.clone()))
            .expect("the mock backend offers a lane accepting raster and copy work"),
        native,
    )
}

/// A native entry point for a portable test that needs a
/// [`ShaderModule`](crate::api::shader::ShaderModule) but is not about the shader
/// lowering.
///
/// The command and pipeline chapters build modules as *inputs* to their own
/// descriptors — a raster pass names one, a pipeline names its stages — and none of
/// those tests is about how a module is made. Without this they would each have to
/// invent a backend object, and an invented one would be a second answer to "what
/// does a module hold".
///
/// It is the mock's own, so a test that reads it back gets the artifact it passed
/// and nothing else: no compilation is modelled, and none is claimed.
pub(crate) fn module_backend_for_test(
    artifact: &crate::api::shader::ShaderArtifact,
) -> Box<dyn crate::api::shader::backend::ShaderModuleBackend> {
    Box::new(MockShaderModule::new(artifact.clone()))
}

/// A portable device that consumes exactly the code forms a test names.
///
/// [`device_for_test`] reports empty facts, and for the shader chapter an empty
/// table is a real answer rather than a hole: the accepted forms are a relation
/// over an unbounded key space, so a device that recorded none refuses every
/// artifact — which is what [`MockShaderModule`] deserves, since it holds an
/// artifact and compiles nothing. A test that wants acceptance to *succeed* has to
/// state which forms this device consumes, and it has to state them here rather
/// than by narrowing the rule: the rule reads a recorded device fact, and a mock
/// that could accept without one would be proving the shortcut section 6.3 forbids.
///
/// `forms` is the caller's because that is what a test varies — the two-form case
/// is what shows the record is a set rather than a single answer.
///
/// Returned as a pair for the same reason [`paired_device_for_test`] is: the
/// assertions worth making about a *refused* creation are about what the backend
/// was never asked — see [`MockDevice::shader_modules`].
pub(crate) fn shaders_for_test(
    identity: DeviceIdentity,
    forms: &[AcceptedCodeForm],
) -> (Device, Arc<MockDevice>) {
    let mut facts = CapabilityFacts::empty();
    for form in forms {
        facts.record_code_form(*form);
    }

    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        default_lanes(),
    );
    (
        Device::new(identity, observed_backend(native.clone()))
            .expect("the mock backend offers a lane accepting raster and copy work"),
        native,
    )
}

/// A portable device whose buffer table answers the whole key space.
///
/// [`device_for_test`] reports empty facts, and an empty table is the wrong
/// answer to a *buffer* query specifically: `BufferUsage`'s sixty-four masks are
/// a space enumeration could have covered, so a miss is a hole and the query
/// panics rather than answering "no". A test that creates a buffer therefore
/// cannot use the default mock, and this is the smallest table that is not a
/// lie — every non-empty mask supported, up to `max_size`.
///
/// `max_size` is the caller's because it is what a test varies: the descriptor
/// rule `size <= max_size` is only exercised by a device whose ceiling is
/// reachable.
///
/// Returned as a pair for the same reason [`paired_device_for_test`] is: the
/// handle owns the backend, and the assertions worth making about a *refused*
/// creation are about what the backend was never asked — see
/// [`MockDevice::allocations`].
pub(crate) fn buffers_for_test(
    identity: DeviceIdentity,
    max_size: u64,
) -> (Device, Arc<MockDevice>) {
    let mut facts = CapabilityFacts::empty();
    let limits = BufferSupportLimits::new(max_size);
    for usage in BufferUsage::all() {
        let support = if usage.is_empty() {
            BufferSupport::Unsupported
        } else {
            BufferSupport::Supported(limits)
        };
        facts.record_buffer_support(usage, support);
    }

    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        default_lanes(),
    );
    (
        Device::new(identity, observed_backend(native.clone()))
            .expect("the mock backend offers a lane accepting raster and copy work"),
        native,
    )
}

/// A mock device that advertises the complete portable query vocabulary.
pub(crate) fn query_device_for_test(identity: DeviceIdentity) -> Device {
    let mut facts = CapabilityFacts::empty();
    let limits = BufferSupportLimits::new(1 << 20);
    for usage in BufferUsage::all() {
        facts.record_buffer_support(
            usage,
            if usage.is_empty() {
                BufferSupport::Unsupported
            } else {
                BufferSupport::Supported(limits)
            },
        );
    }
    for feature in [
        crate::api::platform::OptionalFeature::OcclusionQuery,
        crate::api::platform::OptionalFeature::TimestampQuery,
        crate::api::platform::OptionalFeature::TimestampInsideEncoder,
        crate::api::platform::OptionalFeature::TimestampInsideRasterScope,
        crate::api::platform::OptionalFeature::TimestampInsideComputeScope,
        crate::api::platform::OptionalFeature::PipelineStatisticsQuery,
        crate::api::platform::OptionalFeature::QueryResolve,
        crate::api::platform::OptionalFeature::Compute,
        crate::api::platform::OptionalFeature::IndirectDispatch,
        crate::api::platform::OptionalFeature::ClearBuffer,
        crate::api::platform::OptionalFeature::ClearTexture,
    ] {
        facts.record_feature(feature);
    }
    facts.record_limit(crate::api::platform::LimitKey::MaxQueriesPerQuerySet, 8);
    facts.record_limit(
        crate::api::platform::LimitKey::QueryResolveBufferAlignment,
        8,
    );
    facts.record_pipeline_statistics(crate::api::query::PipelineStatistics::ALL);
    facts.record_timestamp_queries(
        crate::api::query::TimestampQueryCapabilities::new(1.0, None, true)
            .expect("mock timestamp facts are valid"),
    );
    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        query_lanes(),
    );
    Device::new(identity, observed_backend(native))
        .expect("query mock exposes the base submission lane")
}

/// A mock device that advertises native debugger capture and accepts its two calls.
pub(crate) fn native_capture_device_for_test(identity: DeviceIdentity) -> Device {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(crate::api::platform::OptionalFeature::NativeGraphicsCapture);
    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        default_lanes(),
    );
    Device::new(identity, observed_backend(native))
        .expect("capture mock exposes base submission lanes")
}

/// A mock device with exactly the optional features a façade test needs.
pub(crate) fn device_with_features_for_test(
    identity: DeviceIdentity,
    features: &[crate::api::platform::OptionalFeature],
) -> Device {
    let mut facts = CapabilityFacts::empty();
    for feature in features {
        facts.record_feature(*feature);
    }
    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        default_lanes(),
    );
    Device::new(identity, observed_backend(native))
        .expect("feature mock exposes base submission lanes")
}

/// A mock capable of importing the one ordinary external-memory texture used by façade tests.
pub(crate) fn external_memory_device_for_test(identity: DeviceIdentity) -> Device {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(crate::api::platform::OptionalFeature::ExternalMemory);
    let query = crate::api::format::TextureSupportQuery::new(
        crate::api::resource::TextureDimension::D2,
        crate::api::format::TextureFormat::Rgba8Unorm,
        crate::api::resource::TextureUsage::SAMPLED,
        1,
    );
    facts.record_texture_support(
        &query,
        crate::api::format::TextureSupport::Supported(
            crate::api::format::TextureSupportLimits::new(
                crate::api::resource::Extent3d::d2(4096, 4096),
                1,
                1,
            ),
        ),
    );
    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        default_lanes(),
    );
    Device::new(identity, observed_backend(native))
        .expect("external-memory mock exposes base submission lanes")
}

/// A mock device with explicit general-mapping facts.
pub(crate) fn mapped_buffers_for_test(
    identity: DeviceIdentity,
    coherent: bool,
) -> (Device, Arc<MockDevice>) {
    mapped_buffers_with_options_for_test(identity, coherent, false)
}

/// A mapping-capable mock that also permits an open mapping across submission.
pub(crate) fn persistent_mapped_buffers_for_test(
    identity: DeviceIdentity,
) -> (Device, Arc<MockDevice>) {
    mapped_buffers_with_options_for_test(identity, true, true)
}

fn mapped_buffers_with_options_for_test(
    identity: DeviceIdentity,
    coherent: bool,
    persistent: bool,
) -> (Device, Arc<MockDevice>) {
    let mut facts = CapabilityFacts::empty();
    let limits = BufferSupportLimits::new(1 << 20);
    for usage in BufferUsage::all() {
        facts.record_buffer_support(
            usage,
            if usage.is_empty() {
                BufferSupport::Unsupported
            } else {
                BufferSupport::Supported(limits)
            },
        );
    }
    facts.record_feature(crate::api::platform::OptionalFeature::MappablePrimaryBuffers);
    if coherent {
        facts.record_feature(crate::api::platform::OptionalFeature::CoherentMapping);
    }
    if persistent {
        facts.record_feature(crate::api::platform::OptionalFeature::PersistentMapping);
    }
    facts.record_limit(crate::api::platform::LimitKey::MapAlignment, 4);
    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        default_lanes(),
    );
    (
        Device::new(identity, observed_backend(native.clone()))
            .expect("the mapping mock offers base submission lanes"),
        native,
    )
}

/// A mock backend for a device, under the DX12 family and one adapter.
fn mock_native(backend: BackendKind) -> Arc<MockDevice> {
    MockDevice::new(
        backend,
        MockProvider::new(backend, DeviceInstanceId::new(1)).adapter(),
    )
}

/// A recorder over a mock device whose capability snapshot the test states.
///
/// The snapshot is the whole reason this exists. A recorder holds the device's
/// facts rather than the device, so a test that wants a verb to decide *against* a
/// device answer has to state that answer at construction — and the honest way to
/// state it is to build the device and let the recorder take its snapshot, rather
/// than to hand the recorder a table the device never reported.
///
/// Built through `Device::create_recorder` rather than through
/// `CommandRecorder::new`, so a test exercising a verb is also exercising the real
/// creation path: if that path stopped agreeing with the constructor, these would
/// stop compiling or stop being about the same thing.
pub(crate) fn recorder_for_test(
    identity: DeviceIdentity,
    facts: CapabilityFacts,
    lanes: SubmissionCapabilities,
) -> crate::api::command::CommandRecorder {
    let native = MockDevice::with_capabilities(
        BackendKind::Dx12,
        MockProvider::new(BackendKind::Dx12, DeviceInstanceId::new(1)).adapter(),
        facts,
        lanes,
    );
    let device = Device::new(identity, observed_backend(native))
        .expect("the caller states a lane set that satisfies section 10's base guarantee");
    device
        .create_recorder(&crate::api::command::RecorderDescriptor::new())
        .expect("a live device creates a recorder")
}

/// A recorder over a mock device that reports nothing and therefore refuses every
/// device-gated verb.
///
/// The device-gated verbs are exact: a device with no enabled feature and no
/// recorded route answers `Unsupported` to each of them, which is the behaviour
/// tests of that refusal want and is a *steadier* fixture than the panics it
/// replaces.
pub(crate) fn recorder_without_facts_for_test(
    identity: DeviceIdentity,
) -> crate::api::command::CommandRecorder {
    recorder_for_test(identity, CapabilityFacts::empty(), default_lanes())
}
