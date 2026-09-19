//! Platform, provider, adapter, and device lifecycle for RHI API v1.
//!
//! This module owns rhi-design sections 3 to 7: opaque identity, the unified
//! error vocabulary, `PlatformProvider` / `AdapterInfo` / `DeviceRequest` /
//! `Device`, and the device-requirement vocabulary that selects between
//! available, required, and enabled capability facts.
//!
//! # What it is
//!
//! One `PlatformProvider` is one backend family. Several providers coexist in a
//! process, so several independent devices exist at once, and the portable
//! contract is that resources never cross between them:
//!
//! ```text
//! Device::clone()            -> the same DeviceIdentity
//! independent request_device -> a new DeviceIdentity and generation domain
//! device loss                -> terminal for that whole identity
//! retry after loss           -> a new identity/generation domain
//! ```
//!
//! Loss is never repaired by incrementing a field. A handle carrying a lost
//! identity answers `DeviceLost` through its own device and `WrongDevice`
//! through any other, and that check happens in this crate before any backend
//! object is touched (section 6.9).
//!
//! # What it deliberately does not own
//!
//! No native handle, pointer, LUID, `HWND`, `HINSTANCE`, `VkInstance`, `CAMetalLayer`,
//! JS `GPU` object, `HGLRC`, or browser context appears here or in any public
//! signature. A provider is created by platform/host integration and reaches the
//! portable surface only through [`PlatformProvider`], whose inner backend object
//! is private. Capability is never inferred from Rust trait presence, from
//! [`BackendKind`], or from the `AdapterInfo` snapshot; correctness depends only
//! on [`Device::capabilities`].

use core::fmt;
use std::sync::Arc;

use super::binding::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BindingLimitClass,
};
use super::capability::{AvailableCapabilities, EnabledCapabilities};
use super::command::{CommandRecorder, RecorderDescriptor};
use super::format::{BufferSupportQuery, RouteQuery, TextureSupportQuery};
use super::pipeline::{
    ComputeLimits, ComputePipeline, ComputePipelineDescriptor, InterfaceLimits, PipelineInterface,
    PipelineInterfaceDescriptor, RasterLimits, RasterPipeline, RasterPipelineDescriptor,
    validate_compute_descriptor, validate_pipeline_interface, validate_raster_descriptor,
};
use super::presentation::{
    ConfiguredPresentation, PresentReceiptId, PresentState, PresentationConfiguration,
    PresentationTarget, PresentationTargetCapabilities,
};
use super::resource::{
    Buffer, BufferDescriptor, BufferUploadDescriptor, Sampler, SamplerDescriptor, Texture,
    TextureDescriptor, TextureUploadDescriptor, TextureView, TextureViewDescriptor,
    UploadDescriptor, UploadJob,
};
use super::shader::{ArtifactAcceptance, ShaderArtifact, ShaderModule, ShaderStage};
use super::submission::{CompletionPoint, CompletionState, SubmissionPlan, SubmissionReceipt};

/// Opaque identity of one Fluxel logical device execution domain.
///
/// A caller may compare, hash, and print this token but cannot construct one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DeviceInstanceId(u64);

impl DeviceInstanceId {
    /// The underlying opaque value. It is stable only within this process.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Opaque generation within a Fluxel logical device execution domain.
///
/// Part of public identity, not a recovery counter. Nothing may increment it to
/// revive an old handle; a later `request_device()` returns a new identity
/// carrying a new generation instead.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DeviceGeneration(u64);

impl DeviceGeneration {
    /// The underlying opaque value. It is stable only within this process.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A complete Fluxel logical device execution domain.
///
/// P0 has no transparent recovery: loss is terminal for the whole identity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DeviceIdentity {
    instance: DeviceInstanceId,
    generation: DeviceGeneration,
}

impl DeviceIdentity {
    /// The instance half of this identity.
    pub fn instance(self) -> DeviceInstanceId {
        self.instance
    }

    /// The generation half of this identity.
    pub fn generation(self) -> DeviceGeneration {
        self.generation
    }
}

/// The in-process logical ID of an RHI object.
///
/// It is not a native handle. It is unique within the process, carries no
/// cross-process stability, and capture artifacts reassign their own local IDs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ObjectId(u64);

impl ObjectId {
    /// The underlying opaque value.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Optional human-readable label. Labels never participate in compatibility,
/// canonical hashing, or identity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Label(pub Option<String>);

impl Label {
    /// An absent label.
    pub fn none() -> Self {
        Self(None)
    }

    /// A label carrying `value`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(Some(value.into()))
    }

    /// The label text, if any.
    pub fn as_deref(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

mod mint {
    //! Process-wide opaque token minting.
    //!
    //! One monotonic counter per token domain is what makes "the pair
    //! (instance, generation) never repeats" and "ObjectId is unique within the
    //! process" true without a registry. A backend never mints a token itself:
    //! only these functions do, so no backend can construct an identity that
    //! another backend believes it owns.

    use core::sync::atomic::{AtomicU64, Ordering};

    use super::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, ObjectId};

    static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);
    static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
    static NEXT_OBJECT: AtomicU64 = AtomicU64::new(1);

    /// Mints a fresh device identity domain.
    pub(crate) fn next_identity() -> DeviceIdentity {
        let instance = DeviceInstanceId(NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed));
        let generation = DeviceGeneration(NEXT_GENERATION.fetch_add(1, Ordering::Relaxed));
        DeviceIdentity {
            instance,
            generation,
        }
    }

    /// Mints a fresh object id.
    pub(crate) fn next_object_id() -> ObjectId {
        ObjectId(NEXT_OBJECT.fetch_add(1, Ordering::Relaxed))
    }
}

/// Mints a fresh device identity domain.
///
/// Nothing outside a backend's own `request_device` path should call this: a
/// non-backend that can mint an identity can invent a domain no device owns,
/// which is exactly the opaque-identity invariant of design-rhi section 3.11.
/// It is reachable crate-internally only because the mock backend and the
/// contract suites need a domain to test against.
#[cfg(any(test, feature = "test-support"))]
pub(crate) use mint::next_identity;

pub(crate) use mint::next_object_id;

/// The unified portable error vocabulary.
///
/// Portable validation never defers to a driver, validation layer, or browser to
/// discover that a call was illegal: a refusal that this crate can detect is
/// returned here, before any backend side effect.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RhiErrorKind {
    /// A range, alignment, usage, state, or argument error.
    InvalidUsage,

    /// No adapter/device in this provider satisfies the request.
    NoSuitableAdapter,

    /// The device, format, or route does not support the requested operation.
    Unsupported,

    /// A binding, interface, or pipeline-interface mismatch.
    IncompatibleInterface,

    /// A required GPU happens-before dependency is missing.
    MissingDependency,

    /// An object belongs to a different device identity.
    WrongDevice,

    /// Allocation failed.
    OutOfMemory,

    /// The presentation target is outdated.
    TargetOutdated,

    /// The presentation target is lost.
    TargetLost,

    /// The device identity is lost. Loss is terminal.
    DeviceLost,

    /// The backend failed for a reason the portable contract does not classify.
    BackendFailure,
}

/// A portable structured error.
#[derive(Debug)]
pub struct RhiError {
    kind: RhiErrorKind,
    message: String,
    object: Option<ObjectId>,
    operation: Option<&'static str>,
}

impl RhiError {
    /// The classification of this failure.
    pub fn kind(&self) -> RhiErrorKind {
        self.kind
    }

    /// The human-readable detail. It never carries a native handle or pointer.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The object this failure is about, when one is known.
    pub fn object(&self) -> Option<ObjectId> {
        self.object
    }

    /// The portable operation that failed, when one is known.
    pub fn operation(&self) -> Option<&'static str> {
        self.operation
    }

    pub(crate) fn new(kind: RhiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            object: None,
            operation: None,
        }
    }

    pub(crate) fn invalid_usage(message: impl Into<String>) -> Self {
        Self::new(RhiErrorKind::InvalidUsage, message)
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::new(RhiErrorKind::Unsupported, message)
    }

    pub(crate) fn wrong_device(message: impl Into<String>) -> Self {
        Self::new(RhiErrorKind::WrongDevice, message)
    }

    pub(crate) fn device_lost(message: impl Into<String>) -> Self {
        Self::new(RhiErrorKind::DeviceLost, message)
    }

    pub(crate) fn backend_failure(message: impl Into<String>) -> Self {
        Self::new(RhiErrorKind::BackendFailure, message)
    }

    pub(crate) fn at(mut self, operation: &'static str) -> Self {
        self.operation = Some(operation);
        self
    }

    pub(crate) fn on(mut self, object: ObjectId) -> Self {
        self.object = Some(object);
        self
    }
}

impl fmt::Display for RhiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)?;
        if let Some(operation) = self.operation {
            write!(f, " (in {operation})")?;
        }
        if let Some(object) = self.object {
            write!(f, " (object {})", object.as_u64())?;
        }
        Ok(())
    }
}

impl std::error::Error for RhiError {}

/// A portable RHI result.
pub type RhiResult<T> = Result<T, RhiError>;

/// The backend family of a provider or device.
///
/// It is used for diagnostics, selection and capture provenance, backend-specific
/// shader acceptance, and tooling UI. It never substitutes for a capability
/// query.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BackendKind {
    /// Direct3D 12.
    Dx12,
    /// Vulkan.
    Vulkan,
    /// Metal.
    Metal,
    /// Browser WebGPU.
    WebGpu,
    /// Desktop OpenGL.
    OpenGl,
    /// Browser WebGL2 (and the GLES profile it shares a family with).
    WebGl2,
}

/// An optional portable feature that a device may enable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptionalFeature {
    /// The compute pipeline and dispatch vocabulary.
    Compute,

    /// Anisotropic sampler filtering.
    ///
    /// WebGL2 needs an extension and Vulkan needs the corresponding device
    /// feature, so `max_anisotropy > 1` never implies this.
    SamplerAnisotropy,

    /// Fixed-length buffer/texture/sampler binding arrays.
    ///
    /// Runtime-sized, partially-bound, update-after-bind, and arbitrary
    /// descriptor indexing remain future extensions.
    BindingArrays,
}

/// A portable device limit.
///
/// `MaxFoo` limits are stronger when larger; `MinFooAlignment` limits are
/// stronger when smaller, which is why device requirements are stated with
/// [`LimitRequirement::AtLeast`] and [`LimitRequirement::AtMost`] rather than a
/// single "minimum".
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LimitKey {
    /// Largest buffer size in bytes.
    MaxBufferSize,

    /// Largest 1D texture dimension.
    MaxTexture1dDimension,
    /// Largest 2D texture dimension.
    MaxTexture2dDimension,
    /// Largest 3D texture dimension.
    MaxTexture3dDimension,
    /// Largest texture array layer count.
    MaxTextureArrayLayers,

    /// Largest number of bind groups in a pipeline interface.
    MaxBindGroups,
    /// Largest number of entries in one bind group layout.
    MaxBindingsPerGroup,
    /// Backends that constrain bind groups plus vertex buffers report it here.
    MaxBindGroupsPlusVertexBuffers,
    /// Largest visible uniform buffer binding range.
    MaxUniformBufferBindingSize,
    /// Largest visible storage buffer binding range.
    MaxStorageBufferBindingSize,
    /// Largest dynamic uniform buffer count in a pipeline interface.
    MaxDynamicUniformBuffersPerPipelineLayout,
    /// Largest dynamic storage buffer count in a pipeline interface.
    MaxDynamicStorageBuffersPerPipelineLayout,

    /// Largest anisotropy value. Meaningful only with
    /// [`OptionalFeature::SamplerAnisotropy`].
    MaxSamplerAnisotropy,

    /// Largest color attachment count.
    MaxColorAttachments,
    /// Largest bytes per sample across all color attachments.
    MaxColorAttachmentBytesPerSample,

    /// Largest vertex buffer count.
    MaxVertexBuffers,
    /// Largest vertex attribute count.
    MaxVertexAttributes,
    /// Largest vertex buffer array stride.
    MaxVertexBufferArrayStride,
    /// Largest inter-stage shader variable count.
    MaxInterStageShaderVariables,

    /// Largest compute invocations per workgroup.
    MaxComputeInvocationsPerWorkgroup,
    /// Largest compute workgroup X dimension.
    MaxComputeWorkgroupSizeX,
    /// Largest compute workgroup Y dimension.
    MaxComputeWorkgroupSizeY,
    /// Largest compute workgroup Z dimension.
    MaxComputeWorkgroupSizeZ,
    /// Largest compute workgroup count per dimension.
    MaxComputeWorkgroupsPerDimension,
    /// Largest compute workgroup storage size in bytes.
    MaxComputeWorkgroupStorageSize,

    /// Smallest uniform buffer offset alignment.
    MinUniformBufferOffsetAlignment,
    /// Smallest storage buffer offset alignment.
    MinStorageBufferOffsetAlignment,
}

/// One requirement on a portable device limit.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LimitRequirement {
    /// For example `MaxBufferSize >= value`.
    AtLeast {
        /// The limit being constrained.
        key: LimitKey,
        /// The required lower bound.
        value: u64,
    },

    /// For example `MinUniformBufferOffsetAlignment <= value`.
    AtMost {
        /// The limit being constrained.
        key: LimitKey,
        /// The required upper bound.
        value: u64,
    },
}

/// What a device request requires and prefers.
///
/// The three layers stay distinct: this states requirements, [`AdapterInfo`]
/// reports availability, and [`Device::capabilities`] reports what the created
/// device actually enabled.
#[derive(Clone, Debug, Default)]
pub struct DeviceRequirements {
    required_features: Vec<OptionalFeature>,
    preferred_features: Vec<OptionalFeature>,
    limit_requirements: Vec<LimitRequirement>,
    required_buffers: Vec<BufferSupportQuery>,
    required_textures: Vec<super::format::TextureSupportQuery>,
    required_bindings: Vec<super::binding::BindingSupportQuery>,
    required_routes: Vec<RouteQuery>,
}

impl DeviceRequirements {
    /// No required or preferred facts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requires `feature`. Its absence fails the whole device request.
    pub fn require_feature(mut self, feature: OptionalFeature) -> Self {
        self.required_features.push(feature);
        self
    }

    /// Enables `feature` when the adapter has it. Absence does not fail.
    pub fn prefer_feature(mut self, feature: OptionalFeature) -> Self {
        self.preferred_features.push(feature);
        self
    }

    /// Requires `value` as a lower bound for `key`.
    pub fn require_limit_at_least(mut self, key: LimitKey, value: u64) -> Self {
        self.limit_requirements
            .push(LimitRequirement::AtLeast { key, value });
        self
    }

    /// Requires `value` as an upper bound for `key`.
    pub fn require_limit_at_most(mut self, key: LimitKey, value: u64) -> Self {
        self.limit_requirements
            .push(LimitRequirement::AtMost { key, value });
        self
    }

    /// Requires the complete buffer semantics named by `query`.
    pub fn require_buffer_support(mut self, query: BufferSupportQuery) -> Self {
        self.required_buffers.push(query);
        self
    }

    /// Requires the complete texture semantics named by `query`.
    pub fn require_texture_support(mut self, query: super::format::TextureSupportQuery) -> Self {
        self.required_textures.push(query);
        self
    }

    /// Requires the binding semantics named by `query`.
    pub fn require_binding_support(mut self, query: super::binding::BindingSupportQuery) -> Self {
        self.required_bindings.push(query);
        self
    }

    /// Requires the transfer route named by `query`.
    pub fn require_route(mut self, query: RouteQuery) -> Self {
        self.required_routes.push(query);
        self
    }

    /// The features whose absence fails the request.
    pub fn required_features(&self) -> &[OptionalFeature] {
        &self.required_features
    }

    /// The features enabled when available.
    pub fn preferred_features(&self) -> &[OptionalFeature] {
        &self.preferred_features
    }

    /// The stated limit requirements.
    pub fn limit_requirements(&self) -> &[LimitRequirement] {
        &self.limit_requirements
    }

    /// The required buffer semantics.
    pub fn required_buffers(&self) -> &[BufferSupportQuery] {
        &self.required_buffers
    }

    /// The required texture semantics.
    pub fn required_textures(&self) -> &[super::format::TextureSupportQuery] {
        &self.required_textures
    }

    /// The required binding semantics.
    pub fn required_bindings(&self) -> &[super::binding::BindingSupportQuery] {
        &self.required_bindings
    }

    /// The required transfer routes.
    pub fn required_routes(&self) -> &[RouteQuery] {
        &self.required_routes
    }

    /// Answers whether `available` satisfies every requirement recorded here.
    ///
    /// This is one lowering shared by every backend: availability is compared
    /// with the requirements once, so a backend never re-decides what a
    /// requirement means.
    pub(crate) fn satisfied_by(&self, available: &AvailableCapabilities) -> bool {
        self.required_features
            .iter()
            .all(|feature| available.supports_feature(*feature))
            && self.limit_requirements.iter().all(|requirement| {
                let (key, required, at_least) = match *requirement {
                    LimitRequirement::AtLeast { key, value } => (key, value, true),
                    LimitRequirement::AtMost { key, value } => (key, value, false),
                };
                match available.limit(key) {
                    Some(actual) if at_least => actual >= required,
                    Some(actual) => actual <= required,
                    None => false,
                }
            })
            && self
                .required_buffers
                .iter()
                .all(|query| available.buffer_support(query).is_supported())
            && self
                .required_textures
                .iter()
                .all(|query| available.texture_support(query).is_supported())
            && self
                .required_bindings
                .iter()
                .all(|query| available.binding_support(query).is_supported())
            && self
                .required_routes
                .iter()
                .all(|query| available.route(query).is_supported())
    }
}

/// How a provider should choose between its adapters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterSelection {
    /// The provider, OS, or browser selects its default candidate.
    Default,

    /// A preference for a high-performance adapter. Not a guarantee of a
    /// discrete GPU.
    PreferHighPerformance,

    /// A preference for a low-power adapter. Not a guarantee of an integrated
    /// GPU.
    PreferLowPower,

    /// An adapter returned by optional enumeration.
    Explicit(AdapterId),
}

/// Provider-scoped opaque adapter identity.
///
/// It is not an enumeration index, not a native pointer, LUID, or physical
/// device, and is guaranteed only to be passed back to the provider that
/// produced it. It carries no cross-process hardware identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdapterId {
    provider: u64,
    serial: u64,
}

impl AdapterId {
    /// The provider half of this identity.
    pub fn provider(self) -> u64 {
        self.provider
    }

    /// The adapter serial within that provider.
    pub fn serial(self) -> u64 {
        self.serial
    }

    pub(crate) fn new(provider: u64, serial: u64) -> Self {
        Self { provider, serial }
    }
}

/// A discovery snapshot for one adapter.
///
/// It reports what the adapter offers, not what a device enabled. It is not a
/// stable cross-run key and must not be used to infer capability.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    id: AdapterId,
    name: String,
    backend: BackendKind,
    vendor_id: Option<u32>,
    device_id: Option<u32>,
    available: AvailableCapabilities,
}

impl AdapterInfo {
    pub(crate) fn new(
        id: AdapterId,
        name: impl Into<String>,
        backend: BackendKind,
        vendor_id: Option<u32>,
        device_id: Option<u32>,
        available: AvailableCapabilities,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            backend,
            vendor_id,
            device_id,
            available,
        }
    }

    /// This adapter's provider-scoped identity.
    pub fn id(&self) -> AdapterId {
        self.id
    }

    /// The adapter's human-readable name, for diagnostics only.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The backend family this adapter belongs to.
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// The vendor id, when the provider can supply it safely.
    pub fn vendor_id(&self) -> Option<u32> {
        self.vendor_id
    }

    /// The device id, when the provider can supply it safely.
    pub fn device_id(&self) -> Option<u32> {
        self.device_id
    }

    /// The facts available on this adapter.
    pub fn available_capabilities(&self) -> &AvailableCapabilities {
        &self.available
    }
}

/// Targets and facts a device request must be able to serve.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DeviceRequestDescriptor {
    selection: AdapterSelection,
    requirements: DeviceRequirements,
    presentation_targets: Vec<PresentationTarget>,
}

impl DeviceRequestDescriptor {
    /// A request for `selection` satisfying `requirements`.
    pub fn new(selection: AdapterSelection, requirements: DeviceRequirements) -> Self {
        Self {
            selection,
            requirements,
            presentation_targets: Vec::new(),
        }
    }

    /// Requires the final device to have a portable presentation route for
    /// `target`.
    ///
    /// Preflighting an adapter is not enough on its own: the queue family,
    /// execution/presentation route, and backend presentation support must be
    /// selected during device creation, so a target that needs presentation
    /// enters this request.
    pub fn require_presentation_target(mut self, target: PresentationTarget) -> Self {
        self.presentation_targets.push(target);
        self
    }

    /// The requested adapter selection.
    pub fn selection(&self) -> AdapterSelection {
        self.selection
    }

    /// The stated requirements.
    pub fn requirements(&self) -> &DeviceRequirements {
        &self.requirements
    }

    /// The presentation targets the device must serve.
    pub fn presentation_targets(&self) -> &[PresentationTarget] {
        &self.presentation_targets
    }
}

/// The observation of a request that may still be pending.
#[derive(Debug)]
pub enum RequestStatus<T> {
    /// The request has not produced its result yet.
    Pending,

    /// The request produced its result.
    Ready(T),
}

/// A runtime-agnostic, single-shot device request.
///
/// RHI binds to no async runtime: there is no `Tokio`, `async-std`,
/// `async_trait`, or JS `Promise` ABI here. The host keeps pumping its own loop
/// and calls [`DeviceRequest::poll`], which advances only RHI bookkeeping.
pub struct DeviceRequest {
    backend: Option<Box<dyn DeviceRequestBackend>>,
    completed: bool,
}

impl DeviceRequest {
    /// Non-blockingly observes or advances this request.
    ///
    /// It never runs a browser or OS event loop. After the first
    /// [`RequestStatus::Ready`] or terminal error the request is complete and a
    /// later call returns [`RhiErrorKind::InvalidUsage`].
    pub fn poll(&mut self) -> RhiResult<RequestStatus<Device>> {
        if self.completed {
            return Err(RhiError::invalid_usage(
                "device request already produced its result",
            )
            .at("DeviceRequest::poll"));
        }
        let backend = self
            .backend
            .as_mut()
            .ok_or_else(|| RhiError::invalid_usage("device request has no backend").at("poll"))?;
        match backend.poll()? {
            RequestStatus::Pending => Ok(RequestStatus::Pending),
            RequestStatus::Ready(device) => {
                self.completed = true;
                self.backend = None;
                Ok(RequestStatus::Ready(device))
            }
        }
    }

    pub(crate) fn new(backend: Box<dyn DeviceRequestBackend>) -> Self {
        Self {
            backend: Some(backend),
            completed: false,
        }
    }

    /// A request that is already resolved.
    pub(crate) fn ready(device: Device) -> Self {
        Self {
            backend: Some(Box::new(ReadyRequest { device: Some(device) })),
            completed: false,
        }
    }
}

impl fmt::Debug for DeviceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceRequest")
            .field("completed", &self.completed)
            .finish_non_exhaustive()
    }
}

struct ReadyRequest {
    device: Option<Device>,
}

impl DeviceRequestBackend for ReadyRequest {
    fn poll(&mut self) -> RhiResult<RequestStatus<Device>> {
        let device = self
            .device
            .take()
            .ok_or_else(|| RhiError::invalid_usage("ready request already consumed"))?;
        Ok(RequestStatus::Ready(device))
    }
}

/// The lifecycle status of a device identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceStatus {
    /// The identity is usable.
    Active,
    /// The identity is terminally lost.
    Lost,
}

/// A stable summary of why a device identity was lost.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceLossInfo {
    message: String,
}

impl DeviceLossInfo {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The human-readable loss summary.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// A shared logical device execution domain.
///
/// Cloning a `Device` yields the same [`DeviceIdentity`]. Dropping the last
/// public handle does not destroy the domain: created objects, recorded work,
/// and completions hold enough internal shared ownership to reach GPU-safe
/// retirement, and a normal `Drop` never requires the caller to synchronize
/// `wait_idle()`.
#[derive(Clone)]
pub struct Device {
    inner: Arc<dyn DeviceBackend>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("backend", &self.inner.backend())
            .field("identity", &self.inner.identity())
            .finish_non_exhaustive()
    }
}

impl Device {
    pub(crate) fn new(inner: Arc<dyn DeviceBackend>) -> Self {
        Self { inner }
    }

    pub(crate) fn backend_ref(&self) -> &dyn DeviceBackend {
        self.inner.as_ref()
    }

    /// This domain's identity.
    pub fn identity(&self) -> DeviceIdentity {
        self.inner.identity()
    }

    /// The backend family that owns this domain.
    pub fn backend(&self) -> BackendKind {
        self.inner.backend()
    }

    /// The adapter actually selected, when creation succeeded.
    pub fn adapter_info(&self) -> &AdapterInfo {
        self.inner.adapter_info()
    }

    /// The facts this device enabled. This is the only correctness source.
    pub fn capabilities(&self) -> &EnabledCapabilities {
        self.inner.capabilities()
    }

    /// Whether this identity is active or terminally lost.
    pub fn status(&self) -> DeviceStatus {
        self.inner.status()
    }

    /// The loss summary, absent while the identity is active.
    pub fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.inner.loss_info()
    }

    /// Non-blockingly processes RHI-observable completion, loss, callbacks, and
    /// backend bookkeeping.
    ///
    /// It is not a host or browser event-loop pump.
    pub fn poll(&self) -> RhiResult<()> {
        self.inner.poll()
    }

    /// Blocks until this device's accepted work has completed.
    ///
    /// For shutdown and diagnostics only. It is never a per-frame retirement
    /// mechanism, and a restricted host or backend may return
    /// [`RhiErrorKind::Unsupported`].
    pub fn wait_idle(&self) -> RhiResult<()> {
        self.inner.wait_idle()
    }

    /// The presentation facts of `target` under this device.
    ///
    /// The answer is a snapshot: resize, display move, host context recreation,
    /// compositor change, or surface loss may make it stale, so configuration
    /// validates again.
    pub fn presentation_capabilities(
        &self,
        target: &PresentationTarget,
    ) -> RhiResult<PresentationTargetCapabilities> {
        self.require_active("Device::presentation_capabilities")?;
        self.inner.presentation_capabilities(target)
    }

    /// Creates a buffer.
    ///
    /// The descriptor is validated and the usage set queried against this
    /// device's enabled facts *before* a backend sees it, so "the device
    /// refuses this" is one structured answer produced in one place rather than
    /// a different answer from each backend. A backend is handed a descriptor
    /// that has already passed, which is why it may lower it without repeating
    /// the portable checks.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Buffer> {
        const OP: &str = "Device::create_buffer";
        self.require_active(OP)?;
        descriptor.normalize_and_validate().map_err(|e| e.at(OP))?;
        let support = self.capabilities().buffer_support(&BufferSupportQuery::new(descriptor.usage));
        let Some(limits) = support.limits() else {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not support the requested buffer usage set",
            )
            .at(OP));
        };
        if descriptor.size > limits.max_size() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "buffer size exceeds the device's supported maximum for this usage set",
            )
            .at(OP));
        }
        self.inner.create_buffer(descriptor)
    }

    /// Creates a texture.
    pub fn create_texture(&self, descriptor: &TextureDescriptor) -> RhiResult<Texture> {
        const OP: &str = "Device::create_texture";
        self.require_active(OP)?;
        // Canonicalization comes first because validation is written against
        // the canonical form: `view_formats` is a set, and an un-canonicalized
        // list is not an identity for this texture.
        let canonical = descriptor.canonicalized();
        canonical.normalize_and_validate().map_err(|e| e.at(OP))?;
        let mut query = TextureSupportQuery::new(
            canonical.dimension,
            canonical.format,
            canonical.usage,
            canonical.sample_count,
        )
        .with_view_compatibility(canonical.view_compatibility);
        for format in &canonical.view_formats {
            query = query.with_view_format(*format);
        }
        if !self.capabilities().texture_support(&query).is_supported() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not support the requested texture shape",
            )
            .at(OP));
        }
        self.inner.create_texture(&canonical)
    }

    /// Creates a view onto `texture`.
    pub fn create_texture_view(
        &self,
        texture: &Texture,
        descriptor: &TextureViewDescriptor,
    ) -> RhiResult<TextureView> {
        const OP: &str = "Device::create_texture_view";
        self.require_active(OP)?;
        if texture.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "texture belongs to another device identity",
            )
            .at(OP)
            .on(texture.id()));
        }
        descriptor.validate_for(texture).map_err(|e| e.at(OP))?;
        let format = descriptor.resolved_format(texture);
        if !self
            .capabilities()
            .texture_view_format_compatible(texture.descriptor().format, format)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not permit the requested view format",
            )
            .at(OP));
        }
        self.inner.create_texture_view(texture, descriptor)
    }

    /// Creates a sampler.
    pub fn create_sampler(&self, descriptor: &SamplerDescriptor) -> RhiResult<Sampler> {
        const OP: &str = "Device::create_sampler";
        self.require_active(OP)?;
        descriptor.normalize_and_validate().map_err(|e| e.at(OP))?;
        self.inner.create_sampler(descriptor)
    }

    /// Creates a retained host-to-buffer upload.
    pub fn create_buffer_upload(
        &self,
        descriptor: BufferUploadDescriptor,
    ) -> RhiResult<UploadJob> {
        const OP: &str = "Device::create_buffer_upload";
        self.require_active(OP)?;
        descriptor.validate(self.identity()).map_err(|e| e.at(OP))?;
        self.inner
            .create_upload(UploadDescriptor::Buffer(descriptor))
    }

    /// Creates a retained host-to-texture upload.
    pub fn create_texture_upload(
        &self,
        descriptor: TextureUploadDescriptor,
    ) -> RhiResult<UploadJob> {
        const OP: &str = "Device::create_texture_upload";
        self.require_active(OP)?;
        descriptor.validate(self.identity()).map_err(|e| e.at(OP))?;
        self.inner
            .create_upload(UploadDescriptor::Texture(descriptor))
    }

    /// Creates a shader module from a portable artifact.
    pub fn create_shader(&self, artifact: &ShaderArtifact) -> RhiResult<ShaderModule> {
        const OP: &str = "Device::create_shader";
        self.require_active(OP)?;
        let acceptance = self.shader_acceptance(artifact);
        if acceptance != ArtifactAcceptance::Accepted {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!("this device does not accept the shader artifact: {acceptance:?}"),
            )
            .at(OP));
        }
        self.inner.create_shader(artifact)
    }

    /// Creates a bind group layout.
    pub fn create_bind_group_layout(
        &self,
        descriptor: &BindGroupLayoutDescriptor,
    ) -> RhiResult<BindGroupLayout> {
        const OP: &str = "Device::create_bind_group_layout";
        self.require_active(OP)?;
        let canonical = descriptor.canonicalized();
        let max_bindings = self
            .capabilities()
            .limit(LimitKey::MaxBindingsPerGroup)
            .unwrap_or(u64::MAX);
        super::binding::validate_layout_entries(&canonical.entries, max_bindings)
            .map_err(|e| e.at(OP))?;
        self.inner.create_bind_group_layout(&canonical)
    }

    /// Creates a bind group over `descriptor`'s layout.
    pub fn create_bind_group(&self, descriptor: &BindGroupDescriptor) -> RhiResult<BindGroup> {
        const OP: &str = "Device::create_bind_group";
        self.require_active(OP)?;
        let layout = descriptor.layout.clone();
        if layout.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "bind group layout belongs to another device identity",
            )
            .at(OP)
            .on(layout.id()));
        }
        let canonical = descriptor.canonicalized();
        // Every bound resource is checked against this identity here rather
        // than in each backend: a packet naming a buffer from another device
        // must fail in the library, not in the driver.
        for entry in &canonical.entries {
            for id in entry.resource.device_objects() {
                let owner = id.1;
                if owner != self.identity() {
                    return Err(RhiError::new(
                        RhiErrorKind::WrongDevice,
                        "bind group entry names a resource from another device identity",
                    )
                    .at(OP)
                    .on(id.0));
                }
            }
        }
        super::binding::validate_bind_group_entries(
            &canonical.entries,
            &layout.descriptor().entries,
        )
        .map_err(|e| e.at(OP))?;
        self.inner.create_bind_group(&canonical)
    }

    /// Creates a pipeline interface.
    ///
    /// The portable validation runs here, before the backend is asked to lower
    /// anything, so a descriptor that mixes devices or exceeds a declared limit
    /// is refused by this device rather than by whichever backend happens to
    /// re-check it. Section 4 of the device chapter states the rule: a problem
    /// portable validation can find may not be handed to a driver to discover.
    pub fn create_pipeline_interface(
        &self,
        descriptor: &PipelineInterfaceDescriptor,
    ) -> RhiResult<PipelineInterface> {
        const OP: &str = "Device::create_pipeline_interface";
        self.require_active(OP)?;
        let capabilities = self.capabilities();
        let binding_limit =
            |stage: ShaderStage, class: BindingLimitClass| capabilities.binding_limit(stage, class);
        validate_pipeline_interface(
            self.identity(),
            descriptor,
            InterfaceLimits::from_capabilities(capabilities),
            &binding_limit,
        )
        .map_err(|e| e.at(OP))?;
        self.inner.create_pipeline_interface(descriptor)
    }

    /// Creates a raster pipeline.
    ///
    /// See [`Self::create_pipeline_interface`] for why the portable validation
    /// is here. The merge outcomes this validation derives are a lowering input
    /// rather than a legality decision, so the backend derives its own copy.
    pub fn create_raster_pipeline(
        &self,
        descriptor: &RasterPipelineDescriptor,
    ) -> RhiResult<RasterPipeline> {
        const OP: &str = "Device::create_raster_pipeline";
        self.require_active(OP)?;
        let capabilities = self.capabilities();
        let binding_limit =
            |stage: ShaderStage, class: BindingLimitClass| capabilities.binding_limit(stage, class);
        let limit = |key: LimitKey| capabilities.limit(key);
        validate_raster_descriptor(
            self.identity(),
            descriptor,
            RasterLimits::from_capabilities(capabilities),
            &binding_limit,
            &limit,
        )
        .map_err(|e| e.at(OP))?;
        self.inner.create_raster_pipeline(descriptor)
    }

    /// Creates a compute pipeline.
    ///
    /// See [`Self::create_pipeline_interface`] for why the portable validation
    /// is here.
    pub fn create_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> RhiResult<ComputePipeline> {
        const OP: &str = "Device::create_compute_pipeline";
        self.require_active(OP)?;
        let capabilities = self.capabilities();
        validate_compute_descriptor(
            self.identity(),
            descriptor,
            ComputeLimits::from_capabilities(capabilities),
            capabilities.supports_feature(OptionalFeature::Compute),
        )
        .map_err(|e| e.at(OP))?;
        self.inner.create_compute_pipeline(descriptor)
    }

    /// Creates a command recorder.
    pub fn create_recorder(
        &self,
        descriptor: &RecorderDescriptor,
    ) -> RhiResult<CommandRecorder> {
        const OP: &str = "Device::create_recorder";
        self.require_active(OP)?;
        self.inner.create_recorder(descriptor)
    }

    /// Submits a validated plan.
    ///
    /// The plan was validated when it was built, against this device; the
    /// identity check here is what makes that validation mean something, since
    /// a plan built against another device is not a plan for this one.
    pub fn submit(&self, plan: SubmissionPlan) -> RhiResult<SubmissionReceipt> {
        const OP: &str = "Device::submit";
        self.require_active(OP)?;
        if plan.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "submission plan was validated against another device identity",
            )
            .at(OP));
        }
        self.inner.submit(plan)
    }

    /// Reads the completion state of `point`.
    pub fn completion_state(&self, point: CompletionPoint) -> RhiResult<CompletionState> {
        const OP: &str = "Device::completion_state";
        if point.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "completion point belongs to another device identity",
            )
            .at(OP));
        }
        // Deliberately not gated on an active identity: the state of work
        // accepted before a loss is exactly what a caller needs after one.
        self.inner.completion_state(point)
    }

    /// Configures `target` for presentation under this device.
    pub fn configure_presentation(
        &self,
        target: &PresentationTarget,
        configuration: &PresentationConfiguration,
    ) -> RhiResult<ConfiguredPresentation> {
        const OP: &str = "Device::configure_presentation";
        self.require_active(OP)?;
        // A target is provider-scoped, not device-scoped: the same provider may
        // hand out two devices that both answer for one surface. Legality of
        // *this* pair is the question `presentation_capabilities` answers, and
        // configuration re-asks it, so no identity check belongs here.
        self.inner.configure_presentation(target, configuration)
    }

    /// Reads the display state of a present receipt.
    pub fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        const OP: &str = "Device::present_state";
        if receipt.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "present receipt belongs to another device identity",
            )
            .at(OP));
        }
        self.inner.present_state(receipt)
    }

    /// This identity's statistics domain.
    ///
    /// There is no global statistics singleton, and the returned handle is
    /// scoped to this identity, so two devices cannot be summed by accident.
    /// Like [`Self::drain_diagnostics`] this does not require an active
    /// identity: the numbers a device accumulated before it was lost are still
    /// the numbers it accumulated.
    pub fn statistics(&self) -> super::statistics::DeviceStatistics {
        self.inner.statistics().clone()
    }

    /// Moves every pending diagnostic into `out`, oldest first.
    ///
    /// Pull model, so RHI never imposes a callback threading policy on a host.
    /// It is safe to call on a lost identity: diagnostics emitted while the
    /// device was failing are exactly the ones worth reading, so this is one of
    /// the few operations that does not require an active identity.
    pub fn drain_diagnostics(&self, out: &mut Vec<super::diagnostics::DiagnosticEvent>) {
        self.inner.diagnostics().drain(out);
    }

    /// This device's tooling SPI surface.
    ///
    /// The returned access is device-scoped and versioned by
    /// [`ToolingSpiVersion`], not by the crate's semver: it is a separately
    /// frozen seam and is `#[doc(hidden)]`. It deliberately does not require an
    /// active identity — a capture coordinator must still be able to describe
    /// what it saw and drain terminal observation after the identity is lost.
    ///
    /// A backend that implements no observer registry still returns an access;
    /// its `subscribe` answers [`RhiErrorKind::Unsupported`] rather than
    /// accepting an observer it would never call.
    ///
    /// [`ToolingSpiVersion`]: super::tooling::ToolingSpiVersion
    #[doc(hidden)]
    pub fn tooling(&self) -> super::tooling::ToolingAccess {
        super::tooling::ToolingAccess::new(self.identity(), self.inner.tooling())
    }

    /// Refuses with [`RhiErrorKind::DeviceLost`] once this identity is lost.
    ///
    /// Every public operation that would touch backend state calls this first,
    /// so a lost identity never reaches a native object whose value the driver
    /// may already have reused.
    pub(crate) fn require_active(&self, operation: &'static str) -> RhiResult<()> {
        match self.status() {
            DeviceStatus::Active => Ok(()),
            DeviceStatus::Lost => {
                let message = self
                    .loss_info()
                    .map(|info| info.message)
                    .unwrap_or_else(|| "device identity is lost".to_string());
                Err(RhiError::device_lost(message).at(operation))
            }
        }
    }
}

/// A portable, opaque device-creation entry point.
///
/// One provider is one backend family. It is created by platform/host
/// integration; the portable core exposes no native-handle constructor.
#[derive(Clone)]
pub struct PlatformProvider {
    inner: Arc<dyn ProviderBackend>,
}

impl fmt::Debug for PlatformProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlatformProvider")
            .field("backend", &self.inner.backend())
            .finish_non_exhaustive()
    }
}

impl PlatformProvider {
    pub(crate) fn new(inner: Arc<dyn ProviderBackend>) -> Self {
        Self { inner }
    }

    /// The backend family this provider serves.
    pub fn backend(&self) -> BackendKind {
        self.inner.backend()
    }

    /// Attempts portable adapter enumeration.
    ///
    /// `Ok(Some(list))` means the provider enumerates and the list is its
    /// candidates. `Ok(None)` means the provider does not expose portable
    /// enumeration, which is legitimate for WebGPU and adopted-context
    /// providers. `Ok(Some(vec![]))` means the provider enumerates and
    /// currently has no candidate. `Err` means enumeration itself failed.
    ///
    /// It is never a prerequisite for [`PlatformProvider::request_device`].
    pub fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        self.inner.enumerate_adapters()
    }

    /// Preflights whether `adapter` can present to `target`.
    ///
    /// For adapter pickers and diagnostics only. A device that needs
    /// presentation must still name the target in
    /// [`DeviceRequestDescriptor::require_presentation_target`].
    pub fn supports_presentation(
        &self,
        adapter: AdapterId,
        target: &PresentationTarget,
    ) -> RhiResult<bool> {
        self.inner.supports_presentation(adapter, target)
    }

    /// The canonical device-creation path.
    pub fn request_device(&self, desc: DeviceRequestDescriptor) -> RhiResult<DeviceRequest> {
        self.inner.request_device(desc)
    }
}

/// The backend half of a [`PlatformProvider`].
pub(crate) trait ProviderBackend: Send + Sync + 'static {
    /// The backend family this provider serves.
    fn backend(&self) -> BackendKind;

    /// Portable adapter enumeration, with the `None`/empty distinction intact.
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>>;

    /// Presentation preflight for one enumerated adapter.
    fn supports_presentation(
        &self,
        adapter: AdapterId,
        target: &PresentationTarget,
    ) -> RhiResult<bool>;

    /// Starts a device request.
    fn request_device(&self, desc: DeviceRequestDescriptor) -> RhiResult<DeviceRequest>;
}

/// The backend half of a [`DeviceRequest`].
pub(crate) trait DeviceRequestBackend: Send + 'static {
    /// Advances the request by one non-blocking step.
    fn poll(&mut self) -> RhiResult<RequestStatus<Device>>;
}

/// The backend half of a [`Device`].
pub(crate) trait DeviceBackend: Send + Sync + 'static {
    /// This domain's identity.
    fn identity(&self) -> DeviceIdentity;

    /// The backend family that owns this domain.
    fn backend(&self) -> BackendKind;

    /// The adapter actually selected.
    fn adapter_info(&self) -> &AdapterInfo;

    /// The facts this device enabled.
    fn capabilities(&self) -> &EnabledCapabilities;

    /// Whether this identity is active or terminally lost.
    fn status(&self) -> DeviceStatus;

    /// The loss summary, absent while active.
    fn loss_info(&self) -> Option<DeviceLossInfo>;

    /// Non-blocking bookkeeping advance.
    fn poll(&self) -> RhiResult<()>;

    /// Blocks until accepted work completes, or refuses.
    fn wait_idle(&self) -> RhiResult<()>;

    /// Presentation facts for `target`.
    fn presentation_capabilities(
        &self,
        target: &PresentationTarget,
    ) -> RhiResult<PresentationTargetCapabilities>;

    // Every `create_*` below receives a descriptor that has already been
    // canonicalized, validated, and checked against this device's enabled
    // facts by [`Device`]. A backend therefore lowers what it is given rather
    // than re-deciding portable legality: two backends disagreeing about what
    // is legal is a portable-contract bug, and the only way to make that
    // impossible is to have exactly one place that decides.

    /// Creates a buffer.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Buffer>;

    /// Creates a texture.
    fn create_texture(&self, descriptor: &TextureDescriptor) -> RhiResult<Texture>;

    /// Creates a view onto `texture`.
    fn create_texture_view(
        &self,
        texture: &Texture,
        descriptor: &TextureViewDescriptor,
    ) -> RhiResult<TextureView>;

    /// Creates a sampler.
    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> RhiResult<Sampler>;

    /// Creates a retained upload job.
    ///
    /// One entry point rather than two, because a backend retains the payload
    /// the same way for both forms and the two would otherwise differ only in
    /// which variant they wrap.
    fn create_upload(&self, descriptor: UploadDescriptor) -> RhiResult<UploadJob>;

    /// Creates a shader module.
    fn create_shader(&self, artifact: &ShaderArtifact) -> RhiResult<ShaderModule>;

    /// Creates a bind group layout.
    fn create_bind_group_layout(
        &self,
        descriptor: &BindGroupLayoutDescriptor,
    ) -> RhiResult<BindGroupLayout>;

    /// Creates a bind group.
    fn create_bind_group(&self, descriptor: &BindGroupDescriptor) -> RhiResult<BindGroup>;

    /// Creates a pipeline interface.
    fn create_pipeline_interface(
        &self,
        descriptor: &PipelineInterfaceDescriptor,
    ) -> RhiResult<PipelineInterface>;

    /// Creates a raster pipeline.
    fn create_raster_pipeline(
        &self,
        descriptor: &RasterPipelineDescriptor,
    ) -> RhiResult<RasterPipeline>;

    /// Creates a compute pipeline.
    fn create_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> RhiResult<ComputePipeline>;

    /// Creates a command recorder.
    fn create_recorder(&self, descriptor: &RecorderDescriptor) -> RhiResult<CommandRecorder>;

    /// Accepts a plan for execution.
    fn submit(&self, plan: SubmissionPlan) -> RhiResult<SubmissionReceipt>;

    /// Reads the completion state of `point`.
    fn completion_state(&self, point: CompletionPoint) -> RhiResult<CompletionState>;

    /// Configures `target` for presentation.
    fn configure_presentation(
        &self,
        target: &PresentationTarget,
        configuration: &PresentationConfiguration,
    ) -> RhiResult<ConfiguredPresentation>;

    /// Reads the display state of `receipt`.
    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState>;

    /// This device's diagnostic log.
    ///
    /// The log is owned by the device rather than kept in a process-wide
    /// registry so that diagnostics from a lost identity can never appear in a
    /// replacement identity's stream.
    fn diagnostics(&self) -> &super::diagnostics::DiagnosticLog;

    /// This identity's statistics domain.
    ///
    /// Like the diagnostic log, the domain belongs to the device rather than to
    /// a process-wide singleton: two identites must never be summed into one
    /// counter, and a replacement identity must not inherit its predecessor's
    /// numbers. The backend constructs it once, with this identity, and the
    /// device hands out clones.
    fn statistics(&self) -> &super::statistics::DeviceStatistics;

    /// This device's tooling backend, when this backend implements the seam.
    ///
    /// The default answers `None`. A backend that has not implemented the
    /// tooling SPI still satisfies the device contract, and
    /// [`Device::tooling`] reports `Unsupported` for the operations that need
    /// one rather than pretending to observe something.
    fn tooling(&self) -> Option<Arc<dyn super::tooling::ToolingBackend>> {
        None
    }
}

impl Device {
    /// Asks the device whether it accepts `artifact`.
    pub fn shader_acceptance(&self, artifact: &ShaderArtifact) -> super::shader::ArtifactAcceptance {
        self.capabilities().shader_acceptance(artifact)
    }
}

#[cfg(test)]
mod tests;
