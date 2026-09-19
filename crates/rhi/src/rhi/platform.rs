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

use super::capability::{AvailableCapabilities, EnabledCapabilities};
use super::format::{BufferSupportQuery, RouteQuery};
use super::presentation::{PresentationTarget, PresentationTargetCapabilities};
use super::shader::ShaderArtifact;

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
}

impl Device {
    /// Asks the device whether it accepts `artifact`.
    pub fn shader_acceptance(&self, artifact: &ShaderArtifact) -> super::shader::ArtifactAcceptance {
        self.capabilities().shader_acceptance(artifact)
    }
}
