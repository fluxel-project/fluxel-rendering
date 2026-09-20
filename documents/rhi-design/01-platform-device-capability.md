# RHI API freeze v13. Platform, device, and capability

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and this module in full before implementation. No other
> document may redefine the interfaces in this module.

> Status: **RHI API freeze v13 — normative for the 0.16 implementation target**
> Scope: Fluxel RHI’s portable public Rust API and Capture/Replay’s mandatory capability requirements for RHI.
> Baseline: DX12/Vulkan/Metal/WebGPU/OpenGL/WebGL2.
> No freezing: native lowering, backend internal objects, ABI, Capture file format, ReplayRuntime.

---

## 0. Final design principles

```text
RHI
   ↓ portable execution semantic
Backend
   ↓ native lowering
DX12 / Vulkan / Metal / WebGPU / GL
```

RHI 0.16 freezes the following principles:

1. **Capability is an instance fact of Device / Format / Surface / Route, not a Rust trait. **
2. **Base only guarantees at least one ordered submission lane. **Multiple lanes are capabilities; no real hardware overlap is promised.
3. **Ordinary public RHI does not expose barrier / fence / semaphore / native queue / descriptor heap / native heap. **
4. **Recorder generates actual `rhi::command::ResourceUse` from portable commands.** `ResourceUse` is RHI’s own execution vocabulary; RHI knows no rendering-scheduler contract.
5. **`BindGroup` is a logical validated resource packet and does not promise native descriptor object. **
6. **`FrameAttachment` is not equal to `TextureView`. ** GL/WebGL2 default framebuffer can only have attachment semantics.
7. **Present enters `SubmissionPlan` before submit. ** `PresentMode` belongs to the presentation configuration, not the frame-by-frame request parameter.
8. **submit accepted != GPU complete != present outcome. ** The three must be separated.
9. **Transient resource API is frozen now.** Every backend implements `Dedicated`; `Aliasing` is a later optimization without an API change.
10. **RHI must make portable semantics observable and reconstructable to support Capture/Replay; but RHI does not implement Artifact/ReplayRuntime. **
11. **Statistics is portable logical observability, not native profiler. ** RHI freezes the unified logical counting caliber and resource inventory/video memory estimation; native barrier, real queue engine, driver allocation, GPU timestamp, etc. are not disguised as portable facts.
12. **Async is only for operations that may await a future event, not for every concurrent operation.** Logical resource creation and recording remain synchronous.
13. **Do not pre-build empty capability trait/empty handle. ** Known long-lived capability such as Transient freezes its complete contract now; other capabilities wait for real semantics.

### 0.1 Async boundaries

Async operations are adapter/device discovery, shader and pipeline compilation, submission acceptance, GPU completion, readback readiness, presentation configuration/acquisition/abandonment/present outcome, and `wait_idle`. Capability queries, descriptor validation, logical Buffer/Texture/View/Sampler/BindGroup/Layout/PipelineInterface creation, command recording, statistics, and diagnostics remain synchronous.

> **Concurrency capability ≠ `async fn`.** A concurrent `create_buffer()` has no future event to await and must not be disguised as a Future.

Public inherent APIs use `async fn` and require no Tokio or async-std. `Device::poll()` remains a synchronous opportunistic progress hook; normal async correctness must use backend wake integration rather than a caller busy-loop.

---

# 1. Freeze range

## 1.1 FROZEN / P0

```text
Platform / Adapter / Device
Device identity
Capability / limit / format / route / surface facts
Buffer / Texture / TextureView / Sampler
Transient resource allocation contract
Upload / Readback
ShaderArtifact
BindGroupLayout / BindGroup / PipelineInterface
RasterPipeline / ComputePipeline
CommandRecorder / RasterScope / ComputeScope / Copy
RecordedWork / ResourceUse
Submission lanes / SubmissionPlan / Completion
PresentationTarget / Configuration / Acquire / Present outcome
Validation / diagnostics / labels / markers
Statistics / frame sampling / live inventory / logical memory estimate
RHI semantic observability required for Capture/Replay
```

## 1.2 DEFERRED

0.16 does not export the following empty APIs:

```text
Query / timestamp
Indirect / MultiDraw / Count
Bindless / descriptor indexing
General map / persistent mapping
Inline parameters / push constants
External memory / external sync
Pipeline cache persistence
Mesh / task / geometry / tessellation
Ray tracing
Sparse / tiled / residency
Device address
Work graphs
GPU-generated commands
Multi-GPU
XR custom present
Portable Capture Artifact format
ReplayRuntime
```

---

# 2. Module layout

```rust
pub mod rhi {
    pub mod platform;
    pub mod capability;
    pub mod format;
    pub mod resource {
        pub mod transient;
    }
    pub mod shader;
    pub mod binding;
    pub mod pipeline;
    pub mod command;
    pub mod submission;
    pub mod presentation;
    pub mod statistics;
    pub mod diagnostics;

/// Engine tooling SPI used by Capture / GPU debugger / trace.
    ///
/// The tooling SPI is versioned separately from the normal RHI public surface.
    #[doc(hidden)]
    pub mod tooling;
}
```

Backend：

```text
crate::backend::dx12
crate::backend::vulkan
crate::backend::metal
crate::backend::webgpu
crate::backend::gl
```

Backend-private objects must not be returned from the portable API.

---

# 3. Basic identity type

All IDs are **opaque token**. The caller can compare, hash, and print, but cannot construct any valid token by itself.

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceInstanceId(u64);

impl DeviceInstanceId {
    pub fn as_u64(self) -> u64 { self.0 }
}

/// A Fluxel logical Device execution domain.
///
/// P0 does not provide transparent device recovery:
/// Device loss is terminal; re-request_device() obtains a new identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceIdentity(DeviceInstanceId);

impl DeviceIdentity {
    pub fn instance(self) -> DeviceInstanceId { self.0 }
}

/// The in-process logical ID of the RHI object.
///
/// - not equal to native handle;
/// - Globally unique within the process;
/// - Cross-process stability is not guaranteed;
/// - Capture Artifact reassigns the capture-local typed ID.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ObjectId(u64);

impl ObjectId {
    pub fn as_u64(self) -> u64 { self.0 }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Label(pub Option<String>);
```

## 3.1 Multi-device isolation — P0 correctness invariant

The same process must allow multiple Providers/Devices to exist at the same time:

```text
DX12 Provider   -> Device A
Vulkan Provider -> Device B
Metal Provider  -> Device C
...
```

It also allows the creation of multiple independent logical devices with the same backend.

Freeze rules:

```text
Device::clone()
-> Same DeviceIdentity

Standalone request_device()
-> New DeviceIdentity

Device loss
-> The DeviceIdentity terminal lost

Retry request_device() after loss
-> New DeviceIdentity

```

P0 **None**:

```text
The same DeviceIdentity is automatically restored
Transparent replacement of a lost DeviceIdentity
Old resources are automatically migrated to the new Device
```

This way the public error model is simpler:

```text
target DeviceIdentity != object DeviceIdentity
    -> WrongDevice

The identity is the same but the Device is lost
    -> DeviceLost
```

All Device-owned objects must be bound to the `DeviceIdentity` they were created with:

```text
Buffer / Texture / TextureView / Sampler
ShaderModule
BindGroupLayout / BindGroup
PipelineInterface / RasterPipeline / ComputePipeline
CommandRecorder / RecordedWork
ReadbackTicket
SubmissionPlan / CompletionPoint / PresentReceipt
ConfiguredPresentation / AcquiredFrame
```

Any public operation must first perform O(1) identity validation before touching the backend.

For example:

```rust
let dx_buffer = dx12.create_buffer(...)?;
let vk_buffer = vk.create_buffer(...)?;
let mut vk_recorder = vk.create_recorder(...)?;

vk_recorder.copy_buffer(&BufferCopy {
    src: dx_buffer,
    dst: vk_buffer,
    // ...
})?;
// => Err(RhiErrorKind::WrongDevice)
```

You must not panic, and you must not hand over the DX12 native handle to the Vulkan backend.

## 3.2 Identity validation is always on

Identity check is a correctness requirement, not a diagnostics feature:

```text
Debug build -> must check
Release build -> must be checked
Statistics off -> must be checked
Diagnostics off -> must be checked
```

## 3.3 P0 does not support implicit cross-Device interop

When the normal API encounters cross-Device objects, it will always be `WrongDevice`.

Disable automatic:

```text
Cross-Device copy
Cross-Device resource binding
cross backend native-handle unwrap
CPU staging bridge
peer-GPU transfer
```

In the future, external memory / external sync / multi-GPU must be explicit extension families.

## 3.4 Public handle lifetime

`PlatformProvider`, `Device`, and Resource handle are all shared logical owners.

Freeze rules:

```text
Provider public handle drop
!= The created Device will expire immediately

Device public handle drop
!= Its resources/native domain will be destroyed immediately
```

Created objects/RecordedWork/Completion etc. must hold sufficient internal shared ownership,
Until GPU-safe retirement / terminal loss.

When the last internal owner is released, go to no-throw backend shutdown/reclaim;
The caller must not be required to synchronize `wait_idle()` in a normal `Drop`.

---


# 4. Error model

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhiErrorKind {
    InvalidUsage,

/// The current Provider cannot find the Adapter/Device that satisfies the request.
    NoSuitableAdapter,

    Unsupported,
    IncompatibleInterface,
    MissingDependency,
    WrongDevice,
    OutOfMemory,
    TargetOutdated,
    TargetLost,
    DeviceLost,
    BackendFailure,
}

#[derive(Debug)]
pub struct RhiError {
    kind: RhiErrorKind,
    message: String,
    object: Option<ObjectId>,
    operation: Option<&'static str>,
}

impl RhiError {
    pub fn kind(&self) -> RhiErrorKind { self.kind }
    pub fn message(&self) -> &str { &self.message }
    pub fn object(&self) -> Option<ObjectId> { self.object }
    pub fn operation(&self) -> Option<&'static str> { self.operation }
}

pub type RhiResult<T> = Result<T, RhiError>;
```

Require:

```text
There is no Adapter/Device that meets the requirements -> NoSuitableAdapter
unsupported device/format/route   -> Unsupported
Cross Device -> WrongDevice
binding/pipeline interface mismatch -> IncompatibleInterface
Missing GPU happens-before dependency -> MissingDependency
range/alignment/usage error -> InvalidUsage
OOM                               -> OutOfMemory
device lost                       -> DeviceLost
```

Problems that have been discovered by portable validation are not allowed to be deliberately sent to the backend and then relied on driver validation.

---

# 5. Platform / Adapter / Device creation

This chapter has entered **FROZEN/P0**.

Core model:

```text
Platform/Host integration
    ├─ DX12 PlatformProvider
    ├─ Vulkan PlatformProvider
    ├─ Metal PlatformProvider
    ├─ WebGPU PlatformProvider
    └─ GL / WebGL2 adopted-context PlatformProvider

canonical path:
    PlatformProvider
        -> request_device(DeviceRequestDescriptor).await
        -> Device

optional inspection path:
    PlatformProvider
        -> enumerate_adapters().await
        -> AdapterInfo / AdapterId
        -> AdapterSelection::Explicit
```

**A `PlatformProvider` corresponds to a backend family. **

So the same process can simultaneously:

```text
dx12_provider.request_device(...)  -> Device A
vk_provider.request_device(...)    -> Device B
```

The `DeviceIdentity` of the two Devices are different, and the resources cannot be used interchangeably.

---

## 5.1 Provider source

`PlatformProvider` is a portable opaque object, but portable core does not provide a native-handle constructor.

It is created by Fluxel host/platform integration.

For example:

```text
Windows host
-> Create DX12 Provider
-> Create Vulkan Provider

Apple host
-> Create Metal Provider

Browser host
-> Create WebGPU Provider
-> Create adopted WebGL2 Provider

Desktop GL host
-> Create adopted OpenGL Provider
```

portable RHI is not public:

```text
HWND
HINSTANCE
IDXGIAdapter native pointer
VkInstance / VkPhysicalDevice
CAMetalLayer
GPU / GPUAdapter JS object
HGLRC / EGLContext / WebGLRenderingContext
```

These only belong to the platform integration / backend-private seam.

---

## 5.2 BackendKind

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Dx12,
    Vulkan,
    Metal,
    WebGpu,
    OpenGl,
    WebGl2,
}
```

`BackendKind` is only used for:

```text
diagnostics
selection / capture provenance
backend-specific shader acceptance
tooling UI
```

It cannot be used in place of capability query.

mistake:

```rust
if device.backend() == BackendKind::Vulkan {
// Directly assume feature X exists
}
```

correct:

```rust
if device.capabilities().foo(...) {
    // ...
}
```

---

## 5.3 `PlatformProvider`

```rust
#[derive(Clone)]
pub struct PlatformProvider {
    /* opaque backend/provider state */
}

impl PlatformProvider {
    /// The backend family corresponding to this Provider.
    pub fn backend(&self) -> BackendKind;

    /// Attempts to enumerate the Adapters this Provider can expose explicitly.
    ///
    /// - Ok(Some(list)):
    ///     Provider supports portable enumeration.
    ///
    /// - Ok(None):
    ///     Provider does not expose portable enumeration.
    ///     WebGPU and adopted-context providers may legitimately do this.
    ///
    /// - Err(...):
    ///     Enumeration itself failed.
    ///
    /// enumerate_adapters() is not a prerequisite for request_device().
    pub async fn enumerate_adapters(
        &self,
    ) -> RhiResult<Option<Vec<AdapterInfo>>>;

    /// Performs presentation preflight for an enumerated Adapter.
    ///
    /// This is only for adapter pickers and diagnostics. Final Device creation
    /// must still include the target in DeviceRequestDescriptor when
    /// presentation is required.
    pub fn supports_presentation(
        &self,
        adapter: AdapterId,
        target: &PresentationTarget,
    ) -> RhiResult<bool>;

    /// Canonical Device creation path.
    pub async fn request_device(
        &self,
        desc: DeviceRequestDescriptor,
    ) -> RhiResult<Device>;
}
```

`None` and `Some(vec![])` are distinct:

```text
None
    = the Provider does not expose enumeration

Some(empty)
    = the Provider can enumerate, but currently has no candidate Adapter
```

## 5.4 Adapter identity

~~~rust
/// Provider-scoped Adapter identity.
///
/// Opaque; callers cannot construct it themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterId {
    /* provider identity + opaque serial */
}
~~~

Rules:

- AdapterId is not an enumeration index;
- it is not a native pointer / LUID / VkPhysicalDevice;
- it is guaranteed only to be passed back to **the same Provider that produced it**;
- passing Provider A's AdapterId to Provider B -> InvalidUsage;
- it is not a persistent cross-process hardware ID.

If a future pipeline cache / persistent adapter preference needs a hardware fingerprint, it is frozen separately and does not reuse AdapterId.

---

## 5.5 AdapterInfo

~~~rust
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    id: AdapterId,
    name: String,
    backend: BackendKind,

    /// Present only when the provider can safely provide it.
    vendor_id: Option<u32>,
    device_id: Option<u32>,

    /// AvailableOnAdapter, not what is ultimately enabled on Device.
    available: AvailableCapabilities,
}

impl AdapterInfo {
    pub fn id(&self) -> AdapterId;
    pub fn name(&self) -> &str;
    pub fn backend(&self) -> BackendKind;
    pub fn vendor_id(&self) -> Option<u32>;
    pub fn device_id(&self) -> Option<u32>;
    pub fn available_capabilities(&self) -> &AvailableCapabilities;
}
~~~

AdapterInfo is a discovery snapshot.

Do not use:

~~~text
enumeration index
name
vendor/device id
~~~

as a stable cross-run key.

---

## 5.6 Adapter selection

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterSelection {
    /// Provider / OS / browser selects the default candidate.
    Default,

    /// A preference; not a guarantee of a discrete GPU.
    PreferHighPerformance,

    /// A preference; not a guarantee of an integrated GPU.
    PreferLowPower,

    /// An explicit Adapter returned by optional enumeration.
    Explicit(AdapterId),
}
~~~

AdoptedContext is **not a selection variant**.

An adopted GL/WebGL2 context is a Provider source:

~~~text
host has already supplied a context
    -> host creates adopted-context provider
    -> provider.request_device(Default, ...)
~~~

It is not GPU-selection policy for every request.

---

## 5.7 Device requirements

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OptionalFeature {
    /// Compute pipeline / dispatch vocabulary.
    Compute,

    /// Sampler anisotropic filtering.
    ///
    /// WebGL2 requires an extension; Vulkan requires the corresponding feature;
    /// support cannot be inferred merely from max_anisotropy > 1.
    SamplerAnisotropy,

    /// Fixed-length Buffer / Texture / Sampler binding arrays.
    ///
    /// runtime-sized / partially-bound / update-after-bind / arbitrary indexing
    /// remain future bindless/indexing extensions.
    BindingArrays,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LimitKey {
    MaxBufferSize,

    MaxTexture1dDimension,
    MaxTexture2dDimension,
    MaxTexture3dDimension,
    MaxTextureArrayLayers,

    MaxBindGroups,
    MaxBindingsPerGroup,

    /// Some backends (for example WebGPU) also constrain bind groups + vertex buffers;
    /// backends without this constraint may return None.
    MaxBindGroupsPlusVertexBuffers,

    MaxUniformBufferBindingSize,
    MaxStorageBufferBindingSize,

    MaxDynamicUniformBuffersPerPipelineLayout,
    MaxDynamicStorageBuffersPerPipelineLayout,

    /// Meaningful only when OptionalFeature::SamplerAnisotropy is enabled.
    MaxSamplerAnisotropy,

    MaxColorAttachments,
    MaxColorAttachmentBytesPerSample,

    MaxVertexBuffers,
    MaxVertexAttributes,
    MaxVertexBufferArrayStride,
    MaxInterStageShaderVariables,

    MaxComputeInvocationsPerWorkgroup,
    MaxComputeWorkgroupSizeX,
    MaxComputeWorkgroupSizeY,
    MaxComputeWorkgroupSizeZ,
    MaxComputeWorkgroupsPerDimension,
    MaxComputeWorkgroupStorageSize,

    /// A smaller value for this kind of limit is more permissive, so a requirement
    /// cannot uniformly be called a minimum.
    MinUniformBufferOffsetAlignment,
    MinStorageBufferOffsetAlignment,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitRequirement {
    /// For example, MaxBufferSize >= value.
    AtLeast {
        key: LimitKey,
        value: u64,
    },

    /// For example, MinUniformBufferOffsetAlignment <= value.
    AtMost {
        key: LimitKey,
        value: u64,
    },
}

#[derive(Clone, Debug, Default)]
pub struct DeviceRequirements {
    required_features: Vec<OptionalFeature>,
    preferred_features: Vec<OptionalFeature>,
    limit_requirements: Vec<LimitRequirement>,

    /// Requires that the resulting Device can create these buffer semantics.
    required_buffers: Vec<BufferSupportQuery>,

    /// Requires that the resulting Device can create these texture semantics.
    required_textures: Vec<TextureSupportQuery>,

    /// Requires that the resulting Device can express these binding semantics.
    required_bindings: Vec<BindingSupportQuery>,

    /// Requires that the resulting Device has these transfer/resolve/blit routes.
    required_routes: Vec<RouteQuery>,
}

impl DeviceRequirements {
    pub fn new() -> Self;

    /// Absence makes the entire Device request fail.
    pub fn require_feature(
        self,
        feature: OptionalFeature,
    ) -> Self;

    /// Enable when possible; absence does not make the request fail.
    pub fn prefer_feature(
        self,
        feature: OptionalFeature,
    ) -> Self;

    pub fn require_limit_at_least(
        self,
        key: LimitKey,
        value: u64,
    ) -> Self;

    pub fn require_limit_at_most(
        self,
        key: LimitKey,
        value: u64,
    ) -> Self;

    pub fn require_buffer_support(
        self,
        query: BufferSupportQuery,
    ) -> Self;

    /// Requires complete texture semantics.
    ///
    /// Does not reuse an ambiguous “require format”:
    /// whether the same format is usable also depends on dimension / usage / sample_count
    /// and view-compatibility intent at creation.
    pub fn require_texture_support(
        self,
        query: TextureSupportQuery,
    ) -> Self;

    pub fn require_binding_support(
        self,
        query: BindingSupportQuery,
    ) -> Self;

    pub fn require_route(
        self,
        query: RouteQuery,
    ) -> Self;
}
~~~

The three layers must remain distinct:

~~~text
AvailableOnAdapter
        ↓
Required / Preferred
        ↓
EnabledOnDevice
~~~

Final correctness may depend only on:

~~~rust
device.capabilities()
~~~

---

## 5.8 Presentation requirement enters Device creation

Preflighting only:

~~~text
Can the Adapter present to the Surface?
~~~

is insufficient to guarantee that logical/native Device creation selected the correct:

~~~text
queue family
execution/presentation route
backend-specific presentation support
~~~

A target requiring presentation must therefore enter Device request.

~~~rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DeviceRequestDescriptor {
    selection: AdapterSelection,
    requirements: DeviceRequirements,

    /// Targets that the Device must be able to serve after creation.
    /// An empty list means headless / compute-only / offscreen-only.
    presentation_targets: Vec<PresentationTarget>,
}

impl DeviceRequestDescriptor {
    pub fn new(
        selection: AdapterSelection,
        requirements: DeviceRequirements,
    ) -> Self;

    /// Requires the final Device to have a portable presentation route for this target.
    ///
    /// Concrete Surface Facts such as format / present mode / extent
    /// are still queried by the Presentation chapter.
    pub fn require_presentation_target(
        self,
        target: PresentationTarget,
    ) -> Self;
}
~~~

If no candidate Adapter can satisfy the request:

~~~text
NoSuitableAdapter
~~~

If an explicitly selected Adapter lacks required feature/route/target support:

~~~text
Unsupported
~~~

---

## 5.9 Async cancellation

Dropping a `request_device().await` Future cancels receiving its result. A backend may safely complete or cancel its underlying OS/browser request, but must not leak a half-initialized Device. Cancellation does not promise to forcibly cancel an already-issued OS or browser device request.

---

# 6. Device

This chapter has entered **FROZEN/P0**.

```rust
#[derive(Clone)]
pub struct Device {
    /* opaque shared logical execution domain */
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceStatus {
    Active,
    Lost,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DeviceLossInfo {
    message: String,
}

impl DeviceLossInfo {
    pub fn message(&self) -> &str;
}
```

---

## 6.1 Device clone and independent Device

```text
Device::clone()
    = the same Fluxel Device domain
    = the same DeviceIdentity

A separate request_device()
    = a new Fluxel Device domain
    = a new DeviceInstanceId
```

therefore:

```rust
let a2 = a.clone();

assert_eq!(a.identity(), a2.identity());
```

but:

```rust
let dx = dx12_provider.request_device(...);
let vk = vulkan_provider.request_device(...);

assert_ne!(dx.identity(), vk.identity());
```

This is a portable contract.

Even if a backend happens to reuse a native device internally, this logical isolation cannot be changed.

---

## 6.2 Identity

```rust
impl Device {
    pub fn identity(&self) -> DeviceIdentity;
}
```

All Device-owned objects must be traceable to the same identity. A loss is
terminal for that identity; a later independent or retry `request_device()`
creates a new identity and never revives an old handle.

For unified verification rules, see **3.1 Multi-device isolation**.

---

## 6.3 Backend / Adapter provenance

```rust
impl Device {
    pub fn backend(&self) -> BackendKind;

    /// Snapshot of the Adapter actually selected.
    ///
    /// Even when Provider does not support enumerate_adapters(),
    /// Device should provide the actual selection whenever possible after creation succeeds.
    pub fn adapter_info(&self) -> &AdapterInfo;
}
```

This information is used to:

```text
diagnostics
UI
Capture provenance
benchmark report
```

Cannot be used to guess capabilities.

---

## 6.4 Enabled capabilities

```rust
impl Device {
    pub fn capabilities(&self) -> &EnabledCapabilities;
}
```

The conceptual relationship must be satisfied:

```text
EnabledOnDevice ⊆ AvailableOnAdapter
```

But portable caller only depends on `EnabledCapabilities`.

---

## 6.5 Status / loss

```rust
impl Device {
    pub fn status(&self) -> DeviceStatus;

    /// None while Active.
    ///
    /// Returns a stable loss summary after Lost.
    pub fn loss_info(&self) -> Option<DeviceLossInfo>;
}
```

v13 deliberately has **no** `Device::lost()` future, no `wait_lost()`, and no
separate public device-loss event. Loss is an execution-domain terminal state,
not a second event stream a caller must subscribe to or drain. `status()` and
`loss_info()` are synchronous, stable observations: once a Device reports
`Lost`, later calls on that same identity report the same terminal state and
loss summary.

After Device loss:

```text
Buffer / Texture / View / Sampler
Shader / BindGroup / Pipeline
Recorder / RecordedWork
CompletionPoint / ReadbackTicket
ConfiguredPresentation / AcquiredFrame / PresentReceipt
```

All enter the terminal domain of the lost DeviceIdentity.

Loss does not perform transparent recovery. A later Device request obtains a
new DeviceIdentity; old handles return `WrongDevice` when passed to it, and
`DeviceLost` when used through their lost original Device.

Loss must wake and terminate every operation that is already waiting on that
domain:

```text
pending CompletionPoint / wait_completion()  -> CompletionState::DeviceLost
pending ReadbackTicket::read()                -> DeviceLost error/state
pending surface.acquire()                     -> AcquireErrorKind::DeviceLost
pending PresentReceipt / wait_present()       -> PresentState::DeviceLost
wait_idle()                                   -> DeviceLost error
```

No such operation may remain pending forever after loss is observed. Subsequent
operations on the lost original Device return structured `DeviceLost` after the
normal ownership/identity checks; they do not reach a native backend merely to
rediscover loss.

An implementation is not required to poll solely to discover a loss when it has
no pending RHI operation. In that idle case it may first observe and publish the
loss on the next RHI call. This does not weaken the wakeup rule above: once loss
is observed, all registered pending operations must be released promptly. If a
real future use case needs proactive notification while completely idle, it may
add a separately designed `wait_lost` API then; v13 does not reserve or imply
one.

---

## 6.6 `Device::poll`

```rust
impl Device {
    /// Non-blockingly processes RHI-observable completion, loss, callbacks,
    /// and backend bookkeeping.
    ///
    /// This is not a host/browser event-loop pump.
    pub fn poll(&self) -> RhiResult<()>;
}
```

Normal async APIs do not require this erroneous assumption:

```text
while pending {
    device.poll();
}
```

And believe that this will definitely promote host async work on all platforms.

`poll()` is an opportunistic hook for hosts without an async executor,
explicit event-loop integration, and diagnostics. Correct async operation is:

```text
host main loop / runtime advances normally
        +
device.poll() non-blockingly publishes/consumes RHI bookkeeping
```

---

## 6.7 `wait_idle`

```rust
impl Device {
    /// For shutdown / diagnostics only.
    ///
    /// - Must not be used for per-frame retirement;
    /// - Must not become a normal render-loop correctness mechanism;
    /// - A restricted host/backend may return Unsupported.
    pub async fn wait_idle(&self) -> RhiResult<()>;
}
```

The frame-by-frame lifecycle can only be used with:

```text
SubmissionPoint
CompletionPoint
terminal completion / loss
```

---

## 6.8 Multi-device validation examples

### Resource creation / binding

```rust
let tex_a = device_a.create_texture(...)?;
let sampler_b = device_b.create_sampler(...)?;

device_a.create_bind_group(
    &BindGroupDescriptor::new(layout_a)
        .texture(slot0, tex_a.create_view(...)?)   // OK
        .sampler(slot1, sampler_b),                // WrongDevice
)?;
```

`create_bind_group()` must fail before entering backend descriptor creation.

### Pipeline

```text
Device A ShaderModule
+
Device B PipelineInterface
    -> Device A/B create_pipeline(...)
    -> WrongDevice
```

### Recording

```text
Device A CommandRecorder
+
Device B Buffer / Texture / Pipeline / BindGroup
    -> WrongDevice
```

### Submission

```text
Device A RecordedWork
+
Device B SubmissionPlan / submit()
    -> WrongDevice
```

### Presentation

```text
Device A AcquiredFrame
+
Device B SubmissionPlan
    -> WrongDevice
```

### Device loss + recreate

```text
old Device A Texture
+
newly requested Device B Recorder
    -> WrongDevice

Device A identity
+
transparent replacement after loss
    -> prohibited

old Device A Recorder after Device A lost
    -> DeviceLost
```

---

## 6.9 It is not allowed to replace RHI validation with "the bottom layer will report an error"

Portable contract explicitly requires:

```text
wrong device / device lost
    ↓
Fluxel RHI structured validation
    ↓
Err(WrongDevice | DeviceLost)
```

instead of:

```text
Pass an invalid handle to DX12 / Vulkan / Metal / WebGPU / GL
    ↓
let the validation layer / driver / browser handle it unpredictably
```

reason:

- native validation may not be enabled in the release environment;
- Different APIs fail in different ways when mixing error objects;
- Some errors may evolve into backend failure, device loss or even undefined behavior;
- Fluxel public API must give callers unified and testable error semantics.

Therefore identity validation is RHI's own correctness responsibility.

---

# 7. Capability schema

This chapter has entered **FROZEN/P0**.

Capability is not a "big hardware capability structure", but a fact database at different levels:

```text
AvailableOnAdapter
    ├─ portable feature availability
    ├─ available limits
    ├─ format facts
    ├─ texture creation support
    └─ route facts

EnabledOnDevice
    ├─ enabled portable features
    ├─ enabled limits
    ├─ format facts after feature enablement
    ├─ texture creation support
    ├─ route facts
    ├─ actual logical submission lanes
    └─ shader artifact acceptance

SurfaceFact(Device, PresentationTarget)
    └─ presentation formats / modes / extent
```

It is forbidden to degenerate into:

```text
HardwareCapabilities {
    supports_x: bool,
    supports_y: bool,
    supports_z: bool,
    ...
}
```

---

## 7.1 Capability compatibility identity / fingerprint

RenderGraph compiled plan requires a compatibility token without hash-collision correctness risk;
cache/tooling uses fingerprint instead.

```rust
/// Process-local exact capability-contract intern token.
///
/// Produced by RHI through process-wide interning of canonical EnabledCapabilities semantics.
/// A caller cannot construct it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityCompatibilityId(u64);

/// Cache / diagnostics / Capture provenance fingerprint.
///
/// Equal hashes cannot alone carry correctness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityFingerprint(pub [u8; 32]);
```

`CompiledGraph` correctness reuse uses:

```text
CapabilityCompatibilityId
+ resource/import/presentation contracts
```

Fingerprint is only used for:

```text
cache key candidate
logs
artifact provenance
```

The re-created Device can obtain the same compatibility id even if the capability facts are exactly the same;
But all Device-owned handles are still strictly isolated by `DeviceIdentity`.

---

## 7.2 Available vs Enabled

```rust
#[derive(Clone, Debug)]
pub struct AvailableCapabilities {
    /* opaque adapter/provider query database */
}

#[derive(Clone, Debug)]
pub struct EnabledCapabilities {
    /* opaque immutable device query database */
}
```

Both expose the same portable fact vocabulary:

```rust
impl AvailableCapabilities {
    pub fn supports_feature(
        &self,
        feature: OptionalFeature,
    ) -> bool;

    pub fn limit(
        &self,
        key: LimitKey,
    ) -> Option<u64>;

    pub fn format(
        &self,
        format: TextureFormat,
    ) -> Option<FormatFacts>;

    pub fn buffer_support(
        &self,
        query: &BufferSupportQuery,
    ) -> BufferSupport;

    pub fn texture_support(
        &self,
        query: &TextureSupportQuery,
    ) -> TextureSupport;

    pub fn binding_support(
        &self,
        query: &BindingSupportQuery,
    ) -> BindingSupport;

    /// Portable binding-count limit for one shader stage and resource class.
    ///
    /// None means the stage / resource class is inapplicable to the current capability contract.
    pub fn binding_limit(
        &self,
        stage: ShaderStage,
        class: BindingLimitClass,
    ) -> Option<u32>;

    pub fn route(
        &self,
        query: &RouteQuery,
    ) -> RouteSupport;
}

impl EnabledCapabilities {
    pub fn compatibility_id(&self) -> CapabilityCompatibilityId;
    pub fn fingerprint(&self) -> CapabilityFingerprint;

    pub fn supports_feature(
        &self,
        feature: OptionalFeature,
    ) -> bool;

    pub fn limit(
        &self,
        key: LimitKey,
    ) -> Option<u64>;

    pub fn format(
        &self,
        format: TextureFormat,
    ) -> Option<FormatFacts>;

    pub fn buffer_support(
        &self,
        query: &BufferSupportQuery,
    ) -> BufferSupport;

    pub fn texture_support(
        &self,
        query: &TextureSupportQuery,
    ) -> TextureSupport;

    pub fn binding_support(
        &self,
        query: &BindingSupportQuery,
    ) -> BindingSupport;

    /// Portable binding-count limit for one shader stage and resource class.
    ///
    /// None means the stage / resource class is inapplicable to the current capability contract.
    pub fn binding_limit(
        &self,
        stage: ShaderStage,
        class: BindingLimitClass,
    ) -> Option<u32>;

    pub fn route(
        &self,
        query: &RouteQuery,
    ) -> RouteSupport;

    pub fn submission(&self) -> &SubmissionCapabilities;

    pub fn shader_acceptance(
        &self,
        artifact: &ShaderArtifact,
    ) -> ArtifactAcceptance;
}
```

### Semantics of `Option<FormatFacts>`

```text
Some(...)
    = this format is supported by the current Adapter / Device contract

None
    = the format is unavailable to the current contract
```

For platforms such as WebGPU where "format requires feature enablement":

```text
AvailableCapabilities::format(format) -> Some(...)
EnabledCapabilities::format(format)   -> None
```

is legal as long as the Device is created without enabling the required semantics of the format.

Therefore correctness always depends on:

```rust
device.capabilities()
```

instead of Adapter snapshot.

---

## 7.3 No longer retain duplicate capability bool

Remove the following types of repetitive structures:

```text
ResourceCapabilities {
    supports_texture_1d
    supports_texture_3d
}

ShaderCapabilities {
    vertex
    fragment
    compute
}

PipelineCapabilities {
    raster
    compute
}

BindingCapabilities {
    max_bind_groups
    ...
}
```

reason:

- Base Raster / Vertex / Fragment are RHI core contracts and do not require bool;
- Compute uses `OptionalFeature::Compute`;
- Texture dimension/usage/sample count uses `TextureSupportQuery`;
- The total number of Binding groups, the total number of dynamic-buffers, and alignment use `DeviceLimits`;
- per-stage resource count uses `binding_limit(stage, class)`;
- binding kind/count/dynamic legality using `BindingSupportQuery`;
- Binding arrays use `OptionalFeature::BindingArrays` + BindingSupport;
- format-specific capabilities provided by `FormatFacts`;
- presentation by Surface Facts.

A fact can only have one canonical source.

---

## 7.4 DeviceLimits

```rust
#[derive(Clone, Debug)]
pub struct DeviceLimits {
    /* opaque immutable map */
}

impl DeviceLimits {
    pub fn get(
        &self,
        key: LimitKey,
    ) -> Option<u64>;
}

impl EnabledCapabilities {
    pub fn limits(&self) -> &DeviceLimits;
}
```

`LimitKey` uses `#[non_exhaustive]`, and adding limit in the future will not change the overall API.

Notice:

```text
MaxFoo
    larger values are stronger

MinFooAlignment
    smaller values are stronger
```

So Device request uses:

```text
AtLeast
AtMost
```

The two requirements do not allow for a unified `minimum_limit()` to change semantics.

---
