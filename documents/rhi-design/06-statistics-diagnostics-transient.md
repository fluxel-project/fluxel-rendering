# RHI API freeze v13. Statistics, diagnostics, and transient resources

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and this module in full before implementation. Statistics are
> portable logical observations; transient allocation is a pure RHI capability.

# 47. Statistics — portable logical statistics

This chapter is **FROZEN / P0**.

Goal:

> Enable Debug HUD / benchmark / engine tooling to obtain **Fluxel logical statistics with the same definitions** across five backends.

This does not replace PIX / RenderDoc / Xcode / vendor profilers.

---

## 47.1 Frozen boundary

RHI Statistics measure:

```text
portable commands
portable state changes
logical submission structure
presentation lifecycle
logical object lifecycle
live inventory
descriptor-based logical resource bytes
caller-defined frame interval / FPS
```

They do not measure or promise:

```text
native barrier count
native descriptor bind count
native command buffer count
native queue switch
actual queue overlap
driver PSO cache hit/miss
hardware counter / occupancy / cache miss
actual VRAM allocation / residency / fragmentation
GPU timestamp/frame time
display scan-out FPS
```

These will later be independently provided by:

```text
Query
backend tooling
allocator telemetry
present timing
```

---

## 47.2 Device-scoped service

Each DeviceIdentity has an independent statistics domain.

```rust
#[derive(Clone)]
pub struct DeviceStatistics {
    /* opaque; DeviceIdentity scoped */
}

impl Device {
    pub fn statistics(
        &self,
    ) -> DeviceStatistics;
}
```

There is no global statistics singleton.

If upper layers simultaneously run:

```text
DX12 Device A
Vulkan Device B
```

they aggregate the two device snapshots themselves; the RHI does not combine them into one counter.

---

## 47.3 Collection detail / epoch

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatisticsDetail {
    /// Low-frequency data such as inventory/lifecycle/submission/presentation.
    Minimal,

    /// Adds draw/dispatch/copy/scope/bind-call.
    Basic,

    /// Adds effective state changes and interval working set.
    Detailed,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatisticsConfig {
    pub detail: StatisticsDetail,
}

impl StatisticsConfig {
    pub fn minimal() -> Self;
    pub fn basic() -> Self;
    pub fn detailed() -> Self;
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self::minimal()
    }
}
```

Switching detail changes which counters are collected.

Therefore, `configure()` must start a new **collection epoch**:

```rust
impl DeviceStatistics {
    pub fn configure(
        &self,
        config: StatisticsConfig,
    ) -> RhiResult<()>;

    pub fn config(&self) -> StatisticsConfig;
    pub fn collection_epoch(&self) -> u64;
}
```

Freeze rule:

```text
configure()
    -> collection_epoch += 1
    -> event cumulative counters restart at 0
    -> live inventory is not reset
```

This avoids the ambiguity of:

```text
Minimal for the first half hour
Detailed for the second half hour
yet treating detailed cumulative values as “since Device creation”
```



`configure()` is thread-safe and does not require the GPU to be idle.

---

## 47.4 Snapshot consistency

```rust
#[derive(Clone, Debug)]
pub struct StatisticsSnapshot {
    device: DeviceIdentity,
    collection_epoch: u64,
    sequence: u64,

    /// Monotonic CPU time on the Device statistics clock.
    cpu_time_ns: u64,

    cumulative: CumulativeStatistics,
}

impl StatisticsSnapshot {
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn collection_epoch(&self) -> u64;
    pub fn sequence(&self) -> u64;
    pub fn cpu_time_ns(&self) -> u64;
    pub fn cumulative(&self) -> &CumulativeStatistics;

    pub fn delta_since(
        &self,
        previous: &StatisticsSnapshot,
    ) -> RhiResult<IntervalStatistics>;
}

impl DeviceStatistics {
    pub fn snapshot(&self) -> StatisticsSnapshot;
}
```

`delta_since()` requires:

```text
same DeviceIdentity
same collection_epoch
self.sequence >= previous.sequence
```

Otherwise it returns `InvalidUsage`.

A snapshot does not wait for the GPU.

Concurrent events may land before or after a snapshot, but one logical event must not be “torn in half”;
a snapshot must read an internally consistent counter set.

Recommended implementation:

```text
Recorder-local counters
    -> bulk merge at finish()

Submission/Present events
    -> atomic/batched merge

Resource inventory
    -> object lifecycle table

snapshot
    -> versioned consistent read
```

---

## 47.5 CumulativeStatistics

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CumulativeStatistics {
    pub commands: CommandStatistics,
    pub bindings: BindingStatistics,
    pub submissions: SubmissionStatistics,
    pub presentation: PresentationStatistics,
    pub resources: ResourceLifecycleStatistics,
    pub transient: TransientMemoryStatistics,
}
```

Counters use `u64`.

Implementations must not wrap; after reaching `u64::MAX`, they saturate.
This is stable semantics for exceptionally long-running systems; normal programs will almost never reach it.

---

## 47.6 Command statistics

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CommandStatistics {
    pub recorders_finished: u64,

    pub raster_scopes: u64,
    pub compute_scopes: u64,

    pub draw_calls: u64,
    pub draw_indexed_calls: u64,
    pub dispatch_calls: u64,

    pub buffer_copies: u64,
    pub buffer_to_texture_copies: u64,
    pub texture_to_buffer_copies: u64,
    pub texture_copies: u64,

    pub resolves: u64,
    pub blits: u64,

    pub upload_commands: u64,
    pub readback_commands: u64,

    pub debug_markers: u64,
}
```

These are Fluxel semantic command counts.

They are not equal to the backend's final:

```text
VkCmd count
D3D12 command count
Metal encoder calls
WebGPU internal commands
```

---

## 47.7 Binding / switch statistics

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct BindingStatistics {
    pub pipeline_bind_calls: u64,

    /// Effective Pipeline ObjectId change within a scope.
    pub pipeline_changes: u64,

    /// Effective executable shader set change.
    ///
    /// Raster = (vertex ShaderModule ObjectId, optional fragment ObjectId)
    /// Compute = compute ShaderModule ObjectId
    ///
    /// Do not use hashes/fingerprints to determine correctness.
    pub shader_set_changes: u64,

    pub bind_group_bind_calls: u64,

    /// Effective binding tuple change:
    ///
    /// (BindGroup ObjectId, dynamic_offsets[])
    pub bind_group_changes: u64,

    /// set_vertex_buffer / set_index_buffer API calls.
    pub buffer_bind_calls: u64,

    /// Effective buffer binding element change.
    ///
    /// Includes:
    /// - vertex/index binding
    /// - BindGroup buffer binding
    /// - effective range changes caused by dynamic offsets
    pub buffer_binding_changes: u64,

    /// Number of effective Texture element changes caused by BindGroup state changes.
    pub texture_binding_changes: u64,

    /// Number of effective Sampler element changes caused by BindGroup state changes.
    pub sampler_binding_changes: u64,

    /// The attachment set differs between adjacent RasterScopes in the same Recorder.
    ///
    /// The first set, transitioning from “no target” to an actual target, also counts as 1.
    pub render_target_set_changes: u64,
}
```

### Scope reset

Command state such as Pipeline / BindGroup is considered unbound at the start of every Raster/Compute scope.

Therefore:

```text
Scope A set_pipeline(P) -> pipeline_changes +1
Scope B set_pipeline(P) -> pipeline_changes +1
```

because a new scope does not inherit old state.

### RT switch

`render_target_set_changes` is deliberately defined over:

```text
the RasterScope sequence of the same Recorder
```

rather than the whole Device.

Multiple Recorders may record in parallel and later execute on different lanes/batches,
so no reliable system-wide “previous RT” exists.

### Binding changes ≠ shader usage

When a Pipeline change causes a shader to start/stop reading an already-bound resource:

```text
texture_binding_changes does not automatically increase
```

because bind state did not change.

Actual resource use is represented by actual command use in `WorkingSetStatistics`.

---

## 47.8 Submission statistics

With multiple lanes there is no global total order, so remove the old metric:

```text
logical_lane_changes
```

There is no portable definition of “how many times adjacent execution batches switch from queue A to queue B”.

Freeze as:

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct SubmissionStatistics {
    /// Number of Device::submit() calls.
    pub submission_calls: u64,

    /// Number of plans for which at least some native work was accepted.
    pub plans_accepted: u64,

    /// Number of plans fully rejected before submission, with no native work accepted.
    pub plans_rejected: u64,

    /// Total logical batches declared in accepted plans.
    pub batches_planned: u64,

    /// Number of logical batches that actually entered backend acceptance successfully.
    pub batches_accepted: u64,

    /// Number of RecordedWork items in accepted batches.
    pub recorded_work_items_accepted: u64,

    /// Number of explicit cross-lane dependencies within this plan.
    pub cross_lane_dependencies: u64,

    /// Number of prior CompletionPoint -> current PlanPoint dependencies in accepted plans.
    pub external_dependencies: u64,

    /// Number of final lowerings that choose GPU-side dependencies.
    pub gpu_dependency_routes: u64,

    /// Number of final lowerings that require lane collapse.
    pub collapsed_dependency_routes: u64,
}
```

If:

```text
batch A accepted
batch B immediate failure
```

then:

```text
plans_accepted += 1
batches_planned += plan.batch_count
batches_accepted += actual accepted count
```

Do not incorrectly count the entire plan as rejected.

---

## 47.9 Per-lane interval usage

“How much each queue/lane was used” is exposed through data that is actually definable:

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct LaneIntervalStatistics {
    pub lane: SubmissionLaneId,
    pub batches_accepted: u64,
    pub recorded_work_items: u64,
}

#[non_exhaustive]
pub struct IntervalStatistics {
    pub device: DeviceIdentity,
    pub collection_epoch: u64,

    pub elapsed_cpu_ns: u64,

    pub commands: CommandStatistics,
    pub bindings: BindingStatistics,
    pub submissions: SubmissionStatistics,
    pub presentation: PresentationStatistics,
    pub resources: ResourceLifecycleStatistics,

    /// Lists only lanes actually used during this interval.
    pub lanes: Vec<LaneIntervalStatistics>,

    pub working_set: Option<WorkingSetStatistics>,
}
```

`lanes` must be canonically sorted by `SubmissionLaneId`.

Upper layers may display:

```text
Graphics lane: 5 batches / 11 work items
Transfer lane: 2 batches / 2 work items
Unique lanes: 2
Cross-lane deps: 3
```

rather than inventing native queue switches.

---

## 47.10 Presentation statistics

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct PresentationStatistics {
    pub acquires_succeeded: u64,

    pub acquire_not_ready: u64,
    pub acquire_timeout: u64,
    pub acquire_outdated: u64,
    pub acquire_target_lost: u64,

    /// Number of times a frame is consumed by present_after().
    pub presents_planned: u64,

    pub presents_accepted: u64,
    pub presents_outdated: u64,
    pub presents_target_lost: u64,
    pub presents_failed: u64,

    /// Explicit abandon + Drop safety abandonment.
    pub frames_abandoned: u64,
}
```

Do not count:

```text
scan-out count
displayed frame count
vsync count
```

unless a Presentation Timing extension is added later.

Acquire refusal and submitted-present outcome are disjoint accounting domains.
`NotReady`, `Timeout`, `FrameOutstanding`, zero-size suspension, and acquire-time
out-of-memory do not increment any `presents_*` field because no present was
planned or accepted. Acquire `TargetOutdated` and `TargetLost` increment only
their acquire fields. Device loss is reported by the device/loss statistics and
must not be relabeled as a present outcome. A terminal `PresentState` increments
exactly one applicable `presents_*` category.

---

## 47.11 Resource lifecycle statistics

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ResourceLifecycleStatistics {
    pub buffers_created: u64,
    pub textures_created: u64,
    pub texture_views_created: u64,
    pub samplers_created: u64,

    pub shader_modules_created: u64,
    pub bind_group_layouts_created: u64,
    pub bind_groups_created: u64,
    pub pipeline_interfaces_created: u64,
    pub raster_pipelines_created: u64,
    pub compute_pipelines_created: u64,

    pub buffers_reclaimed: u64,
    pub textures_reclaimed: u64,
    pub texture_views_reclaimed: u64,
    pub samplers_reclaimed: u64,

    pub shader_modules_reclaimed: u64,
    pub bind_group_layouts_reclaimed: u64,
    pub bind_groups_reclaimed: u64,
    pub pipeline_interfaces_reclaimed: u64,
    pub raster_pipelines_reclaimed: u64,
    pub compute_pipelines_reclaimed: u64,
}
```

`reclaimed` means:

```text
not a public handle drop
instead, backing has satisfied completion-safe reclaim conditions
```

---

## 47.12 Working set

Detailed mode collects actual-use unique objects per interval:

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct WorkingSetStatistics {
    pub unique_buffers: u64,
    pub unique_textures: u64,
    pub unique_frame_attachments: u64,

    pub unique_samplers: u64,
    pub unique_bind_groups: u64,

    pub unique_shader_modules: u64,
    pub unique_raster_pipelines: u64,
    pub unique_compute_pipelines: u64,

    pub unique_submission_lanes: u64,
}
```

All “unique” values use logical IDs:

```text
ObjectId
AcquiredFrameId
SubmissionLaneId
```

not native handles.

---

## 47.13 Frame sampler / FPS

The RHI does not know what a “game frame” is.

Upper layers choose their sample boundary:

```rust
pub struct FrameStatisticsSampler {
    statistics: DeviceStatistics,
    previous: StatisticsSnapshot,
}

#[derive(Clone, Debug)]
pub struct FrameStatistics {
    interval: IntervalStatistics,
    fps: Option<f64>,
}

impl DeviceStatistics {
    pub fn frame_sampler(
        &self,
    ) -> FrameStatisticsSampler;
}

impl FrameStatisticsSampler {
    pub fn sample_frame(
        &mut self,
    ) -> RhiResult<FrameStatistics>;
}

impl FrameStatistics {
    pub fn interval(&self) -> &IntervalStatistics;
    pub fn fps(&self) -> Option<f64>;
}
```

`fps()`：

```text
elapsed_cpu_ns > 0
    -> 1e9 / elapsed_cpu_ns

elapsed_cpu_ns == 0
    -> None
```

If the statistics collection epoch changes after the sampler is created:

```text
sample_frame() -> InvalidUsage
```

The caller recreates the sampler.

This avoids calculating an incorrect delta across detail levels.

This FPS is:

> caller-defined Renderer loop CPU sample rate.

It is not GPU FPS / scan-out FPS.

---

## 47.14 Live inventory

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct LiveObjectCounts {
    pub buffers: u64,

    pub textures: u64,

    /// An overlapping subset of textures.
    pub render_target_textures: u64,
    pub color_attachment_textures: u64,
    pub depth_stencil_textures: u64,

    pub texture_views: u64,
    pub samplers: u64,

    pub shader_modules: u64,

    pub bind_group_layouts: u64,
    pub bind_groups: u64,

    pub pipeline_interfaces: u64,
    pub raster_pipelines: u64,
    pub compute_pipelines: u64,

    /// Number of frames currently in Acquired / PlannedForPresent.
    pub outstanding_frames: u64,
}
```

Definition:

> A unique logical object in RHI inventory that has not yet been finally reclaimed/terminal.

If one Buffer handle is cloned 100 times:

```text
buffers == 1
```

Likewise, cloning FrameAttachment does not increase `outstanding_frames`.

---

## 47.15 Logical memory estimate

What is frozen is a **logical resource bytes estimate**, not VRAM.

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEstimateQuality {
    LogicalEstimate,
    Unknown,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct MemoryEstimate {
    pub logical_estimated_bytes: Option<u64>,
    pub quality: MemoryEstimateQuality,
}

impl MemoryEstimate {
    pub const fn unknown() -> Self;

    pub const fn logical(
        bytes: u64,
    ) -> Self;
}

impl Default for MemoryEstimate {
    fn default() -> Self;
}

#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ResourceMemoryStatistics {
    /// Non-overlapping top-level classifications.
    pub buffers: MemoryEstimate,
    pub textures: MemoryEstimate,
    pub total_resources: MemoryEstimate,

    /// Overlapping analytical subsets of Texture; do not add them to textures again.
    pub render_target_textures: MemoryEstimate,
    pub color_attachment_textures: MemoryEstimate,
    pub depth_stencil_textures: MemoryEstimate,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct InventoryStatistics {
    pub device: DeviceIdentity,
    pub objects: LiveObjectCounts,
    pub memory: ResourceMemoryStatistics,
}

impl DeviceStatistics {
    pub fn inventory(
        &self,
    ) -> RhiResult<InventoryStatistics>;
}
```

---

## 47.16 Buffer estimate

```text
logical_estimated_bytes = BufferDescriptor.size
```

It excludes native allocation padding/page/metadata.

---

## 47.17 Texture estimate

If:

```rust
FormatFacts::logical_bytes_per_block()
```

is `Some(bytes)`:

```text
mip_width  = max(1, width  >> mip)
mip_height = max(1, height >> mip)
mip_depth  = max(1, depth  >> mip)

blocks_x = ceil(mip_width  / block_width)
blocks_y = ceil(mip_height / block_height)

mip_bytes =
    blocks_x
  * blocks_y
  * mip_depth
  * bytes_per_block
  * array_layers
  * sample_count
```

Sum all mips, using checked arithmetic for every intermediate multiplication and addition.

For any overflow:

```text
MemoryEstimate::Unknown
```

For implementation-defined backing such as `Depth24Plus`:

```text
logical_bytes_per_block == None
    -> Unknown
```

This estimate excludes:

```text
tiling/swizzle
row alignment
driver metadata
compression
mip tail
heap fragmentation
alias reuse
residency
```

Therefore, the name is frozen as:

```text
logical_estimated_bytes
```

It must never be called:

```text
vram_bytes
gpu_memory_bytes
physical_bytes
```

---

## 47.18 Per-object estimate

```rust
impl DeviceStatistics {
    pub fn estimate_buffer_memory(
        &self,
        buffer: &Buffer,
    ) -> RhiResult<MemoryEstimate>;

    pub fn estimate_texture_memory(
        &self,
        texture: &Texture,
    ) -> RhiResult<MemoryEstimate>;
}
```

Use only canonical descriptors + format facts.

Do not wait for the GPU or query native heaps.

---

## 47.19 Implementation constraints

Statistics must not change RHI semantics.

Allowed:

```text
recorder-local counters
finish-time merge
submission-time merge
atomic/versioned lifecycle inventory
Detailed mode maintains local state/unique sets
```

Forbidden:

```text
insert GPU commands
insert barriers
wait for the GPU
change lane assignment
change aliasing
change present behavior
force serialization of parallel recorders
```

Enabling/disabling statistics may affect only CPU/memory overhead.

---

## 47.20 Capture relation

Statistics and Capture may share internal semantic-observation plumbing, but:

```text
Statistics
    = aggregate counters / inventory

Capture
    = ordered reconstructable semantic events
```

Statistics：

```text
are not the Replay source of truth
do not enter the Capture correctness contract
do not require bit-exact counters after Replay
```

---

## 47.20.1 Rust semver rule

Every externally returned Statistics data struct to which metrics may be added in the future uses `#[non_exhaustive]`;
upper layers may read fields stably, while future metrics can be added without breaking existing construction/pattern-matching APIs.

---

## 47.21 Frozen metric definitions

0.16 freezes the following definitions:

```text
pipeline_changes
    = scope-local effective Pipeline ObjectId change

shader_set_changes
    = effective ShaderModule ObjectId tuple change

bind_group_changes
    = effective (BindGroup ObjectId + dynamic offsets) change

buffer_binding_changes
    = effective logical buffer element/range change

texture_binding_changes
    = logical texture element change caused by a BindGroup state change

render_target_set_changes
    = same-Recorder adjacent RasterScope target-set change

submission_calls
    = Device::submit() calls

plans_accepted / rejected
    = according to the freeze v13 submit acceptance contract

cross_lane_dependencies
    = explicit dependency in the same plan whose logical lanes differ

external_dependencies
    = prior CompletionPoint -> current PlanPoint dependency

LaneIntervalStatistics
    = accepted batch/work counts per logical lane

fps
    = caller-defined CPU sampling rate

logical_estimated_bytes
    = descriptor/format-based logical resource estimate
```

Explicitly remove:

```text
logical_lane_changes
native queue switches
```

because multiple lanes have no portable global adjacent execution order.

# 48. Diagnostics

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DiagnosticEvent {
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub object: Option<ObjectId>,
    pub label: Option<String>,
    pub operation: Option<&'static str>,
    pub backend_detail: Option<String>,
}

impl Device {
    /// Pull model; avoids imposing a callback threading policy.
    pub fn drain_diagnostics(&self, out: &mut Vec<DiagnosticEvent>);
}
```

Backend details are for diagnostics only and do not participate in portable correctness.

---

# 48.1 Canonicalization requirements

Every descriptor that participates in:

```text
compatibility id
fingerprint
Capture definition
statistics identity comparison
```

must be canonicalized first.

Set-like vectors:

```text
TextureDescriptor.view_formats
DeviceRequirements required/preferred feature sets
ShaderInterface resources/inputs/outputs
BindGroupLayout entries
BindGroup entries
```

must:

```text
stable sort
reject semantic duplicates or deduplicate (as defined by each type)
```

Ordered-semantic vectors:

```text
PipelineInterface.groups
color target locations
Submission batches/work
commands
```

retain their semantic order and must not be sorted.

`Label` and diagnostic strings do not participate in compatibility/fingerprint correctness.

---

# 49. Validation requirements

At minimum, validate the following before entering the backend:

```text
DeviceIdentity
Buffer range / alignment
Texture format / usage / sample count
TextureView reinterpretation / aspect / mip / layer
copy/upload/readback layout
BindGroupLayout / PipelineInterface compatibility
dynamic offsets
shader requirements / artifact acceptance
pipeline target signature
actual ResourceUse legality
command scope legality
lane legality
SubmissionPlan topology / token ownership
frame single-consume
CompletionPoint / PresentReceipt device source
surface configuration against target facts
statistics snapshot DeviceIdentity / collection epoch compatibility
```

---

# 50. Transient resource model

Transient allocation is a frozen, pure RHI capability. It creates short-lived
ordinary `Buffer` and `Texture` handles; no external scheduling contract,
native heap, placed-resource offset, or aliasing barrier is public.
`ResourceUse` remains `rhi::command::ResourceUse` and is recorded from actual
portable commands.

## 50.1 Capability and requirements

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransientAllocationSupport {
    /// Correctness-complete baseline: one independent backing per resource.
    Dedicated,
    /// Physical memory may be reused between proven non-overlapping lifetimes.
    Aliasing,
}

#[derive(Clone, Copy, Debug)]
pub struct TransientCapabilities {
    pub buffers: TransientAllocationSupport,
    pub textures: TransientAllocationSupport,
    pub mixed_resource_aliasing: bool,
}

impl EnabledCapabilities {
    pub fn transient(&self) -> TransientCapabilities;
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum TransientResourceDescriptor {
    Buffer(BufferDescriptor),
    Texture(TextureDescriptor),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransientCompatibilityClass(u64);

#[derive(Clone, Copy, Debug)]
pub struct TransientAllocationRequirements {
    pub logical_size: Option<u64>,
    pub physical_size: Option<u64>,
    pub alignment: Option<u64>,
    pub class: Option<TransientCompatibilityClass>,
}

impl Device {
    pub fn transient_requirements(
        &self,
        desc: &TransientResourceDescriptor,
    ) -> RhiResult<TransientAllocationRequirements>;
}
```

`Dedicated` is mandatory whenever the corresponding ordinary descriptor is
supported. `physical_size`, `alignment`, and `class` are device/backend facts,
not portable capture correctness data. `mixed_resource_aliasing == false` only
forbids buffer/texture sharing; it does not forbid aliasing within either kind.

## 50.2 Plan-point lifetime and allocation

Transient lifetime uses the existing RHI submission vocabulary only:

```rust
let mut plan = SubmissionPlanBuilder::new(&device);
let a = plan.reserve_batch(lane)?;
let b = plan.reserve_batch(lane)?;
let transient = plan.transient_allocator();
let texture = transient.create_texture(
    &desc,
    TransientLifetime::new(a).release_at(b),
)?;
```

```rust
#[derive(Clone, Debug)]
pub struct TransientLifetime {
    acquire: PlanPoint,
    release_frontier: Vec<PlanPoint>,
}

impl TransientLifetime {
    pub fn new(acquire: PlanPoint) -> Self;
    pub fn release_at(self, point: PlanPoint) -> Self;
    pub fn acquire(&self) -> PlanPoint;
    pub fn release_frontier(&self) -> &[PlanPoint];
}

#[derive(Clone)]
pub struct TransientAllocator { /* Device + SubmissionPlan scoped */ }

impl TransientAllocator {
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn plan_id(&self) -> SubmissionPlanId;
    pub fn create_buffer(&self, desc: &BufferDescriptor, lifetime: TransientLifetime)
        -> RhiResult<Buffer>;
    pub fn create_texture(&self, desc: &TextureDescriptor, lifetime: TransientLifetime)
        -> RhiResult<Texture>;
}
```

The allocator is obtained through the `SubmissionPlanBuilder::transient_allocator`
method defined with the complete builder interface in section 40.

All lifetime points must belong to the same plan; the frontier is non-empty;
`acquire` happens-before every frontier point. Every actual use must be after
or equal to acquire and able to reach at least one release frontier. Otherwise
the builder returns `InvalidUsage`. Returned handles use normal view, binding,
recording, copy, and attachment APIs; native materialization may be deferred to
async submit preflight.

Transient handles carry their plan identity. `set_batch`/`add_batch` reject a
recorded use belonging to another plan. On plan drop, rejection, or terminal
completion/loss, its transient resources expire and cannot subsequently enter
GPU execution.

# 51. Transient allocation and aliasing lowering

`SubmissionPlanBuilder::build()` validates the PlanPoint DAG, actual
`ResourceUse`, and `TransientLifetime`. Physical realization may wait until
`device.submit(plan).await` preflight. If it fails, submit returns `Err` before
any native work from that plan is accepted.

Aliasing is legal only when all release-frontier points of one resource
happen-before the other's acquire (or vice versa), and compatibility class,
alignment, physical size, resource type, and backend restrictions agree.
Potentially parallel lifetimes never alias.

The RHI, not the caller, lowers the physical reuse relation together with
PlanPoint ordering and actual uses into required DX12 aliasing barriers, Vulkan
memory/image dependencies, Metal heap/resource synchronization, or a no-op.
An implementation that cannot lower safely must use `Dedicated`.

WebGPU, OpenGL, WebGL2, and incomplete native allocators may implement
`TransientAllocationSupport::Dedicated`: each transient gets normal independent
backing and is reclaimed/recycled after plan-terminal completion. Backing must
survive through its last actual-use `CompletionPoint`; CPU plan drop is never a
license for early reuse.

### 51.1 Placed-heap implementation route

Placed heaps/pages, allocation pools, and aliasing-barrier bookkeeping are
backend-private realizations of this section. They require no additional
portable allocation, heap-offset, or barrier API. A native backend may first
ship `Dedicated`, then add a placed allocator and advertise `Aliasing` only
after it proves all of the following for every realized reuse:

```text
compatible resource/memory class and alignment
non-overlap from TransientLifetime + PlanPoint ordering
actual ResourceUse boundaries lowered to native synchronization
old and new backing remain completion-safe
device loss and submit rejection retain/retire backing safely
```

Until then, a private TODO must identify `Dedicated` as the active correct
fallback. It must not be a `todo!()` or `unimplemented!()` reachable from
`TransientAllocator::create_*` or `Device::submit`.

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct TransientMemoryStatistics {
    pub logical_bytes: u64,
    pub physical_backing_bytes: u64,
    pub resources_realized: u64,
    pub alias_reuses: u64,
}
```

These are implementation-observable transient backing statistics, not VRAM or
residency measurements. Capture records descriptor, PlanPoint lifetime, and
normal portable commands only; replay may re-alias or use Dedicated backing.
