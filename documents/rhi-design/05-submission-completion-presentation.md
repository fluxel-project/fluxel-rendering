# RHI API freeze v13. Submission, completion, and presentation

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and this module in full before implementation. Submission
> acceptance, GPU completion, and presentation outcome are distinct contracts.

# 39. SubmissionPlan identity / tokens

This chapter is **FROZEN / P0**.

The Submission API exposes only logical plans/tokens, never:

~~~text
native queue
fence
semaphore
event
timeline value
~~~

---

## 39.1 Plan identity

Every Builder receives an opaque, Device-scoped plan identity when created.

~~~rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionPlanId {
    /* device identity + opaque serial */
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionBatchId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanPoint {
    plan: SubmissionPlanId,
    batch: SubmissionBatchId,
}

impl PlanPoint {
    pub fn batch(self) -> SubmissionBatchId;
}
~~~

These types have no public constructors.

Therefore:

~~~text
PlanPoint from Builder A
passed to Builder B
    -> InvalidUsage
~~~

A user cannot forge batch = 0 to bypass the invariant.

---

## 39.2 Submitted tokens

~~~rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionPoint {
    device: DeviceIdentity,
    serial: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompletionPoint {
    device: DeviceIdentity,
    serial: u64,
}

impl SubmissionPoint {
    pub fn device_identity(self) -> DeviceIdentity;
}

impl CompletionPoint {
    pub fn device_identity(self) -> DeviceIdentity;
}
~~~

CompletionPoint is Fluxel's terminal-completion token.

It is not a native fence/timeline value; distinct logical points may map to the same native completion primitive on a backend.

---

# 40. SubmissionPlanBuilder

~~~rust
pub struct SubmissionPlan {
    /* opaque validated plan */
}

impl SubmissionPlan {
    pub fn id(&self) -> SubmissionPlanId;
    pub fn device_identity(&self) -> DeviceIdentity;
}

pub struct SubmissionPlanBuilder {
    /* opaque, bound to DeviceIdentity + SubmissionPlanId */
}

impl SubmissionPlanBuilder {
    pub fn new(
        device: &Device,
    ) -> Self;

    /// Reserves an empty logical point before its RecordedWork exists.
    ///
    /// This freezes transient lifetime frontiers before recording.
    pub fn reserve_batch(
        &mut self,
        lane: SubmissionLaneId,
    ) -> RhiResult<PlanPoint>;

    /// Fills one previously reserved point exactly once.
    pub fn set_batch(
        &mut self,
        point: PlanPoint,
        work: Vec<RecordedWork>,
    ) -> RhiResult<()>;

    /// Convenience for reserve_batch + set_batch.
    pub fn add_batch(
        &mut self,
        lane: SubmissionLaneId,
        work: Vec<RecordedWork>,
    ) -> RhiResult<PlanPoint>;

    /// Returns an allocator scoped to this plan and Device.
    pub fn transient_allocator(
        &self,
    ) -> TransientAllocator;

    /// Adds cross-batch happens-before.
    pub fn add_dependency(
        &mut self,
        before: PlanPoint,
        after: PlanPoint,
    ) -> RhiResult<()>;

    /// Establishes happens-before from prior submitted-work completion to this
    /// plan batch.
    ///
    /// A GPU-side wait, an already ordered execution domain, or a proven
    /// Collapse route is accepted. It returns Unsupported only when none of
    /// those routes proves the required order; a caller may instead wait for
    /// completion before building/submitting this plan.
    pub fn add_external_dependency(
        &mut self,
        before: CompletionPoint,
        after: PlanPoint,
    ) -> RhiResult<()>;

    /// Includes frame presentation in the plan.
    pub fn present_after(
        &mut self,
        frame: AcquiredFrame,
        after: PlanPoint,
    ) -> RhiResult<PresentPlanId>;

    pub fn build(
        self,
    ) -> RhiResult<SubmissionPlan>;
}
~~~

---

## 40.1 Transient plan integration

Transient allocation is an RHI capability, not a render-scheduling contract.
`TransientAllocator` creates ordinary `Buffer` and `Texture` handles scoped to
this builder's `SubmissionPlanId`; its full resource contract is defined in the
resource/transient module. A lifetime uses the existing execution vocabulary:

~~~rust
let mut plan = SubmissionPlanBuilder::new(&device);
let a = plan.reserve_batch(lane)?;
let b = plan.reserve_batch(lane)?;

let transient = plan.transient_allocator();
let texture = transient.create_texture(
    &desc,
    TransientLifetime::new(a).release_at(b),
)?;
~~~

`TransientLifetime` contains an acquire `PlanPoint` and a non-empty
`release_frontier: Vec<PlanPoint>`. Its points must belong to this plan,
acquire must happen-before every release point, and every actual use must lie
within that lifetime. `set_batch`/`add_batch` reject transient resources owned
by a different plan.

At async submit preflight, RHI combines PlanPoint ordering, actual
`command::ResourceUse`, and any physical alias relation to lower native alias
synchronization. `Dedicated` is the required correct fallback; `Aliasing` is
an optional performance implementation. Callers never encode alias barriers.

---

## 40.2 Batch invariants

reserve_batch() creates an empty logical point; set_batch() fills it exactly
once. At build():

~~~text
every reserved point has non-empty work
~~~

add_batch() is the reserve + set convenience.

Batch validation:

~~~text
work non-empty

lane belongs to current Device
every RecordedWork DeviceIdentity identical

for every work:
    lane.domains contains work.work_domains()
~~~

The order in Vec<RecordedWork> is the batch's logical work order.

For one lane, Builder insertion order forms logical submission order:

~~~text
Lane X:
    Batch A
    Batch B
    Batch C
~~~

It inherently has:

~~~text
A before B before C
~~~

RHI must still generate required memory/execution synchronization from actual-use sequence; “same lane has order” does not mean a command-buffer boundary supplies a memory barrier. Vulkan explicitly separates submission order from actual memory/execution dependency (see References).

---

## 40.3 Cross-lane dependencies

Different lanes default to:

~~~text
unordered
~~~

Only add_dependency(before, after) creates portable happens-before.

Builder handles:

~~~rust
device
    .capabilities()
    .submission()
    .dependency_route(from_lane, to_lane)
~~~

as:

~~~text
Ordered
    -> actually already in one ordered execution domain

Gpu
    -> backend lowers a GPU-side dependency

Collapse
    -> backend collapses logical lanes while retaining logical order

Unsupported
    -> add_dependency/build returns Unsupported
~~~

### HostWait removed from P0 lane routes

HostWait is no longer a LaneDependencyRoute.

A:

~~~text
host wait
~~~

would turn a relationship that should be a GPU execution plan into CPU orchestration policy, and may require blocking/runtime/background workers.

If this is genuinely needed later:

~~~text
GPU work A
-> CPU completion callback
-> GPU work B
~~~

it belongs to an upper-layer host node / continuation, not a pretend GPU lane dependency.

---

## 40.4 Cross-plan dependency

One logical lane remains lane-ordered across multiple Device::submit() calls.

But:

~~~text
prior plan Lane A
    -> next plan Lane B
~~~

has no implicit happens-before.

It must use:

~~~text
add_external_dependency(previous_completion, next_point)
~~~

or observe previous completion Complete on CPU before submitting the next plan.

`add_external_dependency()` succeeds when the backend can establish the edge
through a GPU-side wait, or when the relevant source and destination execution
domains are already ordered, including a proven `Collapse` lowering that retains
the required logical order. A backend returns `Unsupported` only if it can prove
none of those routes. It must not reject an already ordered/no-op dependency
merely because it has no separate native wait primitive.

This prevents multi-lane semantics from breaking at frame/plan boundaries. Vulkan queues have no implicit inter-queue order; Metal likewise expresses different-queue dependencies through event wait/signal.

---

## 40.5 Unordered hazard validation

build() must use RecordedWork actual uses to inspect every batch pair with **no happens-before path**.

If:

~~~text
the same Buffer overlapping byte range
or the same Texture overlapping mip/layer/aspect
or the same AcquiredFrameId
~~~

and at least one side writes:

~~~text
READ/WRITE
WRITE/READ
WRITE/WRITE
~~~

then one of these must exist:

~~~text
same-lane order
or explicit PlanPoint dependency path
~~~

Otherwise:

~~~text
Err(RhiErrorKind::MissingDependency)
~~~

READ/READ needs no dependency.

RHI may not allow direct-RHI users to form an unsynchronized cross-queue data race.

---

## 40.6 Dependency graph validation

Before any native submission, build() must complete:

~~~text
PlanPoint belongs to this plan
dependency is not self-loop
cross-lane route supported
DAG has no cycle
external CompletionPoint DeviceIdentity valid
external CompletionPoint route is GPU, Ordered, or proven Collapse
no unordered overlapping write hazard
present after-point valid
each AcquiredFrame consumed only once
all object / work DeviceIdentity and liveness valid
~~~

Implicit same-lane order and explicit dependencies jointly participate in cycle detection.

---

# 41. Submission / Completion

## 41.1 Completion state

The old draft had only:

~~~text
Pending / Complete / DeviceLost / Failed
~~~

but asynchronous failure had no structured reason.

Replace it with:

~~~rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct CompletionFailure {
    message: String,
}

impl CompletionFailure {
    pub fn message(&self) -> &str;
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum CompletionState {
    Pending,
    Complete,

    DeviceLost(DeviceLossInfo),

    Failed(CompletionFailure),
}
~~~

Query:

~~~rust
impl Device {
    pub fn completion_state(
        &self,
        point: CompletionPoint,
    ) -> RhiResult<CompletionState>;

    /// Waits for this point to reach a terminal state.
    pub async fn wait_completion(
        &self,
        point: CompletionPoint,
    ) -> RhiResult<CompletionState>;
}
~~~

CompletionPoint.device_identity() must be validated.

This is a non-blocking query.

Progress comes from:

~~~text
normal host/runtime progress
+
Device::poll()
+
backend completion callbacks
~~~

---

## 41.2 SubmissionReceipt

One plan needs two completion levels:

~~~text
overall completion
per-batch / PlanPoint completion
~~~

Readback, transient retirement, and resource reuse should not be forced to await an unrelated slowest batch in the plan.

~~~rust
pub struct SubmissionReceipt {
    /* opaque */
}

impl SubmissionReceipt {
    pub fn device_identity(&self) -> DeviceIdentity;

    /// Logical serial for acceptance of this plan.
    pub fn submitted(&self) -> SubmissionPoint;

    /// Terminal completion of all GPU work in this plan.
    ///
    /// Does not include display scan-out completion.
    pub fn completion(&self) -> CompletionPoint;

    /// Terminal completion of work corresponding to a PlanPoint.
    ///
    /// A backend unable to provide finer completion may return the same
    /// token as overall completion.
    pub fn completion_for(
        &self,
        point: PlanPoint,
    ) -> RhiResult<CompletionPoint>;

    pub fn presents(&self) -> &[PresentReceipt];
}
~~~

When WebGPU has only a single queue completion primitive, multiple PlanPoints may conservatively share a completion; GPUQueue::onSubmittedWorkDone() expresses completion of work submitted as of its call, so coarse mapping is correct but less precise (see References).

---

## 41.3 Device::submit() acceptance contract

~~~rust
impl Device {
    /// A successful return means RHI entered “accepted submission” state.
    ///
    /// Err is allowed only when RHI can guarantee that:
    ///     no GPU work in this plan was accepted by native backend.
    pub async fn submit(
        &self,
        plan: SubmissionPlan,
    ) -> RhiResult<SubmissionReceipt>;
}
~~~

This is a critical frozen semantic.

### Phase A — preflight

Before any native queue submit/commit call, complete:

~~~text
plan/device identity
lane/work-domain legality
dependency DAG
resource/object identity
present relation
RecordedWork validity
backend lowering prerequisites
~~~

If Phase A fails:

~~~text
Err(...)
and guarantee that no plan work was submitted
~~~

### Phase B — backend acceptance

Once any native queue/work is accepted, Device::submit() **may not return an Err that makes the caller believe “nothing happened.”**

If a later lane submit immediately fails, device is lost, or backend cannot continue:

~~~text
return Ok(SubmissionReceipt)
~~~

and move:

~~~text
corresponding CompletionPoint
PlanPoint that could not continue submission
PresentReceipt
~~~

to Failed / DeviceLost terminal state.

This avoids:

~~~text
batch A already runs on GPU
batch B submit fails
API nevertheless returns Err
upper layer believes the entire plan did not execute
~~~

Vulkan requires ordinary submit failure to leave resource/synchronization state unaffected when no side effect can be guaranteed; otherwise it follows a device-lost-class path. This is why the portable API must explicitly distinguish Rejected from Accepted/terminal failure (see References).

---

## 41.4 Cross-plan in-flight hazard validation

SubmissionPlanBuilder::build() can see only its own plan; Device::submit() preflight must additionally inspect **existing submissions not yet terminal** on the current Device.

If current plan and prior in-flight work:

~~~text
access the same logical resource overlapping range/subresource
and at least one writes
and are not on the same ordered lane
~~~

then at least one must hold:

~~~text
current plan has add_external_dependency(prior_completion, current_point)
or prior completion is already Complete
or backend has collapsed/ordered relevant logical lanes into one execution domain
~~~

Otherwise:

~~~text
Err(RhiErrorKind::MissingDependency)
~~~

RHI must retain sufficient accepted-work use history until relevant completion is terminal so direct-RHI correctness does not break at Device::submit() boundaries.

For same-lane cross-submit resource transitions/memory dependency, RHI automatically lowers from the prior accepted use; caller still does not write barriers.

---

## 41.5 Completion ordering

completion_for(point) means:

> by this logical PlanPoint, work that the dependency graph requires it to await, and this point's own work, are terminal.

For one lane:

~~~text
earlier batch completion
    happens-before
later batch completion
~~~

For different lanes:

~~~text
no dependency
    -> no completion-order guarantee
~~~

Vulkan different queues likewise have no implicit ordering and require explicit synchronization (see References).

---

## 41.6 Readback binding

If a ReadbackTicket is encoded into PlanPoint P, after successful submit:

~~~text
ticket.completion()
    =
receipt.completion_for(P)
~~~

or a more conservative completion token chosen by backend.

State:

~~~text
NotSubmitted -> Pending
~~~

When completion is terminal:

~~~text
Complete   -> Ready
DeviceLost -> DeviceLost
Failed     -> Failed
~~~

---

## 41.7 Retirement

RHI internal retirement may use **completion of the last batch that actually referenced the object**:

~~~text
resource R
    last used by PlanPoint P

P terminal
+
no CPU logical owner
    ->
native backing may reclaim
~~~

It need not make every resource await receipt.completion().

The RHI transient allocator may also use completion_for(point) for fine-grained reuse.

---

## 41.8 SubmissionPoint vs CompletionPoint

~~~text
SubmissionPoint
    = logical serial where RHI accepted the plan

CompletionPoint
    = GPU-work terminal-observation token
~~~

They are not the same concept.

Forbidden:

~~~text
submit() returns
    => GPU complete

SubmissionPoint serial
    => native fence value
~~~

---

## 41.9 Device loss

After Device loss is observed, every pending CompletionPoint for that
DeviceIdentity must be woken and enter:

~~~text
CompletionState::DeviceLost(...)
~~~

It may not remain Pending forever.

Already Complete tokens remain Complete.

This is one part of the device-loss rule in section 6.5. The same observation
terminates pending readback, surface acquire, present/wait_present, and
wait_idle; there is no independent `Device::lost()` future or public loss-event
stream. A backend with no pending RHI operation need not poll just to discover
loss, but once it observes loss it must release every registered waiter.

---

## 41.10 Plan abandonment

When SubmissionPlan is dropped before successful submit:

~~~text
every included RecordedWork is released from plan ownership
unsubmitted ReadbackTicket -> Abandoned
unsubmitted Present plan -> no-submit terminal path
~~~

The Builder owns every `AcquiredFrame` consumed by `present_after()` until
ownership transfers into a successfully built `SubmissionPlan`. If `build()`
returns `Err`, Builder `Drop` performs the same no-submit abandonment
bookkeeping for every still-consumed frame. It performs no submission.

The presentation chapter owns the underlying cleanup details for an acquired
frame on its presentation target; a frame token must not silently leak.

---

## 41.11 Completion waiting is async

P0 provides no:

~~~rust
completion.wait();
~~~

Normal frame lifetime uses the async wait where awaiting completion is needed:

~~~text
device.wait_completion(point).await
~~~

`completion_state()` remains the non-blocking fast query. `Device::poll()` is
only an opportunistic synchronous progress hook, never the sole way an async
future can make progress.

`device.wait_idle().await` remains only for:

~~~text
shutdown
recovery
diagnostics
~~~

This prevents browser/host-restricted backends from being forced to implement synchronous blocking fence waits.

---

## 41.12 Submission freeze decision

0.16 stable semantics:

~~~text
opaque plan identity / PlanPoint
logical ordered lanes
explicit cross-lane happens-before
GPU dependency or lane collapse
preflight before native submit
accepted != complete
overall + per-point completion
structured async terminal failure
completion-safe readback / retirement
~~~

Explicitly absent from the public API:

~~~text
Fence
Semaphore
Event
timeline value
queue-family index
native command queue
HostWait as lane dependency
~~~

---

## 41.13 Private execution enhancement route

The following are deliberately backend-private improvements to the frozen
submission contract. They do not justify a public fence, queue, barrier,
descriptor heap, command packet, or synchronization API.

| Improvement | Existing carrying semantics | Backend obligation before it is enabled |
| --- | --- | --- |
| Batch-local state-difference tracker | ordered `command::ResourceUse` and PlanPoint dependencies | Produce every required state/memory transition, including cross-submit uses, without changing the semantic use trace. |
| Shared fence/completion waiter | `CompletionPoint` terminal observation | Multiplex native waits and wake every completion/readback/present/idle waiter on success, failure, and DeviceLost. |
| Narrower submit synchronization | Phase A/Phase B acceptance contract | Preserve cross-plan hazard validation and ensure no accepted work is reported as a rejected plan. |
| Multi-queue lowering | `SubmissionLane` and `LaneDependencyRoute` | Use native queue overlap only for routes that can be proven; otherwise retain ordered or Collapse lowering. |
| Recorder arena/command packet cache | synchronous recorder and reconstructable `RecordedWork` | Preserve recorder validation, command order, actual uses, diagnostics, and tooling descriptions. |
| Descriptor retirement | last actual-use completion rule | Keep all native descriptor/table backing live through the relevant completion point. |

The implementation may introduce private `TODO` markers for one of these
improvements, but the marker must state its current correct fallback and must
not be reachable from a public call as `todo!()`/`unimplemented!()`. A backend
must not advertise a capability or route merely because a private type exists;
unsupported optional routes return structured `Unsupported`.


# 42. Presentation surface facts

This chapter is **FROZEN / P0**.

Surface Facts must be queried by:

```text
Device + PresentationTarget
```

Presentation must not be degraded into a Device-global boolean.

---

## 42.1 PresentationTarget / configuration lease

```rust
#[derive(Clone)]
pub struct PresentationTarget {
    /* opaque host/platform target */
}

impl PresentationTarget {
    pub fn id(&self) -> ObjectId;
}
```

`PresentationTarget` is a host object and does not belong to the Device execution domain.

The same target may undergo capability preflight by multiple Providers/Devices,
but **there may be only one active `ConfiguredPresentation` lease at a time**.

Therefore:

```text
Device A configure(target) -> active lease

At this point, Device B / Device A configures the same target again
    -> InvalidUsage / Unsupported host ownership
```

To change backend/device, the old configuration must first be dropped/closed.

Portable RHI does not expose:

```text
HWND
CAMetalLayer
VkSurfaceKHR
canvas/context
```

---

## 42.2 PresentMode

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentMode {
    Automatic,
    Fifo,
    Mailbox,
    Immediate,
}
```

`Automatic` is the only portable mode that every presentation backend must support.

It does not promise mapping to a particular fixed native mode.

Browser WebGPU/GL may expose only:

```text
[Automatic]
```

Do not disguise host/compositor policy as FIFO.

---

## 42.3 Extent ownership

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent2d {
    pub width: u32,
    pub height: u32,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationExtentControl {
    HostManaged {
        current: Option<Extent2d>,
    },

    Configurable {
        min: Extent2d,
        max: Extent2d,
    },
}
```

`HostManaged`:

```text
browser canvas
adopted GL context
some surface/window paths
```

`Configurable`:

```text
the RHI may request an exact drawable extent
```

The concrete backend mapping is private.

---

## 42.4 PresentationTargetCapabilities

P0 **no longer exposes an ordinary drawable TextureView**.

The reason is that the ordinary `TextureView` contract requires it to belong to an ordinary `Texture`,
while a GL/WebGL2 default framebuffer is not a Texture at all;
forcibly creating a “surface Texture” for a few backends would pollute resource identity/lifetime/inventory.

P0 freezes only:

```rust
#[derive(Clone, Debug)]
pub struct PresentationTargetCapabilities {
    /* opaque snapshot */
}

impl PresentationTargetCapabilities {
    pub fn formats(&self) -> &[TextureFormat];
    pub fn present_modes(&self) -> &[PresentMode];
    pub fn extent_control(&self) -> PresentationExtentControl;
}

impl Device {
    pub fn presentation_capabilities(
        &self,
        target: &PresentationTarget,
    ) -> RhiResult<PresentationTargetCapabilities>;
}
```

If a real consumer needs the following in the future:

```text
sample/copy/read back an acquired drawable
```

design a `PresentationTexture` extension separately;
do not sneak it into an ordinary `TextureView`.

---

## 42.5 Snapshot semantics

Surface Facts are snapshots at query time.

All of the following changes may make them stale:

```text
resize
display move
host context recreation
compositor change
surface lost/outdated
```

Therefore:

```text
query capability
    !=
permanent guarantee that configure succeeds
```

`configure/reconfigure` must validate again.

---

## 42.6 Do not freeze image count

P0 does not expose:

```text
min_image_count
max_image_count
desired_image_count
```

This is because WebGPU/GL/Metal/native swapchains do not have a unified buffering-ownership model.

Review frame-pacing / latency extensions separately in the future.

---

# 43. Presentation configuration

## 43.1 Extent request

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationExtent {
    HostManaged,
    Exact(Extent2d),
}
```

`Exact` is legal only when the capability is `Configurable`.

---

## 43.2 Configuration

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PresentationConfiguration {
    format: TextureFormat,
    present_mode: PresentMode,
    extent: PresentationExtent,
}

impl PresentationConfiguration {
    pub fn new(
        format: TextureFormat,
    ) -> Self;

    pub fn with_present_mode(
        self,
        mode: PresentMode,
    ) -> Self;

    pub fn with_extent(
        self,
        extent: PresentationExtent,
    ) -> Self;

    pub fn format(&self) -> TextureFormat;
    pub fn present_mode(&self) -> PresentMode;
    pub fn extent(&self) -> PresentationExtent;
}
```

P0 configuration guarantees only:

```text
FrameAttachment may be used as the final color render target
```

It does not guarantee:

```text
sampled
copy source/destination
storage
readback
```

---

## 43.3 ConfiguredPresentation

```rust
pub struct ConfiguredPresentation {
    /* opaque, owns target configuration lease */
}

impl ConfiguredPresentation {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn target_id(&self) -> ObjectId;
    pub fn configuration(&self) -> &PresentationConfiguration;

    /// Reconfigure on the same Device + target.
    ///
    /// There must be no outstanding frame.
    pub async fn reconfigure(
        &mut self,
        config: &PresentationConfiguration,
    ) -> RhiResult<()>;
}

impl Device {
    pub async fn configure_presentation(
        &self,
        target: &PresentationTarget,
        config: &PresentationConfiguration,
    ) -> RhiResult<ConfiguredPresentation>;
}
```

`configure/reconfigure` must validate again:

```text
target lease
format
present mode
extent ownership
zero-size/suspended
target/device loss
```

---

## 43.4 One outstanding frame — P0 rule

P0 permits at most **one outstanding `AcquiredFrame`** per `ConfiguredPresentation`.

Reasons:

- Do not freeze swapchain image count;
- Prevent upper layers from implicitly depending on a 2/3-buffer count;
- Web/Metal/GL/native backends can all implement it;
- Frame pipelining is still completed by GPU work and scheduling of the next acquire.

Therefore:

```text
acquire()
    -> outstanding = 1

acquire() again
    -> FrameOutstanding
```

Only after that frame:

```text
enters an accepted present plan
or is explicitly abandoned
or device/target loss occurs
```

may the next acquire be allowed.

If a real consumer needs multiple outstanding acquired frames in the future,
review it as a frame-pacing extension.

---

# 44. Acquire / FrameAttachment

## 44.1 Acquire error

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquireErrorKind {
    FrameOutstanding,
    ZeroSizeOrSuspended,
    Outdated,
    TargetLost,
    DeviceLost,
    OutOfMemory,
}

#[derive(Debug)]
pub struct AcquireError {
    kind: AcquireErrorKind,
    message: String,
}

impl AcquireError {
    pub fn kind(&self) -> AcquireErrorKind;
    pub fn message(&self) -> &str;
}
```

---

## 44.2 Frame identity / state

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AcquiredFrameId {
    device: DeviceIdentity,
    serial: u64,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquiredFrameState {
    Acquired,
    PlannedForPresent,
    PresentAccepted,
    Abandoned,
    Outdated,
    TargetLost,
    DeviceLost,
}
```

`AcquiredFrameId` has no public constructor.

---

## 44.3 FrameAttachment

```rust
#[derive(Clone)]
pub struct FrameAttachment {
    /* opaque reference to AcquiredFrame state */
}

impl FrameAttachment {
    pub fn frame_id(&self) -> AcquiredFrameId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn format(&self) -> TextureFormat;

    /// Current drawable texel extent.
    pub fn extent(&self) -> Extent3d;

    /// P0 presentation frames are fixed to single-sample.
    pub fn sample_count(&self) -> u32;
}
```

FrameAttachment guarantees only:

```text
color render attachment
```

It is not a Texture and cannot enter:

```text
BindGroup
copy_buffer_to_texture
copy_texture
readback
storage
```

---

## 44.4 AcquiredFrame

```rust
pub struct AcquiredFrame {
    /* opaque, non-Clone owner token */
}

impl ConfiguredPresentation {
    /// Non-blocking fast path.
    pub fn try_acquire(
        &mut self,
    ) -> Result<Option<AcquiredFrame>, AcquireError>;

    /// Waits until the next drawable/frame can be acquired.
    pub async fn acquire(
        &mut self,
    ) -> Result<AcquiredFrame, AcquireError>;
}

impl AcquiredFrame {
    pub fn id(&self) -> AcquiredFrameId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn state(&self) -> AcquiredFrameState;

    pub fn attachment(&self) -> FrameAttachment;

    /// Explicitly means “do not present this frame.”
    ///
    /// This is a lifecycle escape and does not promise to be cheap.
    /// The backend may release the acquired image, retire/recreate the swapchain,
    /// drop the drawable, and so on, to ensure that the presentation system does
    /// not retain the frame permanently.
    pub async fn abandon(
        self,
    ) -> RhiResult<()>;
}
```

Remove the old APIs:

```text
drawable_view()
discard()
```

### Why `abandon` instead of `discard`

“discard” is easily understood to mean:

```text
every backend has a cheap release-acquired-image primitive
```

This is not true.

Vulkan's explicit `vkReleaseSwapchainImagesKHR` is a maintenance1 capability; the ownership in the standard acquire→present lifecycle is released by present after acquisition. (See References at the end.)

Therefore, the portable semantic promises only:

> Do not present; this RHI is responsible for safely terminating frame ownership.

It does not promise backend cost.

---

## 44.5 Drop safety

The normal path requires an explicit:

```text
present_after(frame, ...)
or
frame.abandon()
```

But Rust `Drop` cannot return an error.

If an `AcquiredFrame` in the `Acquired` state is directly dropped:

```text
the RHI must perform no-throw abandonment bookkeeping
emit a DiagnosticEvent
when necessary, mark ConfiguredPresentation as Outdated/NeedsRecovery
```

It must not leak an acquired image / drawable permanently.

Before cleanup/reconfigure completes, `try_acquire()` may return `None`; an
async acquire may report:

```text
Outdated
```

---

## 44.6 Attachment validity

`FrameAttachment` may be Clone, but it is not an ownership token.

Every command that uses it must check:

```text
frame state == Acquired
or
belongs to the same present-plan closure that is being built and has not yet been accepted
```

All of the following must fail:

```text
recording the same FrameAttachment again after present is accepted
continuing to record after abandon
continuing to record after target/device loss
use by another Device
```

Errors:

```text
InvalidUsage / WrongDevice / TargetLost / DeviceLost
```

No stale native drawable/swapchain image will be sent into the backend.

---

# 45. Present planning / outcome

## 45.1 Present plan identity

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentPlanId {
    plan: SubmissionPlanId,
    local: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentReceiptId {
    device: DeviceIdentity,
    serial: u64,
}
```

There is no public constructor.

After `present_after()` consumes `AcquiredFrame`:

```text
Acquired -> PlannedForPresent
```

---

## 45.2 Frame use enters the ResourceUse model

A Frame is not a Texture, so actual use must be a separate variant:

```rust
command::ResourceUse::Frame(
    FrameAttachmentUse {
        frame,
        stages,
        access,
    }
)
```

Raster:

```text
Load       -> COLOR_READ
Clear/draw -> COLOR_WRITE
```

This preserves actual RHI resource-use validation without disguising a frame as a Texture.

---

## 45.3 Plan closure validation

For every frame, `SubmissionPlanBuilder::build()` validates:

```text
at most one PresentPlan per AcquiredFrame

any SubmissionPlan containing a FrameAttachment GPU use
    must also contain that frame's PresentPlan

all RecordedWork in this plan that references that FrameAttachment
    belongs to that PresentPlan closure

all frame-use PlanPoints
    must happen-before the point specified by present_after
    or be that point

there must be no work using the frame after the present point
```

The RHI automatically adds the acquire dependency:

```text
native acquire readiness
    ->
first frame-use batch
```

If the frame has no GPU use and is presented directly:

```text
acquire dependency
    ->
present
```

Ordinary callers do not touch acquire semaphores/fences.

Vulkan itself requires waiting for acquire synchronization before an application uses an acquired image; Metal also associates drawable presentation with command-buffer scheduling. (See References at the end.)

---

## 45.4 Present state

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PresentFailure {
    message: String,
}

impl PresentFailure {
    pub fn message(&self) -> &str;
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum PresentState {
    Pending,

    /// The presentation system / host lifecycle has accepted and consumed frame ownership.
    ///
    /// This does not mean scan-out/display completion.
    Accepted,

    Outdated,
    TargetLost,
    DeviceLost(DeviceLossInfo),
    Failed(PresentFailure),
}
```

```rust
pub struct PresentReceipt {
    id: PresentReceiptId,
    plan_id: PresentPlanId,
}

impl PresentReceipt {
    pub fn id(&self) -> PresentReceiptId;
    pub fn plan_id(&self) -> PresentPlanId;
}

impl Device {
    pub fn present_state(
        &self,
        receipt: PresentReceiptId,
    ) -> RhiResult<PresentState>;

    /// Waits for presentation ownership to reach a terminal outcome.
    pub async fn wait_present(
        &self,
        receipt: PresentReceiptId,
    ) -> RhiResult<PresentState>;
}
```

---

## 45.5 GPU completion and present outcome are independent

```text
submit(plan) accepted
    ├─ CompletionPoint
    │      -> GPU work terminal
    │
    └─ PresentReceipt
           -> presentation ownership outcome
```

The following may occur:

```text
GPU Complete
Present Outdated

GPU Complete
Present TargetLost

GPU DeviceLost
Present DeviceLost
```

Present failure must not be written back as:

```text
work was not submitted
```

`Accepted` also does not mean “the screen has already displayed it.”

Metal's `present(_:)` schedules drawable presentation during command-buffer scheduling; it is not a portable scan-out-completion primitive. (See References at the end.)

---

## 45.6 Submit rejection / partial failure

If `SubmissionPlanBuilder::build()` fails after consuming a frame, or `Device::submit(plan)` returns Err **before any native work is accepted**:

```text
frame -> no-submit terminal path
the presentation lease must be safely released/recovered
```

If the Builder itself is dropped before `build()` succeeds, its owned consumed
frames take this same no-submit terminal path. This bookkeeping is an abandon /
recovery action only; it does not submit GPU work or present the frame.

If some work has already been accepted:

```text
submit returns Receipt
the frame/present eventually terminates through PresentState / CompletionState
```

Frame ownership must not be lost.

---

# 46. Presentation final route

## 46.1 P0 canonical final route

Because P0 neither disguises FrameAttachment as Texture
nor provides a frame copy/blit command:

```text
the portable Base route for final output to a presentation frame
    =
RasterScope color attachment
```

Typical postprocessing:

```text
scene / postprocess
    -> ordinary intermediate Texture

final fullscreen raster
    -> FrameAttachment

present
```

MSAA route:

```text
multisampled color Texture
    -> RasterScope resolve
    -> FrameAttachment
```

Both routes cover DX12 / Vulkan / Metal / WebGPU / GL/WebGL2.

---

## 46.2 Do not secretly copy/blit

If a backend can do the following in the future:

```text
Texture -> swapchain image direct copy
blit to default framebuffer
sample acquired drawable
```

P0 does not use these capabilities to alter public semantics.

When a real consumer requires performance optimization, add:

```text
PresentationTexture / PresentationCopyRoute
```

as an independent capability family.

This follows:

> Do not constrain future lower-layer capabilities to the greatest common denominator; but do not prematurely pollute the Base API when no complete five-platform contract exists.

---

## 46.3 Configuration / frame drop

Dropping `ConfiguredPresentation`:

```text
releases the target configuration lease
```

If an outstanding frame remains:

```text
first enter the no-throw abandonment/recovery path
then release the lease
```

The target must not be left permanently in an “acquired frame already exists” state.

---

## 46.4 Presentation freeze decision

0.16 frozen semantics:

```text
Device + Target surface facts
exclusive configuration lease
Automatic present policy
HostManaged / Exact extent
one outstanding frame
FrameAttachment != Texture
explicit acquire dependency
Frame use is first-class ResourceUse
present is part of SubmissionPlan
GPU completion != presentation outcome
safe abandon/drop recovery
final Base route = raster to FrameAttachment; direct MSAA resolve only when facts prove it
```

Explicitly deferred:

```text
surface drawable TextureView
direct copy/blit to frame
multiple outstanding acquired frames
frame latency/buffering controls
HDR metadata
present timing / scan-out completion
XR/custom compositor
```
