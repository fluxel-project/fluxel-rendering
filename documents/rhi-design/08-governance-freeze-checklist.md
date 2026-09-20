# RHI API freeze v13. Governance and freeze checklist

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and every affected module before using this checklist. Deferred
> names are not P0 support claims and must not be predeclared as empty API.

# 59. Capability traits explicitly absent from the public API

0.16 does not export:

```rust
trait IndirectApi;
trait QueryApi;
trait BindlessApi;
trait RayTracingApi;
trait MeshShaderApi;
trait AsyncComputeApi;
trait MultiviewApi;
trait DeviceAddressApi;
trait SparseResourceApi;
```

It also does not predefine:

```rust
struct NativeQueue;
struct NativeFence;
struct NativeSemaphore;
struct NativeBarrier;
struct DescriptorHeap;
struct GpuAddress(u64);
```

---

# 60. Deferred feature gate

Before any P1/P2 family enters the public API, it must submit:

```text
1. A real consumer
2. A DX12 / Vulkan / Metal / WebGPU / GL capability matrix
3. The owner layer
4. Portable semantics
5. Facts / limits / routes
6. Fallback / unsupported behavior
7. Lifetime / synchronization
8. Validation
9. Tests
10. Capture/Replay implications
```

### P1 candidates

```text
Typed/texel BufferView
Query/timestamp
Indirect/multi-draw/count
Inline parameters
General mapping
External resource import/export
Pipeline cache identity
HDR / frame pacing
Transient alias implementation
```

### P2 candidates

```text
Bindless
Mesh/task/geometry/tessellation
Ray tracing
Sparse/tiled/residency
Device address
Work graphs
GPU-generated commands
External memory/sync
Cross-device interop / multi-GPU resource sharing
XR compositor
Crash dump / breadcrumbs
```

---

# 61. Minimal direct-RHI example

```rust
async fn render_one_frame(
    provider: &PlatformProvider,
    target: PresentationTarget,
) -> RhiResult<()> {
    let device = provider.request_device(
        DeviceRequestDescriptor::new(
            AdapterSelection::PreferHighPerformance,
            DeviceRequirements::new(),
        ).require_presentation_target(target.clone()),
    ).await?;

    let format = device.presentation_capabilities(&target)?.formats()[0];
    let mut surface = device.configure_presentation(
        &target, &PresentationConfiguration::new(format),
    ).await?;

    let vertex_shader = device.create_shader(&make_vertex_artifact()).await?;
    let fragment_shader = device.create_shader(&make_fragment_artifact()).await?;
    let interface = device.create_pipeline_interface(
        &PipelineInterfaceDescriptor::new(vec![]),
    )?;
    let pipeline = device.create_raster_pipeline(
        &RasterPipelineDescriptor::new(vertex_shader, interface)
            .with_fragment(fragment_shader)
            .with_color_target(ShaderLocation::new(0), ColorTargetState::new(format)),
    ).await?;

    let frame = surface.acquire().await?;
    let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
    let mut raster = recorder.begin_raster(
        &RasterScopeDescriptor::new().with_color(
            ShaderLocation::new(0),
            ColorAttachment {
                view: ColorAttachmentView::Frame(frame.attachment()),
                load: LoadOp::Clear(ColorClearValue::Float([0.1, 0.1, 0.1, 1.0])),
                store: StoreOp::Store,
                resolve: None,
            },
        ),
    )?;
    raster.set_pipeline(&pipeline)?;
    raster.draw(0..3, 0..1)?;
    raster.end()?;

    let work = recorder.finish()?;
    let lane = device.capabilities().submission().lanes().iter()
        .find(|lane| lane.domains().contains(LaneWorkDomains::RASTER))
        .ok_or_else(|| RhiError::unsupported("no raster lane"))?.id();
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan.add_batch(lane, vec![work])?;
    plan.present_after(frame, point)?;
    let receipt = device.submit(plan.build()?).await?;

    match device.wait_completion(receipt.completion()).await? {
        CompletionState::Complete => {}
        CompletionState::DeviceLost(info) => return Err(RhiError::device_lost(info.message())),
        CompletionState::Failed(err) => return Err(RhiError::backend(err.message())),
        CompletionState::Pending => unreachable!(),
    }
    Ok(())
}
```

Direct-RHI users do not write:

```text
ResourceUse declaration
barrier
fence
semaphore
native queue
swapchain image state
```

The RHI generates `rhi::command::ResourceUse` and hazard semantics exclusively from actual portable commands. RenderGraph remains an RHI client and is not part of the RHI public object model.

---

## 61.1 Minimal transient example

```rust
async fn transient_example(
    device: &Device,
    lane: SubmissionLaneId,
) -> RhiResult<()> {
    let mut plan = SubmissionPlanBuilder::new(device);

    let produce = plan.reserve_batch(lane)?;
    let consume = plan.reserve_batch(lane)?;
    plan.add_dependency(produce, consume)?;

    let transient = plan.transient_allocator();
    let temp = transient.create_texture(
        &TextureDescriptor::new_2d(
            1920,
            1080,
            TextureFormat::Rgba16Float,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
        ),
        TransientLifetime::new(produce).release_at(consume),
    )?;

    let view = device.create_texture_view(
        &temp,
        &TextureViewDescriptor::whole(
            &temp,
            TextureViewDimension::D2,
        )?,
    )?;

    // Record work that uses `view`, then fill both reserved points.
    // plan.set_batch(produce, vec![...])?;
    // plan.set_batch(consume, vec![...])?;

    // Physical backing and aliasing are realized during async submit preflight.
    Ok(())
}
```

---

# 62. Lifecycle matrix

| Object | DeviceIdentity scoped | Clone | GPU-safe retirement | Canonically describable |
|---|---:|---:|---:|---:|
| Buffer | Yes | Yes | after completion/loss | Yes |
| Texture | Yes | Yes | after completion/loss | Yes |
| TextureView | Yes | Yes | after completion/loss | Yes |
| Sampler | Yes | Yes | after completion/loss | Yes |
| ShaderModule | Yes | Yes | after completion/loss | Yes |
| BindGroup | Yes | Yes | after completion/loss | Yes |
| Pipeline | Yes | Yes | after completion/loss | Yes |
| CommandRecorder | Yes | No | N/A | semantic tooling |
| RecordedWork | Yes | No | after completion/loss | command→use IR via tooling |
| CompletionPoint | Yes | Copy token | N/A | Yes |
| PresentReceiptId | Yes | Copy token | N/A | Yes |
| AcquiredFrame | Yes | **No** | present/abandon/loss | Yes |
| ReadbackTicket | Yes | Yes | completion-aware | Yes |
| DeviceStatistics | Yes | Yes | N/A | statistical observation; does not participate in Capture correctness |

Device loss is terminal; P0 recreation obtains a new `DeviceIdentity`.

---

# 63. Freeze checklist

## Platform / Device / Async

- [x] Provider is backend-family scoped; multiple backend Devices may coexist in one process.
- [x] `enumerate_adapters().await` and `request_device(...).await` are the async discovery/creation boundary; no public poll-state request object exists.
- [x] `Device::poll()` is only an opportunistic synchronous progress hook and normal completion does not require a busy loop.
- [x] `create_shader`, `create_raster_pipeline`, and `create_compute_pipeline` are async; logical resource and binding creation remain synchronous.
- [x] `submit`, `wait_completion`, and `wait_idle` are async operations.
- [x] Readback is `readback.read().await -> ReadbackView<'_>`; the RAII view closes any backend mapping lease on Drop.
- [x] Presentation configure/reconfigure/acquire/abandon/wait-present operations are async.
- [x] Concurrent invocation does not imply `async fn`: capability queries, descriptor validation, command recording, statistics, and diagnostics stay synchronous.

## Identity

- [x] `DeviceIdentity` contains opaque instance plus generation and names one
  terminal logical execution domain; P0 never revives it through generation++
  transparent recovery.
- [x] Tokens cannot be arbitrarily constructed by users.
- [x] Completion/Present tokens are device-scoped.
- [x] Device loss terminates that identity; requesting again obtains a new identity.

## Platform / Adapter / Device

- [x] One `PlatformProvider` corresponds to one backend family.
- [x] `request_device().await` is the canonical path; adapter enumeration is optional async inspection.
- [x] WebGPU / adopted contexts are not forced to fabricate complete Adapter enumeration.
- [x] `AdapterId` is a Provider-scoped opaque token, not an index/native handle.
- [x] The presentation requirement enters the request at Device creation.
- [x] Device-request futures are not bound to a specific async runtime.
- [x] Device loss wakes pending completion/readback/acquire/present/wait-idle
  into terminal states; v13 has no `Device::lost()` future or separate public
  loss-event API, and idle loss may first be observed by the next RHI call.
- [x] `wait_idle().await` is only for shutdown/diagnostics.

## Capability / Format / Route / Surface

- [x] Capability is instance data, not a capability trait.
- [x] AvailableOnAdapter and EnabledOnDevice are separate.
- [x] RenderGraph may use an opaque `CapabilityFingerprint` for plan invalidation.
- [x] Limit requirements distinguish `AtLeast` and `AtMost`, and do not reverse the direction of minimum alignment.
- [x] No capability bool duplicates Feature/Limit/TextureSupport.
- [x] Buffer creation also uses `BufferSupportQuery`; usages such as STORAGE are not deferred until after creation.
- [x] `FormatFacts` and `TextureSupportQuery` are layered separately.
- [x] Sampleable and filterable are not conflated into one bool; `TextureSampleType` is used.
- [x] Storage read/write/read-write are expressed separately.
- [x] Texture support includes dimension / usage / sample count / view formats / view compatibility intent.
- [x] TextureSupport returns descriptor-specific maximum extent / mip / array-layer limits.
- [x] TextureView format compatibility is queried separately.
- [x] Buffer copy offset/size alignment and texel-copy row/offset alignment are both Route Facts.
- [x] The Route key includes texture dimension/sample count that affect legality; `rows_per_image_alignment` is removed.
- [x] An unsupported route does not permit the backend to silently insert a shader/CPU fallback.
- [x] Lane legality uses explicit `LaneWorkDomains`; it is not inferred from lane class.
- [x] Lane dependencies are queried per pair and distinguish Ordered/Gpu/Collapse.
- [x] There is no `supports_real_overlap`.
- [x] Surface Facts are queried by Device + PresentationTarget; P0 does not fabricate drawable TextureView usage.
- [x] PresentMode has `Automatic`; WebGPU/host compositor is not disguised as FIFO.
- [x] Extent ownership distinguishes HostManaged / Configurable.
- [x] The portable API does not freeze a swapchain image-count contract.
- [x] A Surface Fact is a snapshot and is revalidated at configuration.
- [x] Sampler anisotropy is expressed through Feature + Limit, not guessed from backend names.
- [x] Binding kind/count/dynamic legality is queried through `BindingSupportQuery`.
- [x] Per-stage binding resource counts are queried through `binding_limit(stage, class)` and are no longer compressed into one all-stage minimum.

## Resource / Upload / Readback

- [x] Buffer / Texture creation both have descriptor-dependent capability queries.
- [x] Buffer size > 0 and usage is non-empty; zero sizes are not accepted by relying on the backend.
- [x] P0 Buffer is byte-addressed; `element_stride_hint` is removed.
- [x] P0 has no public map; therefore `HostAccessIntent / HostPreferred` are removed.
- [x] Memory placement retains only the `Automatic / DeviceLocalPreferred` performance hint.
- [x] Texture `extent.depth` and `array_layers` are separate.
- [x] D1 / D2 / D3 / MSAA creation invariants are explicit.
- [x] Cube view compatibility is declared at Texture creation, not by adding a native flag at view creation.
- [x] `TextureSubresourceRange` and `TextureSubresourceLayers` are separate.
- [x] D2 array layers and D3 Z-slices are not conflated into `depth_or_layers`.
- [x] TextureView uses generic `whole(texture, dimension)`; `whole_2d()` is removed.
- [x] TextureView is a first-class object and validates view format/aspect/range/dimension/cube compatibility.
- [x] Sampler anisotropy is capability-gated; a WebGL extension does not impersonate Base.
- [x] Upload source `HostTexelLayout` and GPU copy route layout are separate.
- [x] Upload may privately repack/stage; an ordinary Copy command must not use an implicit CPU/shader fallback.
- [x] Upload / Readback enforce the COPY_DST / COPY_SRC usage contract.
- [x] The Readback ticket has a closed NotSubmitted/Pending/Ready/Abandoned/DeviceLost/Failed state machine.
- [x] Readback layout explicitly returns row/image stride and does not pretend to be tightly packed.
- [x] ReadbackTicket binds its Device itself; callers need not pass the Device again.
- [x] Resource backing retires safely with respect to completion.
- [x] Device loss terminates pending readback state.
- [x] Descriptor/Upload mutation/Readback layout satisfy Capture observability.
- [x] `ReadbackView` exposes the ready bytes and layout only while its RAII lease is held; it is not a bare borrowed slice detached from backend unmap requirements.

## Transient resource / aliasing

- [x] The public transient API is frozen in 0.16; it is not a deferred graph/service contract.
- [x] `TransientAllocationSupport::{Dedicated, Aliasing}` is capability data: Dedicated is the required correct fallback and Aliasing is an optimization.
- [x] `TransientLifetime { acquire: PlanPoint, release_frontier: Vec<PlanPoint> }` uses only RHI execution vocabulary.
- [x] `SubmissionPlanBuilder::reserve_batch`, `set_batch`, and `transient_allocator` allow lifetime reservation before recording work.
- [x] Transient allocation returns ordinary Buffer/Texture handles, is bound to its plan, and does not fork binding or command APIs.
- [x] Backends lower aliasing from PlanPoint ordering, actual `rhi::command::ResourceUse`, and physical alias relations; native heaps, offsets, and barriers remain private.
- [x] Aliasing is permitted only for proven non-overlap and otherwise falls back to Dedicated; realization occurs during async submission preflight before acceptance.

## Shader / Binding / Pipeline

- [x] GL/GLSL, WebGL2/GLSL ES, WebGPU/WGSL, Vulkan/SPIR-V, DX12/DXIL, and Metal/MSL/metallib can all be expressed.
- [x] Backend-consumable ShaderCode is separate from replay/toolchain provenance.
- [x] `ShaderAbiVersion` makes clear that logical group/slot is not native register/set/index.
- [x] The RHI does not take responsibility for shader cross-compilation.
- [x] P0 artifacts retain no unresolved pipeline specialization constants.
- [x] ShaderInterface uses the same `BindingKind/BindingCount` as the Binding API; it does not maintain a second ShaderBindingKind.
- [x] ShaderInterface includes the location/type/components/interpolation metadata required for Vertex->Fragment stage IO linkage.
- [x] ShaderInterface resource and location arrays are duplicate-free and
  canonically ordered before hashing, stage merge, capture, or pipeline use.
- [x] Artifact identity includes producer/toolchain identity and a defined
  canonical hash domain that excludes labels; compiler options are unique and
  sorted, and executable-only provenance carries an explicit acceptance scope.
- [x] ShaderRequirements uses `LimitRequirement` and does not reintroduce the error that all limits are minimums.
- [x] Compute requirements explicitly reflect workgroup dimensions, total
  invocations, and workgroup storage bytes and are validated before backend
  pipeline creation.
- [x] Binding legality uses `BindingSupportQuery`; per-stage/dynamic aggregate limits are validated uniformly at PipelineInterface level, not misjudged at an individual BGL level.
- [x] Dynamic offsets are valid only for buffers, their type is frozen as `u32`, and ordering is canonicalized by slot/array element.
- [x] BindGroup is an immutable, logical, validated packet.
- [x] BindGroupLayout / PipelineInterface compatibility uses a Device-scoped exact intern token; hash/fingerprint alone does not carry correctness.
- [x] The vector index of PipelineInterface groups is explicitly equal to `BindGroupIndex`.
- [x] Vertex input can be validated numerically by components against ShaderInterface inputs.
- [x] Strip topology's pipeline-time index format is explicitly modeled.
- [x] Blend constant factors align with the `set_blend_constant()` API.
- [x] The multisample mask uses 32-bit portable semantics.
- [x] Color target / Raster attachment support sparse locations: `Vec<Option<...>>`.
- [x] Raster pipeline descriptors / Compute pipeline descriptors both have public constructors/builders.
- [x] Before entering the backend, Raster pipelines validate shader resources, stage linkage, vertex input, fragment outputs, target facts, alpha-to-coverage, and limits.
- [x] `PipelineInterfaceDescriptor` has no reserved fake field.
- [x] Multiview is removed from P0 rather than half-frozen.

## Recording / Raster / Compute / Copy

- [x] Ordinary Recorders accept no external scheduling contract and are not bound to a submission lane.
- [x] The Recorder state machine explicitly distinguishes Open / RasterScope / ComputeScope / Poisoned.
- [x] Parameter validation errors do not automatically poison; only backend finalize/internal failure poisons.
- [x] Scopes must explicitly `end()`; Drop does not perform potentially failing native finalization.
- [x] Raster attachments retain only one definition and support sparse color locations.
- [x] Depth/Stencil read-only uses mode and no longer allows the conflicting `read_only + Clear` combination.
- [x] Raster attachment format/extent/layer/sample-count invariants are checked in `begin_raster()`.
- [x] Clear values match the attachment numeric class; depth clear is finite
  and within `[0, 1]`.
- [x] A FrameAttachment primary color target must Store.
- [x] `set_bind_group` uses `BindGroupIndex + &[u32] dynamic offsets`.
- [x] viewport/scissor/blend constant/stencil reference all have portable defaults.
- [x] Only draw/dispatch consume shader bindings and generate shader actual use.
- [x] A shader-reachable `Fixed(n)` binding array conservatively marks all `n`
  elements actually used unless certified static element metadata exists.
- [x] Extra BindGroup entries that are not referenced by the shader do not create a hazard.
- [x] Attachment Clear/Load/Store/Discard produce correct semantics even without a draw.
- [x] Copy/Resolve/Blit explicitly define usage, route, format/aspect/sample-count, and overlap rules.
- [x] Direct Resolve / Blit P0 freezes only Color semantics.
- [x] Direct resolve carries explicit source and destination origins and
  validates origin-plus-extent ranges.
- [x] Upload host layout and GPU copy layout remain layered separately.
- [x] Debug groups must be balanced and cannot carry state across scopes covertly.
- [x] Recorder-open, RasterScope, and ComputeScope own independent debug stacks;
  each corresponding finish/end operation requires its own stack to be empty.
- [x] Upload/readback command actual uses are COPY-domain GPU uses; P0 `ResourceUse` contains no graph/host scheduling vocabulary.
- [x] RHI internals retain a command-level actual-use sequence; the public summary does not carry synchronization lowering.
- [x] RecordedWork exposes `work_domains()` for Submission lane legality validation.
- [x] RecordedWork strongly retains execution dependencies.
- [x] `ResourceUse` is `rhi::command::ResourceUse`; RHI derives its command-level actual-use sequence without an external cross-check layer.

## Submission / Completion

- [x] `SubmissionPlan` is opaque; PlanPoint carries opaque plan identity and cannot be fabricated across builders/plans.
- [x] `add_batch` validates RecordedWork device + `work_domains()` against lane domains.
- [x] The same lane has logical order by batch insertion order; it still does not impersonate a memory barrier.
- [x] Different lanes are unordered by default; only an explicit dependency creates happens-before.
- [x] `HostWait` is removed from the P0 lane dependency route; CPU continuation belongs to a higher layer.
- [x] The dependency graph performs cycle/route/full-plan validation before native submission.
- [x] An unordered overlapping-write hazard in the same plan returns `MissingDependency`; `Device::submit` also checks pending prior submissions for cross-plan hazards.
- [x] A `CompletionPoint -> PlanPoint` external dependency resolves cross-plan
  happens-before through GPU wait, an already ordered domain, or a proven
  order-preserving collapse; only the absence of all three is `Unsupported`.
- [x] An Err from `Device::submit(...).await` guarantees that no plan work was accepted by the backend.
- [x] Once any native work is accepted, a subsequent immediate failure is reported through the Receipt/Completion terminal state rather than returning an ambiguous Err.
- [x] Builder ownership of a frame is total: `build()` failure or builder Drop
  performs no-submit abandonment/recovery bookkeeping.
- [x] Submission accepted != GPU complete.
- [x] CompletionState carries structured Failed / DeviceLost reasons.
- [x] Receipt provides both overall completion and `completion_for(PlanPoint)`.
- [x] A backend lacking fine-grained completion may let multiple PlanPoints share a more conservative token.
- [x] After a successful submission, ReadbackTicket binds the corresponding point completion.
      The portable half is in place: `ReadbackTicket::completion()` returns
      `Option<CompletionPoint>`, and a crate-private `set_completion` records the point once,
      first-write-wins, before the status advances. The call that performs the binding belongs
      to the submit path, which is not built yet — so the *binding* is unwitnessed until a
      backend port lands, while the contract and the accessor are frozen.
- [x] RHI retirement may depend on the completion of the last batch actually used, without forcing whole-plan completion.
- [x] Pending completion must become terminal after DeviceLost, not remain Pending forever.
- [x] P0 provides no blocking completion wait; `wait_completion().await` and `wait_idle().await` are async.
- [x] Present outcome is independent from GPU Completion, and `Accepted` does not mean scan-out completion.

## Presentation

- [x] `PresentMode` is in configuration; `Automatic` is the only all-platform-required policy.
- [x] Surface facts are queried by Device + target and are only snapshots.
- [x] A PresentationTarget has only one active configuration lease at a time.
- [x] `ConfiguredPresentation::reconfigure(...).await` requires no outstanding frame.
- [x] P0 permits at most one outstanding frame per ConfiguredPresentation and does not imply a swapchain image count.
- [x] AcquiredFrame is non-Clone and is the unique ownership token; FrameAttachment is only a Cloneable reference.
- [x] `drawable_view()` is removed; a surface image is not smuggled into an ordinary TextureView.
- [x] `discard()` is removed and replaced with the explicit `abandon().await` lifecycle escape; backend cost is not promised.
- [x] AcquiredFrame Drop has a no-throw recovery path and does not leak an acquired image permanently.
- [x] FrameAttachment has format/extent/sample-count, but is not a Texture.
- [x] FrameAttachment use state is validated with the frame lifecycle and cannot be reused after present/abandon.
- [x] FrameAttachment use is a separate `ResourceUse::Frame`, not a disguised TextureUse.
- [x] The RHI automatically joins acquire synchronization to the first frame-use/present point.
- [x] Any plan containing FrameAttachment GPU use must also consume the corresponding AcquiredFrame for presentation; all frame-use work happens-before the present point.
- [x] PresentState provides structured failure; Accepted does not mean scan-out completion.
- [x] GPU Completion and Present outcome are independent.
- [x] GL/WebGL2 default framebuffer is valid.
- [x] The unconditional P0 final route is a single-sample raster write to
  `FrameAttachment`; direct MSAA raster resolve to it requires proved
  presentation/route facts, otherwise an intermediate resolve plus final raster
  write is used.
- [x] Direct frame copy/blit, drawable texture, multi-acquired-frame, HDR/timing are all deferred.

## Statistics

- [x] Statistics are Device-scoped portable logical counters and do not impersonate a native profiler.
- [x] Changing `StatisticsDetail` starts a collection epoch, avoiding an invalid delta across detail levels.
- [x] A snapshot delta requires the same Device + same epoch; snapshots do not wait for the GPU.
- [x] Pipeline/shader/bind-group/buffer/texture/RT changes all have stable logical definitions.
- [x] A bind-group change includes dynamic offsets; shader-set uses an ObjectId tuple rather than a fingerprint.
- [x] RT switching is defined only within the same Recorder; it does not fabricate a global previous target across parallel recorders.
- [x] `logical_lane_changes` is removed; multiple lanes have no portable global adjacent order.
- [x] Lane usage is expressed as per-lane accepted batches/work + cross-lane/internal-external dependencies/collapse.
- [x] submission_calls / plans_accepted / rejected / batches_planned / accepted distinguish partial acceptance.
- [x] Acquire/Present use separate `PresentationStatistics` and are not mixed into submission counters.
- [x] Acquire-only refusal never increments a submitted-present terminal
  category; device loss remains a device/loss statistic.
- [x] WorkingSet uses actual-use unique IDs and includes FrameAttachment / ShaderModule / lane.
- [x] Inventory counts unique logical objects, not handle clones.
- [x] An outstanding frame enters inventory, but FrameAttachment clones are not counted repeatedly.
- [x] A render-target texture is an overlapping subset of Texture inventory.
- [x] Buffer / Texture memory fields are named `logical_estimated_bytes` and do not impersonate real VRAM.
- [x] The Texture estimate considers mip / extent / format block / layer / sample count and uses checked arithmetic.
- [x] Unknown backing such as `Depth24Plus` does not fabricate a fixed byte count.
- [x] A frame sampler fails directly across a collection epoch; FPS remains explicitly a caller-defined CPU sample rate.
- [x] Statistics inserts no GPU command, waits for no GPU work, and does not alter execution semantics.

## Capture / Replay readiness

- [x] Live RecordedWork must retain reconstructible portable semantics throughout its lifetime, not merely a native command handle.
- [x] An ObjectCreated event carries the borrowed definition directly, avoiding a reclaim-before-lazy-query race; a pre-existing object can still be lazily described.
- [x] Tooling object definitions use an ObjectId reference graph and do not contain live BindGroup/Shader/Pipeline handles.
- [x] A TextureView definition explicitly references a Texture ObjectId.
- [x] BindGroup/PipelineInterface/Pipeline definitions explicitly reference dependency ObjectIds.
- [x] ToolingAccess supports lazy `describe_object()` so objects that existed before capture scope begins can receive definitions.
- [x] ToolingAccess supports `describe_work(ObjectId)` and does not depend on the caller still holding a RecordedWork handle.
- [x] Upload mutation includes destination/layout/retained bytes.
- [x] A Readback request is observable; snapshots/checkpoints continue to reuse the formal Readback API.
- [x] CapturedCommand binds each PortableCommand to its actual uses; every value contains only logical IDs, not native handles.
- [x] Submission tooling IR retains all batch/work/lane/internal+external dependency/present relationships, not only counts.
- [x] `CapturedObjectDefinition` describes presentation-target fixtures and
  configured-presentation leases, so every target/configuration `ObjectId` in a
  `FrameAcquired` event is resolvable.
- [x] Tooling subscription start/drop is linearized; drop drains in-flight
  callbacks, callbacks cannot self-drop, mutate-reenter the same device, wait on
  GPU completion, or block for long periods.
- [x] CompletionState / PresentState / DeviceLost all have terminal semantic events.
- [x] SemanticEventId represents CPU observation order only and does not impersonate GPU order.
- [x] Observer callback lifetime/non-reentrancy/no-GPU-wait constraints are explicit.
- [x] Tooling SPI and Artifact schema are versioned separately.
- [x] The RHI does not implement artifact storage / dependency closure / snapshot policy / ReplayRuntime.
- [x] The ordinary Replay source of truth is captured RHI semantics; FrozenGraphIR is for provenance/comparison.

---

# 64. Final decision

The stable public surface of Fluxel RHI 0.16 should be understood as:

```text
portable GPU execution vocabulary
+ async lifecycle operations
+ instance capability facts
+ persistent / transient resource model
+ strict validation
+ opaque logical handles
+ explicit hazard/dependency validation
+ logical submission/completion/presentation
+ portable logical statistics / inventory
```

It is not:

```text
wgpu-rs wrapper
Vulkan wrapper
DX12 wrapper
UE RHI clone
the greatest common denominator of all platforms
```

The frozen RHI module layout is:

```text
rhi
├─ platform / capability / format
├─ resource
│  └─ transient
├─ shader / binding / pipeline
├─ command                 (`ResourceUse`)
├─ submission / presentation
├─ statistics / diagnostics
└─ tooling                 (hidden SPI)

backend
└─ DX12 / Vulkan / Metal / WebGPU / GL
```

The relationship between Capture/Replay and the RHI:

```text
The RHI must make its portable semantics observable, describable, and reconstructible;
but Capture Artifacts, dependency closure, snapshot policy, and ReplayRuntime
belong to the tooling/runtime layer outside the RHI.
```

After this version, adding Query, Indirect, Bindless, RT, Sparse, and similar capabilities must only add new capability-gated vocabulary and corresponding facts/limits/routes; it must not change the RHI's foundational identity, resource, recording, submission, or presentation model.

---

## Final freeze completed

```text
Platform / Adapter / Device                    ✅ FROZEN
Async lifecycle contract                       ✅ FROZEN
Capability / Format / Route / Surface Facts   ✅ FROZEN
Persistent Resource / Upload / Readback       ✅ FROZEN
Transient Resource / Aliasing Contract         ✅ FROZEN
Shader / Binding / Pipeline                    ✅ FROZEN
Recording / Raster / Compute / Copy            ✅ FROZEN
Submission / Completion                        ✅ FROZEN
Presentation                                   ✅ FROZEN
Statistics                                     ✅ FROZEN
Capture / Replay tooling seam                  ✅ FROZEN
```

Next item:

```text
Global cross-review / public type closure / normative freeze v13
```

# 65. v13 global cross-review conclusions

This pass does not merely continue confirming earlier conclusions; it re-derives the design from the following six chains:

```text
A. Identity / lifetime / loss
B. Capability / descriptor legality
C. Resource -> Binding -> Pipeline -> Recording
D. Recording -> Hazard -> Submission -> Completion -> Present
E. Whether Statistics only observes and does not alter semantics
F. Whether Capture is reconstructible without controlling execution in reverse
```

## 65.1 Key corrections from v12 to v13

| Area | v12 | v13 |
|---|---|---|
| RHI module boundary | extra upper-layer bridge module | removed; RHI knows only its own types |
| Resource use | bridge namespace | `rhi::command::ResourceUse` |
| external work coverage | public RHI validation contract | removed from RHI |
| Transient | partial bridge/service | frozen first-class RHI API |
| Transient implementation | API coupled to aliasing | `Dedicated` baseline; `Aliasing` optimization |
| Transient lifetime | external opaque plan | `PlanPoint` acquire/release frontier |
| Device creation | hand-written poll state | `request_device().await` |
| Shader/pipeline | synchronous create | asynchronous create |
| Submit | synchronous | async, allowing deferred realization in preflight |
| Completion | query/poll centered | query plus `wait_completion().await` |
| Readback | borrowed bytes without a guard | async read plus RAII `ReadbackView` |
| Presentation | synchronous configure/acquire | async lifecycle |

### Additional P0 cross-review ledger

| Issue | Earlier-draft risk | v13 decision |
|---|---|---|
| Device generation | generation++ recovery semantics were defined without a recovery API | P0 loss is terminal; requesting again obtains a new DeviceIdentity |
| Capability fingerprint | a hash was described as carrying plan correctness | add exact `CapabilityCompatibilityId`; fingerprint is only for cache/tooling |
| Buffer copy alignment | the BufferToBuffer route did not expose offset/size alignment | add `BufferCopyLayoutLimits`; upload/readback/copy share legality |
| Texture route key | copy/blit routes did not include dimension/sample count | add dimension/sample count to RouteQuery |
| Texture mip legality | only device max was examined, without limiting to the complete mip chain of descriptor extent | add mip-chain validation |
| TextureView aspects | sampled/storage/attachment aspect rules were incomplete | P0 explicitly defines Color/Depth/Stencil aspect legality |
| Layout compatibility ID | BGL and PipelineInterface shared one token type | split into two typed exact tokens |
| Storage access merge | merging multi-stage access for the same binding was undefined | define a ReadOnly/ReadWrite merge lattice |
| Binding aggregate limits | per-pipeline limits were checked at the individual BGL stage, which is the wrong layer | a single BGL validates only local constraints; PipelineInterface aggregates stage/dynamic limits |
| Integer attachment clear | `LoadOp<Color>` could not precisely express integer RT clear | add `ColorClearValue::{Float,Sint,Uint}` |
| Basic raster state | common shadow-map depth bias was missing | P0 adds constant+slope depth bias; clamp is deferred |
| Fragment target linkage | it forced every shader output to have a target and did not constrain target-without-output | permit unused shader output; a target without output must have write mask NONE |
| Alpha-to-coverage | its relation to fragment/sample-mask/target0 was not closed | add target0 alpha/output/sample-mask validation |
| Compute diagnostics | ComputeScope lacked descriptor/label | add `ComputeScopeDescriptor` |
| Scope debug markers | draw/dispatch could not be marked inside a scope | add debug group/marker to Raster/Compute scopes |
| ResourceUse vocabulary | HOST/PRESENT were half-frozen while P0 had no corresponding execution path | remove them from P0 `rhi::command::ResourceUse` |
| Command semantic closure | attachment Store/Discard and similar contradictions could escape validation | recorder validates portable command semantics while deriving actual uses |
| Cross-lane hazards | direct RHI could omit a dependency and form a data race | build validates overlap/write hazards; a missing edge returns `MissingDependency` |
| Cross-plan in-flight hazard | the second submission could not see pending resource use from a prior plan | `Device::submit` performs global hazard preflight over pending submissions |
| Cross-plan multi-lane | Completion could not be the GPU predecessor of the next plan | add `add_external_dependency(CompletionPoint, PlanPoint)` |
| Frame ownership | work that used a frame could be submitted without including the frame in a present plan | forbid it: a plan with frame GPU use must also consume/present that frame |
| Statistics extensibility | adding fields to metric structs would break Rust semver | metric structs are `#[non_exhaustive]` |
| Capture object race | ObjectCreated held only an ID, allowing lazy describe to fail after reclaim | ObjectCreated directly borrows the definition |
| Capture command/use | commands and actual uses were flattened, losing their one-to-one ordering | `CapturedCommand { command, actual_uses }` |
| Capture pre-recorded work | `describe_work(&RecordedWork)` required the caller to retain a handle | change to `describe_work(ObjectId)` |
| Capture cross-plan dependency | tooling IR could represent only PlanPoint->PlanPoint | dependency source also supports a prior CompletionPoint |
| Document coherence | the minimal example still used removed legacy API | update everything to the v13 frozen API |

## 65.2 Pure-RHI source of truth

```text
Capability facts
    -> whether an operation is supported
Descriptors / interfaces
    -> what an object is
Portable commands + actual ResourceUse
    -> what was actually done
Submission dependencies
    -> which work happens before which work
Transient lifetime
    -> which short-lived resources may safely reuse backing
Completion / PresentState
    -> what finally happened
Statistics
    -> how much happened
Tooling
    -> how those semantics are copied for diagnostics/reconstruction
```

No other module owns barrier state, native state, or resource identity truth.

## 65.3 Final async boundary

```text
enumerate_adapters / request_device                       async
create_shader / create_raster_pipeline / create_compute_pipeline async
submit / wait_completion / wait_idle                      async
readback.read                                              async
configure / reconfigure / acquire / abandon / wait_present async
```

Logical resource creation, binding/layout/interface creation, command
recording, capability queries, state queries and `try_*` fast paths,
statistics snapshots, and diagnostics pulls remain synchronous. Concurrency
support by itself does not give an operation future-event semantics.

## 65.4 Transient cross-backend check

- DX12 may use placed resources, heap reuse, and aliasing barriers.
- Vulkan may use aliased device memory plus explicit execution/memory dependencies.
- Metal may use `MTLHeap` and resource aliasability strategies.
- WebGPU, OpenGL, and WebGL2 remain correct with
  `TransientAllocationSupport::Dedicated`.

The same public semantics therefore support both optimized native aliasing and
the minimum correct dedicated implementation without a later API change.

## 65.5 Final gates

```text
Pure RHI layering                    PASS
No upper scheduling contract in RHI PASS
Public type closure                  PASS
Async lifecycle closure              PASS
Multi-device structured errors       PASS
Capability single-source             PASS
Persistent resource closure          PASS
Transient resource closure           PASS
Resource/binding/pipeline closure     PASS
Command state machine                PASS
Cross-lane/cross-plan hazard safety  PASS
Completion/retirement closure        PASS
Presentation ownership closure       PASS
Statistics observer-only boundary    PASS
Capture reconstructability           PASS
Backend-private boundary             PASS
```

> **Fluxel RHI 0.16 public semantic API: FINAL FREEZE CANDIDATE V13.**

### Additional five-backend conclusions

### DX12 / Vulkan / Metal

Can fully lower:

```text
resource/view/binding/pipeline
raster/compute/copy
multiple logical lanes
GPU dependency
completion
presentation
```

However, the portable API does not expose their respective native queue/barrier/fence/descriptor/heap objects.

### WebGPU

Naturally contracts to:

```text
usually one logical lane
Unsupported where no direct general blit route exists
surface presentation policy primarily Automatic
completion may be coarse to queue submitted-work completion
```

It need not fabricate native multi-queue or present modes.

### OpenGL / WebGL2

Naturally contracts to:

```text
a single ordered lane
Compute: WebGL2 Unsupported; desktop GL according to provider capabilities
FrameAttachment may be only the default framebuffer
no explicit native barrier/fence vocabulary exposed to callers
```

Therefore, v13 remains a **capability-layered RHI**, rather than cutting native backends down to the lowest common public API.

### Additional source-of-truth notes

```text
Capability facts
    -> “whether it can be done”

Descriptor / Pipeline interface
    -> “what the object is”

Portable commands + command-level actual uses
    -> “what was actually done”

Submission dependencies
    -> “which work comes before which other work”

Completion / PresentState
    -> “what finally happened to GPU / presentation”

Statistics
    -> “how much happened”

Capture tooling
    -> “copies the above semantics in order for reconstruction”
```

No two layers simultaneously own the source of truth for barriers/native state.

### Additional detailed freeze gates

```text
Public type graph                      PASS
Multi-device misuse -> structured Err PASS
Device-loss lifecycle                 PASS
Capability single-source              PASS
Resource/binding/pipeline closure      PASS
Command state machine                 PASS
Cross-lane hazard safety              PASS
Cross-plan dependency + in-flight hazard PASS
Completion/retirement closure         PASS
Presentation ownership closure        PASS
Statistics observer-only boundary     PASS
Capture reconstructability            PASS
Backend-private boundary              PASS
```

### Change-control status

> **Fluxel RHI 0.16 public semantic API: FINAL FREEZE CANDIDATE V13.**

Afterwards, the 0.16 implementation phase permits:

```text
private lowering adjustments
backend internal structure adjustments
performance optimization
error message improvements
tests / diagnostic additions
```

Without a new API review, it does not permit:

```text
changing the frozen public semantics in this document
exposing native barrier/fence/queue to ordinary callers
changing Capability Facts back into trait presence
allowing a backend to silently execute an undeclared fallback
allowing Statistics/Capture to alter execution results
```

Query / Indirect / Inline Parameters / Bindless / RT / Sparse and similar features continue to be incrementally frozen as independent capability families.

---

# 66. References used for v13 cross-validation

This pass uses these materials only as evidence that native semantics can support the Fluxel contract; it does not copy their API shapes.

- Vulkan Synchronization and Cache Control: https://docs.vulkan.org/spec/latest/chapters/synchronization.html
- Vulkan Fundamentals / Queue Operation: https://docs.vulkan.org/spec/latest/chapters/fundamentals.html
- Vulkan Queue Guide: https://docs.vulkan.org/guide/latest/queues.html
- Vulkan Robustness: https://docs.vulkan.org/guide/latest/robustness.html
- Vulkan Shader Interfaces / limits: https://docs.vulkan.org/spec/latest/chapters/interfaces.html
- D3D12 `ExecuteCommandLists`: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-executecommandlists
- D3D12 command queue/list design: https://learn.microsoft.com/en-us/windows/win32/direct3d12/design-philosophy-of-command-queues-and-command-lists
- WebGPU specification: https://gpuweb.github.io/gpuweb/
- WebGPU `GPUQueue`: https://gpuweb.github.io/types/interfaces/GPUQueue.html
- WebGPU `GPUCanvasContext`: https://gpuweb.github.io/types/interfaces/GPUCanvasContext.html
- Metal `MTLCommandBuffer`: https://developer.apple.com/documentation/metal/mtlcommandbuffer
- Metal `MTLEvent`: https://developer.apple.com/documentation/metal/mtlevent
- Metal `MTLSharedEvent`: https://developer.apple.com/documentation/metal/mtlsharedevent

Validation date: 2026-09-19.
