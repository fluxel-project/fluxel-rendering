# RHI API freeze v13. Tooling and capture prerequisites

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and this module in full before implementation. This module
> freezes RHI reconstructability and tooling SPI, not artifact storage or
> ReplayRuntime.

# 52. Capture/Replay: Capabilities that RHI must implement

This chapter has entered the **FROZEN/P0 infrastructure contract**.

The full Capture/Replay product is not part of RHI core,
However, if RHI lacks the following capabilities, it will not be able to reliably implement portable capture/replay in the future.

---

## 52.1 Stable logical identity

All observable objects must have:

```text
ObjectId
DeviceIdentity
canonical/reconstructable semantic definition
```

Can't just have native handle.

Include at least:

```text
Buffer / Texture / TextureView / Sampler
ShaderModule
BindGroupLayout / BindGroup
PipelineInterface / RasterPipeline / ComputePipeline
UploadJob
ReadbackTicket
RecordedWork
Submission / Completion
AcquiredFrame / Present
```

---

## 52.2 Live RecordedWork must be rebuildable

Although the normal public API is not exposed:

```text
RecordedWork::portable_ir()
```

But **RHI internal RecordedWork must retain sufficient portable semantics** within its live lifetime,
Make tooling retrievable:

```text
portable commands
actual command/use sequence
resource/object references
debug markers
```

You can’t just leave:

```text
VkCommandBuffer
ID3D12GraphicsCommandList*
MTLCommandBuffer/Encoder
GPUCommandBuffer
```

Then when capture is turned on, it is discovered that the original portable command cannot be restored.

Implementations can use compact internal IR;
There is no requirement to create an additional second debug IR when tooling observer is not enabled.

---

## 52.3 Object reconstruction graph

Capture tooling must obtain an object definition graph that does not contain live Rust handles.

For example:

```text
TextureView
    -> Texture ObjectId

BindGroup
    -> BindGroupLayout ObjectId
    -> Buffer/TextureView/Sampler ObjectId

PipelineInterface
    -> BindGroupLayout ObjectIds

RasterPipeline
    -> ShaderModule ObjectIds
    -> PipelineInterface ObjectId
    -> fixed state
```

This is the source of future artifact ObjectTable.

---

## 52.4 Upload mutation

CPU upload is an observable mutation.

Tooling must obtain:

```text
UploadJob ID
destination ObjectId
offset/subresource
HostTexelLayout
source bytes
```

Upload source has been retained by `UploadJob`,
So capture does not need to read back the CPU data just uploaded.

---

## 52.5 Readback primitive

RHI official readback provides:

```text
Buffer/Texture readback
row/image layout
completion-aware readiness
`readback.read().await -> ReadbackView<'_>`
```

`ReadbackView` is an RAII mapping lease. Tooling must consume or copy the view
while it is live; it must not model readiness as an indefinitely valid bare
`&[u8]`, because backend Drop may need to unmap the resource.

Capture Coordinator can be used to:

```text
initial snapshot
checkpoint
observation
```

RHI does not decide when to snapshot.

---

## 52.6 Submission / present visibility

Tooling must be able to observe complete relationships, not counts:

```text
PlanPoint
lane
RecordedWork IDs
cross-batch dependencies
PresentPlan(frame, after)
SubmissionPoint
per-PlanPoint CompletionPoint
PresentReceipt
terminal outcome
```

You can't just record:

```text
batch_count
dependency_count
```

Because that cannot be Replayed.

---

## 52.7 Transient reconstruction contract

Transient resources are ordinary logical Buffer/Texture objects for command,
binding, and capture purposes. Their capture definition additionally retains:

```text
descriptor
TransientLifetime { acquire: PlanPoint, release_frontier }
owning SubmissionPlan identity
```

Capture never records a native heap/page, alias offset, alias barrier, or
backend synchronization primitive. On replay, RHI reconstructs the same
portable transient lifetime and selects `Dedicated` or `Aliasing` internally.

---

## 52.8 Shader provenance

Must be retained:

```text
ShaderArtifact.code
ShaderAbiVersion
ShaderInterface
ShaderRequirements
ShaderProvenance
```

ReplayRuntime/toolchain judges based on this:

```text
Can the target backend accept it directly?
Must it be recompiled?
Or is executable-only -> Unsupported?
```

RHI does not do a cross-backend compiler.

---

## 52.9 Device loss / terminal events

After Device loss:

```text
CompletionPoint
ReadbackTicket
AcquiredFrame
PresentReceipt
tooling subscription
```

Terminal observation must be obtained.

Capture does not allow waiting forever for an event that the RHI already knows will not complete.

---

# 53. Tooling SPI — engine/tooling surface

This is not a normal renderer API, but the interface needs to be frozen for:

```text
RenderGraph diagnostics
Capture Coordinator
Replay/debug tooling
```

use.

Recommended independent Cargo feature:

```text
rhi-tooling
```

Ordinary release build does not need to enable tooling observer surface.
But the reconstructable semantic contract of RecordedWork inside RHI still exists.

---

## 53.1 SPI version

```rust
#[doc(hidden)]
pub mod tooling {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ToolingSpiVersion {
        pub major: u16,
        pub minor: u16,
    }

    pub const TOOLING_SPI_VERSION: ToolingSpiVersion =
        ToolingSpiVersion { major: 1, minor: 0 };
}
```

Tooling SPI must bump major when changing incompatible semantic.

It is not a Capture Artifact schema version.

---

## 53.2 ToolingAccess / subscription

```rust
#[doc(hidden)]
pub mod tooling {
    use std::sync::Arc;

    #[derive(Clone)]
    pub struct ToolingAccess {
        /* device-scoped */
    }

    pub struct ToolingSubscription {
        /* unregister-on-drop */
    }

    pub trait SemanticObserver: Send + Sync + 'static {
        /// The callback is a synchronous observation callback.
        ///
        /// If the observer needs to retain data, it must copy/own it before
        /// returning.
        ///
        /// The callback must not re-enter a mutating RHI API on the same
        /// `Device` or DeviceIdentity, wait for GPU completion, or block.
        /// In particular, it must not drop its own ToolingSubscription while
        /// this callback is running.
        fn on_event(
            &self,
            event: SemanticEvent<'_>,
        );
    }

    impl Device {
        #[doc(hidden)]
        pub fn tooling(
            &self,
        ) -> ToolingAccess;
    }

    impl ToolingAccess {
        pub fn subscribe(
            &self,
            observer: Arc<dyn SemanticObserver>,
        ) -> RhiResult<ToolingSubscription>;

        /// Queries the definition of an object that is currently still live.
        pub fn describe_object(
            &self,
            id: ObjectId,
        ) -> RhiResult<CapturedObjectDefinition>;

        /// Obtains complete portable semantics for live `RecordedWork`.
        pub fn describe_work(
            &self,
            work: ObjectId,
        ) -> RhiResult<CapturedRecordedWork>;
    }
}
```

`subscribe()` has one start linearization point: the successful insertion of the
observer into the Device's observer set. It returns only after that point. The
returned subscription receives every event whose `SemanticEventId` is assigned
after the point and before its Drop unregistration point, exactly once and in
increasing `SemanticEventId` order; it does not receive an event assigned before
the point. This is the boundary between lazy description of pre-existing live
objects and observation of new semantic events.

Dropping `ToolingSubscription` unregisters the observer and waits for every
callback that began before unregistration to return. Once Drop returns, no
callback for that subscription can begin or remain active. An observer must not
drop its own `ToolingSubscription` from `on_event()`: doing so would wait for
the callback currently executing and is prohibited.

`describe_object()` solves:

> The object has been created before capture scope starts, so the capture observer has not seen the ObjectCreated event.

Capture Coordinator can lazy pull definition when it sees an ObjectId.

---

## 53.3 Observation ordering

```rust
#[doc(hidden)]
pub mod tooling {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct SemanticEventId(u64);
}
```

Unique, monotonic assignment within each DeviceIdentity.

but:

> `SemanticEventId` is CPU observation order, not GPU execution order.

Under multi-threaded recorder:

```text
Event 100 < Event 101
```

Doesn't mean GPU work 100 happens-before 101.

Real GPU order only looks at:

```text
order within PortableCommand
SubmissionPlan lane order
explicit dependencies
present relation
```

---

# 54. Captured object definitions

Tooling definition must contain only:

```text
value types
ObjectId
AcquiredFrameId
```

Live RHI handles must not be included.

```rust
#[doc(hidden)]
pub mod tooling {
    #[non_exhaustive]
    #[derive(Clone)]
    pub enum CapturedBindingResource {
        Buffer {
            buffer: ObjectId,
            range: BufferRange,
        },

        TextureView {
            view: ObjectId,
        },

        Sampler {
            sampler: ObjectId,
        },

        BufferArray(Vec<(ObjectId, BufferRange)>),
        TextureViewArray(Vec<ObjectId>),
        SamplerArray(Vec<ObjectId>),
    }

    #[derive(Clone)]
    pub struct CapturedBindGroupEntry {
        pub slot: BindingSlotId,
        pub resource: CapturedBindingResource,
    }

    #[derive(Clone)]
    pub struct CapturedBindGroupDefinition {
        pub label: Label,
        pub layout: ObjectId,
        pub entries: Vec<CapturedBindGroupEntry>,
    }

    #[derive(Clone)]
    pub struct CapturedPipelineInterfaceDefinition {
        pub label: Label,
        pub groups: Vec<ObjectId>,
    }

    #[derive(Clone)]
    pub struct CapturedRasterPipelineDefinition {
        pub label: Label,

        pub vertex: ObjectId,
        pub fragment: Option<ObjectId>,
        pub interface: ObjectId,

        pub vertex_input: VertexInputState,
        pub primitive: PrimitiveState,
        pub depth_stencil: Option<DepthStencilState>,
        pub multisample: MultisampleState,
        pub color_targets: Vec<Option<ColorTargetState>>,
    }

    #[derive(Clone)]
    pub struct CapturedComputePipelineDefinition {
        pub label: Label,
        pub shader: ObjectId,
        pub interface: ObjectId,
    }

    #[non_exhaustive]
    #[derive(Clone)]
    pub enum CapturedObjectDefinition {
        Buffer {
            id: ObjectId,
            descriptor: BufferDescriptor,
        },

        Texture {
            id: ObjectId,
            descriptor: TextureDescriptor,
        },

        TextureView {
            id: ObjectId,
            texture: ObjectId,
            descriptor: TextureViewDescriptor,
        },

        Sampler {
            id: ObjectId,
            descriptor: SamplerDescriptor,
        },

        Shader {
            id: ObjectId,
            artifact: ShaderArtifact,
        },

        BindGroupLayout {
            id: ObjectId,
            descriptor: BindGroupLayoutDescriptor,
        },

        BindGroup {
            id: ObjectId,
            definition: CapturedBindGroupDefinition,
        },

        PipelineInterface {
            id: ObjectId,
            definition: CapturedPipelineInterfaceDefinition,
        },

        RasterPipeline {
            id: ObjectId,
            definition: CapturedRasterPipelineDefinition,
        },

        ComputePipeline {
            id: ObjectId,
            definition: CapturedComputePipelineDefinition,
        },
    }
}
```

Capture Artifact Layer subsequently maps runtime `ObjectId` to capture-local typed ID;
RHI tooling SPI itself does not define disk typed-ID encoding.

---

# 55. Captured mutation / command value types

## 55.1 Raster values

```rust
#[doc(hidden)]
pub mod tooling {
    #[derive(Clone)]
    pub enum CapturedColorAttachmentView {
        TextureView(ObjectId),
        Frame(AcquiredFrameId),
    }

    #[derive(Clone)]
    pub struct CapturedColorAttachment {
        pub view: CapturedColorAttachmentView,
        pub load: LoadOp<Color>,
        pub store: StoreOp,
        pub resolve: Option<CapturedColorAttachmentView>,
    }

    #[derive(Clone)]
    pub struct CapturedDepthStencilAttachment {
        pub view: ObjectId,
        pub depth: Option<DepthAttachmentMode>,
        pub stencil: Option<StencilAttachmentMode>,
    }

    #[derive(Clone)]
    pub struct CapturedRasterScope {
        pub label: Label,
        pub colors: Vec<Option<CapturedColorAttachment>>,
        pub depth_stencil: Option<CapturedDepthStencilAttachment>,
    }
}
```

## 55.2 Copy values

```rust
#[doc(hidden)]
pub mod tooling {
    #[derive(Clone)]
    pub struct CapturedBufferCopy {
        pub src: ObjectId,
        pub src_offset: u64,
        pub dst: ObjectId,
        pub dst_offset: u64,
        pub size: u64,
    }

    #[derive(Clone)]
    pub struct CapturedBufferTextureCopy {
        pub buffer: ObjectId,
        pub buffer_offset: u64,
        pub bytes_per_row: u32,
        pub rows_per_image: u32,

        pub texture: ObjectId,
        pub texture_subresource: TextureSubresourceLayers,
        pub texture_origin: Origin3d,
        pub extent: Extent3d,
    }

    #[derive(Clone)]
    pub struct CapturedTextureCopy {
        pub src: ObjectId,
        pub src_subresource: TextureSubresourceLayers,
        pub src_origin: Origin3d,

        pub dst: ObjectId,
        pub dst_subresource: TextureSubresourceLayers,
        pub dst_origin: Origin3d,

        pub extent: Extent3d,
    }

    #[derive(Clone)]
    pub struct CapturedResolve {
        pub src: ObjectId,
        pub src_subresource: TextureSubresourceLayers,

        pub dst: ObjectId,
        pub dst_subresource: TextureSubresourceLayers,

        pub extent: Extent3d,
    }

    #[derive(Clone)]
    pub struct CapturedBlit {
        pub src: ObjectId,
        pub src_subresource: TextureSubresourceLayers,
        pub src_origin: Origin3d,
        pub src_extent: Extent3d,

        pub dst: ObjectId,
        pub dst_subresource: TextureSubresourceLayers,
        pub dst_origin: Origin3d,
        pub dst_extent: Extent3d,

        pub filter: BlitFilter,
    }
}
```

## 55.3 Upload / readback definitions

```rust
#[doc(hidden)]
pub mod tooling {
    #[non_exhaustive]
    #[derive(Clone)]
    pub enum CapturedUploadDefinition {
        Buffer {
            id: ObjectId,
            dst: ObjectId,
            dst_offset: u64,
            bytes: std::sync::Arc<[u8]>,
        },

        Texture {
            id: ObjectId,
            dst: ObjectId,
            subresource: TextureSubresourceLayers,
            origin: Origin3d,
            extent: Extent3d,
            source_layout: HostTexelLayout,
            bytes: std::sync::Arc<[u8]>,
        },
    }

    #[non_exhaustive]
    #[derive(Clone)]
    pub enum CapturedReadbackRequest {
        Buffer {
            ticket: ObjectId,
            src: ObjectId,
            range: BufferRange,
        },

        Texture {
            ticket: ObjectId,
            src: ObjectId,
            subresource: TextureSubresourceLayers,
            origin: Origin3d,
            extent: Extent3d,
        },
    }
}
```

Upload bytes is debug/capture-sensitive content;
The Artifact layer is responsible for redact/encrypt/filter policy.

---



# 56. PortableCommand / RecordedWork tooling IR

PortableCommand is an **in-memory semantic IR**, not a file format.

~~~rust
#[doc(hidden)]
pub mod tooling {
    #[non_exhaustive]
    #[derive(Clone)]
    pub enum PortableCommand {
        BeginRaster(CapturedRasterScope),
        EndRaster,

        SetRasterPipeline(ObjectId),
        SetComputePipeline(ObjectId),

        SetBindGroup {
            index: BindGroupIndex,
            group: ObjectId,
            dynamic_offsets: Vec<u32>,
        },

        SetVertexBuffer {
            slot: u32,
            buffer: ObjectId,
            range: BufferRange,
        },

        SetIndexBuffer {
            buffer: ObjectId,
            range: BufferRange,
            format: IndexFormat,
        },

        SetViewport(Viewport),
        SetScissor(Rect),
        SetBlendConstant(Color),
        SetStencilReference(u32),

        Draw {
            vertices: std::ops::Range<u32>,
            instances: std::ops::Range<u32>,
        },

        DrawIndexed {
            indices: std::ops::Range<u32>,
            base_vertex: i32,
            instances: std::ops::Range<u32>,
        },

        BeginCompute { label: Label },
        EndCompute,

        Dispatch {
            x: u32,
            y: u32,
            z: u32,
        },

        Upload {
            upload: ObjectId,
        },

        Readback {
            ticket: ObjectId,
        },

        CopyBuffer(CapturedBufferCopy),
        CopyBufferToTexture(CapturedBufferTextureCopy),
        CopyTextureToBuffer(CapturedBufferTextureCopy),
        CopyTexture(CapturedTextureCopy),
        Resolve(CapturedResolve),
        Blit(CapturedBlit),

        PushDebugGroup(String),
        PopDebugGroup,
        DebugMarker(String),
    }

    #[derive(Clone)]
    pub struct CapturedCommand {
        pub command: PortableCommand,
        pub actual_uses: Vec<CapturedResourceUse>,
    }

    #[derive(Clone)]
    pub struct CapturedRecordedWork {
        pub work: ObjectId,
        pub device: DeviceIdentity,
        pub domains: LaneWorkDomains,
        pub commands: Vec<CapturedCommand>,
        pub merged_use_summary: Vec<CapturedResourceUse>,
    }
}
~~~

### CapturedResourceUse

A tooling-owned value may not retain a live Buffer/Texture handle.

~~~rust
#[doc(hidden)]
pub mod tooling {
    #[non_exhaustive]
    #[derive(Clone)]
    pub enum CapturedResourceUse {
        Buffer {
            buffer: ObjectId,
            range: BufferRange,
            stages: command::PipelineScope,
            access: command::AccessMask,
        },

        Texture {
            texture: ObjectId,
            subresources: TextureSubresourceRange,
            stages: command::PipelineScope,
            access: command::AccessMask,
            intent: command::TextureUseIntent,
        },

        Frame {
            frame: AcquiredFrameId,
            stages: command::PipelineScope,
            access: command::AccessMask,
        },
    }
}
~~~

A Rust enum discriminant / memory layout must never be written directly as an artifact opcode.

The Artifact Layer must separately provide:

~~~text
tagged
versioned
bounds-checked
canonical encoding
~~~

---

# 57. Submission / presentation tooling IR

The old draft retained only:

~~~text
work_count
dependency_count
present_count
~~~

This is insufficient for Replay.

Freeze the complete logical relation:

~~~rust
#[doc(hidden)]
pub mod tooling {
    #[derive(Clone)]
    pub struct CapturedSubmissionBatch {
        pub point: PlanPoint,
        pub lane: SubmissionLaneId,
        pub work: Vec<ObjectId>,
    }

    #[non_exhaustive]
    #[derive(Clone, Copy, Debug)]
    pub enum CapturedDependencySource {
        PlanPoint(PlanPoint),
        PriorCompletion(CompletionPoint),
    }

    #[derive(Clone, Copy, Debug)]
    pub struct CapturedPlanDependency {
        pub before: CapturedDependencySource,
        pub after: PlanPoint,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct CapturedPresentPlan {
        pub id: PresentPlanId,
        pub frame: AcquiredFrameId,
        pub after: PlanPoint,
    }

    #[derive(Clone)]
    pub struct CapturedSubmissionPlan {
        pub device: DeviceIdentity,
        pub plan: SubmissionPlanId,
        pub batches: Vec<CapturedSubmissionBatch>,
        pub dependencies: Vec<CapturedPlanDependency>,
        pub presents: Vec<CapturedPresentPlan>,
    }

    #[derive(Clone)]
    pub struct CapturedSubmissionReceipt {
        pub submitted: SubmissionPoint,
        pub overall_completion: CompletionPoint,

        pub point_completions: Vec<(PlanPoint, CompletionPoint)>,

        pub presents: Vec<(PresentPlanId, PresentReceiptId)>,
    }
}
~~~

The Artifact Layer subsequently canonicalizes runtime PlanPoint/IDs into capture-local IDs.

---

# 58. Semantic events / parts RHI does not own

## 58.1 SemanticEvent

~~~rust
#[doc(hidden)]
pub mod tooling {
    #[non_exhaustive]
    pub enum SemanticEvent<'a> {
        ObjectCreated {
            event: SemanticEventId,
            definition: &'a CapturedObjectDefinition,
        },

        /// GPU-safe backing has actually been reclaimed from RHI inventory.
        ObjectReclaimed {
            event: SemanticEventId,
            object: ObjectId,
        },

        UploadDefined {
            event: SemanticEventId,
            upload: &'a CapturedUploadDefinition,
        },

        ReadbackDefined {
            event: SemanticEventId,
            request: &'a CapturedReadbackRequest,
        },

        WorkFinished {
            event: SemanticEventId,
            work: &'a CapturedRecordedWork,
        },

        SubmissionAccepted {
            event: SemanticEventId,
            plan: &'a CapturedSubmissionPlan,
            receipt: &'a CapturedSubmissionReceipt,
        },

        CompletionChanged {
            event: SemanticEventId,
            point: CompletionPoint,
            state: &'a CompletionState,
        },

        FrameAcquired {
            event: SemanticEventId,
            target: ObjectId,
            configured_presentation: ObjectId,
            frame: AcquiredFrameId,
            configuration: &'a PresentationConfiguration,
            extent: Extent3d,
        },

        PresentChanged {
            event: SemanticEventId,
            receipt: PresentReceiptId,
            state: &'a PresentState,
        },

        DeviceLost {
            event: SemanticEventId,
            info: &'a DeviceLossInfo,
        },

        Diagnostic {
            event: SemanticEventId,
            diagnostic: &'a DiagnosticEvent,
        },
    }
}
~~~

ObjectCreated directly borrows the complete definition, avoiding a race where the object is reclaimed soon after observer return and its definition has disappeared by a later lazy query.

`FrameAcquired.target` and `FrameAcquired.configured_presentation` must each
refer to a corresponding `CapturedObjectDefinition::PresentationTarget` and
`CapturedObjectDefinition::ConfiguredPresentation`, respectively. The latter
references the former through `definition.target`; therefore every `ObjectId`
carried by `FrameAcquired` has a describable logical definition without
serializing a host/native presentation handle.

For live objects that already existed before capture scope was enabled, use:

~~~rust
ToolingAccess::describe_object(id)
~~~

For work recorded before scope activation but still retained by plan/submission ownership, use:

~~~rust
ToolingAccess::describe_work(work_id)
~~~

---

## 58.2 Event callback lifetime

References in SemanticEvent<'a> are valid only during the on_event() call.

An observer that needs retention must:

~~~text
clone/copy into its own queue
~~~

RHI does not wait for subsequent observer processing.

A tooling observer:

~~~text
must not re-enter a mutating API on the same Device or DeviceIdentity
must not wait for GPU completion or otherwise block
must not drop its own ToolingSubscription
must not alter command/submission semantics
~~~

It may add capture/debug CPU overhead but may not change GPU results.

---

## 58.3 RHI explicitly does not implement the Artifact Layer

RHI does not own:

~~~text
magic
schema major/minor
chunk
manifest
compression
dedup
blob hashing
signature
encryption
artifact migration
~~~

Tooling SPI version and Artifact schema version are separate things.

---

## 58.4 RHI does not implement capture dependency closure

RHI does not answer:

~~~text
Capture Passes 30~40:
which producers must also be added beforehand?

A persistent Texture was written in an earlier frame:
must it be snapshotted?

How many frames of a history resource must be included?
~~~

This belongs to:

~~~text
RenderGraph
+
Capture Coordinator
~~~

---

## 58.5 RHI does not decide snapshot policy

RHI provides:

~~~text
Readback
resource descriptors
actual use
~~~

Capture decides:

~~~text
snapshot point
subresource
full bytes / hash-only
size budget
redaction
external fixture
~~~

---

## 58.6 RHI does not implement ReplayRuntime

ReplayRuntime owns:

~~~text
read/safely parse artifact
schema migration
select target Device
capability negotiation
shader rebuild
object-graph rebuild
command rebuild
submission rebuild
external fixtures
observation/diff
step debugger
~~~

It ultimately calls only normal RHI API.

---

## 58.7 Replay source of truth

Normal Replay executes:

~~~text
CapturedObjectDefinition graph
+ initial snapshots / upload mutations
+ PortableCommand
+ CapturedSubmissionPlan
+ external fixtures
~~~

as its source of truth.

FrozenGraphIR belongs to RenderGraph and primarily provides:

~~~text
why
pass/resource provenance
visualization
~~~

If a future:

~~~text
FrozenGraphIR
    -> new Graph compiler
~~~

regenerates a plan, that is:

> **Recompile Comparison Mode**

not normal Replay.

---

## 58.8 Security boundary

Tooling semantics may not contain:

~~~text
native pointer
OS handle
GPU virtual address
descriptor heap index
absolute host-memory address
credential
~~~

Shader source, labels, and upload bytes may contain sensitive contents; whether to:

~~~text
redact
omit
encrypt
~~~

belongs to Artifact/Capture policy.

---

## 58.9 Capture tooling freeze decision

RHI 0.16 must implement:

~~~text
live RecordedWork reconstructable semantics
ObjectId-based object-definition graph
upload mutation observability
readback primitive
full submission/present relation
structured terminal events
Shader provenance
engine/tooling observer + lazy describe seam
~~~

RHI 0.16 **does not implement**:

~~~text
capture file
snapshot selection
Graph dependency closure
cross-backend compatibility decision
ReplayRuntime
diff/debugger UI
~~~

This lets Capture/Replay build on RHI without placing a “debug product” into the core execution API.
