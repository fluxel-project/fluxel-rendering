# The capability-oriented RHI interface

This document is the interface contract introduced by lead 3F / 0.16. It describes
what the RHI's internal platform layer looks like, so a reader can implement a
backend against it or judge a change to it without reading the whole crate.

**Design principle: unify semantics and ownership, not hardware capability.**

**Scope, as re-cut:** this layer is designed to the five-backend standard — no
backend mechanism may leak into it, and the family vocabulary is written for five —
but **0.16 requires only the three native backends** (DX12, Vulkan, Metal) to
implement it. The GL family, the browser WebGPU adapter and compressed formats are
0.17. The goal of 0.16 is not to finish every GPU platform; it is to prove that this
model holds up. See section 20 of the lead 3F plan.

A familiar failure mode is avoided on purpose. This is *not* one wide interface
whose methods a backend may decline:

```rust
// Not this. Every backend must answer for capabilities it may not have, and the
// result is `if backend == …`, `if supported …` and `Unsupported` everywhere.
trait Device {
    fn draw();
    fn dispatch();           // what does WebGL2 answer?
    fn storage_texture();    // and when the device has none?
    fn timeline_semaphore(); // Metal is not this model
}
```

Instead there is a small required base, one trait per capability family, and
negotiation.

## 1. Three layers

```text
                    RHI Base
     identity / resources / lifetime / submission / capability query
                        |
               Capability Families
  Graphics / Compute / StorageBuffer / StorageTexture / Indirect /
  Multiview / AsyncCompute / TransferQueue / ...
                        |
             Backend Implementations
        DX12 / Vulkan / Metal / WebGPU / GL family
```

Nothing in `common` names a platform crate. That is enforced by inspection, not by
convention: the only occurrence of `ash`, `windows`, `objc2`, `web_sys` or `glow`
under `crates/rhi/src/common/` is the doc line stating the rule.

## 2. Base

The base carries only what all five backends must agree on for Fluxel's own
semantics to hold. The membership test: **a base item is something all five
backends must agree on for Fluxel's own semantics to hold.**

| Item | Type | What it is |
| --- | --- | --- |
| identity | `base::stamp::DeviceStamp` | `(DeviceIdentity, generation)`; identity alone cannot answer "was this object created before this device was replaced" |
| resources | `base::resource::{ResourceId, BufferId, TextureId}` | device stamp + opaque `PhysicalResourceIdentity`; the kind is a type parameter, so a wrong-kind mistake does not compile |
| lifetime | `base::lifetime::may_release(CompletionStatus) -> bool` | ADR-0004's rule: only terminal completion releases, and `Unknown` is not terminal |
| submission | `base::submission::Disposition` | `Observed` / `Abandoned` / `Terminal`: who holds the leases while completion is not terminal |
| capability query | `api::negotiate::{CapabilitySource, require}` | how a family is obtained from a device |

Not in the base, deliberately:

| Not in the base | Where it belongs | Why |
| --- | --- | --- |
| `draw`, `dispatch` | the graphics and compute families | a base method forces every backend to answer for a capability it may not have |
| `storage_texture` | the storage-texture family | the capability varies per adapter, not only per backend |
| copies | the copy family | every backend serves them today, and that is a fact about today's backends rather than a semantic requirement — the same argument that keeps `Graphics` a family |
| `graphics_queue`, `compute_queue`, `transfer_queue` | the queue-shape rows | DX12 and Vulkan expose several queues, Metal's model differs, WebGPU has its own constraints and WebGL2 is not the same thing at all |

## 3. Capability rows and the ledger

`common::caps::Capability` has **exactly one variant per optional family**, and
`CapabilityLedger` records what one *device* proved. A row is enabled only when
three independent conditions hold:

| Condition | Meaning |
| --- | --- |
| evidence | a route provides the domain (`Core`, or a named optional facility) |
| limit floor | the numeric requirement for the row is satisfied |
| probe outcome | `Passed` (a real command ran), or `NotRequired` (this backend's proof is structural) |

`Failed` and `NotRun` both leave the row disabled, and they are different
sentences: one says the domain does not work here, the other says nobody has
established that it does. An unexamined row answers exactly like a refused one.

**Whether a command probe is required is a backend-level decision, not a
row-level one.** The GL family establishes compute by running a dispatch; Vulkan
establishes it by creating a device on a queue family that reports compute. The
backend states how it proved the row; the ledger refuses what was not proved.

Rows today: `Graphics`, `Compute`, `StorageBuffer`, `StorageImage`,
`IndirectDraw`, `IndirectDispatch`, `MultiDrawIndirect`, `MultiDraw`, `Multiview`,
`AsyncCompute`, `TransferQueue`, `OcclusionQuery`, `ElapsedQuery`,
`TimestampQuery`, `BaseVertex`, `FirstInstance`, `AnisotropicFiltering`.

## 4. Families

A family is **vocabulary, not permission**. Each is bounded on
`api::handle::FamilyApi`, declares its own `Error`, and takes only base vocabulary
plus its own associated types — so no descriptor is invented here that a backend
would have to accept.

### 4.1 `FamilyApi` — the supertrait every handle implements

```rust
trait FamilyApi {
    /// The device generation this handle was negotiated on.
    fn stamp(&self) -> DeviceStamp;
}
```

Callers verify resource ids against **the handle's stamp, never a freshly read
one**: after a device replacement a re-read returns the replacement's stamp and a
stale id would verify successfully. `verify_buffer` / `verify_texture` accept only
the handle, so the one correct source is the only input.

### 4.2 `GraphicsApi`

```rust
trait GraphicsApi: FamilyApi {
    type Error; type Pipeline; type Bindings;

    fn begin_raster(&mut self, d: &RasterPassDescriptor<'_, TextureId>) -> Result<(), Self::Error>;
    fn end_raster(&mut self) -> Result<(), Self::Error>;
    fn set_raster_pipeline(&mut self, p: &Self::Pipeline) -> Result<(), Self::Error>;
    fn set_bindings(&mut self, b: &Self::Bindings) -> Result<(), Self::Error>;
    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferId, offset: u64) -> Result<(), Self::Error>;
    fn set_index_buffer(&mut self, buffer: BufferId, offset: u64, format: IndexFormat) -> Result<(), Self::Error>;
    fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Self::Error>;
    fn set_scissor(&mut self, scissor: ScissorRect) -> Result<(), Self::Error>;
    fn draw(&mut self, vertices: Range<u32>, instance_count: u32) -> Result<(), Self::Error>;
    fn draw_indexed(&mut self, indices: Range<u32>, instance_count: u32) -> Result<(), Self::Error>;
}
```

The current single-queue implementation may use the handle as its recording context,
**but the contract does not require capability negotiation and recording context to
remain the same object** (plan section 21). Freezing that identity would have to be
undone the moment there are graphics, compute and transfer queues with parallel
recording. The pass descriptor is `fluxel_rendergraph`'s, already generic over the
resource type — the GL family instantiates it with its own texture id today, so the
pass vocabulary is not restated. Its optional depth-stencil attachment is why the
current "no depth recipe" refusal is *expressible* rather than hidden.

Instance *counts* rather than instance ranges, so the first instance is fixed at
zero. A range can express a non-zero first instance, which is the `FirstInstance`
family, and **a family's parameter space must not be able to name another family's
capability** — otherwise every backend without that family must refuse at run time,
which is the shape this layer removes. Base vertex is the `BaseVertex` family for the
same reason; neither is here. The rule applies next to depth-stencil state,
multiview, variable-rate shading, mesh shaders and ray tracing.

### 4.3 `ComputeApi`

```rust
trait ComputeApi: FamilyApi {
    type Error; type Pipeline; type Bindings;

    fn begin_compute(&mut self) -> Result<(), Self::Error>;
    fn end_compute(&mut self) -> Result<(), Self::Error>;
    fn set_compute_pipeline(&mut self, p: &Self::Pipeline) -> Result<(), Self::Error>;
    fn set_bindings(&mut self, b: &Self::Bindings) -> Result<(), Self::Error>;
    fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), Self::Error>;
}
```

Every group dimension must be non-zero: a zero-sized dispatch is a legal driver
no-op, so one arriving here is a caller's mistake rather than a request to do
nothing.

### 4.4 `StorageBufferApi` / `StorageTextureApi`

```rust
trait StorageBufferApi: FamilyApi {
    type Error; type Binding;
    fn create_storage_binding(&mut self, buffer: BufferId, offset: u64, size: u64)
        -> Result<Self::Binding, Self::Error>;
}

trait StorageTextureApi: FamilyApi {
    type Error; type Binding;
    fn create_storage_binding(&mut self, texture: TextureId) -> Result<Self::Binding, Self::Error>;
}
```

These are **resource roles**, not command domains: they gate how a binding is
built. They are traits anyway, because a backend that cannot serve the role must be
structurally unable to build the binding. They carry no capability query — asking
whether a role is available is `require::<StorageBuffer>()`, and a second way to ask
would be a second answer.

### 4.5 `IndirectDrawApi` / `IndirectDispatchApi`

```rust
trait IndirectDrawApi: FamilyApi {
    type Error;
    fn draw_indirect(&mut self, commands: BufferId, offset: u64, count: u32, stride: u32)
        -> Result<(), Self::Error>;
}

trait IndirectDispatchApi: FamilyApi {
    type Error;
    fn dispatch_indirect(&mut self, commands: BufferId, offset: u64) -> Result<(), Self::Error>;
}
```

**These are two traits, not one, and an earlier version of this document had them
merged.** That version violated the rule the ledger already encoded: it carried
`IndirectDraw` and `IndirectDispatch` as two rows, while the trait asked a backend to
implement both verbs. A platform with indirect draw and no indirect dispatch would
have been forced to write a refusing `dispatch_indirect` — the lowest-common-
denominator shape this whole design exists to remove.

The rule that decides every split, now frozen: **one independently negotiable batch
of API per family, and one ledger row per family.** The test is not whether two verbs
sound alike; it is whether they always appear, are always proved, and always fail
together.

`stride` is stated rather than assumed: a caller may pack records with padding, and
an assumed tight packing reads the wrong offsets.

### 4.6 `CopyApi`

```rust
trait CopyApi: FamilyApi {
    type Error;
    fn copy_buffer(&mut self, source: BufferId, destination: BufferId, region: BufferCopyRegion)
        -> Result<(), Self::Error>;
    fn copy_texture(&mut self, source: TextureId, destination: TextureId, region: TextureCopyRegion)
        -> Result<(), Self::Error>;
}
```

A **family rather than floor**, by the same argument that keeps `Graphics` a family:
every backend in the set serves copies today, and "all five happen to have it" is a
fact about today's backends rather than a requirement of Fluxel's semantics. Keeping
copy a family is also where the facade migration points: the retired `CopyBackend`
tier becomes "this device implements `Copy`" — a ledger row instead of a parallel
type hierarchy.

### 4.7 Markers without traits

`Multiview`, `AsyncCompute` and `TransferQueue` have ledger rows and markers but no
trait: no retained recipe declares a multiview attachment, and the execution model
is one queue. The markers already let a caller state the requirement.

## 5. Negotiation

```text
require::<Compute>(&device)
  |
  |- D: Provides<Compute>          compile time: the backend has the vocabulary
  |- device.ledger() proves ROW    run time: this device proved the capability
  `- Ok(handle)                    execution no longer asks again
```

```rust
fn require<'d, D, F>(device: &'d D) -> Result<<D as Provides<F>>::Api<'d>, UnsupportedCapability>
where D: CapabilitySource + Provides<F>, F: CapabilityFamily;
```

**The check comes before the backend is asked for a handle**, so a refused
negotiation cannot have had a side effect; a test asserts that by counting provider
calls. A handle borrows the device, so it cannot outlive the ledger that justified
it.

Requirements are stated in capability terms and never in backend terms: a node
requires `Graphics + StorageBuffer + Indirect`, not `Vulkan`. A refusal carries the
row and which condition failed — never examined, no route, limits unsatisfied,
probe failed, probe not run — because a caller needs to know whether *another*
device could run the graph.

## 6. Implementing a backend

1. Discover facts. Create the device; read limits and features. Report only what
   was read; an unproved fact keeps the value that rejects work.
2. Build the ledger. Record a row per fact you actually established, with its
   evidence, its limit floor, and how you proved it (`Passed` or `NotRequired`).
3. Implement `CapabilitySource` returning that ledger.
4. Implement `Provides<F>` for each family you hold, with its handle type, and
   implement that family's trait on the handle. A family you cannot serve is one
   you do not implement — there is then no call site for it at all.

Two refusals stay distinct, and collapsing them is what produces a
lowest-common-denominator API:

1. **the backend type has no such domain** — the vocabulary is absent, so no call
   site could have been written (a trait bound);
2. **the backend has the domain and this device did not prove it** — the refusal
   happens before any object, extension or command side effect (a value).

## 7. Deliberately absent

No `dyn` in the submission path: dispatch is static, and which backend exists is
decided by Cargo features at compile time. No descriptor set, pipeline barrier,
root signature or encoder-hazard type in `common` — those are each backend's own
mechanism, and lowering one into `common` to make a backend easier is how the
shared layer would become a lowest-common-denominator Vulkan. No public API: this
layer is private (`crates/rhi/src/common/`), and ADR-0006/0007 continue to hold.

## 8. Status

| Surface | State |
| --- | --- |
| `base::{stamp, resource, lifetime, submission}` | implemented, tested |
| `caps` (limits, rows, ledger) | implemented, tested |
| `api::{family, handle, negotiate}` | implemented, tested |
| `api::{graphics, families}` | vocabulary complete; no backend implements them yet |
| `vertex`, `sampler`, `binding`, `pipeline` | implemented, tested |
| Vulkan (`native::vulkan`) | steps 1-3 done and proven on real hardware; format/dimension/usage lowering, real buffer and image handles, and the resource table that owns buffer handles, texture handles (image plus the view sampled through it) and sampler handles over one allocator done; **step 4 complete**; **step 5 complete** — shader module, pipeline layout, descriptor set layout, compute pipeline and raster pipeline landed and proven against a real driver, the raster pipeline lowered from the common fixed-function vocabulary with the render pass that creation needs owned only for the call; **step 6 complete** — the retained WGSL lowered to SPIR-V with Naga's `wgsl-in` + `spv-out`, with the dialect, entry-point, stage and SPIR-V 1.0 profile checks decided before the driver is reached and the writer options pinned field by field (SPIR-V passthrough remains the other route); **step 7 complete** — the semantic access states lowered onto pipeline-stage/access masks and image layouts, and one command pool plus the one recording encoder that records those barriers, with `before == after` still a barrier; **step 8 complete** — the two copy routes this family owns, `vkCmdCopyBuffer` and `vkCmdCopyImage`, lowered from the portable regions with the alignment and range checks repeated at the driver boundary and the image aspect derived from the mapped format, recorded and proven against a real driver (`vkCmdCopyBufferToImage` / `vkCmdCopyImageToBuffer` stay with the staging path that needs them); **step 9 complete** — one unsignaled fence per execution, one `vkQueueSubmit` on the device's single queue, and the completion state machine over `common::base::{lifetime, submission}`, with a timeout reported as `Pending` and accepted-unknown work quarantined rather than released; the encoder's ended-recording handoff is the `Finished` type that submission alone accepts; **step 10 complete** — the fixed presentation contract (`R8G8B8A8_UNORM` + `SRGB_NONLINEAR`, `FIFO`, `OPAQUE`, a clamped image count and the extent the surface reports) is decided from a surface's own format, present-mode and capability lists and lowered into the `VkSwapchainCreateInfoKHR`, with sRGB deliberately not claimed and the extent never invented; the surface instance extensions (`VK_KHR_surface` + `VK_KHR_win32_surface`) are verified against the loader inventory and enabled on a surface-capable instance whose distinct `SurfaceInstance` witness makes a headless instance unable to reach the surface call, and an owned `VkSurfaceKHR` is created from a host Win32 window and destroyed before its borrowing parents; the surface-facts query (the driver's capability, format and present-mode lists, read into the inputs the pure contract decides against) and the per-family presentation-support answer are read from a real surface through the owned handle, with the selected graphics family reporting presentation support on the named board; **step 10's swapchain half landed** — a device can be opened with `VK_KHR_swapchain` verified against the physical device's own extension inventory *before* creation and enabled only through the distinct `SwapchainDevice` witness a headless device cannot produce; the `VkSwapchainKHR` is created from the fixed contract over the owned surface after the selected queue family's presentation support is read and refused by name when it cannot present, the surface-reported `IDENTITY` transform is now part of the contract rather than a deferred caller argument, and the images `Vulkan` creates with the swapchain are read into the handle and destroyed with it; **step 10's acquire half is landed** — a real `vkAcquireNextImageKHR` over the swapchain's own semaphore, lowered by a pure module into its named refusals (timeout, not-ready, out-of-date, surface-lost, driver result), with the lease that owns the acquired image and that semaphore, and an unpresented drop poisoning the surface and leaving the semaphore undestroyed rather than guessing reuse; **step 10's present half is landed** — a real `vkQueuePresentKHR` that consumes the lease by handing its acquire semaphore to the present as the wait, lowered by a pure module into success, success-plus-suboptimal and the out-of-date/surface-lost/driver refusals, with the waited-on semaphore retained per image index and released only when `vkAcquireNextImageKHR` hands that image back (the ANGLE inference, because returning from present is not proof its wait is consumed) and every remaining slot destroyed after a device-idle wait at swapchain teardown; **step 10 is complete** — reconfigure rebuilds over `old_swapchain` against the surface's current facts, retires the predecessor and destroys it through the same device-idle teardown on both the success and the failure path (the specification retires the old swapchain even when creation fails, so the method consumes it) and refuses a quarantined surface by name before the driver is reached, with recovering such a surface meaning replacing the surface itself; the draw submission not started, and **step 11 in progress** — its per-format evidence half landed: `common::formats` promotes the GL family's evidence-carrying per-format fact table (the eight facts, keyed on `(format, sample count)`, with storage facts refused without operation evidence), and `native::vulkan::format_facts` lowers one `vkGetPhysicalDeviceFormatProperties` answer into those facts and queries and records every format this backend maps, with the remaining ledger rows owed; and its core-row half records the rows the created device already proves — `Copy` from the API version on the graphics family, `IndirectDispatch` beside a proved compute row and keeping that row's numeric floor, and `TimestampQuery` from the selected family's own `timestamp_valid_bits` report carried on `SelectedQueue` — with `capability::capabilities` reading `raster`, `copy` and the timestamp placement from the ledger rather than writing them beside the queue; and its storage half reads the adapter's own `VkPhysicalDeviceFeatures`, enables `fragmentStoresAndAtomics` and `vertexPipelineStoresAndAtomics` only where the adapter reported them, and records the `StorageBuffer` row from both halves with the driver's binding-size report as its floor; and its storage-image half makes the device own that per-format evidence table — `format_facts::record_mapped` runs in `device::create` before `vkCreateDevice`, so a driver that cannot answer for a mapped format refuses the open with nothing created — and records the `StorageImage` row from the same stage-gated pair with `FormatTable::has_storage_read_write` as its resource floor, leaving the query, draw-parameter and multi-draw rows owed |
| DX12, Metal | not started; both are 0.16 |
| GL family, browser WebGPU, compressed formats | **0.17**, not 0.16 |

Test evidence: `common::` and `native::` hold 190+ unit tests, and the Vulkan tests
that open a real instance, adapter and `VkDevice` on this machine now cover memory
selection, suballocation, buffer and image handles, sampler handles (including a
comparison sampler and the disabled-comparison lowering a filtering sampler needs),
the resource table that owns buffers plus textures and their views plus samplers,
the real `VkShaderModule` and `VkComputePipeline` a compute pipeline is built
from, a real `VkDescriptorSetLayout` created from the common bind-group layout
vocabulary and owned by the pipeline layout that names it, a real
`VkGraphicsPipeline` for both the colour-only and the depth-attached fixed-function
descriptions, created against a render pass built from the same attachment
signature, and the retained WGSL artifacts lowered to SPIR-V by Naga and handed to
the driver as real modules (one retained vertex artifact and one retained compute
artifact). Step 7 adds a real command pool and a real recording encoder that records
buffer and image barriers driven by the portable access states, and the pure barrier
lowering is covered on its own: every named state lowers to a non-empty stage mask,
a texture-only state is refused for a buffer and a buffer-only state for an image,
the sampled-read layout follows the mapped format, and a same-state transition is
still a barrier. Step 8 adds both copy routes to that recording: a real
`vkCmdCopyBuffer` between two device-local buffers and a real `vkCmdCopyImage`
between two device-local images, recorded over their transfer transitions, with the
pure copy lowering covered on its own — the field-for-field buffer record, the
zero/misaligned/overflow/out-of-bounds refusals by name, the same four for textures
plus the differing-format, unknown-mip, zero-extent, layer-origin and layer-count
refusals, the mip-halved bounds, the depth aspect a depth format selects without
being named, and the one-layer subresource the record writes. Step 9 adds the
submission path: one real fence per execution, one real `vkQueueSubmit` on the
device's queue, a real fence wait that reports `Complete`, and a released submission
that refuses further observation — plus the pure lowerings of a fence's answers
(`VK_TIMEOUT` is pending and not a failure, `VK_ERROR_DEVICE_LOST` is named, and a
wait duration too large for the driver clamps to "forever"). Step 10's pure half adds
the surface-side lowerings: the present format pinned against the portable
`TextureFormat::Rgba8Unorm` and pinned as *not* its sRGB sibling, the driver's
`current_extent` winning over the caller's request, each missing piece of the fixed
contract as its own refusal, the image count clamped into the driver's own range, the
usage fold reaching every declared `TextureUsageKind`, and the swapchain create-info's
field-for-field contents. Step 10's owning half adds the surface object to that: the
two surface instance extensions verified against the loader inventory before creation
and enabled only on the `open_with_surface` path (the headless order is unchanged),
the distinct `SurfaceInstance` witness that makes the unresolved-entry-point crash
unrepresentable, the Win32 window lowering with its two named refusals, and a real
`VkSurfaceKHR` created from a real hidden window and destroyed before the instance
and the window it borrows. Step 10's query half adds the facts read through that
surface: a real surface reports non-empty format and present-mode lists that
`surface::contract` accepts with the contract's own format, colour space and present
mode, and every reported queue family answers `supports_presentation` with at least
one answering true, so the fact step 2's queue rule has to be told is read rather
than assumed. Step 10's swapchain half adds the swapchain object to that: a real
`VkSwapchainKHR` created over a real surface through a real device that enabled
`VK_KHR_swapchain` (the extension read from the physical device's own inventory
first, and refused by name when absent), with its real images read back and the
contract's format, extent and `IDENTITY` transform asserted, and the device-level
path smoked on this machine as well. Step 10's acquire half adds the lease to that: a
real `vkAcquireNextImageKHR` returns an in-range index whose image is the one at that
index, the lease carries a real semaphore, an unpresented drop poisons the surface
and leaves that semaphore undestroyed rather than guessing it is reusable, and a
poisoned surface refuses a second acquire by name before the driver is reached — with
the pure half covered on its own for the timeout/not-ready, out-of-date/surface-lost
and driver-result distinctions. Step 10's present half adds the present call to that:
a real `vkQueuePresentKHR` presents an image acquired from a real swapchain, the
surface is **not** poisoned (which is what distinguishes a present from an
unpresented drop), the semaphore present waited on is asserted retained in the slot
its image names, and a second acquire and present over the same swapchain proves the
surface stays live and the retained semaphore of whichever image returns is released
rather than reused or leaked — with the pure present lowering covered on its own for
the success/suboptimal pair and the out-of-date/surface-lost/driver-result
distinctions. Step 10's reconfigure half adds a real `vkCreateSwapchainKHR` over its
own predecessor: the predecessor's retained present semaphore is retired through the
same device-idle teardown, the replacement reads its own real images and acquires and
presents a frame of its own, and a surface an unpresented acquire quarantined refuses
a rebuild by name before the driver is reached — with the create-info's
`old_swapchain` field pinned both null and non-null in the pure half. Step 11's per-format half
adds the format-evidence table: on a real adapter every format this backend maps is queried once
and recorded in `MAPPED` order with `OperationProbed` evidence at sample count one, and the facts
`Vulkan` makes mandatory are asserted from the driver's own answer (`R8G8B8A8_UNORM` sampled,
renderable, blendable and copyable in both directions; its sRGB sibling sampled and renderable;
`D32_SFLOAT` renderable and copyable in both directions and *not* blendable), with a repeated
discovery accepted and changing nothing. The pure half covers each optimal-tiling flag lowering to
its own fact, a flag the table does not model (`BLIT_SRC`) lowering to none, the two attachment
kinds reaching one `renderable` fact, the single `STORAGE_IMAGE` bit proving both storage
directions, a storage fact passing the table's probe rule, and a driver answer with no flags being a
proved negative the table keeps rather than an absent row. Step 11's lowering half
adds `native::vulkan::capability`: the ledger `require` reads, the adapter's limits and that
evidence table are lowered onto `fluxel_rendergraph::DeviceCapabilities`, reading the
optional-domain rows from the ledger rather than re-deriving them, and reporting the recording and
transition facts this backend's command-buffer model implies. On a real adapter the lowering
reports one raster/copy queue whose compute row is the ledger's, the adapter's colour count and
uniform-offset alignment, the workgroup dimensions only where compute was proved, no surface (this
is the headless half), and exactly the mapped formats in table order — `R8G8B8A8_UNORM` sampled,
linearly filterable, a colour attachment at one sample and copyable both ways, and `D32_SFLOAT` a
depth-stencil attachment rather than a colour one and copyable both ways. The pure half covers the
fail-closed floor (no optional row, the workgroup dimensions left at zero), both storage directions
from one storage row, each indirect row reaching the buffer flag, the backend's own
recording/transition shape, the attachment-side split, the sample-count fold, the
absent-versus-recorded pair and the report order. Step 11's core-row half adds three ledger
rows the created device already proved, from facts the open path had already read and with no
device feature enabled: `Copy` from the API version on the graphics family (which the lowering
now reads instead of writing `true` beside the queue), `IndirectDispatch` only beside a proved
compute row and keeping that row's numeric floor, and `TimestampQuery` from the selected
family's own `timestamp_valid_bits` report carried on `SelectedQueue` as the driver's number.
`capability::capabilities` lowers `raster`, `compute`, `copy` and the timestamp placement from
the ledger — `TimestampCapabilities::PassBoundaries` only where the row is proved, `Unsupported`
otherwise — and `indirect_read` widened with no line written because it already folded the three
indirect rows. The pure tests cover each row's proving shape and its discriminating pair, and
assert the six feature-gated or preserved rows (`IndirectDraw`,
`MultiDrawIndirect`, `AnisotropicFiltering`, `Multiview`, `AsyncCompute`, `TransferQueue`) stay
both disabled and unexamined — the storage-buffer row left that list in step 11's storage half
and the storage-image row in step 11's storage-image half, both below; against a real adapter the created device proves copy, proves
indirect dispatch exactly where the family reports compute, and proves timestamps exactly where
the family's valid-bit report is non-zero. Step 11's storage half adds the `StorageBuffer` row and
the device-feature query it needs: `native::vulkan::features` reads the adapter's own
`VkPhysicalDeviceFeatures` and requests `fragmentStoresAndAtomics` and
`vertexPipelineStoresAndAtomics` only where the adapter reported them (an unreported feature would
make `vkCreateDevice` fail rather than leave a row unproved), and nothing else — a reported
`samplerAnisotropy` or `multiDrawIndirect` still comes out disabled. The row is recorded only when
**both** halves are enabled, because it describes a read *and* write domain whose write side is
gated per stage, and its numeric floor is the driver's own `max_storage_buffer_binding_size`
report. The pure tests cover each half alone as a refusal, both halves as a `NotRequired` proof,
the zero-binding-size refusal, and the no-inheritance rule; against a real adapter the device test
reads the report again, applies the same pure request, and asserts the device's enabled set equals
that and the ledger row equals the pair. Step 11's storage-image half makes the device own the
per-format evidence table — `device::create` runs `format_facts::record_mapped` before
`vkCreateDevice`, so a driver that cannot answer for a mapped format refuses the open with nothing
created, and the table the row is read from is the same one the capability lowering folds — and
records the `StorageImage` row from that same stage-gated pair beside
`FormatTable::has_storage_read_write` (a recorded row proving both storage directions) as the row's
resource floor. The two halves fail in different ledger fields, which the pure tests pin: without
the pair the row was never examined, while with the pair and no storage-capable format it is
examined and its floor refuses it, and a format proving one direction alone is refused the same
way. Against a real adapter the device test asserts the row equals the pair *and* the device's own
table's answer, and that the table holds every mapped format in `MAPPED` order — on the named board
the adapter enables both features and reports `STORAGE_IMAGE` for `R8G8B8A8_UNORM`, `B8G8R8A8_UNORM`
and `R16G16B16A16_SFLOAT` but not for the sRGB sibling or `D32_SFLOAT`, so the row is proved there.
Step 11's query-row half records the two remaining query rows from facts the open path already read:
`OcclusionQuery` from the graphics family the device was created on — the core *imprecise* query, so
the `occlusionQueryPrecise` feature this device leaves disabled is not part of the proof — and
`ElapsedQuery` from that same family's own timestamp-valid-bit report, because a `Vulkan` elapsed
interval is two timestamp writes and their difference rather than a separate query type; neither row
borrows a numeric floor it does not depend on. The pure tests pin each row's proving shape (occlusion
proved in a ledger whose every reported number is absent, and elapsed examined only where the family
reported usable timestamps), and against a real adapter the created device proves occlusion
unconditionally and elapsed exactly where timestamps were reported.
Clippy is clean
under `-D warnings` for all-features and no-default-features.
