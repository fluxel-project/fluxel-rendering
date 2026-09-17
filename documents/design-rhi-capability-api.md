# The capability-oriented RHI interface

This document is the interface contract introduced by lead 3F / 0.16. It describes
what the RHI's internal platform layer looks like, so a reader can implement a
backend against it or judge a change to it without reading the whole crate.

**Design principle: unify semantics and ownership, not hardware capability.**

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
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<(), Self::Error>;
    fn draw_indexed(&mut self, indices: Range<u32>, instances: Range<u32>) -> Result<(), Self::Error>;
}
```

The handle *is* the recording context, so no encoder type appears in this
vocabulary. The pass descriptor is `fluxel_rendergraph`'s, already generic over the
resource type — the GL family instantiates it with its own texture id today, so the
pass vocabulary is not restated. Its optional depth-stencil attachment is why the
current "no depth recipe" refusal is *expressible* rather than hidden.

Instance ranges are present because plain instancing is part of the draw
vocabulary and every backend serves it. A range starting at a non-zero instance is
the `FirstInstance` family instead, and base vertex is the `BaseVertex` family;
neither is here.

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

### 4.5 `IndirectApi`

```rust
trait IndirectApi: FamilyApi {
    type Error;
    fn draw_indirect(&mut self, commands: BufferId, offset: u64, count: u32, stride: u32)
        -> Result<(), Self::Error>;
    fn dispatch_indirect(&mut self, commands: BufferId, offset: u64) -> Result<(), Self::Error>;
}
```

`stride` is stated rather than assumed: a caller may pack records with padding, and
an assumed tight packing reads the wrong offsets.

### 4.6 Markers without traits

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
| `vertex`, `sampler` | implemented, tested |
| Vulkan (`native::vulkan`) | steps 1-2 done and proven on real hardware; steps 3-11 not started |
| DX12, Metal | not started |
| GL family, browser WebGPU convergence | not started |

Test evidence: `common::` and `native::` hold 100+ unit tests, and two
Vulkan tests open a real instance, adapter and `VkDevice` on this machine. Clippy
is clean under `-D warnings` for all-features and no-default-features.
