# Fluxel RHI

`fluxel-rhi` is Fluxel's portable GPU execution layer.  It gives a renderer one
vocabulary for devices, resources, shader artifacts, recorded commands,
submission, completion, and presentation, while keeping DX12, Vulkan, Metal,
WebGPU, and the GL family behind the backend seam.

It is intended to be used below a renderer or render graph, not as a scene
graph, asset cache, shader compiler, or native-handle wrapper.  In particular,
the public API never exposes `ID3D12Device`, `VkDevice`, `MTLDevice`,
`GPUDevice`, or a GL context.

## Start here

Most callers use these public modules:

| Need | Public module |
| --- | --- |
| Discover/open a device and inspect its state | `api::platform` |
| Ask what the opened device can do | `api::capability`, `api::format` |
| Create buffers, textures, views, samplers, uploads, and readbacks | `api::resource` |
| Describe shader artifacts and interfaces | `api::shader` |
| Define bind-group layouts, bind groups, and pipeline interfaces | `api::binding`, `api::pipeline` |
| Record raster, compute, copy, resolve, upload, and readback work | `api::command` |
| Build a dependency-aware submission and observe completion | `api::submission` |
| Configure a target, acquire a frame, and observe presentation | `api::presentation` |
| Use query sets | `api::query` |

`api::RhiError` / `RhiErrorKind` are the structured error surface.  Logical
handles are opaque and device-owned: passing a handle to another `Device` is an
immediate `WrongDevice` error rather than a driver call.

## Typical frame workflow

The provider is supplied by Fluxel's platform/host integration. Applications
never construct one from raw native objects: on backends with process-owned
discovery they call the matching public composition function (for example
`create_dx12_provider` or `create_vulkan_provider`); adopted-context backends
need a platform bridge that creates the same portable provider without exposing
a native context. That GL/browser bridge is a tracked composition TODO, not a
reason to pass a native context through the RHI API.

```rust,no_run
use fluxel_rhi::api::{
    command::RecordedWork,
    platform::{AdapterSelection, DeviceRequestDescriptor, DeviceRequirements, PlatformProvider},
    submission::SubmissionPlanBuilder,
};

async fn submit_work(
    provider: &PlatformProvider,
    work: RecordedWork,
) -> fluxel_rhi::api::RhiResult<()> {
    // Enumeration is optional: browser and adopted-context providers may return
    // Ok(None), and request_device remains usable in that case.
    let _adapters = provider.enumerate_adapters().await?;

    let device = provider
        .request_device(DeviceRequestDescriptor::new(
            AdapterSelection::Default,
            DeviceRequirements::new(),
        ))
        .await?;

    // Resource, layout, bind-group, recorder, and command calls are synchronous
    // logical operations. Shader and raster/compute pipeline creation may await
    // backend compilation.
    let lane = device.capabilities().submission().lanes()[0].id();
    let mut plan = SubmissionPlanBuilder::new(&device);
    plan.add_batch(lane, vec![work])?;
    let receipt = device.submit(plan.build()?).await?;

    // Acceptance, GPU completion, and presentation are intentionally separate.
    let _state = device.wait_completion(receipt.completion()).await?;
    Ok(())
}
```

In normal code the `RecordedWork` above comes from this shape:

```text
Device::create_recorder
  -> CommandRecorder
  -> begin_raster / begin_compute / copy / encode_upload / encode_readback
  -> CommandRecorder::finish
  -> RecordedWork
  -> SubmissionPlanBuilder
  -> Device::submit(...).await
```

For presentation, configure a `PresentationTarget`, then repeat:

```text
Device::configure_presentation(...).await -> ConfiguredPresentation
ConfiguredPresentation::acquire().await   -> AcquiredFrame
plan.present_after(frame, point)
Device::submit(plan).await
Device::wait_present(receipt_id).await     -> PresentState
```

`AcquiredFrame` is a non-cloneable lease.  Submit it through `present_after`, or
explicitly call `frame.abandon().await`; do not treat a frame attachment as an
ordinary persistent `TextureView`.

## Capability-first programming

Do not branch on `BackendKind` to infer GPU support.  Backend family is useful
for diagnostics, selection, shader-artifact provenance, and tooling; it is not
a feature level.  Query the opened device instead:

```rust,ignore
if device.capabilities().supports_feature(OptionalFeature::SamplerAnisotropy) {
    // Request and use anisotropy within MaxSamplerAnisotropy.
}

let support = device.capabilities().texture_support(TextureSupportQuery::new(/* ... */));
```

The same rule applies to formats, binding forms, routes, presentation modes,
query profiles, and limits. The frozen contract requires a finalized immutable
snapshot whose public queries are total. Construction-time finalization of
every finite query domain is still an implementation closure item: the current
snapshot can panic when a backend omits a required finite-domain entry. That is
a backend-construction defect, not a supported public outcome, and must be
rejected before the device is exposed. A capability reported as unsupported
must be handled by a fallback or rejected by the caller; the RHI does not
silently emulate a feature with different semantics.

## Async and lifetime model

Only operations that wait for an external or GPU event are async:

- adapter/device acquisition;
- shader and raster/compute pipeline creation;
- submission, completion, idle, readback, presentation configuration,
  acquire/abandon/reconfigure, and present waiting;
- buffer mapping and `ReadbackTicket::read()`.

Logical object construction (`create_buffer`, `create_texture`, views, samplers,
bind groups, layouts, pipeline interfaces, recorders) is synchronous.  This is
intentional: thread-safe construction is not by itself a future-producing
operation.

`ReadbackTicket::read().await` returns a `ReadbackView` guard rather than a bare
slice.  Keep that guard alive only while consuming the mapped bytes; dropping it
ends the backend mapping lease when one exists.

A device loss is terminal for that `DeviceIdentity`.  Pending completion,
readback, acquire, and presentation futures resolve to a terminal result, later
operations return `DeviceLost`, and callers can synchronously inspect
`Device::status()` and `Device::loss_info()`.  Request a new device instead of
trying to revive old resources.

## Submission and transient resources

`RecordedWork` contains the command-derived `command::ResourceUse` summary.
The submission builder uses that actual use to validate lane compatibility and
unordered write hazards.  There is no render-graph declared-use contract in the
RHI; graph scheduling remains an upper-layer responsibility.

For plan-scoped temporary resources, reserve plan points before recording their
lifetime:

```rust,ignore
let mut plan = SubmissionPlanBuilder::new(&device);
let first = plan.reserve_batch(lane)?;
let last = plan.reserve_batch(lane)?;
let transient = plan.transient_allocator();
let texture = transient.create_texture(
    &descriptor,
    TransientLifetime::new(first).release_at(last),
)?;
// Record work using `texture`, then fill both reserved batches with set_batch.
```

All backends provide the correctness baseline (`Dedicated`).  Aliasing is a
separate capability and backend-private allocation strategy; callers do not
write aliasing barriers.

## Backend boundary and availability

Cargo features select code that can provide a backend; they are not support
claims.  Actual availability is determined by target platform, host/platform
integration, runtime probing, the device capability snapshot, and conformance
evidence.  The feature names are `dx12`, `vulkan`, `metal`, `webgpu`, and
`gl-family` (with `native-gl-wgl`, `native-gles-egl`, or `webgl2` integrations
where applicable).

Native resource state transitions, descriptor allocation, queue/fence details,
browser objects, context state, and native synchronization remain backend
private.  The portable API exposes logical lanes, plan points, completion
points, and resource intent instead.

Some vocabulary intentionally remains provisional/experimental and fail-closed
until the complete portable path is available on a backend. Consult the
capability snapshot rather than assuming that a descriptor type implies support;
advanced mesh/task shader, ray-tracing, cooperative-matrix, and
transient-aliasing families are staged work in the design/ADR set and do not
freeze descriptor ABI.

## Examples and further reading

- [`tests/`](tests/) is the real-adapter conformance baseline: focused
  readback assertions are distinct from portable unit tests and from visible
  surface smoke tests.
- [`examples/`](examples/) explains the host-injected v13 workloads and the
  public provider-composition entry points. Examples may create a portable
  provider through those functions, but never receive a native provider or
  platform handle.
- [`crates/rendergraph/examples/`](../rendergraph/examples/) shows render-graph
  declaration and compilation on top of the RHI vocabulary.
- [`crates/renderer/examples/01_headless_frame.rs`](../renderer/examples/01_headless_frame.rs)
  shows a renderer-level headless lifecycle.
- [`examples/android-vulkan-wsi/`](../../examples/android-vulkan-wsi/) is the
  Android Vulkan WSI evidence harness.
- [`examples/windows-dx12/`](../../examples/windows-dx12/) is retained as
  historical presentation evidence; read its warning before treating it as a
  current API tutorial.
- [`../../documents/adr/`](../../documents/adr/) records architectural decisions;
  [`documents/design-rhi.md`](documents/design-rhi.md) is the compact RHI design
  companion and TODO index.

Rustdoc on the types named above is the authoritative detail for descriptors,
validation, and errors.  The design documents explain *why* the public shape is
that way; backend implementation details are deliberately not part of this
user-facing contract.
