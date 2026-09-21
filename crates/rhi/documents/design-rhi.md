# Fluxel RHI design

This is the short architecture guide for `fluxel-rhi`. The public Rust API,
its rustdoc, and its positive/negative/boundary tests are the executable API
reference. This document records the stable model, points to the decisions
that explain it, and names work that is deliberately not enabled yet.

## Scope

RHI owns portable GPU execution: device-scoped resources, shader and pipeline
creation, recording actual uses, plans, submission, completion, presentation,
readback, retirement, capability reporting, diagnostics, and observability.

It does not own a RenderGraph, renderer scheduling, material policy, native
handles, native synchronization objects, capture artifacts, or a replay
runtime. Backends keep their native objects and strategy private.

## Public model

```text
PlatformProvider -> Adapter -> Device
Device -> resource / shader / binding / pipeline / command
CommandRecorder -> RecordedWork -> SubmissionPlan -> CompletionPoint
ConfiguredPresentation -> AcquiredFrame -> FrameAttachment -> present outcome
```

The public modules are `platform`, `capability`, `format`, `resource`,
`shader`, `binding`, `pipeline`, `command`, `submission`, `presentation`,
`statistics`, `diagnostics`, and crate-private tooling support.

## Invariants

- Every object belongs to one opaque `DeviceIdentity`; loss terminates that
  identity and a replacement device receives a new one.
- Capability facts are the authority. A supported fact requires validation,
  native lowering, lifetime/loss handling, and conformance evidence.
- `command::ResourceUse` is derived from recorded commands. It includes
  scheduling-only query-slot writes and resolve reads without pretending a
  `QuerySet` is a buffer. RenderGraph declarations and scheduler contracts are
  intentionally outside RHI.
- Only operations which may wait for a future event are async. Logical object
  creation, validation, capability queries, and recording are synchronous.
- `submit(Err)` accepts no native work. `submit(Ok)` transfers plan ownership;
  acceptance, GPU completion, and present outcome remain distinct states.
- `FrameAttachment` is not a `Texture` or `TextureView`. It is valid only for
  the acquired-frame lifecycle that produced it.
- Transient lifetimes use `PlanPoint` ordering. `Dedicated` allocation is the
  correct baseline; aliasing is a private, capability-gated optimization.

## Decision records

| ADR | Decision |
| --- | --- |
| [0012](../../../documents/adr/0012-pure-rhi-execution-boundary.md) | Keep RHI limited to portable execution. |
| [0013](../../../documents/adr/0013-async-operation-boundary.md) | Make only future-event operations asynchronous. |
| [0014](../../../documents/adr/0014-terminal-device-loss.md) | Treat device loss as a terminal identity state. |
| [0015](../../../documents/adr/0015-plan-scoped-transient-allocation.md) | Freeze plan-scoped transient lifetimes and dedicated fallback. |
| [0016](../../../documents/adr/0016-capability-claims-require-lowering-closure.md) | Publish facts only after lowering closure. |
| [0017](../../../documents/adr/0017-presentation-is-not-a-texture-view.md) | Keep presentation frames separate from textures and completion. |
| [0018](../../../documents/adr/0018-capture-observability-without-capture-ownership.md) | Preserve observability without owning capture/replay. |
| [0019](../../../documents/adr/0019-portable-logical-statistics.md) | Keep statistics logical rather than native profiling. |
| [0020](../../../documents/adr/0020-optional-feature-family-admission.md) | Admit optional feature families as complete portable contracts. |
| [0021](../../../documents/adr/0021-shader-owned-immediate-abi.md) | Derive executable immediate-data ABI from shader artifacts, not interface supersets. |

Existing ADRs cover unsafe/native containment, serial lowering policy,
conformance evidence, platform test gates, and the GL-family private boundary.

## Current TODO boundary

The public vocabulary already carries advanced mesh/task shaders, ray tracing,
cooperative matrices, aliasing, external interop, multiplanar formats, native
debug capture, HDR/timing, and advanced descriptor indexing. A backend keeps
each incomplete fact disabled and returns structured `Unsupported` before
native work. It must not replace an incomplete path with a panic, dummy
success, or silent fallback.

Current examples include transient physical aliasing, counted indirect paths,
Metal timestamp/statistics, WebGPU timestamp placement/conversion, and
WebGPU host-image interop. See ADR-0020 and the capability facts for the exact
per-backend answer.

## Change rule

For a new stable boundary: write or update one focused ADR, update public API
and rustdoc, add positive/negative/boundary tests, then implement and test
every backend that publishes the capability. Performance-only changes stay
backend-private provided observable execution, loss, completion, and tooling
semantics remain unchanged.
