# Fluxel RHI public API v1

> Status: normative target for Fluxel Rendering 0.16.
> Scope: portable public Rust API, the RenderGraph-to-RHI engine bridge, and
> the RHI reconstructability required by capture/replay.
> Baseline: DX12, Vulkan, Metal, WebGPU, OpenGL, GLES, and WebGL2.
> Not frozen here: native lowering, backend-internal objects, ABI, persistent
> capture artifact format, and ReplayRuntime.

This file is the sole entry point and conflict authority for RHI semantics.
The files under [`rhi-design/`](rhi-design/) are normative modules of this
same specification, not independent design documents. Every public interface is
defined exactly once in one module. README files, ADRs, historical stage
journals, backend notes, RenderGraph documents, and implementation code may
consume this specification but may not redefine it.

## 1. Required reading protocol

Before changing RHI or an RHI consumer:

1. read this file in full;
2. read the complete owning module from the table below;
3. read every adjacent module named by that module's dependency note;
4. identify the numbered API sections and validation rules affected;
5. implement one version-plan gate at a time;
6. run the owning contract tests and all affected ecosystem integration tests;
7. update evidence without changing the normative contract implicitly.

Do not infer API from old code, README examples, ADR-0006/0007, or
`.github/stages/stage-03f-common-rhi-layer.md`. When a required change crosses
module boundaries, stop coding until the impact set and compatibility decision
are written down.

## 2. Normative module map

| Module | Owns | Required with |
| --- | --- | --- |
| [01 Platform, device, and capability](rhi-design/01-platform-device-capability.md) | principles; freeze scope; modules; identity; errors; provider/adapter/device; capability/format/route/lane facts | always read for device, capability, error, or backend admission work |
| [02 Resources, upload, and readback](rhi-design/02-resources-transfer.md) | buffers; textures; views; samplers; usage; host layouts; upload jobs; encoded readback; retirement | module 01; module 04 for encoding |
| [03 Shader, binding, and pipeline](rhi-design/03-shader-binding-pipeline.md) | shader code/interface/provenance; layouts; bind groups; pipeline interface; raster/compute pipelines | modules 01 and 02 |
| [04 Recording and actual resource uses](rhi-design/04-recording-resource-uses.md) | recorder state machines; scopes; commands; actual uses; RecordedWork; declared-use validation | modules 02 and 03 |
| [05 Submission, completion, and presentation](rhi-design/05-submission-completion-presentation.md) | plans; points; hazards; acceptance; receipts; completion; target/configure/acquire/present/abandon | modules 01 and 04 |
| [06 Statistics, diagnostics, and graph bridge](rhi-design/06-statistics-diagnostics-graph-bridge.md) | logical counters; diagnostics; canonicalization; validation; Graph bridge; transient allocation service | modules 01, 04, and 05 |
| [07 Tooling and capture prerequisites](rhi-design/07-tooling-capture-prerequisites.md) | tooling SPI; object/work descriptions; portable command/submission IR; semantic events; RHI/capture ownership split | modules 02 through 06 |
| [08 Governance and freeze checklist](rhi-design/08-governance-freeze-checklist.md) | forbidden public shapes; deferred-feature gate; lifecycle matrix; P0 checklist; final cross-review decisions | all affected modules |

Section numbers 0-66 remain stable across the modules. A tool may load a module
by section range, but it must not load a section excerpt without the module's
introductory contract and this root file.

## 3. Global invariants

The following apply to every module and backend:

1. Capability is instance data for adapter, device, format, surface, or route;
   it is never inferred from Rust trait presence.
2. Base guarantees at least one ordered submission lane. Multiple logical lanes
   do not promise physical queues or hardware overlap.
3. Public RHI exposes no native handle, barrier, fence, semaphore, queue,
   descriptor heap, native memory type, heap offset, encoder, or resource state.
4. RenderGraph owns declarations, versions, dependencies, culling, scheduling,
   logical lifetime, and presentation intent. Recorder owns command-ordered
   actual uses. The graph bridge validates coverage without creating a second
   dependency truth.
5. `BindGroup` is a validated logical resource packet, not a promise of a
   native descriptor object.
6. `FrameAttachment` is neither `Texture` nor `TextureView`.
7. Presentation is planned before submit. Submit acceptance, GPU completion,
   and present outcome are distinct.
8. RHI retains canonical reconstructable semantics from 0.16 but does not own
   capture dependency closure, artifact storage, snapshots policy, or replay.
9. Statistics are portable logical observations, not native profiling facts.
10. P1/P2 vocabulary is absent until a real consumer, portable semantics,
    capability facts, validation, lifetime, tests, and capture implications are
    reviewed together.
11. Public ownership is opaque device/context identity plus generation. Loss is
    terminal: restoration creates a new identity/generation domain and never
    increments a field to revive old objects.
12. Browser and mini-game adapters use the same RHI resource model. Browser
    session/token types are forbidden in public and backend resource
    architecture; native WebGPU/WebGL objects remain backend-private.

## 4. v1 closure corrections

The former freeze candidate was adopted in full after cross-review, with the
following corrections. The owning modules must express these decisions
directly; this list is the audit ledger, not an alternate API.

- Device identity includes the ecosystem-required generation component while
  preserving terminal loss and no transparent recovery.
- An external completion dependency succeeds on `Ordered` or proven
  `Collapse` routes as an ordered-domain relation. It is `Unsupported` only
  when neither GPU dependency nor proven ordering/collapse can satisfy it.
- P0 guarantees raster output to `FrameAttachment`. Direct MSAA resolve into a
  frame is legal only when presentation/route facts prove it; otherwise the
  portable path resolves to an intermediate texture and performs a final
  single-sample raster write.
- `SubmissionPlanBuilder` owns a consumed frame. Build failure or builder drop
  performs no-submit abandonment bookkeeping; no acquired frame is leaked.
- Presentation targets and configured-presentation leases have canonical
  tooling definitions, so every `ObjectId` emitted by presentation events can
  be described.
- Recorder, raster scope, and compute scope each own an independent debug-group
  stack; closing a scope or finishing a recorder requires its corresponding
  stack to be empty.
- Compute shader reflection includes workgroup dimensions, total invocations,
  and workgroup/shared-memory requirements, which are validated before pipeline
  creation.
- Shader interface arrays are canonical and duplicate-free. Artifact identity
  includes producer/toolchain identity and a specified canonical hash domain;
  executable-only provenance states its replay acceptance scope.
- A fixed binding array is conservatively considered fully used unless future
  certified element-use metadata proves a narrower set.
- Clear values are validated against attachment numeric class; depth clear is
  finite and within the portable depth range.
- Texture resolve has explicit source and destination origins.
- Recorder actual uses encode upload/readback in the copy domain. Host access
  bits are reserved for Graph/tooling host observations and are not emitted as
  GPU-command scope uses.
- Tooling subscription defines start/drop linearization and callback lifetime;
  a callback cannot drop its own subscription or reenter a mutating operation
  on the same device.
- Logical presentation statistics distinguish acquire refusal from submitted
  present outcomes; acquire-only failures do not increment a present-terminal
  category.

No implementation may silently revert one of these corrections to match an old
prototype.

## 5. Layer ownership

```text
Renderer / material graph / custom pipeline policy
    -> RenderGraph declaration and object recipes
    -> GraphExecutionPlan
    -> RHI RecordedWork + SubmissionPlan
    -> backend-private lowering
    -> DX12 | Vulkan | Metal | WebGPU | GL family
```

Materials own shader composition, parameters, variants, and authoring
provenance. Custom render pipelines own frame topology and renderer policy.
RenderGraph owns dependency and lifetime compilation. RHI owns portable
execution, device validation, submission, completion, presentation, retirement,
logical observation, and backend lowering. A fixed renderer is only the first
consumer of these contracts.

## 6. Change control

A public RHI change is admissible only when its proposal names:

- the owning module and exact affected sections;
- a real consumer;
- portable semantics and structured refusal behavior;
- adapter/device/format/surface/route facts;
- lifetime, loss, completion, and presentation consequences;
- RenderGraph and tooling/capture consequences;
- shared contract tests and the complete backend evidence matrix;
- ecosystem dependency revisions and integration artifacts.

Changing code first and repairing the specification later is not accepted.
Optimizations may change private lowering only when the serial semantic oracle,
actual-use trace, output, terminal states, and capture-visible behavior remain
equivalent.

## 7. Delivery authority

[The five-version plan](version-plan.md) controls implementation order and
evidence gates. It cannot weaken or postpone an API v1 P0 requirement.
[The foundation contract](design-foundation-interfaces.md) controls cross-layer
ownership. [RenderGraph](design-rendergraph.md) and
[capture/replay](design-capture-replay.md) consume the engine/tooling bridges
defined here.

RHI implementation starts only at 0.16 and closes native backends before 0.17
closes WebGPU and the GL family. RenderGraph implementation starts only after
all RHI platforms pass. Capture/replay product implementation starts after
RenderGraph closes, while its RHI tooling prerequisites are implemented and
tested from 0.16.
