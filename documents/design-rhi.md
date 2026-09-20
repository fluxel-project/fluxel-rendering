# Fluxel RHI 0.16 public API freeze v13

> Status: final freeze candidate v13; normative target for Fluxel Rendering 0.16.
> Scope: portable public Rust API and the RHI observability required by
> capture/replay tooling.
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
| [02 Resources, upload, and readback](rhi-design/02-resources-transfer.md) | buffers; textures; views; samplers; usage; host layouts; upload jobs; readback guard; retirement | modules 01 and 04 |
| [03 Shader, binding, and pipeline](rhi-design/03-shader-binding-pipeline.md) | shader code/interface/provenance; layouts; bind groups; pipeline interface; raster/compute pipelines | modules 01 and 02 |
| [04 Recording and actual resource uses](rhi-design/04-recording-resource-uses.md) | recorder state machines; scopes; commands; `command::ResourceUse`; `RecordedWork` | modules 02 and 03 |
| [05 Submission, completion, and presentation](rhi-design/05-submission-completion-presentation.md) | plans; plan/completion points; transient allocator access; hazards; async acceptance and completion; async presentation lifecycle | modules 01, 02, and 04 |
| [06 Statistics, diagnostics, validation, and transient lowering](rhi-design/06-statistics-diagnostics-transient.md) | logical counters; diagnostics; canonicalization; validation; transient allocation and aliasing lowering | modules 01, 02, 04, and 05 |
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
4. Recorder derives command-ordered actual `rhi::command::ResourceUse` values.
   Upper-layer scheduling contracts and coverage checks do not enter RHI.
5. `BindGroup` is a validated logical resource packet, not a promise of a
   native descriptor object.
6. `FrameAttachment` is neither `Texture` nor `TextureView`.
7. Presentation is planned before submit. Submit acceptance, GPU completion,
   and present outcome are distinct.
8. RHI retains canonical reconstructable semantics from 0.16 but does not own
   capture dependency closure, artifact storage, snapshots policy, or replay.
9. Statistics are portable logical observations, not native profiling facts.
10. Transient allocation is a frozen RHI capability. Every backend provides
    the correct `Dedicated` baseline; `Aliasing` is an optimization capability.
11. Async marks operations that may wait for a future event. Thread-safe or
    concurrent synchronous work does not become async merely for uniformity.
12. P1/P2 vocabulary is absent until a real consumer, portable semantics,
    capability facts, validation, lifetime, tests, and capture implications are
    reviewed together.
13. Public ownership is an opaque device/context identity. Its uniqueness
    includes the lifecycle generation internally; loss is terminal and a new
    device request creates a new identity rather than reviving old objects.
14. Browser and mini-game adapters use the same RHI resource model. Browser
    session/token types are forbidden in public and backend resource
    architecture; native WebGPU/WebGL objects remain backend-private.

## 4. v13 closure decisions

The owning modules express these decisions directly. This list is an audit
ledger, not an alternate API definition.

- The public module tree is `platform`, `capability`, `format`, `resource`
  (including `resource::transient`), `shader`, `binding`, `pipeline`, `command`,
  `submission`, `presentation`, `statistics`, `diagnostics`, and hidden
  `tooling`. There is no upper-layer scheduling bridge module.
- `ResourceUse` belongs to `rhi::command` and is derived from recorded portable
  commands. External work declarations and coverage checks do not exist in the
  RHI.
- `PlatformProvider::enumerate_adapters`, `PlatformProvider::request_device`,
  shader/pipeline creation, submission, completion waits, readback readiness,
  presentation configuration/acquire/abandon, present waits, and `wait_idle`
  are async. Capability queries, logical resource/binding/interface creation,
  command recording, statistics, and diagnostics remain synchronous.
- `ReadbackTicket::read().await` returns a `ReadbackView<'_>` RAII guard so a
  backend can end a map/unmap lease on `Drop`.
- Transient lifetime is expressed solely with `PlanPoint`: an acquire point and
  a release frontier. `SubmissionPlanBuilder::reserve_batch`, `set_batch`, and
  `transient_allocator` make that lifetime constructible before work is added.
- RHI lowers transient reuse from PlanPoint ordering, actual `ResourceUse`, and
  the physical alias relation. DX12/Vulkan/Metal may advertise `Aliasing`;
  WebGPU/GL-family backends can remain correct with `Dedicated`.

- Device loss is terminal. Re-requesting creates a fresh `DeviceIdentity`;
  public APIs expose no mutable generation counter and never revive old
  resources. v13 exposes stable synchronous `status()` / `loss_info()`, not a
  `Device::lost()` future or separate loss event: once loss is observed, pending
  completion/readback/acquire/present/wait-idle operations are woken into their
  terminal DeviceLost outcomes, while an idle backend may defer discovery until
  the next RHI call.

No implementation may silently revert one of these corrections to match an old
prototype.

## 5. Layer ownership

```text
Renderer / material graph / custom pipeline policy
    -> optional upper-layer scheduling
    -> RHI RecordedWork + SubmissionPlan + TransientLifetime
    -> backend-private lowering
    -> DX12 | Vulkan | Metal | WebGPU | GL family
```

Materials own shader composition, parameters, variants, and authoring
provenance. Custom render pipelines own frame topology and renderer policy.
An upper layer may own its own declarations and scheduling, but none of that
vocabulary enters RHI. RHI owns portable execution, actual resource uses,
transient allocation semantics, device validation, submission, completion,
presentation, retirement, logical observation, and backend lowering. A fixed
renderer is only the first consumer of these contracts.

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
evidence gates. It cannot weaken or postpone an API freeze v13 P0 requirement.
[The foundation contract](design-foundation-interfaces.md) controls cross-layer
ownership. [RenderGraph](design-rendergraph.md) and
[capture/replay](design-capture-replay.md) consume the portable RHI and hidden
tooling seam defined here.

RHI implementation starts only at 0.16 and closes native backends before 0.17
closes WebGPU and the GL family. RenderGraph implementation starts only after
all RHI platforms pass. Capture/replay product implementation starts after
RenderGraph closes, while its RHI tooling prerequisites are implemented and
tested from 0.16.
