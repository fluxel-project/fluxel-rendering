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
| [08 Governance and freeze checklist](rhi-design/08-governance-freeze-checklist.md) | forbidden native public shapes; capability-family admission gate; lifecycle matrix; cross-review decisions | all affected modules |
| [09 Capability-complete feature families](rhi-design/09-capability-complete-feature-families.md) | optional GPU feature vocabulary, capability closure, wgpu-hal 30.0.1 correspondence, backend admission, and definition of done | module 01 and every owning module named by a feature family |

Section numbers 0-67 remain stable across the modules. A tool may load a module
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
12. Every mature portable GPU feature family has a public semantic vocabulary.
    Whether an adapter/device can execute it is decided by capability facts and
    structured `Unsupported`, never by deleting the vocabulary because a
    baseline backend lacks it. A feature may be admitted only with portable
    semantics, validation, lifetime/loss rules, capture implications, and
    backend conformance evidence.
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

## 4.1 Capability-complete feature admission

The v13 foundation is frozen; [module 09](rhi-design/09-capability-complete-feature-families.md)
extends it with the feature families established by `wgpu-hal 30.0.1`. These
families are part of the RHI contract, not a deferred/minimal subset. Each one
has a public semantic, precise capability or descriptor-dependent support
query, requirements negotiation, portable validation, loss behaviour, and a
backend conformance case. A backend reports `Unsupported` before native work
when it cannot lower a requested optional feature.

HAL mechanisms which already have an RHI equivalent are deliberately not
duplicated: HAL barriers map to `command::ResourceUse`, fences to completion,
queues to lanes/plans, encoder recycling to private lowering, and native
destroy to ownership plus completion-safe retirement. This is equivalence, not
feature exclusion.

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

## 5.1 Ten-class native-lowering status matrix

The v13 public surface already carries all ten cross-platform native-lowering
classes below. They are implementation work, not deferred public API design.
The status is deliberately about what a device may publish today, rather than
what its native API happens to name. DX12 and Vulkan are the reference native
backends; other backends follow the same admission rule. An implementation may
add private modules, traits, data structures, and conformance tests without
widening the public API.

| Native-lowering class | Complete public carrier | DX12 / Vulkan published status | Before-native refusal or required fallback | Admission condition for a future stronger lowering |
| --- | --- | --- | --- | --- |
| 1. Persistent RTV/DSV or equivalent attachment allocator | `TextureView`, `FrameAttachment`, attachment uses in `RecordedWork` | Private baseline only; neither backend publishes a public descriptor-allocator capability. | No optional public request exists: allocate/reuse private descriptors as needed and preserve attachment semantics. | A persistent pool must prove descriptor lifetime through the last use `CompletionPoint`, recycle only after retirement, survive loss without stale leases, and pass repeated view/frame attachment conformance. Native descriptor storage remains private. |
| 2. Resource-state difference tracker | command-ordered `command::ResourceUse`, `RecordedWork`, `PlanPoint` order | Private correctness lowering; no public "state tracker" feature is published. | No optional public request exists: emit conservative legal DX12 transitions or Vulkan layouts/dependencies rather than guessing a state. | Track every accepted use, queue/layout/state boundary and alias relation; emit only legal differences; retain correct final state across batches; pass cross-batch, read/write, presentation and loss conformance. |
| 3. Shared completion/fence waiter | `CompletionPoint`, `completion_state`, `wait_completion`, `wait_idle` | The completion API is published; waiter topology is private on both backends. | N/A to capability: each pending completion must still settle as Complete, Failed, or DeviceLost. | One native notification mechanism may multiplex waiters only after race-free registration/removal, no lost wakeups, loss wakeup of every pending future, and cancellation/idle conformance. |
| 4. Descriptor and resource retirement | ownership plus last actual-use `CompletionPoint` | Required baseline semantic on DX12 and Vulkan; allocator strategy is private. | N/A to capability: native backing must not be reused/destroyed before completion-safe retirement. | Deferred pools/heaps may be introduced only with exact last-use retention, loss draining, no descriptor reuse while GPU-visible, bounded diagnostics, and drop/submit/readback/present conformance. |
| 5. Submit-lock narrowing | `submit(plan).await` acceptance and terminal-state contract | Private synchronization policy; no backend publishes lock granularity. | N/A to capability: retain a conservative lock if needed. `Err` still means zero native work accepted. | Split locks only after preflight/commit atomicity, cross-plan hazard ordering, serial allocation, post-commit loss publication and simultaneous-submit conformance remain identical. |
| 6. Recorder arena/packet plus binding/state-difference cache | synchronous recorder, `RecordedWork`, canonical tooling descriptions | Private baseline on DX12 and Vulkan; no native encoder/packet type is public. | N/A to capability: direct private recording/lowering remains valid. | Arena reuse and deduplication must reconstruct every frozen command and exact `ResourceUse`, retain every referenced backing through acceptance, invalidate on pipeline/heap/loss changes, and pass capture/replay plus state-change conformance. |
| 7. Pipeline cache | async shader/pipeline creation; canonical shader/layout/pipeline descriptors; `PipelineCache` facts | DX12 and Vulkan currently publish cache and serialization through their wired native cache paths. A backend without that complete path publishes neither fact. | `PipelineCache` request/serialization is rejected as `Unsupported` before native work when the corresponding fact is absent; uncached pipeline creation remains the required fallback. | Probe/cache-device identity and validation key; enable/create the native cache; apply declared invalid-data policy; feed every supported pipeline creation through it; retain it through outstanding creation; serialize/restore and test invalid, loss and driver-mismatch cases. No cache-file ABI is frozen. |
| 8. Placed heap / transient aliasing | `TransientAllocationSupport`, `TransientLifetime`, `PlanPoint`, actual `ResourceUse` | DX12 and Vulkan currently publish `Dedicated`; `Aliasing` is not published. | A requested aliasing path is Unsupported before native work; `Dedicated` is the correct universally required allocation fallback. | Probe allocation/memory requirements; create compatible placed/shared allocations; prove non-overlap from lifetime frontiers; lower DX12 alias barriers or Vulkan memory/layout dependencies; retain physical memory until all aliases retire; pass overlap, ordering, loss and reuse conformance. |
| 9. Multiple native queues | `SubmissionLane`, dependency routes, `PlanPoint`, `CompletionPoint` | DX12 and Vulkan publish their currently implemented ordered lane set; unimplemented queue classes/routes are not published. | Unsupported route/lane requests fail before native queue work. Collapsing to one ordered native queue is valid only for already advertised semantics. | Probe queue families/classes and presentation compatibility; enable/select queues; lower every inter-lane dependency to native signal/wait and ownership transfer where needed; retain synchronization through completion; pass concurrent, dependency-cycle, presentation and loss conformance. |
| 10. Primary-buffer mapping and persistent mapping leases | `MapMode`, range RAII, flush/invalidate, mapping capabilities, `DeviceIdentity`, completion/loss | DX12/Vulkan publish only the exact mappable primary-buffer masks their memory/heap path can uphold. `PersistentMapping` remains unadvertised until its ownership contract is met. | Unsupported map mask, range, coherence operation or persistent lease is rejected before native map; no hidden staging or implicit CPU/GPU race is permitted. | Probe legal heap/memory type and coherency; enable/select it at allocation; wait for last GPU use before grant; lower atom-aligned Vulkan flush/invalidate or truthful DX12 coherence; retain the map lease and wake/cancel it on loss; pass read/write, exact-end, non-coherent, overlap and loss conformance. |

An unavailable enhancement is not an API stub. Capabilities must describe only
lowerings that are implemented and tested: unsupported optional behavior
returns structured `Unsupported` at the capability/operation boundary, while a
correct fallback (`Dedicated`, ordered lane, or private uncached lowering) is
used where the frozen contract requires one. A public or reachable backend
execution path must never use `todo!()` or `unimplemented!()` as its result.
Private `TODO` comments are permitted only when they name the carrying
semantics, the required fallback, and the condition for advertising the
enhancement.

## 5.2 Advanced native-lowering TODO boundary

The following ten families are **not deferred public API design**: their public
types, capability/limit vocabulary, portable validation, positive/negative/
boundary contract tests, resource-use representation, and tooling schema are
part of v13. What may remain for a backend is only the advanced native lowering.
Until every admission condition in a row is met, DX12/Vulkan must keep the
corresponding fact disabled and return structured `Unsupported` before the first
native operation. A comment may say `TODO(native-lowering)`; a reachable
`todo!()`, `unimplemented!()`, panic, dummy success, or silent no-op is forbidden.

| Advanced family | Frozen public carrier | Current correct DX12/Vulkan behavior | Required evidence before enabling |
| --- | --- | --- | --- |
| Mesh / task shaders | mesh/task stages, `MeshPipeline`, direct/indirect/count mesh commands and mesh limits | Capability false and pre-native `Unsupported` unless a backend has the complete path | Native tier/extension probe and device enablement; shader acceptance; pipeline creation; dispatch lowering; argument/resource retention; loss and conformance tests. |
| Ray system | BLAS/TLAS descriptors and sizing, build/update/copy/compaction, ray-query requirements, ray pipeline/groups, SBT and `trace_rays` | Keep each independently incomplete fact false; no AS/pipeline/dispatch placeholder may succeed | Exact size/alignment query; native allocation/build barriers; update/compaction; descriptor binding; pipeline/SBT construction; trace dispatch; lifetime/loss and positive/negative/boundary tests. |
| Cooperative matrix | structured matrix properties and shader requirements | No matching property means Unsupported | Exact native configuration enumeration, logical-device enablement, shader compiler acceptance and a conformance shader for every published tuple. |
| Transient physical aliasing | `TransientAllocationSupport`, `TransientLifetime`, `PlanPoint`, actual `ResourceUse` | Publish `Dedicated`; do not publish `Aliasing` | Compatible heap/memory placement, lifetime non-overlap proof, DX12 alias barrier/Vulkan dependency, completion-safe reuse and overlap/loss tests. |
| External interop | external image/texture and typed external-memory import SPI | Capability false where platform ownership and synchronization are not implemented | Handle/source provenance, format/usage validation, ownership transfer, native synchronization, retirement and platform integration tests. |
| NV12/P010 multiplanar | planar formats/aspects, copy layout, view and support queries | Per-format Unsupported where plane-aware views/copies are absent | Per-plane format probe, plane view/copy lowering, Vulkan YCbCr or DX12 plane semantics, row/layout validation and sampling/copy conformance. |
| Native debugger capture | capability-gated begin/end native capture | Unsupported when PIX/RenderDoc/native integration is unavailable | Runtime integration probe, balanced begin/end state, loss/error handling and coexistence with semantic capture. |
| Advanced allocator/memory diagnostics | allocator report with optional committed/resident/aliased/retired/budget fields | Return only truthful known fields; capability false if no useful report exists | Defined measurement source and quality, overflow-safe aggregation, loss behavior and tests distinguishing unknown from zero. |
| HDR and presentation timing | color-space pairs, HDR display data, generic timing capabilities/timestamps | Do not advertise HDR/timing routes not consumed by swapchain/present lowering | Surface-specific probe, configuration consumption, timestamp conversion/order, reconfigure/loss behavior and real WSI tests. |
| Advanced descriptor indexing | runtime-sized arrays, partially-bound/non-uniform indexing and their limits | Fixed arrays may remain supported; advanced facts stay false independently | Descriptor-indexing feature probe and device enablement, layout/pool flags, shader acceptance, bounds/lifetime rules and per-route conformance. |

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
