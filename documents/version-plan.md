# Fluxel rendering foundation: versions 0.16-0.20

> Status: normative execution order.
> Supersedes: any older plan that resumes the scene/renderer roadmap before the
> RHI, RenderGraph, and capture/replay foundation is closed.

This plan deliberately pauses new high-level rendering work for five minor
versions. The goal is not to keep the current fixed renderer running through
every intermediate commit; the goal is to rebuild and prove the foundation in
dependency order, then reconnect the higher layers once. The stable RHI
architecture target is [RHI design](../crates/rhi/documents/design-rhi.md);
public details live in rustdoc and contract tests. The
[workspace architecture](design-overview.md) owns only the
cross-layer framework and Graph/capture integration.

## 1. Non-negotiable order

```text
0.16  RHI contract + DX12/Vulkan/Metal
  ↓ all 0.16 gates close
0.17  WebGPU + GL family + cross-platform RHI closure
  ↓ all RHI platforms and ecosystem consumers close
0.18  RenderGraph authoring/compiler/instantiation core
  ↓ all 0.18 graph gates close
0.19  RenderGraph allocation/scheduling/trace closure
  ↓ all scoped RenderGraph behavior closes
0.20  portable capture/replay implementation and artifact freeze
  ↓ replay closes on the declared target matrix
resume renderer/scene/canvas/runtime work from the architecture roadmap
```

No implementation work from a later row may be used to declare an earlier row
complete. The only capture/replay work allowed before `0.20` is the RHI/Graph
interface and diagnostic hook that must exist from their first definition.

“Complete” means the scoped contract and its refusal behavior are implemented,
tested, documented, and retained as evidence. It does not mean every P2 GPU
family listed for future review has been implemented.

## 2. Worktree policy during the foundation train

High-level crates may be temporarily disabled, feature-gated, or commented out
while their dependency is replaced. In particular, `fluxel-renderer`, browser
WASM entry points, examples, and fixed recipes do not constrain an intermediate
RHI internal shape. This permission has limits:

- keep source history; do not delete higher-level behavior merely to make the
  workspace compile;
- keep the workspace manifest parseable and make intentionally dormant targets
  explicit rather than letting them fail accidentally;
- do not replace a typed contract with `dyn Any`, strings, native handles,
  browser session/token types, or backend conditionals in public APIs;
- do not publish a release while a crate advertised by that release is only
  commented out;
- reactivate and migrate every affected target in the release that owns its
  integration gate;
- do not use old high-level tests as proof of a newly written lower layer; add
  contract and backend evidence at the owning boundary.

Each minor version begins from the previous version's released tag, not from an
unrecorded local state. Every cross-repository consumer is pinned to an exact
commit during validation and to a released tag before closure.

## 3. Common completion gate

Every version must satisfy all applicable rows before the next version opens.

1. Public RHI interfaces match [RHI design](../crates/rhi/documents/design-rhi.md), while cross-layer
   interfaces match the [workspace architecture](design-overview.md).
   Deviations require an accepted ADR and an update to the owning contract.
2. Unit/contract tests cover success, every named structured refusal, stale
   generation, and loss/terminal lifecycle.
3. Formatting, MSRV, full-feature tests, Clippy with warnings denied, and docs
   succeed for every active target.
4. Real GPU/platform runs use the production route, required native validation
   where available, structured diagnostics, deterministic inputs, an independent
   oracle, and completion-safe teardown.
5. Evidence records repository and dependency SHAs, OS/target, backend, adapter,
   device/driver/browser, commands, results, diagnostics, and artifact hashes.
6. Participating repository integration passes at exact revisions. A local
   single-repository green run is insufficient.
7. Documentation describes only observed support. Compile-only or emulator
   evidence is labeled as such and never upgraded to a real-target claim.

Baseline repository commands remain:

```text
cargo +1.87.0 fmt --all -- --check
cargo +1.87.0 test --workspace --all-targets --all-features --locked
cargo +1.87.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo +1.87.0 doc --workspace --all-features --no-deps --locked
```

Platform scripts may add steps but may not omit the contract tests. Native GPU
work also runs `scripts/conformance.ps1` or its versioned platform equivalent.

## 4. Version 0.16 - RHI protocol and native backend closure

### Goal

Freeze the portable semantic vocabulary and implement it end to end for DX12,
Vulkan, and Metal. This is the first half of RHI platform closure, not permission
to start RenderGraph replacement work.

### Required implementation

- Complete API v1 provider/adapter/device discovery, runtime-agnostic request,
  presentation-target requirements, `None` versus empty enumeration semantics,
  opaque device/context identity plus generation, terminal loss, and fail-
  closed opening. Recreation creates a new identity domain and never revives
  old handles.
- Available/required/enabled capability separation; per-format, per-route,
  shader-artifact, presentation, and pairwise lane-dependency facts. Delete any
  `supports_real_overlap` contract and do not freeze one global GPU-lane bool.
- Canonical API v1 buffer/texture/view/sampler descriptors and memory
  preferences; retained-byte `UploadJob` encoded into recorder order; explicit
  copies; encoded readback with `NotSubmitted`/`Abandoned` and actual returned
  layout; and completion-safe destruction. Mapping, general host access,
  placement, and compressed/planar formats remain deferred.
- Opaque transient allocation-requirements query (size, alignment,
  compatibility class, dedicated preference). No public heap/memory type and no
  alias implementation is required in this release.
- Shader artifact/replay payload, bind-group layout and validated bind group,
  pipeline interface fingerprint, raster pipeline, capability-gated compute
  pipeline, and render-target signature.
- Recorder/scopes/P0 command vocabulary, command-ordered actual uses,
  `RecordedWork`, `SubmissionPlanBuilder`, same-plan and in-flight-plan hazard
  preflight, external completion dependencies, `SubmissionReceipt`, overall and
  per-`PlanPoint` terminal completion, and retirement. Submit `Err` proves zero
  native work accepted.
- Presentation configuration/acquire/attachment-only `FrameAttachment`, one
  outstanding frame, plan-owned `present_after`, independent present outcome,
  and explicit `abandon`; present mode is configuration.
- Complete API v1 unified error model, logical statistics, diagnostics,
  `graph_bridge`, and separately versioned tooling SPI. RHI retains canonical
  object/command/submission semantics from 0.16 but owns no capture artifact or
  ReplayRuntime.
- Native implementations on DX12, Vulkan, and Metal. Backend objects, native
  synchronization, descriptors, heaps, memory, and unsafe code remain private.
- Remove the borrowed native implementation only after the replacement passes
  equivalent real-target gates; dependency removal is an exit condition, not an
  early cleanup step.

### Required proof

- CPU/mock conformance suite shared by all three backends.
- Windows DX12 and Vulkan: real headless raster/compute/copy/upload/readback,
  surface lifecycle, required validation availability, empty unexpected native
  diagnostics, loss and completion/retirement cases.
- Metal: Apple-target compile plus the same contract suite, followed by a named
  real macOS/Apple-GPU run before a Metal support claim. Compile-only Metal does
  not close `0.16`.
- Pairwise wrong-device/old-identity tests for every device-affine family,
  proving terminal loss and no transparent generation recovery.
- API v1 freeze-checklist conformance, including descriptor canonicalization,
  declared-versus-actual Graph coverage, upload/readback ordering, partial
  backend acceptance, per-point completion, frame ownership, logical
  statistics, and tooling reconstruction events.
- Contract tests prove external completion dependencies succeed through GPU
  waits, already ordered domains, and proven collapse; only absence of all
  legal order routes is `Unsupported`.
- Frame tests cover builder `build()` failure and builder Drop no-submit
  abandonment, conditional direct-MSAA admission plus intermediate fallback,
  and separate acquire/present statistics.
- Tooling tests cover subscribe/drop linearization, callback self-drop and
  same-device reentry refusal, lazy description of pre-existing objects/work,
  and complete presentation-target/configuration object definitions.
- Canonical semantic snapshots proving that no event contains a native handle,
  pointer, descriptor index, GPU address, barrier bit, or Rust-layout encoding.
- Dependency audit proving the native execution route no longer uses the
  replaced HAL/types packages in any published feature combination.

### Exit artifact

A `v0.16.0` release and retained native-RHI evidence. The renderer and browser
adapters may still be dormant; `0.16` makes no five-backend claim.

## 5. Version 0.17 - all RHI platforms and ecosystem integration

### Goal

Put browser WebGPU and the GL family behind the same RHI semantic contract, then
close the RHI phase on every declared platform/profile before RenderGraph work
begins.

The implementation families are:

- DX12;
- Vulkan;
- Metal;
- browser WebGPU;
- GL family, with separately tested desktop GL 4.x, GLES 3.x, and WebGL2
  profiles.

GL profiles are one backend family with different facts, not three public RHIs.
Browser/mini-game objects stay backend-private; JS bridge adapts canvas/surface,
resize, foreground/background, loss, restore, and host scheduling only.

### Required implementation

- WebGPU device request, enabled facts, resource/binding/pipeline/command routes,
  completion, terminal device loss followed by a new identity/generation-domain
  request, canvas epochs, and async disposal.
- GL-family `api -> state -> compat` implementation of the common RHI contract.
  Its desired/applied/unknown state machine remains private. Default framebuffer
  maps to `FrameAttachment`, never a fake texture.
- Separate, fail-closed facts/refusals for desktop GL, GLES, and WebGL2. No
  compute/storage emulation on a profile that lacks them.
- Structured rejection of deferred compressed/planar formats on every backend;
  they are not predeclared as P0 capability vocabulary.
- One common capability lowering path and one shared contract suite, while
  retaining backend-specific discovery and diagnostics.
- Re-enable or minimally migrate `fluxel-renderer`, `fluxel-rendering-wasm`,
  examples, `fluxel-jsbridge`, and required `fluxel-host` consumers only to run
  RHI integration proof and remove obsolete parallel resource models. This gate
  adds no renderer, material, scene, canvas, or runtime feature.
- Verify the public API contains no `WebGpuSession`, `WebGpuAssetToken`, GL
  context lease, or equivalent platform-specific resource model.

### Required proof

- Re-run the `0.16` DX12/Vulkan/Metal suite unchanged on the final `0.17`
  source; prior evidence cannot prove the integrated implementation.
- Named Chrome WebGPU: resource/command/readback oracle, resize/zero-size,
  hidden/visible, terminal device loss and new identity/generation-domain
  request, stale-domain handle rejection, bounded completion, disposal, and no
  leaked pending work.
- Named Chrome WebGL2: common floor, structured unsupported compute/storage,
  default-framebuffer presentation, resize/visibility/context loss/restore.
- Desktop GL on a named real implementation, GLES on a named real device for a
  portable GLES claim, and their exact extension-route/refusal matrices. An
  emulator may be retained only as auxiliary evidence.
- Cross-repository browser package/build/integration tests pinned to the exact
  rendering SHA.
- Same semantic test vectors and independent output oracle across all capable
  backends; unsupported cells prove zero side effects.

### Exit artifact

A `v0.17.0` release whose evidence matrix states exactly which profile supports
each resource, command, format, presentation, and loss route. At this point—and
not before—the RHI phase is complete.

## 6. Version 0.18 - RenderGraph semantic core

### Entry gate

`0.17` is released and every RHI target above has retained passing evidence.
Graph development consumes only the frozen RHI facts, allocation-requirements
query, recording, submission, presentation, completion, and trace seams.

### Goal

Complete authoring, validation, logical compilation, and per-frame
instantiation without yet claiming aliasing/parallel/multi-lane optimization.

### Required implementation

- Mutable `RenderGraph`, typed logical buffer/texture versions, definedness,
  whole/partial/discard writes, and mip/layer/aspect or byte-range inheritance.
- Stable `PassKind::{Raster, Compute, Copy}` and full buffer/texture/attachment/
  copy/readback pass-local builder and resolver authority. `Host` remains
  unfrozen without a real consumer.
- Pass declarations, effects, scheduling hints, explicit non-resource
  dependencies, and rejection of undeclared/global resource access.
- Import contracts with `initial_use`, contents, ownership, usage, identity and
  lease; export contracts with `final_use` and retention. No native “state” API.
- Observable export/present/readback/external roots, reverse-reachability
  culling, deterministic cull reasons, and no `NeverCull` escape hatch.
- RAW/WAR/WAW and memory-dependency construction, subresource overlap, cycle
  detection, capability/route validation, and a serial logical schedule.
- Target-aware immutable `CompiledGraph`, capability/allocation profile
  fingerprints, cache invalidation, per-frame `GraphInstantiation`, and explicit
  `GraphExecutionPlan -> SubmissionPlan` lowering.
- Query the RHI allocation requirements but emit a conservative one-slot-per-
  resource/no-alias plan. Use only the guaranteed serial lane; pairwise
  multi-lane scheduling and alias realization remain disabled until `0.19`.
- Present root and exactly-once acquired-frame consumption.
- Deterministic `CompileReport`, plan visualization, structured authoring/
  compile/instantiate/record/lower errors, and `FrozenGraphIR` export hook.

### Required proof

- Table-driven CPU tests for every version/definedness transition, partial-write
  inheritance, overlapping/non-overlapping ranges, RAW/WAR/WAW, same-layout
  memory dependency, cycles, roots/culling, wrong-pass handles, undeclared use,
  imports/exports, stale generation, frame double consume, and side-effect-free
  preflight failure.
- Setup-to-handle-to-resolver-to-kind-restricted-recorder coverage; import slot
  to initial version; copy/readback ticket mapping; present-final-version
  selection; and missing/duplicate/unknown instantiation bindings.
- Compiled-template versus per-frame execution-plan separation, rejection on a
  foreign target device/presentation generation, and acquired-frame reducer
  behavior on preflight rejection, accepted work, `abandon`, loss, and present.
- Property/model tests comparing compiler dependency/liveness results with a
  small independent reference model.
- Deterministic reports/IR across repeated processes for identical inputs.
- `TestRhi` proves the exact plan/lowering protocol.
- On every `0.17` backend, at least one common raster/copy graph; compute and
  storage graphs run only where facts allow and fail before side effects
  elsewhere.

### Exit artifact

A `v0.18.0` release with the complete correctness model. Cross-lane execution,
same-frame aliasing, parallel recording, and durable diagnostic artifact remain
off, so they cannot hide a core compiler defect.

## 7. Version 0.19 - RenderGraph execution and diagnostics closure

### Entry gate

`0.18` is released with the semantic compiler and all graph validation green.

### Goal

Finish the scoped RenderGraph contract: target-aware allocation/reuse/alias,
pairwise lane routing and legal serial fallback, optional parallel lowering,
real-backend closure for the `0.18` completion-aware readback contract, and a
durable non-replay trace artifact.

### Required implementation

- Activate reuse/alias planning from the already-frozen RHI opaque requirements
  and implement `TransientAllocationService::realize`.
- Lifetime intervals, cross-frame compatible reuse, same-frame alias planning,
  explicit `AliasBoundary`, imported/exported/presentable exclusion, and a
  mandatory no-alias fallback.
- Pairwise lane-route scheduling (`Ordered`, `Gpu`, `Collapse`, or
  `Unsupported`), execution dependencies, and correct single-lane lowering.
  Host waiting is not an RHI lane route.
- Completion-safe allocation reuse/retirement through preflight rejection,
  accepted receipt with pending/failed terminal state, and device loss.
- Close the already-required `0.18` graph readback root/ticket mapping on real
  allocation, lane, completion, retirement, and trace paths. This adds no new
  readback authoring semantic. A general Host pass is added only if an accepted
  real consumer proves its execution model; otherwise it remains a documented
  candidate and does not block closure.
- Optional parallel recording, batching, raster-scope merge, and transition
  coalescing behind equivalence tests. No public backend or `NeverParallel`/
  `NeverMerge` knob.
- Versioned, non-replayable `GraphTraceArtifact` with typed IDs, capability and
  plan facts, lifetimes/alias statistics, validation events, checkpoints, and
  native-tool correlation. It is explicitly not `CaptureArtifact`.
- Implement and verify the API v1 tooling/Graph hooks already frozen in `0.16`;
  do not redefine their public semantics. No disk capture schema is frozen yet.

### Required proof

- Allocation requirements influence compatibility; alignment, class, dedicated
  allocation, overlap, first-use contents, and imported/exported exclusion each
  have positive and negative tests.
- No-alias and alias plans produce identical observations. Reuse occurs only
  after known completion; pending/unknown work prevents checkout.
- Every pairwise lane route preserves happens-before; lane collapse produces
  the same result on serial backends. No test claims hardware overlap.
- Parallel and merged lowerings, when enabled, compare against the serial oracle
  and preserve command/use coverage.
- Real-backend transient/reuse/readback workloads run across the `0.17` matrix;
  only supported lane routes are exercised.
- Trace round-trip/version tests prove deterministic diagnostics and explicitly
  prove that the artifact cannot be accepted by `ReplayRuntime`.

### Exit artifact

A `v0.19.0` release. The scoped RenderGraph is now complete and tested; later
blackboard, conditional/history, multi-device/XR/video/sparse nodes and
cost-model optimization remain separate evidence-gated features, not omissions.

## 8. Version 0.20 - portable capture and replay

### Entry gate

`0.19` is released. All reconstructability hooks, shader artifacts, canonical
objects/commands/submissions, graph IR, readback, completion, and external-input
classification already exist in normal production paths.

### Goal

Implement portable capture/replay without serializing `RecordedWork` or native
state. Freeze the first artifact schema only after end-to-end replay proves it.

### Required implementation

- `CaptureRequest`, all scope kinds, explicit begin/end, dependency closure,
  included producers or legal snapshot boundary, and observation selection.
- Capture-local monotonic typed IDs; canonical object table; immutable blob
  content; mutation and lifecycle events.
- `FrozenGraphIR` for provenance/validation and `PortableCommandIR` as normal
  replay truth. Cross-validation rejects commands not covered by graph uses.
- Resource initial snapshots, CPU uploads, external fixtures, checkpoints,
  undefined/unverifiable regions, texture block/row/image layouts, and
  completion-aware observation readback.
- Shader portability classes and recreation of interfaces, bindings, and
  pipelines without requiring backend binaries.
- Canonical batches, dependencies, acquire/present/retirement semantics and
  lane-collapse adaptation. Presentation mode is captured from configuration,
  and frames are reconstructed through the normal plan/present receipt model.
- External input policies and `ReplayProvider` boundary. No host-native object
  is consulted implicitly.
- `ReplayDecision::{Direct, Adapted, Unsupported}`, headless/windowed execution,
  validation levels, tolerance, query-kind policy, checkpoints, debugger/
  inspection mode, diff, first-failure provenance, and `ReplayReport`.
- `Complete`/`Partial`/`Failed` status, replayability, omissions, crash/device-
  loss finalization, version migration/refusal, redaction/privacy, manifest and
  blob integrity, optional signatures, and bounded untrusted parsing.
- Separate `ReplayCapturedCommands` and `RecompileComparison` modes.

### Required proof

- Requested pass/submission/frame ranges whose producers lie outside the range
  close by inclusion/snapshot or fail explicitly; no hidden warm-up dependency.
- `NextGraphExecution` and a multi-frame artifact containing multiple graph
  executions/submissions per frame round-trip through the same closure rules.
- Same-backend replay for every declared backend/profile and cross-backend
  replay wherever shader portability and capability negotiation permit.
- Direct and adapted cases (including lane collapse/pipeline recompile), plus
  feature/limit/format/shader/external-provider unsupported cases.
- Snapshot/upload/copy/clear/discard/undefined mutation cases and deterministic
  hash/tolerant image observations.
- Partial capture on device loss only when the retained prefix is dependency
  closed; truncated/missing/corrupt required data is never reported complete.
- Parser fuzz/corpus tests for bounds, overflow, recursion/count limits,
  decompression bombs, unknown required/optional sections, hash failures, and
  redacted artifacts.
- Required/optional opcodes, header major/minor/endianness/producer/features,
  append/finalize behavior, manifest/chunk hashes, deduplicated compressed blobs,
  lazy verified reads, and migration/refusal each have persistence tests.
- Every external-input policy and class is tested, including missing provider
  and proof that replay never reads a current host object implicitly.
- Memoryless/tile-local observation succeeds only through a legal readback route
  and otherwise returns side-effect-free unsupported.
- Each missing required snapshot, command tail, shader/pipeline description,
  external input, and integrity record is non-complete/non-replayable; missing
  optional backend diagnostics alone may be partial/replayable.
- Redacting diagnostics or unselected contents outside the replay closure may
  remain replayable. Redacting any closure snapshot, required fixture,
  shader/pipeline payload, command reconstruction data, or required observation
  makes the artifact non-complete/non-replayable, and replay rejects it before
  any driver submission.
- `Complete` and `Partial { ReplayableClosedPrefix }` admission is tested; the
  latter submits only the declared closed prefix and reports partial status.
  `Partial { DiagnosticOnly }`, `Partial { NotReplayable }`, and `Failed` each
  remain parseable but prove zero replay object creation and zero driver
  submission. Finalization also distinguishes a saved failed diagnostic artifact
  from `Err(CaptureError)` where no finalized artifact can be produced.
- Negotiation and replay use the same explicit live target device identity and
  generation. Target loss/replacement after an earlier decision requires
  re-negotiation and is rejected before object creation/submission.
- Capture instrumentation preserves semantic output and is never labeled a
  representative performance trace. Native attachment correlation by
  `CaptureId`/marker is tested, and a missing/corrupt attachment cannot affect
  portable replay truth.
- Replay reports assert source/target observations, capability differences,
  adaptations, first failing pass/work/command, and markers. Timestamp/GPU
  clock/completion timing/pipeline-statistics and each enabled query kind follow
  their explicit non-portable/result policy.
- Normal replay uses captured Command IR. A test with an intentionally changed
  compiler proves that recompile comparison can differ without changing normal
  replay output.
- Native RenderDoc/PIX/Xcode attachment correlation is optional and never used
  as replay truth.

### Exit artifact

A `v0.20.0` release containing the selected versioned artifact schema,
`CaptureSession`, `ReplayRuntime`, CLI/test fixtures, and retained compatible-
target evidence. The portable promise is reconstruction of verifiable Fluxel
semantics, not native barriers, memory placement, driver commands, or timing.

## 9. Resume gate after 0.20

Only after all five releases and their ecosystem evidence are complete may work
resume from the renderer/scene/canvas/runtime architecture. The first resume
change must:

1. pin `fluxel-rendering` `v0.20.0` and the exact compatible revisions of
   `fluxel-bases`, `fluxel-host`, and `fluxel-jsbridge`;
2. reactivate any remaining dormant higher-level code without public native or
   platform resource leakage;
3. run the full RHI, graph, capture/replay, renderer, browser, and host
   integration gates on the same source set;
4. choose the next scene/runtime vertical slice in the ecosystem roadmap; and
5. avoid reopening the foundation contract without a new consumer, ADR,
   migration plan, and all affected backend evidence.

## 10. Explicitly not scheduled in these five versions

The interface contract's deferred P1/P2 list is preserved but not implied by
the word “complete”. In particular, this train does not promise ray tracing,
bindless, work graphs, sparse resources, general mapping, public native sync,
multi-device, XR, a general Host pass, a general material/pipeline authoring
API, scene API, Canvas/UI, or application runtime. Such work resumes only after
the gate above and enters the roadmap through a concrete consumer.

## 11. Claude Code execution protocol

The implementation agent starts from the first incomplete gate and continues
strictly in version order. It must read `AGENTS.md`,
`crates/rhi/documents/design-rhi.md`, the affected ADR, this plan, and the affected
layer design before editing code. It may temporarily comment out or feature-
gate higher-level consumers while rebuilding RHI, but it may not delete them,
invent a compatibility API, or use old implementation shapes as authority.

For each gate the agent must:

1. name the version, normative sections, affected repositories, and proof plan;
2. implement only that gate and its required migration;
3. run the specified local contract/backend/ecosystem checks;
4. retain exact evidence and update the plan/roadmap status honestly;
5. continue only when every exit condition is satisfied;
6. stop and report a real blocker rather than weakening or bypassing a contract;
7. avoid new renderer/material/custom-pipeline feature work until `0.20` closes.

The bootstrap command from the `fluxel-rendering` repository is:

```powershell
claude --add-dir .. "Read AGENTS.md and documents/version-plan.md in full. Follow section 11 as the controlling workflow. Treat crates/rhi/documents/design-rhi.md, the relevant ADR, rustdoc, and contract tests as the RHI authority. Start at the first incomplete 0.16 gate, implement and verify gates one at a time, and continue through 0.20 only after each prior exit gate and affected Fluxel ecosystem integration gate truly passes. Do not implement from historical stage journals or legacy code when they conflict with the public API; do not omit, rename, or defer any P0 interface. Higher layers may be temporarily commented out or feature-gated during RHI replacement, but preserve their source and reactivate them at the owning integration gate. Record exact evidence and never claim completion from a single-library green test."
```
