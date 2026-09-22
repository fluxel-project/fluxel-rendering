# Fluxel portable capture and replay architecture

This document explains the planned portable capture/replay architecture. RHI
tooling, object, command, submission, and presentation semantics are defined by
[RHI API architecture](../crates/rhi/documents/design-rhi.md). Capture
artifact/runtime interfaces are owned by this document and the
[implementation plan](version-plan.md).

RHI tooling SPI v1 is part of the completed RHI baseline. This document does
not authorize a persistent capture file ABI until the capture/replay plan has
an end-to-end proof.

## Boundary

Runtime recording and persistent capture are different systems:

```text
Recorder -> RecordedWork
```

is process-local, device-generation-affine work used for one execution, while:

```text
CaptureRequest -> CaptureSession -> CaptureArtifact
CaptureArtifact -> ReplayRuntime -> ReplayReport
```

reconstructs portable semantics at another time or on another compatible
device. `RecordedWork` is never serialized.

Capture records no native command bytes, handles, pointers, descriptor indices,
GPU addresses, barrier bits, queue objects, fence values, OS handles, or Rust
memory layout.

## Layer ownership

| Layer | Capture responsibility | Replay responsibility |
| --- | --- | --- |
| Renderer | request range, observations, and external-input policy | choose mode and consume report |
| RenderGraph | emit `FrozenGraphIR` and declared-use mapping | validate provenance; recompile only in comparison mode |
| RHI tooling SPI | expose borrowed canonical objects, command-ordered actual uses, mutations, submission, present, completion, and terminal loss | no replay product responsibility |
| Backend | optional native-tool attachment and diagnostics | lower through its normal production path |
| Artifact layer | schema, typed IDs, manifest, blobs, integrity, privacy | bounded parsing and version handling |
| Replay runtime | dependency closure and finalization orchestration | rebuild through normal RHI API, execute, observe, and retire |
| Validator/debugger | markers, checkpoints, observation metadata | diff and first-failure provenance without changing valid replay |

Backend native-capture files can be correlated by `CaptureId` and markers, but
they are never portable replay truth. The ID is generated at arm time and stored
in the finalized artifact header; it is provenance/correlation only, not a
device identity, native handle, integrity hash, object ID, or replay key.

## Two IRs, one replay source of truth

An artifact contains both:

- `FrozenGraphIR`: why work was retained and ordered—pass declarations,
  logical resource versions/uses, dependencies, roots/culling, lanes,
  import/export, lifetime, and alias provenance;
- `PortableCommandIR`: the canonical operations that were actually accepted
  for execution, plus object definitions and submission semantics.

Commands must be covered by the corresponding declared graph uses. This is a
validation relation, not permission to regenerate one IR from the other.

Normal replay always replays `PortableCommandIR`. Graph compiler versions,
culling, scheduling, merging, and lowering evolve, so recompiling
`FrozenGraphIR` would not reliably reproduce the captured work. The optional
`RecompileComparison` mode exists specifically to expose those differences and
must label its result as comparison, not replay.

## Requested scope and replayable closure

A capture request may name the next graph execution, a frame range, a submission
range, or a pass range. That requested range is not automatically self-contained.

```text
requested scope
  -> traverse resource and explicit dependencies
  -> include required producer work where legal
     or snapshot the required boundary state
  -> produce a closed replay range
```

If captured work reads persistent content created or modified before the range,
capture supplies a canonical initial snapshot or fails explicitly. A pass range
cannot silently assume that earlier producers will run. Begin/end are explicit;
markers do not widen them.

## Object and identity model

Every artifact reference uses a capture-local typed ID. Buffer, texture, view,
sampler, shader, pipeline interface, bind group, pipeline, graph version, pass,
recorded work, submission, present, and readback IDs occupy distinct domains.
IDs are monotonic and never reused.

A graph logical resource/version is not a physical allocation. Alias placement
is retained for provenance and diagnostics, but replay may choose different
storage while preserving logical lifetime and contents. Source device identity
and generation are provenance, not replay keys.

Object definitions contain canonical descriptors, labels, dependency IDs, and
content references. RHI events distinguish object creation from backing
reclamation; Rust handle drop is not a destroy event. Descriptor/opcode encodings
are tagged and versioned rather than raw Rust enum discriminants.

Polymorphic object-table references use a closed tagged `CapturedObjectId` union
of buffer, texture, view, sampler, shader, pipeline-interface, bind-group, and
pipeline IDs. Pass, graph-resource-version, recorded-work, submission, present,
and readback IDs stay in their own typed domains. A bare integer can never cross
one of these domains.

## Content and mutation

Capture accounts for every mutation that can affect an observation:

- initial contents at the closed capture boundary;
- retained CPU upload;
- clear, copy, resolve, and GPU producer commands;
- undefined/discard operations;
- external input fixtures; and
- explicit checkpoints.

GPU-produced contents are normally recomputed rather than stored. Contents that
pre-date the range, uploads, fixtures, and checkpoints need blobs. Buffer
snapshots name byte range and layout. Texture snapshots name format, extent,
mip/layer/aspect, block layout, row pitch, and image pitch.

Undefined contents stay undefined. If an observation reads them, the artifact
and report mark that observation unverifiable; they never turn undefined bytes
into zeros. Memoryless/tile-local contents require a legal copy/readback route
or capture returns structured unsupported.

Snapshot/checkpoint insertion can perturb timing, so semantic capture does not
claim to be a representative performance trace.

## Shader and pipeline reconstruction

The shader artifact created by normal RHI operation already declares its replay
provenance and acceptance scope:

- portable source/IR with canonical producer identity and compile options can be
  regenerated for compatible backend families;
- executable-only artifacts carry an explicit backend-family or exact
  capability-contract acceptance scope.

Bind-group layouts, pipeline-interface fingerprints, render-target signatures,
specialization, and fixed state are canonical object data. A backend pipeline
binary can be optional diagnostic/cache data, never mandatory portable data.

This makes cross-backend replay a negotiated fact rather than a blanket promise.

## Submission and presentation

Artifact submission semantics include plan/work/lane IDs, ordered work,
`PlanPoint` dependencies, external `CompletionPoint` dependencies, receipt
mapping, acquire relations, present-plan/present-receipt relations, terminal
completion/present states, and retirement. They contain no native queue or
synchronization object.

Replay may collapse lanes where the target route proves it equivalent and must
record the adaptation. It may not remove a happens-before edge.

Surface images are not serialized as ordinary textures. Captured plan semantics
keep the frame/`PlanPoint` relation and independent present outcome.
Presentation configuration separately
records present mode and relevant surface facts. Headless replay replaces the
visible present with an output observation; a window is optional UI, not replay
semantics.

The artifact object table assigns capture-local definitions to both the
external `PresentationTarget` fixture and its `ConfiguredPresentation` lease,
so every target/configuration `ObjectId` in `FrameAcquired` is describable. No
native target is serialized. `ReplayProvider` maps the fixture to a compatible
live target, while headless replay may map it to a headless output sink.

## External inputs

Each external class—imported resource, video/XR image, external memory/sync,
host callback, dynamic asset, and network input—selects an explicit policy:

- reject capture;
- snapshot a supported input into a portable replacement;
- replace it with a named fixture; or
- require a replay provider.

A missing required provider is a replay failure. Replay never consults whatever
external object happens to exist on the target host. Closures and native/OS
handles cannot be artifact data.

## Negotiation and validation

Requirements cover protocol sections, features, limits, format/render/copy
routes, shader portability/ABI, and optional diagnostics. The target decision is:

- `Direct`;
- `Adapted { adaptations }`; or
- `Unsupported { reasons }`.

There is no `Exact` promise. A pipeline recompile or lane collapse can preserve
semantics without bit identity. Proven adaptations are reported. Missing
semantic capability, incompatible format/route, or insufficient shader artifact
rejects replay. Shader substitution, format approximation, precision reduction,
and sample-count reduction are never implicit.

Validation levels are command legality, deterministic hash, and
observation-specific tolerant diff. Timestamps, GPU clocks, completion duration,
and pipeline statistics are not portable expected values by default. Each query
kind declares its result policy. API v1 logical statistics and backend
diagnostics may be captured as optional evidence, but they are neither replay
commands nor bit-exact replay truth and cannot affect reconstruction legality.

## Replay transaction

Replay is fail-closed and ordered:

1. parse with size/count/recursion/decompression bounds and validate integrity;
2. inspect requirements and choose provider/device;
3. negotiate and produce the initial decision/report;
4. load/recompile accepted shader artifacts and create layouts/pipelines;
5. create resources, views, samplers, and bind groups;
6. restore snapshots and external fixtures;
7. cross-validate Graph IR, Command IR, objects, and declared uses;
8. reconstruct `RecordedWork` and `SubmissionPlan`;
9. execute headless or through optional presentation UI;
10. wait through normal completion and collect observations;
11. diff and report the first failing pass/work/command and related markers;
12. retire every object through normal completion-safe RHI behavior.

Failure at any step forbids submitting later unvalidated work.

Negotiation and execution are bound to the same explicit live `Device` identity
and generation. Replay revalidates negotiation for the device passed to the
execution call; an earlier displayed decision is not reusable authority. A
device/generation change requires a new decision and fails before object
creation or submission if stale.

## Partial capture

A capture is complete, partial, or failed. Missing a required snapshot, command
tail, shader/pipeline descriptor, external input, or integrity record can never
produce a complete replayable artifact. A dependency-closed prefix finalized
before device loss may be partial and replayable. Missing optional backend
diagnostics may also yield a replayable partial artifact.

`Complete` and `Partial { ReplayableClosedPrefix }` are executable; the latter
replays only its declared prefix and remains labeled partial in the report.
`Partial { DiagnosticOnly | NotReplayable }` and `Failed` remain parseable but
are refused before negotiation, object creation, or submission. `finalize`
returns these as `Ok(artifact)` when it can write a valid finalized diagnostic
artifact; `Err(CaptureError)` means no valid finalized artifact could be
produced.

The finalized manifest, not the mere presence of chunks, decides what is closed.

## Storage, security, and privacy

The intended storage properties are an append-only write path, finalized
manifest, content-addressed deduplicated blobs, optional compression, chunk and
manifest integrity, optional signature/envelope metadata, and lazy blob access.
The concrete chunk/schema technology is selected and frozen only after an
end-to-end capture/replay proof.

Artifact input is untrusted. The parser checks bounds and integer overflow,
limits object/chunk/recursion counts and decompression ratios, rejects unknown
required sections/opcodes, and reports skipped optional diagnostics. Canonical
IR is validated through normal RHI checks again before driver entry.

Capture supports resource filtering/redaction, source omission or IR-only
shaders, and label/path scrubbing. Credentials, absolute paths, native handles,
and host addresses are forbidden. Encryption and access control belong to the
storage/transport layer, not the RHI ABI.

Redaction is closure-aware. Removing optional diagnostics or unselected content
outside the replay closure may preserve replayability. Removing a required
snapshot/fixture, shader or pipeline reconstruction payload, command data, or
required observation input must fail finalization or produce a partial,
non-replayable artifact. It can never be marked complete or a replayable closed
prefix, and replay rejects an inconsistent manifest before driver submission.

## Completion criteria

The capture/replay architecture closes only when the gates in the
[implementation plan](version-plan.md) pass: dependency-closed ranges, mutation and
snapshot coverage, compatible same/cross-backend replay, direct/adapted/
unsupported negotiation, loss/partial handling, malicious-input parser tests,
and proof that normal replay is independent of current graph compiler output.
