# S4 CPU Closure Record

> Status: independently reviewed CPU closure. This record is deliberately
> limited to the S4 data contract and public provisional CPU
> facade. It is not an S5/Forward result, GPU evidence, a support declaration,
> or full-workspace regression.

## Product and evidence identity

The product revision is `e4ec252b65fafa72c412ec006c36f9337a8ea5c8`. The complete
retained attempt is:

```text
examples/external-renderer/evidence/s4-cpu-reference/
  e4ec252b65fafa72c412ec006c36f9337a8ea5c8/attempt-009/
```

`input.json` binds every tracked executable input to both a target Git blob and
a SHA-256 value, including the workspace `Cargo.lock`, the external consumer
`Cargo.toml`/`Cargo.lock`, the private-Graph negative fixture, Renderer scene
and feature sources, and their contract tests. It also records the copied
executing `harness.py` with SHA-256
`2fcbc0158d45c9e0a1a0504462c86328ddbfc3c9800e90f53d80d30bc4466c0a`.
`hashes.json` recursively hashes the logs and generated records. The external
consumer keeps its own lockfile; the private-Graph fixture has a separate
lockfile. These locks, rather than the current working tree, are the dependency
identity for this attempt.

The attempt ran from `examples/external-renderer` with Rust 1.87.0:

```powershell
cargo +1.87.0 test --locked
cargo +1.87.0 run --locked
```

`actual.json` records zero exit status for both commands. The test command
discovered and passed two tests: the public direct/ECS consumer contract and the
independent private-Graph compile-fail negative. The executable published scene
versions 1 and 2. The command logs are `test.stdout.log`, `test.stderr.log`,
`run.stdout.log`, and `run.stderr.log` in the attempt directory.

The runner isolates pre-existing raw records before execution, writes input,
independent expectation, actual result, and error record before evaluating its
oracle, and preserves those records on a deliberate failure path. The retained
attempt contains the fresh `raw-evidence/object-adapter/` and
`raw-evidence/ecs-adapter/` sets, their prior isolated copies, and `error.txt`
(`no error`). Earlier attempts remain audit history only; attempt-009 is the
current complete record.

## S4 data-contract assessment

The original S4 requirement remains authoritative: generational render and view
identities; dense render-only records; views/cameras; finite transforms/bounds;
logical geometry/material references; transactional typed changes; snapshots;
incremental updates; stale/remove/reuse/conflict/capacity/dirty-range refusal;
ordinary and ECS-shaped adapters; and independently typed Feature data without a
universal layout.

| Contract | CPU evidence and boundary |
| --- | --- |
| Scene identity, record/view updates, and snapshots | The external direct and ECS-shaped adapters independently apply the public batch API and compare their public observations. The attempt retains their source input, expected facts, and actual facts. |
| Dense removal and generational safety | The Renderer S4 contract tests at the bound product cover compact removal, moved stable IDs, stale-slot refusal, reuse, and dirty-slot effects. The external artifact exercises the public update/snapshot boundary but is not a benchmark or storage-layout claim. |
| Atomicity, capacity, invalid values, and dirty ranges | Renderer contract tests cover whole-batch refusal, foreign/conflicting changes, capacity/value refusal, snapshot versions, and precise dirty ranges. The external artifact retains an independently checked public update result. |
| Snapshot stability and incremental reuse | Renderer contract tests cover old-snapshot stability, concurrent CPU reads, changed-chunk sharing, and persistent-directory reuse. The external artifact retains version 1 while publishing version 2. |
| Asset identity and generation | The external consumer uses the pinned Bases asset store and S3 material assets. Its output confirms stable logical material identity with the replacement generation advancing from 1 to 2. This is logical CPU identity only, not GPU residency. |
| Adapter and layout boundary | Ordinary object-map and ECS-shaped column adapters produce the same public scene result. Their resource and feature payload layouts remain application-owned; Renderer does not impose an ECS, gameplay hierarchy, universal layout, or heterogeneous feature registry. |
| Typed feature boundary | The external consumer implements the static public `CpuFeature<R>` trait. `FeatureIssuer` creates opaque handles and controlled contexts; issuer-bound `TypedFeatureStorage<T>` retains typed snapshots; `CpuPhaseContributions` accepts only matching scene/issuer/feature/version/phase contributions and rejects replays. Version, scene, issuer, handle, storage, collector, and prepared-data failures are structured `FeatureError` values. |
| Privacy boundary | The external private-Graph fixture imports public `RenderScene` but its attempt to name `fluxel_renderer::graph` fails with E0603. The public CPU facade grants no Graph, RHI, backend object, submission, allocator, scheduler, or device authority. |

The in-tree contract tests named above are product inputs bound by
attempt-009. The two tests executed by that attempt are the external consumer
tests listed in `test.stdout.log`; this record does not represent that command
as a full workspace or GPU test run.

## Closure limits

This record supports only the exercised S4 CPU data and facade boundary. The
facade is provisional and has one CPU reference phase. It does not define a
public `RenderFeature` or `FramePipeline`, declare Graph resources or passes,
perform preparation, create a device, retain GPU resources, submit work, present,
or establish culling, indirect-work, backend, or performance behavior.

S5 consumes a closed S4 data boundary for Forward/GPU integration; it does not
move incomplete S4 work into a later stage. No native evidence is added here:
the existing DX12/WGL evidence remains limited to the earlier S0/S1/S2/S3
fixtures. Full regression remains deferred until the planned S-stage test
sequence is complete.

## Review condition

Independent review confirmed the attempt's target binding, 73 product blobs,
locks, command logs, two-test discovery count, raw records, stale isolation,
executing-harness identity, and all 26 recursive hashes. This is therefore the
closure reference for the stated S4 CPU boundary only.
