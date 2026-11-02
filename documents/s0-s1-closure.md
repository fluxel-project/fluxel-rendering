# S0/S1 foundation acceptance record

S0/S1 foundation acceptance is complete for the named DX12 and WGL matrix using
the repaired RHI revision below. The [capability audit](capability-audit.md)
records the source inventory, Core and Metal fixes, and hardware coverage limits.
Earlier native observations remain historical and do not certify the new revision.

The foundation implementation and its evidence are bounded to the named Windows
DX12 and desktop OpenGL/WGL matrix below. The machine-readable closure review is
the acceptance record; prose, a green library test, or a historical screenshot
cannot establish a pass by itself. S2 Shader, S3 Material and later public
Feature/FramePipeline contracts remain unimplemented.

## Inputs and retained artifacts

The Rendering-owned executable is `examples/rendering-integration`, with its own
workspace and lock. It consumes the public Renderer/RHI/Assets interfaces; Host
exists only in this independent consumer. The exact selected dependency commits
remain:

| Dependency | Commit |
| --- | --- |
| RHI | `0f7f785d4f0ccc62e125dd6cf560d91333ec85cf` |
| Bases | `c5fdaa993c13feb6b55b23fc766c22c9733836f2` |
| Host | `33a6bbb3fa15500e928b4192ca8f280b735034d9` |

Each selected attempt retains its exact Rendering worktree input archive and
manifest, both lock hashes, actual Cargo package identities, clean Git checkout
revisions, executable bytes and before/after hashes, commands, diagnostics and
raw workload artifacts. The Rendering HEAD is recorded as dirty; the source
archive and per-file hashes identify the implementation, not HEAD alone. This
is a local foundation acceptance record, not a published release or a clean Git
tag. Neither local instructions nor planning drafts are staged by this work.

The evidence index is
`target/integration/s0-s1/fccbeb9578652806c5e139fa6133a4ac2f86e91f/manifest.json`.
The explicit selected attempt IDs and review links are in
`target/reviews/20261102-foundation/closure-review.json` and
`target/reviews/20261102-foundation/acceptance.json`. Failed and superseded
attempts remain in the index and cannot certify newer source.

## Named real targets

| Route | Observed environment | Native workload |
| --- | --- | --- |
| DX12 | Windows 11 build 26300, AMD Radeon 780M Graphics, driver `32.0.21028.2002`, Rust 1.87 MSVC | Copy, Raster, Compute, dynamic uniform/texture read-set, Surface resize/restore, replacement device |
| Desktop WGL | Same OS/adapter/driver; OpenGL `4.6.0 Core Profile Context 25.10.28.02.250904` | Same workload, with independent original/replacement WGL contexts |

Every route retains raw inputs, independently retained expected output and full
readbacks before its oracle. Copy, Raster, Compute and read-set each retain
three generations before the first completion wait and check per-frame snapshot
retirement. The read-set shader consumes both a changing uniform window and
non-white sampled texel; omitting either input fails its dense pixel oracle.
Old pipeline and opaque submission handles are refused by the replacement
device before backend work, while logical snapshot bytes are rendered using a
new-device pipeline and checked again.

DX12 validation evidence combines a successful device-open marker with the
exact pinned fail-fast debug-layer activation branch, verified compiled source
identity and callback stderr. WGL uses scoped native interop on both actual
contexts, enables debug output, and observes an exact application marker.
Application push/pop notifications are excluded at admission so they cannot fill
the driver's bounded queue; API/driver errors and warnings remain enabled. A
full, unknown-capacity or non-progressing log fails validation. All other admitted
messages are retained. Pinned glow Debug metadata is decoded strictly because
the selected type has no public field accessors; unknown formats fail closed.

## Acceptance mapping

| Requirement | Current evidence |
| --- | --- |
| Pinned ecosystem composition and source provenance | Independent manifest/lock, actual Cargo Git checkout checks, frozen source archives and retained native executables |
| Resource versions, definedness, RAW/WAR/WAW, deterministic DAG, culling and lifetimes | Current Renderer compiler/access/state tests and native multi-pass fixtures using the same core |
| Definition/device-plan/frame-binding separation | Actual `ReadSetRasterFrame` candidates describe physical ranges; compatible candidates reuse `ReadSetRasterPlan`; source, target and descriptors remain per-frame |
| Compatible instance/range changes and incompatible contracts | Native plan reuse across three owners; actual-plan preflight tests refuse descriptor, state/import, alias and device mismatches; topology/access/dependency/layout tests establish refusal boundaries |
| Completion-owned storage and retirement | Native three-owner/per-frame release checks, exact covering-checkpoint selection, Witness completion markers and cancellation/loss tests |
| Clean rejection, partial acceptance, partial immediate issue and indeterminate effect | Deterministic execution-backend tests retain original outcomes and actual owner witnesses, finish after recording errors and release only on explicit no-effect or covering completion/idle evidence |
| Device loss and replacement | Injected loss at allocation/map/flush/begin/record/finish/wait/readback/idle blocks old work and preserves uncompleted owners; both native routes create distinct replacement devices and verify provenance refusal plus snapshot reuse |
| Real Surface and resize output | Actual Host resize events/client extents, completion-before-reconfigure checks and three native HWND client captures per route |
| Output, diagnostics and evidence review | Recursive raw-oracle/hash review, explicit visual/test/fault/read-set review files, saturation-aware diagnostics and schema validation |

With the revised dependency, the affected Renderer, consumer and tool suites
passed 86, four and 30 test functions respectively. The RHI Core passed 129
tests, two GL feature-route tests passed, and Metal cross-compiled for an Apple
target. Logs are in `target/reviews/20261102-capabilities/`. Earlier workspace
Clippy and rustdoc results are retained as historical evidence, not attributed to
the revised dependency. Shader and Material skeletons intentionally discover
zero tests; that is not coverage for those future stages.

Native client captures are distinct from RHI readbacks. The consumer uses
per-monitor-aware pixel coordinates, a temporary raise of its own window without
focus transfer, compositor synchronization and desktop client cropping. The
fixture requests square DWM corners. The visual contract allows no corner mask;
only the frozen two-pixel triangle-edge band is excluded from dense RGB checks.
All six actual images are inspected in addition to that dense check. This proves
three fixed completed presentations and resize/restore, not continuous animation
or an unrestricted flicker claim.

## Limits retained at closure

- Partial/indeterminate execution and DeviceLost evidence uses deterministic
  fault injection. Physical GPU removal, OS driver reset and recovery of a lost
  device are not claimed. A failed wait never becomes retirement evidence.
- The selected RHI CompletionPoint exposes device provenance but no portable
  completion identity. Source-level exact checkpoint selection, distinct Witness
  markers and native per-frame isolation provide the stated completion proof;
  no native completion-ID claim or new public generation field is invented.
- GPU inputs are host-uploaded or graph-produced in these fixtures. No external
  GPU-producer completion/import API, general Material/Shader authoring, physical
  alias allocation optimization or multi-queue rendering is claimed.
- Unknown native effects may be disposed through successful idle shutdown. A
  lost/uncompleted owner remains quarantined; no generic recovery framework or
  replay of an accepted prefix is introduced.
- Vulkan, browser WebGPU and browser WebGL2 remain deferred with the owners and
  restoration gates in the support decision. The named matrix does not certify
  untested adapters, drivers, operating systems or later extension APIs.
- Raw timings and counters are diagnostic baseline observations, not a speedup,
  GPU-time measurement or cross-driver performance claim.
