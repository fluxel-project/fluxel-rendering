# S0-S3 integration fixture

This independent consumer was rebuilt from the current workspace. Its
previous sources targeted removed APIs and are retired. No source is recovered
from repository history, and previous execution results are not current evidence.
The Copy/Raster/Compute/Surface inputs remain bounded fixtures, while direct S2
Shader and S3 Material workloads prove their current composition path. None of
these inputs establishes future Geometry, Feature, FramePipeline, or RenderScene
contracts.

Current native DX12/WGL attempts retain dynamic read-set execution, actual HWND
client captures, device replacement, S2 shader routes, and S3 material results.
The [foundation closure review](../../documents/s0-s1-closure.md) remains the
S0/S1 record. Each attempt archive identifies its exact source snapshot and
target observations; historical source snapshots do not certify later edits.

## Selected current inputs

Selection was audited against the current sibling source and clean checkout
metadata before adding the new dependency edges.

| Repository | Revision | Contract and role |
| --- | --- | --- |
| Rendering | Current source snapshot; runner records revision and dirty state | Renderer facade, private Graph, S2 Shader and S3 Material consumers |
| RHI | `0f7f785d4f0ccc62e125dd6cf560d91333ec85cf` | `fluxel-rhi =0.1.0`, defaults disabled; portable common execution, mapping, explicit copy transitions and readback |
| Bases | `c5fdaa993c13feb6b55b23fc766c22c9733836f2` | `fluxel-assets =0.13.4`, immutable snapshots and content replacement |
| Host | `33a6bbb3fa15500e928b4192ca8f280b735034d9` | `fluxel-host =0.1.0`; owns native windows and lifecycle events in this executable, not a Renderer dependency |

The first workload uses three immutable source snapshots under the same logical
asset identity. They carry asymmetric sentinel prefixes and suffixes around
16-, 24-, and 12-byte payloads containing respectively byte values 17, 18 and
19. The payload ranges are `[4, 20)`, `[8, 32)`, and `[4, 16)`, so every Copy
submission uses a nonzero source offset. All three uploads are submitted before
the first host completion wait. This workload is intended to verify that compatible Copy uses with
changed source descriptors and ranges refresh frame bindings while reusing one
device submission plan. Each frame's logical declaration is compiled and checked
against that cached topology; its candidate binding manifest and current
recording range remain frame-specific. The output records one device plan and
three binding refreshes, preserves full source, expected-payload, and readback
artifacts, and records source byte lengths, ranges, and output byte lengths.
Caller asset handles are dropped before waiting. The oracle compares each
readback exactly with its selected payload, excluding the sentinels. Empty input,
capacity saturation and foreign/stale submission handles are structured refusals.

Weak references to all three snapshot values prove they remain alive after
caller/store release and disappear only after their covering readbacks finish.
The output contains every input/readback byte, actual asset generations, RHI
counters, adapter identity, and raw submit/readback timings. Timings are diagnostic
samples, not a performance comparison or benchmark gate.

The same executable also runs fixed WGSL Raster and supported Compute fixtures.
Raster uploads three generations of `{ vec2<f32> position, vec4<f32> color }`
vertices and renders asymmetric red, green, and blue triangles to separate
128-by-128 RGBA8 targets. A dense CPU oracle checks interior and clear pixels,
excluding a declared two-pixel edge band. Raw RGBA and PPM files are saved before
the oracle, including on failure. This proves offscreen output, not a Surface or
window screenshot.

Compute transforms three 96-element storage-buffer generations with
`u32.wrapping_mul(3).wrapping_add(7)`, including overflow inputs and a partial
workgroup. It compares every output byte and retains the original values.
Unavailable compute is recorded before submission; Renderer also has a typed
unsupported result. Each fixture submits all three generations before waiting
and checks snapshot ownership through completion.
Each readback also proves per-frame retirement isolation: the completed
generation is released while every later generation remains retained until its
own readback. Raster records one plan compilation for all three submissions;
Compute records the same plan-reuse and per-frame retirement facts.

The Surface fixture uses the same fixed vertex WGSL and public Renderer path to
render and present three asset generations. It completion-drains each frame,
uses the Host window's native HWND to request a 320×180 → 256×144 resize and
restore, verifies both Host resize events and queried client extents, then
reconfigures/acquires the RHI Surface. The HWND remains alive through the
surface, device and all leases. This is a fixed fixture, not a general frame API.

Renderer owns the pending resources and original RHI outcomes. Its private Graph
compiles upload, raster, present, and readback steps on one ordered graphics lane, including the
COPY_WRITE to COPY_READ transition. Only a checkpoint covering the concrete
readback step permits mapping and retirement. Partial or uncertain effects retain
their original payloads and owners; a whole graph is never replayed implicitly.

## S2 Shader and S3 Material workloads

The S2 workload composes a direct Shader artifact, validates its logical
interface, maps reflected vertex requirements to the concrete fixture layout,
checks device capabilities and limits, and uses the exact `ShaderDesc` for both
route selection and native creation. Its device-local executable cache proves a
same-device hit and replacement-device isolation without treating a backend name
as a capability decision.

The S3 workload compiles several Material fragments and typed resources into a
generic Shader contribution. It executes both original and replacement material
generations through the existing read-set Raster path. Dynamic uniform values
share a static variant; a changed source generation changes the variant and
logical shader artifact. Binding preflight rejects incompatible reflection and
resource tokens before RHI work.

The runner retains source, independent expected values, raw readbacks, route and
capability evidence, diagnostics, and workload metadata before evaluating an
oracle. These runs demonstrate only their named DX12 and WGL configurations.

## Artifact and evidence boundary

This directory owns its own Cargo workspace and lock. The parent workspace
excludes it. `tools/integration/smoke.py` is the Rendering-owned runner, and
`documents/integration-manifest.schema.json` describes retained evidence under
`target/integration/s0-s3/<rendering-revision>/manifest.json`.

The first active target is Windows DX12 (`dx12`). Desktop OpenGL 4+ through WGL
(`native-gl-wgl`) follows with the same semantic oracle and Host-owned context.
The closure review records the exact exercised matrix and its evidence limits.
Vulkan, WebGPU and WebGL2 remain deferred under the reviewed support decision.

Run from the repository root with the retained-evidence runner:

```powershell
python tools/integration/smoke.py --backend dx12
python tools/integration/smoke.py --backend native-gl-wgl
```

For a direct development run, build this manifest with exactly one backend
feature and pass `--output <directory>` to the binary. The old retirement marker
has been removed. S0/S1 acceptance also
requires Raster, supported Compute, Graph semantics, failure/loss behavior,
inspected real output and the complete target matrix; buffer readback alone
cannot close the foundation work.

## Retained evidence from earlier source snapshots

These observations predate the reviewed execution/range/shutdown batch. They
do not prove its nonzero-offset path, shared execution integration or shutdown
behavior on native GPUs; focused logic/build results are recorded separately.

Copy attempts on 2026-10-30 passed on AMD Radeon 780M (Windows driver
`32.0.21028.2002`) through DX12 and WGL OpenGL 4.6 Core. Both verified all bytes,
snapshot ownership, one compiled plan, three outstanding submissions, and the
structured refusal cases. The retained index records a dirty current source
snapshot and individual source hashes; it is not a frozen release revision.
Native validation activation remains unverified. Subsequent three-workload
attempts `dx12-20261030T072817221585Z-21108` and
`native-gl-wgl-20261030T072827433722Z-5148` also passed Raster and Compute. Both
routes produced byte-identical Raster RGBA, and all 288 Compute values per route
matched the independent CPU oracle. Offscreen visual review is retained under
`reviews/raster-20261030/`. Surface/present and actual HWND resize have since
passed both native routes; visual inspection and the remaining failure/loss
lifecycle gates were open at that historical point.

Earlier Rust 1.87 retained attempts include
`dx12-20261030T081646487800Z-23880` and
`native-gl-wgl-20261030T081653574811Z-26212`. Both exited successfully with
stable source snapshots and verified Cargo source identities. Raster and
Compute workloads each retain three outstanding submissions, verify per-frame
snapshot retirement, and report one compiled-plan reuse. Each surface workload
records three generations, Host resize/restore events, client extent checks,
and completion before reconfiguration. These are partial route facts, not a
closure claim. Later offset-zero dynamic-length attempts
`dx12-20261030T084557779060Z-23108` and
`native-gl-wgl-20261030T084610799916Z-9016` verified 16/24/12-byte Copy outputs,
one device plan and three binding refreshes for their source snapshots. They
still predate the current nonzero-offset and common execution edits.
See the attempt index under
`target/integration/s0-s1/fccbeb9578652806c5e139fa6133a4ac2f86e91f/manifest.json`.

## Dynamic uniform and texture read-set fixture

The `read-set/` workload uses `Renderer::submit_read_set_raster` with the fixed
`assets/read-set.wgsl`. It submits three immutable generations before waiting,
changes uniform destination windows (`0..256`, `256..512`, `512..768`) and texture
contents, and requires one device plan with fresh frame bindings. Its independent
expected images use distinct non-white texels and tint masks, so omitting either
shader input fails the dense pixel oracle. Full 20-byte inputs, expected RGBA,
readbacks and structured preflight refusals are retained before the oracle.
The same Graph engine and completion-owned slots serve this path and the other
fixtures; no preparation-only alternative cache supplies its proof.

Each native route also creates a replacement device. It rejects the old pipeline
and opaque submission before backend work, then reuses the logical snapshot with
a replacement pipeline and verifies its image. WGL uses an independent Host
window and context, with separate diagnostic activation and message capture.
Surface frames retain client BMP captures made from their live HWND with GDI;
these are distinct from GPU readbacks and require inspection before a visual
claim. The source and lock inputs are archived per attempt. Implementation and
compilation alone do not close S0/S1 or certify these new native paths.
