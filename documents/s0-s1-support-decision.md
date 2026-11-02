# S0/S1 Foundation Closure Support Decision

## Decision

The rebuilt Renderer foundation closure is limited to these named native routes.
Copy, offscreen Raster, supported Compute, dynamic uniform/texture read-set,
Renderer-owned Surface presentation/resize, and replacement-device consumers
are implemented. Fresh native attempts retain their exact source archives,
executables, locks, diagnostics and client captures. The acceptance mapping and
named hardware limits are recorded in the [foundation closure review](s0-s1-closure.md).

Native normal, preflight-refusal, replacement and pixel evidence is distinct
from deterministic fault-backend evidence for partial/indeterminate execution,
device loss and cancellation. Physical GPU removal is not claimed. Static
Feature/Pipeline composition remains a later architecture constraint, not an
additional S0/S1 public API or an expansion of this target matrix.

| Route | Closure status | Required evidence |
| --- | --- | --- |
| Windows DX12 | Target scope | A rebuilt integration artifact must execute its asset replacement, Copy, Raster, supported Compute, three-frame completion/retirement, failure, surface, readback, and visual oracle matrix on a named DX12 adapter. |
| Desktop OpenGL 4+ through WGL | Target scope | The rebuilt artifact must execute the same logical cases on a named WGL adapter/profile. Compute is executed only when the observed capability set supports it; otherwise it is structurally refused before backend work. |
| Vulkan | Deferred | No migrated Renderer support claim is made for this closure. |
| Browser WebGPU | Deferred | No migrated Renderer or browser-consumer support claim is made for this closure. |
| Browser WebGL2 | Deferred | No migrated Renderer or browser-consumer support claim is made for this closure. |

This decision adjusts the S0/S1 closure matrix only. It does not reduce the
resource-validity, structured-refusal, completion, failure-state, lockfile,
real-consumer, or evidence requirements imposed by the organization roadmap and
evidence policy. S0 and S1 still close together. This decision does not close
either stage, and the reset does not authorize S2 before a rebuilt in-scope
matrix is complete.

## Affected consumer inventory

The [reset inventory](foundation-migration.md) records the current module and
consumer state. Removed tests and commands are explicitly excluded from current
test counts.

| Consumer or boundary | Current state | S0/S1 closure treatment | Owner |
| --- | --- | --- | --- |
| `fluxel-rendergraph` | Package and directory removed | The rebuilt buffer/texture definition/compiler and Copy/Raster/Compute lowering are private Renderer modules. Fixed consumers share one Graph compiler/access/state/execution core; no public scheduler or general frame facade is exposed. | `fluxel-rendering` |
| `fluxel-renderer` | Initial fixture facade, bounded pending owners, and private Graph | Three-frame resource changes, exact completion, owner retention, read-set validation and native Surface/device-replacement cases are mapped separately to logical and native evidence in the closure review. | `fluxel-rendering` |
| Renderer examples/tests | Public directories remain placeholders; private logic tests are implemented | They prove only the exercised private subset. | `fluxel-rendering` |
| Renderer features | `dx12` and `native-gl-wgl` explicitly forward to the same RHI feature names; default features are empty; no `gpu` feature | Prove each route independently. Features select build inputs and do not certify support. | `fluxel-rendering` |
| Former `crates/rendering-wasm` | Historical browser capsule removed | Direct legacy WASM builds are unavailable. Browser restoration requires a new producer through unified RHI, without public session/token resources. | `fluxel-rendering`, with lifecycle integration owned by `fluxel-jsbridge` |
| `fluxel-jsbridge` browser and mini-game adapters | Existing lifecycle adapter boundary | Not a S0/S1 runnable consumer. It continues to own canvas/surface lifecycle, resize, visibility, loss/restore, and diagnostics when browser work resumes. | `fluxel-jsbridge` |
| `fluxel-host` | Native platform lifecycle and window boundary | Supplies the window/display lifetime to the native integration executable; Renderer does not depend on Host as a library. | `fluxel-host` / integration executable owner |
| `examples/rendering-integration` | Newly implemented Copy/Raster/Compute/Surface consumer with independent manifest and lock | Native S0/S1 artifact owner, with per-attempt source/executable archives and actual client captures; historical source snapshots do not certify current code. | `fluxel-rendering` |

## Deferred-route restoration requirements

| Route | Responsible owner | Required restoration evidence |
| --- | --- | --- |
| Vulkan | `fluxel-rendering`, consuming `fluxel-rhi` | Pin the Renderer/RHI revisions and locks; run the same fixture on a named Vulkan adapter with Copy, Raster, supported Compute, overlapping submissions, completion-safe retirement, failure/loss behavior, surface output, validation, readback, and visual evidence. |
| Browser WebGPU | `fluxel-rendering` and `fluxel-jsbridge` | Migrate the WASM producer and browser consumer to the selected RHI without public session/token resources. Pin Rust and JavaScript/package or bundle inputs and output hashes. Prove a named browser/version/GPU route for resize, visibility, loss/recovery, Graph completion, visual output, and browser-specific diagnostics. |
| Browser WebGL2 | `fluxel-rendering` and `fluxel-jsbridge` | Use the unified RHI backend with the JSBridge lifecycle boundary. Prove a named browser/version/GPU route for Copy/Raster, three in-flight frames and completion-safe retirement, context loss/restore, resize/visibility, and visual/readback evidence. Requests for unavailable features, including Compute, must fail structurally before backend work. |

Each restoration is a separate compatibility task created after a future S0/S1
closure. It must retain the applicable manifest, exact revisions, lock hashes,
commands, capability snapshots, raw output, and target evidence. Shared semantic
changes require a rerun of the affected DX12 and WGL oracles before expanding a
support claim.

## Scope review

This target matrix remains aligned with the organization roadmap and evidence
policy. A read-only scope audit on 2026-10-30 verified the named matrix,
deferred-route owners and restoration gates against the organization roadmap.
This is not an implementation, lifecycle, or stage-close approval.

The active Rendering-owned runner is `tools/integration/smoke.py`. The independent
artifact's `dx12` feature enables Renderer/RHI DX12 and Host; `native-gl-wgl`
enables Renderer/RHI WGL and Host. Both consumers create a Host-owned window,
render through Renderer to a Surface, and resize the HWND after completion.
Renderer has no Host dependency. The current workloads are Copy, offscreen and
Surface Raster, conditional Compute, dynamic read-set Raster and device replacement.
The runner retains source/executable hashes and diagnostics; the closure review
links actual validation activation, logical fault evidence and visual inspection.

## Closure record requirements

The eventual `target/integration/s0-s1/<rendering-sha>/manifest.json` must link
this decision and record the exact consumer matrix, participating repository
revisions, dirty states, lock hashes, feature selections, commands, target facts,
capability snapshots, visual/readback artifacts, diagnostics, focused test
results, and the deferred routes above. Local tests, a host-target compilation of
the WASM capsule, or old screenshots do not substitute for this evidence.
