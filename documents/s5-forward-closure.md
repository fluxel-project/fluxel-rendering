# S5 Forward closure record

## Status

S5 is closed for the bounded Forward candidate on Windows DX12 and desktop
OpenGL/WGL. The six selected native attempts pass all 12 independent reviewer
checks. Both routes also have five-sample release baselines with stable source
inputs. The requirement mapping below, retained focused checks, raw native
artifacts and measurements support this decision. S6 same-frame external
Feature composition and product MVP acceptance remain open.

The exact evidence source is the portable archive
`documents/evidence/s5-forward-candidate/native-evidence.zip`
(`sha256: b1f009b2ca236749f9c37f383a5d67baf9914e3eff92a656c4d813b89a8e2d92`).
Its `index.json` records six selected attempts and states that all six bind the
same exact dirty-worktree input list in their `source-inputs.zip` archives.
`36dcbbb1da0fed7ef43b3f8cc842bf1f4900c210` is only the Git baseline locator,
not the identity of the tested S5 source.

## Supported evidence scope

The six native attempts are:

| Backend | Forward | Device isolation | Surface |
| --- | --- | --- | --- |
| Windows DX12 | `dx12-20261104T114305300070Z-16472` | `dx12-20261104T114443846572Z-4176` | `dx12-20261104T114551726433Z-15564` |
| Windows desktop OpenGL/WGL | `native-gl-wgl-20261104T114652422233Z-19916` | `native-gl-wgl-20261104T114758621513Z-5244` | `native-gl-wgl-20261104T114907805020Z-3480` |

For both routes, the review validates the Forward raw artifacts/oracle/cache,
device isolation, and surface capture review.  The Forward workload retains
frozen geometry, material, texture, frame-input, expected-image and readback
artifacts; its oracle/cache/measurement facts are checked from the nested
workload rather than inferred from a process exit.  The surface runs retain
two resized client captures and their capture/verdict sidecars.  Readback and
window capture are separate artifact classes.

`visual-review.json` records an assistant's `view_image` inspection of the
selected DX12 and WGL captures.  Its legacy `human_verdict` field must not be
read as human approval: the file explicitly identifies the reviewer as
`Codex visual inspection via view_image` and `reviewer_kind: "assistant"`.

The archive and manifests bind the dependency inputs actually used by the
attempts: Fluxel RHI `b5d569feace548a6ed498901deaf31a5efcaca61`, Fluxel Bases
`ee640d57a8d7c32d251252d0e8b32950636a770c`, and Fluxel Host
`33a6bbb3fa15500e928b4192ca8f280b735034d9` for the rendering-integration
route.  The root, rendering-integration, and external-renderer locks pin the
RHI/Bases revisions; the integration lock also pins Host.  The root lock hash
recorded by the native manifest is
`c407323ad1de8c767fe70aff1490eb07ac46f8c102d36f2604310df6e482dfa8`.

## Requirement-to-evidence mapping

This mapping groups the S5 acceptance in [the implementation plan](version-plan.md)
into six implementation work items. Source
entries below are the implementation and focused-test audit; native entries
are the six archived attempts above.  A fault-injection test is not presented
as real-GPU evidence.

| Requirement | Concrete implementation / focused evidence | Native evidence and supported conclusion |
| --- | --- | --- |
| F1: bounded public frame construction; private Graph and submission; contribution validation and compatible-plan reuse | `crates/renderer/src/pipeline/mod.rs` exports `FramePipeline`, `ForwardPhaseContext`, opaque and clear-only authorities. `examples/external-renderer/src/lib.rs` implements two statically dispatched pipelines; its contract tests include private-Graph and private-phase-context compile failures. `pipeline/contract.rs`, `renderer/forward/contribution.rs`, and `renderer/forward/surface_contribution.rs` cover invalid/missing/duplicate/wrong contribution cases, including cached paths. | Offscreen Forward and Surface paths execute the selected controlled Forward contributions on both backend routes. This supports the bounded Forward seam exercised here; it does not stabilize a general external GPU extension API. |
| F2: runtime geometry/material/shader preparation, three layouts, rejection of invalid inputs, and generation-sensitive replacement | `preparation/{geometry,material,forward,forward_validation}.rs` prepares the first Forward products; `tests/preparation/{geometry,forward_batches}.rs` and the external material-preparation contract exercise interleaved, separate-stream, mixed-stream, Position-only/custom-attribute cases and validation. | Both Forward attempts retain four layout/material/texture input groups, frame outputs, old/new replacement inputs, independent expected pixels and readbacks. The claim is the tested first Forward templates and fixture, not arbitrary material or shader support. |
| F3: CPU visibility, deterministic opaque ordering, compatibility batches, and two concrete providers | `pipeline/mod.rs` supplies `ForwardCpuPipeline<V>`/`BoundsVisibility`; external consumer code composes built-in and interval visibility providers and independently checks candidates/order/batches. `tests/preparation/forward_batches.rs` covers compatible batching. | Forward cache/measurement facts retain candidate, visible/batch/draw facts for the selected fixture. Evidence supports only opaque Forward's tested order and providers; it does not claim transparent global composition, indirect work, or a general GPU-product API. |
| F4: persistent per-device residency, completion-gated frame reuse and owner-based refusal | `residency/{forward,forward_cache}.rs` key device-local products by asset generation and `DeviceId`; `frame_data/{forward,pool}.rs` implements frame allocation/reuse. `tests/residency/forward.rs`, `tests/frame_data/forward.rs`, and `tests/renderer/forward/{pending,lifecycle,cache}.rs` cover reuse, pending owners, capacity and foreign/descriptor cases. | Forward replacement/cache facts and the two device-isolation workloads verify the selected native routes. Failure/loss and completion-edge tests are fault injection: they demonstrate the local ownership policy, not a native device-loss event. |
| F5: concrete access/binding checks, plan reuse or rejection, and pending/retirement lifecycle | `graph/definition/` and `graph/{forward,raster,read_set_raster}.rs` and `renderer/{admission,validation,shutdown,readback}.rs` retain the Graph path. `tests/graph/{binding,compiler,forward_offscreen_plan,forward_surface,forward_surface_plan}.rs` and Forward lifecycle tests cover aliases, ranges, dependencies, cache reuse/recompile and injected submission/wait/readback failures. | The native Forward review checks raw/oracle/cache facts; device-isolation attempts check device-keyed cache separation. Injected partial-failure, wait-failure, DeviceLost and readback cases remain explicitly non-native evidence. |
| F6: real consumer, retained inputs/outputs, independent oracle, archive and review | `examples/rendering-integration` supplies `--s5-only`; `tools/integration/smoke.py`, `s5_review.py` and the archived manifests enumerate nested artifacts and attempt IDs. `documents/s5-measurement-baseline.md` defines a reproducible baseline protocol and its limits. | The archive contains the six selected DX12/WGL attempts, review command, selected-attempt record and hashes. It establishes only the two Windows native routes actually run, with assistant—not human—visual inspection. |

## Explicit limits and remaining work

- No general backend/platform support follows from these artifacts. Vulkan,
  WebGPU, WebGL2, Metal/Apple hardware and browser routes are not supported by
  this record.
- The selected six attempts form the complete selected S5 native matrix. They
  do not rerun every historical workload or establish full workspace regression.
- The retained measurements establish the first workload baseline; they do not
  demonstrate a performance improvement or measure GPU timestamp duration.
- S6 remains pending. In particular, this does not prove one runtime frame
  combining an external `FramePipeline`, typed external Feature, custom
  material contribution, built-in mesh path and a distinct provider under the
  final global composition rule.

## Measurements and source reconciliation

[The measurement record](s5-measurement-baseline.md#retained-first-baseline)
reports all five-sample distributions, executable sizes and actual allocation,
upload, cache and pending facts. [The baseline archive index](evidence/s5-forward-candidate/baseline-index.json)
retains commands, raw logs and outputs, exact source inputs and executed binaries.
It is separate from correctness/diagnostic evidence and makes no speedup claim.

[Source reconciliation](evidence/s5-forward-candidate/source-reconciliation.json)
proves that the six native attempts share 380 identical inputs, and both
baselines use those same inputs plus the two benchmark Python files. After
capture, the three explicitly listed README files changed to update status and
procedures, and extra empty lines at EOF were removed from the listed Python
files. Python executable statements are unchanged; Rust, WGSL, manifests and
locks remain byte-identical to the measured/tested worktree. Raw archives are
preserved; documentation follow-up is not presented as a new GPU run.

[Focused-check logs](evidence/s5-forward-candidate/focused-checks/index.json)
retain the affected Renderer, Material/Shader, independent consumer, integration,
Clippy and Python-tool results. The Renderer batch passed 160 tests; a later
Forward-only check after test-helper relocation passed 40. The external consumer
passed 6 tests, DX12 integration 28, and WGL integration 31. The Python batch
ran 72 tests, with 71 passed and one platform skip. These are scoped checks,
not a full workspace regression. A failed runner bootstrap is retained separately
and is not counted as native execution.

The runner's root ledgers remain raw capture records; their provisional root
summary fields are not rewritten into a closure verdict. The independent
`review/review.json`, per-attempt raw artifacts, source reconciliation and this
explicit bounded decision are the review record. Exact archive bytes and Git
index correspondence are checked before publication.
