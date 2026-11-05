# S8 measured validation strategies and final native closure

The selected result is bounded to Windows DX12 and desktop OpenGL/WGL over the
retained material-heavy Forward inputs. Independent archive review verifies the
selected native matrix, raw pixels, measured comparisons, ownership evidence,
exact runtime sources and CPU regression logs after relocation. These bounded
results close the selected S8 scope.

## Selected implementation

`TextureBindingManifest::validate` uses private requirement/binding and actual
instance indices instead of repeated linear searches. It preserves cardinality,
input error ordering, first conflicting alias selection, and actual descriptor,
device provenance, access and incoming-state validation. `PassAccess` groups
immutable declarations by slot and exact contract and normalizes buffer ranges
for lookup. It caches no actual frame binding or owner fact. Cache hits still
consume this frame's declarations and validate concrete owners before the
protected operation. Graph remains the execution authority.

These changes select CPU/API validation strategies. They do not add a residency
budget, streaming/LOD policy, GPU culling, indirect draws, memory aliasing,
multi-queue execution, or parallel recording. CPU assets, completed GPU
residency, per-submission frame storage and Graph transients remain distinct.

## Retained performance result

Each backend has seven unique alternating pairs for each primary scenario and
seven pairs of the small 100-draw/10-texture guard. Primary inputs contain
10,000 visible draws, 1,000 real texture assets and four material variants at
512x512. Cold, warm and ten-percent generation replacement have identical
baseline/candidate inputs, independently checked raw pixels and matching
allocation, upload, draw, batch and plan-counter contracts.

| Backend | Scenario | Baseline median (ms) | Candidate median (ms) | Improvement |
| --- | --- | ---: | ---: | ---: |
| DX12 | Cold | 7347.4724 | 736.5368 | 89.976% |
| DX12 | Warm | 8149.7976 | 684.7604 | 91.598% |
| DX12 | 10% replacement | 7730.5796 | 605.6272 | 92.166% |
| WGL | Cold | 7366.8673 | 610.4103 | 91.714% |
| WGL | Warm | 8306.8045 | 682.8619 | 91.779% |
| WGL | 10% replacement | 7701.1725 | 583.0218 | 92.429% |

Every primary scenario exceeds the predeclared 10% median improvement threshold.
All small-scene guards pass the allowed `max(0.5 ms, 10%)` regression boundary.
The measured interval is preparation/submit API elapsed time and ends before
readback. It is not a GPU timestamp, CPU-cycle measurement, frame-rate claim or
universal renderer speed-up. GPU duration and heap profiling are unobserved.
Retained logical allocation/upload counts do not claim driver heap usage.

## Ownership and failure evidence

Separate, non-timed native cases use the full 10k/1k Forward input on two live
devices. They retain the old submission while accepting a replacement, reject
a foreign submission without retiring the old owner, check the pending sequence
`[1, 2, 2, 1, 0]`, verify both generations' raw pixels, require a new device's
cold allocations and zero completed-product hits, then return to the old device
and reuse its 1,000 exact completed textures. Inactive-generation pruning does
not remove an owner retained by pending work. Three pending slots remain the
existing bound; this is not a newly introduced eviction/budget API.

Four scripted tests additionally submit actual 12-texture Forward scenes through
the same manifest/access path. They cover replacement while pending, cancelled
wait ownership, map/flush and partial-recording failure without reusable-cache or
frame-pool publication, and renderer/device isolation. Broader completion,
backpressure, device-loss and deferred-cache failures remain covered by the full
Renderer regression. Scripted failures are logic evidence, not native driver
fault injection.

## Final execution and visible output

The final affected native matrix contains 18 attempts: S0-S3 foundation, S5
Forward/Surface/device isolation, S6 composition/Surface, and S7 Deferred
offscreen/Surface/device isolation on both backends. The final workspace MSRV
regression, strict Clippy, independent consumers and evidence-tool behavior logs
are retained separately and linked by the aggregate review; an empty per-attempt
logical-test list is not counted as a passing CPU test.

Four real-window demos cover Forward and Deferred on both backends. Each saves
11 distinct same-snapshot companion readbacks and window captures at frames
0/5/10. Window images are independently checked and actually inspected. Normal
demos remain interactive until closed; bounded evidence mode does not define
their normal lifetime.

Offscreen WGL roots stay hidden from creation. Visible windows are displayed
behind existing work without activation or topmost placement. DX12 can complete
a hidden first frame before revealing; WGL maps its background drawable before
the first presentation. Windows Graphics Capture reads the actual HWND without
raising it. Only the bottom two 16x16 DWM decoration corners are excluded from
schema-2 window pixel comparison; RHI raw pixels and historical schema-1 masks
have no such exclusion. The S6 reader correction is retained as a separate
checker-version supplement with the original failed pixel reports and negative
tests. Original producer bytes are unchanged.

## Revisions, artifacts and limits

| Dependency | Exact selected revision |
| --- | --- |
| RHI | `b5d569feace548a6ed498901deaf31a5efcaca61` |
| Bases | `ee640d57a8d7c32d251252d0e8b32950636a770c` |
| Host | `8d101624c603a6c1e0badf729768c6b26e70fba8` |

Rendering inputs are identified by exact retained worktree snapshots and actual
executable hashes; the Git HEAD recorded during execution is the published S6
baseline, with the later source bytes explicitly frozen. Dependency version
strings and sibling checkout HEADs are not substitutes for those pins.

The [final evidence index](evidence/s8-final-candidate/README.md) locates the
deduplicated archive, original producer ledgers, independent relocation review,
raw bytes, source/lock snapshots and tool supplements. Failed/interrupted probes
are diagnostic-only and are not mixed into accepted samples. The historical
[S7 acceptance record](s7-deferred-closure.md) remains unchanged in scope.

This result does not certify browser WebGPU/WebGL, Metal or Vulkan, a general
multi-mesh Deferred renderer, configurable scene lighting, IBL, normal mapping,
Forward+, shadows, post-processing or arbitrary SRP pass chains. The fixed-light
PBR and external position deformation retain their documented combination
limits. Native consumer evidence does not certify JSBridge browser lifecycle.
