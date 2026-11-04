# S6 composed Forward measurement baseline

## Purpose and boundary

This document defines the first reproducible S6 baseline. It is a baseline
record only, not an optimization result or a regression threshold. The primary
metric is end-to-end process wall time for the frozen eleven-submission S6
offscreen typed-feature workload (`rendering-integration --s6-only`). One
process sample includes all eleven accepted offscreen submissions and their
readbacks, process/device startup, shader preparation and artifact I/O; it is
not a per-frame latency measurement.

The secondary observations are cold and warm release-build wall time, release
executable bytes, accepted-submission preparation time, cache-miss Graph
compile time, accepted allocation counts/requested bytes, resident-copy and
frame-upload bytes, cache hits/reuse, opaque candidate/batch/draw counts,
pending peak, and plan compilations. Their definitions are the fields retained
in `s6-composed-measurements.json`; they describe this fixture's accepted
Forward work, not general renderer or system resource consumption.
Build time and executable bytes cover the complete selected-backend integration
binary, which also contains other named workload entry points. They do not
isolate the marginal cost of static dispatch or of this single Feature.

`--s6-surface-only` is a correctness/lifecycle guard: it proves presentation,
resize and native capture behavior, but is intentionally excluded from timing.
Do not compare these results with S5: its workload, submission count, cache
interval and feature composition differ. No result here may claim a speedup,
memory reduction, or performance regression without a same-workload,
same-machine interleaved candidate comparison.

## Capture

Capture only after the selected source, locks, tool and RHI revision are
frozen. `tools/bench/s6_baseline.py` archives/hashes the complete relevant
source input closure, records toolchain and environment facts, rechecks source
stability before and after sampling, retains commands/logs, and refuses an
existing output directory. A failed build or process sample aborts the capture
while retaining its partial evidence.

Stabilize the power mode and close unrelated GPU-intensive work. Capture at
least five samples per backend in a new unique directory:

```powershell
python tools/bench/s6_baseline.py --backend dx12 --samples 5 `
  --output target/bench/s6-composed/<frozen-revision>/dx12-<utc>
```

When WGL is included, make a separate capture with `--backend native-gl-wgl`
and its own `native-gl-wgl-<utc>` directory. The tool takes five isolated cold
builds, one untimed warm-up build, five warm builds, and five complete direct
process samples. Every process sample must pass the independent S6 offscreen
review and retain exactly eleven accepted submissions.

## Interpretation

Report individual values plus minimum/median/maximum distributions. Treat
spread as evidence, not as data to discard. `preparationElapsedNs` is accepted
CPU-side preparation wall time up to Graph execution; `graphCompileElapsedNs`
is accepted cache-miss compilation time. Allocation/upload values are the
instrumented physical Forward-owner footprints (including renderer alignment
where applied), not GPU heap residency or process-wide allocation.

This baseline does **not** measure a production scene, GPU timestamp duration,
present-to-display latency, backend heap residency, allocator activity, or a
general batching policy. Add profiler or counter evidence before attributing a
future difference to a particular implementation mechanism.

## Captured baseline

Five cold builds, five warm builds and five complete process samples were
retained for each route on Windows 11 (10.0.26300), Ryzen 7 8845H, Radeon 780M,
driver 32.0.21028.2002, with the High performance power scheme. Rust/Cargo
1.87.0 targeted `x86_64-pc-windows-msvc`, using the release profile. All ten
process samples passed the actual independent S6 input/pixel/lifecycle reviewer.

| Observation (median [minimum, maximum]) | DX12 | Desktop OpenGL/WGL |
| --- | --- | --- |
| Cold build, seconds | 42.794 [42.329, 44.143] | 46.109 [45.088, 47.923] |
| Warm build, seconds | 0.462 [0.454, 0.472] | 0.476 [0.463, 0.494] |
| Complete eleven-submission process, seconds | 1.048 [1.042, 1.299] | 0.520 [0.512, 0.921] |
| Accepted preparation, milliseconds | 149.147 [141.277, 160.940] | 16.506 [15.544, 43.221] |
| Cache-miss Graph compilation, milliseconds | 0.124 [0.114, 0.172] | 0.131 [0.100, 0.212] |
| Release executable, bytes | 8,047,104 [8,047,104, 8,047,104] | 7,624,192 [7,624,192, 7,624,192] |

The following accepted-owner counters were identical across all five samples
of each route. They are total fixture observations, not peak heap residency.

| Counter per eleven-submission process | DX12 | Desktop OpenGL/WGL |
| --- | --- | --- |
| Buffer allocations | 24 | 24 |
| Requested buffer bytes | 157,300 | 152,116 |
| Texture allocations | 7 | 7 |
| Resident copy bytes | 1,780 | 1,780 |
| Frame upload bytes | 9,728 | 1,520 |
| Geometry hits | 15 | 15 |
| Texture hits | 15 | 15 |
| Frame allocation reuse | 8 | 8 |
| Pending peak | 3 | 3 |
| Graph plan compilations | 5 | 5 |
| Opaque candidates / batches / draws | 19 / 19 / 19 | 19 / 19 / 19 |

The full individual samples, commands, outcomes, environment, source snapshots,
logs and release binaries are retained by the
[baseline index](evidence/s6-typed-forward-candidate/baseline-index.json) and its
`baseline-evidence.zip`. The [native index](evidence/s6-typed-forward-candidate/index.json)
contains the separate correctness/diagnostic matrix. Exact inputs shared by
native and baseline captures match; benchmark Python tools are additional
baseline inputs. Publication-only documentation changes are listed separately
in `source-reconciliation.json`.

The spread remains part of the observation, including the first-sample highs.
These small integration workloads are not evidence of a production-scene
speedup, a backend ranking, or the marginal cost of generic dispatch. A later
optimization requires the same representative workload and an interleaved
baseline/candidate comparison.
