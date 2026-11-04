# S5 Forward measurement baseline

## Decision

The first S5 record is a reproducible baseline, not an optimization result.
It must not be used to claim that S5 is faster or smaller than an earlier
implementation.  A future candidate needs an interleaved comparison against a
baseline captured with the same workload, machine, backend, toolchain, and
power configuration.

The primary product metric is end-to-end wall time for the offscreen S5 Forward
external-consumer workload. Its secondary measurements are clean and warm
build wall time, optimized executable bytes, and the workload's emitted
preparation and Graph-compilation elapsed time, accepted allocation and upload
facts, pending peak, candidate, batch, draw, cache-hit, and reuse facts. Surface
presentation remains a guard/correctness workload; it is deliberately not used
as a latency proxy.

## Workload boundary

| Role | Workload | Models | Does not model |
| --- | --- | --- | --- |
| Correctness gate | `tools/integration/smoke.py --workload s5-forward` plus the selected focused checks | Frozen inputs, independent readbacks, diagnostic collection, cache ownership, replacement, and completion semantics. | A timing sample or a performance decision. |
| Representative | `rendering-integration --s5-only` | The public external `FramePipeline` path through four layouts/materials/textures, repeated snapshots, cache reuse, replacement and offscreen readback. | A production scene corpus, presentation latency, a general batching policy, allocator cost, or GPU timestamp duration. |
| Guard | `s5-surface` and `s5-device-isolation` integration workloads | Surface lifecycle/resize and cache isolation must remain correct. | Comparable timing: their window and device lifecycle work is intentionally different. |
| Diagnostics | `s5-forward-cache-facts.json` and `s5-forward-measurements.json` | Accepted-submission preparation elapsed time, cache-miss Graph compilation elapsed time, buffer/texture allocation counts and requested buffer bytes, static/dynamic upload bytes, pending peak, cache hits/reuse, candidate/batch/draw counts. | GPU timestamp duration, backend heap residency, or process-wide allocator activity. |

The fixture has four deliberately distinct layout/material pairs.  Its expected
ten candidate/batch/draw facts describe this fixture's preserved-order path;
they do not establish that all real opaque scenes should have one draw per
batch.

## Capture procedure

Do not capture while the Renderer source, either lock, the baseline Python
tools, or the selected RHI Git revision is changing. A dirty Rendering
worktree is permitted: this is necessary when the already-smoked S5 work has
not yet been committed. Before any build, the tool reuses the smoke input
closure, adds every `tools/bench/*.py` input, hashes each file, and archives
the exact bytes as `source-inputs.zip`. It then checks the closure again before
measurement and after measurement. Any edit, removal, or newly relevant Python
input rejects the capture; the partial directory and all logs remain for
inspection.

`baseline.json` records Git HEAD and porcelain status only as a convenient
baseline locator. They are not the source identity for a dirty capture. The
archive hash and per-file hashes are the exact source identity, including
untracked files. First retain a passing smoke attempt for that same frozen
source snapshot. Then make power mode stable, close unrelated GPU-intensive
work, and capture at least five samples in a new directory:

```powershell
python tools/bench/s5_baseline.py --backend dx12 --samples 5 `
  --output target/bench/s5-forward/<frozen-revision>/dx12-<utc>
```

Use `native-gl-wgl` in a separate capture when that guard route is selected.
The command creates one empty target directory per cold build, performs one
untimed warm build, then records warm builds and direct workload-process
samples. It stores commands, return values, individual wall-clock samples and
minimum/median/maximum summaries, executable hashes/bytes, the exact frozen
source archive and per-file hashes, Git baseline locator, both lock hashes,
toolchain, OS, CPU/GPU, and active Windows power scheme in `baseline.json`.
Each workload sample must emit both cache and measurement facts; missing or
non-integer facts reject capture.
Every run output directory is unique; the tool refuses an existing result path.

The build and direct-process measurements are performance observations only.
The integration smoke runner remains the retained correctness and native
diagnostic evidence path.  A failed workload sample aborts the capture and its
log remains in the new directory.

## Interpretation and measurement boundary

Report distributions (minimum, median, maximum, and the individual values),
not a single remembered number.  Compare later revisions with adjacent or
randomized interleaved samples.  If a difference overlaps ordinary machine
noise, collect better evidence instead of attributing it to a cache, compiler,
driver, or scheduling mechanism.

`preparationElapsedNs` starts immediately before pipeline selection and ends
immediately before Graph execution. It includes synchronous API time for CPU
preparation, resource planning, owner allocation, pipeline/bind-group assembly,
and is only accumulated for accepted submissions. It is a wall-clock boundary,
not a CPU-cycle counter, and it excludes Graph execution, GPU completion and
readback. `graphCompileElapsedNs` measures only accepted cache-miss Graph
compilation. `residentStaticUploadBytes` and `frameUploadBytes` are physical
copy footprints, including alignment/padding where the renderer applies it.
Accepted buffer/texture allocation counts and requested buffer bytes describe
the Forward owners created by accepted submissions, not backend heap residency
or process-wide allocation activity. `pendingPeak` is the highest Renderer
pending count observed immediately after accepting Forward work; in the
isolated baseline all of those pending frames belong to Forward.

No GPU timestamp-query duration, backend heap-residency counter, or process-
wide allocator counter is configured. Those are not substituted with process
wall time and remain outside this baseline's claims.

Repeat the baseline when the representative fixture, FramePipeline/Forward
preparation path, RHI revision, Rust toolchain, feature set, optimizer profile,
or measurement machine/power configuration changes.  Before proposing an
optimization, first collect profiler, allocation, or stage-counter evidence
linking a measured cost to a concrete implementation mechanism.

## Retained first baseline

The first capture uses Rust/Cargo 1.87.0, `x86_64-pc-windows-msvc`, release
profile, Windows 11 build 26300, AMD Ryzen 7 8845H and Radeon 780M
(driver `32.0.21028.2002`), with the recorded High Performance power scheme.
Each backend has five isolated cold builds, one untimed warm-up build, five
warm builds and five complete Forward process samples. Both captures bind the
same exact source input list and passed the final source-consistency check.

| Observation (median [minimum, maximum]) | DX12 | Desktop OpenGL/WGL |
| --- | --- | --- |
| Cold build, seconds | 42.790 [40.883, 43.733] | 45.849 [44.367, 46.067] |
| Warm build, seconds | 0.489 [0.482, 0.508] | 0.568 [0.563, 0.659] |
| Complete process, seconds | 1.106 [1.057, 1.374] | 0.511 [0.492, 0.945] |
| Accepted preparation, milliseconds | 146.975 [143.079, 176.080] | 15.771 [14.862, 51.776] |
| Graph compilation, milliseconds | 0.197 [0.179, 0.220] | 0.179 [0.165, 0.235] |
| Release executable, bytes (all five cold builds) | 7,718,912 | 7,311,872 |

All five samples per backend report 28 accepted buffer allocations, 9 texture
allocations, pending peak 3, 1,748 resident static upload bytes, and 10
candidates/batches/draws in the selected cache interval. DX12 requests 214,580
buffer bytes and uploads 11,296 frame bytes; WGL requests 202,484 buffer bytes
and uploads 1,792 frame bytes. These are the instrumented physical footprints,
not an equal-backend memory contract or a heap-residency estimate.

The spread, especially the WGL process/preparation samples, is retained rather
than discarded. There is no candidate comparison, speedup claim, GPU timestamp
measurement, or steady-state frame-latency inference. The first baseline has
no performance pass/fail threshold; a future optimization must specify its
noise-aware comparison and acceptable regression before candidate measurement.
Correctness and resource-lifetime regressions remain failures regardless of speed.

[The baseline index](evidence/s5-forward-candidate/baseline-index.json) binds
all individual values, commands, logs, run artifacts, source archives and the
executed warm-build binaries in `baseline-evidence.zip`. Build cache directories
are excluded. [The native evidence index](evidence/s5-forward-candidate/index.json)
retains the separate correctness/diagnostic matrix. These captures establish an
initial reproducible workload baseline, not a representative production corpus.
