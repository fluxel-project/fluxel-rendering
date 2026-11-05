# Final S8 native evidence

This archive retains the final native matrix, measured strategies, visible demos
and resource-lifecycle evidence described by the
[S8 closure record](../../s8-measured-closure.md). Independent archive review
accepts the selected native scope after isolated relocation; publication status
is determined separately by the repository commit and remote revision.

- `evidence.zip`: 52,778,935 bytes; SHA-256
  `c24fb5c04be7722052ea1422a07d1d5a671b2d9a344da3f74915679325d12051`.
- `archive.json`: original archive size/hash and inventory summary.
- ZIP `manifest.json`: 6,277 logical files backed by 1,252 hash-addressed blobs.
  Deduplication preserves original bytes; logical paths are restored before
  review. Original producer JSON is never rewritten to relocate it.
- `archive-review.json` and retained reviewer code: verified integrity, source,
  behavior, pixel, ownership, threshold and relocation results. The dedicated
  `native-oracle-review.json` covers all 18 selected native attempts and eight
  CPU regression logs; source and measured-comparison subgates remain explicit.

The accepted selection is explicit in the archived producer ledgers:

| Evidence | Logical locator inside the materialized archive |
| --- | --- |
| Final CPU tests, strict Clippy and source snapshot | `target/final-regression-20261105T0825/` |
| 18 actual native workload attempts | `target/final-regression-20261105T0825/native-matrix/native-matrix.json` |
| Four final visible demos | `target/final-regression-20261105T0825/demos/attempts.json` |
| Two final alternating performance comparisons | `target/final-regression-20261105T0825/performance/comparisons.json` |
| Native high-cardinality lifecycle, DX12 | `target/bench/s8/final-lifecycle-dx12-20261105T0915/attempt.json` |
| Native high-cardinality lifecycle, WGL | `target/bench/s8/final-lifecycle-native-gl-wgl-20261105T0915/attempt.json` |
| S6 window checker correction and negative tests | `target/final-regression-20261105T0825/reviewer-probes/` |

Each performance record retains its original baseline, executable, exact source
archive, input/readback hashes, alternating samples, counters and small guards.
GPU duration is explicitly unobserved. Native lifecycle outputs contain no timed
sample claims. Full CPU behavior evidence is global and is not fabricated as
repeated per-native-attempt tests.

Source snapshots distinguish the exact producer code from later read-only
checker/documentation changes. The S6 checker supplement applies the already
defined schema-2 DWM corner policy to window BMPs only; it does not change native
workloads, RHI readbacks, archived manifests, or historical schema-1 semantics.
Contact sheets and typed inspection records bind actual window/readback images.

Failed startup/capture probes, deliberately corrupted oracle inputs and the
interrupted older WGL comparison remain diagnostic-only. Explicit selection
prevents those records, or historical attempts mentioned by an original closure
manifest, from being accepted as part of the final result.
