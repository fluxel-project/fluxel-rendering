# Shader cold-compile baseline

## Decision

Keep the current capability-inference implementation unchanged. This record
establishes a repeatable baseline before considering an optimization; it does
not attribute cost to a particular operation or claim a cross-revision
improvement.

The primary product metric is latency of a representative logical shader
composition cache miss. The cache-hit path is a guard workload: a regression
there requires investigation, but it does not decide an optimization of cold
compilation. Allocation counts, CPU profiles, GPU work, and native shader
creation are out of scope for this first baseline.

## Workloads

`crates/shader/benches/cold_compile.rs` uses only the public
`ShaderComposer::compose` API.

| Role | Workload | What it models | Exclusions |
| --- | --- | --- | --- |
| Representative | A fresh `ShaderComposer` composing three ordered WGSL contributors with vertex and fragment entries, three bindings, two vertex inputs, one callable, and one static option. | A multi-contributor graphics program requiring composition, WGSL parsing, full Naga validation, Naga capability inference, reflection, and interface checks while the logical artifact cache misses. | It is not a renderer asset corpus, native backend compilation, GPU pipeline creation, or an allocation measurement. |
| Guard | A second compose of the same complete request in one composer, with the first compose completed before timing. | The public logical-artifact cache-hit path. | It deliberately excludes cache population from timing. |
| Correctness gate | The harness asserts that the two artifacts are pointer-identical; `tests/cache.rs` checks complete request identity. | That the guard actually takes the intended cache path. | It is not a timing result. |

The harness accepts `--smoke` to execute each workload once without treating it
as a measurement. Its normal mode records individual nanosecond samples after
five warmups; the default is 41 samples. The command uses Cargo's optimized
`bench` profile and has no benchmark-framework dependency.

## Recorded measurement

The baseline source was revision
`2ce58d599ee7bdfdea46b4ebfac487da94667aae`. The benchmark harness was an
uncommitted addition at measurement time; its SHA-256 was
`d0703724bf33f6a2804d2da519de24a2e41be69053b5388ef3a7f3cd2af84de1`.
No production Shader source changed for this measurement.

| Field | Value |
| --- | --- |
| Date | 2026-11-03 (Asia/Tokyo) |
| OS | Windows 11 Home, version 10.0.26300, build 26300 |
| CPU | AMD Ryzen 7 8845H with Radeon 780M Graphics; 8 cores / 16 logical processors |
| Power scheme | High performance (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`) |
| Toolchain | `rustc 1.87.0 (17067e9ac 2025-05-09)`, LLVM 20.1.1; `cargo 1.87.0 (99624be96 2025-05-06)`; `x86_64-pc-windows-msvc` |
| Dependency | `naga 27.0.3` from the locked workspace dependency graph |
| Command | `cargo +1.87.0 bench -p fluxel-shader --bench cold_compile -- --samples=41` |
| Parameters | 5 warmups and 41 timed samples per workload; no profiler or allocation counter collected |

The final measurement produced the following durable raw summaries, in
nanoseconds. Values are sorted by the harness before output, so their order is
not execution order.

```text
representative-cold-cache-miss
minimum=248400 median=257600 maximum=335500
[248400, 250000, 251000, 251600, 251800, 252200, 252400, 254100, 254200,
 254800, 254900, 254900, 255000, 255100, 255400, 255700, 256800, 257200,
 257300, 257300, 257600, 257900, 258100, 259200, 259600, 260900, 262100,
 264400, 264900, 265600, 267200, 276000, 284000, 287500, 292100, 303400,
 310400, 310700, 312600, 319400, 335500]

guard-cache-hit
minimum=700 median=800 maximum=1600
[700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
 700, 700, 700, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800,
 800, 800, 800, 800, 800, 800, 800, 900, 900, 900, 900, 1200, 1600]
```

Two adjacent valid repetitions produced representative medians of 215.5 and
341.3 microseconds, and cache-hit medians of 0.7 and 1.1 microseconds. That
variation makes this a same-machine baseline and regression guard, not a basis
for claiming a small candidate speedup without randomized or interleaved A/B
measurement under a controlled load.

## Validation and retest conditions

Smoke passed with:

```text
cargo +1.87.0 bench -p fluxel-shader --bench cold_compile -- --smoke
```

The focused cache correctness gate also passed:

```text
cargo +1.87.0 test -p fluxel-shader --test cache --locked
```

Repeat this baseline when the capability inference algorithm, the representative
Shader request contract, Naga version, or Rust toolchain changes. Before an
optimization is proposed, collect profiler or allocation evidence that links a
measured cold-compile cost to a specific operation, then compare an isolated
candidate against this workload with interleaved samples and rerun the guard.

## Post-measurement boundary

After this measurement, a lint-only cleanup boxed the private Naga validation
error transport and removed a redundant iterator conversion. It did not change
the successful capability-inference loop, its probe order, or the benchmark
input. No new timing was collected after that source edit, so the raw numbers
above remain evidence for the recorded revision only and are not presented as
measurements of the later source.
