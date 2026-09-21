# ADR-0005: Require GPU conformance evidence for native correctness claims

**Status:** Accepted

## Context

Compilation, mocks, and CPU protocol tests cannot prove that DX12 and Vulkan
commands, barriers, shaders, and readback agree with portable semantics.

## Decision

Claim native correctness only when the same compiled ExecutionPlan runs on
DX12 and Vulkan with Required validation and an independent CPU oracle.
Artifacts identify the exact commit, backend, hardware/driver, inputs,
expected/actual result, completion, outgoing state, and diagnostics.

`scripts/conformance.ps1` is the sole release entry point for these fixtures.
It accepts only a clean Windows `x86_64-pc-windows-msvc` checkout, rejects a
conflicting `CARGO_BUILD_TARGET`, injects the checked-out 40-character SHA as
`FLUXEL_TEST_COMMIT`, and runs workspace ignored tests with all targets and
features. It writes `target/conformance/<sha>/manifest.json` and `cargo.log`;
the manifest records the command, toolchain, target, SHA, environment, GPU
identity, timestamps, exit status, and observed test-binary/case summaries.
The script retains both files when Cargo fails and returns Cargo's nonzero
status. These local artifacts are release evidence, not source-controlled
claims of success. After the matching annotated tag is pushed, both artifacts
are attached to that tag's GitHub Release; the durable asset URLs close the
evidence chain without changing the tested commit.

## Alternatives

- Treat CI compilation, TestRhi, or one backend demo as conformance.
- Let readback assume a convenient state instead of consuming an export's
  reported outgoing state.

## Consequences

Hardware fixtures are distinct from ordinary tests and are rerun on the final
commit before release through that script. CPU and compile evidence remain
useful but are labeled as such; they do not substitute for native oracle
evidence. A release is not tagged when either backend's required fixture is
unavailable or fails.

## Evidence

0.1.2 C01–C03, 0.1.3 K01–K02, and 0.1.4 R01/R02/X01 established the pattern.
0.2.0–0.2.7 extended it through U01–U08 and exact-SHA release reruns.

See [RHI design](../../crates/rhi/documents/design-rhi.md) for the current test-evidence boundary.
