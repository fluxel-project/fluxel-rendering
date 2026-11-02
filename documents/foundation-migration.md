# Renderer reset inventory

The current directory is the starting point for a new renderer implementation.
This document records what is present now; it is not completion evidence and it
does not authorize recovery of removed source or tests.

The current S0/S1 implementation uses one private Graph for Copy, Raster,
Compute, uniform/texture read-set and Surface consumers. Current native matrix,
logical fault tests, source archives and limitations are described in the
[foundation closure review](s0-s1-closure.md). The dated review below is a
historical source snapshot, not the current verification summary.
The architecture still requires static batch extensions under
[ADR-0025](adr/0025-composable-rendering-policies.md) and
[ADR-0026](adr/0026-static-extension-composition.md); foundation fixtures do not
implement or freeze those later public APIs.

## Workspace state

| Boundary | Current state |
| --- | --- |
| `fluxel-renderer` | Eight private ownership boundaries; the entry facade contains bounded Copy, offscreen Raster, supported Compute, fixed Surface raster/present, and completion-based ownership. Current logical and native evidence is linked in the closure review, with injected loss distinguished from native replacement. Scene, pipeline, feature, preparation and frame data remain shells. No stable extension API exists. |
| `fluxel-shader` | A scaffold for the S2 target crate. No shader composition, validation, reflection, or RHI execution is implemented. |
| `fluxel-material` | A scaffold for the S3 target crate. No material graph, lowering, asset, instance, or shader-generation API is implemented. |
| Standalone `fluxel-rendergraph` | Removed. A fresh private buffer/texture definition/compiler and fixed Copy/Raster/Compute lowering now live under Renderer. The exercised S1 foundation subset is recorded in the closure review; later public extension APIs are not implied. |
| `rendering-wasm` | Removed. Browser restoration requires a new producer using the unified RHI and the JSBridge lifecycle boundary. |
| Renderer GPU features and native routes | Explicit `dx12` and `native-gl-wgl` forward to pinned RHI; default features are empty. No `gpu` feature or full native support claim exists. |
| Renderer examples and tests | Public example/test directories remain placeholders; private Graph and ownership tests exercise implemented logic. Neither establishes complete API compatibility or route support. |
| `examples/rendering-integration` and `tools/integration` | Independent Copy/Raster/Compute/read-set/Surface/replacement consumer and active evidence runner. Historical Rust sources/results are retired. Fresh attempts prove only their exercised subset. |

## Focused consistency review, 2026-10-30

This section records only that historical source state. Its open items and test
counts are superseded by the current closure review above.

The organization roadmap, design documents, manifests/locks, independent consumer
and current source were reviewed together. Stage order and ownership remain
unchanged. Shader/Material and the public static extension facade remain future
work; fixture implementation does not establish those APIs.

The review corrected texture definedness across adjacent subresource writes,
same-state texture hazards, exact plan/resource-instance checks for Compute
fixture bindings, and device-loss handling during completion/readback. Shutdown
tests are now connected to the crate test modules. The consumer retains Copy
inputs and raw readbacks before byte oracles; the runner retains a missing-manifest
failure and hashes its current diagnostic policy instead of a historical waiver.

Focused verification of the local working tree used Rust 1.87.0:

- Renderer library tests: 61 passed in the batch, followed by one newly added
  readback-loss test passing in a targeted run (62 distinct cases).
- The independent consumer passed `cargo check --locked --offline` separately
  with `--no-default-features --features dx12` and `native-gl-wgl`.
- Runner checks: nine passed in the batch; after fixing the new test's temporary
  root isolation, its missing-manifest case passed separately (10 distinct cases).

These are logic and compilation results, not native execution evidence. The
nonzero Copy offsets and shared execution path still needed hardware validation
at that point. Dynamic bindings/aliases, device replacement, Surface faults,
validation and visual inspection remained open in that snapshot. S0/S1 was not
complete then. No RHI interface was changed and that review did not run a full
workspace or dual-backend GPU regression.

## Reset rules

The removed fixed-frame, adapter, browser-session, and standalone-Graph public
surfaces are not compatibility APIs. New implementation begins from the current
module boundaries, selected RHI contracts, and the
stage order in [the implementation plan](version-plan.md). It must not recreate
public session/token resources, a standalone Graph crate, or a second graph
engine.

No test, integration command, diagnostic, screenshot, or hardware result from a
removed implementation is current coverage. New claims require a newly built
consumer, exact dependency inputs, and fresh evidence under the applicable
stage's acceptance criteria.
