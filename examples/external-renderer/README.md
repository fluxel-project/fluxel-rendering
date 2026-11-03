# External Renderer CPU Reference

This independent consumer exercises the public RenderScene batch boundary without
an ECS, GPU device, Host window, or access to Renderer-internal Graph state. It
keeps two application-side adapters: a direct object map and an ECS-shaped set of
component columns. Both translate the same logical objects and views into public
`RenderChangeSet` values and check their observable Scene results. Both retain an
old immutable `FrameSnapshot` through a later update, then bind two differently
shaped local feature snapshots (dense instance colors and per-view labels) to the
matching Scene version.

The reference owns its geometry marker but uses the real S3
`fluxel_material::MaterialAsset` and `MaterialDefinition` through the pinned
Bases `AssetStore`. It compiles both material generations through the S3
material compiler before publishing them to S4. The update uses a replacement
material snapshot: the logical `AssetId` remains stable while
`ContentGeneration` advances. No GPU residency or backend object enters the
Scene record.

It implements the provisional public Renderer CPU extension facade: the static
`CpuFeature` trait prepares dense colors and view labels from the same immutable
scene snapshot. Renderer-owned `FeatureIssuer` creates opaque feature handles
and contexts; issuer-bound `TypedFeatureStorage` retains each feature snapshot.
Both features contribute to the controlled CPU phase collector. Version mismatch,
cross-scene equal-version mismatch, same-scene cross-issuer payload mismatch,
issuer-bound storage mismatch, mixed-version/replayed collector contributions,
and rejected cross-feature contributions are structured errors. The facade
exposes neither Graph nor GPU authority.

The `compile-fail/private-graph` fixture is an independent crate. It confirms
that the public `RenderScene` import is usable but that attempting to import
Renderer's private `graph` module fails with Rust privacy error E0603.

The executable is the retained S4 CPU reference artifact described by the
[S4 CPU closure record](../../documents/s4-cpu-closure.md). It validates data
ownership, identity, atomic changes, and later snapshot/typed-feature contracts.
It does not render, submit GPU work, create a FramePipeline, or claim culling,
indirect work, or a supported graphics backend.

Before either adapter starts scene construction, each executable attempt saves
its source input, independently specified expected facts, an initially unavailable
actual record, and a pending error record under
`target/external-renderer-evidence/`. It writes the actual facts before running
the oracle and records any construction or oracle error. The contract test also
intentionally fails the real entry path and verifies those four records remain
available.

This manifest is intentionally a standalone workspace with its own lockfile. It
depends on the public Renderer crate and the pinned Bases asset crate used to
prove asset generations. A compilation or run here is evidence for this bounded
API, not permission to access private Graph or RHI execution state.

Run from this directory:

```powershell
cargo +1.87.0 test --locked
cargo +1.87.0 run --locked
python scripts/capture_s4_evidence.py --attempt attempt-002
```
