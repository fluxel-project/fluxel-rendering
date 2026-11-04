# External Renderer Reference

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

The `compile-fail/private-graph` and `compile-fail/private-phase-context`
fixtures are independent crates. They confirm that external code cannot import
Renderer's private `graph` module or construct the renderer-owned phase context
needed to declare and submit work.

## Bounded S5 FramePipeline policy

The crate also implements two statically dispatched public `FramePipeline`
policies: `ExternalOpaquePipeline` and `ExternalClearPipeline`. They reuse an
external interval-based visibility provider, prepare an immutable scene snapshot,
and select one of the two renderer-owned Forward products. The integration
workload invokes these policies on the actual Scene-to-Forward path.

The policy declares its bounded opaque or clear pass and the required
vertex/uniform/texture/color accesses through the controlled phase context on
the actual Graph transaction, including on cache hits. It cannot create arbitrary
Graph resources, record commands, submit work, or declare arbitrary GPU accesses.
It is evidence for the bounded S5 API only; it is not a
general external GPU Feature, custom GPU product, or same-frame composition
claim.

The executable retains the S4 CPU reference artifact described by the
[S4 CPU closure record](../../documents/s4-cpu-closure.md). It validates data
ownership, identity, atomic changes, and later snapshot/typed-feature contracts.
It also checks the bounded S5 policies and two concrete visibility-provider
compositions. It does not itself own a GPU device, render or submit GPU work, or
claim a general Feature system, indirect work, or a supported graphics backend.

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

A fixed three-record scene has two independently expected visible records; the
external provider deliberately traverses candidates in reverse order to check
deterministic normalization. This local preparation check alone does not close S5 and
does not establish S6: external GPU resource contributions, heterogeneous
Feature composition, and an external pipeline plus Feature in the same frame
remain pending.

The [S5 closure record](../../documents/s5-forward-closure.md) combines this
independent consumer with the native integration matrix and measurement baseline.
