# External Renderer Reference

This standalone workspace is an independent public-API consumer of
`fluxel-renderer`. It intentionally has no access to private Graph state,
command recording, allocation, or submission authority.

The executable contains two application-side scene adapters: a direct object
map and ECS-shaped component columns. Both emit public `RenderChangeSet` values
and verify equivalent observable `RenderScene` results. It also exercises
immutable snapshots, version-bound typed CPU feature storage, controlled CPU
phase contributions, material compilation, geometry preparation, and deterministic
opaque batching.

The crate includes compile-fail fixtures proving that external code cannot
import Renderer's private `graph` module or construct the renderer-owned phase
contexts that declare and submit GPU work. Its typed `QuadFeature` and external
pipeline demonstrate the controlled opaque Forward composition seam: the Feature
provides prepared requests and receives a resolved read scope, while Renderer
retains Graph, RHI, and lifetime ownership.

Its `BendFeature` implements `DeferredPositionDeformer`: typed parameters and a
WGSL function feed a Renderer-owned compute pass whose position buffer is read
by the subsequent Deferred geometry pass. Native offscreen and Surface consumers
live in `../rendering-integration`; this CPU reference alone does not certify
that GPU execution.

Run from this directory:

```powershell
cargo +1.87.0 test --locked
cargo +1.87.0 run --locked
```

The manifest has its own lockfile and pins `fluxel-assets` at
`124e825f82e5cd0eb485c95fe1281b11c8dbc844`. It is a CPU/public-contract
consumer; it does not create a GPU device or certify backend support. Native
fixture procedures and evidence schema are documented under
[`tools/integration`](../../../../tools/integration/README.md) and
[`manifest.schema.json`](../../../../tools/integration/manifest.schema.json).

For architecture and limits, read the Renderer
[design](../../documents/design.md).
