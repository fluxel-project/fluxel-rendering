# Rendering Integration Fixtures

This standalone workspace owns runnable Windows DX12 and desktop OpenGL/WGL
fixture consumers. It builds with exactly one backend feature, drives the public
Renderer/RHI/Host boundary, and retains inputs, expected results, raw output,
diagnostics, and execution facts for the integration tooling to review.

The fixtures cover fixed Copy, Raster, supported Compute, Surface presentation,
direct Shader and Material composition, bounded Forward work, controlled opaque
Feature composition, selected Deferred work, and the material-heavy validation
workload. These are concrete consumers, not a general graphics API.

The Deferred deformation workloads use the independent consumer's `BendFeature`
to produce positions in a Graph compute pass and consume them as geometry vertex
input. Offscreen and Surface paths exercise this bounded cross-pass GPU product;
its source, parameters, readbacks and pending-owner facts belong to each evidence
attempt. Historical acceptance does not certify a changed checkout.

For an interactive result, select `dx12` or `native-gl-wgl` and run a demo:

```powershell
cargo +1.87.0 run --manifest-path crates/renderer/examples/rendering-integration/Cargo.toml --features dx12 -- --demo-forward --output target/demo-forward
```

Use `--demo-deferred` for the Deferred demo. Demos keep automatic display behind
existing windows without activation. Offscreen routes keep any required host
window hidden. The tooling guide describes bounded visual capture and review;
a readback is not a substitute for a window image.

The executable selectors include `--s5-only`, `--s6-only`, `--s6-surface-only`,
and the selected Deferred and material-heavy routes. They are fixture selectors,
not public API commitments. Run retained-evidence procedures from the repository
root:

```powershell
python tools/integration/smoke.py --backend dx12
python tools/integration/smoke.py --backend native-gl-wgl
```

The manifest pins RHI
`b5d569feace548a6ed498901deaf31a5efcaca61`, Bases
`124e825f82e5cd0eb485c95fe1281b11c8dbc844`, and Host
`8d101624c603a6c1e0badf729768c6b26e70fba8`. The root workspace excludes this
consumer; its lockfile is part of the executable input.

See [`tools/integration/README.md`](../../../../tools/integration/README.md) for
procedures and [`manifest.schema.json`](../../../../tools/integration/manifest.schema.json)
for retained-manifest structure. Renderer architecture and scope live in
[its design](../../documents/design.md).
