# Fluxel Rendering

Fluxel Rendering provides scene preparation, logical shader composition and
materials over the independent [Fluxel RHI](https://github.com/fluxel-project/fluxel-rhi).
The workspace uses Rust 2024 and requires Rust 1.87 or newer.

| Crate | Responsibility | Design |
| --- | --- | --- |
| `fluxel-renderer` | Scene snapshots, static Feature/FramePipeline composition, private RenderGraph, residency and completion-owned execution | [Renderer](crates/renderer/documents/design.md) |
| `fluxel-shader` | WGSL composition, Naga validation/reflection, logical artifacts and requirements | [Shader](crates/shader/documents/design.md) |
| `fluxel-material` | Material definitions, typed functions/Code Nodes, variants, instances and Shader contributions | [Material](crates/material/documents/design.md) |

Hosts own windows, canvas lifecycle, input and frame pacing. Bases owns logical
asset identity and content generations. RHI owns GPU realization, recording,
submission, completion and presentation. Renderer owns dependency planning,
actual-owner validation, caches and completion-based resource retirement.

## Current scope

Forward supports multiple meshes, vertex layouts, materials and textures, plus a
statically composed external quad Feature in the same opaque pass. Deferred is
currently a one-mesh product with fixed lighting, selected metallic/roughness
PBR and an optional external position producer. These are bounded products;
there is no public general Graph API, arbitrary pass-chain interface, dynamic
plugin registry or full PBR system. Runtime scene and material data remain
ordinary values; concrete extension implementations use generics and traits.

The native consumers select Windows DX12 or desktop OpenGL through WGL. Feature
selection, CPU tests and compilation alone do not establish GPU or platform
support. Other backend/browser routes need their own current executable
consumer, device facts and evidence. The crate designs distinguish current
behavior from future direction.

## Layout

```text
crates/
  renderer/
    documents/design.md
    examples/
      external-renderer/       # independent public-API consumer and compile failures
      rendering-integration/   # native workloads and visible demos
    src/                      # unit tests grouped under src/tests/
    tests/
  shader/
    documents/design.md
    benches/
    src/
    tests/
  material/
    documents/design.md
    src/
    tests/
tools/integration/            # runner, oracles, policy and manifest schema
tools/bench/                  # workload measurement and comparison tools
```

The two renderer consumers remain standalone Cargo workspaces with independent
lockfiles. Their location expresses ownership, not membership in the library
workspace. Each library crate contains its README, design and dual-license
texts. Local plans, reviews and historical evidence are kept under ignored
`draft/`; they are not public design documents or current support claims.

## Historical acceptance index

The S0--S8 records were intentionally removed from the live source tree during
the layout consolidation. They remain auditable at rendering revision
[`c24f9adf17cfd12ed2bc3035f45cd1f477879afc`](https://github.com/fluxel-project/fluxel-rendering/tree/c24f9adf17cfd12ed2bc3035f45cd1f477879afc):

| Scope | Historical record |
| --- | --- |
| S0--S1 foundation | [`documents/s0-s1-closure.md`](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/s0-s1-closure.md) |
| S2--S3 Shader and Material | [`documents/s2-s3-closure.md`](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/s2-s3-closure.md) |
| S4 RenderScene CPU boundary | [`documents/s4-cpu-closure.md`](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/s4-cpu-closure.md) |
| S5 Forward | [`documents/s5-forward-closure.md`](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/s5-forward-closure.md) |
| S6 typed Forward Feature | [`documents/s6-typed-forward-closure.md`](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/s6-typed-forward-closure.md) |
| S7 bounded Deferred | [`documents/s7-deferred-closure.md`](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/s7-deferred-closure.md) |
| S8 validation-index strategy | [`documents/s8-measured-closure.md`](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/s8-measured-closure.md) and its [final evidence index](https://github.com/fluxel-project/fluxel-rendering/blob/c24f9adf17cfd12ed2bc3035f45cd1f477879afc/documents/evidence/s8-final-candidate/README.md) |

These are bounded historical results for the source, inputs, and native matrix
named by each record. They do not certify this checkout, expand backend or host
support, or replace current executable validation.

## Development

```powershell
cargo +1.87.0 test --workspace --lib --tests --locked
cargo +stable clippy --workspace --all-targets --locked -- -D warnings
cargo +stable fmt --all -- --check
cargo +1.87.0 test --manifest-path crates/renderer/examples/external-renderer/Cargo.toml --locked
python -m unittest discover -s tools/integration -p "test_*.py"
python -m unittest discover -s tools/bench -p "test_*.py"
```

For a real-window Forward demo on Windows:

```powershell
cargo +1.87.0 run --manifest-path crates/renderer/examples/rendering-integration/Cargo.toml --features dx12 -- --demo-forward --output target/demo-forward
```

Use `native-gl-wgl` for the WGL route or `--demo-deferred` for the Deferred
product. Visible demos render to a real Surface; unattended tests keep required
host windows hidden. See the [renderer examples](crates/renderer/examples/README.md),
[integration procedures](tools/integration/README.md),
[evidence policy](tools/integration/policy.md), and [release procedure](RELEASING.md).

The crates are currently unpublished (`publish = false`). Git consumers select
exact revisions and retain their dependency locks. Licensed under
`MIT OR Apache-2.0`.
