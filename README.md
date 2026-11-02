# Fluxel Rendering

Fluxel Rendering is a host-independent Rust rendering workspace. Its target is
an extensible, data-oriented renderer built on the independent
[`fluxel-rhi`](https://github.com/fluxel-project/fluxel-rhi) and the logical asset
core in [`fluxel-bases`](https://github.com/fluxel-project/fluxel-bases).

The renderer consumes scene/view updates, resource inputs, a host-supplied surface,
and explicit frame calls. Hosts and browser/mini-game adapters own platform I/O,
windows/canvas lifecycle, input, and frame pacing. Renderer does not own a gameplay
scene, mandatory ECS, event loop, or TypeScript gameplay runtime.

The target exposes Rust Feature and FramePipeline contracts over typed batch data.
Applications assemble concrete geometry, visibility, ordering and deformation
providers through generics, traits and monomorphization at compile time. Different
providers can coexist in one frame; no trait-object dispatch or runtime type
erasure is used for extensibility. SoA/AoS layouts remain workload choices, and
all frame GPU work uses the internal Graph. See the
[Renderer design](documents/design-renderer.md) for the target contracts; these
extension APIs are not yet implemented.

## Current implementation and target

The checked-out `0.16` workspace contains the S0/S1 foundation implementation
and scaffolds for later stages:

| Workspace crate | Current role |
| --- | --- |
| `fluxel-renderer` | Initial Copy, offscreen and Surface Raster, and supported Compute fixtures with private Graph and completion-owned resources; scene and extension APIs remain to be built |
| `fluxel-shader` | Minimal crate scaffold; composition and reflection APIs are not implemented yet |
| `fluxel-material` | Minimal crate scaffold; material functions and lowering APIs are not implemented yet |

```text
crates/
  renderer/          # fluxel-renderer; includes the private graph module
  shader/            # fluxel-shader
  material/          # fluxel-material
```

The standalone `rendergraph` and historical `rendering-wasm` crate directories
have been removed. RHI remains in its independent repository.

All previous Renderer implementations and crate-local examples have been removed.
The rebuilt Renderer exposes narrow foundation-fixture entry points for the S0/S1
consumer, with pinned Bases/RHI dependencies and explicit DX12/WGL features.
The remaining scene and extension boundaries are scaffolds, not a complete
rendering API. The [foundation closure review](documents/s0-s1-closure.md) maps
the current logical tests and retained DX12/WGL native artifacts separately.

The previous integration sources have been removed. The newly implemented
[independent consumer](examples/rendering-integration/README.md) checks three
asset generations, Copy/Compute readback, offscreen Raster pixels, Surface
present/resize, and structured refusals. Old logs and images are not
compatibility or support evidence for the rebuilt path.

## S0/S1 support matrix

S0/S1 foundation acceptance is scoped to Windows DX12 and desktop OpenGL 4+
through WGL. The closure review identifies the exact observed adapter/profile,
source hashes, locks, native outputs and deterministic failure witnesses. It does
not certify untested hardware, a general renderer API or later stages.

Vulkan, browser WebGPU, and browser WebGL2 are deferred from this closure. The
historical `fluxel-rendering-wasm` capsule has been removed. Browser restoration
must supply a producer using the current RHI architecture. The
route-by-route consumer status, owner, and restoration evidence are in the
[S0/S1 support decision](documents/s0-s1-support-decision.md).

The target has three principal crates:

| Target crate/module | Responsibility |
| --- | --- |
| `fluxel-shader` | WGSL composition, logical validation/reflection, source artifacts, interfaces, requirements, and cache identity |
| `fluxel-material` | Material functions, parameter/resource schemas, minimal composition IR, evaluation WGSL, assets/instances/variants |
| `fluxel-renderer` | Batch RenderScene/views, preparation, visibility, ordering/batching, GPU residency, frame data, and execution |
| Renderer `graph` module | Pass/resource declarations, versions, hazards, DAG planning, lifetimes, and direct portable RHI execution |
| Renderer `pipeline` / `feature` modules | Private shells for the future S4/S5 Rust extension facade |

Material and Shader do not depend on Renderer. Material emits generic WGSL and
interface contributions; Renderer/FramePipeline assembles the complete
`ShaderCompositionRequest` with pass templates and entry points. Shader produces
validated logical programs and reflection. RHI owns device shader-route selection,
backend ABI/lowering, native compilation, and executable objects. Compilation can
precede rendering and its products can be reused across frames.

The frame path is:

```text
Batched scene changes -> RenderScene / RenderView / Features
    -> FramePipeline policy and preparation -> internal RenderGraph -> RHI
```

Graph declarations cover every bound buffer/texture access, including prepared
material resources. Compatible per-frame resource-instance changes reuse compiled
plans after manifest validation; new hazards/dependencies or incompatible access
contracts require recompilation or refusal. Graph planning uses portable RHI
transitions/dependencies; native synchronization remains backend-private.

Persistent residency and frame-data uploads have distinct lifetimes while sharing
transfer machinery and RHI completion/state authority. Features request preparation
and declare all frame GPU work through Graph. Partial/indeterminate native effects
cannot be treated as clean rejection or permission to recycle resources.

## Delivery

The organization [roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
is the stage-order authority. The [implementation plan](documents/version-plan.md)
defines each stage's modules, executable output, failures, and evidence:

```text
S0 Foundation + S1 Graph -> S2 Shader -> S3 Material + S4 RenderScene
    -> S5 Forward integration -> S6 external extension/product MVP
    -> S7 selected rendering expansion -> S8 measured resource strategies
```

S0/S1 share one foundation closure. S3/S4 may overlap after S2. Feature and phase
seams are exercised early in S4/S5; S6 closes only when an external Rust Feature,
external FramePipeline, and custom material cooperate in the same real frame.
Code Nodes are planned S7 functionality; complete authoring breadth does not block
the MVP. Optimization follows measurements retained from the earlier stages.

Capture/replay, Canvas 2D/basic text, declarative UI, Blender tooling, and JavaScript
wrappers remain undecided. A future game scene/script runtime needs separate
planning and is not a rendering milestone.

## Documentation

- [Product direction](documents/plan-overview.md)
- [Implementation stages and acceptance](documents/version-plan.md)
- [Workspace architecture](documents/design-overview.md)
- [Shader design](documents/design-shader.md)
- [Material design](documents/design-material.md)
- [Renderer design](documents/design-renderer.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [Architecture decisions](documents/adr/README.md)
- [Migration inventory](documents/foundation-migration.md)
- [Shader package scaffold](crates/shader/README.md)
- [Material package scaffold](crates/material/README.md)
- [Current renderer package guide](crates/renderer/README.md)
- [Release and evidence requirements](RELEASING.md)

Package guides distinguish the current implemented subset from target design.
The selected independent RHI's [Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md),
API documentation, and tests define portable execution contracts.

## Verification and source dependencies

Git consumers pin a revision or tag rather than following `main`. Migration must
update affected manifests/locks and verify real producer/consumer combinations;
matching version strings are not integration evidence.

The current workspace uses Rust 1.87 and edition 2024. The following are full
workspace regression commands for completed major refactors or complete goal
implementation and closure. During implementation, validate coherent batches
with only the necessary affected checks; do not run these for each small edit:

```sh
cargo +1.87.0 test --workspace --all-targets --all-features --locked
cargo +1.87.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
```

Compilation and CPU tests do not certify GPU behavior. Real-target evidence must
identify exact revisions, features, target/backend, inputs, inspected output, and
completion/lifecycle outcomes. Follow [RELEASING.md](RELEASING.md) before making a
release or renewed support claim; removed historical harnesses are not runnable
release commands in this checkout.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
