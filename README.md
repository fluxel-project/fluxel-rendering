# Fluxel Rendering

Fluxel Rendering is a host-independent Rust rendering workspace. Its target is
an extensible, data-oriented renderer built on the independent
[`fluxel-rhi`](https://github.com/fluxel-project/fluxel-rhi) and the logical asset
core in [`fluxel-bases`](https://github.com/fluxel-project/fluxel-bases).

The renderer consumes scene/view updates, resource inputs, a host-supplied surface,
and explicit frame calls. Hosts and browser/mini-game adapters own platform I/O,
windows/canvas lifecycle, input, and frame pacing. Renderer does not own a gameplay
scene, mandatory ECS, event loop, or TypeScript gameplay runtime.

The target will expose Rust Feature and FramePipeline contracts over typed batch data.
Applications assemble concrete geometry, visibility, ordering and deformation
providers through generics, traits and monomorphization at compile time. Different
providers can coexist in one frame; no trait-object dispatch or runtime type
erasure is used for extensibility. SoA/AoS layouts remain workload choices, and
all frame GPU work uses the internal Graph. The public subset includes
RenderScene storage, immutable snapshots, a bounded CPU feature facade, S5
Forward's controlled `FramePipeline` preparation and pass/access declarations,
and the demonstrated S6 typed opaque-Forward composition subset, plus the named
S7 Deferred one-mesh acceptance subset, plus the selected S8 measured
resource-strategy acceptance. Neither is a general GPU-product or
cross-pass Feature contract. See the
[Renderer design](documents/design-renderer.md) for the target contracts.

## Current implementation and target

The checked-out `0.16` workspace contains the S0/S1 foundation plus the first
implemented S2 Shader, S3 Material, closed S4 CPU scene/feature data boundary,
S5 Forward integration, a bounded S6 external typed-Feature composition, and a
bounded S7 Deferred pipeline acceptance, and the selected S8 measured
resource-strategy acceptance.
S5 renders multiple meshes, layouts, materials and textures through Scene ->
FramePipeline -> Graph -> RHI. S6 adds one independently built typed quad
Feature, custom material, and external composed pipeline alongside those meshes
in the same opaque Forward Graph pass; it does not claim general GPU products.
S7 adds one selected Deferred path sharing scene/material inputs, with optional
static external deformation, selected CodeNode and metallic/roughness PBR cases,
and fixed-light output; it does not claim a general PBR or pipeline framework.
S8 measures only the private validation-index strategy used by the material-heavy
Forward consumer; it preserves the existing Graph and owner-validation authority
and does not turn that consumer into a general optimization API.

| Workspace crate | Current role |
| --- | --- |
| `fluxel-renderer` | Private Graph, Copy/Raster/supported Compute, RenderScene snapshots, bounded CPU feature and Forward pipeline facades, per-device residency, frame data and completion-owned resources |
| `fluxel-shader` | WGSL composition, Naga validation/reflection, source provenance, interface checks, requirement inference, and logical artifact caching |
| `fluxel-material` | Rust-defined fragments, typed parameters/resources, validated call graphs, static variants, instances, and generic WGSL contributions |

```text
crates/
  renderer/          # fluxel-renderer; includes the private graph module
  shader/            # fluxel-shader
  material/          # fluxel-material
```

The standalone `rendergraph` and historical `rendering-wasm` crate directories
have been removed. RHI remains in its independent repository.

All previous Renderer implementations and crate-local examples have been removed.
The rebuilt Renderer exposes narrow fixture entry points with pinned Bases/RHI
dependencies and explicit DX12/WGL features. The S2/S3 consumers compose
validated logical shaders, query the selected device route and capabilities, and
then use the existing Graph path. RenderScene now exposes its bounded CPU update
and snapshot API. It also exports the provisional CPU-only `CpuFeature` facade;
S5 adds the bounded Forward `FramePipeline` path. The
[foundation closure review](documents/s0-s1-closure.md) maps the S0/S1 logical
tests and retained DX12/WGL native artifacts separately.

The previous integration sources have been removed. The
[independent consumer](examples/rendering-integration/README.md) covers three
asset generations, Copy/Compute readback, offscreen Raster pixels, Surface
present/resize, direct S2 shader execution, and S3 material contribution and
replacement, plus S5 multi-layout Forward, device isolation and Surface resize.
It records structured refusals as well as accepted routes. Old logs
and images are not compatibility or support evidence for the rebuilt path.

The consumer also provides persistent `--demo-forward` and `--demo-deferred`
windows on DX12/WGL. They start behind existing work without activation;
offscreen workloads keep their context hosts hidden. See the
[demo guide](tools/integration/README.md#visible-demos-and-hidden-tests) for the
commands and separate window-capture/readback verification.

## Observed native evidence

S0/S1 foundation acceptance is scoped to Windows DX12 and desktop OpenGL 4+
through WGL. The independently reviewed S2/S3 fixture closure uses those same
two named routes with Bases `ee640d5`, archived dirty-source inputs, shader
route/capability facts, readbacks, native diagnostics, and inspected Surface
captures. It does not establish a general renderer API, S4/S5 GPU behavior, or
support for untested targets; see the [S2/S3 named-Windows closure](documents/s2-s3-closure.md).

The [S4 CPU closure record](documents/s4-cpu-closure.md) binds the independently
reviewed external CPU reference to product `e4ec252`. It builds against the public RenderScene and
CPU-feature APIs with its own manifest and lock. It compares ordinary and
ECS-shaped adapters, preserves an old snapshot across an update, and uses
renderer-issued opaque handles, issuer-bound typed storage, and a controlled
CPU phase collector. The static `CpuFeature` trait validates snapshot version,
scene, issuer, handle, and contribution identity through structured errors. This
is a bounded provisional CPU facade, not a FramePipeline or GPU API: it creates
no device, Graph work, submission, culling, indirect work, or backend support
claim. Its compile-fail negative confirms that the external dependency cannot
name Renderer-private Graph state.

The [S5 Forward record](documents/s5-forward-closure.md) maps the bounded
pipeline/preparation/residency contract to focused checks and six inspected
DX12/WGL Forward, device-isolation and Surface attempts. Exact source archives,
locks, raw artifacts and the independent reviewer report are retained in the
[S5 evidence index](documents/evidence/s5-forward-candidate/index.json).
The [measurement baseline](documents/s5-measurement-baseline.md) records the
separate release measurement procedure and its limits. The
[S6 typed Forward record](documents/s6-typed-forward-closure.md) records the
separate bounded external composition acceptance: eleven offscreen submissions
per backend, pending replacement/refusal coverage, and a two-frame native
Surface resize matrix on DX12 and WGL. Its retained inputs and outputs are in
the [S6 evidence index](documents/evidence/s6-typed-forward-candidate/index.json),
and its release baseline is recorded separately in
[the S6 measurement baseline](documents/s6-measurement-baseline.md). The
[S7 Deferred acceptance record](documents/s7-deferred-closure.md) records six
reviewed native attempts for the named DX12/WGL subset. The
[S8 measured closure](documents/s8-measured-closure.md) records the final
source-bound DX12/WGL material-heavy Forward evidence, lifecycle proof, and
affected full-workspace regression. Its CPU/API elapsed measurements are not GPU
durations: cold/warm/replacement results are 89.976/91.598/92.166% on DX12 and
91.714/91.779/92.429% on WGL. This closes only the selected private-index
strategy, not streaming, LOD, GPU culling, indirect work, browser, Metal,
Vulkan, an SRP, Forward+, shadows, PostFX, or full PBR.

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
| Renderer `feature` module | Public bounded CPU facade and demonstrated typed opaque-Forward Feature seam; Graph/RHI authority remains private |
| Renderer `pipeline` module | Public bounded Forward pipeline/composition and controlled phase/access facade; Graph internals remain private |

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
seams are exercised early in S4/S5; S6 closes its demonstrated subset through an
external Rust Feature, external FramePipeline, and custom material cooperating in
the same real frame. Code Nodes are planned S7 functionality; complete authoring
breadth does not block the MVP. Optimization follows measurements retained from
the earlier stages.

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
- [S4 CPU closure record](documents/s4-cpu-closure.md)
- [S2/S3 named-Windows closure](documents/s2-s3-closure.md)
- [S6 typed Forward closure record](documents/s6-typed-forward-closure.md)
- [S7 Deferred closure record](documents/s7-deferred-closure.md)
- [S8 measured closure record](documents/s8-measured-closure.md)
- [S6 measurement baseline](documents/s6-measurement-baseline.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [Architecture decisions](documents/adr/README.md)
- [Migration inventory](documents/foundation-migration.md)
- [Shader package guide](crates/shader/README.md)
- [Material package guide](crates/material/README.md)
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
