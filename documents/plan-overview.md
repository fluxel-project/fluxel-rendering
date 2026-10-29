# Fluxel Rendering Product Plan

> Status: active target direction. The organization [roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
> owns stage order; [version-plan.md](version-plan.md) defines executable work
> packages and acceptance. Target architecture does not imply implementation.

## Product direction

Fluxel provides an extensible, data-oriented real-time renderer without requiring
an ECS, gameplay scene model, host, or fixed rendering strategy. Renderer owns
views, visibility, preparation, sorting/batching, resource coordination, and frame
execution. Logical assets remain in Bases; portable GPU execution remains in RHI.

The target has three principal crates: `fluxel-shader`, `fluxel-material`, and
`fluxel-renderer`. Graph and FramePipeline are renderer-owned modules, with a
controlled public Rust facade for pipeline/Feature implementations. Shader and
Material are independent of Renderer; Material consumes Shader's generic source
and interface vocabulary. See [workspace architecture](design-overview.md).

Material emits evaluation WGSL and logical interface contributions. Renderer/
FramePipeline combines them with pass templates and entry points into a complete
`ShaderCompositionRequest`. Shader composes, validates, and reflects the logical
program. RHI selects the device route and performs backend lowering and native
compilation. Prepared programs/variants can be reused across frames; this
compilation flow is not a mandatory per-frame call chain.

## Delivery and first product milestone

| Work packages | Product outcome |
| --- | --- |
| S0 Foundation + S1 Graph | Pinned Bases/RHI integration, internal Graph execution, reusable plans, safe overlapping frames |
| S2 Shader | Independent WGSL programs compile, reflect, and execute through real RHI routes |
| S3 Material + S4 RenderScene | Minimal custom material composition and batch-oriented scene snapshots; independent tasks may overlap after S2 |
| S5 Forward integration | Multiple meshes/materials render through the full path; early external consumers shape phase/Feature seams |
| S6 Extension MVP | External Feature, external FramePipeline, and custom material cooperate in one frame; first product MVP closes |
| S7 Rendering expansion | Code Nodes, selected advanced rendering, a second pipeline, materially different Features |
| S8 Measured strategies | Workload-driven improvements with bounded memory and preserved lifecycle correctness |

S0/S1 form one foundation closure. Direct RHI fixtures and prepared shaders make
Graph execution possible before Shader/Material authoring. S3 needs only the
functions, parameters, and call composition consumed by the first scene. Full
MaterialGraph breadth and Code Node authoring experience do not block S6; Code
Nodes remain planned S7 functionality. Feature interfaces start during S4/S5;
S6 proves their joint use rather than adding a late plugin framework.

Measurements begin with runnable workloads. S8 does not postpone basic batching,
safe lifetime handling, or bounded frame storage. Rust crate plugins are in scope;
dynamic DLL loading and a stable plugin binary ABI are unscheduled.

## MVP acceptance

```text
Bases snapshots + batched scene changes
  -> RenderScene / RenderView + built-in Mesh + external Feature
  -> custom Material + Shader preparation + external FramePipeline
  -> visibility / phase ordering / batching -> internal Graph -> stable RHI
  -> observed textured 3D output in the same frame
  -> completion-governed reuse and retirement across overlapping frames
```

An independent Rust consumer uses only the public facade. Adding its typed Feature
cannot require a core object-type switch, private Graph internals, or independent
GPU submission. Ordinary and ECS-style adapters share the batch boundary without
an ECS dependency. Stabilize only the exercised extension subset; a second
materially different Feature must exercise broader cross-pass/shading contracts
before those stabilize.

Every bound material contributes its reflected, concrete resource-use manifest.
Compatible resource-instance changes reuse a Graph plan after validation; changed
hazards, dependencies, or access/layout constraints require recompilation or
refusal. A changed texture identity alone does not imply new graph topology.

Persistent residency covers textures, meshes, and streamed tiles. Frame Data Upload
covers transforms, bone palettes, and other frame-owned inputs. They may share
transfer machinery, with explicit producer dependencies, visibility/state
transitions, and last-use retention. Conditional use of pending content differs
from observed completion. GPU skinning, simulation, culling, and other frame GPU
work remain Graph passes.

Each closure retains exact revisions/locks, focused tests, structured refusals,
and the affected real-target evidence. Partial/indeterminate execution retains
affected resources without publishing guessed state. RHI stability does not
certify every optional feature/backend; its [open work](https://github.com/fluxel-project/fluxel-rhi/blob/main/TODO.md)
and [verification limits](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
remain explicit.

## Undecided and future scope

Capture/replay, Canvas 2D/basic text, declarative UI, Blender tooling, and JavaScript
wrappers remain undecided, with no retained API design or delivery order. The old
standalone Graph, fixed renderer, and WASM capsule are migration inputs.

A future Rust + TypeScript game engine needs separate planning for Game Scene,
gameplay/script execution, the game loop, bindings, and hot reload. Adapters may
map external identity to RenderIds and submit batches. `fluxel-jsbridge` adapts
browser/mini-game lifecycle; it does not own a future gameplay VM. Those systems
are not rendering MVP gates.
