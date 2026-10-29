# Fluxel Rendering

`fluxel-rendering` is Fluxel's host-agnostic Rust rendering kernel. Its typed
render graph and renderer layer prepare portable drawing work and execute it
through the separate `fluxel-rhi` contract without owning an application host.
Cross-frame logical asset identity, platform services, runtime packaging, and
language SDKs belong to adjacent Fluxel repositories described in the
[Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).

The kernel accepts a host-supplied surface target, resize information, resource
bytes, scene or canvas updates, and explicit render calls. It does not create a
window, run an event loop, read files or URLs, collect input, provide storage,
audio, video, networking, or decide when the next frame runs. Browser and
mini-game adapters drive its WASM form; native hosts drive its library form.

## Workspace

The table below describes the rendering workspace and its dependency on the
separate RHI repository. The next work reuses the Bases asset core and establishes
Renderer-internal Graph execution against stable RHI before the Shader/Material
and Renderer MVP. Older interfaces are migration inputs.

The current workspace contains these crates:

The renderer and hardware examples pin one RHI Git revision. RHI-specific
diagnostic scripts use a sibling `fluxel-rhi` checkout when run locally.

| Crate | Responsibility |
| --- | --- |
| `fluxel-rendergraph` | GPU-work planning: typed declarations, versions, validation, dependencies, culling, lifetimes, and immutable graph plans. It directly uses RHI portable contracts. |
| `fluxel-renderer` | RenderScene and FramePipeline policy: scene preparation, material/shader resolution, culling, sorting, and graph construction. |
| `fluxel-rendering-wasm` | Non-published wasm-bindgen capsule for the named WebGL2 and WebGPU browser proofs; JavaScript remains the RAF and DOM lifecycle owner. |

The target has three principal crates: `fluxel-shader`, `fluxel-material`, and
`fluxel-renderer`. Graph and the replaceable `FramePipeline` SPI move inside
Renderer. Shader is reusable compilation infrastructure; Material consumes
Shader and owns material semantics. Neither depends on Renderer. This
documentation update does not create crates or migrate the existing code.

The optional `gpu-upload` slice publishes immutable GPU snapshot generations
after native uploads complete. It can lower an insertion-ordered `DrawList`
and its matching ready indexed snapshots into an owned, opaque, device-affine
`RenderPacket`, then execute all of its legacy unlit indexed draws through one
compiled graph, raster pass, and submission. Existing single-draw paths also
cover closed textured, vertex-color, and fixed-Lambert recipes.
Each legacy-unlit packet draw may carry an independent affine model-to-world
placement; the renderer lowers it into the existing camera/material uniform
contract without exposing a general scene or pipeline API.
The default build remains the portable headless domain model.

The fixed renderer and RHI slices are split across their dedicated repositories.
Seven proven raster paths share one private, closed recipe mapping.

Package READMEs are the detailed user documentation published with each crate;
this file is only the workspace entry point.

The architecture is an ownership DAG, not a universal linear dependency path:
frame execution is `RenderScene -> FramePipeline -> RenderGraph -> Shader /
Pipeline + Material Runtime -> RHI`, while material compilation is
`MaterialGraph -> Material IR -> evaluation WGSL / ShaderCompositionRequest
-> Shader System + pass template -> ShaderArtifact / Pipeline
requirements`. Renderer, RenderGraph, shader/pipeline code, and material
runtime may each reuse RHI portable contracts where needed; normal frame work
continues to preserve RenderGraph pass-local resource authority.

Material owns material functions, parameters/uniform schemas, and call-code
generation; Shader owns complete programs/modules, entry points, compilation,
metadata, and reflection, without importing Material IR. A Graph pass describes
GPU work and resource dependencies, and may use multiple prepared pipelines.
Residency uploads use renderer coordination and share RHI dependency/completion
tracking with Graph consumers. See the [architecture boundaries](documents/design-overview.md).

## Released source dependency

Git consumers must pin a tag or revision rather than follow `main`.
`fluxel-rhi` has its own stable releases. The following records this workspace's
older integration pin, not a recommendation for the new architecture. The first
implementation milestone must select an exact stable RHI dependency and migrate
affected manifests, locks, examples, and real consumers with integration evidence:

```toml
fluxel-rendergraph = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.16.0" }
fluxel-rhi = { git = "https://github.com/fluxel-project/fluxel-rhi", rev = "97a003fb11e30bd97ec62efe82e031f7bcf26ab5" }
fluxel-renderer = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.16.0" }
```

See [RELEASING.md](RELEASING.md) for the rendering workspace release gate.

## Documentation

- [Workspace architecture](documents/design-overview.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [RHI design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md)
- [Renderer design](documents/design-renderer.md)
- [Implementation plan](documents/version-plan.md)
- [Architecture decisions](documents/adr/README.md)
- [Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
- [RenderGraph guide](crates/rendergraph/README.md)
- [RHI guide](https://github.com/fluxel-project/fluxel-rhi)

## Quick verification

Rust MSRV is 1.87 (edition 2024).

These commands verify the checked-out baseline; by themselves they do not close
any future version's real-platform or cross-repository evidence gate.

```sh
cargo +1.87.0 test --workspace --all-targets --all-features --locked
cargo +1.87.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
```

The graph compiler and `TestRhi` are CPU-only. `fluxel-rhi` provides
headless DX12 and Vulkan fixed Raster, Compute, and Copy execution on Windows
when the native loader and driver are available. CI is a compile/link gate;
real-GPU conformance remains an explicit local release gate.

On a clean Windows `x86_64-pc-windows-msvc` checkout, the release-only GPU
gate is:

```powershell
./scripts/conformance.ps1
```

It runs every workspace ignored fixture, injects the checked-out commit through
`FLUXEL_TEST_COMMIT`, and preserves a manifest and complete log under
`target/conformance/<sha>/`, including failures. Successful release artifacts
are identified by the release tag and retained evidence records described in
[RELEASING.md](RELEASING.md); the release process does not create a GitHub Release.

## Roadmap

The organization
[roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md) is the
only stage/status authority. This README records only the workspace's current
supported paths and recommended entry points.

Reuse the existing Bases asset core, establish internal Graph/stable RHI
integration, then deliver Shader/Material and a data-oriented Renderer MVP.
The MVP includes built-in Mesh plus a minimal Feature in an independent Rust
crate, proving batch updates, pipeline-phase participation, and graph resource
declarations. Broader material/pipeline/Feature work and measured performance
and resource strategies complete the active implementation scope.
Capture/replay, Canvas 2D and basic text, declarative UI, a Blender plugin,
and JavaScript wrappers are undecided, with no retained design or delivery order.
The executable breakdown is
the [implementation plan](documents/version-plan.md). The sole normative
RHI API architecture source is [RHI design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md);
rustdoc and contract tests define descriptor-level detail. The
[workspace architecture](documents/design-overview.md) records cross-layer
invariants and integration boundaries.

The `0.15` retained historical baseline closures are:

- a fixed resource floor on DX12, Vulkan, WebGPU, and WebGL2;
- fixed compute plus storage-buffer paths on DX12, Vulkan, and WebGPU;
- fixed storage-texture writes on DX12, Vulkan, and WebGPU, and fixed
  storage-texture reads on Vulkan only;
- structured, zero-side-effect WebGL2 rejection for Compute and every storage
  resource operation;
- a GL-family execution path behind one internal three-layer boundary
  (`api/` → `state/` → `compat/`), covering desktop GL 4.x over WGL, GLES 3.x
  over EGL, and browser WebGL2, as a peer of DX12/Vulkan rather than a second
  public graphics API; the GLES half of this was proven against a GLES 3.1
  implementation driven through EGL, not against a physical GLES 3.1 device,
  and the desktop half's profile floor is enforced from a spec-derived table
  rather than from a measured one;
- headless DX12/Vulkan execution and real-GPU conformance;
- visible Windows DX12/Vulkan surface lifecycle and bounded completion;
- the retained RGB scene on the named Chrome/WebGL2 target; and
- the same scene on the named Chrome/WebGPU target, including canvas epochs,
  bounded completion, controlled destroyed-device recovery, and async disposal.

This is a capability matrix, not a portable feature tier: DX12 storage-texture
read fails closed on the observed driver, and WebGPU storage-texture read is
not currently promised. WebGL2 does not emulate Compute or storage. Unsupported
graph declarations expose a typed `UnsupportedCapability` requirement with the
observed capability snapshot, so a host can choose a fallback before context or
resource work begins.

For application integration, a `FramePipeline` resolves material/shader
variants, builds a typed RenderGraph, and records it directly through the
selected RHI device's portable contract. Per-frame imports can bind RHI
resources directly; backends and native synchronization remain private.
RenderGraph analyzes logical transient lifetimes while RHI realizes allocation,
so callers must never depend on a transient's physical identity. Details are in
[the RenderGraph design](documents/design-rendergraph.md), [the RHI design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md),
and [ADR-0009](documents/adr/0009-resource-floor-and-reuse-safety.md).

The corresponding supported-target limits and release evidence follow the
ecosystem evidence policy and tagged release records. The historical resource floor remains a
closed set of recipes; it does not add a general shader, pipeline, descriptor,
or resource-builder API.

### Unscheduled optimizations

- [ ] Batch compatible render packets while preserving declared ordering.
- [ ] Lower compatible repeated draws as instances when the selected recipe
  permits it.
- [ ] Reuse compatible renderer-side pipelines and bindings across a frame.
- [ ] Add multi-queue scheduling, parallel command recording, recording caches,
  transient resource aliasing, or more aggressive GPU scheduling and memory
  optimization only after representative benchmarks or profiling demonstrate a
  concrete need.

### Out of scope

Durable logical asset identity and CPU caching stay in `fluxel-bases`.
General file/URL loading, game scene graphs/ECS, gameplay animation state,
and host policy remain outside Renderer. Renderer owns rendering views/cameras,
visibility, sorting, batching, and selected rendering Features such as GPU
skinning, terrain, water, particles, and area-light integration. JS adaptation
is outside the active Rust implementation scope.
The separate `fluxel-rhi` repository owns backend maintenance and the
portable presentation boundary. This workspace owns renderer-private per-device
GPU residency needed to prove renderer behavior; it does not own a general
platform runtime, host services, or a public asset cache API.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
