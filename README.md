# Fluxel Rendering

`fluxel-rendering` is Fluxel's host-agnostic Rust rendering kernel. Its typed
render graph, native RHI boundary, and renderer layer provide portable drawing
semantics and GPU execution without owning an application host. Cross-frame
logical asset identity, platform services, runtime packaging, and language SDKs belong
to adjacent Fluxel repositories described in the
[Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).

The kernel accepts a host-supplied surface target, resize information, resource
bytes, scene or canvas updates, and explicit render calls. It does not create a
window, run an event loop, read files or URLs, collect input, provide storage,
audio, video, networking, or decide when the next frame runs. Browser and
mini-game adapters drive its WASM form; native hosts drive its library form.

## Workspace

The table below describes the retained historical workspace and the completed
RHI baseline. The next implementation plan starts the shader assembly and
material-system route; older higher-level interfaces are not the replacement
contract.

The workspace is organised around four crates:

| Crate | Responsibility |
| --- | --- |
| `fluxel-rendergraph` | GPU-work planning: typed declarations, versions, validation, dependencies, culling, lifetimes, and immutable graph plans. It directly uses RHI portable contracts. |
| `fluxel-rhi` | Portable GPU execution: device-affine resources, capabilities, recording, submission, completion, and presentation; native backends remain private. |
| `fluxel-renderer` | RenderScene and FramePipeline policy: scene preparation, material/shader resolution, culling, sorting, and graph construction. |
| `fluxel-rendering-wasm` | Non-published wasm-bindgen capsule for the named WebGL2 and WebGPU browser proofs; JavaScript remains the RAF and DOM lifecycle owner. |

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

The fixed renderer and RHI slices are split into single-responsibility modules.
Seven proven raster paths share one private, closed recipe mapping.

Package READMEs are the detailed user documentation published with each crate;
this file is only the workspace entry point.

The architecture is an ownership DAG, not a universal linear dependency path:
frame execution is `RenderScene -> FramePipeline -> RenderGraph -> Shader /
Pipeline + Material Runtime -> RHI`, while material compilation is
`MaterialGraph -> Material IR -> Shader System -> ShaderArtifact / Pipeline
requirements`. Renderer, RenderGraph, shader/pipeline code, and material
runtime may each reuse RHI portable contracts where needed; normal frame work
continues to preserve RenderGraph pass-local resource authority.

## Released source dependency

The workspace releases its three publishable crates together. Git consumers must pin the
release tag rather than follow `main`:

```toml
fluxel-rendergraph = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.16.0" }
fluxel-rhi = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.16.0" }
fluxel-renderer = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.16.0" }
```

`v0.16.0` and each publishable package's `0.16.0` version identify the same workspace
release. See [RELEASING.md](RELEASING.md) for the release gate.

## Documentation

- [Workspace architecture](documents/design-overview.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [RHI design](crates/rhi/documents/design-rhi.md)
- [Portable capture/replay design](documents/design-capture-replay.md)
- [Renderer design](documents/design-renderer.md)
- [Implementation plan](documents/version-plan.md)
- [Architecture decisions](documents/adr/README.md)
- [Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
- [RenderGraph guide](crates/rendergraph/README.md)
- [RHI guide](crates/rhi/README.md)

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
are retained on the matching [GitHub Release](https://github.com/fluxel-project/fluxel-rendering/releases).

## Roadmap

The organization
[roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md) is the
only stage/status authority. This README records only the workspace's current
supported paths and recommended entry points.

The completed RHI baseline is followed by material/shader work, a minimal
scene and Forward proof, scene preparation and a Deferred proof, Blender
authoring, preview/runtime equivalence and export, capture/replay, JavaScript,
Canvas 2D with text, and finally declarative UI. This order intentionally keeps
JavaScript before Canvas/text and declarative UI. The executable breakdown is
the [implementation plan](documents/version-plan.md). The sole normative
RHI API architecture source is [RHI design](crates/rhi/documents/design-rhi.md);
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
[the RenderGraph design](documents/design-rendergraph.md), [the RHI design](crates/rhi/documents/design-rhi.md),
and [ADR-0009](documents/adr/0009-resource-floor-and-reuse-safety.md).

The corresponding supported-target limits and release evidence are maintained
in the ecosystem roadmap and GitHub Releases. The resource floor remains a
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

Durable logical asset identity and caching, general file or URL loading, broad
scene graphs, animation, application host policy, and higher-level JS or UI
APIs belong to other Fluxel layers, principally `fluxel-bases` for logical
assets. This workspace owns backend bring-up, the minimal presentation boundary,
and renderer-private per-device GPU residency needed to prove renderer behavior;
it does not own a general platform runtime, host services, or a public asset
cache API.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
