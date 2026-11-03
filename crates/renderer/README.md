# Fluxel Renderer

Renderer is being rebuilt following the organization architecture. Narrow Copy,
offscreen and Surface Raster, and supported Compute fixtures serve the S0/S1
consumer. The public S4 subset adds CPU-only RenderScene changes, immutable
snapshots, and a bounded CPU feature facade. FramePipeline, preparation, and GPU
scene execution remain private scaffolds.

```text
renderer/
  Cargo.toml
  README.md
  LICENSE-APACHE
  LICENSE-MIT
  src/
    lib.rs
    renderer/mod.rs
    renderer/cache.rs
    renderer/copy.rs
    renderer/raster.rs
    renderer/compute.rs
    renderer/read_set.rs
    renderer/readback.rs
    renderer/error.rs
    renderer/submission.rs
    renderer/validation.rs
    renderer/shutdown.rs
    render_scene/mod.rs
    feature/mod.rs
    pipeline/mod.rs
    preparation/mod.rs
    graph/mod.rs
    graph/definition.rs
    graph/compiler.rs
    graph/binding.rs
    graph/access.rs
    graph/state.rs
    graph/execution.rs
    graph/copy.rs
    graph/raster.rs
    graph/compute.rs
    graph/read_set_raster.rs
    residency/mod.rs
    residency/upload.rs
    frame_data/mod.rs
    tests/README.md
    tests/graph/{access,binding,compiler,compute,copy,raster,state}.rs
    tests/renderer/{contracts,lifecycle,raster,read_set_raster,surface,shutdown}.rs
  examples/README.md
  tests/README.md
```

| Private module | Responsibility |
| --- | --- |
| `renderer` | Entry facade, service ownership, and frame lifecycle coordination |
| `render_scene` | Public render-only storage, batched changes, views, bounds, and immutable snapshots |
| `feature` | Prepared feature data and declared phase contributions |
| `pipeline` | Frame policy, controlled context, phases and outputs |
| `preparation` | Visibility, sorting, batching, draw preparation and concrete bindings |
| `graph` | Resource versions, definedness, hazards, plans and direct RHI execution |
| `residency` | Persistent asset realization and completion-safe GPU retention |
| `frame_data` | Per-frame transient uploads with their own lifetime |

The fixture facade provides `submit_surface_raster` and `complete` for a
caller-configured RHI Surface. It retains the exclusive image lease and native
execution outcome until an accepted upload/raster/present sequence and covering
completion evidence are observed. This narrow path does not introduce a public
Graph, surface factory, or general frame API.

These are ownership boundaries, not a set of completed or stabilized subsystems.
Only add further files and dependencies when an implemented consumer needs them.
Future public exports must preserve Graph/compiler/scheduler privacy.

## Current RenderScene subset

`RenderScene<R>` is a CPU-only render-domain store whose record resource bundle
is chosen by the consumer. `RenderRecord<R>` keeps that bundle alongside a finite
transform and bounds; `R` must be cloneable so immutable snapshots can retain it.
`RenderResources<G, M>` is a convenience bundle for one geometry and one material
asset reference, not a per-scene restriction or a heterogeneous Feature registry.
Views are separate generational identities.

`RenderChangeSet` applies inserts, updates, and removals transactionally. Mutable
records live in dense storage; removal compacts that storage while generational
identities remain stable and affected slot ranges are reported. It rejects foreign
or stale identities, conflicting writes in one batch, invalid
transforms/bounds/views, capacity overflow, and identity/version exhaustion.
Successful changes publish an immutable `FrameSnapshot` with exact dirty ranges.
Snapshots retain their prior values through later updates and can be read while
the mutable scene advances. This API performs no preparation, Graph declaration,
RHI call, GPU residency, submission, or completion management.

`examples/external-renderer/` is the independent CPU reference consumer. Its
ordinary object-map and ECS-shaped adapters produce equivalent public snapshots;
their typed color and label data are application-owned and explicitly tied to
snapshot versions. It implements the public static `CpuFeature` trait using
renderer-issued opaque handles, issuer-bound `TypedFeatureStorage`, controlled
`FeatureContext`, and the reference `CpuPhase` collector. The facade rejects
foreign scenes/issuers, handles, versions, mixed/replayed contributions, and
invalid prepared data with structured errors. It is provisional and CPU-only:
it does not implement `RenderFeature` or `FramePipeline`, declare Graph work,
or provide GPU evidence. Its compile-fail negative verifies that the external
consumer cannot name Renderer-private Graph state.

The target uses batch-level Rust traits with generics and monomorphization for
Feature and frame-policy composition. External crates supply concrete local
visibility, geometry, ordering, batching and deformation providers. Applications
assemble them through typed tuples or ordinary aggregates at compile time;
trait-object dispatch, erased registries and runtime type erasure are excluded.
Runtime enablement uses state, while explicit enum branches select known types.
See [ADR-0026](../../documents/adr/0026-static-extension-composition.md). Several providers can coexist; core does not require a
single global algorithm, fixed vertex struct, universal SoA layout or CPU DrawList.
The phase owner coordinates ordering across Features. See
[the Renderer design](../../documents/design-renderer.md) for the detailed target
and [ADR-0025](../../documents/adr/0025-composable-rendering-policies.md) for the
decision. This describes future contracts, not the fixture API's current ability.

## Frame flow and dependency rules

```text
render_scene + feature
    -> pipeline policy + preparation
    -> graph declarations, planning, and RHI execution
    -> completion evidence -> residency / frame_data retirement
```

`renderer` coordinates these boundaries; it does not implement another execution
path. `pipeline` chooses passes and outputs. `feature` contributes data and work
through that path. `graph` knows resource accesses and dependencies, not material
models or rendering algorithms. `residency` owns persistent GPU realizations;
`frame_data` owns transient frame inputs. Their retirement follows RHI evidence.

The following subdivisions describe where implementation will grow. They are
not empty files to create in advance or public API commitments.

| Boundary | Add private submodules when exercised |
| --- | --- |
| `renderer` | Configuration, services, frame lifecycle, structured errors |
| `render_scene` | Scene storage, changes, views, snapshots |
| `feature` | Extraction, prepared contributions, phase participation |
| `pipeline` | Controlled context, phases, frame outputs |
| `preparation` | Visibility, sorting, batching, draw packets, binding manifests |
| `graph` | Declarations, validation, dependencies, plans, recording, completion |
| `residency` | Realization keys, uploads, last use, retirement |
| `frame_data` | Frame allocation, uploads, completion-safe reuse |

## Material and Shader integration

```text
fluxel-material: MaterialGraph -> private IR -> generic source + interfaces
pipeline / preparation: add pass template, entry points and geometry requirements
fluxel-shader: complete composition request -> logical program + reflection
preparation: resolve concrete bindings and resource access manifest
graph: validate declared accesses and execute through RHI
```

Material owns parameters, functions, evaluation code, and material semantics.
Shader owns logical composition, reflection, variants, and program identity.
Neither depends on Renderer. Shader does not receive Material IR. Renderer owns
their integration and concrete resources; RHI owns native compilation and GPU
pipeline objects. There is no Renderer-local `material/` or `shader/` subsystem.
Material/Shader dependency edges will be added with real implementations rather
than scaffolds. The current Copy path uses only Bases and RHI.

S0/S1 fixture entry points and fixed pass assembly are provisional consumers.
Later preparation replaces those adapters while retaining Graph validation,
device execution planning and completion-based ownership. Copy now builds a
logical graph and concrete binding manifest for each frame's byte extent,
checks compatibility with the cached topology, and records the current frame's
ranges. This reuses one device submission plan across different buffer sizes;
it does not reuse the frame's allocations or skip binding validation.

## Examples and tests

`examples/` is reserved for small runnable consumers of public Renderer APIs.
Shared example helpers belong under `examples/common/` only after duplication
appears. Complete Host/RHI lifecycle composition and retained backend evidence
belong in the independent root `examples/rendering-integration/` consumer.

Crate-root `tests/` is reserved for integration tests of public contracts. Unit
tests live under `src/tests/`, grouped by the production component they cover.
Production modules reference those files with test-only module declarations so
private access and logical test-module paths remain intact. No placeholder executable or test is
included in the crate-local examples directory. The independent fixture consumer
and private logic tests establish only their exercised subset.

Material owns material semantics and emits generic Shader contributions. Shader
owns complete logical program composition and reflection. Renderer prepares
concrete resources and frame work; RHI owns device routes, physical resources,
recording, submission, completion and presentation. Host and JSBridge retain
platform lifecycle responsibilities.

The root integration consumer uses `Renderer::submit`, `submit_raster`,
`submit_compute`, and `readback`. The Raster/Compute entry points deliberately
accept a fixed fixture schema and already-created public RHI pipelines; they
are not Material/Shader composition or a general frame extension API.
It retains three immutable asset generations per workload and their
original native outcomes until exact covering completion. Copy checks each candidate graph against a cached device plan; the other fixture
caches still use byte length and, for Raster, target extent. These are provisional
fixture caches, not persistent asset residency. All four submit paths share one outcome classification. Only an accepted
execution without recording or execution errors returns a `Submission`.
`RetainedExecution` reports the original status and both error channels while
keeping the original native outcome and affected owners. Explicit no-effect
rejections release their local owners; partial and unknown effects remain
retained until a successful idle shutdown. Every observed `DeviceLost`, including
allocation, upload mapping, begin, recording and finish, invalidates reusable
plans and rejects new work on that Renderer. Begin is always paired with finish,
including after a recording failure. Unproven pending work remains
conservatively retained on Drop. Shutdown tests are wired into the crate and
cover idle success, failure/cancellation and device loss. Completion-wait loss
retains uncompleted owners and disables new work; a later readback/map failure
does not undo a successfully observed completion. The current logical/native
evidence and named target limits are mapped in the
[foundation closure review](../../documents/s0-s1-closure.md).
Follow [the implementation plan](../../documents/version-plan.md) and the
[Renderer design](../../documents/design-renderer.md) as targets, not completion
claims. The RHI interface has not been changed by this reset.

The fixed `submit_read_set_raster` consumer multiplies a 1-by-1 RGBA8 texture by
one uniform tint and reads back an 8-by-8 target. Its 20-byte immutable payload
contains four little-endian floats followed by one texel. The 256-byte destination
window may change between frames. Each candidate Graph declares the actual
16-byte physical uniform access at that window's offset; its owner retains the
allocation descriptor, texture state and resource instance identities. The device
submission topology is reused only after the candidate Graph's dependencies,
layouts and access contracts remain compatible. Graph records the actual upload,
uniform/texture reads, target write and readback. Padded upload storage satisfies
the selected DX12 texture-copy alignment without changing the logical asset.
These fixture inputs are not a public Material, Shader or frame-pipeline model.
