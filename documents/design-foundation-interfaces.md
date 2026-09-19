# Fluxel rendering foundation contract

> Status: normative cross-layer contract for the 0.16–0.20 foundation train.
>
> Exact RHI types, signatures, errors, and semantics are defined only by
> [design-rhi.md](design-rhi.md) and [RHI design modules](rhi-design/).
> Exact graph and capture interfaces are defined by
> [design-rendergraph.md](design-rendergraph.md) and
> [design-capture-replay.md](design-capture-replay.md). Release order and
> evidence gates are defined by [version-plan.md](version-plan.md).

## 1. Authority

This document defines cross-layer ownership only. It does not duplicate or
amend RHI, RenderGraph, or capture/replay API definitions.

Authority order is:

1. RHI design modules and design-rhi.md: portable RHI API and semantics.
2. design-rendergraph.md: graph authoring, compilation, instantiation, bridge.
3. design-capture-replay.md: capture/replay and artifact/runtime ownership.
4. version-plan.md: release sequence, validation, and evidence.
5. This document: cross-layer integration.

Higher authority resolves conflict. Old stages, fixed recipes, examples, ADRs,
and backend details cannot create an API commitment.

## 2. Layer model

~~~text
MaterialGraph / compiled material artifacts
        ↓
Renderer policy / CustomPipeline / RenderFeature
        ↓
RenderGraph authoring, compilation, instantiation
        ↓
portable RHI recording, submission, completion, presentation
        ↓
backend-private lowering
        ↓
DX12 | Vulkan | Metal | WebGPU | GL family
~~~

MaterialGraph owns asset-level graph lowering, shader payload/provenance,
parameters, domains, and variant requirements. It never owns GPU realization,
frame scheduling, graph resources, or native objects.

Renderer and CustomPipeline own scene/view preparation, variant selection,
feature order, and logical pass/resource policy. They cannot bypass RenderGraph
to manipulate RHI/native objects.

RenderGraph owns versions, declared uses, definedness, DAG, roots, culling,
lifetime, logical scheduling, and transient packing decisions. It owns no
scene/assets/material policy, native states, or artifact policy.

RHI owns device-affine realization, command recording, actual uses, submission,
completion, presentation, retirement, and observability. It owns neither
renderer/graph policy nor capture dependency closure, artifact storage, or
ReplayRuntime.

## 3. Device/context identity and assets

Every public GPU-affine value belongs to opaque Device/Context identity plus
generation. GPUDevice, WebGL2RenderingContext, GPUBuffer, GL names, Vulkan
handles, Metal objects, DX12 interfaces, and private leases remain backend
details.

Every operation combining GPU-affine values validates identity/generation before
backend work. Loss terminates that complete identity. A new request creates a
new identity/generation domain; transparent generation increment, handle revival,
automatic transplantation, and implicit cross-device interoperation are
forbidden. Old handles remain invalid.

Platform adapters handle canvas/surface, resize, foreground/background, and
loss/restore without introducing browser session/token resource types.

RHI owns device-generation-affine objects and completion-safe retirement.
RenderGraph owns per-frame logical declarations and lifetime reasoning.
fluxel-bases owns durable asset identity, content generation, loading, and cache
policy. Renderer may retain per-device realization keyed by asset identity,
content generation, and DeviceIdentity. A pass receives resolved frame inputs,
never an AssetStore.

## 4. MaterialGraph, CustomPipeline, and RenderGraph

MaterialGraph and RenderGraph are distinct graphs. A MaterialGraph artifact
describes how a surface or screen effect is calculated. It may carry shader
payload/provenance, interface/parameter layout, domain, vertex/target, and
capability requirements, but does not create graph resources or record commands.

CustomPipeline is Renderer policy: it chooses material variants, features,
logical resources, and pass topology, then authors graph passes. The fixed
renderer is only one consumer; its recipes must use the same artifact, binding,
pipeline, graph, and RHI seams as future material/custom-pipeline work.

RenderGraph does not know GBuffer, SceneColor, bloom, UI, or material meanings.
It validates generic resource/version/use semantics and lowers only through the
standard graph-to-RHI bridge.

## 5. Declared and actual use

RenderGraph declares:

~~~text
versions; declared uses; definedness; dependencies; roots; culling
lifetime; lanes; imports/exports; readback/present relation; logical packing
~~~

RHI records:

~~~text
actual command/resource uses; RecordedWork; submission dependencies
completion; retirement; presentation outcome; backend-private lowering
~~~

The bridge is the sole declared-versus-actual coverage check. A normal recorder
does not accept graph declarations, and graph execution cannot receive arbitrary
device objects that hide use. Imports/exports use portable semantic state,
ownership/lease, and definedness; no native layout/state/barrier/fence/queue is
exposed. CompiledGraph is immutable logical data; instantiation resolves current
imports and frame attachments without retaining native objects.

## 6. Submission, presentation, capture

RHI owns recording, acceptance, execution lowering, completion, retirement,
configuration, acquire, and present outcome. Accepted submission, GPU completion,
and present outcome are distinct. Frame attachments are presentation semantics,
not drawable TextureViews. Work using an acquired frame must be in a valid
present plan. Lanes are device facts and do not promise native queue count or
hardware overlap.

Normal replay truth is:

~~~text
object-definition graph + snapshots/upload mutations
+ PortableCommandIR with actual uses
+ submission/presentation relations + external fixtures
~~~

PortableCommandIR is replay truth. FrozenGraphIR is graph provenance for
explanation, declared-use checks, and visualization; recompilation is comparison
mode, not normal replay. RHI provides reconstructable semantics and readback,
but not artifact schema, dependency closure, snapshot policy, negotiation, or
ReplayRuntime. Observers cannot change execution.

## 7. Capability, validation, and order

Capability is instance data: available adapter facts, requested requirements,
and enabled Device facts are distinct. Unsupported behavior returns structured
failure; no hidden shader, CPU round trip, format conversion, copy, or weakened
synchronization is allowed. Diagnostics/statistics are observational only.

~~~text
0.16 RHI + DX12/Vulkan/Metal
0.17 WebGPU + GL family + ecosystem RHI integration
0.18 RenderGraph semantic core
0.19 RenderGraph execution/diagnostics closure
0.20 portable capture/replay closure
then scene/material/renderer/CustomPipeline work
~~~

Later work cannot establish earlier closure. Public MaterialGraph, CustomPipeline,
and renderer-plugin APIs remain post-foundation, though this contract protects
their integration seams.

## 8. Prohibitions

Do not introduce browser session/token models, native escape hatches, implicit
cross-device use, fixed recipes as public ABI, hidden graph resource use, graph
trace as normal replay input, execution-changing capture/statistics, undocumented
fallback, or fast-iteration deferral of foundational semantics. A new RHI API
must be designed in the RHI authority, never sketched here.
