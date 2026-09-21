> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the
> root specification, module 01, and the owning resource/recording/pipeline/
> presentation module before changing one of these feature families.

# 67. Capability-complete feature families

This module adopts the useful GPU semantics present in `wgpu-hal 30.0.1`; it
does not copy that crate's native trait shapes. It replaces every earlier
"DEFERRED", "minimal set", or "backend does not need it" exclusion for the
families named here.

## 67.1 Admission, refusal, and HAL correspondence

A public method/type says that Fluxel defines a semantic. It never says that a
particular adapter supports it. Optional use follows this mandatory sequence:

```text
public vocabulary -> capability/support query -> DeviceRequirements negotiation
-> descriptor/command validation -> native lowering -> conformance case
```

The first unsupported boundary returns `RhiErrorKind::Unsupported`; no backend
may defer that decision to a driver, browser, or an `unimplemented!()` path.
Capabilities are adapter/device/format/surface/route instance data. They may
vary by driver, browser extension, enabled feature, memory type, format, usage,
dimension, sample count, stage, and limits. Traits and enum variants are not
capability claims.

The following HAL mechanics already have exactly one Fluxel carrier and are
therefore backend-private: transitions/barriers (`command::ResourceUse`),
fences and waits (`CompletionPoint`), native queues (`SubmissionLane` and
`SubmissionPlan`), encoder recycling, native destroy, native GPU addresses,
descriptor heaps, and allocation offsets. Their absence from public API is not
an unsupported feature.

## 67.2 Module map and mandatory capability closure

| Family | Public owner | Capability / limits owner | Recording or lifecycle owner |
| --- | --- | --- | --- |
| query, clear, indirect, external copy | `command`, `resource` | `capability` | recording |
| mapping, formats, sampler, external texture | `resource`, `format` | `capability` | resource transfer |
| immediates, cache, vertex, raster, binding arrays | `pipeline`, `binding`, `shader` | `capability` | recording/pipeline |
| multiview, mesh, AS, ray tracing, cooperative matrix | dedicated public modules | `capability` | recording/submission |
| surface HDR/timing | `presentation` | presentation capability facts | presentation |
| debugger, allocator/memory reports | `diagnostics` | `capability` | device lifecycle |

Every supported capability has at least one backend lowering conformance case;
every negative capability has validation coverage proving structured refusal.
All commands derive actual `ResourceUse`; all objects participate in identity,
loss, completion-safe retirement, statistics, and tooling descriptions.

### wgpu-hal 30.0.1 correspondence

| wgpu-hal surface | Fluxel carrier | Deliberate non-copy |
| --- | --- | --- |
| `Instance` / adapter enumeration/open | `PlatformProvider`, adapter facts, `DeviceRequirements` | no native instance/adapter handles |
| Surface configure/acquire/discard/capabilities | presentation target/configuration/frame/capability facts | no drawable texture impersonation |
| Device create/map/query/cache/debugger/AS | resource, query, pipeline, diagnostics, acceleration modules | no native device address or allocator handle |
| Queue submit/wait/timestamp period | submission/completion/query facts | no native Queue/Fence |
| CommandEncoder copy/clear/query/indirect/pass operations | recorder and scopes | no transition/barrier API |
| `Features`, `DownlevelFlags`, `Limits` | `OptionalFeature`, exact support queries and limit keys/typed facts | no backend-name boolean |

The `wgpu-types 30.0.1` feature inventory is carried as follows: compression
(BC/ETC2/ASTC including sliced-3D/HDR) by per-format support; all query,
mapping, indirect, clear, sampler-address, polygon/conservative, multiview,
vertex-64, NV12/P010, external texture, pipeline cache, mesh, ray and
cooperative-matrix entries by their named families below; shader atomic/numeric/
subgroup/builtin/memory entries by `ShaderRequirements`; binding-array and
non-uniform entries by binding-array facts; display timing by generic
presentation timing; and platform external-memory entries by extension SPI.
`RG11B10Ufloat` renderability, BGRA storage, Float32 filterability/blendability,
depth32-stencil8, and adapter-specific texture capabilities are per-format
facts. This is exhaustive by semantic family, not a claim that every backend
supports every member.

## 67.3 Queries

`rhi::query` defines `QuerySet`, `QuerySetDescriptor`, and
`QueryType::{Occlusion, Timestamp, PipelineStatistics(selection)}`. Pipeline
statistics are an explicit selection bitset (vertex invocations, clipper
invocations/primitives, fragment invocations, and compute invocations), never a
claim that every native counter exists. Raster/compute scopes can begin/end a
query and write timestamps; the recorder can write timestamps and resolve a
query range into a `QUERY_RESOLVE` buffer.

Capabilities are separate for occlusion, timestamp, timestamp inside encoder,
timestamp inside raster scope, timestamp inside compute scope, pipeline
statistics, resolve mode, nonblocking resolve, timestamp period/conversion,
and valid bits where meaningful. `MaxQueriesPerQuerySet` and
`QueryResolveBufferAlignment` are explicit limits; missing or invalid values
fail closed instead of selecting a hidden library default. Timestamp conversion
is exposed by `TimestampQueryCapabilities` (`period_nanos`, optional
`valid_bits`, and `non_blocking_resolve`). Resolve index/count/offset/alignment,
query state nesting, buffer usage, completion, and device-loss wakeup are validated.
Query reset is deliberately not a public command: each backend derives the
exact written slots from immutable `RecordedWork` and emits a legal reset in
the command-buffer preamble before any scope begins. This is Fluxel's equivalent
of HAL `reset_queries`, avoids exposing backend reuse mechanics, and guarantees
that a slot is reset exactly for the work that writes it.
WebGL2 probes its actual query and timer-query extension; pipeline statistics
remain Unsupported where unavailable.

### WebGPU query-profile boundary

WebGPU has a direct `resolveQuerySet()` command, with a 256-byte aligned
destination offset. That alone cannot enable the Fluxel query family: a useful
resolve first requires a query set that can be legally recorded. A WebGPU
render-pass descriptor chooses exactly one `occlusionQuerySet` when the pass
begins, whereas one portable `RasterScope` may legally begin/end slots from
different sets in sequence. Rejecting the second set only during submission
would make a public capability promise a program that fails after recording.

Timestamp writes have the same semantic impedance: current WebGPU pass
`timestampWrites` represent beginning/end descriptor boundaries, while the RHI
records an exact arbitrary command-stream position. Rewriting the latter into a
boundary changes the measured interval. Consequently WebGPU deliberately
publishes none of `OcclusionQuery`, `TimestampQuery`,
`TimestampInsideEncoder`, `TimestampInsideRasterScope`,
`TimestampInsideComputeScope`, or `QueryResolve`, even when the browser offers
`timestamp-query`; it publishes neither query limits nor a fictitious 8-byte
alignment. A future WebGPU query profile must first express one fixed pass set
and explicit pass-boundary timestamps in public recording vocabulary, then
enable creation, validation, lowering, and resolve together.

## 67.4 Resources, mapping, formats, and samplers

`BufferUsage` includes `MAP_READ`, `MAP_WRITE`, `INDIRECT`, `QUERY_RESOLVE`,
`BLAS_INPUT`, and `TLAS_INPUT`. `ResourceUse::AccessMask` has matching
portable map, indirect, query, acceleration-build, and ray-data semantics.
`Device::map_buffer(...)?.await` uses `MapMode::{Read, Write}` and borrowing RAII mapped ranges; unmap,
flush, and invalidate define range/alignment, exclusivity, coherent versus
explicit-cache memory, persistent mappings, GPU-use legality, completion, and
loss. Dropping a pending mapping cancels it and releases its reservation; loss
wakes every pending mapping to `DeviceLost`. `MappablePrimaryBuffers`,
`PersistentMapping`, and coherent/explicit
flush-invalidate facts are distinct capabilities. Persistent mapping keeps the
native mapping lease alive across submission; it does not grant coherent
simultaneous ownership of overlapping bytes. The caller must flush CPU writes
before an overlapping GPU read, wait for GPU writes before reading and then
invalidate non-coherent memory, and otherwise externally order overlapping
CPU/GPU accesses. Coherency removes cache maintenance only. A backend that
cannot uphold this allocation-and-barrier contract must not publish
`PersistentMapping`.

`TextureFormat` covers the complete normal, depth/stencil, planar/video, BC,
ETC2/EAC, and ASTC families: R/RG/RGBA 16-bit norm formats, RGB9E5,
RGB10A2 Uint/Unorm, RG11B10 Ufloat, R64Uint, Stencil8, NV12, P010, BC1--7,
ETC2/EAC, and all standard ASTC block sizes in linear, sRGB, and HDR forms.
`TextureAspect` has Plane0, Plane1, and Plane2. Compressed and planar layout
calculations use their real block/plane geometry for mip dimensions, copy
alignment, row pitch, logical estimates, views, and capture.

`Astc*Hdr` denotes the ASTC HDR codec form, not a promise that it is available
where ordinary ASTC is.  Each block size is queried independently.  Vulkan may
publish it only after `VK_EXT_texture_compression_astc_hdr` is enabled, its
feature bit is enabled on the logical device, and the exact `*_SFLOAT_BLOCK`
format probe accepts the requested use.  DX12 publishes it as Unsupported
because DXGI has no ASTC HDR format.  HDR, linear UNORM, and sRGB ASTC facts
are independent: support for one must never be inferred from another.

Format support is always per `TextureSupportQuery`, not a compressed-texture
boolean. `FormatFacts`/support queries represent sampling, linear/minmax
filtering, storage read/write/read-write/atomic, color/depth-stencil attachment
and blending, copy source/destination, sample counts 2/4/8/16, resolve,
dimension, sliced-3D compression, view compatibility, and descriptor-specific
limits. Float32 filterability is a probed device fact. Compressed, planar, and
multisample combinations not accepted by native queries are Unsupported.

Sampler facts independently cover anisotropy plus `MaxSamplerAnisotropy`,
comparison samplers, clamp-to-zero, clamp-to-border, custom border color, and
border-color restrictions. A descriptor requesting one validates the relevant
feature and limit before backend lowering; it is never silently changed to a
different addressing/filtering mode.

`TextureViewDescriptor::usage` is a subset of its texture's usage. View
format/aspect/plane/usage compatibility is a descriptor-dependent query, not a
backend-name rule. `ExternalTexture` and external-image copy use opaque public
source descriptors: origin, extent, flip-Y, premultiplied-alpha and color-space
intent. Browser/native object types never enter public API. Capabilities state
the exact external-copy and unrestricted-copy restrictions.

## 67.5 Commands: clear, indirect, and immediates

The recorder supports validated `clear_buffer` and `clear_texture`; clear is
only advertised where it has a direct native semantic and creates actual uses.
It must not be silently emulated by an inserted draw/compute pass.
`clear_buffer` uses the common native fill contract: its offset and size are both
four-byte aligned, including the exact-end boundary.

`clear_texture` validates that its selected aspects belong to the format before
recording. Color clear requires `COPY_DST`; depth or stencil clear additionally
requires `DEPTH_STENCIL_ATTACHMENT`, because a native backend may lower it via
a depth-stencil view rather than a copy footprint. A backend that advertises the
feature must lower every creatable aspect it accepts at record time; it may not
record a depth/stencil clear and discover only during submission that its color
route cannot express it.

Raster supports direct/indirect indexed and non-indexed draw, multi-draw,
count-buffer multi-draw, and the corresponding indexed forms. Compute supports
indirect dispatch. Required distinctions are `IndirectDraw`,
`IndirectDispatch`, `MultiDrawIndirect`, `MultiDrawIndirectCount`,
`IndirectFirstInstance`, and `BaseVertex`. Alignment, stride, count-buffer
range, `INDIRECT` usage and bounds are validated. Nonzero `base_vertex` is
rejected unless `BaseVertex` is enabled.

`IndirectFirstInstance` is distinct because argument bytes are GPU-owned and
cannot be inspected by portable recording. A backend must not publish executable
`IndirectDraw` unless its native route guarantees the non-zero first-instance
semantic (for example Vulkan's drawIndirectFirstInstance feature); it cannot let
the driver discover an unsupported argument value after work is accepted.

Pipeline interfaces declare `ImmediateData` ranges and stage visibility;
raster, compute, and ray scopes set bytes by offset. `Immediates`,
`MaxImmediateSize`, and `ImmediateDataAlignment` are independent facts. No
backend may silently substitute an ordinary uniform binding.

## 67.6 Pipeline, binding, shader, and raster families

`PipelineCache` has descriptor, validation key, optional input data, explicit
invalid-data policy, serialization data, and pipeline-descriptor references.
Cache availability and serialization are capabilities; invalid data follows its
declared reject/ignore policy rather than driver-defined behaviour.

Vertex formats cover all wgpu 30 numeric layouts including 8/16/32-bit
integer/norm/float, BGRA8 norm, packed 10:10:10:2, Float16, and Float64
vectors. Byte size, numeric class, components, validation, lowering and capture
are defined per format. Float64 uses `VertexAttribute64Bit`.

Raster capabilities independently gate polygon line/point, conservative
rasterization, depth clip control, depth-bias clamp (including legal topology),
dual-source blending, independent blend, and multisampled shading. Read-only
depth/stencil, comparison samplers, NPOT mipmapping, cube arrays, uint32 index
coverage, depth/stencil copies, and binding alignment are likewise concrete
support/route facts, not assumed Base behaviour.

Binding arrays distinguish texture, buffer, storage-resource, uniform-buffer,
and acceleration-structure arrays; fixed and runtime-sized counts;
partially-bound resources; sampled/storage non-uniform indexing; writable
storage; stage and resource kind; and maximum elements. `BindingCount` includes
`RuntimeSized`. Limits cover general, sampler, AS, and non-sampler arrays.

Shader requirements can declare F16/F64/I16/I64, float/int64/texture atomics,
early-depth, subgroups (including vertex and barriers plus min/max size),
barycentrics, per-vertex data, draw/primitive index, clip distances,
coherent/volatile memory decorations, F16-in-F32, and required builtins.
Passthrough shader acceptance is separately capability-gated and still requires
an ABI/version-compatible validated interface, provenance and trusted boundary;
native bytecode is not implicitly trusted.

## 67.7 Multiview, mesh, acceleration structures, ray tracing, matrices

Raster/pipeline descriptors carry a validated multiview mask. Facts distinguish
multiview, selective multiview, multisample arrays, and mesh multiview; limits
state maximum view counts. Color attachments can select a validated 3D depth
slice, which is reflected in subresource uses.

Task and mesh shader stages, mesh pipeline path, direct/indirect/count mesh
draws, and mesh points are public semantics. Facts and limits cover task/mesh
workgroup counts/dimensions/invocations, payload, output vertices/primitives/
layers, points, and multiview. Unsupported platforms return Unsupported.

A `MeshPipelineDescriptor` replaces only `VertexInputState` and the vertex
entry point with mesh/task stages. It still carries the complete graphics fixed
state: `PrimitiveState`, optional `DepthStencilState`, `MultisampleState`, an
optional multiview mask, and sparse `ColorTargetState` locations. Its target
signature is derived from that state, never supplied as an independent
potentially-divergent attachment contract.

`rhi::acceleration` defines BLAS/TLAS, triangle/AABB geometry, instances,
build sizes/modes, build/copy/compaction operations, AS bindings, and ray-query
capabilities. It does not expose GPU virtual addresses. Facts/limits cover ray
query, hit-vertex return, extended vertex formats, AS arrays, primitive/
geometry/instance counts and shader-stage AS/buffer counts.

BLAS input validation is intentionally exact before native sizing or lowering:
the sum of triangle and AABB primitives is bounded by
`MaxBlasPrimitiveCount`; every vertex, index, and AABB range must cover its
declared `count × stride` without arithmetic overflow (an exact end-of-buffer
range is valid). `Float32x3` positions are the baseline; any other declared
`AccelerationStructureVertexFormat` requires
`ExtendedAccelerationStructureVertexFormats`. TLAS transforms contain finite
floating-point components only, and `shader_record_offset` is limited to the
portable 24-bit subset so a backend never silently truncates it. These are
portable refusals, not backend repair opportunities.

`rhi::ray_tracing` defines pipelines, shader groups, ray-generation/miss/hit/
intersection stages, ray scopes, shader-table/group data, and `trace_rays`.
Facts include ray-tracing pipelines; limits/alignment facts include dispatch,
recursion, scratch, group-data size/alignment/offset and TLAS instance size.
`CooperativeMatrix` publishes structured supported M/N/K, input/result types,
stages and scope properties, which shader requirements select exactly.

## 67.8 Presentation, diagnostics, memory, and external extension SPI

Surface facts provide format+color-space pairs, usage, view formats, present
modes, composite alpha, current/controlled extent, maximum frame latency,
display HDR data, and generic presentation timing capability. Configuration
selects usage, view formats, color space, composite alpha and frame latency;
an acquired frame reports `suboptimal`. Unconfigure/close has explicit lifecycle
semantics. HDR and timing are not deferred and do not use Vulkan-specific names.

`RayHitVertexReturn` is consumed by shader reflection through
`ShaderBuiltin::RayHitVertexPosition`; it is not implied by `RayQuery` or by an
acceleration-structure binding. Artifact acceptance therefore rejects the
builtin unless this separate capability was enabled.

The diagnostics module provides capability-gated native graphics-debugger
capture start/stop, allocator/backend diagnostic reports, OOM-safe allocation
checks, and truthful counters for allocated, committed, resident, aliased and
retired memory. These are observations, not invented VRAM guarantees.

External-memory **import** is a separate backend-extension SPI with typed handle
class, extension-owned native handle and synchronization, immutable
format/usage/size validation, and completion-safe retirement. The frozen v13
surface does not export an RHI allocation or transfer ownership of a raw native
handle; an exporter remains platform/backend extension work until a portable
ownership and synchronization contract is specified. Win32/FD/DMA-BUF names do
not enter ordinary resource vocabulary.

## 67.9 Backend requirements

DX12, Vulkan, Metal, WebGPU, GL, GLES, and WebGL2 each probe and publish only
their real support. Browser/GL extensions are probed individually (including
S3TC/ETC2/ASTC, timer query, multiview, and external-copy restrictions), never
assumed from API family. Dedicated transient allocation is the required fallback;
`Aliasing` is advertised only after placed/shared allocation, non-overlap proof,
alias hazard/barrier, and completion-safe reuse are lowered. DX12/Vulkan/Metal
must implement their native aliasing mechanisms before advertising it.

### 67.9.1 Native-lowering admission matrix

The ten-class status matrix in [the root design](../design-rhi.md#51-ten-class-native-lowering-status-matrix)
is normative for enhancements whose native mechanism is deliberately private.
The following is the capability-facing reading of that matrix. Every row already
has a complete portable carrier; an absent native lowering is never grounds for
adding a second public API.

| Class | DX12 / Vulkan current publication | Pre-native behavior | Required admission evidence before publication or optimization |
| --- | --- | --- | --- |
| Persistent attachment descriptors | No public capability; private baseline allocator only. | Preserve attachment semantics with private temporary/reused descriptors. | Completion-safe descriptor retirement, loss drain, repeated attachment/view conformance. |
| State-difference tracker | No public capability; conservative legal barriers/layouts remain valid. | Never infer a state; emit a legal conservative transition/dependency. | Complete actual-use trace, cross-batch final-state proof, alias/present/loss cases. |
| Shared completion waiter | Completion semantics are published; waiter topology is private. | Every future still resolves Complete, Failed, or DeviceLost. | Race-free multiplexing, cancellation, idle and all-pending-on-loss wakeup tests. |
| Resource/descriptor retirement | Required baseline semantic; private allocator policy. | Keep backing alive through last-use completion. | Exact last-use tracking, no early reuse, loss drain and bounded-retirement diagnostics. |
| Submit-lock narrowing | No public lock capability. | Retain conservative serialization; `Err` accepts zero native work. | Atomic preflight/commit, concurrent-plan hazard and terminal-publication tests. |
| Recorder packets and state cache | No public packet/encoder capability. | Lower frozen `RecordedWork` directly. | Exact command/use reconstruction, backing retention, cache invalidation and capture conformance. |
| Pipeline cache | DX12 and Vulkan currently publish cache/serialization through wired native paths; a backend without that complete path publishes Unsupported. | Reject requested cache capability before native work when the fact is absent; uncached creation remains valid. | Native cache creation/identity, invalid-data policy, pipeline feed-through, serialization and loss/driver-mismatch tests. |
| Transient aliasing | `Dedicated` is published; `Aliasing` is not published until complete. | Reject aliasing before allocation; use dedicated backing. | Memory/heap probe, device enablement, non-overlap proof, native alias barrier/dependency, retirement and overlap/loss tests. |
| Multiple native queues | Publish only implemented lane classes and dependency routes. | Reject an unadvertised lane/route before queue work; an advertised ordered lane may collapse privately. | Queue-family/class/present probe, device enablement, native waits/signals/ownership and dependency/present/loss tests. |
| Primary and persistent mapping | Publish only exact mappable buffer masks; `PersistentMapping` stays absent until proven. | Reject unsupported mask/range/lease before native map; no hidden staging. | Heap/memory probe and selection, completion-before-grant, cache maintenance/coherence truth, lease retirement and non-coherent/overlap/loss tests. |

"Device enablement" includes any native extension, feature chain, queue family,
memory type, heap flag or cache object required by that row. A positive native
probe alone is insufficient: the selected device must enable it, lowering must
consume it, and retention must keep every affected native object valid until its
portable completion/loss terminal state. The corresponding positive, negative
and boundary conformance tests are part of the admission evidence.

No row may be represented by reachable `todo!()` or `unimplemented!()` code.
For a public optional capability, the pre-native result is structured
`Unsupported`; for a transparent private optimization, the row's stated
correct fallback is mandatory. A private `TODO` comment is permitted only when
it names this carrier, fallback, and admission evidence.

### 67.9.2 Advanced native TODO families

[The root design's advanced native-lowering matrix](../design-rhi.md#52-advanced-native-lowering-todo-boundary)
is the normative scope boundary for Mesh/Task, the ray system, cooperative
matrix, transient physical aliasing, external interop, NV12/P010 multiplanar
handling, native debugger capture, advanced allocator diagnostics, HDR/timing,
and advanced descriptor indexing. Their public vocabulary and portable tests are
complete v13 requirements. Only their native DX12/Vulkan lowering may remain a
documented TODO, with its capability absent and a pre-native `Unsupported`
result. These rows must not be confused with the private performance
enhancements in section 67.9.1.

## 67.10 Definition of done

A feature family is complete only when all are true:

1. A public, backend-neutral semantic exists with no native handles/types.
2. Capabilities, limits and `DeviceRequirements` express exact support.
3. Validation rejects unsupported/illegal use before native work.
4. Every advertised backend path has concrete lowering; reachable code contains
   neither `todo!()` nor `unimplemented!()`.
5. Commands derive correct actual resource uses, hazards, lifetime and
   completion/loss behaviour.
6. Tooling/capture descriptions and logical statistics remain reconstructible.
7. Positive and negative conformance tests cover every advertised capability;
   platform tests verify the backend matrix.

This is the definition used for the wgpu-hal 30.0.1 feature audit, its
downlevel flags, and its limits. A feature is not complete merely because its
enum is present or because one backend can compile a descriptor.
