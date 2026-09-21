# ADR-0011: Build the GL family through three private layers

**Status:** Accepted; implementation is complete through the static closure
gate. The private layers live under `crates/rhi/src/backend/gl/` (`api/`,
`state/`, `native/`, `browser/`, `platform/`, and feature-specific lowering
modules). The shared state authority, native/browser resource and submission
paths, completion/readback/presentation paths, and capability-to-lowering gate
are in place. This is not runtime release evidence: Win11 WGL, Android
EGL/GLES, Chrome WebGL2, and Edge WebGL2 still have to run against the finished
v13 adapter.

## Context

The 0.15 ecosystem closure adds one GL-family implementation for desktop GL
4.x, GLES 3.x, and browser WebGL2 without weakening the common RHI contract.
The existing WebGL2 adapter is a retained 0.14 fixed-scene path with direct
browser calls. It remains operational during migration, but it is not a base
for the new common adapter or state machine.

The implementation is clean-room work. Its inputs are Fluxel's common RHI and
RenderGraph contracts, the neutral 0.15 plan and state-machine handoff, public
Khronos specifications, and independently authored tests. The unlicensed
reference snapshot named by the local plan is observation material only; its
code, names, structure, shaders, tests, tables, data layout, and control flow
must not enter this repository.

The exact ecosystem starting revisions are:

- ecosystem `.github`: `023f3d7b1a1be333c97d308c93f706a3b680905f`;
- `fluxel-bases`: `22c4eb0e199575aa71b59f3abc6ec3f72d934b9a`;
- `fluxel-rendering`: `ad37a14da5415b880c4746ce3238e83aa8ea933a`;
- `fluxel-host`: `7d43134460217c636d8881dbf76771e494c19e28`;
- `fluxel-jsbridge`: `c010016701f566123040d2735de1c2b9549bdca2`.

## Decision

The private implementation lives under `crates/rhi/src/backend/gl/`, while
shared private types use the `GlFamily` name. There is one dependency direction:

```text
api <- native | browser | mock
api <- state
api + state + common RHI <- platform/lowering
```

`api` owns the neutral typed GL-family vocabulary and structured failures.
`native` and `browser` own actual context calls and discovery. `state` owns
desired/applied/unknown knowledge, deterministic dirty reconciliation,
counters, and bounded structurally keyed caches. `platform` and the lowering
modules alone translate the common RHI objects and commands.

The following boundaries are mandatory:

- `api` imports no platform implementation, state, renderer, or
  RenderGraph execution code;
- native/browser/mock implementations do not define a second state authority;
- state imports only the neutral API and names no `web_sys`, `glow`, EGL/WGL/GLX/CGL
  provider, renderer, residency, or RenderGraph type;
- platform/lowering imports no renderer, residency, legacy WebGL2 session, or
  public raw GL type;
- raw handles and extension objects never appear in public rustdoc;
- unsupported optional domains reject before object, extension, or command
  side effects.

The GL-family implementation may in principle execute through a provider whose
internal translation target is DX11, such as an ANGLE configuration. DX11 is
not a separate Fluxel backend or 0.15 semantic profile: this release adds no
DX11 selection API, feature, capability claim, test dimension, or evidence
gate. The implemented and reported profiles remain WebGL2, desktop GL 4.x, and
GLES 3.x.

The old adapter is observation material for behavior and state-reduction
strategy only. New code must not copy its renderer-shaped session into the API,
state, or resource-ownership model.

### Context providers and initial evidence matrix

GL-family dispatch uses the Rust `glow` 0.18.0 API on native and browser
targets. Native EGL loading uses `khronos-egl` 6 with dynamic loading, and
Windows WGL loading uses `glutin_wgl_sys` 0.6. These libraries are contained by
Layer 1; none defines Fluxel state, cache, lifetime, or common RHI semantics.
Host supplies only `raw-window-handle` window/display handles; RHI continues to
own context, surface, and presentation:

- `DesktopGl4`: Windows WGL through `glutin_wgl_sys`, requesting the highest
  available core context from 4.6 down to the 4.0 floor on Windows
  11 25H2 build 26200.9445, AMD Radeon 780M Graphics, driver
  32.0.21028.2002. The runtime probe, not the request, records and authorizes
  the actual GL/GLSL version, flags, extensions, limits, and operations.
- `Gles30` and `Gles31Plus`: system EGL through dynamically loaded
  `khronos-egl`, with a real system or device
  GLES implementation. The first development target is Ubuntu 24.04.3 under
  WSL2 with the system `libEGL.so.1.1.0`; separate probes request ES 3.0 and
  ES 3.1 and must record the actual renderer before any capability claim. A
  physical non-translation GLES 3.1 device remains required for portable GLES
  release evidence.
- `WebGl2`: Google Chrome and Microsoft Edge stable on the same Windows/AMD
  system, each using an independently created real WebGL2 context. Firefox and
  Safari remain auxiliary future probes and do not broaden support unless an
  exact target is added to the ecosystem ledger before implementation.

A real GLES 3.1 device is a release evidence requirement for a portable GLES
claim. Missing hardware leaves that target gate open rather than narrowing the
implementation. ANGLE is neither a default dependency nor a required provider;
it may be evaluated later only if an explicitly selected target needs it.

Native features are split by provider. A platform-neutral `gl-family` feature
may compile API/state/mock contracts without native or browser dependencies;
`native-gl-wgl`, `native-gles-egl`, and the existing `webgl2` feature add only
their own provider. DX12/Vulkan defaults and WebGPU remain unchanged.

Production code must not depend on `winit`, `glutin-winit`, `glutin`, or ANGLE.

Across GL 4.x, GLES 3.x, and WebGL2, 0.15 prefers the newest applicable core
or standardized extension route and uses older/vendor aliases only when they
are the available, semantically equivalent path. Every proved usable route is
normalized toward a standard WebGPU feature/limit/format and enabled; reported
non-WebGPU feature families do not enter the common capability surface. This
extension closure is part of 0.15 rather than deferred follow-up work.

In particular, the Host window contract is not replaced by another window or
event-loop library, and private binaries installed by unrelated applications
are never context-provider inputs.

At the public RHI and RenderGraph boundary, extension-backed functionality is
represented only as optional common/WebGPU capabilities, exact format facts,
and limits. Extension names never escape Layer 1. Runtime availability is
optional per adapter; implementing discovery, normalization, validation, and
structured rejection for every in-scope capability is mandatory for 0.15.
RenderGraph compilation checks its required capability/format set against the
selected device before execution. Parallel compilation, robustness, and debug
paths stay backend-private, while common operations such as texture copy may
use an extension, a core command, or an equivalent validated fallback without
exposing that lowering choice publicly.

### Shader artifacts

Shader routing is backend-native passthrough first, rather than a mandatory
Naga intermediate representation. WGSL goes directly to wgpu; HLSL and DXIL
go directly to DX12/DXC; SPIR-V goes directly to Vulkan. GLSL/ESSL goes
directly to GL only after its exact profile/version/dialect check. A source
that a selected target cannot accept natively uses Naga only when it belongs to
Naga's supported translation domain; otherwise it rejects before backend work.
Validated reflection and logical binding metadata are shared, but not a
universal Naga IR. RenderGraph refers only to artifacts, not source languages.

The GL-family endpoint accepts only already-lowered GLSL/ESSL artifacts. Each
one carries its stage, entry point, source hash, and explicit dialect: WebGL2
and GLES 3.0 use ESSL 300; GLES 3.1/3.2 use ESSL 310/320; desktop GL 4.x uses
the exact GLSL target for the actual profile. The dialect/profile check occurs
before driver shader creation. This keeps profile diagnostics precise without
making Fluxel WGSL-only or exposing a frontend choice in public RHI.

`ShaderAbiVersion { 1, 0 }` deliberately does not contain a GL
reflection-name table: `ShaderResourceRequirement` identifies logical
group/slot/kind/count, but it is not a GLSL identifier allocation convention.
Consequently the GL backend must fail closed for public resource-binding facts
unless a selected artifact provides a validated backend-private mapping from
that logical requirement to the linked program's uniform/block/image name and
index. It may still compile and use resource-free programs. This is a deliberate
structured `Unsupported` boundary, not a deferred `todo!()` implementation and
not permission to invent names from group/slot numbers. A later ABI revision or
artifact metadata extension can add the mapping without leaking GL names into
the public API.

## Performance decision

The primary product goal is lower owner-thread CPU submission latency and fewer
actual GL/WebGL calls for 2,000 resident indexed draws with repeated pipeline
and material bindings, bounded mesh/texture sets, depth, blend,
viewport/scissor, and an offscreen-to-present pass. Correctness, pixels, draw
order, completion, lifetime, and recovery override speed.

Compressed texture support is format-exact rather than family-global and is
limited to WebGPU's standard BC, ETC2/EAC, and ASTC feature families. DXT1/3/5
map to BC1/2/3; BC4/5/6H/7 require exact RGTC/BPTC evidence. PVRTC and ETC1 are
excluded. Block geometry, encoded byte size, color space, and extension/core
evidence are validated per format before upload; compressed formats never
inherit render, blend, or storage-image support from an unrelated family
capability.

The cache-disabled mode is the adjacent baseline and correctness oracle. The
only initial candidates are grouped dirty reconciliation, structural VAO/FBO
reuse, and allocation-free bind-group short-circuiting. Each is screened in
isolation with interleaved samples and call/allocation counters. No candidate
is retained unless the representative improvement exceeds observed noise and
the guard workloads remain understood.

The concrete state model follows the proven `webgl2_performance` architecture,
adapted to v13 identities rather than copied source. One context has one state
authority through which all mutable GL calls pass. A raster-pipeline request
first compares its stable pipeline identity, then independently reconciles the
program, raster, depth, stencil, blend/color, multisample, viewport, and scissor
blocks. Bind groups are retained as logical packets plus a dirty mask and are
resolved into per-slot buffer, texture, sampler, image, and storage bindings
immediately before draw or dispatch. Vertex and index bindings form a structural
geometry key used for VAO reuse; framebuffer attachment tuples similarly key
derived FBO reuse.

The authority also mirrors active texture, every indexed binding slot,
framebuffer/read/draw selection, renderbuffer, clear values and masks,
pixel-store state, and active query state. Resource retirement invalidates both
the exact bound slots and every derived VAO/FBO dependency before native
deletion. A helper that performs raw GL calls must declare the domains it
mutates and invalidate them; an undeclared helper invalidates all domains.
Context loss or replacement forgets all claims and starts the next generation
as Unknown. Stable object IDs, generations, or canonical structural keys replace
the old implementation's pointer comparisons, so this optimization does not
introduce extra shared-ownership wrappers.

### v13-shaped state authority

The old implementation supplies the optimization model, not the object model.
The v13 recorder does not replay `set_pipeline` and `set_bind_group` setters:
each `RasterDraw`, `RasterIndirect`, `ComputeDispatch`, or `ComputeIndirect`
already owns the complete logical snapshot that was current when it was
recorded. GL lowering therefore has the following two stages:

```text
RecordedPayload + command ResourceUse
    -> Phase A: resolve public handles into owned, generation-safe GL packets
    -> Phase B: reconcile one context's state authority, then issue work
```

Phase A performs no GL call. A resolved raster packet names the backend-private
pipeline identity, its immutable pipeline record, resolved bind-group packets,
vertex/index ranges, dynamic raster values, and draw scalars. `ResourceUse` is
retained separately for memory visibility and alias/hazard lowering; it is not
used to reconstruct bindings already present in the command snapshot.

One WGL, EGL, or WebGL2 context owner holds exactly one `ContextState`. The
public API exposes neither that context nor a session/token. `ContextState` is
split according to the mutable GL machine actually reached by v13 commands:

```text
ContextState
|- PassState             DRAW/READ FBO, draw buffers, attachment key
|- RasterState           program and independent fixed/dynamic raster leaves
|- GeometryState         VAO plus vertex/index structural key
|- BindingState          active unit; texture/sampler/UBO/SSBO/image slots
|- ComputeState          compute program and dispatch binding epoch
|- PixelTransferState    pack/unpack, PBO and temporary copy bindings
|- QueryState            active query per legal target
|- VisibilityState       ResourceUse-derived memory visibility
|- DerivedCaches         canonical state blocks, VAOs, and FBOs
`- DirtyState            Unknown/known facts and scoped raw-access damage
```

Submission completion and presentation lease state are owned by the same
execution owner but are not GL setter-cache domains: fences, draw/dispatch,
pass load/store work, clears, queries, barriers, acquire, present, and abandon
are ordering operations and are never elided.

At pipeline creation, the private pipeline table stores its real
`GlObjectName`, program/link epoch, vertex-layout key, and canonical IDs for
the independently mutable raster, depth, stencil, blend/color, and multisample
blocks. Phase B first compares the pipeline `GlObjectName` in O(1). An exact hit
skips every immutable pipeline leaf. A miss compares the canonical block IDs
and emits only changed leaves. This preserves the reference implementation's
whole-object and sub-object pointer fast paths without using pointer identity,
collision-prone hashes, or extra `Arc`/`Rc` ownership.

The exact-identity gate is valid only while every leaf owned by that pipeline
is still known. GL slots shared by command families have one owner: raster and
compute use the same current-program slot, copy and render work may use the same
READ/DRAW framebuffer slots, and raw helpers may touch bindings used by later
draws. A compute-program bind therefore makes the raster program leaf unknown
without discarding unrelated raster leaves; an unchanged raster pipeline must
then restore only its program. No domain may keep a second, contradictory copy
of a shared GL slot.

The draw packet's groups are compared by group index, resolved group identity,
dynamic offsets, and program binding epoch. Changed groups set dirty bits;
program/link changes dirty every affected assignment. Immediately before the
draw or dispatch, only dirty slots are written. Geometry is keyed by the
pipeline vertex layout, vertex buffer identities/ranges, index identity/format,
and any capability-dependent base-instance contribution. The resulting VAO is
looked up structurally and bound only when its identity changes.

Pass begin always binds or creates the attachment-keyed FBO and performs its
load operations even if the preceding pass used the same descriptor. Copy,
upload, and readback operations either reconcile their temporary READ/DRAW FBO,
PBO, texture, and pixel-store slots through `ContextState`, or invalidate the
precise affected domains before returning; resetting a raw binding to zero is
not a substitute for restoring the authority's known value.

Every state claim begins as Unknown. A successful native call alone publishes
the new known value. A failed or partially applied call poisons the affected
leaf. Resource retirement clears exact binding slots and reverse-invalidates
dependent VAO/FBO/cache entries before native deletion. Context loss,
restoration, or generation replacement forgets every claim and every raw object
from the old generation.

Reconciliation is prepare/commit, not optimistic mutation. Preparing a pass,
pipeline, geometry, binding, transfer, or query transition returns the calls
required against the last known state but does not publish them. The owner
commits each leaf only after the corresponding GL call succeeds; an error marks
the attempted leaf Unknown. This rule also prevents an exact pipeline hit after
a partially failed install.

## Consequences

0.15 is intentionally larger than a WebGL2 cleanup: GL4, GLES, and WebGL2 share
one neutral API and state model, then meet the existing common RHI semantics.
The compatibility layer cannot accumulate direct calls, and the common RHI is
not reduced to WebGL2's capability floor. An underlying provider's translation
target, including a possible DX11-backed ANGLE implementation, remains a
provider fact rather than a new public backend. Context epochs and full
structural identity prevent raw-name reuse from selecting stale state or cache
entries.

Checkpoint A closes only the architecture and selected evidence identities.
Runtime discovery must still prove every requested profile and capability;
declarations, extension strings, and mock tests alone do not establish support.

The implementation's static closure is intentionally narrower than a runtime
claim. Registry routes normalize core promotion, extension availability, and
required entry points; the lowering closure publishes a capability only when
the corresponding native/browser v13 path exists. The binding exception above
remains fail-closed. Real context creation, driver behavior, WSI, context-loss
recovery, and browser security/runtime conditions are established only by the
four-device/browser evidence matrix, not by `cargo check`.

Transactional submission is preserved across the GL family's eager execution
model. Phase A performs every fallible portable validation before a native call.
Phase B reserves its completion serial before the first GL call; once any work
is accepted, a later action, fence, or flush failure produces an accepted
submission whose completion is `Failed` or `DeviceLost`, never a false
`submit Err`. Readback and presentation waiters observe the same terminal
state. A backend-private loss authority connects DOM/native context loss to the
public device's stable `status()` and `loss_info()` without exposing a context,
session, or token.

Texture usage facts are also execution promises, not raw driver-format facts.
The current GL-family `COPY_SRC` rows are limited to the RGBA8 formats supported
by the readback lowerer. `COPY_DST` rows are limited to RGBA8 and the exact
compressed formats whose whole-mip block layout is validated and executable.
Broader native copy-image support must not publish a usage combination for
which another public transfer route would fail.
