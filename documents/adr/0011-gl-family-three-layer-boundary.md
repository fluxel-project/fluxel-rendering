# ADR-0011: Build the GL family through three private layers

**Status:** Accepted, and implemented in the 0.15 release. The three layers
exist under `crates/rhi/src/webgl2/` (`api/`, `state/`, `compat/`), the boundary
below is enforced by `scripts/check_gl_architecture.py` rather than by
convention, and the release gate in the 0.15 plan records what was proven on
real hardware and what was not.

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

The private implementation lives under `crates/rhi/src/webgl2/`, while shared
types use the `GlFamily` name. There is one dependency direction:

```text
api/core <- api/native | api/browser | api/mock
api/core <- state
api/core + state + common RHI <- compat
```

`api` owns context calls, discovery, typed objects, and structured driver or
browser failures. `state` owns desired/applied/unknown knowledge, deterministic
dirty reconciliation, counters, and bounded structurally keyed caches.
`compat` alone translates common RHI commands and RenderGraph requirements.

The following boundaries are mandatory:

- `api/core` imports no platform implementation, state, compat, renderer, or
  RenderGraph execution code;
- platform and mock API implementations import neither state nor compat;
- state imports only API core and names no `web_sys`, `glow`, EGL/WGL/GLX/CGL
  provider, renderer, residency, or RenderGraph type;
- compat imports no renderer, residency, legacy WebGL2 session, or raw GL type;
- raw handles and extension objects never appear in public rustdoc;
- unsupported optional domains reject before object, extension, or command
  side effects.

The GL-family implementation may in principle execute through a provider whose
internal translation target is DX11, such as an ANGLE configuration. DX11 is
not a separate Fluxel backend or 0.15 semantic profile: this release adds no
DX11 selection API, feature, capability claim, test dimension, or evidence
gate. The implemented and reported profiles remain WebGL2, desktop GL 4.x, and
GLES 3.x.

The existing `experimental::webgl2` adapter remains the migration oracle and
keeps its current feature/API until the new WebGL2 slice passes the common
compatibility gates. New code must not copy its renderer-shaped session into
the API or state layers.

### Context providers and initial evidence matrix

GL-family dispatch uses the Rust `glow` 0.18.0 API on native and browser
targets. Native EGL loading uses `khronos-egl` 6 with dynamic loading, and
Windows WGL loading uses `glutin_wgl_sys` 0.6. These libraries are contained by
Layer 1; none defines Fluxel state, cache, lifetime, or common RHI semantics.
Host supplies only `raw-window-handle` window/display handles; RHI continues to
own context, surface, and presentation:

- `DesktopGl4`: Windows WGL through `glutin_wgl_sys`, requesting a core 4.3 context on Windows
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
- `WebGl2`: Google Chrome Stable 153.0.8010.36 on the same Windows/AMD system.
  This preserves the existing named browser support claim. Firefox and Safari
  remain auxiliary future probes and do not broaden 0.15 support unless an
  exact target is added to the ecosystem ledger before its implementation.

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
