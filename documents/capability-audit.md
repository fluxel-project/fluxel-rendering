# Capability admission audit

The original source audit covered RHI revision
`e39c0e50406d764e6f4fb53950f7c8bc00469d04`. The identified Core gates and Metal
capability omission are fixed in the pushed revision
`0f7f785d4f0ccc62e125dd6cf560d91333ec85cf`. The new exact dependency is integrated
and the named DX12/WGL native matrix has been reviewed.

## Capability sources and boundaries

The audit inventory enumerates all 94 public `DeviceCapabilities` fields and all
15 `DeviceLimits` fields. The retained source review is
`target/reviews/20261102-capabilities/rhi-audit.json`; its revised-source
reconciliation is `target/reviews/20261102-capabilities/rhi-audit-0f7f785.json`.
The audit maps DX12, GL, Vulkan,
WebGPU and Metal assignments to their native queries, selected API/profile
guarantees, backend implementation bounds, or unsupported defaults. Cargo features
select compiled code and do not establish runtime adapter support.

The GL-specific inventory additionally covers all 24 declared feature routes,
10 discovery capabilities, 36 entry-point facts and 50 known extension names.
Device requests, shader feature requirements, resource format/usage queries and
each backend's surface support path are included. Vendor extensions outside the
selected catalog and every extension on physical drivers are not certified.

A published `false` for an unimplemented optional route means the backend refuses
that route; it does not prove that the hardware lacks the extension. Conversely,
baseline guarantees require the selected API/profile to have been admitted at
device open. For example, Microsoft's [D3D12 feature-level table](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-feature-levels)
distinguishes guaranteed compute support from tiered optional features. The
[OpenGL 4.3 core specification](https://registry.khronos.org/OpenGL/specs/gl/glspec43.core.pdf)
provides compute, shader storage and debug-output core routes; lower profiles
require the relevant extension and loaded entry points.

This is a source admission audit of every exposed field, not a native workload
for every optional feature. Only the named Windows DX12 and WGL S0/S1 workloads
have hardware evidence for the revised RHI source. Vulkan, Metal, browser
WebGPU/WebGL2 and EGL remain without hardware evidence in this closure.

## Findings and changes

| Area | Finding | Current resolution |
| --- | --- | --- |
| Evidence reviewer | WGL compute support was incorrectly required to be false | Use actual complete device facts and fixed-fixture requirements; either supported execution or an evidenced pre-submission refusal follows those facts on every backend |
| Renderer | Fixture allocations, bindings, texture limits and dynamic uniform offsets were incompletely checked | Private preflight validators run before cache creation, allocation, mapping and recording; negative tests vary every used capability and limit |
| Consumer | Only a small capability subset was retained | Exhaustive Rust destructuring records every capability/limit, exact matrix tuples, lane identities and wait routes; a new RHI field stops compilation until evidence is updated |
| Consumer Surface | Preferred presentation attributes were assumed | Select advertised attributes and record the complete capabilities/configuration plus exact `configuration_support` result before configuration |
| WGL diagnostics | Core debug support used a fixed version-prefix list | Numeric version comparison or `GL_KHR_debug` admits the route, with the basis retained in both context observations |
| RHI CAP-003 | Ordinary storage resources and layouts lacked capability gates | Fixed in RHI Core before backend support, creation and layout calls, with negative SPI-entry tests |
| RHI CAP-004 | Encoder native interop did not check the device capability | Fixed by retaining and checking that capability in the encoder before backend interop |
| RHI CAP-005 | Query and pass-statistics admission lacked negotiated capability checks | Fixed by requiring both the relevant capability and the backend's exact support predicate |
| Metal statistics | Pass-statistics fields were implemented and queried but the capability was omitted | Fixed by deriving the capability from the native counter-field probe; generic query support remains independently checked |

The Rendering-side review compares the complete runtime JSON schema to the
actual Cargo-built RHI declarations, validates types and alias consistency,
and checks lane device provenance and route targets. Fixed-compute admission
checks are retained before pipeline creation and independently recomputed from
the raw capabilities. The reviewer does not infer support from a backend name.

The RHI fixes introduce no new public feature-negotiation API. Historical native
evidence remains identified by its captured source and does not certify the new
revision. Current integration evidence must establish the named DX12/WGL paths.

The pushed Core passed 129 tests; two GL feature-route tests and an Apple-target
Metal compile check also passed. These are source/contract checks, not an Apple
GPU run. Raw logs are retained under `target/reviews/20261102-capabilities/`.
