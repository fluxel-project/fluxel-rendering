# RHI hardware conformance cases

This directory owns Fluxel RHI's real-hardware baseline. Every portable
workload has exactly one implementation under `common/`; a backend test is
only a fixture that supplies code-form-specific shader artifacts, selects a
compatible lane, or owns native surface-host glue. Public applications and
examples construct process-owned providers through `create_*_provider`; only
the in-crate hardware fixture may use the backend-private constructor when it
needs a deliberately controlled native surface/context.

`common/mod.rs` and `harness/` are included by the library's native test build
so they can accept backend-injected shader/surface fixtures while still using
the same public provider/device API. `harness` owns async completion/readback policy and the
Pass/Unsupported/Skipped/Failure vocabulary; it never contains a native handle
or a backend name.
`common/cases/` owns portable recording, submission, completion and CPU
assertion behaviour. `fixtures/` and the thin modules under each backend own
only the private bridge. New cross-backend cases belong in `common/cases/`
first.

| Case | Portable owner | Fixture responsibility |
| --- | --- | --- |
| Empty occlusion query → resolve → readback | `common` | provider/device, compatible lane, executor |
| Indirect compute → readback | `common` | shader artifact, pipeline/bind group, compatible lane |
| Device identity and core resource creation | `common/cases/core_device.rs` | provider/device availability classification |
| GL-family core creation and empty submission | `common/cases/core_device.rs` | adopted WGL/EGL/WebGL owner and host drawable |
| Buffer / texture transfer | `common/cases/transfer.rs` | device and exact capability gate |
| Storage-buffer compute and offscreen raster | `common/cases/compute.rs`, `common/cases/raster.rs` | code-form-specific shader fixture |
| Depth/stencil | `common/cases/depth_stencil.rs` | compatible format and shader fixture |
| Surface lifecycle | `common/cases/presentation.rs` | HWND/ANativeWindow/CAMetalLayer/canvas target registration, configuration chosen from fresh facts, logical lane |

## Baseline matrix

The baseline is intentionally enumerated here so a backend cannot claim broad
conformance merely because its provider can be constructed.  A fixture should
wire each row to the shared case (or report `Unsupported` from the published
capability facts):

| ID | Workload | Shared owner / gate |
| --- | --- | --- |
| 01 | triangle | `cases/raster.rs`, raster + RGBA8 |
| 02 | indexed triangle | `cases/raster.rs`, index-buffer route |
| 03 | uniform + texture + sampler | `cases/compute.rs`/binding fixture, sampled-texture facts |
| 04 | storage-buffer compute | `cases/compute.rs`, storage-buffer + dispatch |
| 05 | texture upload / copy / readback | `cases/transfer.rs`, copy-src/dst |
| 06 | depth + stencil | `cases/depth_stencil.rs`, depth/stencil attachment facts |
| 07 | MSAA + resolve | raster resolve route and sample-count capability |
| 08 | indirect draw / dispatch | indirect draw/dispatch facts and argument validation |
| 09 | occlusion query | query begin/end/resolve/readback |
| 10 | multiview | multiview view-count and attachment route |
| 11 | mapping | asynchronous `ReadbackTicket::read` RAII lease |
| 12 | BC / ETC2 / ASTC compressed textures | per-format support and block-aware transfer |
| 13 | presentation / resize / recreate | `cases/presentation.rs`, host target lifecycle |

Rows 01, 04, 05, 06, 08, 09, 11 and 13 already have executable native
fixtures in this tree.  Rows 02, 03, 07, 10 and 12 share the same public API
and are capability-gated; a backend must add its code-form-specific shader or
surface fixture before advertising them as hardware evidence.  Keeping the
matrix explicit prevents a small green smoke set from being mistaken for full
wgpu-style conformance.

Each case records portable work, submits it to a real adapter, waits for a
terminal completion, and asserts a readback or surface lifecycle outcome.
Shader artifacts intentionally remain backend-specific (DXIL, SPIR-V, MSL,
WGSL, or GLSL): sharing a source form would test a compiler or translator,
not the selected backend lowering.

The public case must distinguish these outcomes precisely:

- `Pass`: an advertised capability completed and met its CPU assertion.
- `Unsupported`: the exact capability/format/route is not published; the case
  is not run and no success is fabricated.
- `Skipped`: the required native loader, adapter, or host surface is absent.
- `Failure`: an advertised route rejected the case, lost the device, failed, or
  produced wrong readback data.

## Running a baseline

On Windows, DX12 and Vulkan headless cases are ordinary crate tests. The exact
filters evolve with the suite, so start with:

```powershell
cargo test -p fluxel-rhi --all-features backend::dx12 -- --nocapture
cargo test -p fluxel-rhi --all-features backend::vulkan -- --nocapture
cargo test -p fluxel-rhi --features native-gl-wgl backend::gl::native::wgl_core_conformance -- --nocapture
```

The WGL fixture is intentionally an adopted-context case: it creates its Host
window and private WGL worker itself, then crosses only the public provider and
device boundary into `common`. GLES and WebGL2 must use the same fixture shape
through their respective owner (EGL / browser canvas); they must not obtain
coverage by exporting an EGL context or `WebGl2RenderingContext` from RHI.

The release evidence gate is [`../../../scripts/conformance.ps1`](../../../scripts/conformance.ps1).
It records the commit, target, adapter and driver; use it for a claim of
hardware evidence rather than treating a local green unit-test run as one.

Android and browser fixtures require their respective host adapters. See
`../../../examples/android-vulkan-wsi/`, `../../../scripts/android_*_evidence.py`,
and `../../../scripts/browser/`; a desktop result does not substitute for them.
Metal fixtures compile cross-target on Windows but execute only on an Apple
host.

Result labels and the distinction between portable, headless, and presentation
evidence are fixed by [ADR-0022](../../../documents/adr/0022-hardware-conformance-evidence.md).
