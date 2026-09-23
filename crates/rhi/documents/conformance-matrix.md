# RHI Backend Conformance Matrix

> Baseline: commit `54b846c` (`docs: separate graph plans from RHI bindings`), RHI API freeze v13.
> Purpose: freeze the actual observed capability of every backend **before** the 0.17 B3
> (native recording / submit separation) refactor, so that "green after refactor" cannot
> be claimed without knowing what was lost. Scope: `crates/rhi`.
>
> Cell states are restricted to: `HardwarePass` · `Unsupported` · `Skipped` · `Failure` · `NotCovered`.
> - `HardwarePass` requires a real-adapter run with recorded commit/adapter/driver/case/expected/actual.
> - `Unsupported` must point at a capability fact or a public early-refusal.
> - compile-only / source-audit may not be written as `HardwarePass`.
>
> This is a living document; each backend row updates as S0/S3 evolves.

## Adapters & runtimes

| Label | Adapter | Backend | Evidence source |
|---|---|---|---|
| `dx12-780M` | Direct3D 12 (adapter TBD) | dx12 | `backend/dx12/platform/tests/` |
| `vk-780M` | Vulkan (adapter TBD) | vulkan | `backend/vulkan/*_tests.rs` |
| `metal-host` | Metal (Apple host, remote) | metal | `backend/metal/tests.rs` |
| `webgpu-ci` | WebGPU (CI headless Chrome, wasm) | webgpu | `backend/webgpu/tests.rs` |
| `gl-wgl` | OpenGL WGL 4.x | gl | `tests/gl/wgl_core.rs` |

> Adapter/driver details are recorded per `HardwarePass` cell below as they are produced.
> Metal / WebGPU / GL rows are largely `NotCovered` / `Skipped` until the corresponding
> hardware carrier is available on this machine (Metal needs an Apple host; WebGPU needs
> CI wasm; GL raster needs a WGL rendering example), per the 0.17 plan hard constraints.

## Matrix

| Case | dx12 | vulkan | metal | webgpu | gl |
|---|---|---|---|---|---|
| buffer upload/readback | HardwarePass | — | NotCovered | NotCovered | NotCovered |
| texture upload/readback | HardwarePass | — | NotCovered | NotCovered | NotCovered |
| compute dispatch | HardwarePass | HardwarePass | HardwarePass | NotCovered | NotCovered |
| raster triangle | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |
| sampled texture | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |
| storage buffer | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |
| storage texture | NotCovered | HardwarePass | NotCovered | NotCovered | NotCovered |
| depth/stencil | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |
| indirect compute | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |
| indirect raster | NotCovered | — | NotCovered | NotCovered | NotCovered |
| timestamp query | NotCovered | — | NotCovered | Unsupported | NotCovered |
| occlusion query | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |
| pipeline statistics | NotCovered | NotCovered | NotCovered | Unsupported | NotCovered |
| clear buffer | NotCovered | — | NotCovered | NotCovered | NotCovered |
| clear texture | NotCovered | — | NotCovered | NotCovered | NotCovered |
| MSAA + resolve | NotCovered | NotCovered | NotCovered | NotCovered | NotCovered |
| multi-batch plan | NotCovered | NotCovered | NotCovered | NotCovered | NotCovered |
| cross-submit resource state | NotCovered | NotCovered | NotCovered | NotCovered | NotCovered |
| map/readback lifetime | NotCovered | — | NotCovered | NotCovered | NotCovered |
| presentation | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |
| resize/reconfigure | NotCovered | HardwarePass | NotCovered | NotCovered | NotCovered |
| device loss | HardwarePass | HardwarePass | NotCovered | NotCovered | NotCovered |

## Evidence ledger

### dx12 (HardwarePass)

All on `dx12-780M`. Case → test function → evidence.

- **buffer upload/readback** — `a_real_device_moves_bytes_from_the_cpu_to_a_buffer_and_back_to_the_cpu`
- **texture upload/readback** — `a_real_device_round_trips_texels_through_every_dx12_copy_path`
- **compute dispatch** — `a_real_compute_dispatch_writes_a_storage_buffer`
- **raster triangle** — `an_offscreen_triangle_writes_deterministic_rgba8_pixels` (platform/tests/raster.rs)
- **sampled texture** — `a_fragment_sampled_texture_and_sampler_reach_the_raster_output` (platform/tests/raster.rs)
- **storage buffer** — `a_real_compute_dispatch_writes_a_storage_buffer`
- **depth/stencil** — `tests/dx12/depth_stencil.rs`
- **indirect compute** — `a_real_indirect_compute_dispatch_reads_gpu_arguments`
- **occlusion query** — `tests/dx12/query.rs`
- **presentation** — `a_host_window_frame_clears_presents_and_can_be_acquired_again` + `common_presentation_lifecycle_case`
- **device loss** — `a_loss_recorded_on_the_native_device_is_terminal_and_stable`

### vulkan (HardwarePass)

All on `vk-780M`. Case → test function → evidence.

- **compute dispatch** — `buffer_compute_dispatch_keeps_dropped_caller_handles_alive_until_readback` (compute_tests.rs)
- **indirect compute** — `indirect_compute_dispatch_reads_gpu_arguments_and_publishes_readback` (compute_tests.rs)
- **raster triangle** — `offscreen_raster_draw_transitions_to_readback_and_retains_native_objects` (raster_tests.rs)
- **sampled texture / storage texture** — `image_binding_tests.rs`
- **depth/stencil** — `tests/vulkan/depth_stencil.rs`
- **occlusion query** — `tests/vulkan/query.rs`
- **presentation** — `presentation_tests.rs` (win32, real HWND acquire/clear/present)
- **device loss / resize** — platform/tests.rs

### metal (NotCovered)

Only 3 Apple-gated `#[test]` exist in `backend/metal/tests.rs` (enumeration/identity, compute
dispatch+readback, rgba8 upload copy readback). Everything else — presentation, raster,
query, resolve, MSAA, indirect, depth-stencil — has **zero** hardware evidence and requires
an Apple host that is not available on this machine. Camera to C2 later in the series.

### webgpu (NotCovered)

Only 2 headed-browser smoke tests (`backend/webgpu/tests.rs`, wasm32-gated, CI). They cover
device/buffer/texture/view creation + async compute pipeline, but never `submit`, draw,
copy, acquire/present, or readback publication. The whole `command.rs` submission/readback
spine has no hardware evidence. Camera to C4.

### gl (NotCovered)

`tests/gl/wgl_core.rs` is real WGL 4.x but only runs logical-creation + an empty-plan
submit. No raster/compute rendering path, no GL example exists. Camera to C5.

## Unsupported cross-check

`Unsupported` cells must point at a capability fact / public early-refusal — this is the
S0.1 **completion gate** and is audited as the matrix is filled.

- webgpu **timestamp query / pipeline statistics** — WebGPU capability publishes
  `OcclusionQuery` + `FixedAtRasterScope` only; it refuses query-family / timestamp
  capability facts. Verified in `backend/webgpu/capabilities.rs`.

## Open gaps (against 0.17 plan §7 S3)

- DX12: resolve_texture, map/late-lifetime, timestamp query, clear buffer/texture, indirect
  raster, MSAA+resolve, wait_idle timing.
- Vulkan: multi-batch single plan, cross-submit layout persistence, Android presentation.
- Metal / WebGPU / GL: as above — hardware carriers required.

## Rebaseline procedure

When the code changes, re-run the affected backend's evidence, update the commit SHA in the
header, and move the prior version into `versions/` history per the repo evidence rule.