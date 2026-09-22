# Fluxel RHI 0.16

## Completed in this release

0.16 keeps the v13 public RHI contract frozen. The main work in this release
is cross-platform test and example infrastructure rather than a new API revision:

- Public provider-composition entry points: `create_dx12_provider`,
  `create_vulkan_provider`, `create_metal_provider`, and the wasm WebGPU
  entry point. Higher-level examples no longer access crate-private providers
  or native handles.
- A public conformance-workload matrix covering 13 categories: triangle,
  indexed triangle, uniform/texture/sampler, storage compute,
  transfer/readback, depth/stencil, MSAA/resolve, indirect, occlusion query,
  multiview, mapping, BC/ETC2/ASTC, and presentation resize/recreate. Each
  category has a fixed capability gate and `Pass`/`Unsupported`/`Skipped`/
  `Failure` outcome rule.
- Example entry points: `triangle`, `textured_cube`, `compute`, `msaa`,
  `indirect`, and `query`, plus the existing `offscreen_triangle`,
  `headless_compute`, `clear_present`, and `provider_probe`.
- Public GL/WGL fixture integration. GL/GLES/WebGL2 still use a host-owned
  context adapter; a native context does not enter the public RHI object model.
- DX12 completion ordering fix: an earlier unobservable signal is published in
  queue order when a later fence succeeds; transfer resources remain alive
  until their corresponding fence completes.
- Completed Metal/WebGPU branches in provider probing. The wasm path uses
  browser Promise scheduling and never blocks the browser thread.

## Test results

The following numbers come from commands actually run on this release worktree;
`cargo check` and no-run compilation are not counted as hardware passes.

| Environment | Command/entry point | Result |
| --- | --- | --- |
| API-only | `cargo test -p fluxel-rhi --no-default-features --lib` | **597 passed** |
| Windows Vulkan | `cargo test -p fluxel-rhi --no-default-features --features vulkan --lib -- --test-threads=1` | **610 passed, 0 failed** |
| Windows GL/WebGL backend logic | `cargo test -p fluxel-rhi --no-default-features --features webgl2 --lib -- --test-threads=1` | **727 passed, 0 failed** |
| Chrome headed WebGL2 wasm suite | `scripts/browser/wasm_test_headed.py --features webgl2` | **13 browser tests passed**, capture produced; screenshot evidence written under `target/evidence/` |
| Examples | `cargo check -p fluxel-rhi --features examples --examples` | passed for all registered examples |
| DX12 | full run attempted with DX12+Vulkan | initial GPU work passed, then adapter entered `DEVICE_REMOVED/DEVICE_PAUSED` after TDR; subsequent failures are terminal-device cascade and are not a clean DX12 verdict |

### Environments without a passing evidence record

- Chrome WebGPU headed suite: this release does not yet have a complete
  `test result: ok.` record.
- Android GLES 3.x / Vulkan: scripts and host fixtures exist, but this release
  contains no complete device-side pass log.
- macOS Metal: static compilation and API/logic checks are complete; Metal
  hardware cannot be executed from Windows.
- DX12: rerun after the system/driver recovers from TDR; the current terminal
  device state is not evidence about the implementation.

## Example results

All ten RHI examples compile using the public API. They use host-fixture
injection: a fixture can be implemented by `fluxel-host` or an application's
window/browser lifecycle layer, while the workload accepts only public types
such as `PlatformProvider`, `Device`, resources, and pipelines.

Examples with a complete public record/submit/readback path:

- `headless_compute`
- `offscreen_triangle`
- `clear_present`
- `provider_probe`

The new `triangle`, `textured_cube`, `compute`, `msaa`, `indirect`, and `query`
examples provide public fixture entry points for the same workloads. Backend
integration fixtures supply the shader code form, window target, and
capability-specific resources.

## Code changes at a glance

- `crates/rhi/examples/`: six public API examples and cross-platform usage notes.
- `crates/rhi/tests/`: the 13-case conformance matrix, a GL/WGL fixture, and
  public workload/capability-gate documentation.
- `crates/rhi/src/lib.rs`: public backend provider composition functions without
  exposing DXGI, Vulkan, Metal, or WebGL native handles.
- `crates/rhi/src/backend/dx12/command/spine.rs`, `failure.rs`, and
  `platform/device.rs`: fence-order completion and device-loss fixes plus
  regression coverage.
- `crates/rhi/src/backend/gl/native/mod.rs`: private WGL host-fixture owner path.
- Version notes record only proven hardware results; incomplete capabilities
  remain fail-closed and are not represented by templates or no-run builds.

## 0.16 boundary

The public API remains frozen at v13. Indexed raster, uniform+texture/sampler,
MSAA, multiview, compressed textures, and complete WebGPU/Android/Metal
hardware workloads need their own shader/WSI fixtures and device logs before
they can become hardware-conformance passes in a later release. This release
does not mark those missing records as Pass.
