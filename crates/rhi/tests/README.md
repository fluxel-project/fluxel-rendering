# RHI hardware conformance cases

This directory owns Fluxel RHI's real-hardware baseline. Every portable
workload has exactly one implementation under `common/`; a backend test is
only a fixture that constructs its deliberately crate-private provider/device,
supplies code-form-specific shader artifacts, selects a compatible lane, or
owns native surface-host glue. No test requires exporting a DX12, Vulkan,
Metal, browser, or GL context constructor from the public API.

`common/mod.rs` and `harness/` are included by the library's native test build
so they can accept backend-injected fixtures without making provider
construction public. `harness` owns async completion/readback policy and the
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
| Buffer / texture transfer | `common/cases/transfer.rs` | device and exact capability gate |
| Storage-buffer compute and offscreen raster | `common/cases/compute.rs`, `common/cases/raster.rs` | code-form-specific shader fixture |
| Depth/stencil | `common/cases/depth_stencil.rs` | compatible format and shader fixture |
| Surface lifecycle | `common/cases/presentation.rs` | HWND/ANativeWindow/CAMetalLayer/canvas target registration, configuration chosen from fresh facts, logical lane |

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
```

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
