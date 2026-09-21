# RHI example workloads

These are compile-checked, **host-injected workloads**. They deliberately do
not create DX12/Vulkan/Metal/WebGPU/GL objects themselves: Fluxel RHI keeps
native objects below the platform seam. Native applications can use the public
`create_dx12_provider`, `create_vulkan_provider`, or
`create_metal_provider` composition entry point where available, then inject
the resulting portable `PlatformProvider` and their host-owned surface into a
fixture.

Each template supplies a small fixture trait. The application implements it in
its platform integration, retaining native state there while returning only
portable RHI objects to the common workload.

```powershell
cargo check -p fluxel-rhi --features examples --examples
```

The workload binaries print their integration boundary when invoked without a
fixture; copy the corresponding source into a platform test/application and
invoke its `run(&fixture).await` function. `provider_probe` is a complete
public-API discovery example and can be run directly on a supported native
target.

The portable examples are the same small workloads used by the hardware
baseline, parameterised by a host-created public `PlatformProvider`.  The
provider constructors (`create_dx12_provider`, `create_vulkan_provider`,
`create_metal_provider`, and the wasm WebGPU equivalent) are public composition
functions; no example needs to name the crate-private backend provider.

1. **`headless_compute`** — creates a storage buffer, a compute pipeline and a
   bind group, dispatches one workgroup, reads back eight `u32` values, and
   verifies them on the CPU.
2. **`offscreen_triangle`** — uploads a triangle, renders it to an 8×8 RGBA8
   texture, reads the texture back, and checks the centre pixel.
3. **`triangle`** — minimal raster fixture entry point for a window or
   off-screen target.
4. **`textured_cube`** — indexed geometry plus texture/sampler binding.
5. **`compute`** — storage-buffer dispatch and CPU verification.
6. **`msaa`** — capability-gated multisample render/resolve.
7. **`indirect`** — capability-gated indirect draw or dispatch.
8. **`query`** — occlusion query begin/end, resolve and asynchronous readback.

They are deliberately not copies of wgpu's event-loop/framework examples:
their job is to show Fluxel's resource → recorder → `SubmissionPlan` →
completion/readback vocabulary. Concrete shader artifacts remain backend code
forms, so the runnable forms live in the matching hardware fixtures:

- DX12: `src/backend/dx12/platform/tests/mod.rs` and `tests/raster.rs`;
- Vulkan: `src/backend/vulkan/compute_tests.rs` and `raster_tests.rs`;
- Metal: `src/backend/metal/tests.rs`.

`clear_present` is the visible lifecycle counterpart. A host adapter supplies a
presentation target and configuration, after which it exercises configure →
acquire → `present_after` → submit → `wait_present` → reacquire/abandon, with
no native window/session/token type entering the RHI API.

The existing root-level Windows DX12 program predates v13 and is historical
evidence, not a v13 usage tutorial. Windowed application examples use
`fluxel-host` (or the browser host adapter) for lifecycle and surface ownership;
their RHI operations remain these same checked templates.
