# Android Vulkan evidence

`scripts/android_vulkan_evidence.py` runs the RHI's native Vulkan conformance
tests on an attached Android device without creating an Android surface.  It is
deliberately a **test executable harness**, not a second public Vulkan-provider
wrapper: `VulkanProvider` is backend-private, and exporting it merely to make
an example compile would weaken the RHI ownership boundary.

The runner cross-compiles `fluxel-rhi`'s library test binary for the device,
installs that one binary through `adb`, and runs these closed vertical slices:

| Slice | RHI conformance test |
| --- | --- |
| adapter enumeration and device request | `loader_enumeration_request_identity_and_idle` |
| buffer upload/copy/readback | `upload_copy_and_readback_move_bytes_on_a_real_vulkan_queue` |
| texture upload/copy/readback | `texture_upload_copy_and_readback_move_texels_on_a_real_vulkan_queue` |
| compute dispatch and lifetime retention | `buffer_compute_dispatch_keeps_dropped_caller_handles_alive_until_readback` |
| offscreen raster and readback | `offscreen_raster_draw_transitions_to_readback_and_retains_native_objects` |
| sampled image, sampler, storage image | `compute_sampled_filtering_and_storage_images_round_trip_through_vulkan` |

This is intentionally a no-WSI gate.  It validates the Android Vulkan loader,
device, memory, descriptor, command and queue paths without claiming that
`VK_KHR_android_surface` presentation has been implemented.  Android WSI is a
separate backend slice and must add its own acquire/present conformance test
before it can be advertised.

From the repository root on an x86_64 Android emulator:

```powershell
python scripts/android_vulkan_evidence.py --serial emulator-5554
```

The machine-readable report is written to
`target/android-vulkan-evidence/report.json`.  It records the guest identity,
the Vulkan driver facts reported by `cmd gpu vkjson`, and the exact test result
for every vertical slice.  An emulator's advertised GPU profile remains
evidence for that guest/host Vulkan path, not a substitute for testing a
physical Android GPU.
