//! The Vulkan backend, implemented on `ash` with `gpu-allocator`.
//!
//! This module is the W2 work package. It is empty of code by design: W2 is a
//! from-scratch backend, and writing its types before its first real call path
//! exists would be the pre-designed abstraction the plan's third rule forbids.
//! What follows is the ordered step list W2 executes, recorded here so the work
//! starts from the plan rather than from re-derivation.
//!
//! # W2 step list
//!
//! 1. **Instance and validation probe.** Dynamic `ash` loading, instance
//!    extensions, and the probe `Validation::Required` depends on: enumerate
//!    layers and instance extensions and positively confirm a usable
//!    `VK_LAYER_KHRONOS_validation` plus `VK_EXT_validation_features` *before* a
//!    device is returned. `Validation::Disabled` is never evidence that validation
//!    was active.
//! 2. **Adapter enumeration and device open.** Physical-device facts into
//!    [`crate::common::base::stamp::DeviceStamp`] and
//!    [`crate::common::caps::AdapterLimits`]; required limits; one queue. The
//!    adapter index is a bootstrap choice, not an adapter-policy API.
//! 3. **Memory.** `gpu-allocator` for suballocation. Device-local, host-visible
//!    and staging kinds, with the staging path the immutable uploads use.
//! 4. **Resources.** Buffers, textures, texture views and samplers, each created
//!    with the usage set the portable descriptor asked for and no more, and each
//!    stamped with a [`crate::common::base::resource::ResourceId`].
//! 5. **Descriptors and pipelines.** Bind group layouts, pipeline layouts, and
//!    compute and raster pipelines. Shader modules from SPIR-V.
//! 6. **Shaders.** Naga `spv-out` for the retained WGSL artifacts (decision F's
//!    passthrough rule means caller-supplied SPIR-V goes straight in). The dialect
//!    and profile check happens before driver shader creation.
//! 7. **Recording.** One command encoder with explicit
//!    `vkCmdPipelineBarrier` transitions driven by RenderGraph's access states. A
//!    `before == after` transition is still a memory dependency, never a discarded
//!    no-op.
//! 8. **Copies.** `vkCmdCopyBuffer` / `vkCmdCopyBufferToImage` /
//!    `vkCmdCopyImageToBuffer` / `vkCmdCopyImage`, with the alignment and range
//!    checks the safe layer already performs repeated at the boundary.
//! 9. **Submission and completion.** Fences, one submit per graph execution on
//!    logical queue 0, and the completion state machine
//!    ([`crate::common::base::lifetime`], [`crate::common::base::submission`]).
//! 10. **Surface.** `VkSurfaceKHR` from the host's display/window handles,
//!     configuration, acquire lease, present, reconfigure, and the quarantine
//!     rule: because `wgpu-hal` 30 cannot recycle an unpresented Vulkan acquire
//!     semaphore, dropping such an image poisons and quarantines that surface
//!     rather than guessing it can be reused. That rule is a **behavior this
//!     backend must preserve**, not a bug to fix (plan section 4).
//! 11. **Capability lowering.** Physical-device facts into the ledger, including
//!     the per-format evidence table (`sampled`, `filterable`, `renderable`,
//!     `blendable`, `storage_read`, `storage_write`, `copy_source`,
//!     `copy_destination`) that the GL family already models and that W9 will
//!     unify. A format fact is recorded only where
//!     `vkGetPhysicalDeviceFormatProperties` proved it.
//!
//! # Acceptance
//!
//! W2 closes when the W1-frozen oracle passes **on Vulkan** with
//! `Validation::Required` and programmatically empty diagnostics on the named
//! Windows board. That half runs on the owner's machine:
//!
//! ```powershell
//! ./scripts/conformance.ps1
//! ```
//!
//! # What must not change
//!
//! The preserved-semantics table in plan section 4. The two entries this backend
//! can silently lose are the unpresented-acquire quarantine (step 10) and
//! `Validation::Required` fail-closed verification (step 1): both are refusals
//! rather than features, and a fresh implementation is tempted to be permissive.

/// The fail-closed probe step 1 depends on; pure, so it is provable without a
/// device.
pub(crate) mod validation;

/// The FFI half of step 1: reading the instance-level names.
pub(crate) mod inventory;

/// Step 1's instance creation, with the fail-closed order it preserves.
pub(crate) mod instance;

/// Step 2's first half: which adapters exist, and what each reports.
pub(crate) mod adapter;

/// Step 2's second half: the logical device and its one queue.
pub(crate) mod device;

/// Step 3's pure half: which memory type a resource may be placed in.
pub(crate) mod memory;

/// Steps 1 and 2 as one entry point: opening a headless device.
pub(crate) mod open;

/// Step 4's pure half: portable buffer usage lowered onto Vulkan flags.
pub(crate) mod buffer;

/// Step 4's pure half: portable formats lowered onto Vulkan image formats.
pub(crate) mod format;

/// Step 4's pure half: portable texture descriptions lowered onto Vulkan.
pub(crate) mod texture;
