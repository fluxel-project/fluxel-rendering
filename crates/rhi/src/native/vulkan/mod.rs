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
//!    and staging kinds, with the staging path the immutable uploads use. The
//!    dependency is direct now (`gpu-allocator` 0.28 behind the `vulkan` feature,
//!    decision E) rather than only reachable through the borrowed layer, and its
//!    `AllocatorCreateDesc` takes `instance`, `device` and `physical_device` **by
//!    value**, so the allocator is built from those handles and must outlive every
//!    allocation it returns. That is why it belongs beside the resource table that
//!    owns them rather than inside one resource.
//!    `memory::types` and `memory::select` already decide *which* type index a
//!    resource uses; this step binds memory to the handle.
//!    The calls to reach for, read from `gpu-allocator` 0.28 rather than guessed:
//!    `vulkan::Allocator::new(&AllocatorCreateDesc { .. })`, then
//!    `allocate(&AllocationCreateDesc { name, requirements, location, linear,
//!    allocation_scheme })` — those five are the whole Vulkan descriptor,
//!    `requirements` being the `MemoryRequirements` the driver reported for the
//!    handle being bound, and `AllocationScheme::GpuAllocatorManaged` the
//!    suballocating choice. `MemoryLocation` is `Unknown | GpuOnly | CpuToGpu |
//!    GpuToCpu`, which is `memory::MemoryPurpose` one to one:
//!    `DeviceLocal -> GpuOnly`, `UploadStaging -> CpuToGpu`,
//!    `ReadbackStaging -> GpuToCpu`. The returned `Allocation` reports `memory()`,
//!    `offset()` and `size()`, which are what the bind call needs and what the
//!    allocator's matching free consumes — so an allocation must not be freed
//!    through a different allocator, and the resource table has to own both.
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
//!     `vkGetPhysicalDeviceFormatProperties` proved it. The device owns that table,
//!     because the storage-image row's resource half is read from it when the ledger
//!     is recorded -- the ledger is captured at creation and never recomputed, so a
//!     fact discovered later could not reach it.
//!
//!     The occlusion and elapsed query rows are recorded from facts the open path
//!     already reads: occlusion is a core query type on the graphics family the
//!     device was created on, and elapsed is that family's own timestamp-valid-bit
//!     report, because an elapsed interval on `Vulkan` is two timestamp writes and
//!     their difference rather than a separate query type. Every row this step can
//!     prove is proved; the rows that stay absent do so for the reasons `device`'s
//!     ledger function records, and the two draw-parameter rows (`BaseVertex`,
//!     `FirstInstance`) belong to step 12, because they arrive with the draw verbs
//!     whose parameter space they gate.
//!
//! 12. **Raster recording.** The pass bracket on the recording encoder, the
//!     framebuffer an admitted [`render_pass`] description needs, and the draw
//!     verbs: pipeline and binding selection, vertex and index buffers, the dynamic
//!     viewport and scissor, and the non-indexed and indexed draws. This is the step
//!     the two draw-parameter rows step 11 owes arrive with -- they are families
//!     whose parameter space `GraphicsApi`'s draw verbs deliberately cannot name
//!     (plan section 20.1) -- and it is what lets `VulkanDevice` implement
//!     `Provides<Graphics>`. Five bounded pieces have landed: the pure [`render_pass`]
//!     lowering; [`framebuffer`] with the [`command::Encoder`] bracket
//!     (`begin_raster` / `end_raster`) that owns the render pass and framebuffer a
//!     recorded pass needs, refusing a barrier, a copy or an `end` while a pass is
//!     open; the raster state and draw verbs (`set_raster_pipeline`,
//!     `set_vertex_buffer`, `set_index_buffer`, `set_viewport`, `set_scissor`,
//!     `draw`, `draw_indexed`) lowered through [`draw`], which also records the Y
//!     flip the borrowed path preserves and is why the device verifies and enables
//!     `VK_KHR_maintenance1`; and [`bind_group`] with the [`command::Encoder`]
//!     `set_bindings` verb, which owns the `VkDescriptorPool` and `VkDescriptorSet`
//!     the retained textured layout fills and refuses a dynamic binding this
//!     vocabulary cannot supply; and the two draw-parameter rows (`BaseVertex`,
//!     `FirstInstance`) step 11 handed to it, recorded from the core `Vulkan` 1.0
//!     draw parameters -- an indexed draw's `vertexOffset` and a draw's
//!     `firstInstance` -- with the family markers that let a graph require them.
//!     `GraphicsApi`'s draw verbs keep both at zero, because a non-zero value is a
//!     separate family's parameter space (plan section 20.1) and the verbs that name
//!     it arrive with a consumer. Step 12 is complete.
//!
//! 13. **The family wiring.** `VulkanDevice` owns the resource table and the one
//!     command pool and implements the family providers this backend can serve,
//!     handing out one handle type per family: `Provides<Graphics>` yields
//!     [`family::GraphicsRecording`] (`FamilyApi` + `GraphicsApi`), `Provides<Copy>`
//!     yields [`family::CopyRecording`] (`FamilyApi` + `CopyApi`),
//!     `Provides<Compute>` yields [`family::ComputeRecording`] (`FamilyApi` +
//!     `ComputeApi`), `Provides<StorageBuffer>` yields
//!     [`family::StorageBufferBindings`] (`FamilyApi` + `StorageBufferApi`) and
//!     `Provides<StorageTexture>` yields [`family::StorageTextureBindings`]
//!     (`FamilyApi` + `StorageTextureApi`), each over that table and -- for the three
//!     command families -- its own recording.
//!     The handles are distinct types so a caller that negotiated one family cannot
//!     reach another's verbs, and each is its own recording context, which the
//!     contract permits (plan section 21). The compute bracket is this layer's own --
//!     `Vulkan` has no compute-pass command -- and the one recording encoder tracks it
//!     in the same single pass slot the raster bracket uses. Composing two families
//!     into one submission is the execution-layer migration's decision and is
//!     deliberately not invented here.
//!
//!     The storage-buffer family was the first *resource role* wired, and it is where
//!     the compute handle's transitions arrive: a dispatch that reads or writes a
//!     storage binding is the first compute command whose resources need ordering, so
//!     [`family::ComputeRecording::transition_buffer`] /
//!     [`family::ComputeRecording::transition_texture`] land beside it. The
//!     storage-texture role followed it: its handle is the same shape (a table lookup
//!     and a validated value, with no recording at all), and the facts it adds are the
//!     texture's own declared usage and the device's per-format storage evidence,
//!     which is why step 11's format table moved onto the device. The
//!     indirect-dispatch family and the execution-layer migration remain owed by this
//!     step. This step was added when the list above was exhausted, which is section
//!     23.1's rule moving to the W2 work package; it is what makes
//!     `require::<_, Graphics>(&device)`, `require::<_, Copy>(&device)`,
//!     `require::<_, Compute>(&device)`, `require::<_, StorageBuffer>(&device)` and
//!     `require::<_, StorageTexture>(&device)` compile and run against this backend.
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

/// Step 11's per-format half: the portable format lowered onto the facts one
/// `vkGetPhysicalDeviceFormatProperties` answer proves, and the common evidence
/// table those facts are recorded in. A format fact is recorded only where the
/// driver was asked about that format.
pub(crate) mod format_facts;

/// Step 11's lowering half: the discovery results this backend owns -- the ledger
/// `require` reads, the adapter's numeric facts and [`format_facts`]'s evidence
/// table -- lowered onto `fluxel_rendergraph::DeviceCapabilities`. It reports only
/// what discovery proved, so the remaining ledger rows are what widens it.
pub(crate) mod capability;

/// Step 11's storage half: the two shader-store device features the
/// `StorageBuffer` row needs, requested only where the adapter reported them, and
/// the fact that a device created with both proves the row. A feature the adapter
/// did not report stays disabled, because enabling it would fail device creation
/// rather than leave the row unproved.
pub(crate) mod features;

/// Step 3''s second half: suballocation and binding memory to a handle.
pub(crate) mod allocator;

/// Step 4's pure half: portable texture descriptions lowered onto Vulkan.
pub(crate) mod texture;

/// Step 4's sampler half: the portable sampler descriptor lowered onto Vulkan,
/// including the one rule a nullable comparison needs.
pub(crate) mod sampler;

/// Step 4's owning half: the resource table, which holds handles (buffers, images,
/// the views sampled through them and samplers), allocations and identities together
/// so memory cannot be freed through the wrong allocator.
pub(crate) mod resource;

/// Step 5's shader half: a SPIR-V module and the pipeline stage that names it.
pub(crate) mod shader;

/// Step 5's descriptor half: the descriptor set layout, lowered from the common
/// bind-group layout vocabulary and owned by the pipeline layout built over it.
pub(crate) mod descriptor;

/// Step 12's binding selection: the `VkDescriptorPool` and `VkDescriptorSet` the
/// common bind-group value vocabulary fills, owned together so destroying the pool
/// is the set's whole release, and validated against the pipeline layout's own set
/// layout before the driver is reached.
pub(crate) mod bind_group;

/// Step 5's pipeline half: the pipeline layout, the compute pipeline built over it,
/// and the raster pipeline lowered from the common fixed-function vocabulary, with
/// each shader module destroyed as soon as creation has read it.
pub(crate) mod pipeline;

/// The raster pass the draw path begins: the portable attachment set admitted under
/// plan section 4's one-colour-at-index-zero / no-depth-stencil rule and lowered onto
/// the `VkRenderPass` description the recording pass is built from. The colour
/// attachment's description is shared with [`pipeline`]'s creation pass, because
/// `Vulkan` compares two render passes by exactly those facts. Pure: it creates
/// nothing, and its owning half lands with the draw verbs.
pub(crate) mod render_pass;

/// Step 12's owning half: the `VkRenderPass` and `VkFramebuffer` one admitted
/// [`render_pass`] description needs, built together because `Vulkan` makes the
/// framebuffer refer to the render pass, and owned together so field order is that
/// dependency. It refuses a subresource range and an attachment shape a framebuffer
/// cannot carry before the driver is reached, and the recording bracket that begins it
/// lives on [`command::Encoder`].
pub(crate) mod framebuffer;

/// Step 12's pure half: the dynamic viewport and scissor, the index-buffer format
/// and the half-open vertex and index ranges lowered onto the commands the raster
/// recorder issues. The viewport keeps the borrowed path's Y flip, which is why the
/// device enables `VK_KHR_maintenance1`: a negative viewport height is legal only
/// with that extension on a `Vulkan` 1.0 device.
pub(crate) mod draw;

/// W2's compute family, pure half: the workgroup counts one `vkCmdDispatch` takes,
/// with every zero dimension refused before the driver is reached because `Vulkan`
/// would accept it as a legal no-op. The bracket and the commands that use it live
/// on [`command::Encoder`]; the family handle that negotiates them lives in
/// [`family`].
pub(crate) mod compute;

/// Step 13: the family wiring. `VulkanDevice` implements `Provides<Graphics>`,
/// `Provides<Copy>`, `Provides<Compute>`, `Provides<StorageBuffer>` and
/// `Provides<StorageTexture>`, handing out [`family::GraphicsRecording`],
/// [`family::CopyRecording`], [`family::ComputeRecording`],
/// [`family::StorageBufferBindings`] and [`family::StorageTextureBindings`], which
/// implement [`FamilyApi`](crate::common::api::handle::FamilyApi) and their own
/// family's trait over the device's own resource table and (where the family records)
/// their own recording. The handles are distinct types so one family's verbs cannot be
/// reached from the other's call site, and `require::<_, Graphics>(&device)`,
/// `require::<_, Copy>(&device)`, `require::<_, Compute>(&device)`,
/// `require::<_, StorageBuffer>(&device)` and `require::<_, StorageTexture>(&device)`
/// are the real negotiation path for this backend rather than only a contract test.
pub(crate) mod family;

/// Step 13's storage-role half, pure: the two facts one storage binding is admitted
/// against -- the buffer range checked against the buffer's own declared size, and the
/// texture whose declared usage and `(format, sample count)` facts permit a storage
/// binding. It creates nothing, and the family handles that build the bindings live in
/// [`family`].
pub(crate) mod storage;

/// Step 6: the retained WGSL artifact lowered to SPIR-V with Naga's `spv-out`, with
/// the dialect, entry-point, stage and profile checks decided before the driver is
/// reached. Caller-supplied SPIR-V stays a passthrough and meets this path in
/// [`wgsl::create_module`].
pub(crate) mod wgsl;

/// Step 7's pure half: RenderGraph's access states lowered onto the pipeline-stage,
/// access and layout facts one `vkCmdPipelineBarrier` needs, with a same-state
/// transition kept a barrier rather than becoming a no-op.
pub(crate) mod barrier;

/// Step 7's owning half: the command pool on the device's selected queue family and
/// the one recording encoder, which records exactly the barriers [`barrier`] builds.
pub(crate) mod command;

/// Step 8's pure half: the portable buffer and texture copy regions lowered onto
/// `Vulkan` copy records, with the alignment, bounds and layer checks repeated at
/// the boundary that reaches the driver.
pub(crate) mod copy;

/// Step 9's first half: one unsignaled fence per execution, one submit on logical
/// queue 0, and the completion state machine over `common`'s disposition and
/// lifetime rules, with accepted-unknown work quarantined rather than released.
pub(crate) mod submission;

/// Step 10's pure halves: the instance extensions the surface path verifies before
/// creating anything, and the fixed presentation contract decided against one
/// surface's reported formats, present modes and capabilities, lowered into the
/// swapchain create-info the driver is handed. It creates and owns nothing, so
/// every refusal is provable without a window.
pub(crate) mod surface;

/// Step 10's owning half: the `VkSurfaceKHR` created from and bound to a host
/// window through a surface-capable instance, with the parent/child order stated as
/// a borrow; and the facts that surface reports -- the capabilities, formats and
/// present modes the pure contract decides against, plus the per-family
/// presentation-support answer step 2's queue rule has to be told. Reconfigure and
/// the recovery of a poisoned surface remain owed by step 10.
pub(crate) mod presentation;

/// Step 10's acquire half, pure: what `vkAcquireNextImageKHR`'s answer means, and
/// the quarantine rule an unpresented acquire enforces. The lease that owns the
/// acquired image and its semaphore lives beside the swapchain that produced it.
pub(crate) mod acquire;

/// Step 10's present half, pure: what `vkQueuePresentKHR`'s answer means, with a
/// suboptimal present carried as a value rather than folded into a refusal and an
/// out-of-date swapchain kept distinct from a lost surface. The call that consumes
/// an [`acquire`] lease and retains the wait semaphore it used lives in
/// [`swapchain`].
pub(crate) mod present;

/// Step 10's swapchain half: the `VkSwapchainKHR` created from the fixed contract
/// over a surface, and the images `Vulkan` creates with it. It is reachable only
/// through [`device::SwapchainDevice`], the device that verified and enabled
/// `VK_KHR_swapchain`, and it checks the selected queue family's presentation
/// support before the driver is reached. The acquire lease is landed beside it: a
/// real `vkAcquireNextImageKHR` whose unpresented drop poisons the surface and
/// retains the acquire semaphore. Present is landed too: a real `vkQueuePresentKHR`
/// consumes that lease, and the semaphore it waited on is retained until the
/// presentation engine hands the image back, because returning from present is not
/// proof its wait is consumed. Reconfigure is landed too: a real
/// `vkCreateSwapchainKHR` over `old_swapchain` that retires its predecessor through
/// the same teardown and returns a live replacement, refusing a quarantined surface
/// by name before the driver is reached. Step 10 is complete.
pub(crate) mod swapchain;

/// Test-only scaffolding shared by the modules that need a real window or surface.
#[cfg(test)]
pub(crate) mod test_support;
