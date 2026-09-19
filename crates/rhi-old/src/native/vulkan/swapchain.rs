//! Step 10's swapchain half: the `VkSwapchainKHR` and the images it owns.
//!
//! [`super::surface`] decides *whether* this backend's one presentation contract is
//! servable and lowers that decision into a `VkSwapchainCreateInfoKHR`;
//! [`super::presentation`] owns the `VkSurfaceKHR` and reads the facts that decision
//! consumes. This module is the half that owns the driver object those two describe:
//! it creates the swapchain, reads the images `Vulkan` created with it, and destroys
//! it in `Drop`.
//!
//! # The device must have been opened with `VK_KHR_swapchain`
//!
//! `vkCreateSwapchainKHR` exists only on a device that enabled `VK_KHR_swapchain`,
//! and `ash` substitutes a panicking stub for a function the loader did not resolve.
//! [`create`] therefore takes a [`SwapchainDevice`] -- the type only
//! [`super::device::open_with_swapchain`] produces after positively observing the
//! extension -- so "create a swapchain on a device opened for headless work" cannot
//! be written. That is the same witness rule the surface path states for
//! [`super::instance::SurfaceInstance`].
//!
//! # The queue family's presentation support is checked, not assumed
//!
//! Step 2 selected one queue family by rule and without a surface. Whether that
//! family can present to *this* surface is a fact about the pair, and it is read here
//! through [`super::presentation::Surface::supports_presentation`]. A family that
//! cannot present is refused by name **before** a swapchain exists, because
//! `Vulkan` would accept the creation and only fail at the first present.
//!
//! # Ownership is field order, as everywhere else in this backend
//!
//! A swapchain must be destroyed before the device that created it and before the
//! surface it presents to, and the images it owns are destroyed with it -- they are
//! not separately owned objects and must never be wrapped in the resource table,
//! which would try to free memory the swapchain owns. [`Swapchain`] holds both
//! parents by borrow, so "destroy the device or the surface first" is a compile error
//! rather than a comment asking politely.
//!
//! # What is deliberately not here
//!
//! - **No rendering submission.** [`AcquireLease::wait_semaphore`] is the acquire
//!   semaphore a submission drawing to this image must wait on, and
//!   [`AcquireLease::present`] waits on it directly because nothing has consumed it
//!   yet. When the draw-and-present path lands, present's wait set moves to the
//!   submission's render-finished semaphore; the retention rule below is about
//!   whichever semaphore present waited on, so it does not change.
//! - **No sRGB presentation, no format negotiation.** These are
//!   [`super::surface::contract`]'s decisions, and this module only consumes them.
//!
//! # Reconfigure consumes the predecessor, because the specification retires it
//!
//! [`Swapchain::reconfigure`] rebuilds the swapchain against the surface's *current*
//! facts and hands the old handle to the driver as `old_swapchain`. That field has a
//! consequence a caller must not miss: `Vulkan` retires the old swapchain **even if
//! the creation fails**, so after a reconfigure there is never a live predecessor to
//! fall back to. The method therefore takes `self`, and the old swapchain's teardown
//! is its own [`Drop`] -- device idle, retained present semaphores, then the handle --
//! which runs whether the rebuild succeeded or failed.
//!
//! Two things make the rebuild safe to write:
//!
//! - **The lease is the exclusion.** `reconfigure` takes `self`, so it cannot be
//!   called while an [`AcquireLease`] exists; `Vulkan`'s "no image of the old
//!   swapchain may be acquired" rule is the borrow checker's to enforce.
//! - **A poisoned surface is refused first.** The surface's quarantine flag is read
//!   before the driver is reached, so a surface whose acquire semaphore cannot be
//!   proven reusable is never presented to as if it were. Recovering such a surface
//!   means replacing the *surface* -- `VkSurfaceKHR` and window -- not reconfiguring
//!   the swapchain, which is why this is a refusal rather than a repair.
//!
//! # The acquire lease is a borrow, and its drop is the preserved semantic
//!
//! `Vulkan` permits at most one acquired image that has not yet been presented or
//! discarded. [`Swapchain::acquire`] takes `&mut self` and returns an
//! [`AcquireLease`] that holds that borrow, so a second acquire cannot be written
//! while a lease exists: the exclusivity rule is the compiler's rather than a
//! run-time flag that can disagree.
//!
//! The lease also carries the acquire semaphore. [`AcquireLease::present`] is the one
//! path that discharges the lease by handing that semaphore to `vkQueuePresentKHR` as
//! its wait. Dropping the lease without presenting is therefore not a no-op: the
//! presentation engine may still signal that semaphore, `Vulkan` cannot recycle it,
//! and nothing here can establish when it is safe to destroy. So the drop poisons the
//! surface -- no later acquire may guess reuse -- and leaves the semaphore
//! undestroyed for the process's lifetime. That is plan section 4's
//! unpresented-acquire quarantine, and it is the borrowed path's behavior, which
//! forgets the whole native bundle.
//!
//! # A presented semaphore is retained until its image returns
//!
//! A present that happened has no such problem, but it is not free either: the
//! semaphore `vkQueuePresentKHR` waited on [cannot be destroyed or recycled when the
//! call returns][angle], because the presentation engine may still be waiting on it.
//! The one portable proof that it is done with the image is
//! `vkAcquireNextImageKHR` handing that same image index back, which is the
//! inference ANGLE records and the reason [`Swapchain::presented`] is keyed by image
//! index: present fills the slot, the next acquire of that image empties it.
//!
//! [angle]: https://chromium.googlesource.com/angle/angle/+/31c4093651079775acf34ea1bb06bdabb4ea4386/src/libANGLE/renderer/vulkan/doc/PresentSemaphores.md

use core::time::Duration;

use ash::{khr, vk};
use fluxel_rendergraph::TextureUsage;

use super::acquire::{self, AcquireError};
use super::device::SwapchainDevice;
use super::present::{self, PresentError, PresentOutcome};
use super::presentation::{Surface, SurfaceQueryError};
use super::surface::{self, Presentation, PresentationError};

/// Why a swapchain could not be created or read.
///
/// Every variant is a distinct sentence, and the three sources stay distinct: a
/// fact about the device's queue, a fact about the surface, and a driver result from
/// one of the two calls that reach it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SwapchainError {
    /// The device's selected queue family cannot present to this surface.
    ///
    /// Step 2 chose the family without a surface, so this is the first place that
    /// choice is checked against presentation at all. It is refused before any
    /// swapchain exists rather than discovered at the first present.
    QueueCannotPresent {
        /// The family the device was created with.
        family: u32,
    },
    /// The surface's own facts could not be read.
    Facts(SurfaceQueryError),
    /// The fixed presentation contract is not servable by this surface.
    Contract(PresentationError),
    /// The surface is quarantined, so no swapchain may be built or rebuilt for it.
    ///
    /// An acquired image that was neither presented nor discarded leaves an acquire
    /// semaphore the presentation engine may still signal, which puts the **surface**
    /// -- not the swapchain -- beyond reuse (plan section 4). A rebuild would guess
    /// that a surface this backend has quarantined is reusable, so it is refused by
    /// name before the driver is reached. Recovering means replacing the surface.
    Poisoned,
    /// The driver refused to create the swapchain.
    Creation(vk::Result),
    /// The driver refused to report the swapchain's images.
    Images(vk::Result),
    /// The driver reported a created swapchain with no images.
    ///
    /// `Vulkan` guarantees at least one image for a created swapchain, so this is a
    /// driver that contradicted its own contract: refused as a value rather than
    /// producing a swapchain with no presentable image.
    NoImages,
}

/// An owned `VkSwapchainKHR` and the images it created.
///
/// Both parents are borrowed, which is what makes the teardown order a compile-time
/// rule: the swapchain is destroyed before the surface it presents to and before the
/// device that created it.
pub(crate) struct Swapchain<'a> {
    /// The surface this swapchain presents to, held for that lifetime binding.
    surface: &'a Surface<'a>,
    /// The device that created it, held for the same reason.
    device: &'a SwapchainDevice,
    /// The adapter both the surface and the device belong to.
    ///
    /// It is owned rather than re-requested by [`Swapchain::reconfigure`] because the
    /// surface facts a rebuild decides against must be read from the adapter this
    /// swapchain was actually created on; a parameter would let a caller ask a
    /// different adapter about a surface this one created.
    physical_device: vk::PhysicalDevice,
    /// The device-level swapchain entry points, loaded once when the handle was
    /// created.
    loader: khr::swapchain::Device,
    /// The handle itself.
    handle: vk::SwapchainKHR,
    /// The images `Vulkan` created with the swapchain.
    ///
    /// They are owned by the swapchain and destroyed with it: these are handles for
    /// recording, never allocations to release, and they must not enter the resource
    /// table, which owns memory.
    images: Vec<vk::Image>,
    /// The present wait semaphores of already-presented images, indexed by image.
    ///
    /// [`AcquireLease::present`] fills the slot its image names, and the next
    /// [`Swapchain::acquire`] of that image empties it -- see the module docs for why
    /// returning from `vkQueuePresentKHR` is not proof that its wait is consumed.
    /// Every slot left here is destroyed after a device-idle wait in [`Drop`], which
    /// is the borrowed path's teardown order.
    presented: Vec<Option<vk::Semaphore>>,
    /// The presentation decision this swapchain was created with.
    presentation: Presentation,
}

impl Swapchain<'_> {
    /// Returns the swapchain handle.
    ///
    /// Crate-private for the same reason every other handle here is: it stays inside
    /// `native::vulkan`.
    pub(crate) fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    /// Returns the surface this swapchain presents to.
    pub(crate) fn surface(&self) -> &Surface<'_> {
        self.surface
    }

    /// Returns the device that created it.
    pub(crate) fn device(&self) -> &SwapchainDevice {
        self.device
    }

    /// Returns the swapchain's images, in the driver's own order.
    ///
    /// The acquire step indexes into this list with the index
    /// `vkAcquireNextImageKHR` reports.
    pub(crate) fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Returns the presentation decision the swapchain was created with.
    pub(crate) fn presentation(&self) -> &Presentation {
        &self.presentation
    }
}

impl<'a> Swapchain<'a> {
    /// Acquires the next presentable image and leases it to the caller.
    ///
    /// `&mut self` **is** the lease: the returned [`AcquireLease`] holds that borrow,
    /// so `Vulkan`'s "at most one image not yet presented or discarded" rule cannot
    /// be broken by a second call while a lease exists.
    ///
    /// The order is the behavior, as everywhere in step 10. A quarantined surface is
    /// refused before anything is created; the acquire semaphore is created next; and
    /// only then is the driver reached. A driver refusal destroys that semaphore,
    /// because the presentation engine never received it and it is still unsignaled --
    /// unlike the success path, where the semaphore may be signalled later and is
    /// therefore owned by the lease. A successful acquire also releases the present
    /// semaphore retained for the image it returned, because handing the image back is
    /// what proves the presentation engine is done with it.
    pub(crate) fn acquire<'s>(
        &'s mut self,
        timeout: Duration,
    ) -> Result<AcquireLease<'s, 'a>, AcquireError> {
        if self.surface.is_poisoned() {
            return Err(AcquireError::Poisoned);
        }

        // A fresh unsignaled semaphore per acquisition is the readiness mechanism.
        // The fence argument is null because this backend's one submission is already
        // ordered on its single queue and waits for the acquire through this
        // semaphore, so a fence here would be a second signal nothing observes.
        let create_info = vk::SemaphoreCreateInfo::default();
        // SAFETY: no allocation callbacks are supplied, and the device is live and
        // owned by the borrow on `self`.
        let semaphore =
            unsafe { self.device.device().device().create_semaphore(&create_info, None) }
                .map_err(AcquireError::Semaphore)?;

        // SAFETY: the swapchain was created by this device, which is live; the
        // semaphore was created just above and is unsignaled; the fence is null; and
        // the timeout is a scalar. `ash` maps `VK_SUCCESS` and `VK_SUBOPTIMAL_KHR` to
        // `Ok` and every other result to `Err`, which `acquire::outcome` names.
        let answer = unsafe {
            self.loader.acquire_next_image(
                self.handle,
                super::submission::timeout_nanos(timeout),
                semaphore,
                vk::Fence::null(),
            )
        };
        let acquired = match acquire::outcome(answer) {
            Ok(acquired) => acquired,
            Err(error) => {
                // The driver refused, so the presentation engine never received this
                // semaphore and it is still unsignaled. Destroying it is therefore
                // legal and leaves nothing behind -- the opposite of the success
                // path's problem, where the semaphore may be signalled later.
                // SAFETY: the semaphore was created just above through this live
                // device, and no operation was successfully queued on it.
                unsafe {
                    self.device
                        .device()
                        .device()
                        .destroy_semaphore(semaphore, None)
                };
                return Err(error);
            }
        };

        let Some(image) = self.images.get(acquired.index as usize).copied() else {
            // The driver promised an index into its own image list and did not
            // deliver one. The acquire succeeded, so the semaphore may be signalled at
            // a time this backend cannot know: poison the surface and retain the
            // semaphore, which is the same quarantine an unpresented drop takes. A
            // `vk::Semaphore` is a handle rather than an owner, so retaining it is
            // exactly *not* calling `destroy_semaphore` on it.
            self.surface.poison();
            return Err(AcquireError::IndexOutOfRange {
                index: acquired.index,
                images: self.images.len(),
            });
        };

        // The driver just handed this image back, which is the proof that the
        // presentation engine is done with the previous present of it and therefore
        // done waiting on the semaphore that present retained.
        self.reclaim_presented(acquired.index);

        Ok(AcquireLease {
            swapchain: self,
            image,
            index: acquired.index,
            suboptimal: acquired.suboptimal,
            semaphore,
        })
    }

    /// Rebuilds this swapchain against the surface's current facts.
    ///
    /// The old handle is passed to the driver as `old_swapchain`, which is what lets
    /// a resize reuse the presentation engine's resources instead of tearing
    /// everything down. That field also decides the ownership shape: `Vulkan` retires
    /// the old swapchain even when the creation fails, so this method takes `self`
    /// rather than `&mut self` and the old swapchain is destroyed by its own [`Drop`]
    /// on both the success and the failure path. There is deliberately no state in
    /// which a caller holds a swapchain the driver has already retired.
    ///
    /// A quarantined surface is refused first, before any driver call, because a
    /// rebuilt swapchain on it would assume the acquire semaphore an unpresented
    /// image left behind is reusable. The presentation contract itself is re-decided
    /// from the facts the surface reports *now*, so a resize is not lowered against a
    /// stale extent.
    pub(crate) fn reconfigure(
        self,
        requested: vk::Extent2D,
        requested_usage: TextureUsage,
    ) -> Result<Swapchain<'a>, SwapchainError> {
        if self.surface.is_poisoned() {
            return Err(SwapchainError::Poisoned);
        }
        let surface = self.surface;
        let device = self.device;
        // The old handle is valid until `self` drops at the end of this call, which is
        // after `build` has created its replacement; that ordering is what makes
        // `old_swapchain` legal at the driver call.
        build(
            device,
            surface,
            self.physical_device,
            requested,
            requested_usage,
            self.handle,
        )
    }

    /// Destroys the present wait semaphore retained for a previously presented image.
    ///
    /// It is called with the index `vkAcquireNextImageKHR` just returned, and that is
    /// the whole justification: the presentation engine handing the image back proves
    /// it is done waiting on the semaphore, which is the inference ANGLE records.
    /// Destroying it at present time instead would free a handle the engine may still
    /// be waiting on.
    fn reclaim_presented(&mut self, index: u32) {
        if let Some(semaphore) = self.presented[index as usize].take() {
            // SAFETY: the semaphore was created by this device, which is live; the
            // acquire that just returned this index proves the presentation engine is
            // done waiting on it, so no queue operation refers to it any more. No
            // allocation callbacks were supplied.
            unsafe {
                self.device
                    .device()
                    .device()
                    .destroy_semaphore(semaphore, None)
            };
        }
    }

    /// Retains the semaphore a successful present waited on until its image returns.
    ///
    /// The slot must be empty: the acquire that produced the presented lease emptied
    /// it, so a filled slot here would mean a lease was presented twice, which the
    /// lease's own consumption prevents. The assertion states that invariant rather
    /// than overwriting a live handle and leaking it.
    fn retain_present_semaphore(&mut self, index: u32, semaphore: vk::Semaphore) {
        let slot = &mut self.presented[index as usize];
        debug_assert!(
            slot.is_none(),
            "the acquire that produced this lease emptied its present slot"
        );
        *slot = Some(semaphore);
    }
}

/// The one acquired swapchain image a surface may hold before it is presented.
///
/// Holding the swapchain's mutable borrow is the lease rule: a second
/// [`Swapchain::acquire`] cannot be written while this value exists, which is
/// `Vulkan`'s "at most one image not yet presented or discarded" stated in the type
/// system rather than in a run-time flag that can disagree.
pub(crate) struct AcquireLease<'s, 'a> {
    /// The borrow that enforces exclusivity, and the surface an unpresented drop
    /// quarantines.
    swapchain: &'s mut Swapchain<'a>,
    /// The acquired image, resolved from the driver's index.
    image: vk::Image,
    /// The index the driver reported, which is the swapchain's own image index.
    index: u32,
    /// Whether the driver reported the swapchain as suboptimal.
    suboptimal: bool,
    /// The semaphore `vkAcquireNextImageKHR` will signal.
    ///
    /// It is a handle rather than an owner: retaining the driver object the
    /// quarantine rule requires is exactly not calling `destroy_semaphore`, so
    /// there is no `Drop` body to suppress.
    semaphore: vk::Semaphore,
}

impl AcquireLease<'_, '_> {
    /// The acquired image's index in the swapchain's own image list.
    pub(crate) fn image_index(&self) -> u32 {
        self.index
    }

    /// The acquired image itself.
    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    /// The semaphore the first submission drawing to this image must wait on.
    ///
    /// It is the whole acquire-side ordering: `Vulkan` signals it when the image is
    /// ready to be written, so a submission that draws before waiting on it would
    /// race the presentation engine. [`AcquireLease::present`] waits on it directly
    /// because no submission has consumed it yet; when the draw-and-present path
    /// lands, that submission waits on it and present waits on the submission's
    /// render-finished semaphore instead.
    pub(crate) fn wait_semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// Whether the driver reported the swapchain as suboptimal for its surface.
    ///
    /// It is reported rather than acted on: the acquire succeeded and the image is
    /// presentable, while reconfigure is a separate step with its own `old_swapchain`
    /// ownership.
    pub(crate) fn is_suboptimal(&self) -> bool {
        self.suboptimal
    }

    /// Presents the leased image and consumes the lease.
    ///
    /// This is the one path that discharges the lease: the acquire semaphore it
    /// carries is handed to `vkQueuePresentKHR` as the present's wait, so it is
    /// consumed by a queue operation rather than left pending. The semaphore is not
    /// destroyed here -- the presentation engine may still be waiting on it, so it is
    /// retained on the swapchain until its image is acquired again. A refused present
    /// leaves that semaphore pending, so the lease's own `Drop` still takes the
    /// unpresented-acquire quarantine, which is the fail-closed direction.
    pub(crate) fn present(self) -> Result<PresentOutcome, PresentError> {
        let index = self.index;
        let wait_semaphore = self.semaphore;
        let answer = {
            let swapchain = &mut *self.swapchain;
            let wait_semaphores = [wait_semaphore];
            let swapchains = [swapchain.handle];
            let image_indices = [index];
            // One swapchain and one wait semaphore, so the three builder setters --
            // each of which overwrites `swapchain_count` -- all agree. No `p_results`
            // is passed: with one swapchain the call's own result carries the same
            // fact, which is what the borrowed path being replaced reads.
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            // SAFETY: the swapchain and the semaphore belong to this device, which is
            // live for the lease's lifetime; the semaphore was handed to this lease by
            // a successful acquire and no operation has waited on it since; the three
            // arrays are locals that outlive the call; and the device enabled
            // `VK_KHR_swapchain` by construction -- `SwapchainDevice` is that proof --
            // so this is the real entry point rather than `ash`'s unresolved stub. No
            // allocation callbacks are supplied.
            unsafe {
                swapchain
                    .loader
                    .queue_present(swapchain.device.device().queue(), &present_info)
            }
        };
        match present::outcome(answer) {
            Ok(outcome) => {
                // The lease's obligation is discharged by a present that happened:
                // the semaphore is retained until its image returns and the surface is
                // not poisoned. Forgetting the lease is what keeps its `Drop` from
                // taking the unpresented-acquire path after all. The lease is not
                // `Copy` and implements `Drop`, so this is a real suppression and not
                // the no-op that forgetting a handle would be.
                let swapchain = &mut *self.swapchain;
                swapchain.retain_present_semaphore(index, wait_semaphore);
                core::mem::forget(self);
                Ok(outcome)
            }
            // A present that did not happen leaves the acquire semaphore pending, so
            // the lease's `Drop` runs and poisons the surface -- the same quarantine
            // an unpresented drop takes, reached through an error return.
            Err(error) => Err(error),
        }
    }
}

impl Drop for AcquireLease<'_, '_> {
    fn drop(&mut self) {
        // Reaching here means the lease was neither presented nor discarded --
        // `present` is the only path that consumes it, and it discharges the lease by
        // forgetting it rather than by running this body. `Vulkan` cannot recycle the
        // acquire semaphore of an image that was never presented: the presentation
        // engine may signal it at a time this backend cannot know, so destroying it
        // could free a handle the driver still holds, and reusing it could hand the
        // driver a semaphore that is already signalled. The surface is therefore
        // poisoned -- no later acquire may guess reuse -- and the semaphore is left
        // undestroyed. That is plan section 4's preserved semantic, and the borrowed
        // path being replaced does the same thing by forgetting its whole native
        // bundle. There is no `Drop` body to suppress for the handle itself: retaining
        // a `vk::Semaphore` is exactly not calling `destroy_semaphore`, so the driver
        // object outlives this lease for the process's lifetime.
        self.swapchain.surface.poison();
    }
}

impl core::fmt::Debug for AcquireLease<'_, '_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("AcquireLease")
            .field("index", &self.index)
            .field("suboptimal", &self.suboptimal)
            .finish_non_exhaustive()
    }
}

impl Drop for Swapchain<'_> {
    fn drop(&mut self) {
        // A retained present semaphore may still be waited on by the presentation
        // engine. There is no portable way to observe a present's completion, so the
        // device is made idle first -- which completes every queue operation, present
        // included -- and only then are they destroyed. That is exactly the borrowed
        // path being replaced (`vkDeviceWaitIdle`, then its semaphores), and it costs
        // the steady-state path nothing because it happens only at teardown.
        // SAFETY: the device is live and owned by the borrow on `self`, and the call
        // only waits for work already submitted.
        let _ = unsafe { self.device.device().device().device_wait_idle() };
        for semaphore in self.presented.drain(..).flatten() {
            // SAFETY: the semaphore was created by this device, which is live; the
            // idle wait above proves the presentation engine is done waiting on it;
            // and this loop is its only owner. No allocation callbacks were supplied.
            unsafe {
                self.device
                    .device()
                    .device()
                    .destroy_semaphore(semaphore, None)
            };
        }
        // SAFETY: this is the only owner and the handle was created through this
        // same device, which the borrow keeps alive; the surface outlives it for the
        // same reason. Destroying the swapchain also destroys its images, and no
        // child object outlives the call. No allocation callbacks were supplied at
        // creation.
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
    }
}

impl core::fmt::Debug for Swapchain<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Swapchain")
            .field("handle", &self.handle)
            .field("images", &self.images.len())
            .field(
                "presented",
                &self.presented.iter().filter(|slot| slot.is_some()).count(),
            )
            .field("extent", &self.presentation.extent)
            .finish_non_exhaustive()
    }
}

/// Creates the swapchain for `surface` on `device`, and reads its images.
///
/// `physical_device` must be the adapter both the surface and the device belong to.
/// The order is the behavior: the queue family's presentation support is checked
/// first, then the surface's facts are read once, then the *pure* contract decides
/// whether they serve this backend's fixed presentation, and only then is the driver
/// reached. Every refusal therefore happens before anything exists.
pub(crate) fn create<'a>(
    device: &'a SwapchainDevice,
    surface: &'a Surface<'a>,
    physical_device: vk::PhysicalDevice,
    requested: vk::Extent2D,
    requested_usage: TextureUsage,
) -> Result<Swapchain<'a>, SwapchainError> {
    build(
        device,
        surface,
        physical_device,
        requested,
        requested_usage,
        vk::SwapchainKHR::null(),
    )
}

/// Builds the swapchain a caller asked for, over a predecessor it may already have.
///
/// This is [`create`]'s body and [`Swapchain::reconfigure`]'s replacement path at
/// once: the only difference between a fresh creation and a rebuild is the
/// `old_swapchain` handle, so a second copy of the facts read and the contract
/// decision would be a second place for them to drift. `build` owns nothing beyond
/// the value it returns, so every refusal it produces happens before a swapchain
/// exists -- and a refused image read undoes the creation it just made.
fn build<'a>(
    device: &'a SwapchainDevice,
    surface: &'a Surface<'a>,
    physical_device: vk::PhysicalDevice,
    requested: vk::Extent2D,
    requested_usage: TextureUsage,
    old_swapchain: vk::SwapchainKHR,
) -> Result<Swapchain<'a>, SwapchainError> {
    let family = device.device().selected_queue().family;
    let presents = surface
        .supports_presentation(physical_device, family)
        .map_err(SwapchainError::Facts)?;
    if !presents {
        return Err(SwapchainError::QueueCannotPresent { family });
    }

    let facts = surface.facts(physical_device).map_err(SwapchainError::Facts)?;
    let presentation = surface::contract(
        &facts.capabilities,
        &facts.formats,
        &facts.present_modes,
        requested,
        requested_usage,
    )
    .map_err(SwapchainError::Contract)?;

    let create_info = surface::swapchain_create_info(surface.handle(), &presentation, old_swapchain);
    let loader = khr::swapchain::Device::new(
        surface.instance().instance().instance(),
        device.device().device(),
    );
    // SAFETY: `create_info` is a local that outlives the call; the surface and the
    // device are live and were created from the same instance, and the device
    // enabled `VK_KHR_swapchain` by construction, so the entry point is the real one
    // rather than `ash`'s unresolved stub. No allocation callbacks are supplied.
    let handle = unsafe { loader.create_swapchain(&create_info, None) }
        .map_err(SwapchainError::Creation)?;

    // Every failure path undoes its own work: a refused image read must not leave a
    // swapchain behind.
    // SAFETY: the handle was just created by this device and is live; `ash` retries
    // on `VK_INCOMPLETE` rather than truncating the image list.
    let images = match unsafe { loader.get_swapchain_images(handle) } {
        Ok(images) if !images.is_empty() => images,
        Ok(_) => {
            // SAFETY: as in `Drop`; this is the only owner and no child outlives it.
            unsafe { loader.destroy_swapchain(handle, None) };
            return Err(SwapchainError::NoImages);
        }
        Err(error) => {
            // SAFETY: as above.
            unsafe { loader.destroy_swapchain(handle, None) };
            return Err(SwapchainError::Images(error));
        }
    };

    // One slot per image, empty until an image is presented: a slot holds the
    // semaphore the presentation engine may still be waiting on for that image.
    let presented = vec![None; images.len()];
    Ok(Swapchain {
        surface,
        device,
        physical_device,
        loader,
        handle,
        images,
        presented,
        presentation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Validation;
    use crate::native::vulkan::test_support::{TestWindow, presenting_adapter};
    use crate::native::vulkan::{adapter, device, instance, presentation};
    use fluxel_rendergraph::{TextureFormat, TextureUsageKind};

    /// The extent to request when the surface delegates the choice.
    fn requested(facts: &presentation::SurfaceFacts) -> vk::Extent2D {
        let current = facts.capabilities.current_extent;
        if current.width == 0 || current.height == 0 {
            vk::Extent2D {
                width: 64,
                height: 64,
            }
        } else {
            current
        }
    }

    #[test]
    fn a_real_swapchain_owns_real_images() {
        // The step 10 swapchain half against the real driver: a real surface, a real
        // device that enabled `VK_KHR_swapchain`, and a real `VkSwapchainKHR` whose
        // images are read back. Skips only where the machine has no loader, no
        // adapter, or no window station -- none of which is this module's subject.
        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let instance = surface_instance.instance().instance();
        let Ok(adapters) = adapter::enumerate(instance) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let Some(window) = TestWindow::open() else {
            return;
        };
        let Ok(surface) = presentation::create(&surface_instance, window.raw()) else {
            return;
        };
        // Not a skip: a surface this loader created is presentable from an
        // enumerated adapter, so the assertions below are reached rather than
        // bypassed by an early return.
        let (physical_device, facts) = presenting_adapter(&surface, &adapters)
            .expect("a surface this loader created is presentable from an enumerated adapter");

        let adapter_facts = adapter::describe(instance, physical_device);
        let device = device::open_with_swapchain(&surface_instance, physical_device, &adapter_facts.limits)
            .expect("the named Windows board reports VK_KHR_swapchain");
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);

        let swapchain = create(
            &device,
            &surface,
            physical_device,
            requested(&facts),
            usage,
        )
        .expect("the named Windows board serves a swapchain for the fixed contract");

        assert_ne!(swapchain.handle(), vk::SwapchainKHR::null());
        assert!(
            !swapchain.images().is_empty(),
            "a created swapchain owns at least one image"
        );
        assert_eq!(swapchain.presentation().format, surface::PRESENT_FORMAT);
        assert!(
            swapchain.presentation().extent.width > 0,
            "the contract never chooses a zero extent"
        );
        // The portable format the surface texture descriptor names and the image the
        // swapchain creates are the same bytes.
        assert_eq!(
            super::super::format::image_format(TextureFormat::Rgba8Unorm),
            Some(swapchain.presentation().format)
        );
        // Drop order: the swapchain is destroyed before the device and before the
        // surface, and the surface before the instance and the window, all by the
        // borrows on this function's locals.
    }

    #[test]
    fn a_real_acquire_leases_an_image_and_an_unpresented_drop_poisons_the_surface() {
        // The step 10 acquire half against the real driver: a real
        // `vkAcquireNextImageKHR` on a real swapchain, the image the driver's index
        // names, and the two rules a lease carries -- exclusivity is the borrow, and
        // an unpresented drop poisons the surface and retains the semaphore. Skips
        // only where the machine has no loader, no adapter or no window station.
        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let instance = surface_instance.instance().instance();
        let Ok(adapters) = adapter::enumerate(instance) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let Some(window) = TestWindow::open() else {
            return;
        };
        let Ok(surface) = presentation::create(&surface_instance, window.raw()) else {
            return;
        };
        // Not a skip: the assertions below are reached on this machine rather than
        // bypassed by an early return.
        let (physical_device, facts) = presenting_adapter(&surface, &adapters)
            .expect("a surface this loader created is presentable from an enumerated adapter");
        let adapter_facts = adapter::describe(instance, physical_device);
        let device =
            device::open_with_swapchain(&surface_instance, physical_device, &adapter_facts.limits)
                .expect("the named Windows board reports VK_KHR_swapchain");
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);
        let mut swapchain = create(&device, &surface, physical_device, requested(&facts), usage)
            .expect("the named Windows board serves a swapchain for the fixed contract");

        // The images are copied before the lease takes the swapchain's mutable
        // borrow, because that borrow is the exclusivity rule this test is about.
        let images = swapchain.images().to_vec();
        let acquisition = swapchain
            .acquire(Duration::from_secs(5))
            .expect("a FIFO swapchain with no image in flight yields one within the timeout");
        assert!(
            (acquisition.image_index() as usize) < images.len(),
            "the driver's index addresses the swapchain's own image list"
        );
        assert_eq!(
            acquisition.image(),
            images[acquisition.image_index() as usize],
            "the leased image is the one the driver's index names"
        );
        assert_ne!(
            acquisition.wait_semaphore(),
            vk::Semaphore::null(),
            "an acquire carries a real semaphore, which is what the quarantine rule is about"
        );
        assert!(
            !surface.is_poisoned(),
            "a surface is live while its lease has not been released unpresented"
        );

        drop(acquisition);

        // The lease was dropped without being presented: `Vulkan` cannot recycle the
        // acquire semaphore, so the surface is quarantined rather than reused. The
        // refusal happens before the driver is reached, which is what makes it a
        // property of this backend rather than of a particular driver.
        assert!(surface.is_poisoned());
        assert_eq!(
            swapchain.acquire(Duration::ZERO).err(),
            Some(AcquireError::Poisoned),
            "a quarantined surface refuses an acquire"
        );
        // Drop order is unchanged: the swapchain before the device and the surface,
        // the surface before the instance and the window.
    }

    #[test]
    fn a_real_present_consumes_the_lease_and_retains_its_wait_semaphore() {
        // The step 10 present half against the real driver: a real
        // `vkQueuePresentKHR` over a real swapchain, the lease it consumes, and the
        // wait semaphore it retains because returning from present is not proof the
        // presentation engine is done with it. Skips only where the machine has no
        // loader, no adapter or no window station.
        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let instance = surface_instance.instance().instance();
        let Ok(adapters) = adapter::enumerate(instance) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let Some(window) = TestWindow::open() else {
            return;
        };
        let Ok(surface) = presentation::create(&surface_instance, window.raw()) else {
            return;
        };
        // Not a skip: the assertions below are reached on this machine rather than
        // bypassed by an early return.
        let (physical_device, facts) = presenting_adapter(&surface, &adapters)
            .expect("a surface this loader created is presentable from an enumerated adapter");
        let adapter_facts = adapter::describe(instance, physical_device);
        let device =
            device::open_with_swapchain(&surface_instance, physical_device, &adapter_facts.limits)
                .expect("the named Windows board reports VK_KHR_swapchain");
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);
        let mut swapchain = create(&device, &surface, physical_device, requested(&facts), usage)
            .expect("the named Windows board serves a swapchain for the fixed contract");

        // Frame one: the lease's own semaphore is what present waits on, and a present
        // that happened discharges the lease rather than quarantining the surface -- so
        // the semaphore is retained, not destroyed and not left pending.
        let lease = swapchain
            .acquire(Duration::from_secs(5))
            .expect("a FIFO swapchain with no image in flight yields one within the timeout");
        let index = lease.image_index();
        let wait_semaphore = lease.wait_semaphore();
        let outcome = lease
            .present()
            .expect("the named Windows board presents an image it just acquired");
        assert!(
            outcome == PresentOutcome::Presented || outcome.is_suboptimal(),
            "a present is either success or success-plus-suboptimal"
        );
        assert!(
            !surface.is_poisoned(),
            "a present that happened discharges the lease instead of quarantining the surface"
        );
        assert_eq!(
            swapchain.presented[index as usize],
            Some(wait_semaphore),
            "present retains the semaphore it waited on until its image returns"
        );

        // Frame two: the surface stays acquirable, and whichever image comes back has
        // its retained semaphore released rather than reused or leaked. A driver that
        // reuses an index here is the case the reclaim exists for.
        let lease = swapchain
            .acquire(Duration::from_secs(5))
            .expect("a presented surface keeps yielding images");
        let index = lease.image_index();
        assert!(
            !surface.is_poisoned(),
            "presenting leaves the surface live for the next frame"
        );
        let outcome = lease
            .present()
            .expect("a second present over the same swapchain succeeds");
        assert!(
            outcome == PresentOutcome::Presented || outcome.is_suboptimal(),
            "a second present is success or success-plus-suboptimal"
        );
        assert!(
            swapchain.presented[index as usize].is_some(),
            "the second present retains its own wait semaphore"
        );
        assert!(
            !surface.is_poisoned(),
            "two frames through present leave the surface live"
        );
        // Drop order: the swapchain, which idles the device and destroys every
        // retained semaphore, before the device, the surface, the instance and the
        // window.
    }

    #[test]
    fn a_real_reconfigure_retires_the_old_swapchain_and_returns_a_live_one() {
        // The step 10 reconfigure half against the real driver: a real
        // `vkCreateSwapchainKHR` over its own predecessor, the retained present
        // semaphore the predecessor left behind, and the replacement that is usable
        // straight away. Skips only where the machine has no loader, no adapter or no
        // window station.
        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let instance = surface_instance.instance().instance();
        let Ok(adapters) = adapter::enumerate(instance) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let Some(window) = TestWindow::open() else {
            return;
        };
        let Ok(surface) = presentation::create(&surface_instance, window.raw()) else {
            return;
        };
        // Not a skip: the assertions below are reached on this machine rather than
        // bypassed by an early return.
        let (physical_device, facts) = presenting_adapter(&surface, &adapters)
            .expect("a surface this loader created is presentable from an enumerated adapter");
        let adapter_facts = adapter::describe(instance, physical_device);
        let device =
            device::open_with_swapchain(&surface_instance, physical_device, &adapter_facts.limits)
                .expect("the named Windows board reports VK_KHR_swapchain");
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);
        let mut swapchain = create(&device, &surface, physical_device, requested(&facts), usage)
            .expect("the named Windows board serves a swapchain for the fixed contract");

        // A presented frame leaves a semaphore retained on the old swapchain, so a
        // rebuild must retire that too -- this is the teardown path, not only a new
        // handle.
        let lease = swapchain
            .acquire(Duration::from_secs(5))
            .expect("a FIFO swapchain with no image in flight yields one within the timeout");
        lease
            .present()
            .expect("the named Windows board presents an image it just acquired");
        assert!(
            swapchain.presented.iter().any(Option::is_some),
            "a present leaves a retained semaphore for the rebuild to retire"
        );

        let reconfigured = swapchain
            .reconfigure(requested(&facts), usage)
            .expect("a surface that just served a swapchain serves its rebuild");
        assert_ne!(reconfigured.handle(), vk::SwapchainKHR::null());
        assert!(
            !reconfigured.images().is_empty(),
            "a rebuilt swapchain owns at least one image"
        );
        assert_eq!(reconfigured.presentation().format, surface::PRESENT_FORMAT);
        assert!(
            reconfigured.presentation().extent.width > 0,
            "the rebuilt contract never chooses a zero extent"
        );
        assert!(
            reconfigured.presented.iter().all(Option::is_none),
            "a rebuilt swapchain starts with no retained present semaphore"
        );

        // The replacement is live: it acquires and presents a frame of its own.
        let mut reconfigured = reconfigured;
        let lease = reconfigured
            .acquire(Duration::from_secs(5))
            .expect("a rebuilt FIFO swapchain yields an image");
        let outcome = lease
            .present()
            .expect("a rebuilt swapchain presents an image it just acquired");
        assert!(
            outcome == PresentOutcome::Presented || outcome.is_suboptimal(),
            "a present over the rebuilt swapchain is success or success-plus-suboptimal"
        );
        assert!(
            !surface.is_poisoned(),
            "a completed rebuild does not quarantine the surface"
        );
    }

    #[test]
    fn a_poisoned_surface_refuses_reconfigure_before_the_driver() {
        // An unpresented acquire quarantines the surface, and a rebuild on it would
        // assume the semaphore that acquire left behind is reusable. The refusal is a
        // property of this backend, reached before any driver call.
        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let instance = surface_instance.instance().instance();
        let Ok(adapters) = adapter::enumerate(instance) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let Some(window) = TestWindow::open() else {
            return;
        };
        let Ok(surface) = presentation::create(&surface_instance, window.raw()) else {
            return;
        };
        // Not a skip: the assertions below are reached on this machine rather than
        // bypassed by an early return.
        let (physical_device, facts) = presenting_adapter(&surface, &adapters)
            .expect("a surface this loader created is presentable from an enumerated adapter");
        let adapter_facts = adapter::describe(instance, physical_device);
        let device =
            device::open_with_swapchain(&surface_instance, physical_device, &adapter_facts.limits)
                .expect("the named Windows board reports VK_KHR_swapchain");
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);
        let mut swapchain = create(&device, &surface, physical_device, requested(&facts), usage)
            .expect("the named Windows board serves a swapchain for the fixed contract");

        drop(
            swapchain
                .acquire(Duration::from_secs(5))
                .expect("a FIFO swapchain yields an image within the timeout"),
        );
        assert!(surface.is_poisoned());

        assert_eq!(
            swapchain.reconfigure(requested(&facts), usage).err(),
            Some(SwapchainError::Poisoned),
            "a quarantined surface refuses a rebuild by name"
        );
        // The consumed swapchain is dropped, so a poisoned surface ends with no
        // swapchain either way; recovering it means replacing the surface itself.
    }
}
