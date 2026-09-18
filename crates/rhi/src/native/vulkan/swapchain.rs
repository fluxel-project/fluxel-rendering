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
//! - **No acquire, present or reconfigure.** Those are the frame loop: the acquire
//!   lease, `vkQueuePresentKHR`, the `old_swapchain` reconfigure path, and the
//!   unpresented-acquire quarantine plan section 4 preserves. They are the next
//!   pieces of step 10, and they lower from the handle, the images and the
//!   [`surface::Presentation`] this module owns.
//! - **No sRGB presentation, no format negotiation.** These are
//!   [`super::surface::contract`]'s decisions, and this module only consumes them.

use ash::{khr, vk};
use fluxel_rendergraph::TextureUsage;

use super::device::SwapchainDevice;
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

impl Drop for Swapchain<'_> {
    fn drop(&mut self) {
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

    let create_info = surface::swapchain_create_info(surface.handle(), &presentation);
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

    Ok(Swapchain {
        surface,
        device,
        loader,
        handle,
        images,
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
}
