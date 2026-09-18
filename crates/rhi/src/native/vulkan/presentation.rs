//! Step 10's owning half: the `VkSurfaceKHR` created from a host window, and the
//! facts it reports.
//!
//! [`super::surface`] is the pure half -- it decides *what* a conformant surface
//! must report and lowers that decision into a `VkSwapchainCreateInfoKHR`. This
//! module is the half that owns a driver object: it lowers the host's window handle
//! into the platform create-info, creates the `VkSurfaceKHR` through the instance
//! that enabled the surface extensions, and destroys it in `Drop`.
//!
//! # The facts query
//!
//! A swapchain's configuration is decided against three lists the surface itself
//! reports, and [`Surface::facts`] is the one place they are read:
//! `vkGetPhysicalDeviceSurfaceCapabilitiesKHR`,
//! `vkGetPhysicalDeviceSurfaceFormatsKHR` and
//! `vkGetPhysicalDeviceSurfacePresentModesKHR`. [`super::surface::contract`] already
//! decides the fixed presentation contract against exactly those three, so the query
//! produces the decision's inputs and nothing else: the query does not decide, and
//! the decision does not read the driver.
//!
//! [`Surface::supports_presentation`] is a fourth read and a different question:
//! whether **one queue family** can present to this surface. Step 2 chose the queue
//! family by rule and without a surface, so this is the fact that tells the
//! swapchain-owning step whether that selection can present. It is read rather than
//! assumed, and it is a value about one surface rather than a capability row about
//! the device.
//!
//! # Why the instance is a witness type, and the surface borrows it
//!
//! `vkCreateWin32SurfaceKHR` is only loadable from an instance that enabled
//! `VK_KHR_win32_surface`; `ash` substitutes a panicking stub for a function the
//! loader did not resolve, so the mistake would be a crash rather than a refusal.
//! [`create`] therefore takes [`SurfaceInstance`], the type only
//! [`super::instance::open_with_surface`] produces, and a headless
//! `ValidationInstance` cannot be handed here at all.
//!
//! The surface *refers to* its instance -- `Vulkan` requires the parent to outlive
//! the child -- so [`Surface`] holds the instance by borrow rather than by clone.
//! The compiler, not a comment asking politely, is what orders the two teardowns.
//!
//! # What the Win32 lowering requires, and why
//!
//! A `VK_KHR_win32_surface` surface names the window's owning module as well as the
//! window, and the borrowed path being replaced refuses a handle that omits it
//! (`crates/wgpu-hal`, `create_surface`: "Vulkan requires raw-window-handle's
//! Win32::hinstance to be set"). That is preserved here as its own refusal rather
//! than defaulting the module to null and letting the driver guess one.
//!
//! # What is deliberately not here
//!
//! - **No swapchain and no present.** Those lower from
//!   [`super::surface::swapchain_create_info`] over the handle this module owns and
//!   the facts it reads; the swapchain object is [`super::swapchain`]'s and the
//!   present call is still owed by step 10. What this module does own beyond the
//!   handle is the surface's quarantine state, because plan section 4 quarantines the
//!   surface an unpresented acquire could not prove reusable.
//! - **No decision about a family that cannot present.** The query reports the
//!   fact; acting on it -- changing step 2's selection rule so the device is
//!   created on a family that can present -- is the swapchain-owning step's
//!   decision, because it is the step that first needs a presentable queue.

use core::sync::atomic::{AtomicBool, Ordering};

use ash::{khr, vk};
use raw_window_handle::RawWindowHandle;

use super::instance::SurfaceInstance;

/// Why a surface could not be created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SurfaceError {
    /// The host handed a window this target's loader cannot present to.
    NotWin32,
    /// The window is a Win32 one whose owning module was not reported.
    MissingInstance,
    /// The loader refused to create the surface for a window it accepted.
    Creation(vk::Result),
}

/// Lowers a host window handle into the create-info the driver is handed.
///
/// The raw-window-handle enum is `#[non_exhaustive]`, so the mapping answers
/// `Result` rather than inventing a value for a platform this backend has not been
/// taught. Both refusal halves are distinct sentences: a window from another
/// platform is a different loader, while a Win32 window without its module is the
/// same loader missing one fact.
pub(crate) fn win32_create_info(
    window: RawWindowHandle,
) -> Result<vk::Win32SurfaceCreateInfoKHR<'static>, SurfaceError> {
    let RawWindowHandle::Win32(handle) = window else {
        return Err(SurfaceError::NotWin32);
    };
    let hinstance = handle.hinstance.ok_or(SurfaceError::MissingInstance)?;
    Ok(vk::Win32SurfaceCreateInfoKHR::default()
        .flags(vk::Win32SurfaceCreateFlagsKHR::empty())
        .hinstance(hinstance.get())
        .hwnd(handle.hwnd.get()))
}

/// What one surface reports to a physical device.
///
/// These are exactly the three lists [`super::surface::contract`] decides against,
/// read into one snapshot so the fixed contract is judged against one consistent
/// report rather than three reads that a resize could have moved between. The
/// vectors are owned because `Vulkan`'s enumeration is a two-call read whose count
/// may change between the calls; `ash` retries on `VK_INCOMPLETE` rather than
/// truncating.
#[derive(Clone)]
pub(crate) struct SurfaceFacts {
    /// The surface's own limits: image counts, extents, transforms, alpha and usage.
    pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,
    /// Every format/colour-space pair the surface offers.
    pub(crate) formats: Vec<vk::SurfaceFormatKHR>,
    /// Every present mode the surface offers.
    pub(crate) present_modes: Vec<vk::PresentModeKHR>,
}

/// Why a surface's facts could not be read.
///
/// One variant per call: the first three are one surface's own presentation facts,
/// and the fourth is whether a queue family can present to it, which is a different
/// question about a different object. They stay separate sentences because they need
/// different fixes, exactly as [`SurfaceError`]'s variants do.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SurfaceQueryError {
    /// `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` failed.
    Capabilities(vk::Result),
    /// `vkGetPhysicalDeviceSurfaceFormatsKHR` failed.
    Formats(vk::Result),
    /// `vkGetPhysicalDeviceSurfacePresentModesKHR` failed.
    PresentModes(vk::Result),
    /// `vkGetPhysicalDeviceSurfaceSupportKHR` failed.
    PresentationSupport(vk::Result),
}

/// An owned `VkSurfaceKHR`, bound to the instance that created it.
///
/// The instance field is held for that binding rather than read: it is what makes
/// "destroy the instance before the surface" a compile error, which `Vulkan`'s
/// object model requires.
///
/// The poison flag is the surface's own state rather than a swapchain's, because
/// what plan section 4 quarantines is the **surface**: an acquired image that is
/// neither presented nor discarded leaves an acquire semaphore the presentation
/// engine may still signal, so no later acquire -- and, once it exists, no
/// reconfigure -- may guess that the surface is reusable. Placing the flag on the
/// surface is what lets a reconfigure read the same fact the acquire path wrote,
/// rather than a second opinion about it living on a swapchain that reconfigure is
/// about to replace.
pub(crate) struct Surface<'a> {
    /// The instance that created this surface and must outlive it.
    instance: &'a SurfaceInstance,
    /// The surface-level entry points, loaded once when the handle was created.
    loader: khr::surface::Instance,
    /// The handle itself.
    handle: vk::SurfaceKHR,
    /// Whether an unpresented acquired image quarantined this surface.
    poisoned: AtomicBool,
}

impl Surface<'_> {
    /// Returns the surface handle.
    ///
    /// The handle is a raw `Vulkan` value, and it is crate-private for the same
    /// reason every other handle here is: it stays inside `native::vulkan`.
    pub(crate) fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Returns the instance this surface belongs to.
    pub(crate) fn instance(&self) -> &SurfaceInstance {
        self.instance
    }

    /// Reads the three lists the fixed presentation contract decides against.
    ///
    /// `physical_device` must have been enumerated from the instance this surface
    /// was created through; the borrow already keeps that instance alive, and the
    /// three calls create and destroy nothing. A device that does not support this
    /// surface answers with empty lists rather than a driver error, and that is
    /// reported as it arrived: [`super::surface::contract`] is what turns an empty
    /// format list into its own refusal.
    pub(crate) fn facts(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<SurfaceFacts, SurfaceQueryError> {
        // SAFETY: the physical device was enumerated from the same live instance
        // that created this surface, and the surface is live and owned here. Each
        // call either writes one value or fills a vector `ash` sized from the
        // driver's own count; no allocation callbacks are involved.
        let capabilities = unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical_device, self.handle)
        }
        .map_err(SurfaceQueryError::Capabilities)?;
        // SAFETY: as above.
        let formats = unsafe {
            self.loader
                .get_physical_device_surface_formats(physical_device, self.handle)
        }
        .map_err(SurfaceQueryError::Formats)?;
        // SAFETY: as above.
        let present_modes = unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical_device, self.handle)
        }
        .map_err(SurfaceQueryError::PresentModes)?;
        Ok(SurfaceFacts {
            capabilities,
            formats,
            present_modes,
        })
    }

    /// Whether one queue family can present to this surface.
    ///
    /// This is the fact step 2's queue selection has to be told: the family the
    /// device was created with is the only one this backend may present from, so a
    /// surface that family cannot present to has no swapchain path at all. The
    /// answer is a value about one surface rather than a capability row, and it is
    /// read here rather than assumed from the family's graphics flag.
    pub(crate) fn supports_presentation(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> Result<bool, SurfaceQueryError> {
        // SAFETY: as in `facts`; the query writes one boolean into `ash`'s own
        // local and creates nothing.
        unsafe {
            self.loader.get_physical_device_surface_support(
                physical_device,
                queue_family,
                self.handle,
            )
        }
        .map_err(SurfaceQueryError::PresentationSupport)
    }

    /// Whether an unpresented acquired image has quarantined this surface.
    ///
    /// This is read before an acquire creates anything, so a quarantined surface is
    /// refused rather than handed to a driver whose acquire semaphore cannot be
    /// proven reusable.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Quarantines this surface because an acquired image was neither presented nor
    /// discarded.
    ///
    /// The flag is set rather than reset by anything: nothing in this backend can
    /// establish when the presentation engine is done with the semaphore the
    /// unpresented acquire handed it, so the state is permanent until the surface
    /// itself is replaced. That is the preserved semantic of plan section 4 and the
    /// borrowed path's behavior, which retains the whole native bundle for process
    /// lifetime.
    pub(crate) fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }
}

impl Drop for Surface<'_> {
    fn drop(&mut self) {
        // SAFETY: this is the only owner and the handle was created through this
        // same instance, which the borrow keeps alive; no child object outlives
        // the call, and no allocation callbacks were supplied at creation.
        unsafe { self.loader.destroy_surface(self.handle, None) };
    }
}

impl core::fmt::Debug for Surface<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Surface")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

/// Creates a surface from the host's window for a surface-capable instance.
///
/// The window is borrowed only for the call -- `Vulkan` copies the two handles --
/// but the caller must keep the window alive for as long as the window system can
/// be asked to present through this surface, which is the same lifetime the public
/// façade already gives a window.
pub(crate) fn create<'a>(
    instance: &'a SurfaceInstance,
    window: RawWindowHandle,
) -> Result<Surface<'a>, SurfaceError> {
    let create_info = win32_create_info(window)?;
    let win32 = khr::win32_surface::Instance::new(
        instance.instance().entry(),
        instance.instance().instance(),
    );
    // SAFETY: `create_info` is a local that outlives the call; the window it names
    // is the caller's live window; and the instance enabled `VK_KHR_win32_surface`
    // by construction, so the entry point is the real one rather than `ash`'s
    // unresolved stub. No allocation callbacks are supplied.
    let handle = unsafe { win32.create_win32_surface(&create_info, None) }
        .map_err(SurfaceError::Creation)?;
    let loader = khr::surface::Instance::new(
        instance.instance().entry(),
        instance.instance().instance(),
    );
    Ok(Surface {
        instance,
        loader,
        handle,
        poisoned: AtomicBool::new(false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use raw_window_handle::XlibWindowHandle;

    use crate::Validation;
    use crate::native::vulkan::instance;
    use crate::native::vulkan::test_support::{TestWindow, presenting_adapter, win32};
    use crate::native::vulkan::{adapter, surface};
    use fluxel_rendergraph::{TextureUsage, TextureUsageKind};

    #[test]
    fn a_win32_window_lowers_field_for_field() {
        let info = win32_create_info(win32(0x1234, Some(0x5678)))
            .expect("a Win32 window carrying its module is this backend's window");
        assert_eq!(info.hwnd, 0x1234);
        assert_eq!(info.hinstance, 0x5678);
        assert_eq!(info.flags, vk::Win32SurfaceCreateFlagsKHR::empty());
    }

    #[test]
    fn a_window_without_its_owning_module_is_refused_by_name() {
        // The borrowed path refuses this too rather than passing a null module.
        // `ash` derives no `PartialEq` for the create-info struct, so the refusal is
        // compared as `.err()` rather than as a `Result`.
        assert_eq!(
            win32_create_info(win32(0x1234, None)).err(),
            Some(SurfaceError::MissingInstance)
        );
    }

    #[test]
    fn a_window_from_another_platform_is_refused_by_name() {
        // The mapping must not invent a value for a platform this backend has not
        // been taught; `RawWindowHandle` is `#[non_exhaustive]`, so a wildcard arm
        // would do exactly that.
        let xlib = RawWindowHandle::Xlib(XlibWindowHandle::new(1));
        assert_eq!(
            win32_create_info(xlib).err(),
            Some(SurfaceError::NotWin32)
        );
    }

    // --- the real driver ---

    #[test]
    fn a_real_window_yields_a_real_surface_owned_by_its_instance() {
        // Skips where no loader, no adapter or no window station exists: none of
        // those is this module's subject.
        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let Some(window) = TestWindow::open() else {
            return;
        };
        let surface = create(&surface_instance, window.raw())
            .expect("a live window is a surface this loader can present to");
        assert_ne!(surface.handle(), vk::SurfaceKHR::null());
        // Drop order is the borrow's: the surface is destroyed before the instance
        // and before the window that backs it.
    }

    #[test]
    fn a_real_surface_reports_the_facts_the_fixed_contract_decides_against() {
        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let Ok(adapters) = adapter::enumerate(surface_instance.instance().instance()) else {
            return;
        };
        // No adapter at all is a machine without a Vulkan driver, which is not what
        // this test is about; a surface would still be creatable through the loader.
        if adapters.is_empty() {
            return;
        }
        let Some(window) = TestWindow::open() else {
            return;
        };
        let Ok(surface) = create(&surface_instance, window.raw()) else {
            return;
        };
        // Not a skip: a real surface was created through a real loader and a real
        // window, so an enumerated adapter must be attached to it. The alternative
        // would let this test pass without reading a single fact.
        let (physical_device, facts) = presenting_adapter(&surface, &adapters)
            .expect("a surface this loader created is presentable from an enumerated adapter");

        assert!(
            !facts.present_modes.is_empty(),
            "a surface that offers a format also offers a present mode"
        );

        let requested = vk::Extent2D {
            width: 64,
            height: 64,
        };
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);
        // The join this increment exists for: the lists the driver reported are the
        // lists the pure contract decides against, and the decision is the one the
        // frozen oracle was measured with.
        let presentation = surface::contract(
            &facts.capabilities,
            &facts.formats,
            &facts.present_modes,
            requested,
            usage,
        )
        .expect("the named Windows board serves the fixed presentation contract");
        assert_eq!(presentation.format, surface::PRESENT_FORMAT);
        assert_eq!(presentation.color_space, surface::PRESENT_COLOR_SPACE);
        assert_eq!(presentation.present_mode, surface::PRESENT_MODE);

        // SAFETY: the adapter belongs to the same live instance the surface was
        // created through, and the call only reports facts.
        let families = unsafe {
            surface_instance
                .instance()
                .instance()
                .get_physical_device_queue_family_properties(physical_device)
        };
        assert!(!families.is_empty(), "a physical device reports a family");
        // Step 2 chose a family without a surface; this is the read that tells the
        // swapchain step whether its choice can present. Every reported family
        // answers, and at least one says yes: a surface this loader created is
        // presentable from somewhere.
        let supporting = (0..families.len() as u32)
            .filter(|index| {
                surface
                    .supports_presentation(physical_device, *index)
                    .expect("a live surface answers presentation support for a reported family")
            })
            .count();
        assert!(supporting > 0);
    }
}
