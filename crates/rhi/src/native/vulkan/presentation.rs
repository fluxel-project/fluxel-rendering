//! Step 10's owning half: the `VkSurfaceKHR` created from a host window.
//!
//! [`super::surface`] is the pure half -- it decides *what* a conformant surface
//! must report and lowers that decision into a `VkSwapchainCreateInfoKHR`. This
//! module is the half that owns a driver object: it lowers the host's window handle
//! into the platform create-info, creates the `VkSurfaceKHR` through the instance
//! that enabled the surface extensions, and destroys it in `Drop`.
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
//! - **No swapchain, no acquire, no present.** Those are the next pieces of step
//!   10, and they lower from [`super::surface::swapchain_create_info`] over the
//!   handle this module owns.
//! - **No surface-facts query.** `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` and
//!   its two list calls answer what the surface reports; the contract that consumes
//!   them is already pure, and wiring the query is the swapchain-owning step's
//!   business.
//! - **No presentation-support query.** Whether a queue family can present is
//!   device enumeration, and the queue family is chosen by rule in step 2.

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

/// An owned `VkSurfaceKHR`, bound to the instance that created it.
///
/// The instance field is held for that binding rather than read: it is what makes
/// "destroy the instance before the surface" a compile error, which `Vulkan`'s
/// object model requires.
pub(crate) struct Surface<'a> {
    /// The instance that created this surface and must outlive it.
    instance: &'a SurfaceInstance,
    /// The surface-level entry points, loaded once when the handle was created.
    loader: khr::surface::Instance,
    /// The handle itself.
    handle: vk::SurfaceKHR,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroIsize;

    use raw_window_handle::{Win32WindowHandle, XlibWindowHandle};

    use crate::Validation;
    use crate::native::vulkan::instance;

    /// A fabricated Win32 handle pair, for the pure lowering tests only.
    fn win32(hwnd: isize, hinstance: Option<isize>) -> RawWindowHandle {
        let mut handle = Win32WindowHandle::new(
            NonZeroIsize::new(hwnd).expect("the test's hwnd is non-zero"),
        );
        handle.hinstance = hinstance.and_then(NonZeroIsize::new);
        RawWindowHandle::Win32(handle)
    }

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

    #[link(name = "user32")]
    unsafe extern "system" {
        fn CreateWindowExW(
            ex_style: u32,
            class_name: *const u16,
            window_name: *const u16,
            style: u32,
            x: i32,
            y: i32,
            width: i32,
            height: i32,
            parent: isize,
            menu: isize,
            instance: isize,
            param: *mut core::ffi::c_void,
        ) -> isize;
        fn DestroyWindow(hwnd: isize) -> i32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(module_name: *const u16) -> isize;
    }

    fn wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// A hidden top-level window, destroyed when dropped.
    struct TestWindow {
        hwnd: isize,
        hinstance: isize,
    }

    impl TestWindow {
        /// Creates a hidden `STATIC` window, or `None` where the session has no
        /// window station that can back one.
        ///
        /// `STATIC` is a system class, so no class registration is needed; the
        /// window is never shown and exists only to have a real `HWND`.
        fn open() -> Option<Self> {
            let class = wide("STATIC");
            let title = wide("fluxel-rhi surface test");
            // SAFETY: both are the documented Win32 contract. The null module name
            // asks for this process's own module, and the two string pointers are
            // NUL-terminated locals that outlive the call.
            let hinstance = unsafe { GetModuleHandleW(core::ptr::null()) };
            if hinstance == 0 {
                return None;
            }
            // SAFETY: as above; the window's parent and menu are null, so no other
            // object is referenced.
            let hwnd = unsafe {
                CreateWindowExW(
                    0,
                    class.as_ptr(),
                    title.as_ptr(),
                    0x00CF_0000, // WS_OVERLAPPEDWINDOW, never shown
                    0,
                    0,
                    64,
                    64,
                    0,
                    0,
                    hinstance,
                    core::ptr::null_mut(),
                )
            };
            (hwnd != 0).then_some(Self { hwnd, hinstance })
        }

        fn raw(&self) -> RawWindowHandle {
            win32(self.hwnd, Some(self.hinstance))
        }
    }

    impl Drop for TestWindow {
        fn drop(&mut self) {
            // SAFETY: this is the only owner of the handle `open` created.
            unsafe { DestroyWindow(self.hwnd) };
        }
    }

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
}
