//! Shared scaffolding for the Vulkan tests that need a real window or a real
//! surface.
//!
//! Several step 10 modules are only provable against a real window: the surface
//! handle, the facts it reports, and the swapchain created over it. The Win32
//! window lowering and the "which enumerated adapter is this surface on" search are
//! written once here rather than copied into each of their test modules, which is
//! also what keeps the window's lifetime rule in one place.
//!
//! Compiled under `cfg(test)` only: nothing here is part of the backend.

use ash::vk;
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};

use super::presentation::{Surface, SurfaceFacts};

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

/// A fabricated Win32 handle pair, for the pure lowering tests.
pub(crate) fn win32(hwnd: isize, hinstance: Option<isize>) -> RawWindowHandle {
    let mut handle = Win32WindowHandle::new(
        core::num::NonZeroIsize::new(hwnd).expect("the test's hwnd is non-zero"),
    );
    handle.hinstance = hinstance.and_then(core::num::NonZeroIsize::new);
    RawWindowHandle::Win32(handle)
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A hidden top-level window, destroyed when dropped.
pub(crate) struct TestWindow {
    hwnd: isize,
    hinstance: isize,
}

impl TestWindow {
    /// Creates a hidden `STATIC` window, or `None` where the session has no
    /// window station that can back one.
    ///
    /// `STATIC` is a system class, so no class registration is needed; the
    /// window is never shown and exists only to have a real `HWND`.
    pub(crate) fn open() -> Option<Self> {
        let class = wide("STATIC");
        let title = wide("fluxel-rhi vulkan test");
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

    pub(crate) fn raw(&self) -> RawWindowHandle {
        win32(self.hwnd, Some(self.hinstance))
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the handle `open` created.
        unsafe { DestroyWindow(self.hwnd) };
    }
}

/// The physical device that actually owns `surface`, with the facts it reported, or
/// `None` where no enumerated adapter reports a presentable surface.
///
/// A multi-adapter machine can enumerate an adapter the window is not attached
/// to; `Vulkan` answers that with an empty format list rather than a driver
/// error, so the first adapter that reports a format is the one this surface is
/// on.
pub(crate) fn presenting_adapter(
    surface: &Surface<'_>,
    adapters: &[vk::PhysicalDevice],
) -> Option<(vk::PhysicalDevice, SurfaceFacts)> {
    adapters.iter().find_map(|adapter| {
        let facts = surface.facts(*adapter).ok()?;
        (!facts.formats.is_empty()).then_some((*adapter, facts))
    })
}
