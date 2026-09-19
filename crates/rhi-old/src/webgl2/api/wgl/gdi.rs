//! The private GDI/Win32 ABI boundary the WGL provider calls through.
//!
//! Single responsibility: declare the small part of Win32 that a WGL window
//! surface needs and is not already bound by `glutin_wgl_sys` -- the pixel
//! format description, the window RGBA format this provider selects, the device
//! context lease, the buffer swap, and the module-symbol lookup used to reach
//! the base OpenGL entry points that WGL itself does not export.
//!
//! It deliberately owns no provider state: it holds no HDC, no HGLRC, no
//! currentness arbiter, and no lifecycle. Every function here is a thin,
//! exactly-typed call; the transaction that decides *when* to call them, and
//! what a failure means, stays in the parent module. This split exists because a
//! change to that transaction and a change to the C ABI change independently.
//!
//! # Safety
//!
//! Every item that reaches the operating system is an `unsafe fn` carrying its
//! own `# Safety` contract. The handles are raw `Hwnd`/`Hdc` values this crate
//! does not own: the Host lends the window, and the parent module leases and
//! releases the device context. Passing a handle that is not live, or using one
//! from a thread other than the one that created it, is outside Rust's type
//! system and is the caller's contract to uphold.

use core::ffi::{c_char, c_int, c_void};

use super::{Hdc, Hwnd};

const PFD_DOUBLEBUFFER: u32 = 0x0000_0001;
const PFD_DRAW_TO_WINDOW: u32 = 0x0000_0004;
const PFD_SUPPORT_OPENGL: u32 = 0x0000_0020;
const PFD_TYPE_RGBA: u8 = 0;
const PFD_MAIN_PLANE: i8 = 0;

/// The ABI layout used by the small GDI surface boundary below.
///
/// `glutin_wgl_sys` deliberately binds WGL rather than GDI's pixel-format and
/// swap functions.  Keeping this exact private ABI here avoids pulling a
/// windowing framework (or a second Win32 binding crate) into the provider.
#[repr(C)]
pub(super) struct PixelFormatDescriptor {
    size: u16,
    version: u16,
    flags: u32,
    pixel_type: u8,
    color_bits: u8,
    red_bits: u8,
    red_shift: u8,
    green_bits: u8,
    green_shift: u8,
    blue_bits: u8,
    blue_shift: u8,
    alpha_bits: u8,
    alpha_shift: u8,
    accum_bits: u8,
    accum_red_bits: u8,
    accum_green_bits: u8,
    accum_blue_bits: u8,
    accum_alpha_bits: u8,
    depth_bits: u8,
    stencil_bits: u8,
    aux_buffers: u8,
    layer_type: i8,
    reserved: u8,
    layer_mask: u32,
    visible_mask: u32,
    damage_mask: u32,
}

impl PixelFormatDescriptor {
    pub(super) fn window_rgba() -> Self {
        Self {
            size: core::mem::size_of::<Self>() as u16,
            version: 1,
            flags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
            pixel_type: PFD_TYPE_RGBA,
            color_bits: 24,
            red_bits: 0,
            red_shift: 0,
            green_bits: 0,
            green_shift: 0,
            blue_bits: 0,
            blue_shift: 0,
            alpha_bits: 8,
            alpha_shift: 0,
            accum_bits: 0,
            accum_red_bits: 0,
            accum_green_bits: 0,
            accum_blue_bits: 0,
            accum_alpha_bits: 0,
            depth_bits: 24,
            stencil_bits: 8,
            aux_buffers: 0,
            layer_type: PFD_MAIN_PLANE,
            reserved: 0,
            layer_mask: 0,
            visible_mask: 0,
            damage_mask: 0,
        }
    }
}

#[link(name = "user32")]
unsafe extern "system" {
    fn GetDC(hwnd: Hwnd) -> Hdc;
    fn ReleaseDC(hwnd: Hwnd, hdc: Hdc) -> c_int;
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn ChoosePixelFormat(hdc: Hdc, format: *const PixelFormatDescriptor) -> c_int;
    fn GetPixelFormat(hdc: Hdc) -> c_int;
    fn SetPixelFormat(hdc: Hdc, pixel_format: c_int, format: *const PixelFormatDescriptor)
    -> c_int;
    fn SwapBuffers(hdc: Hdc) -> c_int;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleA(module_name: *const c_char) -> *mut c_void;
    #[link_name = "GetProcAddress"]
    fn win32_get_proc_address(module: *mut c_void, name: *const c_char) -> *const c_void;
}

/// Leases the window's device context from GDI.
///
/// # Safety
///
/// `hwnd` must be a live Win32 window handle on the calling thread. The caller
/// owns the returned lease and must release it exactly once with [`release_dc`].
pub(super) unsafe fn get_dc(hwnd: Hwnd) -> Hdc {
    // SAFETY: forwarded from this function's documented contract.
    unsafe { GetDC(hwnd) }
}

/// Releases a device context previously leased with [`get_dc`].
///
/// # Safety
///
/// `hdc` must be a lease from [`get_dc`] on `hwnd` that has not already been
/// released, and no GL or WGL command may still be bound to it.
pub(super) unsafe fn release_dc(hwnd: Hwnd, hdc: Hdc) {
    // SAFETY: forwarded from this function's documented contract.
    let _ = unsafe { ReleaseDC(hwnd, hdc) };
}

/// Returns the pixel format already installed on the device context, if any.
///
/// # Safety
///
/// `hdc` must be a live device context on the calling thread.
pub(super) unsafe fn get_pixel_format(hdc: Hdc) -> c_int {
    // SAFETY: forwarded from this function's documented contract.
    unsafe { GetPixelFormat(hdc) }
}

/// Selects the closest available pixel format for `format`.
///
/// # Safety
///
/// `hdc` must be a live device context on the calling thread and `format` must
/// stay valid for the duration of the call.
pub(super) unsafe fn choose_pixel_format(hdc: Hdc, format: *const PixelFormatDescriptor) -> c_int {
    // SAFETY: forwarded from this function's documented contract.
    unsafe { ChoosePixelFormat(hdc, format) }
}

/// Installs `pixel_format` on the device context.
///
/// # Safety
///
/// `hdc` must be a live device context that does not already have a pixel
/// format installed, and `format` must stay valid for the duration of the call.
pub(super) unsafe fn set_pixel_format(
    hdc: Hdc,
    pixel_format: c_int,
    format: *const PixelFormatDescriptor,
) -> c_int {
    // SAFETY: forwarded from this function's documented contract.
    unsafe { SetPixelFormat(hdc, pixel_format, format) }
}

/// Presents the back buffer of a double-buffered device context.
///
/// # Safety
///
/// `hdc` must be a live device context whose window holds a double-buffered
/// pixel format on the calling thread.
pub(super) unsafe fn swap_buffers(hdc: Hdc) -> c_int {
    // SAFETY: forwarded from this function's documented contract.
    unsafe { SwapBuffers(hdc) }
}

/// Returns the already-linked system OpenGL module, or null.
///
/// # Safety
///
/// This only queries the loader for a module this process already links; it
/// acquires and releases no module ownership.
pub(super) unsafe fn opengl32_module() -> *mut c_void {
    // SAFETY: the byte string is NUL terminated. `opengl32` is linked by
    // `glutin_wgl_sys`; no module ownership is acquired or released here.
    unsafe { GetModuleHandleA(c"opengl32.dll".as_ptr()) }
}

/// Resolves one exported symbol inside a module handle.
///
/// # Safety
///
/// `module` must be a live borrowed module handle and `name` must point at a
/// NUL-terminated symbol name that stays alive for the call.
pub(super) unsafe fn module_proc_address(
    module: *mut c_void,
    name: *const c_char,
) -> *const c_void {
    // SAFETY: forwarded from this function's documented contract.
    unsafe { win32_get_proc_address(module, name) }
}
