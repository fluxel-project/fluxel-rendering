//! Windows WGL context and window-surface ownership.
//!
//! The Host owns the Win32 window and keeps its borrowed raw handle alive.
//! RHI owns the device context lease, pixel format, WGL contexts, currentness,
//! and presentation.  This module deliberately has no event-loop dependency.

#![cfg(all(windows, feature = "native-gl-wgl"))]

use core::cell::Cell;
use core::ffi::{c_char, c_int, c_void};
use core::marker::PhantomData;
use std::ffi::CString;
use std::rc::Rc;

use raw_window_handle::{DisplayHandle, RawDisplayHandle, RawWindowHandle, WindowHandle};

use super::{
    ContextStamp, GlContextLifecycle, GlDiscoverySnapshot, GlError, GlFamilyProfile,
    OwnerThreadIdentity,
};

use glutin_wgl_sys::{wgl, wgl_extra};

type Hwnd = *mut c_void;
type Hdc = *const c_void;

/// The RHI current-binding arbiter is thread-local because WGL currentness is.
/// It never substitutes bookkeeping for a driver bind: every executable seam
/// still calls `wglMakeCurrent` before touching GL or WGL.
#[derive(Default)]
struct CurrentBindingArbiter(Cell<Option<usize>>);

impl CurrentBindingArbiter {
    fn record(&self, context: wgl::types::HGLRC) {
        self.0.set(Some(context as usize));
    }

    fn clear_if(&self, context: wgl::types::HGLRC) {
        if self.0.get() == Some(context as usize) {
            self.0.set(None);
        }
    }

    #[cfg(test)]
    fn current(&self) -> Option<usize> {
        self.0.get()
    }
}

thread_local! {
    static RHI_CURRENT_WGL: CurrentBindingArbiter = CurrentBindingArbiter::default();
}

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
struct PixelFormatDescriptor {
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
    fn window_rgba() -> Self {
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

/// Failure while RHI owns a WGL context or its window surface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WglContextError {
    /// The Host did not supply a Win32 window handle.
    UnsupportedWindowHandle,
    /// The Host did not supply the Windows display kind.
    UnsupportedDisplayHandle,
    /// RHI could not lease an HDC for the still-live Host window.
    AcquireDeviceContext(String),
    /// Pixel formats are immutable for the lifetime of a Win32 window.
    PixelFormatAlreadyConfigured,
    /// The requested double-buffered RGBA pixel format is unavailable.
    ChoosePixelFormat(String),
    /// GDI rejected installation of the selected pixel format.
    SetPixelFormat(String),
    /// The bootstrap compatibility WGL context could not be created.
    CreateBootstrapContext(String),
    /// The bootstrap context could not be made current.
    MakeBootstrapCurrent(String),
    /// `wglCreateContextAttribsARB` was not exported by the driver.
    MissingCreateContextAttribs,
    /// The driver rejected the required desktop core 4.3 context request.
    CreateCoreContext(String),
    /// The final desktop core context could not be made current.
    MakeCoreCurrent(String),
    /// The requested context was created but did not report desktop GL 4.3+
    /// core-profile facts when queried from the actual current context.
    ActualContext { version: String, profile_mask: i32 },
    /// A provider call was issued from another Rust thread.
    WrongThread {
        expected: OwnerThreadIdentity,
        actual: OwnerThreadIdentity,
    },
    /// The provider lifecycle cannot perform the requested operation.
    Lifecycle(GlContextLifecycle),
    /// The WGL driver rejected a currentness or presentation operation.
    Driver {
        operation: &'static str,
        message: String,
    },
}

impl WglContextError {
    fn driver(operation: &'static str) -> Self {
        Self::Driver {
            operation,
            message: std::io::Error::last_os_error().to_string(),
        }
    }

    fn as_gl_error(&self, operation: &'static str) -> GlError {
        match self {
            Self::WrongThread { expected, actual } => GlError::WrongThread {
                operation,
                expected: *expected,
                actual: *actual,
            },
            Self::Lifecycle(lifecycle) => GlError::InvalidLifecycle {
                operation,
                lifecycle: *lifecycle,
            },
            Self::Driver { message, .. } => GlError::Driver {
                operation,
                message: message.clone(),
            },
            other => GlError::Driver {
                operation,
                message: format!("WGL provider setup error: {other:?}"),
            },
        }
    }
}

/// RHI-owned current WGL context and double-buffered window surface.
///
/// The `Rc` marker makes the provider neither `Send` nor `Sync`: WGL's thread
/// currentness requirement is then enforced both structurally and at each
/// executable entry point.  The Host must keep the source window alive until
/// this value has been dropped.
pub(crate) struct WglContextSurface {
    stamp: ContextStamp,
    hwnd: Hwnd,
    hdc: Hdc,
    hglrc: wgl::types::HGLRC,
    glow: glow::Context,
    owner_thread: OwnerThreadIdentity,
    lifecycle: Cell<GlContextLifecycle>,
    extent: Cell<[u32; 2]>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl WglContextSurface {
    /// Creates a double-buffered WGL surface for the Host's borrowed window.
    ///
    /// A legacy context is only a bootstrap loader for
    /// `wglCreateContextAttribsARB`; it is deleted before this succeeds.  The
    /// retained context is a desktop OpenGL 4.3 core context, made current on
    /// this thread before the method returns.
    pub(crate) fn open(
        stamp: ContextStamp,
        window: WindowHandle<'_>,
        display: DisplayHandle<'_>,
        extent: [u32; 2],
    ) -> Result<Self, WglContextError> {
        let hwnd = host_hwnd(window, display)?;
        // SAFETY: `hwnd` came from a borrowed, live `Win32WindowHandle`; the
        // caller's Host-lifetime contract keeps it valid until our Drop.
        let hdc = unsafe { GetDC(hwnd) };
        if hdc.is_null() {
            return Err(WglContextError::AcquireDeviceContext(last_os_error()));
        }

        let mut cleanup = OpenCleanup::new(hwnd, hdc);
        let format = PixelFormatDescriptor::window_rgba();
        // SAFETY: HDC is leased above and `format` is a valid C-layout PFD for
        // the duration of this call.
        let installed = unsafe { GetPixelFormat(hdc) };
        if installed != 0 {
            return Err(WglContextError::PixelFormatAlreadyConfigured);
        }
        // SAFETY: same HDC/PFD validity as above; this only queries GDI.
        let pixel_format = unsafe { ChoosePixelFormat(hdc, &format) };
        if pixel_format == 0 {
            return Err(WglContextError::ChoosePixelFormat(last_os_error()));
        }
        // SAFETY: SetPixelFormat is called once for this newly configured
        // window HDC, using the format returned from ChoosePixelFormat.
        if unsafe { SetPixelFormat(hdc, pixel_format, &format) } == 0 {
            return Err(WglContextError::SetPixelFormat(last_os_error()));
        }

        // SAFETY: HDC has a compatible installed pixel format.  This legacy
        // WGL context is retained only long enough to load extension pointers.
        let bootstrap = unsafe { wgl::CreateContext(hdc) };
        if bootstrap.is_null() {
            return Err(WglContextError::CreateBootstrapContext(last_os_error()));
        }
        cleanup.bootstrap = bootstrap;
        // SAFETY: both handles are owned by this construction transaction.
        if unsafe { wgl::MakeCurrent(hdc, bootstrap) } == 0 {
            return Err(WglContextError::MakeBootstrapCurrent(last_os_error()));
        }

        let extensions = wgl_extra::Wgl::load_with(load_wgl_symbol);
        if !extensions.CreateContextAttribsARB.is_loaded() {
            return Err(WglContextError::MissingCreateContextAttribs);
        }
        let attributes = core_context_attributes();
        // SAFETY: the bootstrap context is current, the extension pointer was
        // loaded after that currentness, and `attributes` is zero terminated.
        let core = unsafe {
            extensions.CreateContextAttribsARB(hdc, core::ptr::null(), attributes.as_ptr())
        };
        if core.is_null() {
            return Err(WglContextError::CreateCoreContext(last_os_error()));
        }
        cleanup.core = core;

        // SAFETY: detach the bootstrap before destroying it; both calls use
        // only owned handles and obey WGL's make-current/delete sequence.
        if unsafe { wgl::MakeCurrent(hdc, core) } == 0 {
            return Err(WglContextError::MakeCoreCurrent(last_os_error()));
        }
        // SAFETY: the legacy context is no longer current and is construction
        // owned.  It is never exposed to a caller.
        unsafe { wgl::DeleteContext(bootstrap) };
        cleanup.bootstrap = core::ptr::null();

        // SAFETY: the core context is current on this thread. `glow` only
        // stores function pointers; command use remains behind `with_current`.
        let glow = unsafe { glow::Context::from_loader_function(load_wgl_symbol) };
        verify_actual_desktop_core_context(&glow)?;
        let result = Self {
            stamp,
            hwnd,
            hdc,
            hglrc: core,
            glow,
            owner_thread: OwnerThreadIdentity::current(),
            lifecycle: Cell::new(if extent.contains(&0) {
                GlContextLifecycle::Suspended
            } else {
                GlContextLifecycle::Active
            }),
            extent: Cell::new(extent),
            _not_send_sync: PhantomData,
        };
        // Only transfer final-context destruction after its actual driver
        // profile has been checked; a rejected profile remains transactional.
        cleanup.core = core::ptr::null();
        cleanup.disarm();
        Ok(result)
    }

    /// Runs discovery or a future executor while this RHI-owned context is current.
    pub(crate) fn with_current<T>(
        &self,
        operation: &'static str,
        callback: impl FnOnce(&glow::Context) -> Result<T, GlError>,
    ) -> Result<T, GlError> {
        self.bind_current(operation)?;
        callback(&self.glow)
    }

    /// Discovers immutable native evidence after checking the actual WGL
    /// context, rather than treating the requested attributes as authority.
    pub(crate) fn discover(&self) -> Result<GlDiscoverySnapshot, GlError> {
        self.with_current("discover WGL context", |glow| {
            // SAFETY: `with_current` just bound this exact RHI-owned context
            // on its owner thread and keeps access serialized for the closure.
            let snapshot = unsafe { super::native::discover_current_glow(glow, self.stamp) }
                .map_err(|error| GlError::Driver {
                    operation: "discover WGL context",
                    message: format!("native GL discovery failed: {error:?}"),
                })?;
            match snapshot.context().profile() {
                GlFamilyProfile::Desktop { major: 4, minor } if minor >= 3 => Ok(snapshot),
                profile => Err(GlError::Driver {
                    operation: "discover WGL context",
                    message: format!(
                        "WGL context did not report desktop core GL 4.3+: {profile:?}"
                    ),
                }),
            }
        })
    }

    /// Presents the RHI-owned back buffer to the Host-owned live window.
    pub(crate) fn present(&self) -> Result<(), GlError> {
        self.with_current("WGL SwapBuffers", |_| {
            // SAFETY: with_current made this provider's HDC current and it
            // remains owned for the complete call.
            if unsafe { SwapBuffers(self.hdc) } == 0 {
                Err(WglContextError::driver("WGL SwapBuffers").as_gl_error("WGL SwapBuffers"))
            } else {
                Ok(())
            }
        })
    }

    /// Requests the optional WGL swap-control extension when the driver exposes it.
    ///
    /// `Ok(false)` means no swap-control extension exists; presentation still
    /// works with the driver's default interval.
    pub(crate) fn set_swap_interval(&self, interval: i32) -> Result<bool, GlError> {
        self.bind_current("wglSwapIntervalEXT")?;
        // Load the extension only after this provider's final core context is
        // current; WGL function availability is context/provider dependent.
        let extensions = wgl_extra::Wgl::load_with(load_wgl_symbol);
        if !extensions.SwapIntervalEXT.is_loaded() {
            return Ok(false);
        }
        // SAFETY: the optional function pointer was checked as loaded after a
        // current WGL context existed; `interval` is the WGL integer contract.
        if unsafe { extensions.SwapIntervalEXT(interval) } == 0 {
            return Err(
                WglContextError::driver("wglSwapIntervalEXT").as_gl_error("wglSwapIntervalEXT")
            );
        }
        Ok(true)
    }

    /// Records the Host window's new drawable size; no WGL swapchain exists to resize.
    pub(crate) fn resize(&self, extent: [u32; 2]) -> Result<(), GlError> {
        self.assert_owner("WGL resize")
            .map_err(|error| error.as_gl_error("WGL resize"))?;
        match self.lifecycle.get() {
            GlContextLifecycle::Active | GlContextLifecycle::Suspended => {
                self.extent.set(extent);
                self.lifecycle.set(if extent.contains(&0) {
                    GlContextLifecycle::Suspended
                } else {
                    GlContextLifecycle::Active
                });
                Ok(())
            }
            lifecycle => Err(WglContextError::Lifecycle(lifecycle).as_gl_error("WGL resize")),
        }
    }

    /// Marks the window drawable unavailable without destroying the context.
    pub(crate) fn suspend(&self) -> Result<(), GlError> {
        self.assert_owner("WGL suspend")
            .map_err(|error| error.as_gl_error("WGL suspend"))?;
        if self.lifecycle.get() == GlContextLifecycle::Active {
            self.lifecycle.set(GlContextLifecycle::Suspended);
            Ok(())
        } else {
            Err(WglContextError::Lifecycle(self.lifecycle.get()).as_gl_error("WGL suspend"))
        }
    }

    /// Returns the requested/current Host drawable extent.
    pub(crate) fn extent(&self) -> [u32; 2] {
        self.extent.get()
    }

    /// Returns the owner thread and generation binding used by the provider's caller.
    pub(crate) fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner_thread
    }

    /// Returns the immutable Fluxel device/generation binding for this context.
    pub(crate) const fn context_stamp(&self) -> ContextStamp {
        self.stamp
    }

    fn assert_owner(&self, _operation: &'static str) -> Result<(), WglContextError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner_thread {
            Ok(())
        } else {
            Err(WglContextError::WrongThread {
                expected: self.owner_thread,
                actual,
            })
        }
    }

    fn assert_ready(&self, operation: &'static str) -> Result<(), WglContextError> {
        self.assert_owner(operation)?;
        match self.lifecycle.get() {
            GlContextLifecycle::Active => Ok(()),
            lifecycle => Err(WglContextError::Lifecycle(lifecycle)),
        }
    }

    /// Binds this exact RHI-owned context for every GL/WGL executable seam.
    ///
    /// This intentionally does not trust the last RHI call, another RHI
    /// context, or foreign WGL code to have preserved currentness.
    fn bind_current(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_ready(operation)
            .map_err(|error| error.as_gl_error(operation))?;
        // SAFETY: affinity was checked and this provider owns both handles.
        // Calling MakeCurrent on each seam is the actual cross-context arbiter.
        if unsafe { wgl::MakeCurrent(self.hdc, self.hglrc) } == 0 {
            return Err(WglContextError::driver(operation).as_gl_error(operation));
        }
        RHI_CURRENT_WGL.with(|arbiter| arbiter.record(self.hglrc));
        Ok(())
    }
}

impl Drop for WglContextSurface {
    fn drop(&mut self) {
        self.lifecycle.set(GlContextLifecycle::Disposed);
        // SAFETY: !Send prevents safe cross-thread destruction.  If unsafe
        // foreign code nevertheless made this HGLRC current elsewhere,
        // DeleteContext fails; in that case we leak rather than release its
        // HDC or pretend teardown succeeded.  On this thread detach only our
        // exact current context before deletion.
        unsafe {
            if wgl::GetCurrentContext() == self.hglrc
                && wgl::MakeCurrent(self.hdc, core::ptr::null()) == 0
            {
                self.lifecycle.set(GlContextLifecycle::Poisoned);
                return;
            }
            if wgl::DeleteContext(self.hglrc) == 0 {
                self.lifecycle.set(GlContextLifecycle::Poisoned);
                return;
            }
            RHI_CURRENT_WGL.with(|arbiter| arbiter.clear_if(self.hglrc));
            let _ = ReleaseDC(self.hwnd, self.hdc);
        }
    }
}

struct OpenCleanup {
    hwnd: Hwnd,
    hdc: Hdc,
    bootstrap: wgl::types::HGLRC,
    core: wgl::types::HGLRC,
    armed: bool,
}

impl OpenCleanup {
    fn new(hwnd: Hwnd, hdc: Hdc) -> Self {
        Self {
            hwnd,
            hdc,
            bootstrap: core::ptr::null(),
            core: core::ptr::null(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for OpenCleanup {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // SAFETY: these handles were created in this transaction and have not
        // escaped. If the driver refuses deletion, leak the dependent HDC
        // rather than releasing a potentially still-bound surface.
        unsafe {
            if !self.core.is_null()
                && wgl::GetCurrentContext() == self.core
                && wgl::MakeCurrent(self.hdc, core::ptr::null()) == 0
            {
                return;
            }
            if !self.core.is_null() && wgl::DeleteContext(self.core) == 0 {
                return;
            }
            if !self.bootstrap.is_null() {
                if wgl::GetCurrentContext() == self.bootstrap
                    && wgl::MakeCurrent(self.hdc, core::ptr::null()) == 0
                {
                    return;
                }
                if wgl::DeleteContext(self.bootstrap) == 0 {
                    return;
                }
            }
            let _ = ReleaseDC(self.hwnd, self.hdc);
        }
    }
}

fn host_hwnd(
    window: WindowHandle<'_>,
    display: DisplayHandle<'_>,
) -> Result<Hwnd, WglContextError> {
    if !matches!(display.as_raw(), RawDisplayHandle::Windows(_)) {
        return Err(WglContextError::UnsupportedDisplayHandle);
    }
    match window.as_raw() {
        RawWindowHandle::Win32(handle) => Ok(handle.hwnd.get() as Hwnd),
        _ => Err(WglContextError::UnsupportedWindowHandle),
    }
}

fn core_context_attributes() -> [c_int; 7] {
    [
        wgl_extra::CONTEXT_MAJOR_VERSION_ARB as c_int,
        4,
        wgl_extra::CONTEXT_MINOR_VERSION_ARB as c_int,
        3,
        wgl_extra::CONTEXT_PROFILE_MASK_ARB as c_int,
        wgl_extra::CONTEXT_CORE_PROFILE_BIT_ARB as c_int,
        0,
    ]
}

fn load_wgl_symbol(symbol: &str) -> *const c_void {
    let Ok(name) = CString::new(symbol) else {
        return core::ptr::null();
    };
    // SAFETY: CString supplies a NUL-terminated symbol name. WGL documents
    // this query for a current context; bootstrap/core is current at each
    // loader construction site. The returned pointer is only inspected here.
    let proc = unsafe { wgl::GetProcAddress(name.as_ptr() as *const c_char) } as *const c_void;
    if usable_wgl_proc(proc) {
        return proc;
    }
    // WGL deliberately does not return OpenGL 1.1 exports. `glow` needs both
    // that base ABI and WGL-loaded modern entry points, so resolve the former
    // from the already-linked system OpenGL module.
    // SAFETY: both byte strings are NUL terminated. `opengl32` is linked by
    // `glutin_wgl_sys`; no module ownership is acquired or released here.
    let module = unsafe { GetModuleHandleA(c"opengl32.dll".as_ptr()) };
    if module.is_null() {
        return core::ptr::null();
    }
    // SAFETY: `module` is a live borrowed module handle and `name` remains
    // alive until the OS has finished copying/reading the symbol string.
    unsafe { win32_get_proc_address(module, name.as_ptr()) }
}

fn usable_wgl_proc(proc: *const c_void) -> bool {
    let value = proc as isize;
    !proc.is_null() && !matches!(value, 1 | 2 | 3 | -1)
}

fn verify_actual_desktop_core_context(glow: &glow::Context) -> Result<(), WglContextError> {
    use glow::HasContext as _;

    // SAFETY: construction made the owned final HGLRC current on this thread.
    // The query checks the driver's actual context, not WGL request attributes.
    let version = unsafe { glow.get_parameter_string(glow::VERSION) };
    // SAFETY: same current-context invariant as the preceding query.
    if unsafe { glow.get_error() } != glow::NO_ERROR {
        return Err(WglContextError::ActualContext {
            version,
            profile_mask: 0,
        });
    }
    // SAFETY: GL_CONTEXT_PROFILE_MASK is valid on the requested desktop 4.3
    // context. A query error is treated as rejection, never as a core claim.
    let profile_mask = unsafe { glow.get_parameter_i32(0x9126) };
    // SAFETY: consumes the error produced by precisely the preceding query.
    if unsafe { glow.get_error() } != glow::NO_ERROR
        || (profile_mask & 0x0000_0001) == 0
        || !is_desktop_gl_43_or_newer(&version)
    {
        return Err(WglContextError::ActualContext {
            version,
            profile_mask,
        });
    }
    Ok(())
}

fn is_desktop_gl_43_or_newer(version: &str) -> bool {
    let Some(prefix) = version.split_whitespace().next() else {
        return false;
    };
    let mut components = prefix.split('.');
    let Some(major) = components
        .next()
        .and_then(|component| component.parse::<u8>().ok())
    else {
        return false;
    };
    let Some(minor) = components
        .next()
        .and_then(|component| component.parse::<u8>().ok())
    else {
        return false;
    };
    major == 4 && minor >= 3
}

fn last_os_error() -> String {
    std::io::Error::last_os_error().to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        CurrentBindingArbiter, core_context_attributes, is_desktop_gl_43_or_newer, usable_wgl_proc,
    };

    #[test]
    fn requests_exact_desktop_core_43_context() {
        let attributes = core_context_attributes();
        assert_eq!(attributes[1], 4);
        assert_eq!(attributes[3], 3);
        assert_eq!(attributes.last(), Some(&0));
    }

    #[test]
    fn rejects_wgl_documented_invalid_proc_sentinels() {
        for value in [0_isize, 1, 2, 3, -1] {
            assert!(!usable_wgl_proc(value as *const core::ffi::c_void));
        }
        assert!(usable_wgl_proc(4_isize as *const core::ffi::c_void));
    }

    #[test]
    fn arbiter_records_each_context_switch() {
        let arbiter = CurrentBindingArbiter::default();
        arbiter.record(11_usize as *const core::ffi::c_void);
        assert_eq!(arbiter.current(), Some(11));
        arbiter.record(12_usize as *const core::ffi::c_void);
        assert_eq!(arbiter.current(), Some(12));
        arbiter.clear_if(11_usize as *const core::ffi::c_void);
        assert_eq!(arbiter.current(), Some(12));
        arbiter.clear_if(12_usize as *const core::ffi::c_void);
        assert_eq!(arbiter.current(), None);
    }

    #[test]
    fn actual_profile_check_requires_desktop_gl_43() {
        assert!(is_desktop_gl_43_or_newer("4.3.0 Vendor"));
        assert!(is_desktop_gl_43_or_newer("4.6 Core Profile"));
        assert!(!is_desktop_gl_43_or_newer("4.2.0 Vendor"));
        assert!(!is_desktop_gl_43_or_newer("OpenGL ES 3.2"));
    }
}
