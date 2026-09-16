//! Windows WGL context and window-surface ownership.
//!
//! The Host owns the Win32 window and keeps its borrowed raw handle alive.
//! RHI owns the device context lease, pixel format, WGL contexts, currentness,
//! and presentation.  This module deliberately has no event-loop dependency.
//!
//! This module does not own Fluxel's GL object tables, the render pass, or the
//! viewport any pass renders with, and it never resizes the Host's window.
//!
//! `gdi` below is the private Win32 ABI boundary this provider calls through;
//! it holds no provider state. Tests live in this module's own `tests/mod.rs`.

#![cfg(all(windows, feature = "native-gl-wgl"))]

use core::cell::Cell;
use core::ffi::{c_char, c_int, c_void};
use core::marker::PhantomData;
use std::ffi::CString;
use std::rc::Rc;

use raw_window_handle::{DisplayHandle, RawDisplayHandle, RawWindowHandle, WindowHandle};

use super::{
    ContextStamp, GlContextFlags, GlContextLifecycle, GlDiscoverySnapshot, GlError,
    GlFamilyApi as _, GlSurfaceLeaseBook, GlVersion, OwnerThreadIdentity,
};

use glutin_wgl_sys::{wgl, wgl_extra};

mod gdi;
#[cfg(test)]
mod tests;

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

/// The desktop core version this provider asks a WGL driver for.
///
/// The request below, the early check on the actual version string, and the
/// check on the discovered profile all read this one constant, so the three
/// cannot ask for, enforce, or accept different things (audit P2-12).
///
/// `native::discovery` owns the same decision as its `REQUIRED_DESKTOP_VERSION`
/// and stamps it into every desktop snapshot as a recorded marker (audit
/// P2-12/`gl.desktop-context-floor`).  That module is private to `native` and
/// re-exports neither the constant nor the marker builder, so this file cannot
/// name them; `verify_recorded_desktop_floor` therefore re-reads the *recorded*
/// marker and fails this provider's `open` if the two copies ever disagree,
/// which turns a silent divergence into a loud one at the only place that can
/// see both.
const REQUIRED_DESKTOP_CONTEXT: GlVersion = GlVersion::new(4, 3);

/// The prefix of the recorded desktop-floor marker in the discovery snapshot.
const RECORDED_DESKTOP_FLOOR: &str = "gl.desktop-context-floor=";

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
    /// The requested context was created but did not report the required
    /// desktop core version and profile when queried from the actual current
    /// context.
    ActualContext { version: String, profile_mask: i32 },
    /// The Host reported a drawable extent this context cannot map to a viewport.
    ///
    /// Native GL has no query for the default framebuffer's extent, so the size
    /// is the Host's fact; the maximum viewport dimensions that bound it are a
    /// fact this provider did read, and a size above them could only be rejected
    /// later by an unrelated pass.
    DrawableExtentExceedsViewportLimit {
        requested: [u32; 2],
        limit: [u32; 2],
    },
    /// The context was usable but discovery could not gather complete facts.
    Discovery(String),
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
            Self::DrawableExtentExceedsViewportLimit { requested, limit } => GlError::Validation {
                operation,
                message: format!(
                    "drawable extent {requested:?} exceeds the maximum viewport dimensions {limit:?}"
                ),
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
    snapshot: GlDiscoverySnapshot,
    surface: GlSurfaceLeaseBook,
    owner_thread: OwnerThreadIdentity,
    lifecycle: Cell<GlContextLifecycle>,
    extent: Cell<[u32; 2]>,
    /// Provider-local slot counter for back-buffer image identities.
    present_slot: Cell<u32>,
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
        let hdc = unsafe { gdi::get_dc(hwnd) };
        if hdc.is_null() {
            return Err(WglContextError::AcquireDeviceContext(last_os_error()));
        }

        let mut cleanup = OpenCleanup::new(hwnd, hdc);
        let format = gdi::PixelFormatDescriptor::window_rgba();
        // SAFETY: HDC is leased above and `format` is a valid C-layout PFD for
        // the duration of this call.
        let installed = unsafe { gdi::get_pixel_format(hdc) };
        if installed != 0 {
            return Err(WglContextError::PixelFormatAlreadyConfigured);
        }
        // SAFETY: same HDC/PFD validity as above; this only queries GDI.
        let pixel_format = unsafe { gdi::choose_pixel_format(hdc, &format) };
        if pixel_format == 0 {
            return Err(WglContextError::ChoosePixelFormat(last_os_error()));
        }
        // SAFETY: SetPixelFormat is called once for this newly configured
        // window HDC, using the format returned from ChoosePixelFormat.
        if unsafe { gdi::set_pixel_format(hdc, pixel_format, &format) } == 0 {
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
        // The driver identity is the platform layer's string and exists only
        // here, so it is read while this exact context is current (audit P2-6).
        let driver_identity = wgl_driver_identity(&glow);
        // Discovery (including the capability operation probes) observes the
        // exact current context before construction completes; a failure here
        // keeps the transaction roll-backed by `cleanup`.
        let snapshot = unsafe {
            super::native::discover_current_glow_identified(
                &glow,
                stamp,
                load_wgl_symbol,
                &driver_identity,
            )
        }
        .map_err(|error| {
            WglContextError::Discovery(format!("native GL discovery failed: {error:?}"))
        })?;
        if !snapshot
            .context()
            .profile()
            .meets(Some(REQUIRED_DESKTOP_CONTEXT), None)
        {
            return Err(WglContextError::Discovery(format!(
                "WGL context did not report the required desktop core version {:?}",
                REQUIRED_DESKTOP_CONTEXT
            )));
        }
        verify_recorded_desktop_floor(snapshot.context().flags())?;
        // The Host's initial drawable size is bounded at the same point as every
        // later resize, so a provider is never handed back for a drawable whose
        // viewport could not exist.
        check_drawable_extent(snapshot.limits().max_viewport_dimensions, extent)?;
        let result = Self {
            stamp,
            hwnd,
            hdc,
            hglrc: core,
            glow,
            snapshot,
            surface: GlSurfaceLeaseBook::new(),
            owner_thread: OwnerThreadIdentity::current(),
            lifecycle: Cell::new(if extent.contains(&0) {
                GlContextLifecycle::Suspended
            } else {
                GlContextLifecycle::Active
            }),
            extent: Cell::new(extent),
            present_slot: Cell::new(0),
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

    /// Returns the discovery evidence gathered when this context opened.
    ///
    /// Discovery ran once against the exact current context during
    /// construction, so it needs no repeat currentness and never re-queries.
    pub(crate) fn discover(&self) -> Result<&GlDiscoverySnapshot, GlError> {
        self.bind_current("discover WGL context")?;
        Ok(&self.snapshot)
    }

    /// Presents the RHI-owned back buffer to the Host-owned live window.
    pub(crate) fn present(&self) -> Result<(), GlError> {
        self.with_current("WGL SwapBuffers", |_| {
            // SAFETY: with_current made this provider's HDC current and it
            // remains owned for the complete call.
            if unsafe { gdi::swap_buffers(self.hdc) } == 0 {
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
    ///
    /// The size is validated against the context's recorded viewport bound
    /// before any state changes, so a window the driver could not address is
    /// rejected instead of being recorded as a usable drawable.
    pub(crate) fn resize(&self, extent: [u32; 2]) -> Result<(), GlError> {
        const OP: &str = "WGL resize";
        self.assert_owner(OP)
            .map_err(|error| error.as_gl_error(OP))?;
        check_drawable_extent(self.snapshot.limits().max_viewport_dimensions, extent)
            .map_err(|error| error.as_gl_error(OP))?;
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
            lifecycle => Err(WglContextError::Lifecycle(lifecycle).as_gl_error(OP)),
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

    /// Reattaches a suspended context after the Host window became available.
    pub(crate) fn resume(&self) -> Result<(), GlError> {
        self.assert_owner("WGL resume")
            .map_err(|error| error.as_gl_error("WGL resume"))?;
        match self.lifecycle.get() {
            GlContextLifecycle::Suspended => {
                self.lifecycle.set(GlContextLifecycle::Active);
                Ok(())
            }
            GlContextLifecycle::Active => Ok(()),
            lifecycle => Err(WglContextError::Lifecycle(lifecycle).as_gl_error("WGL resume")),
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
            gdi::release_dc(self.hwnd, self.hdc);
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
            gdi::release_dc(self.hwnd, self.hdc);
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
        c_int::from(REQUIRED_DESKTOP_CONTEXT.major),
        wgl_extra::CONTEXT_MINOR_VERSION_ARB as c_int,
        c_int::from(REQUIRED_DESKTOP_CONTEXT.minor),
        wgl_extra::CONTEXT_PROFILE_MASK_ARB as c_int,
        wgl_extra::CONTEXT_CORE_PROFILE_BIT_ARB as c_int,
        0,
    ]
}

/// The exact string [`REQUIRED_DESKTOP_CONTEXT`] is recorded as.
///
/// Built here rather than read from the discovery module, because that module's
/// marker builder is not visible outside `native`; `verify_recorded_desktop_floor`
/// is what makes the duplication safe.
fn recorded_desktop_floor() -> String {
    format!(
        "{RECORDED_DESKTOP_FLOOR}{}.{}",
        REQUIRED_DESKTOP_CONTEXT.major, REQUIRED_DESKTOP_CONTEXT.minor
    )
}

/// Requires the floor this provider asked for to be the one discovery recorded.
///
/// A desktop snapshot always carries the marker, so a missing or different one
/// means this file and `native::discovery` disagree about what the family
/// requires -- the two can no longer drift apart silently (audit P2-12).
///
/// It reads the recorded flags rather than the whole snapshot so the agreement
/// it enforces is expressible as a pure check with no context in hand.
fn verify_recorded_desktop_floor(flags: &GlContextFlags) -> Result<(), WglContextError> {
    let expected = recorded_desktop_floor();
    if flags.other.contains(&expected) {
        Ok(())
    } else {
        Err(WglContextError::Discovery(format!(
            "WGL provider requires {expected}, but the discovery record does not carry it"
        )))
    }
}

/// The Windows platform layer's driver identity for the current context.
///
/// Unlike an EGL display, Windows' WGL layer answers no query that names the
/// driver it dispatched to, and the version string is already its own field in
/// the discovery record -- repeating it here is exactly the P2-6 defect.  The
/// installable client driver is the layer that runs these commands, and its
/// vendor and renderer pair is the only identity it publishes, so the pair is
/// what this provider supplies as the platform's driver identity.
///
/// Both reads are this function's own queries, so the error they may raise is
/// consumed here rather than left pending for discovery to misread as a failure
/// of the context it is about to describe.  A blank answer returns an empty
/// string, which discovery records as an unavailable identity instead of as a
/// driver name nobody observed.
fn wgl_driver_identity(glow: &glow::Context) -> String {
    use glow::HasContext as _;

    // SAFETY: the caller made this exact context current on this thread and
    // keeps it current for the whole call; both queries only read.
    let vendor = unsafe { glow.get_parameter_string(glow::VENDOR) };
    // SAFETY: same current-context invariant as the preceding query.
    let renderer = unsafe { glow.get_parameter_string(glow::RENDERER) };
    // SAFETY: same current-context invariant; this only consumes the error the
    // two queries above may have produced.
    let _ = unsafe { glow.get_error() };
    driver_identity_from(&vendor, &renderer)
}

/// Composes the platform driver identity from the driver's own two strings.
fn driver_identity_from(vendor: &str, renderer: &str) -> String {
    match (vendor.trim(), renderer.trim()) {
        ("", "") => String::new(),
        ("", renderer) => renderer.to_owned(),
        (vendor, "") => vendor.to_owned(),
        (vendor, renderer) => format!("{vendor} {renderer}"),
    }
}

/// Whether a Host-reported drawable extent can be a viewport on this context.
///
/// The drawable's size is the Host's fact on this family, and the viewport any
/// pass must map into it is bounded by the driver's queried maximum viewport
/// dimensions.  Recording a size the context cannot map is what would otherwise
/// surface as an unrelated driver error in the first pass that draws into the
/// drawable, so the bound is checked where the size is recorded.
///
/// A zero-area extent is not a rejection: it is how a Host reports a minimized
/// or not-yet-shown window and becomes suspension instead.  A limit of zero
/// means the context reported none, so nothing can be measured against it and
/// the check cannot fail closed on a fact that was never read.
fn within_viewport_limit(limit: [u32; 2], extent: [u32; 2]) -> bool {
    if extent.contains(&0) || limit.contains(&0) {
        return true;
    }
    extent[0] <= limit[0] && extent[1] <= limit[1]
}

/// Resolves a drawable extent against the context's recorded viewport limit.
fn check_drawable_extent(limit: [u32; 2], extent: [u32; 2]) -> Result<(), WglContextError> {
    if within_viewport_limit(limit, extent) {
        return Ok(());
    }
    Err(WglContextError::DrawableExtentExceedsViewportLimit {
        requested: extent,
        limit,
    })
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
    let module = unsafe { gdi::opengl32_module() };
    if module.is_null() {
        return core::ptr::null();
    }
    // SAFETY: `module` is a live borrowed module handle and `name` remains
    // alive until the OS has finished copying/reading the symbol string.
    unsafe { gdi::module_proc_address(module, name.as_ptr()) }
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
        || !meets_required_desktop_context(&version)
    {
        return Err(WglContextError::ActualContext {
            version,
            profile_mask,
        });
    }
    Ok(())
}

/// Whether an actual version string meets [`REQUIRED_DESKTOP_CONTEXT`].
///
/// The requirement is a floor, not a spelling: it compares parsed components
/// against the same constant the context was requested with, so a relaxation
/// cannot move the request without moving the check (audit P2-12).  Which
/// desktop majors this family accepts at all stays where it belongs -- in the
/// profile parser, which answers it for every domain at once.
fn meets_required_desktop_context(version: &str) -> bool {
    parse_version(version).is_some_and(|actual| REQUIRED_DESKTOP_CONTEXT.is_met_by(actual))
}

fn parse_version(version: &str) -> Option<GlVersion> {
    let prefix = version.split_whitespace().next()?;
    let mut components = prefix.split('.');
    let major = components
        .next()
        .and_then(|component| component.parse::<u8>().ok())?;
    let minor = components
        .next()
        .and_then(|component| component.parse::<u8>().ok())?;
    Some(GlVersion::new(major, minor))
}

fn last_os_error() -> String {
    std::io::Error::last_os_error().to_string()
}

/// Family-owner facet of the WGL surface provider.
///
/// The provider owns the context and the surface, so it also answers the
/// owner/lifecycle questions the presentation domain preflights. GL object
/// domains stay with the borrowed-context executor (`NativeGlProvider`),
/// which the caller assembles over `glow` while this provider is current.
impl super::GlFamilyApi for WglContextSurface {
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle.get()
    }

    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner_thread
    }

    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_owner(operation)
            .map_err(|error| error.as_gl_error(operation))
    }

    fn discovery(&self) -> &GlDiscoverySnapshot {
        &self.snapshot
    }

    /// Records loss durably. The native context is Host-owned, so this only
    /// freezes the surface state; every acquire lease is invalidated.
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.assert_owner("context-lost")
            .map_err(|error| error.as_gl_error("context-lost"))?;
        self.lifecycle.set(GlContextLifecycle::Lost);
        let _ = self.surface.invalidate_generation();
        Ok(())
    }

    /// A WGL surface cannot resurrect its own context: the Host recreates it
    /// through [`WglContextSurface::open`], which re-runs discovery under a
    /// new stamp instead of pretending the old generation came back.
    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        Err(GlError::Unsupported {
            operation: "context-restored",
            reason: "re-open the WGL provider with a newly created context instead",
        })
    }
}

/// Surface presentation over the RHI-owned WGL back buffer.
///
/// The Host owns the window and its size events; this domain records the
/// drawable extent, invalidates acquire leases on every transition, and hands
/// the actual flip to `SwapBuffers`. It never schedules frames or touches the
/// event loop.
///
/// # What this domain deliberately does not do (audit P1-11, residue)
///
/// It does not apply the recorded [`super::GlSurfaceSize`] to the GL viewport.
/// That is not an omission that a later call here would fix:
///
/// 1. The GL viewport is a single piece of context-global state with no
///    association to a drawable, and this provider does not own the
///    draw-framebuffer binding. Writing a viewport from a resize would land on
///    whichever framebuffer happened to be bound and would then be overwritten
///    by the next pass's own `GlRasterDescriptor` viewport, so the write would
///    be both unordered and ineffective.
/// 2. No Layer-1 pass can target the default framebuffer at all:
///    `GlFramebufferDescriptor::validate` rejects an attachment-less
///    framebuffer, so the swapchain has no Layer-1 pass to render through yet.
///    Which viewport a default-framebuffer pass renders with is therefore a
///    Layer 2/3 decision, not a provider decision.
///
/// What the provider does enforce is the fact it *can* own: the drawable extent
/// is bounded by the context's recorded maximum viewport dimensions, and an
/// extent the context could not address is rejected fail-closed before any
/// lease or lifecycle state changes. Applying a viewport needs a
/// default-framebuffer pass contract that does not exist in this layer; the
/// viewport residue is NOT CLOSED here and is recorded as such.
impl super::GlSurfacePresentationApi for WglContextSurface {
    fn acquire_surface_image(&mut self) -> Result<super::GlSurfaceAcquire, GlError> {
        const OP: &str = "acquire-surface-image";
        self.assert_owner(OP)
            .map_err(|error| error.as_gl_error(OP))?;
        if self.lifecycle.get() != GlContextLifecycle::Active {
            // A suspended or lost drawable cannot present; report suspension
            // instead of handing out an unbacked lease.
            return Ok(super::GlSurfaceAcquire::Suspended);
        }
        let [width, height] = self.extent.get();
        if width == 0 || height == 0 {
            return Ok(super::GlSurfaceAcquire::Suspended);
        }
        let size = super::GlSurfaceSize { width, height };
        // Surface-image identities come from a provider-local counter because
        // the WGL back buffer is a single implicit allocation.
        self.present_slot
            .set(self.present_slot.get().wrapping_add(1));
        let image = super::SurfaceImageId::new(self.context_stamp(), self.present_slot.get(), 0);
        let lease = self.surface.acquire(image, size)?;
        Ok(super::GlSurfaceAcquire::Lease(lease))
    }

    fn resize_surface(&mut self, size: super::GlSurfaceSize) -> Result<(), GlError> {
        const OP: &str = "resize-surface";
        self.assert_owner(OP)
            .map_err(|error| error.as_gl_error(OP))?;
        // Bounded before the lease invalidation, which is a side effect: an
        // unaddressable drawable must not be recorded as a usable one and must
        // not cost the caller its outstanding leases.
        check_drawable_extent(
            self.snapshot.limits().max_viewport_dimensions,
            [size.width, size.height],
        )
        .map_err(|error| error.as_gl_error(OP))?;
        // Resize always invalidates outstanding acquire leases first. The
        // Host owns the actual Win32 window size; this records the drawable
        // fact the presentation domain must match.
        self.surface.invalidate_generation()?;
        match self.lifecycle.get() {
            GlContextLifecycle::Active | GlContextLifecycle::Suspended => {
                self.extent.set([size.width, size.height]);
                self.lifecycle.set(if size.is_zero() {
                    GlContextLifecycle::Suspended
                } else {
                    GlContextLifecycle::Active
                });
                Ok(())
            }
            lifecycle => Err(WglContextError::Lifecycle(lifecycle).as_gl_error(OP)),
        }
    }

    fn suspend_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "suspend-surface";
        self.assert_owner(OP)
            .map_err(|error| error.as_gl_error(OP))?;
        self.surface.invalidate_generation()?;
        self.suspend()
    }

    fn resume_surface(&mut self) -> Result<(), GlError> {
        const OP: &str = "resume-surface";
        self.assert_owner(OP)
            .map_err(|error| error.as_gl_error(OP))?;
        self.surface.invalidate_generation()?;
        self.resume()
    }

    fn present_surface(&mut self, lease: super::GlSurfaceLease) -> Result<(), GlError> {
        const OP: &str = "present-surface";
        self.validate_object_context(OP, lease.image.context)?;
        self.surface.consume(lease)?;
        // The flip happens after the lease is consumed: a rejected swap must
        // not resurrect a consumed lease, and the caller retries with a new
        // acquisition.
        self.present()
    }
}
