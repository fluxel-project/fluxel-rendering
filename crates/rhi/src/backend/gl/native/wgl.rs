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

use super::discovery::{
    REQUIRED_DESKTOP_VERSION, desktop_context_floor_marker, discover_current_glow_identified,
};
use crate::backend::gl::api::{
    ContextStamp, GlContextFlags, GlContextLifecycle, GlDiscoverySnapshot, GlError, GlVersion,
    OwnerThreadIdentity,
};

use glutin_wgl_sys::{wgl, wgl_extra};

mod gdi;

pub(super) type Hwnd = *mut c_void;
pub(super) type Hdc = *const c_void;

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
/// The request in [`core_context_attributes`], the early check on the actual
/// version string in [`meets_required_desktop_context`], the check on the
/// discovered profile in [`WglContextSurface::open`], and the recorded marker all read
/// this one value, so the four cannot ask for, enforce, accept, or record
/// different things (audit P2-12).
///
/// This used to be a second constant declared here beside
/// `native::discovery`'s, with a cross-check to keep the copies honest. The
/// copies are gone instead: `native` re-exports the constant and the marker
/// builder for exactly this reader, which is what the audit's residual asked
/// for, and a single value cannot drift from itself.
const REQUIRED_DESKTOP_CONTEXT: GlVersion = REQUIRED_DESKTOP_VERSION;

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
    /// The driver rejected every requested desktop core profile down to 4.0.
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
    owner_thread: OwnerThreadIdentity,
    lifecycle: Cell<GlContextLifecycle>,
    extent: Cell<[u32; 2]>,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Immutable observations exported to the fixture layer.  This intentionally
/// contains strings and numbers only; WGL handles and the GL dispatch table
/// remain confined to this module.
#[derive(Clone, Debug)]
pub(crate) struct WglEvidence {
    pub(crate) profile: String,
    pub(crate) version: String,
    pub(crate) shading_language_version: String,
    pub(crate) vendor: String,
    pub(crate) renderer: String,
    pub(crate) driver_or_browser: String,
    pub(crate) debug: bool,
    pub(crate) forward_compatible: bool,
    pub(crate) robust_access: bool,
    pub(crate) no_error: bool,
    pub(crate) other_flags: Vec<String>,
    pub(crate) extensions: Vec<String>,
    /// Typed counterparts of reported extension names. Raw names remain in
    /// `extensions`; this is diagnostic evidence only and never enables a
    /// capability by itself.
    pub(crate) typed_extensions: Vec<(String, String)>,
    pub(crate) limits: Vec<(String, String)>,
    pub(crate) capabilities: Vec<(String, bool)>,
    pub(crate) surface_facts: String,
    pub(crate) drawable_extent: [u32; 2],
}

/// Sendable description of one Host-owned Win32 drawable.
///
/// The descriptor carries an address only long enough for the owner worker to
/// create its private HDC/HGLRC.  It is intentionally not a public RHI object:
/// the Host retains the window and must keep it alive until the returned
/// provider is dropped.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WglWorkerDescriptor {
    stamp: ContextStamp,
    hwnd: usize,
    extent: [u32; 2],
}

impl WglWorkerDescriptor {
    /// Validates Host handles on the caller thread and produces the owned
    /// scalar descriptor that may cross to the GL worker.
    pub(crate) fn new(
        stamp: ContextStamp,
        window: WindowHandle<'_>,
        display: DisplayHandle<'_>,
        extent: [u32; 2],
    ) -> Result<Self, WglContextError> {
        Ok(Self {
            stamp,
            hwnd: host_hwnd(window, display)? as usize,
            extent,
        })
    }
}

/// Creates a complete v13 OpenGL provider on its dedicated WGL owner thread.
///
/// Discovery, capability freezing and dispatch-table creation all happen after
/// the context is current on that worker.  No WGL context, HDC, or `glow`
/// table crosses the thread boundary.
pub(crate) fn spawn_provider(
    instance: crate::api::identity::DeviceInstanceId,
    descriptor: WglWorkerDescriptor,
) -> crate::api::error::RhiResult<crate::backend::gl::platform::GlProvider> {
    let (worker, (facts, name)) = super::NativeOwnerWorker::spawn_with_info(move || {
        let surface = WglContextSurface::open_from_hwnd(
            descriptor.stamp,
            descriptor.hwnd as Hwnd,
            descriptor.extent,
            None,
        )
        .map_err(|error| format!("WGL context creation failed: {error:?}"))?;
        let facts = surface.v13_capability_facts();
        let name = format!("OpenGL ({})", surface.snapshot.context().renderer());
        let provider = surface
            .into_owned_provider()
            .map_err(|error| format!("WGL provider creation failed: {error:?}"))?;
        Ok((provider, (facts, name)))
    })
    .map_err(|error| {
        crate::api::error::RhiError::new(
            crate::api::error::RhiErrorKind::BackendFailure,
            format!("WGL owner worker could not start: {error:?}"),
        )
    })?;
    let owner: std::sync::Arc<dyn super::driver::NativeGlOwner> =
        std::sync::Arc::new(super::driver::NativeProviderOwner::new(worker));
    super::driver::NativeGlDriver::adopt(
        instance,
        crate::api::platform::BackendKind::OpenGl,
        name,
        facts,
        owner,
    )
}

impl WglContextSurface {
    /// Consumes this platform context into the worker-owned v13 executor.
    /// The provider receives a separately loaded dispatch table while this
    /// surface retains HDC/HGLRC ownership and currentness enforcement.
    pub(crate) fn into_owned_provider(
        self,
    ) -> Result<super::driver::NativeOwnedProvider<Self>, GlError> {
        let provider = self.with_current("create WGL owned provider", |_| {
            // SAFETY: `with_current` made this exact WGL context current. The
            // new dispatch table is moved into `NativeGlProvider`, not borrowed
            // from `self`, so the resulting owner has no self-reference.
            let dispatch = unsafe { glow::Context::from_loader_function(load_wgl_symbol) };
            // `open` already collected the authoritative, WGL-identified
            // snapshot for this exact current context.  Re-discovering through
            // the generic constructor would discard that platform identity and
            // make provider capabilities depend on a second observation.
            let mut provider = unsafe {
                super::provider::NativeGlProvider::from_discovered(dispatch, self.snapshot.clone())
            };
            // `open` validated and retained this Host-reported drawable extent
            // for this exact context generation.  `NativeGlProvider` otherwise
            // starts with no extent (which is right for a generic adopted GL
            // context, but would make this concrete WGL surface permanently
            // suspended).  Keep this assignment in the platform factory rather
            // than guessing a default framebuffer size in the executor.
            let extent = self.extent();
            provider.surface_extent = Some(crate::backend::gl::api::GlSurfaceSize {
                width: extent[0],
                height: extent[1],
            });
            provider.surface_suspended = extent.contains(&0);
            Ok(provider)
        })?;
        Ok(super::driver::NativeOwnedProvider::new(self, provider))
    }
    /// Freezes the v13 facts for this exact WGL context generation.
    ///
    /// The returned facts are deliberately derived from the retained discovery
    /// record, not from a version string at adoption time.
    pub(crate) fn v13_capability_facts(&self) -> crate::api::capability::CapabilityFacts {
        super::discovery::v13_capability_snapshot(&self.snapshot).into_facts()
    }
    /// Adopts an already worker-owned route for this exact WGL generation.
    ///
    /// `owner` must own this surface's HDC/HGLRC on its dispatch thread.  This
    /// method intentionally accepts no raw WGL handle, so the common provider
    /// seam receives only facts and executable dispatch.
    pub(crate) fn adopt(
        &self,
        instance: crate::api::identity::DeviceInstanceId,
        owner: std::sync::Arc<dyn super::driver::NativeGlOwner>,
    ) -> crate::api::error::RhiResult<crate::backend::gl::platform::GlProvider> {
        super::adopt_discovered_owner(
            instance,
            format!("OpenGL ({})", self.snapshot.context().renderer()),
            &self.snapshot,
            owner,
        )
    }
    /// Creates a double-buffered WGL surface for the Host's borrowed window.
    ///
    /// A legacy context is only a bootstrap loader for
    /// `wglCreateContextAttribsARB`; it is deleted before this succeeds.  The
    /// retained context is the highest available desktop OpenGL core context
    /// down to 4.0, made current on
    /// this thread before the method returns.
    pub(crate) fn open(
        stamp: ContextStamp,
        window: WindowHandle<'_>,
        display: DisplayHandle<'_>,
        extent: [u32; 2],
    ) -> Result<Self, WglContextError> {
        let hwnd = host_hwnd(window, display)?;
        Self::open_from_hwnd(stamp, hwnd, extent, None)
    }

    /// Opens a desktop core context for a hardware fixture's minimum-version request.
    ///
    /// This is deliberately crate-private: choosing a WGL context version is a
    /// test-harness concern, not portable RHI policy. Unlike [`Self::open`],
    /// this sends one WGL version request rather than applying Fluxel's
    /// highest-to-lowest fallback policy. A driver may legally expose a newer
    /// context than requested; the actual current version is checked to be at
    /// least the requested minimum and is recorded by the fixture.
    pub(crate) fn open_with_minimum_desktop_version(
        stamp: ContextStamp,
        window: WindowHandle<'_>,
        display: DisplayHandle<'_>,
        extent: [u32; 2],
        version: GlVersion,
    ) -> Result<Self, WglContextError> {
        if !is_supported_desktop_test_version(version) {
            return Err(WglContextError::Discovery(format!(
                "WGL fixture only accepts desktop core versions 4.0 through 4.6, got {}.{}",
                version.major, version.minor
            )));
        }
        let hwnd = host_hwnd(window, display)?;
        Self::open_from_hwnd(stamp, hwnd, extent, Some(version))
    }

    /// Opens from a descriptor which was validated before it crossed to the
    /// worker.  Kept private so raw HWND values cannot become a general API.
    fn open_from_hwnd(
        stamp: ContextStamp,
        hwnd: Hwnd,
        extent: [u32; 2],
        requested_version: Option<GlVersion>,
    ) -> Result<Self, WglContextError> {
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
        // WGL has no portable "best version" request. The production path
        // explicitly walks down from 4.6 to the v13 4.0 floor; the fixture path
        // deliberately makes one minimum-version request so each supported
        // floor can be independently exercised on a GL 4.6-capable Windows
        // machine. The driver may return a newer context for that request.
        let create = |version| {
            let attributes = core_context_attributes(version);
            // SAFETY: bootstrap current; attributes are a terminated WGL list.
            let context = unsafe {
                extensions.CreateContextAttribsARB(hdc, core::ptr::null(), attributes.as_ptr())
            };
            (!context.is_null()).then_some(context)
        };
        let core = match requested_version {
            Some(version) => create(version),
            None => DESKTOP_CONTEXT_REQUESTS.into_iter().find_map(create),
        };
        let core = core.unwrap_or(core::ptr::null_mut());
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
        verify_actual_desktop_core_context(&glow, requested_version)?;
        // The driver identity is the platform layer's string and exists only
        // here, so it is read while this exact context is current (audit P2-6).
        let driver_identity = wgl_driver_identity(&glow);
        // Discovery (including the capability operation probes) observes the
        // exact current context before construction completes; a failure here
        // keeps the transaction roll-backed by `cleanup`.
        let snapshot = unsafe {
            discover_current_glow_identified(&glow, stamp, load_wgl_symbol, &driver_identity)
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

    /// Copies only diagnostic facts from the discovery snapshot for fixture
    /// output. It never re-queries a driver, so a report is about the same
    /// context generation that authorised lowering.
    pub(crate) fn evidence(&self) -> WglEvidence {
        use crate::backend::gl::api::GlCapability;
        let context = self.snapshot.context();
        let flags = context.flags();
        let limits = self.snapshot.limits();
        let mut extensions: Vec<_> = self
            .snapshot
            .extensions()
            .raw_reported_names()
            .map(str::to_owned)
            .collect();
        extensions.sort();
        let mut typed_extensions: Vec<_> = extensions
            .iter()
            .filter_map(|raw| {
                let known = crate::backend::gl::api::GlKnownExtension::from_raw_name(raw)?;
                let provenance = self.snapshot.extensions().provenance(known)?;
                Some((known.raw_name().to_owned(), format!("{provenance:?}")))
            })
            .collect();
        typed_extensions.sort();
        let mut other_flags: Vec<_> = flags.other.iter().cloned().collect();
        other_flags.sort();
        let capabilities = [
            ("compute", GlCapability::Compute),
            ("storage-buffer", GlCapability::StorageBuffer),
            ("storage-image", GlCapability::StorageImage),
            ("indirect-draw", GlCapability::IndirectDraw),
            ("indirect-dispatch", GlCapability::IndirectDispatch),
            ("multi-draw", GlCapability::MultiDraw),
            ("multi-draw-indirect", GlCapability::MultiDrawIndirect),
            ("multiview", GlCapability::Multiview),
            ("timer-query", GlCapability::TimerQuery),
        ]
        .into_iter()
        .map(|(name, capability)| {
            (
                name.to_owned(),
                self.snapshot.capabilities().supports(capability),
            )
        })
        .collect();
        WglEvidence {
            profile: format!("{:?}", context.profile()),
            version: context.version().to_owned(),
            shading_language_version: context.shading_language_version().to_owned(),
            vendor: context.vendor().to_owned(),
            renderer: context.renderer().to_owned(),
            driver_or_browser: context.driver_or_browser().to_owned(),
            debug: flags.debug,
            forward_compatible: flags.forward_compatible,
            robust_access: flags.robust_access,
            no_error: flags.no_error,
            other_flags,
            extensions,
            typed_extensions,
            limits: vec![
                (
                    "max_texture_size".into(),
                    limits.max_texture_size.to_string(),
                ),
                (
                    "max_renderbuffer_size".into(),
                    limits.max_renderbuffer_size.to_string(),
                ),
                (
                    "max_vertex_attributes".into(),
                    limits.max_vertex_attributes.to_string(),
                ),
                (
                    "max_viewport".into(),
                    format!(
                        "{}x{}",
                        limits.max_viewport_dimensions[0], limits.max_viewport_dimensions[1]
                    ),
                ),
                ("max_samples".into(), limits.max_samples.to_string()),
                (
                    "uniform_buffer_offset_alignment".into(),
                    limits.uniform_buffer_offset_alignment.to_string(),
                ),
                (
                    "storage_buffer_offset_alignment".into(),
                    limits.storage_buffer_offset_alignment.to_string(),
                ),
                (
                    "max_multiview_view_count".into(),
                    self.snapshot.max_multiview_view_count().to_string(),
                ),
            ],
            capabilities,
            surface_facts: format!("{:?}", self.snapshot.surface_facts()),
            drawable_extent: self.extent(),
        }
    }

    /// Executes a real core-profile triangle workload against the default WGL
    /// framebuffer.  The fixture uses this narrow path to prove the context,
    /// program compiler, vertex path, rasterizer and readback route together;
    /// regular RHI submission remains on the v13 execution driver seam.
    pub(crate) fn draw_evidence(
        &self,
        draws: u32,
        read_colour: bool,
    ) -> Result<Option<Vec<u8>>, GlError> {
        use glow::HasContext as _;
        const OP: &str = "WGL evidence draw";
        let [width, height] = self.extent();
        if width == 0 || height == 0 || draws == 0 {
            return Err(GlError::Validation {
                operation: OP,
                message: "evidence draw requires a nonzero extent and draw count".into(),
            });
        }
        self.with_current(OP, |gl| unsafe {
            let vertex = compile_evidence_shader(gl, glow::VERTEX_SHADER, EVIDENCE_VERTEX, OP)?;
            let fragment =
                match compile_evidence_shader(gl, glow::FRAGMENT_SHADER, EVIDENCE_FRAGMENT, OP) {
                    Ok(shader) => shader,
                    Err(error) => {
                        gl.delete_shader(vertex);
                        return Err(error);
                    }
                };
            let program = gl.create_program().map_err(|message| GlError::Driver {
                operation: OP,
                message,
            })?;
            gl.attach_shader(program, vertex);
            gl.attach_shader(program, fragment);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                let message = gl.get_program_info_log(program);
                gl.delete_program(program);
                gl.delete_shader(vertex);
                gl.delete_shader(fragment);
                return Err(GlError::Driver {
                    operation: OP,
                    message,
                });
            }
            let vao = gl
                .create_vertex_array()
                .map_err(|message| GlError::Driver {
                    operation: OP,
                    message,
                })?;
            gl.bind_vertex_array(Some(vao));
            gl.use_program(Some(program));
            gl.viewport(0, 0, width as i32, height as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            for _ in 0..draws {
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }
            gl.finish();
            let colour = if read_colour {
                let mut bytes = vec![0_u8; width as usize * height as usize * 4];
                gl.read_pixels(
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelPackData::Slice(Some(&mut bytes)),
                );
                Some(bytes)
            } else {
                None
            };
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.delete_vertex_array(vao);
            gl.delete_program(program);
            gl.delete_shader(vertex);
            gl.delete_shader(fragment);
            if gl.get_error() != glow::NO_ERROR {
                return Err(GlError::Driver {
                    operation: OP,
                    message: "OpenGL reported an error while executing the evidence workload"
                        .into(),
                });
            }
            Ok(colour)
        })
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

#[cfg(all(windows, feature = "native-gl-wgl"))]
impl super::driver::NativePlatformContext for WglContextSurface {
    fn make_current(&mut self, operation: &'static str) -> crate::api::error::RhiResult<()> {
        self.bind_current(operation).map_err(|error| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::DeviceLost,
                format!("{operation}: {error:?}"),
            )
        })
    }

    fn supports_presentation(&self) -> bool {
        true
    }

    fn drawable_extent(
        &mut self,
    ) -> crate::api::error::RhiResult<Option<crate::api::presentation::Extent2d>> {
        let extent = self.extent();
        Ok(
            (!extent.contains(&0)).then_some(crate::api::presentation::Extent2d {
                width: extent[0],
                height: extent[1],
            }),
        )
    }

    fn present(&mut self) -> crate::api::error::RhiResult<()> {
        WglContextSurface::present(self).map_err(|error| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::BackendFailure,
                format!("WGL presentation failed: {error:?}"),
            )
        })
    }
}

// At 4x4 these coordinates cover exactly the two pixel centres at row 1,
// columns 1 and 2 in GL's bottom-left readback order. Keep this in lockstep
// with the GLES fixture and scripts/gl_raster_picture.py.
const EVIDENCE_VERTEX: &str = "#version 400 core\nvoid main() { const vec2 p[3]=vec2[3](vec2(-0.6,-0.5),vec2(0.6,-0.5),vec2(0.0,0.0)); gl_Position=vec4(p[gl_VertexID],0.0,1.0); }";
const EVIDENCE_FRAGMENT: &str = "#version 400 core\nout vec4 o; void main() { o=vec4(48.0/255.0,176.0/255.0,112.0/255.0,1.0); }";

fn compile_evidence_shader(
    gl: &glow::Context,
    kind: u32,
    source: &str,
    operation: &'static str,
) -> Result<glow::NativeShader, GlError> {
    use glow::HasContext as _;
    // SAFETY: caller holds this WGL context current for this complete helper.
    unsafe {
        let shader = gl
            .create_shader(kind)
            .map_err(|message| GlError::Driver { operation, message })?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if gl.get_shader_compile_status(shader) {
            Ok(shader)
        } else {
            let message = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            Err(GlError::Driver { operation, message })
        }
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

const DESKTOP_CONTEXT_REQUESTS: [GlVersion; 7] = [
    GlVersion::new(4, 6),
    GlVersion::new(4, 5),
    GlVersion::new(4, 4),
    GlVersion::new(4, 3),
    GlVersion::new(4, 2),
    GlVersion::new(4, 1),
    REQUIRED_DESKTOP_CONTEXT,
];

/// The Windows evidence fixture exercises the complete v13 desktop range.
/// Keep this separate from the production fallback array: the latter is an
/// ordered policy, whereas this predicate makes an explicit fixture request
/// refuse unsupported spelling rather than silently testing some other path.
const fn is_supported_desktop_test_version(version: GlVersion) -> bool {
    version.major == 4 && version.minor <= 6
}

fn core_context_attributes(version: GlVersion) -> [c_int; 7] {
    [
        wgl_extra::CONTEXT_MAJOR_VERSION_ARB as c_int,
        c_int::from(version.major),
        wgl_extra::CONTEXT_MINOR_VERSION_ARB as c_int,
        c_int::from(version.minor),
        wgl_extra::CONTEXT_PROFILE_MASK_ARB as c_int,
        wgl_extra::CONTEXT_CORE_PROFILE_BIT_ARB as c_int,
        0,
    ]
}

/// The exact string [`REQUIRED_DESKTOP_CONTEXT`] is recorded as.
///
/// Built by the module that records it, so the marker this provider requires is
/// the marker discovery stamps rather than a second spelling of it (audit
/// P2-12).
fn recorded_desktop_floor() -> String {
    desktop_context_floor_marker()
}

/// Requires the floor this provider asked for to be the one discovery recorded.
///
/// Both sides of the comparison now come from one source, so this can no longer
/// fail because two constants drifted apart.  It is kept because it still
/// proves something the constant cannot: that the snapshot this provider was
/// handed actually carries the marker, rather than being a snapshot whose
/// profile took a path that never stamped it.  That is a fact about the value
/// in hand, and reading the recorded flags keeps the check pure and free of any
/// context.
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

fn verify_actual_desktop_core_context(
    glow: &glow::Context,
    requested_version: Option<GlVersion>,
) -> Result<(), WglContextError> {
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
    // SAFETY: GL_CONTEXT_PROFILE_MASK is valid on the requested desktop core
    // context. A query error is treated as rejection, never as a core claim.
    let profile_mask = unsafe { glow.get_parameter_i32(0x9126) };
    // SAFETY: consumes the error produced by precisely the preceding query.
    let actual_version = parse_version(&version);
    let version_meets_request = requested_version
        .map(|requested| actual_version.is_some_and(|actual| requested.is_met_by(actual)))
        .unwrap_or_else(|| meets_required_desktop_context(&version));
    if unsafe { glow.get_error() } != glow::NO_ERROR
        || (profile_mask & 0x0000_0001) == 0
        || !version_meets_request
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
