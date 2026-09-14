//! EGL-owned GLES contexts for the native GL-family provider.
//!
//! The Host supplies raw display/window handles and keeps the underlying
//! native objects alive.  This provider owns every EGL object created from
//! those handles: display initialization, config, context, surface, current
//! binding, presentation, and teardown.  It deliberately does not create a
//! window, event loop, or an ANGLE/DX translation provider.
//!
//! This is currently a Linux Xlib/Wayland EGL provider, plus headless pbuffers.
//! `eglGetDisplay` is intentionally limited to those handle pairs. EGL's
//! platform-display extension is not exposed by `khronos-egl`, so accepting
//! XCB, GBM, DRM, or Android handles here would require guessing extension ABI
//! calls. Such handles fail closed and this module makes no portable Android
//! support claim.

use core::ffi::c_void;
use core::ptr;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::{GlContextLifecycle, GlError, GlFamilyProfile, OwnerThreadIdentity};

type Egl = khronos_egl::DynamicInstance<khronos_egl::EGL1_4>;

thread_local! {
    /// EGL currentness is thread-local.  Keeping the display lease registry on
    /// that same thread prevents a second context from terminating a display
    /// still used by its sibling context, without claiming EGL is Send/Sync.
    static DISPLAY_LEASES: RefCell<HashMap<usize, Weak<EglDisplayLease>>> = RefCell::new(HashMap::new());
}

const EGL_CONTEXT_MAJOR_VERSION_KHR: i32 = 0x3098;
const EGL_CONTEXT_MINOR_VERSION_KHR: i32 = 0x30FB;
const EGL_KHR_CREATE_CONTEXT: &str = "EGL_KHR_create_context";

/// GLES context version requested from an EGL provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EglGlesVersion {
    /// OpenGL ES 3.0.
    V3_0,
    /// OpenGL ES 3.1.
    V3_1,
    /// OpenGL ES 3.2.
    V3_2,
}

impl EglGlesVersion {
    const fn minor(self) -> i32 {
        match self {
            Self::V3_0 => 0,
            Self::V3_1 => 1,
            Self::V3_2 => 2,
        }
    }

    const fn profile(self) -> GlFamilyProfile {
        GlFamilyProfile::Embedded {
            major: 3,
            minor: self.minor() as u8,
        }
    }
}

/// Pixel dimensions used for an EGL pbuffer probe surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EglPbufferSize {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl EglPbufferSize {
    /// A small valid default suitable for capability probes.
    pub const PROBE: Self = Self {
        width: 16,
        height: 16,
    };

    fn checked(self) -> Result<(i32, i32), EglProviderError> {
        let width = i32::try_from(self.width).map_err(|_| EglProviderError::InvalidSize(self))?;
        let height = i32::try_from(self.height).map_err(|_| EglProviderError::InvalidSize(self))?;
        if width == 0 || height == 0 {
            return Err(EglProviderError::InvalidSize(self));
        }
        Ok((width, height))
    }
}

/// EGL provider setup or presentation failure.
#[derive(Debug)]
pub(crate) enum EglProviderError {
    /// The dynamic EGL implementation could not be loaded.
    Load(String),
    /// The supplied raw-handle pair has no documented `eglGetDisplay` mapping.
    UnsupportedHandlePair {
        display: &'static str,
        window: &'static str,
    },
    /// The requested EGL display could not be obtained.
    DisplayUnavailable,
    /// EGL did not expose the extension required to request a GLES minor version.
    ExactVersionUnavailable(EglGlesVersion),
    /// No EGL config satisfies the provider's GLES surface requirements.
    ConfigUnavailable,
    /// A raw handle omitted a field required by this provider's documented mapping.
    InvalidNativeHandle(&'static str),
    /// A pbuffer size was zero or outside EGL's signed attribute range.
    InvalidSize(EglPbufferSize),
    /// An EGL call failed.  The operation label is stable for Layer 2 logging.
    Egl {
        operation: &'static str,
        error: khronos_egl::Error,
    },
    /// The current GLES context did not match the version requested by Fluxel.
    UnexpectedProfile {
        requested: EglGlesVersion,
        observed: String,
    },
    /// The provider was used from a thread other than its owner.
    Gl(GlError),
}

impl From<GlError> for EglProviderError {
    fn from(value: GlError) -> Self {
        Self::Gl(value)
    }
}

#[derive(Clone, Copy, Debug)]
enum EglSurfaceKind {
    Window { native_window: *mut c_void },
    Pbuffer(EglPbufferSize),
}

/// Shared initialized EGL display.  It terminates exactly once, when its last
/// context lease is dropped on the owner thread.
struct EglDisplayLease {
    egl: Egl,
    display: khronos_egl::Display,
}

impl Drop for EglDisplayLease {
    fn drop(&mut self) {
        // EGL has no safe cross-thread ownership model. Every safe constructor
        // and context is !Send, so this is the owning thread in normal use.
        let _ = self.egl.terminate(self.display);
    }
}

/// A native EGL GLES context and its owned presentation surface.
///
/// This type is deliberately `!Send` and `!Sync`: EGL context currentness is
/// thread-affine, and all calls verify the captured Rust thread identity before
/// entering EGL.  Destruction is best-effort because `Drop` cannot report a
/// driver error; explicit [`dispose`](Self::dispose) is available to observe it.
pub(crate) struct EglGlesContext {
    lease: Rc<EglDisplayLease>,
    config: khronos_egl::Config,
    context: khronos_egl::Context,
    surface: Option<khronos_egl::Surface>,
    kind: EglSurfaceKind,
    version: EglGlesVersion,
    owner: OwnerThreadIdentity,
    lifecycle: GlContextLifecycle,
    _not_send_sync: core::marker::PhantomData<*mut ()>,
}

impl EglGlesContext {
    /// Creates a presentable context from Host-owned native handles.
    ///
    /// # Safety
    /// The raw display and window must describe the same live native platform
    /// objects, remain valid until this context is disposed, and only be used
    /// from the calling thread.  The Host retains ownership of those objects;
    /// this method owns only EGL objects created from them.
    pub(crate) unsafe fn new_window(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        let (native_display, native_window) = native_window_pair(display, window)?;
        // SAFETY: `native_display` is validated by native_window_pair and the
        // caller upholds its native lifetime/platform contract.
        unsafe {
            Self::new_inner(
                native_display,
                EglSurfaceKind::Window { native_window },
                version,
            )
        }
    }

    /// Creates an off-screen pbuffer context for probes and WSL development.
    ///
    /// The default EGL display is intentionally used; this does not fabricate a
    /// host window and cannot present to one.
    pub(crate) fn new_pbuffer(
        size: EglPbufferSize,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        size.checked()?;
        // SAFETY: EGL_DEFAULT_DISPLAY is the specified null native-display
        // sentinel, and this provider owns all EGL objects subsequently made.
        unsafe { Self::new_inner(ptr::null_mut(), EglSurfaceKind::Pbuffer(size), version) }
    }

    unsafe fn new_inner(
        native_display: *mut c_void,
        kind: EglSurfaceKind,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        let lease = acquire_display(native_display)?;
        if let Err(error) = lease.egl.bind_api(khronos_egl::OPENGL_ES_API) {
            return Err(EglProviderError::Egl {
                operation: "eglBindAPI",
                error,
            });
        }
        let extensions = lease
            .egl
            .query_string(Some(lease.display), khronos_egl::EXTENSIONS)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglQueryString(EGL_EXTENSIONS)",
                error,
            })?
            .to_string_lossy();
        if version.minor() != 0 && !has_extension(&extensions, EGL_KHR_CREATE_CONTEXT) {
            return Err(EglProviderError::ExactVersionUnavailable(version));
        }
        let surface_type = match kind {
            EglSurfaceKind::Window { .. } => khronos_egl::WINDOW_BIT,
            EglSurfaceKind::Pbuffer(_) => khronos_egl::PBUFFER_BIT,
        };
        let config_attributes = [
            khronos_egl::SURFACE_TYPE,
            surface_type,
            khronos_egl::RENDERABLE_TYPE,
            khronos_egl::OPENGL_ES3_BIT,
            khronos_egl::RED_SIZE,
            8,
            khronos_egl::GREEN_SIZE,
            8,
            khronos_egl::BLUE_SIZE,
            8,
            khronos_egl::ALPHA_SIZE,
            8,
            khronos_egl::DEPTH_SIZE,
            24,
            khronos_egl::STENCIL_SIZE,
            8,
            khronos_egl::NONE,
        ];
        let config = match lease
            .egl
            .choose_first_config(lease.display, &config_attributes)
        {
            Ok(Some(config)) => config,
            Ok(None) => return Err(EglProviderError::ConfigUnavailable),
            Err(error) => {
                return Err(EglProviderError::Egl {
                    operation: "eglChooseConfig",
                    error,
                });
            }
        };
        let context_attributes = exact_context_attributes(version);
        let context =
            match lease
                .egl
                .create_context(lease.display, config, None, &context_attributes)
            {
                Ok(context) => context,
                Err(error) => {
                    return Err(EglProviderError::Egl {
                        operation: "eglCreateContext",
                        error,
                    });
                }
            };
        let surface = match create_surface(&lease.egl, lease.display, config, kind) {
            Ok(surface) => surface,
            Err(error) => {
                let _ = lease.egl.destroy_context(lease.display, context);
                return Err(error);
            }
        };
        if let Err(error) =
            lease
                .egl
                .make_current(lease.display, Some(surface), Some(surface), Some(context))
        {
            let _ = lease.egl.destroy_surface(lease.display, surface);
            let _ = lease.egl.destroy_context(lease.display, context);
            return Err(EglProviderError::Egl {
                operation: "eglMakeCurrent",
                error,
            });
        }
        let result = Self {
            lease,
            config,
            context,
            surface: Some(surface),
            kind,
            version,
            owner: OwnerThreadIdentity::current(),
            lifecycle: GlContextLifecycle::Active,
            _not_send_sync: core::marker::PhantomData,
        };
        if let Err(error) = result.verify_profile() {
            drop(result);
            return Err(error);
        }
        Ok(result)
    }

    /// Returns the requested GLES profile.  Actual driver strings are verified
    /// at construction, rather than inferred from EGL configuration bits.
    pub(crate) const fn profile(&self) -> GlFamilyProfile {
        self.version.profile()
    }

    /// Returns the native owner thread captured at construction.
    pub(crate) fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }

    /// Returns the lifecycle of this context/surface pair.
    pub(crate) const fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }

    /// Makes this context current on its owner thread.
    pub(crate) fn make_current(&mut self) -> Result<(), EglProviderError> {
        self.assert_owner("eglMakeCurrent")?;
        if self.lifecycle == GlContextLifecycle::Disposed {
            return Err(GlError::Disposed {
                operation: "eglMakeCurrent",
            }
            .into());
        }
        let surface = self.surface.ok_or(GlError::InvalidLifecycle {
            operation: "eglMakeCurrent",
            lifecycle: self.lifecycle,
        })?;
        self.lease
            .egl
            .make_current(
                self.lease.display,
                Some(surface),
                Some(surface),
                Some(self.context),
            )
            .map_err(|error| EglProviderError::Egl {
                operation: "eglMakeCurrent",
                error,
            })?;
        self.lifecycle = GlContextLifecycle::Active;
        Ok(())
    }

    /// Presents the current window surface.  Pbuffer contexts deliberately
    /// reject presentation instead of pretending off-screen work was shown.
    pub(crate) fn present(&mut self) -> Result<(), EglProviderError> {
        self.assert_active("eglSwapBuffers")?;
        self.make_current()?;
        if matches!(self.kind, EglSurfaceKind::Pbuffer(_)) {
            return Err(GlError::Unsupported {
                operation: "eglSwapBuffers",
                reason: "an EGL pbuffer is not presentable",
            }
            .into());
        }
        let surface = self.surface.expect("active context always owns a surface");
        self.lease
            .egl
            .swap_buffers(self.lease.display, surface)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglSwapBuffers",
                error,
            })
    }

    /// Recreates an owned pbuffer at `size`; native window surfaces resize with
    /// the Host window and therefore require no EGL-side resize operation.
    pub(crate) fn resize(&mut self, size: EglPbufferSize) -> Result<(), EglProviderError> {
        self.assert_owner("eglResize")?;
        let EglSurfaceKind::Pbuffer(_) = self.kind else {
            return Ok(());
        };
        size.checked()?;
        self.make_current()?;
        self.suspend()?;
        let old = self.surface.expect("suspend retains the EGL surface");
        self.lease
            .egl
            .destroy_surface(self.lease.display, old)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglDestroySurface",
                error,
            })?;
        self.surface = None;
        let attributes = pbuffer_attributes(size)?;
        let surface = self
            .lease
            .egl
            .create_pbuffer_surface(self.lease.display, self.config, &attributes)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglCreatePbufferSurface",
                error,
            })?;
        self.surface = Some(surface);
        self.kind = EglSurfaceKind::Pbuffer(size);
        self.resume()
    }

    /// Detaches the context while a Host surface is unavailable.
    pub(crate) fn suspend(&mut self) -> Result<(), EglProviderError> {
        self.assert_owner("eglMakeCurrent")?;
        if self.lifecycle == GlContextLifecycle::Disposed {
            return Err(GlError::Disposed {
                operation: "eglMakeCurrent",
            }
            .into());
        }
        self.detach_self_if_current()?;
        self.lifecycle = GlContextLifecycle::Suspended;
        Ok(())
    }

    /// Reattaches the owned context/surface after suspension.
    pub(crate) fn resume(&mut self) -> Result<(), EglProviderError> {
        self.make_current()
    }

    /// Loads a `glow` context after making this EGL context current.
    ///
    /// The returned context is valid only while this provider remains active on
    /// its owner thread.  Layer 1 discovery may borrow it immediately; callers
    /// must not move it to another thread or use it after suspend/dispose.
    pub(crate) fn load_glow(&mut self) -> Result<glow::Context, EglProviderError> {
        self.make_current()?;
        // SAFETY: make_current above established this exact EGL context on the
        // owner thread; EGL owns the dynamic library and proc-address loader for
        // the returned glow context's entire use interval.
        Ok(unsafe {
            glow::Context::from_loader_function(|name| {
                self.lease
                    .egl
                    .get_proc_address(name)
                    .map_or(ptr::null(), |function| {
                        function as *const () as *const c_void
                    })
            })
        })
    }

    /// Explicitly tears down EGL resources and reports the first driver error.
    pub(crate) fn dispose(&mut self) -> Result<(), EglProviderError> {
        self.assert_owner("eglDestroyContext")?;
        if self.lifecycle == GlContextLifecycle::Disposed {
            return Ok(());
        }
        self.detach_self_if_current()?;
        let mut first = None;
        if let Some(surface) = self.surface.take() {
            if let Err(error) = self.lease.egl.destroy_surface(self.lease.display, surface) {
                first = Some(EglProviderError::Egl {
                    operation: "eglDestroySurface",
                    error,
                });
            }
        }
        if let Err(error) = self
            .lease
            .egl
            .destroy_context(self.lease.display, self.context)
        {
            first.get_or_insert(EglProviderError::Egl {
                operation: "eglDestroyContext",
                error,
            });
        }
        self.lifecycle = GlContextLifecycle::Disposed;
        first.map_or(Ok(()), Err)
    }

    fn assert_owner(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner {
            Ok(())
        } else {
            Err(GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
        }
    }

    fn assert_active(&self, operation: &'static str) -> Result<(), EglProviderError> {
        self.assert_owner(operation)?;
        match self.lifecycle {
            GlContextLifecycle::Active => Ok(()),
            GlContextLifecycle::Disposed => Err(GlError::Disposed { operation }.into()),
            lifecycle => Err(GlError::InvalidLifecycle {
                operation,
                lifecycle,
            }
            .into()),
        }
    }

    /// Detaches only if this exact display/context is current.  `eglMakeCurrent`
    /// with no context would otherwise silently detach a sibling EGL context
    /// installed by another Fluxel device on the same owner thread.
    fn detach_self_if_current(&self) -> Result<(), EglProviderError> {
        let current_context = self.lease.egl.get_current_context();
        let current_display = self.lease.egl.get_current_display();
        if current_binding_is_self(
            current_context,
            current_display,
            self.context,
            self.lease.display,
        ) {
            self.lease
                .egl
                .make_current(self.lease.display, None, None, None)
                .map_err(|error| EglProviderError::Egl {
                    operation: "eglMakeCurrent",
                    error,
                })?;
        }
        Ok(())
    }

    fn verify_profile(&self) -> Result<(), EglProviderError> {
        // SAFETY: construction made this context current on its owner thread.
        let glow = unsafe {
            glow::Context::from_loader_function(|name| {
                self.lease
                    .egl
                    .get_proc_address(name)
                    .map_or(ptr::null(), |function| {
                        function as *const () as *const c_void
                    })
            })
        };
        use glow::HasContext as _;
        // SAFETY: as above; this queries only the current GLES context.
        let observed = unsafe { glow.get_parameter_string(glow::VERSION) };
        if parse_gles_version(&observed) == Some(self.version) {
            Ok(())
        } else {
            Err(EglProviderError::UnexpectedProfile {
                requested: self.version,
                observed,
            })
        }
    }
}

fn current_binding_is_self<C: Eq, D: Eq>(
    current_context: Option<C>,
    current_display: Option<D>,
    owned_context: C,
    owned_display: D,
) -> bool {
    current_context == Some(owned_context) && current_display == Some(owned_display)
}

fn acquire_display(native_display: *mut c_void) -> Result<Rc<EglDisplayLease>, EglProviderError> {
    // SAFETY: khronos-egl validates required symbol loading; the system EGL
    // library is trusted by the OS loader boundary.
    let egl = unsafe { Egl::load_required() }
        .map_err(|error| EglProviderError::Load(error.to_string()))?;
    // SAFETY: the caller has checked that this native display belongs to the
    // selected raw-handle platform and stays live for the context lease.
    let display =
        unsafe { egl.get_display(native_display) }.ok_or(EglProviderError::DisplayUnavailable)?;
    let key = display.as_ptr() as usize;
    if let Some(existing) = DISPLAY_LEASES.with(|leases| {
        let mut leases = leases.borrow_mut();
        let existing = leases.get(&key).and_then(Weak::upgrade);
        if existing.is_none() {
            leases.remove(&key);
        }
        existing
    }) {
        return Ok(existing);
    }
    egl.initialize(display)
        .map_err(|error| EglProviderError::Egl {
            operation: "eglInitialize",
            error,
        })?;
    let lease = Rc::new(EglDisplayLease { egl, display });
    DISPLAY_LEASES.with(|leases| {
        leases.borrow_mut().insert(key, Rc::downgrade(&lease));
    });
    Ok(lease)
}

impl Drop for EglGlesContext {
    fn drop(&mut self) {
        if self.lifecycle != GlContextLifecycle::Disposed
            && OwnerThreadIdentity::current() == self.owner
        {
            let _ = self.dispose();
        }
    }
}

fn native_window_pair(
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<(*mut c_void, *mut c_void), EglProviderError> {
    match (display, window) {
        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let native_display = display
                .display
                .ok_or(EglProviderError::InvalidNativeHandle(
                    "Xlib display pointer",
                ))?;
            if window.window == 0 {
                return Err(EglProviderError::InvalidNativeHandle("Xlib window"));
            }
            Ok((
                native_display.as_ptr(),
                window.window as usize as *mut c_void,
            ))
        }
        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
            Ok((display.display.as_ptr(), window.surface.as_ptr()))
        }
        (display, window) => Err(EglProviderError::UnsupportedHandlePair {
            display: display_name(display),
            window: window_name(window),
        }),
    }
}

fn create_surface(
    egl: &Egl,
    display: khronos_egl::Display,
    config: khronos_egl::Config,
    kind: EglSurfaceKind,
) -> Result<khronos_egl::Surface, EglProviderError> {
    match kind {
        EglSurfaceKind::Window { native_window } => {
            // SAFETY: native_window is paired with display by new_window's
            // explicit unsafe contract and stays Host-owned/live.
            unsafe { egl.create_window_surface(display, config, native_window, None) }.map_err(
                |error| EglProviderError::Egl {
                    operation: "eglCreateWindowSurface",
                    error,
                },
            )
        }
        EglSurfaceKind::Pbuffer(size) => {
            let attributes = pbuffer_attributes(size)?;
            egl.create_pbuffer_surface(display, config, &attributes)
                .map_err(|error| EglProviderError::Egl {
                    operation: "eglCreatePbufferSurface",
                    error,
                })
        }
    }
}

fn exact_context_attributes(version: EglGlesVersion) -> Vec<i32> {
    match version {
        // EGL 1.0's ES client-version attribute is enough for the whole 3.0
        // profile.  Do not pass KHR attributes unless their extension was
        // checked, because several otherwise-valid EGL 1.4 implementations
        // reject unknown attributes.
        EglGlesVersion::V3_0 => vec![khronos_egl::CONTEXT_CLIENT_VERSION, 3, khronos_egl::NONE],
        EglGlesVersion::V3_1 | EglGlesVersion::V3_2 => vec![
            EGL_CONTEXT_MAJOR_VERSION_KHR,
            3,
            EGL_CONTEXT_MINOR_VERSION_KHR,
            version.minor(),
            khronos_egl::NONE,
        ],
    }
}

fn parse_gles_version(value: &str) -> Option<EglGlesVersion> {
    let suffix = value.strip_prefix("OpenGL ES ")?;
    let version = suffix.split_ascii_whitespace().next()?;
    match version {
        "3.0" => Some(EglGlesVersion::V3_0),
        "3.1" => Some(EglGlesVersion::V3_1),
        "3.2" => Some(EglGlesVersion::V3_2),
        _ => None,
    }
}

fn pbuffer_attributes(size: EglPbufferSize) -> Result<[i32; 5], EglProviderError> {
    let (width, height) = size.checked()?;
    Ok([
        khronos_egl::WIDTH,
        width,
        khronos_egl::HEIGHT,
        height,
        khronos_egl::NONE,
    ])
}

fn has_extension(extensions: &str, extension: &str) -> bool {
    extensions
        .split_ascii_whitespace()
        .any(|item| item == extension)
}

fn display_name(handle: RawDisplayHandle) -> &'static str {
    match handle {
        RawDisplayHandle::Xlib(_) => "Xlib",
        RawDisplayHandle::Xcb(_) => "Xcb",
        RawDisplayHandle::Wayland(_) => "Wayland",
        RawDisplayHandle::Drm(_) => "DRM",
        RawDisplayHandle::Gbm(_) => "GBM",
        RawDisplayHandle::Windows(_) => "Windows",
        RawDisplayHandle::Android(_) => "Android",
        _ => "other",
    }
}

fn window_name(handle: RawWindowHandle) -> &'static str {
    match handle {
        RawWindowHandle::Xlib(_) => "Xlib",
        RawWindowHandle::Xcb(_) => "Xcb",
        RawWindowHandle::Wayland(_) => "Wayland",
        RawWindowHandle::Drm(_) => "DRM",
        RawWindowHandle::Gbm(_) => "GBM",
        RawWindowHandle::Win32(_) => "Win32",
        RawWindowHandle::AndroidNdk(_) => "AndroidNdk",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EglGlesVersion, EglPbufferSize, current_binding_is_self, exact_context_attributes,
        has_extension, native_window_pair, parse_gles_version, pbuffer_attributes,
    };
    use raw_window_handle::{
        RawDisplayHandle, RawWindowHandle, XlibDisplayHandle, XlibWindowHandle,
    };

    #[test]
    fn exact_gles_attributes_preserve_minor_version() {
        assert_eq!(
            exact_context_attributes(EglGlesVersion::V3_0),
            [0x3098, 3, 0x3038]
        );
        assert_eq!(
            exact_context_attributes(EglGlesVersion::V3_1),
            [0x3098, 3, 0x30FB, 1, 0x3038]
        );
        assert_eq!(
            exact_context_attributes(EglGlesVersion::V3_2),
            [0x3098, 3, 0x30FB, 2, 0x3038]
        );
    }

    #[test]
    fn extension_match_is_token_not_substring() {
        assert!(has_extension(
            "EGL_KHR_create_context EGL_EXT_x",
            "EGL_KHR_create_context"
        ));
        assert!(!has_extension(
            "EGL_KHR_create_context_extra",
            "EGL_KHR_create_context"
        ));
    }

    #[test]
    fn pbuffer_dimensions_are_checked_before_egl() {
        assert_eq!(
            pbuffer_attributes(EglPbufferSize {
                width: 3,
                height: 5
            })
            .unwrap(),
            [0x3057, 3, 0x3056, 5, 0x3038]
        );
        assert!(
            pbuffer_attributes(EglPbufferSize {
                width: 0,
                height: 5
            })
            .is_err()
        );
    }

    #[test]
    fn profile_verification_parses_the_complete_reported_version() {
        assert_eq!(
            parse_gles_version("OpenGL ES 3.2 Mesa 25"),
            Some(EglGlesVersion::V3_2)
        );
        assert_eq!(
            parse_gles_version("OpenGL ES 3.1"),
            Some(EglGlesVersion::V3_1)
        );
        assert_eq!(parse_gles_version("OpenGL ES 3.20"), None);
        assert_eq!(parse_gles_version("4.6"), None);
    }

    #[test]
    fn a_context_never_detaches_a_sibling_current_binding() {
        assert!(current_binding_is_self(
            Some("ours"),
            Some("display"),
            "ours",
            "display"
        ));
        assert!(!current_binding_is_self(
            Some("sibling"),
            Some("display"),
            "ours",
            "display"
        ));
        assert!(!current_binding_is_self(
            Some("ours"),
            Some("other-display"),
            "ours",
            "display"
        ));
        assert!(!current_binding_is_self::<&str, &str>(
            None, None, "ours", "display"
        ));
    }

    #[test]
    fn incomplete_xlib_handles_fail_before_the_unsafe_egl_boundary() {
        let display = RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0));
        let valid_window = RawWindowHandle::Xlib(XlibWindowHandle::new(1));
        assert!(native_window_pair(display, valid_window).is_err());

        let display = RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0));
        let empty_window = RawWindowHandle::Xlib(XlibWindowHandle::new(0));
        assert!(native_window_pair(display, empty_window).is_err());
    }
}
