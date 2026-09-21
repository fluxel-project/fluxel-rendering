//! EGL-owned GLES contexts for the native GL-family provider.
//!
//! The Host supplies raw display/window handles and keeps the underlying
//! native objects alive.  This provider owns every EGL object created from
//! those handles: display initialization, config, context, surface, current
//! binding, presentation, and teardown.  It deliberately does not create a
//! window, event loop, or an ANGLE/DX translation provider.
//!
//! This currently supports Linux Xlib/Wayland windows, Android NDK windows,
//! and headless pbuffers. Android maps its display-less handle to
//! `EGL_DEFAULT_DISPLAY`, the documented EGL route; other platform-display
//! extension ABIs are not guessed.
//!
//! It does not own Fluxel's GL object tables, the render pass, or the viewport
//! a pass renders with, and it never resizes the Host's window. The drawable
//! extent it observes is bounded by the context's recorded viewport limit; see
//! `presentation.rs` for why applying a viewport is not this layer's job.
//!
//! `presentation.rs` holds the family-owner and surface-presentation facets and
//! `tests/mod.rs` holds the pure-logic tests; this file is context and surface
//! construction, discovery wiring, and teardown.

use core::ffi::c_void;
use core::ptr;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use raw_window_handle::{
    AndroidDisplayHandle, AndroidNdkWindowHandle, RawDisplayHandle, RawWindowHandle,
};

use super::discovery::discover_current_glow_identified;

use crate::backend::gl::api::{
    ContextStamp, GlContextLifecycle, GlDiscoverySnapshot, GlError, GlFamilyProfile,
    OwnerThreadIdentity,
};

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

    /// Whether this observed profile satisfies a fixture's minimum request.
    ///
    /// EGL's ES client-version selection is not an exact-version contract on
    /// every implementation.  Fluxel's Android evidence matrix therefore
    /// treats its requested profile as a floor: ES 3.1 satisfies an ES 3.0
    /// case, while ES 3.0 must never satisfy an ES 3.1 case.
    const fn satisfies(self, requested: Self) -> bool {
        self.minor() >= requested.minor()
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

/// Sendable description of a Host-owned Android NDK window.
///
/// The pointer is converted back into `RawWindowHandle::AndroidNdk` only on
/// the EGL owner thread. It never enters the public RHI and the Host must keep
/// the `ANativeWindow` alive until the returned provider is dropped.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AndroidEglWorkerDescriptor {
    stamp: ContextStamp,
    native_window: usize,
}

impl AndroidEglWorkerDescriptor {
    /// Narrows a Host Android raw window handle into a worker-safe scalar.
    pub(crate) fn new(
        stamp: ContextStamp,
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<Self, EglProviderError> {
        let (native_display, native_window) = native_window_pair(display, window)?;
        if !native_display.is_null() {
            return Err(EglProviderError::UnsupportedHandlePair {
                display: "non-Android display",
                window: "Android NDK window",
            });
        }
        Ok(Self {
            stamp,
            native_window: native_window as usize,
        })
    }
}

/// Creates an Android GLES provider entirely on its EGL owner thread.
///
/// Context construction and capability discovery occur on the same thread that
/// retains the `EGLContext`; only immutable facts and a diagnostic name return
/// to the caller. Presentation is intentionally not advertised until the
/// native owner installs the configure/acquire/present forwarding route.
pub(crate) fn spawn_android_window_provider(
    instance: crate::api::identity::DeviceInstanceId,
    descriptor: AndroidEglWorkerDescriptor,
) -> crate::api::error::RhiResult<crate::backend::gl::platform::GlProvider> {
    let (worker, (facts, name)) = super::NativeOwnerWorker::spawn_with_info(move || {
        let pointer = core::ptr::NonNull::new(descriptor.native_window as *mut c_void)
            .ok_or_else(|| "Android ANativeWindow pointer was null".to_owned())?;
        let display = RawDisplayHandle::Android(AndroidDisplayHandle::new());
        let window = RawWindowHandle::AndroidNdk(AndroidNdkWindowHandle::new(pointer));
        let mut last_error = None;
        for version in [
            EglGlesVersion::V3_2,
            EglGlesVersion::V3_1,
            EglGlesVersion::V3_0,
        ] {
            // SAFETY: the descriptor was validated from this Host-owned NDK
            // handle, and its lifetime is the explicit factory contract.
            match unsafe { EglGlesContext::new_window(descriptor.stamp, display, window, version) }
            {
                Ok(surface) => {
                    let facts = surface.v13_capability_facts();
                    let name = format!("OpenGL ES ({})", surface.snapshot.context().renderer());
                    let provider = surface.into_owned_provider().map_err(|error| {
                        format!("Android EGL provider creation failed for {version:?}: {error:?}")
                    })?;
                    return Ok((provider, (facts, name)));
                }
                Err(error) => last_error = Some(error),
            }
        }
        Err(format!(
            "Android EGL could not create any GLES 3.x window context: {:?}",
            last_error
        ))
    })
    .map_err(|error| {
        crate::api::error::RhiError::new(
            crate::api::error::RhiErrorKind::BackendFailure,
            format!("Android EGL owner worker could not start: {error:?}"),
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

/// Creates a self-contained, non-presentable GLES v13 provider on its EGL
/// owner worker.
///
/// This is useful for headless/probe routes.  It deliberately does not claim
/// window presentation: an EGL pbuffer cannot be swapped to a Host display.
/// Window-surface creation remains a separate Host integration because the
/// native window must remain alive for the provider's whole lifetime.
pub(crate) fn spawn_pbuffer_provider(
    instance: crate::api::identity::DeviceInstanceId,
    stamp: ContextStamp,
    size: EglPbufferSize,
) -> crate::api::error::RhiResult<crate::backend::gl::platform::GlProvider> {
    let (worker, (facts, name)) = super::NativeOwnerWorker::spawn_with_info(move || {
        let mut last_error = None;
        for version in [
            EglGlesVersion::V3_2,
            EglGlesVersion::V3_1,
            EglGlesVersion::V3_0,
        ] {
            match EglGlesContext::new_pbuffer(stamp, size, version) {
                Ok(surface) => {
                    let facts = surface.v13_capability_facts();
                    let name = format!("OpenGL ES ({})", surface.snapshot.context().renderer());
                    let provider = surface.into_owned_provider().map_err(|error| {
                        format!("EGL provider creation failed for {version:?}: {error:?}")
                    })?;
                    return Ok((provider, (facts, name)));
                }
                Err(error) => last_error = Some(error),
            }
        }
        Err(format!(
            "EGL could not create any GLES 3.x pbuffer context: {:?}",
            last_error
        ))
    })
    .map_err(|error| {
        crate::api::error::RhiError::new(
            crate::api::error::RhiErrorKind::BackendFailure,
            format!("EGL owner worker could not start: {error:?}"),
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
    /// The current GLES context was below the version requested by Fluxel.
    ///
    /// A newer observed profile is valid: Android's EGL selection rules permit
    /// a request for an ES 3.x floor to yield a newer ES 3.x context.
    BelowRequestedProfile {
        requested: EglGlesVersion,
        observed: String,
    },
    /// A surface extent this context cannot map to a viewport was observed.
    ///
    /// The extent is a real observation here -- EGL answers the live surface's
    /// dimensions, unlike WGL where the Host's window size is the only source.
    /// The maximum viewport dimensions that bound it come from the same
    /// discovery snapshot, and a surface above them could only be rejected later
    /// by an unrelated pass.
    DrawableExtentExceedsViewportLimit {
        requested: [u32; 2],
        limit: [u32; 2],
    },
    /// The provider was used from a thread other than its owner.
    Gl(GlError),
}

impl From<GlError> for EglProviderError {
    fn from(value: GlError) -> Self {
        Self::Gl(value)
    }
}

impl EglProviderError {
    /// Flattens back into the stable GL error vocabulary the presentation
    /// domain reports, keeping provider diagnostics in the message.
    fn into_gl_error(self) -> GlError {
        match self {
            Self::Gl(error) => error,
            Self::Egl { operation, error } => GlError::Driver {
                operation,
                message: format!("EGL error: {error:?}"),
            },
            // Several presentation entry points observe an extent, and this
            // flattening point is below all of them, so the label names the
            // fact that failed rather than guessing which caller found it.
            Self::DrawableExtentExceedsViewportLimit { requested, limit } => GlError::Validation {
                operation: "drawable-extent",
                message: format!(
                    "drawable extent {requested:?} exceeds the maximum viewport dimensions {limit:?}"
                ),
            },
            other => GlError::Driver {
                operation: "egl-provider",
                message: format!("EGL provider error: {other:?}"),
            },
        }
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
    stamp: ContextStamp,
    snapshot: GlDiscoverySnapshot,
    owner: OwnerThreadIdentity,
    lifecycle: GlContextLifecycle,
    _not_send_sync: core::marker::PhantomData<*mut ()>,
}

#[cfg(feature = "native-gles-egl")]
impl super::driver::NativePlatformContext for EglGlesContext {
    fn make_current(&mut self, operation: &'static str) -> crate::api::error::RhiResult<()> {
        self.with_current(operation, |_| Ok(())).map_err(|error| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::DeviceLost,
                format!("{operation}: {error:?}"),
            )
        })
    }

    fn supports_presentation(&self) -> bool {
        matches!(self.kind, EglSurfaceKind::Window { .. })
    }

    fn drawable_extent(
        &mut self,
    ) -> crate::api::error::RhiResult<Option<crate::api::presentation::Extent2d>> {
        self.make_current().map_err(|error| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::BackendFailure,
                format!("EGL drawable extent query failed: {error:?}"),
            )
        })?;
        let Some(surface) = self.surface else {
            return Ok(None);
        };
        let width = self
            .lease
            .egl
            .query_surface(self.lease.display, surface, khronos_egl::WIDTH)
            .map_err(|error| {
                crate::api::error::RhiError::new(
                    crate::api::error::RhiErrorKind::BackendFailure,
                    format!("eglQuerySurface(EGL_WIDTH) failed: {error:?}"),
                )
            })?;
        let height = self
            .lease
            .egl
            .query_surface(self.lease.display, surface, khronos_egl::HEIGHT)
            .map_err(|error| {
                crate::api::error::RhiError::new(
                    crate::api::error::RhiErrorKind::BackendFailure,
                    format!("eglQuerySurface(EGL_HEIGHT) failed: {error:?}"),
                )
            })?;
        let (Ok(width), Ok(height)) = (u32::try_from(width), u32::try_from(height)) else {
            return Ok(None);
        };
        Ok((width != 0 && height != 0)
            .then_some(crate::api::presentation::Extent2d { width, height }))
    }

    fn present(&mut self) -> crate::api::error::RhiResult<()> {
        EglGlesContext::present(self).map_err(|error| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::BackendFailure,
                format!("EGL presentation failed: {error:?}"),
            )
        })
    }
}

/// Immutable EGL/GLES discovery projection for fixture-only evidence. Native
/// EGL handles and the `glow` table never leave this module.
#[derive(Clone, Debug)]
pub(crate) struct EglEvidence {
    pub(crate) profile: String,
    /// The GLES floor passed to EGL when this context was created.
    pub(crate) requested_version: String,
    /// The GL version string observed from the current context.
    pub(crate) observed_version: String,
    pub(crate) version: String,
    pub(crate) shading_language_version: String,
    pub(crate) vendor: String,
    pub(crate) renderer: String,
    pub(crate) driver_or_browser: String,
    pub(crate) extensions: Vec<String>,
    pub(crate) capabilities: Vec<(String, bool)>,
    pub(crate) limits: Vec<(String, String)>,
    pub(crate) extent: [u32; 2],
}

impl EglGlesContext {
    /// Consumes the EGL context into a worker-owned v13 provider.  EGL handles
    /// stay in `self`; the freshly loaded glow table is moved into the provider.
    pub(crate) fn into_owned_provider(
        mut self,
    ) -> Result<super::driver::NativeOwnedProvider<Self>, EglProviderError> {
        let dispatch = self.load_glow()?;
        // SAFETY: `load_glow` establishes this owned EGL context as current on
        // its owner thread, and the table is moved into the provider.
        let mut provider = unsafe {
            super::provider::NativeGlProvider::from_discovered(dispatch, self.snapshot.clone())
        };
        // A pbuffer has an EGL-owned, exact extent.  A window surface does not:
        // EGL exposes no portable query which would let this layer substitute a
        // guessed Host drawable size.  Leave the latter unconfigured until the
        // host/presentation route reports it.
        if let Some([width, height]) = self.known_drawable_extent() {
            provider.surface_extent =
                Some(crate::backend::gl::api::GlSurfaceSize { width, height });
            provider.surface_suspended = width == 0 || height == 0;
        }
        Ok(super::driver::NativeOwnedProvider::new(self, provider))
    }
    /// Freezes the v13 facts for this exact EGL context generation.
    ///
    /// This is the same discovery-to-facts narrowing WGL uses; EGL extension
    /// strings never become public support on their own.
    pub(crate) fn v13_capability_facts(&self) -> crate::api::capability::CapabilityFacts {
        super::discovery::v13_capability_snapshot(&self.snapshot).into_facts()
    }

    /// Returns an exact drawable extent only where EGL owns that fact.
    ///
    /// Window dimensions stay Host-owned.  Returning `None` there is
    /// intentional: handing the executor a made-up extent would permit an
    /// acquired-frame lease for a default framebuffer with different dimensions.
    fn known_drawable_extent(&self) -> Option<[u32; 2]> {
        match self.kind {
            EglSurfaceKind::Pbuffer(size) => Some([size.width, size.height]),
            EglSurfaceKind::Window { .. } => None,
        }
    }
    /// Adopts an already worker-owned route for this exact EGL generation.
    ///
    /// The owner is required to make this context current synchronously; EGL
    /// display/context objects remain private to the platform adapter.
    pub(crate) fn adopt(
        &self,
        instance: crate::api::identity::DeviceInstanceId,
        owner: std::sync::Arc<dyn super::driver::NativeGlOwner>,
    ) -> crate::api::error::RhiResult<crate::backend::gl::platform::GlProvider> {
        super::adopt_discovered_owner(
            instance,
            format!("OpenGL ES ({})", self.snapshot.context().renderer()),
            &self.snapshot,
            owner,
        )
    }
    /// Creates a presentable context from Host-owned native handles.
    ///
    /// # Safety
    /// The raw display and window must describe the same live native platform
    /// objects, remain valid until this context is disposed, and only be used
    /// from the calling thread.  The Host retains ownership of those objects;
    /// this method owns only EGL objects created from them.
    pub(crate) unsafe fn new_window(
        stamp: ContextStamp,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        let (native_display, native_window) = native_window_pair(display, window)?;
        // SAFETY: `native_display` is validated by native_window_pair and the
        // caller upholds its native lifetime/platform contract.
        unsafe {
            Self::new_inner(
                stamp,
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
        stamp: ContextStamp,
        size: EglPbufferSize,
        version: EglGlesVersion,
    ) -> Result<Self, EglProviderError> {
        size.checked()?;
        // SAFETY: EGL_DEFAULT_DISPLAY is the specified null native-display
        // sentinel, and this provider owns all EGL objects subsequently made.
        unsafe {
            Self::new_inner(
                stamp,
                ptr::null_mut(),
                EglSurfaceKind::Pbuffer(size),
                version,
            )
        }
    }

    unsafe fn new_inner(
        stamp: ContextStamp,
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
        // EGL_CONTEXT_CLIENT_VERSION alone only selects the ES major family.
        // On this Android EGL implementation that means a request for ES 3.0
        // is legally upgraded to 3.1.  When EGL_KHR_create_context is present,
        // include an explicit minor value even for zero so the fixture can
        // pass the requested floor through to EGL.  The evidence path verifies
        // the observed profile is at least that floor, because EGL may legally
        // return a newer ES 3.x context.
        let context_attributes =
            exact_context_attributes(version, has_extension(&extensions, EGL_KHR_CREATE_CONTEXT));
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
        // Verification and discovery observe the exact current context before
        // construction completes; any failure drops the transactional result
        // and rolls the EGL objects back.  That rollback is why the remaining
        // steps run in one fallible scope rather than as bare `?` returns: from
        // here on the context is already current and the pair has no owner yet,
        // so an `Err` that returned without destroying them would leak both for
        // the life of the display and leave a context nothing can name
        // installed on the owner thread.
        let egl_loader = |name: &str| {
            lease
                .egl
                .get_proc_address(name)
                .map_or(ptr::null(), |function| {
                    function as *const () as *const c_void
                })
        };
        let prepared = (|| -> Result<GlDiscoverySnapshot, EglProviderError> {
            // SAFETY: make_current above established this exact EGL context on
            // the owner thread and EGL owns the loader for glow's use.
            let glow = unsafe { glow::Context::from_loader_function(egl_loader) };
            use glow::HasContext as _;
            // SAFETY: as above; this queries only the current GLES context.
            let observed = unsafe { glow.get_parameter_string(glow::VERSION) };
            let observed_profile = parse_gles_version(&observed);
            if !observed_profile.is_some_and(|profile| profile.satisfies(version)) {
                return Err(EglProviderError::BelowRequestedProfile {
                    requested: version,
                    observed,
                });
            }
            // The EGL display's own identity strings exist only here, so they
            // are read while this display is initialized and this exact context
            // is current (audit P2-6).
            let driver_identity = egl_driver_identity(&lease.egl, lease.display);
            // SAFETY: the current context serves every discovery query and probe.
            let snapshot = unsafe {
                discover_current_glow_identified(&glow, stamp, egl_loader, &driver_identity)
            }
            .map_err(|error| {
                EglProviderError::Gl(GlError::Driver {
                    operation: "discover EGL context",
                    message: format!("native GL discovery failed: {error:?}"),
                })
            })?;
            // A pbuffer's extent is fixed by this construction, so it is
            // bounded before the context is handed back; window surfaces report
            // their live extent from EGL at each acquisition and are bounded
            // there.
            if let EglSurfaceKind::Pbuffer(size) = kind {
                check_drawable_extent(
                    snapshot.limits().max_viewport_dimensions,
                    [size.width, size.height],
                )?;
            }
            Ok(snapshot)
        })();
        let snapshot = match prepared {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let _ = lease.egl.destroy_surface(lease.display, surface);
                let _ = lease.egl.destroy_context(lease.display, context);
                return Err(error);
            }
        };
        let result = Self {
            lease,
            config,
            context,
            surface: Some(surface),
            kind,
            version,
            stamp,
            snapshot,
            owner: OwnerThreadIdentity::current(),
            lifecycle: GlContextLifecycle::Active,
            _not_send_sync: core::marker::PhantomData,
        };
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
        self.refuse_unusable("eglMakeCurrent")?;
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
    ///
    /// The new extent is bounded before EGL is touched at all, so a size this
    /// context cannot map leaves the existing surface intact rather than
    /// destroying it and then failing.
    pub(crate) fn resize(&mut self, size: EglPbufferSize) -> Result<(), EglProviderError> {
        self.assert_owner("eglResize")?;
        let EglSurfaceKind::Pbuffer(_) = self.kind else {
            return Ok(());
        };
        size.checked()?;
        check_drawable_extent(
            self.snapshot.limits().max_viewport_dimensions,
            [size.width, size.height],
        )?;
        self.make_current()?;
        self.suspend()?;
        let old = self.surface.expect("suspend retains the EGL surface");
        let attributes = pbuffer_attributes(size)?;
        let surface = self
            .lease
            .egl
            .create_pbuffer_surface(self.lease.display, self.config, &attributes)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglCreatePbufferSurface",
                error,
            })?;
        // The new surface is adopted before the old one is released, so a
        // refused `eglCreatePbufferSurface` -- the plausible failure here --
        // leaves the context holding the surface it already had.  The earlier
        // order destroyed first and left `surface: None`, which has no recovery
        // path short of rebuilding the provider: one driver refusal and the
        // context was permanently unusable rather than merely unresized.  The
        // old surface is not current (`suspend` above detached it), so deleting
        // it after the swap is legal.
        self.surface = Some(surface);
        self.kind = EglSurfaceKind::Pbuffer(size);
        let released = self.lease.egl.destroy_surface(self.lease.display, old);
        self.resume()?;
        released.map_err(|error| EglProviderError::Egl {
            operation: "eglDestroySurface",
            error,
        })
    }

    /// Detaches the context while a Host surface is unavailable.
    pub(crate) fn suspend(&mut self) -> Result<(), EglProviderError> {
        self.assert_owner("eglMakeCurrent")?;
        // Refused for the same reason `make_current` is, and it is not a
        // technicality: suspending would rewrite `Lost` to `Suspended`, and the
        // next resume would then reach `Active` through the door this closes.
        self.refuse_unusable("eglMakeCurrent")?;
        self.detach_self_if_current()?;
        self.lifecycle = GlContextLifecycle::Suspended;
        Ok(())
    }

    /// Refuses a state change that would revive a context EGL reported lost.
    ///
    /// Loss is durable in this provider -- [`Self::context_restored`] refuses
    /// outright, because only the Host can make a new one -- and the WGL
    /// provider refuses the identical sequence, so this is the EGL provider
    /// agreeing with its own trait default (`assert_ready` answers
    /// `ContextLost` for the same state) rather than a rule of its own.  Before
    /// this, `resume_surface` reached `Active` from `Lost` and handed out a
    /// fresh acquire lease from a context that had recorded its own death.
    fn refuse_unusable(&self, operation: &'static str) -> Result<(), EglProviderError> {
        match self.lifecycle {
            GlContextLifecycle::Lost => Err(GlError::ContextLost { operation }.into()),
            GlContextLifecycle::Disposed => Err(GlError::Disposed { operation }.into()),
            _ => Ok(()),
        }
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

    /// Returns the discovery evidence gathered when this context opened.
    ///
    /// Discovery ran once against the exact current context during construction,
    /// so this needs no repeat currentness and never re-queries -- the same
    /// contract, and for the same reason, as the WGL surface's `discover`.
    pub(crate) fn discover(&self) -> Result<&GlDiscoverySnapshot, GlError> {
        self.assert_owner("discover EGL context")?;
        if self.lifecycle == GlContextLifecycle::Disposed {
            return Err(GlError::Disposed {
                operation: "discover EGL context",
            });
        }
        Ok(&self.snapshot)
    }

    /// Copies the discovery snapshot into a handle-free fixture report.
    pub(crate) fn evidence(&self) -> EglEvidence {
        use crate::backend::gl::api::GlCapability;
        let context = self.snapshot.context();
        let limits = self.snapshot.limits();
        let mut extensions: Vec<_> = self
            .snapshot
            .extensions()
            .raw_reported_names()
            .map(str::to_owned)
            .collect();
        extensions.sort();
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
        let extent = match self.kind {
            EglSurfaceKind::Pbuffer(size) => [size.width, size.height],
            EglSurfaceKind::Window { .. } => [0, 0],
        };
        EglEvidence {
            profile: format!("{:?}", context.profile()),
            requested_version: format!("3.{}", self.version.minor()),
            observed_version: context.version().to_owned(),
            version: context.version().to_owned(),
            shading_language_version: context.shading_language_version().to_owned(),
            vendor: context.vendor().to_owned(),
            renderer: context.renderer().to_owned(),
            driver_or_browser: context.driver_or_browser().to_owned(),
            extensions,
            capabilities,
            limits: vec![
                (
                    "max-texture-size".into(),
                    limits.max_texture_size.to_string(),
                ),
                (
                    "max-renderbuffer-size".into(),
                    limits.max_renderbuffer_size.to_string(),
                ),
                (
                    "max-vertex-attributes".into(),
                    limits.max_vertex_attributes.to_string(),
                ),
                (
                    "max-viewport".into(),
                    format!(
                        "{}x{}",
                        limits.max_viewport_dimensions[0], limits.max_viewport_dimensions[1]
                    ),
                ),
                ("max-samples".into(), limits.max_samples.to_string()),
            ],
            extent,
        }
    }

    /// Runs one operation while this context is current, lending it `glow`.
    ///
    /// The counterpart of the WGL surface's `with_current`, with two differences
    /// that are facts about this provider rather than choices.
    ///
    /// It builds a `glow` context per call instead of lending out one loaded at
    /// construction, because [`Self::load_glow`] is this provider's only loading
    /// path and it needs `&mut self`.  The cost is resolving the driver's proc
    /// addresses again on each call, which a one-shot measurement does not
    /// notice; buying it back would mean caching a `glow::Context` in a struct
    /// that currently has no field for one, for a saving nothing has asked for.
    ///
    /// And it takes `&mut self`, which is the honest signature for the same
    /// reason: two of these cannot run at once over one EGL context, and the
    /// borrow checker is a better guarantee of that than a comment would be.
    pub(crate) fn with_current<T>(
        &mut self,
        operation: &'static str,
        callback: impl FnOnce(&glow::Context) -> Result<T, GlError>,
    ) -> Result<T, EglProviderError> {
        self.assert_owner(operation)?;
        let glow = self.load_glow()?;
        callback(&glow).map_err(EglProviderError::Gl)
    }

    /// Executes a real ES 3 triangle workload on the owned pbuffer and
    /// optionally returns unmodified bottom-to-top RGBA bytes from it.
    pub(crate) fn draw_evidence(
        &mut self,
        draws: u32,
        read_colour: bool,
    ) -> Result<Option<Vec<u8>>, EglProviderError> {
        let [width, height] = match self.kind {
            EglSurfaceKind::Pbuffer(size) => [size.width, size.height],
            EglSurfaceKind::Window { .. } => {
                return Err(GlError::Unsupported {
                    operation: "EGL evidence draw",
                    reason: "fixture evidence is defined only for owned pbuffers",
                }
                .into());
            }
        };
        if width == 0 || height == 0 || draws == 0 {
            return Err(GlError::Validation {
                operation: "EGL evidence draw",
                message: "evidence draw requires a nonzero extent and draw count".into(),
            }
            .into());
        }
        self.with_current("EGL evidence draw", |gl| {
            use glow::HasContext as _;
            // SAFETY: with_current has made this owned EGL context current for
            // the complete callback; all objects are deleted before it exits.
            unsafe {
                let vertex = compile_evidence_shader(gl, glow::VERTEX_SHADER, EVIDENCE_VERTEX)?;
                let fragment =
                    match compile_evidence_shader(gl, glow::FRAGMENT_SHADER, EVIDENCE_FRAGMENT) {
                        Ok(shader) => shader,
                        Err(error) => {
                            gl.delete_shader(vertex);
                            return Err(error);
                        }
                    };
                let program = gl.create_program().map_err(|message| GlError::Driver {
                    operation: "EGL evidence draw",
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
                        operation: "EGL evidence draw",
                        message,
                    });
                }
                let vao = gl
                    .create_vertex_array()
                    .map_err(|message| GlError::Driver {
                        operation: "EGL evidence draw",
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
                        operation: "EGL evidence draw",
                        message:
                            "OpenGL ES reported an error while executing the evidence workload"
                                .into(),
                    });
                }
                Ok(colour)
            }
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
}

// This is the GLES spelling of the exact 4x4 picture asserted by the shared
// WGL/GLES evidence checker. See the matching WGL constants for the geometry.
const EVIDENCE_VERTEX: &str = "#version 300 es\nprecision highp float; void main() { const vec2 p[3]=vec2[3](vec2(-0.6,-0.5),vec2(0.6,-0.5),vec2(0.0,0.0)); gl_Position=vec4(p[gl_VertexID],0.0,1.0); }";
const EVIDENCE_FRAGMENT: &str = "#version 300 es\nprecision highp float; out vec4 o; void main() { o=vec4(48.0/255.0,176.0/255.0,112.0/255.0,1.0); }";

fn compile_evidence_shader(
    gl: &glow::Context,
    kind: u32,
    source: &str,
) -> Result<glow::NativeShader, GlError> {
    use glow::HasContext as _;
    // SAFETY: caller owns current EGL context for this helper.
    unsafe {
        let shader = gl.create_shader(kind).map_err(|message| GlError::Driver {
            operation: "EGL evidence shader",
            message,
        })?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if gl.get_shader_compile_status(shader) {
            Ok(shader)
        } else {
            let message = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            Err(GlError::Driver {
                operation: "EGL evidence shader",
                message,
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
        // Android's raw display intentionally carries no native pointer. EGL's
        // documented route is EGL_DEFAULT_DISPLAY plus the ANativeWindow used
        // by eglCreateWindowSurface; do not invent an EGL platform extension
        // call for it.
        (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(window)) => {
            Ok((ptr::null_mut(), window.a_native_window.as_ptr()))
        }
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

fn exact_context_attributes(version: EglGlesVersion, has_create_context: bool) -> Vec<i32> {
    match version {
        // EGL 1.0's ES client-version attribute is enough for the whole 3.0
        // profile.  Do not pass KHR attributes unless their extension was
        // checked, because several otherwise-valid EGL 1.4 implementations
        // reject unknown attributes.
        EglGlesVersion::V3_0 if has_create_context => vec![
            EGL_CONTEXT_MAJOR_VERSION_KHR,
            3,
            EGL_CONTEXT_MINOR_VERSION_KHR,
            0,
            khronos_egl::NONE,
        ],
        EglGlesVersion::V3_0 => {
            vec![khronos_egl::CONTEXT_CLIENT_VERSION, 3, khronos_egl::NONE]
        }
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

/// The EGL display's own vendor and version strings.
///
/// The driver identity is the one fact discovery cannot read from GL: on this
/// family it belongs to the platform layer that owns the display, and EGL is
/// the only layer that publishes it (audit P2-6).  The GL vendor/renderer pair
/// is already recorded as its own field, and the GL version string is recorded
/// as `version`, so repeating either here would reproduce exactly the defect
/// this wiring closes; what is read is the EGL implementation's own two
/// strings, queried from the display this provider initialized.
///
/// A failed query or a blank answer yields an empty string, which discovery
/// records as an unavailable identity instead of as a driver nobody observed.
fn egl_driver_identity(egl: &Egl, display: khronos_egl::Display) -> String {
    let read = |name: khronos_egl::Int| {
        egl.query_string(Some(display), name)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    compose_egl_driver_identity(&read(khronos_egl::VENDOR), &read(khronos_egl::VERSION))
}

/// Composes the platform identity from the EGL implementation's two strings.
///
/// `EGL` names the layer the version belongs to, so the recorded identity
/// cannot read as a GL version even when the numbers coincide.
fn compose_egl_driver_identity(vendor: &str, version: &str) -> String {
    match (vendor.trim(), version.trim()) {
        ("", "") => String::new(),
        ("", version) => version.to_owned(),
        (vendor, "") => vendor.to_owned(),
        (vendor, version) => format!("{vendor} EGL {version}"),
    }
}

/// Whether an observed surface extent can be a viewport on this context.
///
/// The extent is a real EGL observation (`eglQuerySurface` for the live
/// surface), and the maximum viewport dimensions that bound it are a recorded
/// discovery fact; recording a size above them is what would otherwise surface
/// as an unrelated driver error in the first pass that draws into the surface.
///
/// A zero-area extent is not a rejection: it is how a minimized or
/// not-yet-shown surface reports itself and becomes suspension instead.  A
/// limit of zero means the context reported none, so nothing can be measured
/// against it and the check cannot fail closed on a fact that was never read.
fn within_viewport_limit(limit: [u32; 2], extent: [u32; 2]) -> bool {
    if extent.contains(&0) || limit.contains(&0) {
        return true;
    }
    extent[0] <= limit[0] && extent[1] <= limit[1]
}

/// Resolves an observed surface extent against the context's viewport limit.
fn check_drawable_extent(limit: [u32; 2], extent: [u32; 2]) -> Result<(), EglProviderError> {
    if within_viewport_limit(limit, extent) {
        return Ok(());
    }
    Err(EglProviderError::DrawableExtentExceedsViewportLimit {
        requested: extent,
        limit,
    })
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
