//! Rendering-owned backend-neutral presentation façade.
//!
//! This narrow Windows-only boundary binds a native window lifetime to one
//! native surface. It intentionally exposes neither HWND nor DXGI/Vulkan/HAL types.
//! A surface has one portable generation/state façade: resize first retires a
//! configured generation, then either suspends at zero extent or configures a
//! fresh generation. Explicit [`Surface::shutdown`] is the observable teardown operation.
//! Its `Drop` fallback only attempts the same idle-and-unconfigure sequence
//! when no acquired frame remains; accepted-unknown quarantine deliberately
//! keeps the token, native surface, and window lease alive instead.

use core::fmt;
use std::{
    any::Any,
    rc::Rc,
    sync::{Arc, Mutex},
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Device, DeviceOptions, MemoryPolicy, PhysicalResourceIdentity, Texture, TextureDescriptor,
    TextureLease, TextureShared, imp, next_identity,
};
use fluxel_rendergraph::{
    BoundSurfaceTexture, BoundTexture, ResourceAccessState, TextureDesc, TextureDimension,
    TextureFormat, TextureUsage, TextureUsageKind,
};

/// Failure while creating, configuring, acquiring, or shutting down a
/// presentation surface.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SurfaceError {
    /// Window-handle extraction failed before native surface creation.
    WindowHandle(String),
    /// Device/surface bootstrap did not produce a present-compatible adapter.
    Open(crate::OpenError),
    /// The fixed surface configuration could not be installed.
    Configure(String),
    /// The next backbuffer could not be acquired before recording.
    Acquire(String),
    /// Native presentable-image capacity is exhausted for this surface.
    ///
    /// This protects swapchain image ownership only; it is independent from
    /// the renderer's higher-level bounded frames-in-flight policy.
    FrameOutstanding,
    /// Shutdown was requested while an acquired frame still exists.
    FrameStillAcquired,
    /// The surface was already shut down and cannot acquire another image.
    NotConfigured,
    /// The surface is intentionally suspended for a zero-sized target.
    Suspended,
    /// Native retirement or configuration left ownership unknowable; this
    /// surface has quarantined its native state and will never acquire again.
    Poisoned(String),
    /// The finite opaque generation namespace cannot advance safely.
    GenerationExhausted,
    /// A native synchronization or teardown operation failed.
    Shutdown(String),
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WindowHandle(value) => write!(f, "window handle unavailable: {value}"),
            Self::Open(value) => write!(f, "surface open failed: {value}"),
            Self::Configure(value) => write!(f, "surface configure failed: {value}"),
            Self::Acquire(value) => write!(f, "surface acquire failed: {value}"),
            Self::FrameOutstanding => f.write_str("surface has no free presentable-image capacity"),
            Self::FrameStillAcquired => {
                f.write_str("cannot shut down with an acquired surface frame")
            }
            Self::NotConfigured => f.write_str("surface is not configured"),
            Self::Suspended => f.write_str("surface is suspended at zero extent"),
            Self::Poisoned(reason) => write!(f, "surface is poisoned: {reason}"),
            Self::GenerationExhausted => f.write_str("surface generation namespace exhausted"),
            Self::Shutdown(value) => write!(f, "DX12 surface shutdown failed: {value}"),
        }
    }
}

impl std::error::Error for SurfaceError {}

/// Comparable identity for one configured presentable-resource generation.
///
/// Values are allocated only by [`Surface`]; callers can compare and log
/// them but cannot construct or mutate one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SurfaceGeneration(u64);

/// Portable client-pixel extent for a presentation target.
///
/// Zero is deliberately valid and means a suspended, non-drawable surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SurfaceExtent {
    width: u32,
    height: u32,
}

impl SurfaceExtent {
    /// Creates a client-pixel extent; zero dimensions are valid suspension input.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Returns the width in client pixels.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the height in client pixels.
    pub const fn height(self) -> u32 {
        self.height
    }

    fn drawable(self) -> bool {
        self.width != 0 && self.height != 0
    }
}

/// Observable portable lifecycle state of a presentation surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceStatus {
    /// A configured generation can acquire images at this extent.
    Active {
        /// Identity allocated for the currently configured native resources.
        generation: SurfaceGeneration,
        /// Client-pixel size negotiated for that generation.
        extent: SurfaceExtent,
    },
    /// No presentable image exists while the client extent is zero.
    Suspended,
    /// Native ownership became unknown and has been quarantined.
    Poisoned,
    /// Explicit shutdown completed; this façade cannot be configured again.
    Closed,
}

/// A rendering-owned surface that keeps its native window alive.
///
/// It is deliberately !Send and !Sync: window affinity and swapchain lifetime
/// stay on the harness thread in this first visible-image slice.
pub struct Surface {
    device: Device,
    native: Arc<Mutex<imp::NativeSurface>>,
    // The native DXGI surface may outlive an acquired token in accepted-unknown
    // quarantine. Retain the concrete window object, not merely a borrow, so
    // HWND destruction cannot race that token's surface teardown.
    window: Arc<dyn Any>,
    status: SurfaceStatus,
    next_generation: u64,
    closed: bool,
    presentation_tickets: Arc<imp::NativePresentationTickets>,
    #[cfg(feature = "vulkan")]
    vulkan_presentation_sync: Option<Arc<imp::VulkanPresentationSync>>,
    _thread_affinity: Rc<()>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        let frame_outstanding = self.presentation_tickets.any_live();
        let retained = (
            Arc::clone(&self.device.inner),
            Arc::clone(&self.native),
            Arc::clone(&self.window),
            Arc::clone(&self.presentation_tickets),
        );
        // A successful present may have moved only a lightweight gate into a
        // Send-capable completion. If callers drop Surface before that bundle
        // retires, the completion can still own derived render views/queue
        // state. Retain the full native/window ownership here; never rely on
        // automatic field drop merely because no acquire token remains.
        if quarantine_live_drop_ownership(frame_outstanding, retained) {
            return;
        }
        if !may_best_effort_shutdown(matches!(self.status, SurfaceStatus::Active { .. }), false) {
            return;
        }
        // Drop must not turn an unwind/early-return cleanup path into a
        // second failure. `unconfigure_dx12_surface` waits for known queue
        // work before touching swapchain state. An error leaves retirement
        // unknown, so retain the complete ownership bundle for process
        // lifetime rather than letting this Drop release native/window state.
        let retained = (
            Arc::clone(&self.device.inner),
            Arc::clone(&self.native),
            Arc::clone(&self.window),
            Arc::clone(&self.presentation_tickets),
        );
        if !quarantine_after_teardown_failure(
            imp::unconfigure_surface(&self.device.inner, &self.native),
            retained,
        ) {
            self.status = SurfaceStatus::Suspended;
        }
    }
}

fn may_best_effort_shutdown(configured: bool, frame_outstanding: bool) -> bool {
    configured && !frame_outstanding
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SurfaceTransition {
    Noop,
    Suspend,
    Configure,
    RetireThenSuspend,
    RetireThenConfigure,
    RefusePoisoned,
    RefuseClosed,
}

fn classify_transition(status: SurfaceStatus, extent: SurfaceExtent) -> SurfaceTransition {
    match (status, extent.drawable()) {
        (SurfaceStatus::Poisoned, _) => SurfaceTransition::RefusePoisoned,
        (SurfaceStatus::Closed, _) => SurfaceTransition::RefuseClosed,
        (SurfaceStatus::Active { extent: active, .. }, _) if active == extent => {
            SurfaceTransition::Noop
        }
        (SurfaceStatus::Suspended, false) => SurfaceTransition::Suspend,
        (SurfaceStatus::Suspended, true) => SurfaceTransition::Configure,
        (SurfaceStatus::Active { .. }, false) => SurfaceTransition::RetireThenSuspend,
        (SurfaceStatus::Active { .. }, true) => SurfaceTransition::RetireThenConfigure,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShutdownTransition {
    RetireThenClose,
    Close,
    RefusePoisoned,
    RefuseClosed,
}

fn classify_shutdown(status: SurfaceStatus, closed: bool) -> ShutdownTransition {
    if closed || matches!(status, SurfaceStatus::Closed) {
        ShutdownTransition::RefuseClosed
    } else {
        match status {
            SurfaceStatus::Active { .. } => ShutdownTransition::RetireThenClose,
            SurfaceStatus::Suspended => ShutdownTransition::Close,
            SurfaceStatus::Poisoned => ShutdownTransition::RefusePoisoned,
            SurfaceStatus::Closed => unreachable!("handled above"),
        }
    }
}

/// Returns whether a failed best-effort teardown quarantined `ownership`.
///
/// This deliberately leaks only on the branch where native retirement cannot
/// be proven. The caller's ordinary fields may then drop without releasing the
/// cloned device, surface, window, or frame-state ownership bundle.
fn quarantine_after_teardown_failure<T, E>(result: Result<(), E>, ownership: T) -> bool {
    if result.is_err() {
        std::mem::forget(ownership);
        true
    } else {
        false
    }
}

/// Quarantines a Drop-time surface ownership bundle while a frame gate is live.
///
/// A normal successful completion owns only the gate, command bundle and
/// derived views; retaining this separate bundle prevents premature surface or
/// window destruction without making `NativeSubmission` thread-affine.
fn quarantine_live_drop_ownership<T>(frame_outstanding: bool, ownership: T) -> bool {
    if frame_outstanding {
        std::mem::forget(ownership);
        true
    } else {
        false
    }
}

impl Surface {
    /// Opens a device selected specifically for this window. A zero
    /// initial extent is valid and starts suspended without configuring HAL.
    ///
    /// This is the Stage 1 single-surface bootstrap entry point, not a promise
    /// that the eventual multi-surface model makes a surface own device
    /// selection. Callers must not infer future Device/Surface topology from
    /// this proof API.
    pub fn open<W>(
        backend: crate::Backend,
        window: Arc<W>,
        options: DeviceOptions,
        width: u32,
        height: u32,
    ) -> Result<Self, SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + 'static,
    {
        let extent = SurfaceExtent::new(width, height);
        let display = window
            .display_handle()
            .map_err(|e| SurfaceError::WindowHandle(e.to_string()))?
            .as_raw();
        let native_window = window
            .window_handle()
            .map_err(|e| SurfaceError::WindowHandle(e.to_string()))?
            .as_raw();
        let (opened, native_surface) = imp::open_surface(backend, options, display, native_window)
            .map_err(SurfaceError::Open)?;
        let device = Device {
            hardware: opened.hardware.clone(),
            capabilities: opened.capabilities,
            inner: Arc::new(opened),
            identity: fluxel_rendergraph::DeviceIdentity::new(next_identity()),
        };
        let native = Arc::new(Mutex::new(native_surface));
        #[cfg(feature = "vulkan")]
        let vulkan_presentation_sync = imp::create_vulkan_presentation_sync(&device.inner, &native)
            .map_err(SurfaceError::Configure)?;
        let (status, next_generation) = if extent.drawable() {
            // This is initial configuration: no earlier generation, acquired
            // image, or queue submission exists, and every native owner is
            // still local to this constructor. A configure error can therefore
            // return normally without releasing accepted/unknown GPU work.
            let actual_extent = imp::configure_surface(&device.inner, &native, width, height)
                .map_err(SurfaceError::Configure)?;
            (
                SurfaceStatus::Active {
                    generation: SurfaceGeneration(1),
                    extent: SurfaceExtent::new(actual_extent.0, actual_extent.1),
                },
                2,
            )
        } else {
            (SurfaceStatus::Suspended, 1)
        };
        Ok(Self {
            device,
            native,
            status,
            next_generation,
            closed: false,
            window,
            presentation_tickets: imp::NativePresentationTickets::new(imp::PRESENTABLE_IMAGE_COUNT),
            #[cfg(feature = "vulkan")]
            vulkan_presentation_sync,
            _thread_affinity: Rc::new(()),
        })
    }

    /// Returns the device whose adapter was selected for this surface.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Creates the only RasterBackend that advertises this surface's fixed
    /// RGBA8 presentation contract; headless backend construction remains
    /// deliberately non-presentable.
    pub fn raster_backend(&self) -> crate::RasterBackend {
        crate::RasterBackend::for_surface(self.device())
    }

    /// Returns the current portable presentation lifecycle state.
    pub fn status(&self) -> SurfaceStatus {
        if self.presentation_tickets.poisoned() {
            SurfaceStatus::Poisoned
        } else {
            self.status
        }
    }

    /// Retires any active generation then suspends or configures a fresh one.
    ///
    /// A live acquired token refuses the transition. The current serial queue
    /// proves old-generation retirement with `wait_for_idle`; any failure to
    /// establish that proof quarantines ownership and poisons this surface.
    pub fn resize(&mut self, extent: SurfaceExtent) -> Result<SurfaceStatus, SurfaceError> {
        if self.presentation_tickets.poisoned() {
            self.status = SurfaceStatus::Poisoned;
            return Err(SurfaceError::Poisoned(
                "an unpresented native surface image was quarantined".into(),
            ));
        }
        if self.closed {
            return Err(SurfaceError::NotConfigured);
        }
        let transition = classify_transition(self.status, extent);
        if matches!(transition, SurfaceTransition::RefusePoisoned) {
            return Err(SurfaceError::Poisoned(
                "a prior native lifecycle failure was quarantined".into(),
            ));
        }
        if matches!(transition, SurfaceTransition::RefuseClosed) {
            return Err(SurfaceError::NotConfigured);
        }
        if matches!(transition, SurfaceTransition::Noop) {
            return Ok(self.status);
        }
        if self.presentation_tickets.any_live() {
            return Err(SurfaceError::FrameOutstanding);
        }
        if matches!(
            transition,
            SurfaceTransition::RetireThenSuspend | SurfaceTransition::RetireThenConfigure
        ) {
            if let Err(error) = imp::unconfigure_surface(&self.device.inner, &self.native) {
                self.poison_and_quarantine();
                return Err(SurfaceError::Poisoned(format!(
                    "old generation retirement failed: {error}"
                )));
            }
            self.status = SurfaceStatus::Suspended;
        }
        if matches!(
            transition,
            SurfaceTransition::Suspend | SurfaceTransition::RetireThenSuspend
        ) {
            return Ok(self.status);
        }
        let generation = self.reserve_generation()?;
        let actual_extent = match imp::configure_surface(
            &self.device.inner,
            &self.native,
            extent.width,
            extent.height,
        ) {
            Ok(actual_extent) => actual_extent,
            Err(error) => {
                // HAL's unsafe configure contract does not promise that an Err
                // leaves a reusable unconfigured surface. Prefer terminal poison
                // to falsely claiming a retry-safe Suspended state.
                self.poison_and_quarantine();
                return Err(SurfaceError::Poisoned(format!(
                    "new generation configuration failed: {error}"
                )));
            }
        };
        self.status = SurfaceStatus::Active {
            generation,
            extent: SurfaceExtent::new(actual_extent.0, actual_extent.1),
        };
        Ok(self.status)
    }

    /// Acquires one presentable image. Each acquisition owns an independent
    /// private ticket until discard or known completion retirement.
    pub fn acquire(&mut self) -> Result<AcquiredSurfaceFrame, SurfaceError> {
        if self.presentation_tickets.poisoned() {
            self.status = SurfaceStatus::Poisoned;
            return Err(SurfaceError::Poisoned(
                "an unpresented native surface image was quarantined".into(),
            ));
        }
        if self.closed {
            return Err(SurfaceError::NotConfigured);
        }
        let SurfaceStatus::Active { extent, .. } = self.status else {
            return Err(match self.status {
                SurfaceStatus::Suspended => SurfaceError::Suspended,
                SurfaceStatus::Poisoned => SurfaceError::Poisoned(
                    "a prior native lifecycle failure was quarantined".into(),
                ),
                SurfaceStatus::Closed => SurfaceError::NotConfigured,
                SurfaceStatus::Active { .. } => unreachable!(),
            });
        };
        let (acquire_lease, ticket) = self
            .presentation_tickets
            .try_acquire()
            .ok_or(SurfaceError::FrameOutstanding)?;
        let descriptor = TextureDesc {
            dimension: TextureDimension::D2,
            extent: fluxel_rendergraph::Extent3d {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        };
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);
        let (native, token) = match imp::acquire_surface(
            &self.device.inner,
            Arc::clone(&self.native),
            imp::NativeSurfaceAcquireRequest {
                window: Arc::clone(&self.window),
                acquire_lease,
                presentation_ticket: ticket,
                #[cfg(feature = "vulkan")]
                vulkan_presentation_sync: self.vulkan_presentation_sync.clone(),
                descriptor,
                allowed_usage: usage,
            },
        ) {
            Ok(value) => value,
            Err(error) => {
                return Err(SurfaceError::Acquire(error));
            }
        };
        let texture = Texture(Arc::new(TextureShared {
            _native: native,
            descriptor: TextureDescriptor {
                texture: descriptor,
                usage,
                memory: MemoryPolicy::DeviceOnly,
            },
            allowed_usage: usage,
            identity: PhysicalResourceIdentity::new(next_identity()),
            device: self.device.identity(),
        }));
        Ok(AcquiredSurfaceFrame {
            texture,
            token: Some(PresentationToken {
                native: Some(token),
                // The native token owns the acquire flag and window lease
                // until discard or accepted-work retirement.
            }),
        })
    }

    /// Waits for known accepted work and releases the native swapchain.
    ///
    /// The caller must first drop or submit its acquired frame. This explicit
    /// shutdown ordering is the contract later resize generations build on.
    /// A later `Drop` is a no-op after success; if callers skip this method,
    /// `Drop` makes the same best-effort attempt only when it is safe.
    pub fn shutdown(&mut self) -> Result<(), SurfaceError> {
        if self.presentation_tickets.poisoned() {
            self.status = SurfaceStatus::Poisoned;
            return Err(SurfaceError::Poisoned(
                "an unpresented native surface image was quarantined".into(),
            ));
        }
        match classify_shutdown(self.status, self.closed) {
            ShutdownTransition::RefuseClosed => return Err(SurfaceError::NotConfigured),
            ShutdownTransition::RefusePoisoned => {
                return Err(SurfaceError::Poisoned(
                    "a prior native lifecycle failure was quarantined".into(),
                ));
            }
            ShutdownTransition::RetireThenClose | ShutdownTransition::Close => {}
        }
        if self.presentation_tickets.any_live() {
            return Err(SurfaceError::FrameStillAcquired);
        }
        if matches!(
            classify_shutdown(self.status, self.closed),
            ShutdownTransition::RetireThenClose
        ) {
            if let Err(error) = imp::unconfigure_surface(&self.device.inner, &self.native) {
                self.poison_and_quarantine();
                return Err(SurfaceError::Poisoned(format!(
                    "shutdown retirement failed: {error}"
                )));
            }
        }
        self.status = SurfaceStatus::Closed;
        self.closed = true;
        Ok(())
    }

    fn reserve_generation(&mut self) -> Result<SurfaceGeneration, SurfaceError> {
        let value = self.next_generation;
        self.next_generation = self.next_generation.checked_add(1).ok_or_else(|| {
            self.poison_and_quarantine();
            SurfaceError::GenerationExhausted
        })?;
        Ok(SurfaceGeneration(value))
    }

    fn poison_and_quarantine(&mut self) {
        self.status = SurfaceStatus::Poisoned;
        let retained = (
            Arc::clone(&self.device.inner),
            Arc::clone(&self.native),
            Arc::clone(&self.window),
            Arc::clone(&self.presentation_tickets),
        );
        let _ = quarantine_live_drop_ownership(true, retained);
    }
}

/// One acquired presentable image and its unforgeable presentation permission.
pub struct AcquiredSurfaceFrame {
    texture: Texture,
    token: Option<PresentationToken>,
}

impl AcquiredSurfaceFrame {
    /// Converts this acquired image into the graph binding for one imported
    /// surface slot. The token is consumed with the binding, preventing a
    /// second present or a frame that records without a corresponding token.
    pub fn into_binding(mut self) -> BoundSurfaceTexture<Texture, TextureLease, PresentationToken> {
        let texture = self.texture;
        BoundSurfaceTexture {
            texture: BoundTexture {
                device: texture.device_identity(),
                identity: texture.identity(),
                physical: texture.clone(),
                descriptor: texture.descriptor().texture,
                usage: texture.allowed_usage(),
                initial_state: ResourceAccessState::Present,
                lease: texture.lease(),
            },
            presentation: self
                .token
                .take()
                .expect("acquired frame owns its presentation token"),
        }
    }
}

/// Opaque one-shot permission to present an acquired image.
pub struct PresentationToken {
    pub(crate) native: Option<imp::NativePresentationToken>,
}

impl PresentationToken {
    pub(crate) fn into_native(mut self) -> Option<imp::NativePresentationToken> {
        self.native.take()
    }
}

impl Drop for PresentationToken {
    fn drop(&mut self) {
        if let Some(native) = self.native.take() {
            imp::discard_presentation(native);
        }
    }
}

impl fmt::Debug for PresentationToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PresentationToken(..)")
    }
}

#[cfg(test)]
mod tests;
