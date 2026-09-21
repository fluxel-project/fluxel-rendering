//! `CAMetalLayer` presentation lowering.
//!
//! A portable presentation target is only an [`ObjectId`].  This module owns
//! the corresponding `CAMetalLayer` registry, and keeps the native drawable
//! below `FrameAttachmentBackend`.  In particular, a drawable is not converted
//! into a normal `TextureView`: Metal can expose one, but the RHI contract must
//! also represent default-framebuffer backends where that would be a lie.
//!
//! `MTLCommandBuffer::presentDrawable:` is the preferred integration point.  A
//! frame attachment therefore exposes [`schedule_present`] for command lowering:
//! it must be called while the command buffer is still being encoded and before
//! `commit`.  The generic presentation seam subsequently invokes `present`
//! after its validated present relation; when command lowering already scheduled
//! the drawable that call only publishes the portable receipt.  The direct
//! `MTLDrawable::present` branch remains a defensive fallback for a frame that
//! was accepted without raster work.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLDrawable, MTLTexture};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::DeviceLossInfo;
use crate::api::presentation::backend::{
    AcquiredSurfaceFrame, ConfiguredPresentationBackend, FrameAttachmentBackend,
    PresentationBackend,
};
use crate::api::presentation::{
    AcquireError, AcquireErrorKind, AcquiredFrameId, CompositeAlphaMode, Extent2d,
    FrameLatencyRange, PresentMode, PresentReceiptId, PresentState, PresentationColorSpace,
    PresentationConfiguration, PresentationExtent, PresentationExtentControl, PresentationFormat,
    PresentationTarget, PresentationTargetCapabilities, PresentationTimingCapabilities,
};
use crate::api::resource::TextureUsage;

use super::format::metal_format;

/// Shared terminal state for the Metal execution and presentation domains.
///
/// `MetalDevice`/`MetalCommandSpine` integration retains this state and calls
/// [`Self::mark_lost`] from its single native-failure authority.  Keeping the
/// wake list here means acquire and `wait_present` do not rely on a caller
/// polling after device loss.
#[derive(Default)]
pub(crate) struct MetalPresentationLoss {
    state: Mutex<LossState>,
}

#[derive(Default)]
struct LossState {
    info: Option<DeviceLossInfo>,
    waiters: Vec<Waker>,
    handlers: Vec<Arc<dyn Fn(DeviceLossInfo) + Send + Sync>>,
}

impl MetalPresentationLoss {
    pub(crate) fn loss_info(&self) -> Option<DeviceLossInfo> {
        lock(&self.state).info.clone()
    }

    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        let (waiters, handlers) = {
            let mut state = lock(&self.state);
            if state.info.is_some() {
                return;
            }
            state.info = Some(info.clone());
            (std::mem::take(&mut state.waiters), state.handlers.clone())
        };
        for waiter in waiters {
            waiter.wake();
        }
        for handler in handlers {
            handler(info.clone());
        }
    }

    fn register(&self, waker: &Waker) {
        let mut state = lock(&self.state);
        if state.info.is_none() && !state.waiters.iter().any(|known| known.will_wake(waker)) {
            state.waiters.push(waker.clone());
        }
    }

    fn register_handler(&self, handler: Arc<dyn Fn(DeviceLossInfo) + Send + Sync>) {
        let lost = {
            let mut state = lock(&self.state);
            match state.info.clone() {
                Some(info) => Some(info),
                None => {
                    state.handlers.push(handler.clone());
                    None
                }
            }
        };
        if let Some(info) = lost {
            handler(info);
        }
    }
}

/// Private registry of host-owned layers.  The host transfers a retained layer
/// at registration; no raw `NSView`, UIKit view, or Core Animation handle can
/// escape into the public RHI model.
pub(crate) struct MetalPresentation {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    targets: Arc<MetalTargetRegistry>,
    state: Arc<Mutex<PresentationState>>,
    loss: Arc<MetalPresentationLoss>,
}

// Core Animation layers are host-thread-affine.  Every access in this module is
// serialized, and host integration must invoke configure/acquire on the thread
// on which it is legal to message that layer.  The portable backend seam itself
// is Send + Sync, hence these private declarations are required at its boundary.
unsafe impl Send for MetalPresentation {}
unsafe impl Sync for MetalPresentation {}

/// Provider-owned registry of host surfaces. It intentionally exists before a
/// device request: `ProviderBackend::supports_presentation` must preflight a
/// target before a logical device and its device-local presentation facet exist.
pub(crate) struct MetalTargetRegistry {
    targets: Mutex<HashMap<ObjectId, Retained<CAMetalLayer>>>,
}

unsafe impl Send for MetalTargetRegistry {}
unsafe impl Sync for MetalTargetRegistry {}

impl MetalTargetRegistry {
    pub(crate) fn new() -> Self {
        Self {
            targets: Mutex::new(HashMap::new()),
        }
    }

    /// Host integration transfers its retained CAMetalLayer into the provider
    /// registry. The portable target reveals only the generated object identity.
    pub(crate) fn register_layer_target(
        &self,
        layer: Retained<CAMetalLayer>,
    ) -> PresentationTarget {
        let id = ObjectId::next();
        lock(&self.targets).insert(id, layer);
        PresentationTarget::new(id)
    }

    pub(crate) fn contains(&self, target: ObjectId) -> bool {
        lock(&self.targets).contains_key(&target)
    }

    fn layer(&self, target: ObjectId) -> RhiResult<Retained<CAMetalLayer>> {
        lock(&self.targets).get(&target).cloned().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::TargetLost,
                "the presentation target is not registered with this Metal provider",
            )
        })
    }
}

#[derive(Default)]
struct PresentationState {
    leased: HashSet<ObjectId>,
    acquired: HashMap<ObjectId, u64>,
    receipts: HashMap<PresentReceiptId, PresentReceiptState>,
}

struct PresentReceiptState {
    state: PresentState,
    waiters: Vec<Waker>,
}

impl MetalPresentation {
    /// Constructs the device-local presentation facet and its loss bridge.
    /// Device construction keeps the returned `loss` with its execution domain
    /// and calls `mark_lost` whenever Metal reports terminal command failure.
    pub(crate) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        targets: Arc<MetalTargetRegistry>,
        loss: Arc<MetalPresentationLoss>,
    ) -> Self {
        Self {
            device,
            targets,
            state: Arc::new(Mutex::new(PresentationState::default())),
            loss,
        }
    }

    fn layer(&self, target: ObjectId) -> RhiResult<Retained<CAMetalLayer>> {
        self.targets.layer(target)
    }

    fn capabilities_for(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        // The objc2 feature set intentionally does not pull CoreFoundation's
        // `CGSize` vocabulary into the backend merely to query this advisory
        // value.  The authoritative extent is the acquired drawable texture;
        // therefore a pre-acquire snapshot correctly reports host-managed size
        // as unavailable.
        self.layer(target)?;
        Ok(metal_capabilities(None))
    }
}

impl PresentationBackend for MetalPresentation {
    fn capabilities(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        if let Some(info) = self.loss.loss_info() {
            return Err(lost_error(&info, "MetalPresentation::capabilities"));
        }
        self.capabilities_for(target)
    }

    fn configure(
        &self,
        device: DeviceIdentity,
        target: ObjectId,
        config: &PresentationConfiguration,
    ) -> RhiResult<Box<dyn ConfiguredPresentationBackend>> {
        if let Some(info) = self.loss.loss_info() {
            return Err(lost_error(&info, "MetalPresentation::configure"));
        }
        validate_metal_configuration(config)?;
        let layer = self.layer(target)?;
        {
            let mut state = lock(&self.state);
            if !state.leased.insert(target) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this CAMetalLayer already has an active presentation configuration",
                ));
            }
        }
        if let Err(error) = configure_layer(&layer, &self.device, config) {
            lock(&self.state).leased.remove(&target);
            return Err(error);
        }
        let acquire = Arc::new(Mutex::new(AcquireState::default()));
        register_loss_wakeup(&self.loss, &self.state, &acquire, target);
        Ok(Box::new(MetalConfiguredPresentation {
            layer,
            device,
            target,
            state: Arc::clone(&self.state),
            loss: Arc::clone(&self.loss),
            serial: AtomicU64::new(1),
            acquire,
        }))
    }

    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        let state = lock(&self.state);
        receipt_state(&state, receipt)
    }

    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        waker: &Waker,
    ) -> RhiResult<PresentState> {
        let mut state = lock(&self.state);
        let entry = state.receipts.get_mut(&receipt).ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "unknown Metal present receipt")
        })?;
        if matches!(entry.state, PresentState::Pending)
            && !entry.waiters.iter().any(|known| known.will_wake(waker))
        {
            entry.waiters.push(waker.clone());
        }
        Ok(entry.state.clone())
    }
}

struct MetalConfiguredPresentation {
    layer: Retained<CAMetalLayer>,
    device: DeviceIdentity,
    target: ObjectId,
    state: Arc<Mutex<PresentationState>>,
    loss: Arc<MetalPresentationLoss>,
    serial: AtomicU64,
    acquire: Arc<Mutex<AcquireState>>,
}

unsafe impl Send for MetalConfiguredPresentation {}
unsafe impl Sync for MetalConfiguredPresentation {}

/// A single non-blocking portable acquire slot. `nextDrawable` is permitted to
/// wait for Core Animation, so it never runs in `try_acquire` or a future poll.
/// The worker result is deposited here and consumed by the next sample.
struct AcquireState {
    generation: u64,
    worker_active: bool,
    result: Option<AcquireResult>,
    waiter: Option<Waker>,
}

// The contained Objective-C objects are moved from the dedicated acquire
// worker into the serialized slot and then consumed by the caller that polls
// the lease.  They are never accessed concurrently; objc2 correctly marks them
// non-Send by default, so this narrowly scoped boundary is explicit.
unsafe impl Send for AcquireState {}

impl Default for AcquireState {
    fn default() -> Self {
        Self {
            generation: 0,
            worker_active: false,
            result: None,
            waiter: None,
        }
    }
}

enum AcquireResult {
    Drawable {
        drawable: Retained<ProtocolObject<dyn MTLDrawable>>,
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
    },
    NotReady,
}

// A retained Core Animation layer is thread-affine at the host boundary.  The
// worker is the sole caller of `nextDrawable`, while configure/release are
// serialized by the lease and host integration is responsible for registering a
// layer whose platform permits this wait off the application executor thread.
struct DrawableWorkerLayer(Retained<CAMetalLayer>);
unsafe impl Send for DrawableWorkerLayer {}

impl DrawableWorkerLayer {
    fn next_drawable(&self) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        self.0.nextDrawable()
    }
}

impl MetalConfiguredPresentation {
    fn validate_acquire(&self, device: DeviceIdentity) -> Result<(), AcquireError> {
        if device != self.device {
            return Err(acquire(
                AcquireErrorKind::DeviceLost,
                "the Metal presentation lease belongs to another device",
            ));
        }
        if self.loss.loss_info().is_some() {
            return Err(acquire(
                AcquireErrorKind::DeviceLost,
                "the Metal device was lost; this presentation lease is terminal",
            ));
        }
        if lock(&self.state).acquired.contains_key(&self.target) {
            return Err(acquire(
                AcquireErrorKind::FrameOutstanding,
                "a CAMetalLayer drawable remains acquired",
            ));
        }
        Ok(())
    }

    fn take_ready(&self) -> Result<Option<AcquiredSurfaceFrame>, AcquireError> {
        let result = lock(&self.acquire).result.take();
        let Some(result) = result else {
            return Ok(None);
        };
        let AcquireResult::Drawable { drawable, texture } = result else {
            return Ok(None);
        };
        let width = u32::try_from(texture.width()).ok();
        let height = u32::try_from(texture.height()).ok();
        let (Some(width), Some(height)) = (width, height) else {
            return Err(acquire(
                AcquireErrorKind::Outdated,
                "CAMetalLayer drawable extent exceeds the portable range",
            ));
        };
        if width == 0 || height == 0 {
            return Err(acquire(
                AcquireErrorKind::ZeroSizeOrSuspended,
                "CAMetalLayer drawable has a zero extent",
            ));
        }
        let serial = self.serial.fetch_add(1, Ordering::Relaxed);
        lock(&self.state).acquired.insert(self.target, serial);
        Ok(Some(AcquiredSurfaceFrame {
            serial,
            extent: Extent2d { width, height },
            suboptimal: false,
            attachment: Box::new(MetalFrameAttachment {
                drawable: Some(drawable),
                texture: Some(texture),
                target: self.target,
                serial,
                state: Arc::clone(&self.state),
                loss: Arc::clone(&self.loss),
                scheduled: Mutex::new(false),
            }),
        }))
    }

    fn start_worker_or_replace_waker(&self, waker: &Waker) {
        let (generation, start) = {
            let mut acquire = lock(&self.acquire);
            acquire.waiter = Some(waker.clone());
            if acquire.worker_active {
                (acquire.generation, false)
            } else {
                acquire.worker_active = true;
                (acquire.generation, true)
            }
        };
        if !start {
            return;
        }
        let layer = DrawableWorkerLayer(self.layer.clone());
        let acquire_state = Arc::clone(&self.acquire);
        std::thread::spawn(move || {
            let drawable = layer.next_drawable();
            let result = match drawable {
                Some(drawable) => AcquireResult::Drawable {
                    texture: drawable.texture(),
                    drawable: ProtocolObject::from_retained(drawable),
                },
                None => AcquireResult::NotReady,
            };
            let waiter = {
                let mut state = lock(&acquire_state);
                if state.generation != generation {
                    // Release/reconfigure invalidated this request.  Dropping
                    // `result` returns the drawable to Core Animation.
                    return;
                }
                state.worker_active = false;
                state.result = Some(result);
                state.waiter.take()
            };
            if let Some(waker) = waiter {
                waker.wake();
            }
        });
    }

    fn cancel_acquire(&self) {
        let waiter = {
            let mut acquire = lock(&self.acquire);
            acquire.generation = acquire.generation.wrapping_add(1);
            acquire.result = None;
            acquire.waiter.take()
        };
        if let Some(waker) = waiter {
            waker.wake();
        }
    }
}

impl ConfiguredPresentationBackend for MetalConfiguredPresentation {
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities> {
        if let Some(info) = self.loss.loss_info() {
            return Err(lost_error(
                &info,
                "MetalConfiguredPresentation::capabilities",
            ));
        }
        Ok(metal_capabilities(None))
    }

    fn reconfigure_or_register_waker(
        &self,
        config: &PresentationConfiguration,
        _waker: &Waker,
    ) -> Poll<RhiResult<()>> {
        if let Some(info) = self.loss.loss_info() {
            return Poll::Ready(Err(lost_error(
                &info,
                "MetalConfiguredPresentation::reconfigure",
            )));
        }
        if lock(&self.state).acquired.contains_key(&self.target) {
            return Poll::Ready(Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot reconfigure a CAMetalLayer while a frame is acquired",
            )));
        }
        if lock(&self.acquire).worker_active {
            return Poll::Ready(Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot reconfigure a CAMetalLayer while drawable acquisition is pending",
            )));
        }
        // The layer retains its configured device.  The device is set by the
        // first configuration and does not change during a portable lease.
        Poll::Ready(
            validate_metal_configuration(config)
                .and_then(|_| configure_existing_layer(&self.layer, config)),
        )
    }

    fn try_acquire(
        &self,
        device: DeviceIdentity,
    ) -> Result<Option<AcquiredSurfaceFrame>, AcquireError> {
        self.validate_acquire(device)?;
        self.take_ready()
    }

    fn acquire_or_register_waker(
        &self,
        device: DeviceIdentity,
        waker: &Waker,
    ) -> Poll<Result<AcquiredSurfaceFrame, AcquireError>> {
        if let Err(error) = self.validate_acquire(device) {
            return Poll::Ready(Err(error));
        }
        match self.take_ready() {
            Ok(Some(frame)) => Poll::Ready(Ok(frame)),
            Ok(None) => {
                self.loss.register(waker);
                self.start_worker_or_replace_waker(waker);
                Poll::Pending
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn abandon(&self, _: AcquiredFrameId) -> RhiResult<()> {
        if let Some(info) = self.loss.loss_info() {
            return Err(lost_error(&info, "MetalConfiguredPresentation::abandon"));
        }
        lock(&self.state).acquired.remove(&self.target);
        Ok(())
    }

    fn abandon_no_throw(&self, _: AcquiredFrameId) {
        lock(&self.state).acquired.remove(&self.target);
    }

    fn release(&self) {
        self.cancel_acquire();
        let mut state = lock(&self.state);
        state.acquired.remove(&self.target);
        state.leased.remove(&self.target);
    }
}

/// Backend-private acquired `CAMetalDrawable`.  The drawable and its texture
/// stay retained until every portable `FrameAttachment` clone and accepted work
/// releases them.
pub(crate) struct MetalFrameAttachment {
    drawable: Option<Retained<ProtocolObject<dyn MTLDrawable>>>,
    texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    target: ObjectId,
    serial: u64,
    state: Arc<Mutex<PresentationState>>,
    loss: Arc<MetalPresentationLoss>,
    scheduled: Mutex<bool>,
}

unsafe impl Send for MetalFrameAttachment {}
unsafe impl Sync for MetalFrameAttachment {}

impl MetalFrameAttachment {
    /// Native texture for render-pass lowering only.
    pub(crate) fn texture(&self) -> RhiResult<&ProtocolObject<dyn MTLTexture>> {
        self.texture.as_deref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::DeviceLost,
                "the Metal drawable texture was retired",
            )
        })
    }

    /// Schedules this drawable on the command buffer before that buffer commits.
    /// Command lowering calls this after ending the render encoder; it is the
    /// only route that couples rendering work and display ownership atomically.
    pub(crate) fn schedule_present(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> RhiResult<()> {
        let drawable = self.drawable.as_deref().ok_or_else(|| {
            RhiError::new(RhiErrorKind::DeviceLost, "the Metal drawable was retired")
        })?;
        let mut scheduled = lock(&self.scheduled);
        if *scheduled {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the Metal drawable is already scheduled for presentation",
            ));
        }
        command_buffer.presentDrawable(drawable);
        *scheduled = true;
        Ok(())
    }

    fn finish(&self, receipt: PresentReceiptId, answer: PresentState) {
        let waiters = {
            let mut state = lock(&self.state);
            state.acquired.remove(&self.target);
            let entry = state
                .receipts
                .entry(receipt)
                .or_insert(PresentReceiptState {
                    state: PresentState::Pending,
                    waiters: Vec::new(),
                });
            entry.state = answer;
            std::mem::take(&mut entry.waiters)
        };
        for waiter in waiters {
            waiter.wake();
        }
    }
}

impl FrameAttachmentBackend for MetalFrameAttachment {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn present(&self, receipt: PresentReceiptId) {
        let answer = if let Some(info) = self.loss.loss_info() {
            PresentState::DeviceLost(info)
        } else if *lock(&self.scheduled) {
            PresentState::Accepted
        } else if let Some(drawable) = self.drawable.as_deref() {
            // A frame may legally be presented without raster work, so it has
            // no command buffer on which lowering could schedule the drawable.
            drawable.present();
            PresentState::Accepted
        } else {
            PresentState::Failed(crate::api::presentation::PresentFailure::new(
                "the CAMetalDrawable was retired before presentation",
            ))
        };
        self.finish(receipt, answer);
    }

    fn terminate_present(&self, receipt: PresentReceiptId, state: PresentState) {
        self.finish(receipt, state);
    }
}

impl Drop for MetalFrameAttachment {
    fn drop(&mut self) {
        // Drop native retained values before advertising that the outstanding
        // portable lease ended.  This mirrors the DXGI resize ordering rule and
        // avoids a host layer observing a logically free drawable still held by
        // this object.
        drop(self.texture.take());
        drop(self.drawable.take());
        let mut state = lock(&self.state);
        if state.acquired.get(&self.target) == Some(&self.serial) {
            state.acquired.remove(&self.target);
        }
    }
}

/// Downcasts a portable attachment for Metal raster lowering.
pub(crate) fn frame_attachment(
    frame: &crate::api::presentation::FrameAttachment,
) -> RhiResult<&MetalFrameAttachment> {
    frame
        .native()
        .as_any()
        .downcast_ref::<MetalFrameAttachment>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "frame attachment is not backed by Metal",
            )
        })
}

fn receipt_state(state: &PresentationState, receipt: PresentReceiptId) -> RhiResult<PresentState> {
    state
        .receipts
        .get(&receipt)
        .map(|entry| entry.state.clone())
        .ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "unknown Metal present receipt"))
}

fn metal_capabilities(current: Option<Extent2d>) -> PresentationTargetCapabilities {
    PresentationTargetCapabilities::new(
        vec![TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb],
        vec![
            PresentMode::Automatic,
            PresentMode::Fifo,
            PresentMode::Immediate,
        ],
        PresentationExtentControl::HostManaged { current },
    )
    .with_format_color_spaces(vec![
        PresentationFormat {
            format: TextureFormat::Bgra8Unorm,
            color_space: PresentationColorSpace::Srgb,
        },
        PresentationFormat {
            format: TextureFormat::Bgra8UnormSrgb,
            color_space: PresentationColorSpace::Srgb,
        },
    ])
    .with_surface_details(
        TextureUsage::COLOR_ATTACHMENT,
        vec![CompositeAlphaMode::Automatic, CompositeAlphaMode::Opaque],
        // CAMetalLayer owns one more drawable than the portable latency request;
        // its documented maximum drawable count is three.
        Some(FrameLatencyRange { min: 1, max: 2 }),
        Vec::new(),
    )
    .with_timing_and_hdr(PresentationTimingCapabilities { timestamps: false }, None)
}

fn validate_metal_configuration(config: &PresentationConfiguration) -> RhiResult<()> {
    if !matches!(config.extent(), PresentationExtent::HostManaged) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "CAMetalLayer drawable extent is host-managed",
        ));
    }
    if !matches!(
        config.format(),
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
    ) || metal_format(config.format()).is_none()
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "this presentation format has no CAMetalLayer lowering",
        ));
    }
    if !matches!(
        config.present_mode(),
        PresentMode::Automatic | PresentMode::Fifo | PresentMode::Immediate
    ) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "this Metal present mode has no CAMetalLayer lowering",
        ));
    }
    if config.usage() != TextureUsage::COLOR_ATTACHMENT || !config.view_formats().is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "CAMetalLayer supports only the baseline color-attachment presentation usage",
        ));
    }
    if !matches!(
        config.composite_alpha_mode(),
        CompositeAlphaMode::Automatic | CompositeAlphaMode::Opaque
    ) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "CAMetalLayer alpha composition mode is outside the Metal baseline",
        ));
    }
    if !(1..=2).contains(&config.maximum_frame_latency()) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "CAMetalLayer maximum frame latency must be in 1..=2",
        ));
    }
    Ok(())
}

fn configure_layer(
    layer: &CAMetalLayer,
    device: &ProtocolObject<dyn MTLDevice>,
    config: &PresentationConfiguration,
) -> RhiResult<()> {
    configure_existing_layer(layer, config)?;
    layer.setDevice(Some(device));
    Ok(())
}

fn configure_existing_layer(
    layer: &CAMetalLayer,
    config: &PresentationConfiguration,
) -> RhiResult<()> {
    let format = metal_format(config.format()).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "presentation format has no Metal pixel format",
        )
    })?;
    layer.setPixelFormat(format);
    layer.setFramebufferOnly(true);
    // The worker may block here, but a timed-out wait returns `None` instead of
    // indefinitely pinning one of the backend's acquire workers.
    layer.setAllowsNextDrawableTimeout(true);
    layer.setMaximumDrawableCount((config.maximum_frame_latency() + 1) as usize);
    // `displaySyncEnabled` is the only CAMetalLayer control matching portable
    // immediate versus paced presentation. `Automatic` deliberately retains the
    // host/layer default rather than fabricating a pacing promise.
    match config.present_mode() {
        PresentMode::Fifo => layer.setDisplaySyncEnabled(true),
        PresentMode::Immediate => layer.setDisplaySyncEnabled(false),
        PresentMode::Automatic => {}
        PresentMode::Mailbox => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "CAMetalLayer has no mailbox presentation lowering",
            ));
        }
    }
    Ok(())
}

fn acquire(kind: AcquireErrorKind, message: &'static str) -> AcquireError {
    AcquireError::new(kind, message)
}

fn lost_error(info: &DeviceLossInfo, operation: &'static str) -> RhiError {
    RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned()).at(operation)
}

/// Connects one short-lived configuration lease to the device's one-way loss
/// authority without retaining it forever.  Every pending public future gets a
/// wake, and no receipt is left in `Pending` once the execution domain is lost.
fn register_loss_wakeup(
    loss: &Arc<MetalPresentationLoss>,
    presentation: &Arc<Mutex<PresentationState>>,
    acquire: &Arc<Mutex<AcquireState>>,
    target: ObjectId,
) {
    let presentation = Arc::downgrade(presentation);
    let acquire = Arc::downgrade(acquire);
    loss.register_handler(Arc::new(move |info| {
        let mut wake = Vec::new();
        if let Some(presentation) = presentation.upgrade() {
            let mut state = lock(&presentation);
            state.acquired.remove(&target);
            for receipt in state.receipts.values_mut() {
                if matches!(receipt.state, PresentState::Pending) {
                    receipt.state = PresentState::DeviceLost(info.clone());
                    wake.append(&mut receipt.waiters);
                }
            }
        }
        if let Some(acquire) = acquire.upgrade() {
            let mut state = lock(&acquire);
            state.generation = state.generation.wrapping_add(1);
            state.result = None;
            state.worker_active = false;
            if let Some(waker) = state.waiter.take() {
                wake.push(waker);
            }
        }
        for waker in wake {
            waker.wake();
        }
    }));
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
