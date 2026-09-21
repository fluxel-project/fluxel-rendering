//! WebGPU canvas presentation lowering.
//!
//! A `GPUCanvasContext` is owned by the browser thread just like `GPUDevice`.
//! Consequently this module keeps canvas and context values in a TLS registry;
//! the `PresentationBackend` object and every frame attachment contain only
//! opaque registrations and ordinary numeric state.  This is deliberately not
//! a public browser-session model: hosts register a canvas, receive the normal
//! portable `PresentationTarget`, and all browser objects remain private here.

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::HtmlCanvasElement;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::DeviceStatus;
use crate::api::presentation::backend::{
    AcquiredSurfaceFrame, ConfiguredPresentationBackend, FrameAttachmentBackend,
    PresentationBackend,
};
use crate::api::presentation::{
    AcquireError, AcquireErrorKind, AcquiredFrameId, CompositeAlphaMode, Extent2d, PresentMode,
    PresentReceiptId, PresentState, PresentationColorSpace, PresentationConfiguration,
    PresentationExtentControl, PresentationFormat, PresentationTarget,
    PresentationTargetCapabilities, PresentationTimingCapabilities,
};
use crate::api::resource::TextureUsage;

use super::registry::{WebGpuDriver, WebGpuObjectId, WebGpuRegistration};
use super::{js, registry, translate};

/// Browser-thread data for a host-registered canvas.  The context is created
/// lazily, because `getContext("webgpu")` is meaningful only when a device is
/// about to configure this exact target.
struct CanvasTarget {
    canvas: HtmlCanvasElement,
    context: Option<JsValue>,
}

thread_local! {
    static CANVASES: RefCell<HashMap<(WebGpuRegistration, ObjectId), CanvasTarget>> =
        RefCell::new(HashMap::new());
}

/// Registers a browser canvas with one already-settled WebGPU device.
///
/// This is a crate-private host/test integration entry point.  The returned
/// portable target has no browser object in it; it is merely an ID whose native
/// meaning is looked up on the browser owner thread when configured.
pub(crate) fn register_canvas(
    driver: &WebGpuDriver,
    canvas: HtmlCanvasElement,
) -> PresentationTarget {
    let registration = driver.registration();
    let target = ObjectId::next();
    CANVASES.with(|targets| {
        targets.borrow_mut().insert(
            (registration, target),
            CanvasTarget {
                canvas,
                context: None,
            },
        );
    });
    PresentationTarget::new(target)
}

/// Removes browser references when the final owner of a WebGPU device
/// generation disappears. Called from the registry's `WebGpuDriverInner::Drop`;
/// keeping the cleanup there is what makes it correspond to the actual native
/// device lifetime rather than to one portable `Device` clone.
pub(super) fn remove_canvases(registration: WebGpuRegistration) {
    CANVASES.with(|targets| {
        targets
            .borrow_mut()
            .retain(|(registered, _), _| *registered != registration);
    });
}

fn target_exists(registration: WebGpuRegistration, target: ObjectId) -> bool {
    CANVASES.with(|targets| targets.borrow().contains_key(&(registration, target)))
}

fn with_target<T>(
    registration: WebGpuRegistration,
    target: ObjectId,
    f: impl FnOnce(&CanvasTarget) -> T,
) -> Option<T> {
    CANVASES.with(|targets| targets.borrow().get(&(registration, target)).map(f))
}

/// Returns the context without retaining a `RefCell` borrow over a browser
/// call. Browser code may synchronously re-enter wasm, so keeping that borrow
/// across `getContext`, `configure`, or `getCurrentTexture` would turn an
/// otherwise valid host callback into a RefCell panic.
fn canvas_context(registration: WebGpuRegistration, target: ObjectId) -> RhiResult<JsValue> {
    if let Some(context) =
        with_target(registration, target, |entry| entry.context.clone()).flatten()
    {
        return Ok(context);
    }
    let canvas =
        with_target(registration, target, |entry| entry.canvas.clone()).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::TargetLost,
                "the WebGPU canvas target is no longer registered",
            )
        })?;
    let context: JsValue = canvas
        .get_context("webgpu")
        .map_err(|error| {
            RhiError::new(RhiErrorKind::BackendFailure, js::message(&error))
                .at("HTMLCanvasElement.getContext(webgpu)")
        })?
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "this canvas does not expose a WebGPU context",
            )
        })?
        .into();
    CANVASES.with(|targets| {
        let mut targets = targets.borrow_mut();
        let entry = targets.get_mut(&(registration, target)).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::TargetLost,
                "the WebGPU canvas target was unregistered while opening its WebGPU context",
            )
        })?;
        // `getContext` is idempotent for a canvas/type pair. If a re-entrant
        // host path initialized the same entry while our call was in flight,
        // use that retained value rather than replacing it.
        Ok(entry.context.get_or_insert(context).clone())
    })
}

/// Numeric state shared by configured leases and their frame attachments.
/// Browser values deliberately do not occur here.
struct PresentationState {
    leased: HashSet<ObjectId>,
    acquired: HashMap<ObjectId, u64>,
    receipts: HashMap<PresentReceiptId, PresentState>,
}

impl PresentationState {
    fn new() -> Self {
        Self {
            leased: HashSet::new(),
            acquired: HashMap::new(),
            receipts: HashMap::new(),
        }
    }
}

/// The presentation facet of one WebGPU device registration.
pub(crate) struct WebGpuPresentation {
    driver: WebGpuDriver,
    state: Arc<Mutex<PresentationState>>,
}

impl WebGpuPresentation {
    pub(crate) fn new(driver: WebGpuDriver) -> Self {
        Self {
            driver,
            state: Arc::new(Mutex::new(PresentationState::new())),
        }
    }

    fn registration(&self) -> WebGpuRegistration {
        self.driver.registration()
    }

    fn require_active(&self, operation: &'static str) -> RhiResult<()> {
        match registry::device_status(self.registration()) {
            Some(DeviceStatus::Active) => Ok(()),
            Some(DeviceStatus::Lost) => Err(lost_rhi_error(self.registration(), operation)),
            None => Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "the WebGPU owner-thread registration is no longer available",
            )
            .at(operation)),
        }
    }

    fn capabilities_for(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        self.require_active("WebGpuPresentation::capabilities")?;
        let format = with_target(self.registration(), target, |entry| {
            preferred_format(&entry.canvas)
        })
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::TargetLost,
                "the WebGPU canvas target is no longer registered",
            )
        })??;
        let extent = with_target(self.registration(), target, |entry| {
            canvas_extent(&entry.canvas)
        })
        .expect("target existence was checked above");
        Ok(canvas_capabilities(format, extent))
    }
}

impl PresentationBackend for WebGpuPresentation {
    fn capabilities(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        self.capabilities_for(target)
    }

    fn configure(
        &self,
        device: DeviceIdentity,
        target: ObjectId,
        config: &PresentationConfiguration,
    ) -> RhiResult<Box<dyn ConfiguredPresentationBackend>> {
        self.require_active("WebGpuPresentation::configure")?;
        // Query before claiming the lease.  It validates target ownership and
        // keeps a missing target from leaving an unreleaseable numeric lease.
        self.capabilities_for(target)?;
        {
            let mut state = lock(&self.state);
            if !state.leased.insert(target) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this WebGPU canvas already has an active presentation configuration",
                ));
            }
        }
        let configured = configure_canvas(self.registration(), target, config);
        if let Err(error) = configured {
            lock(&self.state).leased.remove(&target);
            return Err(error);
        }
        Ok(Box::new(WebGpuConfiguredPresentation {
            driver: self.driver.clone(),
            state: Arc::clone(&self.state),
            device,
            target,
            serial: AtomicU64::new(1),
        }))
    }

    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        lock(&self.state)
            .receipts
            .get(&receipt)
            .cloned()
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "unknown WebGPU present receipt")
            })
    }

    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        _: &Waker,
    ) -> RhiResult<PresentState> {
        // WebGPU canvas presentation is automatic: queue submission transfers
        // ownership and there is no browser API for a later scan-out receipt.
        // Therefore this lowering never reports Pending and does not retain a
        // waiter merely to manufacture asynchronous behaviour.
        self.present_state(receipt)
    }
}

struct WebGpuConfiguredPresentation {
    // A configured lease and acquired attachments can outlive the public
    // `Device` wrapper. Retaining this driver keeps their owner-thread entry
    // alive without putting a JS value in either portable object.
    driver: WebGpuDriver,
    state: Arc<Mutex<PresentationState>>,
    device: DeviceIdentity,
    target: ObjectId,
    serial: AtomicU64,
}

impl WebGpuConfiguredPresentation {
    fn registration(&self) -> WebGpuRegistration {
        self.driver.registration()
    }

    fn require_active(&self, operation: &'static str) -> RhiResult<()> {
        match registry::device_status(self.registration()) {
            Some(DeviceStatus::Active) => Ok(()),
            Some(DeviceStatus::Lost) => Err(lost_rhi_error(self.registration(), operation)),
            None => Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "the WebGPU device registration is unavailable",
            )
            .at(operation)),
        }
    }

    fn acquire_now(&self, device: DeviceIdentity) -> Result<AcquiredSurfaceFrame, AcquireError> {
        if device != self.device {
            return Err(AcquireError::new(
                AcquireErrorKind::DeviceLost,
                "the WebGPU presentation lease belongs to another device",
            ));
        }
        if registry::device_status(self.registration()) != Some(DeviceStatus::Active) {
            return Err(AcquireError::new(
                AcquireErrorKind::DeviceLost,
                "the WebGPU device was lost; this presentation lease is terminal",
            ));
        }
        if !target_exists(self.registration(), self.target) {
            return Err(AcquireError::new(
                AcquireErrorKind::TargetLost,
                "the WebGPU canvas target is no longer registered",
            ));
        }
        if let Some(serial) = lock(&self.state).acquired.get(&self.target).copied() {
            return Err(AcquireError::new(
                AcquireErrorKind::FrameOutstanding,
                format!("WebGPU canvas frame {serial} remains acquired"),
            ));
        }
        let extent = with_target(self.registration(), self.target, |entry| {
            canvas_extent(&entry.canvas)
        })
        .expect("target existence was checked above");
        if extent.width == 0 || extent.height == 0 {
            return Err(AcquireError::new(
                AcquireErrorKind::ZeroSizeOrSuspended,
                "the WebGPU canvas has a zero drawable extent",
            ));
        }
        let view = match current_texture_view(self.registration(), self.target) {
            Ok(view) => view,
            Err(error) if error.kind() == RhiErrorKind::TargetLost => {
                return Err(AcquireError::new(
                    AcquireErrorKind::TargetLost,
                    error.to_string(),
                ));
            }
            Err(error) if error.kind() == RhiErrorKind::DeviceLost => {
                return Err(AcquireError::new(
                    AcquireErrorKind::DeviceLost,
                    error.to_string(),
                ));
            }
            Err(error) => {
                return Err(AcquireError::new(
                    AcquireErrorKind::Outdated,
                    error.to_string(),
                ));
            }
        };
        let object = match registry::insert_object(self.registration(), view) {
            Some(object) => object,
            None => {
                return Err(AcquireError::new(
                    AcquireErrorKind::DeviceLost,
                    "WebGPU device registration disappeared during acquisition",
                ));
            }
        };
        let serial = self.serial.fetch_add(1, Ordering::Relaxed);
        lock(&self.state).acquired.insert(self.target, serial);
        Ok(AcquiredSurfaceFrame {
            serial,
            extent,
            suboptimal: false,
            attachment: Box::new(WebGpuFrameAttachment {
                driver: self.driver.clone(),
                state: Arc::clone(&self.state),
                target: self.target,
                serial,
                view: Some(object),
            }),
        })
    }
}

impl ConfiguredPresentationBackend for WebGpuConfiguredPresentation {
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities> {
        self.require_active("WebGpuConfiguredPresentation::capabilities")?;
        let format = with_target(self.registration(), self.target, |entry| {
            preferred_format(&entry.canvas)
        })
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::TargetLost,
                "the WebGPU canvas target is no longer registered",
            )
        })??;
        let extent = with_target(self.registration(), self.target, |entry| {
            canvas_extent(&entry.canvas)
        })
        .expect("target existence was checked above");
        Ok(canvas_capabilities(format, extent))
    }

    fn reconfigure_or_register_waker(
        &self,
        config: &PresentationConfiguration,
        _: &Waker,
    ) -> Poll<RhiResult<()>> {
        if let Err(error) = self.require_active("WebGpuConfiguredPresentation::reconfigure") {
            return Poll::Ready(Err(error));
        }
        if lock(&self.state).acquired.contains_key(&self.target) {
            return Poll::Ready(Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot reconfigure a WebGPU canvas while a frame is acquired",
            )));
        }
        Poll::Ready(configure_canvas(self.registration(), self.target, config))
    }

    fn try_acquire(
        &self,
        device: DeviceIdentity,
    ) -> Result<Option<AcquiredSurfaceFrame>, AcquireError> {
        self.acquire_now(device).map(Some)
    }

    fn acquire_or_register_waker(
        &self,
        device: DeviceIdentity,
        _: &Waker,
    ) -> Poll<Result<AcquiredSurfaceFrame, AcquireError>> {
        // `getCurrentTexture` is the WebGPU acquire operation and is synchronous.
        // Browser scheduling does not expose a drawable-ready event, so returning
        // Pending here would violate the seam's retained-waker requirement.
        Poll::Ready(self.acquire_now(device))
    }

    fn abandon(&self, _: AcquiredFrameId) -> RhiResult<()> {
        self.require_active("WebGpuConfiguredPresentation::abandon")?;
        lock(&self.state).acquired.remove(&self.target);
        Ok(())
    }

    fn abandon_no_throw(&self, _: AcquiredFrameId) {
        lock(&self.state).acquired.remove(&self.target);
    }

    fn release(&self) {
        lock(&self.state).acquired.remove(&self.target);
        lock(&self.state).leased.remove(&self.target);
        // `unconfigure` drops the canvas-context's reference to this device.
        // It is intentionally best-effort: Drop cannot report an already-lost
        // canvas/device, and browser automatic presentation has no release error.
        if let Some(context) = with_target(self.registration(), self.target, |entry| {
            entry.context.clone()
        })
        .flatten()
        {
            let _ = call0(&context, "unconfigure");
        }
    }
}

/// One acquired `GPUTextureView`.  The view remains retained in the device's
/// owner-thread object registry until all portable frame-attachment clones are
/// gone, which includes recorded/accepted work that retained the attachment.
struct WebGpuFrameAttachment {
    // FrameAttachment clones may be retained by recorded/accepted work after
    // their configured lease has gone away, so this is the final native-lifetime
    // owner for the retained GPUTextureView.
    driver: WebGpuDriver,
    state: Arc<Mutex<PresentationState>>,
    target: ObjectId,
    serial: u64,
    view: Option<WebGpuObjectId>,
}

impl FrameAttachmentBackend for WebGpuFrameAttachment {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn present(&self, receipt: PresentReceiptId) {
        let state = match registry::device_loss(self.driver.registration()) {
            Some(info) => PresentState::DeviceLost(info),
            None => PresentState::Accepted,
        };
        let mut presentation = lock(&self.state);
        presentation.receipts.insert(receipt, state);
        if presentation.acquired.get(&self.target) == Some(&self.serial) {
            presentation.acquired.remove(&self.target);
        }
    }

    fn terminate_present(&self, receipt: PresentReceiptId, state: PresentState) {
        let mut presentation = lock(&self.state);
        presentation.receipts.insert(receipt, state);
        if presentation.acquired.get(&self.target) == Some(&self.serial) {
            presentation.acquired.remove(&self.target);
        }
    }
}

impl Drop for WebGpuFrameAttachment {
    fn drop(&mut self) {
        if let Some(view) = self.view.take() {
            let _ = registry::remove_object(self.driver.registration(), view);
        }
        let mut presentation = lock(&self.state);
        if presentation.acquired.get(&self.target) == Some(&self.serial) {
            presentation.acquired.remove(&self.target);
        }
    }
}

/// Retrieves the native view after portable frame-attachment validation.  This
/// remains backend-private so command lowering never receives a browser object
/// through a portable RHI type.
pub(crate) fn frame_view(frame: &crate::api::presentation::FrameAttachment) -> RhiResult<JsValue> {
    let attachment = frame
        .native()
        .as_any()
        .downcast_ref::<WebGpuFrameAttachment>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "frame attachment has no WebGPU canvas backing",
            )
        })?;
    let object = attachment.view.ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::DeviceLost,
            "the WebGPU presentation texture view was retired",
        )
    })?;
    registry::with_object(attachment.driver.registration(), object, Clone::clone).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::DeviceLost,
            "the WebGPU presentation texture view is unavailable",
        )
    })
}

fn configure_canvas(
    registration: WebGpuRegistration,
    target: ObjectId,
    config: &PresentationConfiguration,
) -> RhiResult<()> {
    let device = registry::with_device_handles(registration, |handles| handles.device.clone())
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::DeviceLost,
                "the WebGPU device registration is unavailable",
            )
        })?;
    let format = translate::texture_format(config.format()).map_err(|unsupported| {
        RhiError::new(RhiErrorKind::Unsupported, unsupported.what)
            .at("WebGpuPresentation::configure")
    })?;
    let descriptor = Object::new();
    field(&descriptor, "device", device)?;
    field(&descriptor, "format", JsValue::from_str(format))?;
    field(
        &descriptor,
        "usage",
        JsValue::from_f64(translate::texture_usage(config.usage()) as f64),
    )?;
    field(
        &descriptor,
        "alphaMode",
        JsValue::from_str(alpha_mode(config.composite_alpha_mode())?),
    )?;
    if !config.view_formats().is_empty() {
        let formats = Array::new();
        for view_format in config.view_formats() {
            formats.push(&JsValue::from_str(
                translate::texture_format(*view_format).map_err(|unsupported| {
                    RhiError::new(RhiErrorKind::Unsupported, unsupported.what)
                        .at("WebGpuPresentation::configure view format")
                })?,
            ));
        }
        field(&descriptor, "viewFormats", formats.into())?;
    }
    let context = canvas_context(registration, target)?;
    call1(&context, "configure", &descriptor).map(|_| ())
}

fn current_texture_view(registration: WebGpuRegistration, target: ObjectId) -> RhiResult<JsValue> {
    let context = canvas_context(registration, target)?;
    let texture = call0(&context, "getCurrentTexture")?;
    call1(&texture, "createView", &Object::new().into())
}

fn preferred_format(canvas: &HtmlCanvasElement) -> RhiResult<TextureFormat> {
    let _ = canvas; // The preference is a browser GPU property, not canvas-local.
    let gpu = js::browser_gpu().map_err(|error| {
        RhiError::new(RhiErrorKind::Unsupported, js::message(&error)).at("navigator.gpu")
    })?;
    let name = call0(&gpu, "getPreferredCanvasFormat")?
        .as_string()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "navigator.gpu.getPreferredCanvasFormat returned a non-string",
            )
        })?;
    match name.as_str() {
        "bgra8unorm" => Ok(TextureFormat::Bgra8Unorm),
        "rgba8unorm" => Ok(TextureFormat::Rgba8Unorm),
        _ => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!(
                "WebGPU preferred canvas format {name:?} is outside the RHI presentation baseline"
            ),
        )),
    }
}

fn canvas_capabilities(format: TextureFormat, current: Extent2d) -> PresentationTargetCapabilities {
    PresentationTargetCapabilities::new(
        vec![format],
        vec![PresentMode::Automatic],
        PresentationExtentControl::HostManaged {
            current: Some(current),
        },
    )
    .with_format_color_spaces(vec![PresentationFormat {
        format,
        color_space: PresentationColorSpace::Srgb,
    }])
    .with_surface_details(
        TextureUsage::COLOR_ATTACHMENT,
        vec![
            CompositeAlphaMode::Automatic,
            CompositeAlphaMode::Opaque,
            CompositeAlphaMode::PreMultiplied,
        ],
        None,
        Vec::new(),
    )
    .with_timing_and_hdr(PresentationTimingCapabilities { timestamps: false }, None)
}

fn canvas_extent(canvas: &HtmlCanvasElement) -> Extent2d {
    Extent2d {
        width: canvas.width(),
        height: canvas.height(),
    }
}

fn alpha_mode(mode: CompositeAlphaMode) -> RhiResult<&'static str> {
    match mode {
        CompositeAlphaMode::Automatic | CompositeAlphaMode::Opaque => Ok("opaque"),
        CompositeAlphaMode::PreMultiplied => Ok("premultiplied"),
        CompositeAlphaMode::PostMultiplied | CompositeAlphaMode::Inherit => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "WebGPU canvas configuration cannot express this portable alpha-composition mode",
        )),
    }
}

fn field(object: &Object, name: &str, value: JsValue) -> RhiResult<()> {
    Reflect::set(object, &JsValue::from_str(name), &value)
        .map_err(|error| {
            RhiError::new(RhiErrorKind::BackendFailure, js::message(&error))
                .at("WebGPU canvas descriptor")
        })?
        .then_some(())
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "browser rejected a WebGPU canvas descriptor field",
            )
        })
}

fn call0(receiver: &JsValue, name: &'static str) -> RhiResult<JsValue> {
    let function = js::property(receiver, name)
        .map_err(|error| RhiError::new(RhiErrorKind::Unsupported, js::message(&error)).at(name))?
        .dyn_into::<Function>()
        .map_err(|error| RhiError::new(RhiErrorKind::Unsupported, js::message(&error)).at(name))?;
    function
        .call0(receiver)
        .map_err(|error| RhiError::new(RhiErrorKind::BackendFailure, js::message(&error)).at(name))
}

fn call1(receiver: &JsValue, name: &'static str, argument: &JsValue) -> RhiResult<JsValue> {
    let function = js::property(receiver, name)
        .map_err(|error| RhiError::new(RhiErrorKind::Unsupported, js::message(&error)).at(name))?
        .dyn_into::<Function>()
        .map_err(|error| RhiError::new(RhiErrorKind::Unsupported, js::message(&error)).at(name))?;
    function
        .call1(receiver, argument)
        .map_err(|error| RhiError::new(RhiErrorKind::BackendFailure, js::message(&error)).at(name))
}

fn lost_rhi_error(registration: WebGpuRegistration, operation: &'static str) -> RhiError {
    let message = registry::device_loss(registration)
        .map(|info| info.message().to_owned())
        .unwrap_or_else(|| "the WebGPU device was lost".to_owned());
    RhiError::new(RhiErrorKind::DeviceLost, message).at(operation)
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
