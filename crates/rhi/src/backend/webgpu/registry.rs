//! Owner-thread registry for browser handles.
//!
//! The IDs below are intentionally plain copy values.  They are Send/Sync
//! because they contain no JavaScript state; using one off the browser owner
//! thread is rejected at the registry boundary instead of making `JsValue`
//! accidentally Send.  The browser values themselves never leave TLS.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Waker;

use js_sys::Promise;
use wasm_bindgen::JsValue;
use wasm_bindgen::closure::Closure;

use crate::api::platform::{DeviceLossInfo, DeviceStatus};

use super::js;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct WebGpuRegistration(u64);

/// Shared ownership domain for one browser `GPUDevice` generation.
///
/// The `Arc` contains no JavaScript value; it only ensures the owner-thread
/// registry entry outlives every resource, command spine and presentation
/// lease created from the device.  This is the one necessary shared lifetime
/// for a cloneable RHI device generation, not a public browser session/token.
#[derive(Clone)]
pub(super) struct WebGpuDriver {
    inner: Arc<WebGpuDriverInner>,
}

struct WebGpuDriverInner {
    registration: WebGpuRegistration,
}

impl WebGpuDriver {
    pub(super) fn new(registration: WebGpuRegistration) -> Self {
        Self {
            inner: Arc::new(WebGpuDriverInner { registration }),
        }
    }

    pub(super) fn registration(&self) -> WebGpuRegistration {
        self.inner.registration
    }
}

impl Drop for WebGpuDriverInner {
    fn drop(&mut self) {
        super::presentation::remove_canvases(self.registration);
        OWNER.with(|owner| {
            owner.borrow_mut().devices.remove(&self.registration.0);
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct WebGpuRequestId(u64);

/// Opaque reference to one browser resource retained by its owning device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct WebGpuObjectId(u64);

/// Metadata obtained before a device exists.  It is diagnostic input only;
/// capability facts are probed from the selected adapter/device later.
#[derive(Clone, Debug, Default)]
pub(super) struct WebGpuAdapterMetadata {
    pub(super) name: String,
    pub(super) vendor: Option<String>,
    pub(super) architecture: Option<String>,
    pub(super) device: Option<String>,
}

pub(super) struct WebGpuDeviceHandles {
    pub(super) adapter: JsValue,
    pub(super) device: JsValue,
    pub(super) queue: JsValue,
    /// Presentation registrations are populated by the presentation module.
    pub(super) canvas_targets: Vec<JsValue>,
}

struct DeviceEntry {
    handles: WebGpuDeviceHandles,
    status: DeviceStatus,
    loss: Option<DeviceLossInfo>,
    objects: BTreeMap<u64, JsValue>,
    // The closure is retained for exactly as long as the JS device is alive.
    _lost_promise: Promise,
    _lost_ok: Closure<dyn FnMut(JsValue)>,
    _lost_err: Closure<dyn FnMut(JsValue)>,
}

struct RequestEntry {
    _promise: Promise,
    settled: Option<Result<JsValue, String>>,
    /// A browser promise can be observed by more than one RHI future.  In
    /// particular a single `onSubmittedWorkDone` promise fans out to every
    /// completion point in a plan.  Keeping only the last waker loses earlier
    /// waiters and makes completion depend on polling order.
    wakers: Vec<Waker>,
    /// `None` is used for adapter/device requests, which exist before there is
    /// a device generation.  Device-bound promises (mapping and completion)
    /// are terminated when this generation is lost.
    device: Option<WebGpuRegistration>,
    _ok: Closure<dyn FnMut(JsValue)>,
    _err: Closure<dyn FnMut(JsValue)>,
}

struct Registry {
    devices: BTreeMap<u64, DeviceEntry>,
    requests: BTreeMap<u64, RequestEntry>,
}

// TLS maps are per browser owner thread, while this source is process-wide.
// Thus an ID received from another worker cannot accidentally name a local
// object whose per-thread map happened to start at the same integer.
static NEXT_REGISTRY_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_REGISTRY_ID.fetch_add(1, Ordering::Relaxed)
}

thread_local! {
    static OWNER: RefCell<Registry> = RefCell::new(Registry {
        devices: BTreeMap::new(),
        requests: BTreeMap::new(),
    });
}

pub(super) fn start_promise(promise: Promise) -> WebGpuRequestId {
    start_promise_for_device(None, promise)
}

/// Starts a promise whose result is invalid once `registration` is lost.
///
/// The promise is retained only in the owner-thread registry.  This function
/// deliberately accepts the registration rather than a JS device so a native
/// browser value never crosses the Rust RHI boundary.
pub(super) fn start_device_promise(
    registration: WebGpuRegistration,
    promise: Promise,
) -> WebGpuRequestId {
    start_promise_for_device(Some(registration), promise)
}

fn start_promise_for_device(
    device: Option<WebGpuRegistration>,
    promise: Promise,
) -> WebGpuRequestId {
    OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        let id = next_id();
        let success_id = id;
        let ok = Closure::wrap(Box::new(move |value: JsValue| {
            let wakers = OWNER.with(|owner| {
                let mut owner = owner.borrow_mut();
                if let Some(request) = owner.requests.get_mut(&success_id) {
                    // A device-loss transition may have retired this promise
                    // first. Browser settlement afterwards must not resurrect
                    // it or overwrite the terminal reason.
                    if request.settled.is_none() {
                        request.settled = Some(Ok(value));
                    }
                    return std::mem::take(&mut request.wakers);
                }
                Vec::new()
            });
            wake_all(wakers);
        }) as Box<dyn FnMut(JsValue)>);
        let failure_id = id;
        let err = Closure::wrap(Box::new(move |value: JsValue| {
            let wakers = OWNER.with(|owner| {
                let mut owner = owner.borrow_mut();
                if let Some(request) = owner.requests.get_mut(&failure_id) {
                    if request.settled.is_none() {
                        request.settled = Some(Err(js::message(&value)));
                    }
                    return std::mem::take(&mut request.wakers);
                }
                Vec::new()
            });
            wake_all(wakers);
        }) as Box<dyn FnMut(JsValue)>);
        let _ = promise.then2(&ok, &err);
        owner.requests.insert(
            id,
            RequestEntry {
                _promise: promise,
                settled: None,
                wakers: Vec::new(),
                device,
                _ok: ok,
                _err: err,
            },
        );
        WebGpuRequestId(id)
    })
}

pub(super) enum PromisePoll {
    Pending,
    Ready(JsValue),
    Failed(String),
}

pub(super) fn poll_promise(id: WebGpuRequestId, waker: &Waker) -> PromisePoll {
    OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        let Some(request) = owner.requests.get_mut(&id.0) else {
            return PromisePoll::Failed("WebGPU request was retired".into());
        };
        let Some(settled) = request.settled.take() else {
            register_waker(&mut request.wakers, waker);
            return PromisePoll::Pending;
        };
        let _request = owner.requests.remove(&id.0).expect("request checked above");
        match settled {
            Ok(value) => PromisePoll::Ready(value),
            Err(message) => PromisePoll::Failed(message),
        }
    })
}

/// Registers interest without consuming a settled result.
///
/// Composite operations such as queue completion plus readback publication
/// have one public future but several browser promises. Their progress driver
/// owns result consumption; the public future only needs each constituent
/// promise to schedule another poll. If settlement raced this call, wake the
/// supplied task immediately after releasing the registry borrow.
pub(super) fn register_promise_waker(id: WebGpuRequestId, waker: &Waker) {
    let settled = OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        let Some(request) = owner.requests.get_mut(&id.0) else {
            return true;
        };
        if request.settled.is_some() {
            true
        } else {
            register_waker(&mut request.wakers, waker);
            false
        }
    });
    if settled {
        waker.wake_by_ref();
    }
}

/// Detaches an abandoned Rust waiter from a browser Promise.
///
/// JavaScript promises are not cancellable, but retaining their closure pair,
/// result and wakers after the RHI future was dropped would turn a forgotten
/// pipeline compile into a registry leak.  Removing the entry is safe: the
/// callbacks look it up by ID and intentionally do nothing when it is gone.
pub(super) fn retire_promise(id: WebGpuRequestId) {
    let _ = OWNER.with(|owner| owner.borrow_mut().requests.remove(&id.0));
}

#[cfg(test)]
pub(super) fn pending_promise_count() -> usize {
    OWNER.with(|owner| owner.borrow().requests.len())
}

/// Samples a promise without changing its waiter set.
///
/// This is for opportunistic `Device::poll()` progress.  It is intentionally
/// separate from `poll_promise`: a progress probe must never replace or append
/// a synthetic/no-op waker in place of the real futures that are waiting.
pub(super) fn try_take_settled_promise(id: WebGpuRequestId) -> PromisePoll {
    OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        let Some(request) = owner.requests.get_mut(&id.0) else {
            return PromisePoll::Failed("WebGPU request was retired".into());
        };
        let Some(settled) = request.settled.take() else {
            return PromisePoll::Pending;
        };
        let _request = owner.requests.remove(&id.0).expect("request checked above");
        match settled {
            Ok(value) => PromisePoll::Ready(value),
            Err(message) => PromisePoll::Failed(message),
        }
    })
}

fn register_waker(wakers: &mut Vec<Waker>, waker: &Waker) {
    if !wakers.iter().any(|registered| registered.will_wake(waker)) {
        wakers.push(waker.clone());
    }
}

fn wake_all(wakers: Vec<Waker>) {
    // Do not call an arbitrary executor while a `RefCell` borrow is held:
    // browser callbacks are allowed to synchronously re-enter wasm.
    for waker in wakers {
        waker.wake();
    }
}

pub(super) fn register_resolved_device(
    adapter: JsValue,
    device: JsValue,
) -> Result<(WebGpuRegistration, WebGpuAdapterMetadata), String> {
    // A browser call may synchronously re-enter wasm.  Do not retain a RefCell
    // borrow while asking JS for device properties.
    let queue = js::queue(&device).map_err(|error| js::message(&error))?;
    let lost_promise = js::lost(&device).unwrap_or_else(|_| Promise::resolve(&JsValue::NULL));
    let metadata = metadata(&adapter);
    OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        let registration = register_device(&mut owner, adapter, device, queue, lost_promise);
        Ok((registration, metadata))
    })
}

fn metadata(adapter: &JsValue) -> WebGpuAdapterMetadata {
    let info = js::property(adapter, "info").ok();
    let get = |name| {
        info.as_ref()
            .and_then(|value| js::optional_string(value, name))
    };
    WebGpuAdapterMetadata {
        name: get("description").unwrap_or_else(|| "WebGPU adapter".into()),
        vendor: get("vendor"),
        architecture: get("architecture"),
        device: get("device"),
    }
}

fn register_device(
    owner: &mut Registry,
    adapter: JsValue,
    device: JsValue,
    queue: JsValue,
    lost_promise: Promise,
) -> WebGpuRegistration {
    let id = next_id();
    let ok_id = id;
    let ok = Closure::wrap(Box::new(move |value: JsValue| {
        mark_lost(WebGpuRegistration(ok_id), js::message(&value));
    }) as Box<dyn FnMut(JsValue)>);
    let err_id = id;
    let err = Closure::wrap(Box::new(move |value: JsValue| {
        mark_lost(
            WebGpuRegistration(err_id),
            format!("device.lost rejected: {}", js::message(&value)),
        );
    }) as Box<dyn FnMut(JsValue)>);
    let _ = lost_promise.then2(&ok, &err);
    owner.devices.insert(
        id,
        DeviceEntry {
            handles: WebGpuDeviceHandles {
                adapter,
                device,
                queue,
                canvas_targets: Vec::new(),
            },
            status: DeviceStatus::Active,
            loss: None,
            objects: BTreeMap::new(),
            _lost_promise: lost_promise,
            _lost_ok: ok,
            _lost_err: err,
        },
    );
    WebGpuRegistration(id)
}

pub(super) fn mark_lost(registration: WebGpuRegistration, message: String) {
    let wakers = OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        let Some(device) = owner.devices.get_mut(&registration.0) else {
            return Vec::new();
        };
        if device.status == DeviceStatus::Lost {
            return Vec::new();
        }
        device.status = DeviceStatus::Lost;
        let loss = DeviceLossInfo::new(message);
        let terminal = format!("DeviceLost: {}", loss.message());
        device.loss = Some(loss);
        let mut wakers = Vec::new();
        // A lost device cannot later yield mapped bytes or a normal completion.
        // Settle every generation-bound browser promise and wake every waiter.
        for request in owner.requests.values_mut() {
            if request.device == Some(registration) {
                // Even a browser promise that resolved just before this callback
                // has not yet published an RHI completion/mapping result. The
                // v13 execution domain is terminal at loss, so it must not be
                // observed as a successful pending future afterwards.
                request.settled = Some(Err(terminal.clone()));
                wakers.append(&mut request.wakers);
            }
        }
        wakers
    });
    wake_all(wakers);
}

pub(super) fn device_status(registration: WebGpuRegistration) -> Option<DeviceStatus> {
    OWNER.with(|owner| {
        owner
            .borrow()
            .devices
            .get(&registration.0)
            .map(|entry| entry.status)
    })
}

pub(super) fn device_loss(registration: WebGpuRegistration) -> Option<DeviceLossInfo> {
    OWNER.with(|owner| {
        owner
            .borrow()
            .devices
            .get(&registration.0)
            .and_then(|entry| entry.loss.clone())
    })
}

pub(super) fn with_device_handles<T>(
    registration: WebGpuRegistration,
    f: impl FnOnce(&WebGpuDeviceHandles) -> T,
) -> Option<T> {
    OWNER.with(|owner| {
        owner
            .borrow()
            .devices
            .get(&registration.0)
            .map(|entry| f(&entry.handles))
    })
}

/// Retains a JS resource under its device.  A resource wrapper carries only the
/// resulting ID, never the JS object, so it remains safe to move as an opaque
/// RHI handle even though lowering is restricted to the owner thread.
pub(super) fn insert_object(
    registration: WebGpuRegistration,
    value: JsValue,
) -> Option<WebGpuObjectId> {
    OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        let id = next_id();
        let device = owner.devices.get_mut(&registration.0)?;
        device.objects.insert(id, value);
        Some(WebGpuObjectId(id))
    })
}

pub(super) fn with_object<T>(
    registration: WebGpuRegistration,
    object: WebGpuObjectId,
    f: impl FnOnce(&JsValue) -> T,
) -> Option<T> {
    OWNER.with(|owner| {
        owner
            .borrow()
            .devices
            .get(&registration.0)?
            .objects
            .get(&object.0)
            .map(f)
    })
}

pub(super) fn remove_object(
    registration: WebGpuRegistration,
    object: WebGpuObjectId,
) -> Option<JsValue> {
    OWNER.with(|owner| {
        owner
            .borrow_mut()
            .devices
            .get_mut(&registration.0)?
            .objects
            .remove(&object.0)
    })
}
