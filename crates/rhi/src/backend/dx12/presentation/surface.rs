use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Device};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
    DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIFactory4, IDXGISwapChain2,
    IDXGISwapChain3,
};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
use windows::core::Interface;

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
    PresentationTargetCapabilities, PresentationTimingCapabilities,
};
use crate::api::resource::TextureUsage;
use crate::backend::dx12::platform::device::Dx12LossState;

/// Private DXGI target registry. `HWND` crosses only this module's boundary.
pub(crate) struct Dx12Presentation {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    factory: IDXGIFactory4,
    targets: Mutex<HashMap<ObjectId, isize>>,
    // A configured lease can outlive the provider-side presentation facet.  This
    // is the one genuinely shared piece of state: both owners must agree that a
    // target remains leased until the configured lease is dropped.
    leased: Arc<Mutex<HashSet<ObjectId>>>,
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
    loss: Arc<Dx12LossState>,
}

impl Dx12Presentation {
    pub(crate) fn new(
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        loss: Arc<Dx12LossState>,
    ) -> RhiResult<Self> {
        // A debug factory is only requested when the host enabled DXGI debug; requesting
        // it unconditionally makes ordinary retail machines fail before a device exists.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(|error| {
                RhiError::new(RhiErrorKind::BackendFailure, error.to_string())
                    .at("CreateDXGIFactory2")
            })?;
        Ok(Self {
            device,
            queue,
            factory,
            targets: Mutex::new(HashMap::new()),
            leased: Arc::new(Mutex::new(HashSet::new())),
            presents: Arc::new(Mutex::new(HashMap::new())),
            loss,
        })
    }

    /// Host integration entry point. The portable target retains only its ObjectId.
    #[cfg(test)]
    pub(crate) fn register_hwnd_target(&self, hwnd: HWND) -> PresentationTarget {
        let id = ObjectId::next();
        self.targets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, hwnd.0 as isize);
        PresentationTarget::new(id)
    }

    #[cfg(test)]
    pub(crate) fn register_test_hwnd(&self, hwnd: HWND) -> PresentationTarget {
        self.register_hwnd_target(hwnd)
    }

    fn hwnd(&self, target: ObjectId) -> RhiResult<HWND> {
        self.targets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&target)
            .copied()
            .map(|value| HWND(value as *mut _))
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::TargetLost,
                    "the presentation target is not registered with this DX12 device",
                )
            })
    }

    fn capabilities_for(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        let hwnd = self.hwnd(target)?;
        Ok(dx12_capabilities(Some(client_extent(hwnd)?)))
    }
}

impl PresentationBackend for Dx12Presentation {
    fn capabilities(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities> {
        self.capabilities_for(target)
    }

    fn configure(
        &self,
        device: DeviceIdentity,
        target: ObjectId,
        config: &PresentationConfiguration,
    ) -> RhiResult<Box<dyn ConfiguredPresentationBackend>> {
        self.capabilities_for(target)?;
        let mut leased = self.leased.lock().unwrap_or_else(|p| p.into_inner());
        if !leased.insert(target) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "this presentation target already has an active configuration lease",
            ));
        }
        let hwnd = match self.hwnd(target) {
            Ok(hwnd) => hwnd,
            Err(error) => {
                leased.remove(&target);
                return Err(error);
            }
        };
        let created = Dx12ConfiguredPresentation::create(
            self.device.clone(),
            self.factory.clone(),
            self.queue.clone(),
            Arc::clone(&self.leased),
            Arc::clone(&self.presents),
            Arc::clone(&self.loss),
            device,
            target,
            hwnd,
            config,
        );
        match created {
            Ok(lease) => Ok(Box::new(lease)),
            Err(error) => {
                leased.remove(&target);
                Err(error)
            }
        }
    }

    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        self.presents
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&receipt)
            .cloned()
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "unknown DX12 present receipt")
            })
    }

    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        _: &std::task::Waker,
    ) -> RhiResult<PresentState> {
        // DXGI `Present` transfers ownership synchronously. This backend never
        // publishes `Pending`, so there is no native event to retain a waiter
        // for; backends with deferred host presentation implement the registration
        // half of the seam instead of making the public future yield.
        self.present_state(receipt)
    }
}

struct SwapchainState {
    swapchain: IDXGISwapChain3,
    acquired: Option<u64>,
    /// Native back-buffer leases which can still hold an `ID3D12Resource` from
    /// this swapchain generation.
    ///
    /// `ResizeBuffers` has a stronger precondition than "no `AcquiredFrame`":
    /// every reference to an old backbuffer must be released.  A submitted
    /// raster scope deliberately retains its `FrameAttachment` until its fence
    /// completes, and a recorded scope may retain a clone for longer still.
    /// Tracking the backend attachment's final drop turns that DXGI precondition
    /// into a deterministic, retryable RHI refusal instead of relying on a
    /// driver-specific `ResizeBuffers` failure.
    live_backbuffers: HashSet<u64>,
    /// The future waiting for the final portable/native attachment lease to
    /// retire. Registration and the `live_backbuffers` observation share one
    /// lock, which closes the classic "dropped just before waker registration"
    /// lost-wake race.
    // `ConfiguredPresentation::reconfigure` takes `&mut self`, so one lease
    // can have only one live reconfigure future. Keep only that task's latest
    // waker: executors are allowed to replace it across polls, and retaining
    // old ones would turn a long-running resize loop into a waker leak.
    reconfigure_waiter: Option<Waker>,
}

/// One acquired DXGI backbuffer. Kept strictly behind FrameAttachment's private
/// backend seam; public RHI code never observes an ID3D12Resource.
pub(crate) struct Dx12FrameAttachment {
    // Options let `Drop` release the COM references before it makes the
    // corresponding liveness bit observable as clear to a concurrent resize.
    // A direct field would be dropped *after* `Drop::drop` returns, leaving a
    // small but real window where `ResizeBuffers` could race a live reference.
    resource: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    swapchain: Option<IDXGISwapChain3>,
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
    loss: Arc<Dx12LossState>,
    state: Arc<Mutex<SwapchainState>>,
    serial: u64,
}

impl Dx12FrameAttachment {
    pub(crate) fn resource(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Resource {
        // `resource` is taken only by this object's `Drop`; a live shared
        // FrameAttachment can therefore always provide its native backing.
        self.resource
            .as_ref()
            .expect("live DX12 frame attachment has a backbuffer resource")
    }
}

impl FrameAttachmentBackend for Dx12FrameAttachment {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn present(&self, receipt: PresentReceiptId) {
        let state = match self.loss.loss_info() {
            Some(info) => PresentState::DeviceLost(info),
            None => match unsafe {
                self.swapchain
                    .as_ref()
                    .expect("live DX12 frame attachment has its swapchain")
                    .Present(1, windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0))
                    .ok()
            } {
                Ok(()) => PresentState::Accepted,
                Err(error) => match observed_loss(&self.loss, &error, "IDXGISwapChain3::Present") {
                    Some(info) => PresentState::DeviceLost(info),
                    None => PresentState::Failed(crate::api::presentation::PresentFailure::new(
                        error.to_string(),
                    )),
                },
            },
        };
        self.presents
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(receipt, state);
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear_acquired(self.serial);
    }

    fn terminate_present(&self, receipt: PresentReceiptId, state: PresentState) {
        // A Phase-B device loss can happen after an earlier batch was executed
        // but before this relation is reached.  The portable receipt must still
        // have a terminal answer; silently leaving it unknown would strand
        // `wait_present` even though the execution domain is already lost.
        self.presents
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(receipt, state);
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear_acquired(self.serial);
    }
}

impl Drop for Dx12FrameAttachment {
    fn drop(&mut self) {
        // This is the only object that owns the acquired backbuffer COM
        // reference behind the portable `FrameAttachment`.  Clones share its
        // enclosing `FrameAttachmentInner`, therefore this callback runs only
        // after recorded scopes and accepted GPU work stopped retaining it.
        // Drop the resource and swapchain COM references *before* publishing
        // their absence.  Field drops ordinarily happen after this method, so
        // leaving them in place while removing `live_backbuffers` would permit
        // a concurrent `ResizeBuffers` to observe a false precondition.
        drop(self.resource.take());
        drop(self.swapchain.take());
        let waiter = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.live_backbuffers.remove(&self.serial);
            state.clear_acquired(self.serial);
            if state.live_backbuffers.is_empty() {
                state.reconfigure_waiter.take()
            } else {
                None
            }
        };
        if let Some(waker) = waiter {
            waker.wake();
        }
    }
}

impl SwapchainState {
    fn clear_acquired(&mut self, serial: u64) {
        if self.acquired == Some(serial) {
            self.acquired = None;
        }
    }
}

struct Dx12ConfiguredPresentation {
    // The configuration is an independently owned public lease.  Retaining the
    // native device keeps the DXGI objects valid even if the device wrapper is
    // dropped before its frame lease is finished.
    _device: ID3D12Device,
    leased: Arc<Mutex<HashSet<ObjectId>>>,
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
    loss: Arc<Dx12LossState>,
    device: DeviceIdentity,
    target: ObjectId,
    state: Arc<Mutex<SwapchainState>>,
    serial: AtomicU64,
}

impl Dx12ConfiguredPresentation {
    fn create(
        device_native: ID3D12Device,
        factory: IDXGIFactory4,
        queue: ID3D12CommandQueue,
        leased: Arc<Mutex<HashSet<ObjectId>>>,
        presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
        loss: Arc<Dx12LossState>,
        device: DeviceIdentity,
        target: ObjectId,
        hwnd: HWND,
        config: &PresentationConfiguration,
    ) -> RhiResult<Self> {
        let swapchain = create_swapchain(&factory, &queue, hwnd, config, &loss)?;
        let state = Arc::new(Mutex::new(SwapchainState {
            swapchain,
            acquired: None,
            live_backbuffers: HashSet::new(),
            reconfigure_waiter: None,
        }));
        // Loss can be the event that makes a reconfiguration future terminal
        // while a lost queue still retains its old command work.  Do not retain
        // a lease through the device's one-way handler registry: a weak state
        // reference lets a released surface disappear normally.
        let lost_state = Arc::downgrade(&state);
        loss.register_handler(Arc::new(move || {
            let Some(state) = lost_state.upgrade() else {
                return;
            };
            wake_reconfigure_waiters(&state);
        }));
        Ok(Self {
            _device: device_native,
            leased,
            presents,
            loss,
            device,
            target,
            state,
            serial: AtomicU64::new(1),
        })
    }
}

impl ConfiguredPresentationBackend for Dx12ConfiguredPresentation {
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities> {
        // A configured HWND swapchain owns the host-managed size.  Its current
        // `GetDesc1` dimensions are the authoritative answer after resize.
        let desc = unsafe {
            self.state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .swapchain
                .GetDesc1()
        }
        .map_err(|error| native_rhi_error(&self.loss, &error, "IDXGISwapChain3::GetDesc1"))?;
        Ok(dx12_capabilities(Some(Extent2d {
            width: desc.Width,
            height: desc.Height,
        })))
    }
    fn reconfigure_or_register_waker(
        &self,
        config: &PresentationConfiguration,
        waker: &Waker,
    ) -> Poll<RhiResult<()>> {
        if self.loss.loss_info().is_some() {
            return Poll::Ready(Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "the Direct3D 12 device was lost; this presentation lease is terminal",
            )));
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.acquired.is_some() {
            return Poll::Ready(Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot resize a DXGI swapchain while an image is acquired",
            )));
        }
        if !state.live_backbuffers.is_empty() {
            register_reconfigure_waker(&mut state.reconfigure_waiter, waker);
            // The lock makes this test and registration atomic with attachment
            // `Drop`. A drop either sees our registered waker, or completed
            // first and leaves the next poll able to resize immediately.
            if !state.live_backbuffers.is_empty() {
                if let Some(info) = self.loss.loss_info() {
                    return Poll::Ready(Err(RhiError::new(
                        RhiErrorKind::DeviceLost,
                        info.message().to_owned(),
                    )));
                }
                return Poll::Pending;
            }
        }
        let format = match swapchain_format(config) {
            Ok(format) => format,
            Err(error) => return Poll::Ready(Err(error)),
        };
        let (width, height) = swapchain_extent(config);
        let desc = match unsafe { state.swapchain.GetDesc1() }.map_err(|error| {
            native_rhi_error(
                &self.loss,
                &error,
                "IDXGISwapChain3::GetDesc1 before ResizeBuffers",
            )
        }) {
            Ok(desc) => desc,
            Err(error) => return Poll::Ready(Err(error)),
        };
        // A flip-model HWND may have only one associated swapchain.  In
        // particular, do not create a replacement while `state.swapchain` is
        // alive: DXGI rejects that arrangement.  `ResizeBuffers` retains the
        // existing association and changes the format/size in place.  The
        // portable lease has already rejected an outstanding frame, and the
        // `live_backbuffers` guard above additionally proves no portable
        // attachment owner in this lowering still retains a backbuffer. Command
        // submission retirement is represented by that same attachment owner,
        // which `CommittedBatch` keeps alive through its fence.
        let resize = unsafe {
            state.swapchain.ResizeBuffers(
                desc.BufferCount,
                width,
                height,
                format,
                DXGI_SWAP_CHAIN_FLAG(desc.Flags as i32),
            )
        }
        .map_err(|error| native_rhi_error(&self.loss, &error, "IDXGISwapChain3::ResizeBuffers"));
        if let Err(error) = resize {
            return Poll::Ready(Err(error));
        }
        if let Err(error) = configure_frame_latency(&state.swapchain, config, &self.loss) {
            return Poll::Ready(Err(error));
        }
        state.acquired = None;
        Poll::Ready(Ok(()))
    }
    fn try_acquire(
        &self,
        device: DeviceIdentity,
    ) -> Result<Option<AcquiredSurfaceFrame>, AcquireError> {
        if self.loss.loss_info().is_some() {
            return Err(AcquireError::new(
                AcquireErrorKind::DeviceLost,
                "the Direct3D 12 device was lost; this presentation lease is terminal",
            ));
        }
        if device != self.device {
            return Err(AcquireError::new(
                AcquireErrorKind::DeviceLost,
                "the presentation lease belongs to another device",
            ));
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.acquired.is_some() {
            return Err(AcquireError::new(
                AcquireErrorKind::FrameOutstanding,
                "DXGI swapchain image remains acquired",
            ));
        }
        let desc = unsafe { state.swapchain.GetDesc1() }.map_err(|error| {
            acquire_error_from_native(&self.loss, &error, "IDXGISwapChain3::GetDesc1")
        })?;
        if desc.Width == 0 || desc.Height == 0 {
            return Err(AcquireError::new(
                AcquireErrorKind::ZeroSizeOrSuspended,
                "DXGI target has a zero drawable extent",
            ));
        }
        let serial = self.serial.fetch_add(1, Ordering::Relaxed);
        let index = unsafe { state.swapchain.GetCurrentBackBufferIndex() };
        let resource = unsafe {
            state
                .swapchain
                .GetBuffer::<windows::Win32::Graphics::Direct3D12::ID3D12Resource>(index)
        }
        .map_err(|error| {
            acquire_error_from_native(&self.loss, &error, "IDXGISwapChain3::GetBuffer")
        })?;
        state.acquired = Some(serial);
        let inserted = state.live_backbuffers.insert(serial);
        debug_assert!(inserted, "each acquired DXGI frame serial is unique");
        Ok(Some(AcquiredSurfaceFrame {
            serial,
            suboptimal: false,
            extent: Extent2d {
                width: desc.Width,
                height: desc.Height,
            },
            attachment: Box::new(Dx12FrameAttachment {
                resource: Some(resource),
                swapchain: Some(state.swapchain.clone()),
                presents: Arc::clone(&self.presents),
                loss: Arc::clone(&self.loss),
                state: Arc::clone(&self.state),
                serial,
            }),
        }))
    }

    fn acquire_or_register_waker(
        &self,
        device: DeviceIdentity,
        _: &std::task::Waker,
    ) -> std::task::Poll<Result<AcquiredSurfaceFrame, AcquireError>> {
        // DXGI's ordinary flip-model acquisition is immediate once a frame is
        // requested. A future backend using a waitable swapchain object may return
        // Pending here after retaining the waker, without changing the public API.
        std::task::Poll::Ready(self.try_acquire(device).and_then(|frame| {
            frame.ok_or_else(|| {
                AcquireError::new(
                    AcquireErrorKind::NotReady,
                    "DX12 acquisition was not immediately ready",
                )
            })
        }))
    }
    fn abandon(&self, frame: AcquiredFrameId) -> RhiResult<()> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.acquired.take().is_none() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("frame {:?} is not the acquired DXGI image", frame),
            ));
        }
        // Flip-model DXGI has no release-acquired-image call. The frame attachment
        // owns the back-buffer COM reference and is consumed here; command batches
        // that recorded it retain their own reference through fence completion.
        Ok(())
    }
    fn abandon_no_throw(&self, _frame: AcquiredFrameId) {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .acquired = None;
    }
    fn release(&self) {
        self.abandon_no_throw(AcquiredFrameId::new(self.device, 0));
        self.leased
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.target);
    }
}

/// Registers one runtime task without retaining duplicate wakers across its
/// repeated polls.  A future may replace its waker when it migrates executors;
/// `will_wake` keeps one registration per actual wake target.
fn register_reconfigure_waker(slot: &mut Option<Waker>, waker: &Waker) {
    if !slot
        .as_ref()
        .is_some_and(|registered| registered.will_wake(waker))
    {
        *slot = Some(waker.clone());
    }
}

/// Drains waiters outside the swapchain-state lock. A task may synchronously
/// poll and enter `ResizeBuffers` from `wake`, so retaining that lock here would
/// deadlock the exact progress path this callback supplies.
fn wake_reconfigure_waiters(state: &Arc<Mutex<SwapchainState>>) {
    let waiter = {
        let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
        state.reconfigure_waiter.take()
    };
    if let Some(waker) = waiter {
        waker.wake();
    }
}

fn create_swapchain(
    factory: &IDXGIFactory4,
    queue: &ID3D12CommandQueue,
    hwnd: HWND,
    config: &PresentationConfiguration,
    loss: &Dx12LossState,
) -> RhiResult<IDXGISwapChain3> {
    let format = swapchain_format(config)?;
    let (width, height) = swapchain_extent(config);
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: format,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 2,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        // The waitable-object flag is required before DXGI will accept the
        // portable maximum-frame-latency request.  We do not expose that object
        // or use it as a public synchronization primitive; it merely makes the
        // requested queue bound a real native configuration.
        Flags: DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0 as u32,
    };
    let chain = unsafe { factory.CreateSwapChainForHwnd(queue, hwnd, &desc, None, None) }
        .map_err(|error| native_rhi_error(loss, &error, "IDXGIFactory4::CreateSwapChainForHwnd"))?;
    let chain: IDXGISwapChain3 = chain
        .cast()
        .map_err(|error| native_rhi_error(loss, &error, "IDXGISwapChain1::cast"))?;
    configure_frame_latency(&chain, config, loss)?;
    Ok(chain)
}

/// Facts this lowering can actually honour.  This HWND flip-model path has no
/// alpha-composited swapchain, mutable-format view list, HDR colour-space setup,
/// or presentation timestamp source, so those are deliberately absent rather
/// than guessed from DXGI defaults.
fn dx12_capabilities(current: Option<Extent2d>) -> PresentationTargetCapabilities {
    PresentationTargetCapabilities::new(
        vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm],
        // Keep this in lockstep with `Dx12FrameAttachment::present`, which uses
        // `Present(1, 0)`. Immediate is never silently substituted.
        vec![PresentMode::Fifo],
        PresentationExtentControl::HostManaged { current },
    )
    .with_format_color_spaces(vec![
        PresentationFormat {
            format: TextureFormat::Bgra8Unorm,
            color_space: PresentationColorSpace::Srgb,
        },
        PresentationFormat {
            format: TextureFormat::Rgba8Unorm,
            color_space: PresentationColorSpace::Srgb,
        },
    ])
    .with_surface_details(
        TextureUsage::COLOR_ATTACHMENT,
        vec![CompositeAlphaMode::Automatic, CompositeAlphaMode::Opaque],
        Some(FrameLatencyRange { min: 1, max: 16 }),
        Vec::new(),
    )
    .with_timing_and_hdr(PresentationTimingCapabilities { timestamps: false }, None)
}

fn client_extent(hwnd: HWND) -> RhiResult<Extent2d> {
    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }.map_err(|_error| {
        // A client rectangle is a host-target fact, not a device failure.  Keep
        // the portable boundary structured and do not leak a Win32 error value.
        RhiError::new(
            RhiErrorKind::TargetLost,
            "the registered DX12 presentation target no longer has a client rectangle",
        )
        .at("GetClientRect")
    })?;
    Ok(Extent2d {
        width: u32::try_from(rect.right.saturating_sub(rect.left)).unwrap_or(0),
        height: u32::try_from(rect.bottom.saturating_sub(rect.top)).unwrap_or(0),
    })
}

fn configure_frame_latency(
    swapchain: &IDXGISwapChain3,
    config: &PresentationConfiguration,
    loss: &Dx12LossState,
) -> RhiResult<()> {
    let swapchain: IDXGISwapChain2 = swapchain.cast().map_err(|error| {
        native_rhi_error(loss, &error, "IDXGISwapChain3::cast(IDXGISwapChain2)")
    })?;
    unsafe {
        swapchain
            .SetMaximumFrameLatency(config.maximum_frame_latency())
            .map_err(|error| {
                native_rhi_error(loss, &error, "IDXGISwapChain2::SetMaximumFrameLatency")
            })
    }
}

fn swapchain_format(
    config: &PresentationConfiguration,
) -> RhiResult<windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT> {
    match config.format() {
        TextureFormat::Bgra8Unorm => Ok(DXGI_FORMAT_B8G8R8A8_UNORM),
        TextureFormat::Rgba8Unorm => Ok(DXGI_FORMAT_R8G8B8A8_UNORM),
        _ => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "DXGI swapchains support only the queried 8-bit presentation formats",
            ));
        }
    }
}

fn swapchain_extent(config: &PresentationConfiguration) -> (u32, u32) {
    match config.extent() {
        PresentationExtent::Exact(extent) => (extent.width, extent.height),
        PresentationExtent::HostManaged => (0, 0),
    }
}

/// Observes a native HRESULT at the presentation boundary before exposing it.
///
/// DXGI has no separate loss callback: `Present`, `GetBuffer`, or
/// `ResizeBuffers` may be the first call to see a removed/reset/hung device.
/// The loss cell is shared with command submission, so recording it here also
/// wakes its pending completion/readback futures. A later observer keeps the
/// first diagnosis stable rather than overwriting it with a secondary error.
fn observed_loss(
    loss: &Dx12LossState,
    error: &windows::core::Error,
    operation: &'static str,
) -> Option<DeviceLossInfo> {
    let native = crate::backend::dx12::ffi::NativeError::new(error, operation);
    if native.failure().is_terminal() {
        loss.mark_lost(DeviceLossInfo::new(format!(
            "Direct3D 12 reported a terminal failure in {operation}: {}",
            native.as_error().message()
        )));
    }
    loss.loss_info()
}

fn native_rhi_error(
    loss: &Dx12LossState,
    error: &windows::core::Error,
    operation: &'static str,
) -> RhiError {
    match observed_loss(loss, error, operation) {
        Some(info) => {
            RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned()).at(operation)
        }
        None => RhiError::new(RhiErrorKind::BackendFailure, error.to_string()).at(operation),
    }
}

fn acquire_error_from_native(
    loss: &Dx12LossState,
    error: &windows::core::Error,
    operation: &'static str,
) -> AcquireError {
    match observed_loss(loss, error, operation) {
        Some(info) => AcquireError::new(AcquireErrorKind::DeviceLost, info.message().to_owned()),
        None => AcquireError::new(AcquireErrorKind::TargetLost, error.to_string()),
    }
}
