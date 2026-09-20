use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Device};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIFactory4, IDXGISwapChain3,
};
use windows::core::Interface;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::presentation::backend::{
    AcquiredSurfaceFrame, ConfiguredPresentationBackend, FrameAttachmentBackend,
    PresentationBackend,
};
use crate::api::presentation::{
    AcquireError, AcquireErrorKind, AcquiredFrameId, Extent2d, PresentMode, PresentReceiptId,
    PresentState, PresentationConfiguration, PresentationExtent, PresentationExtentControl,
    PresentationTarget, PresentationTargetCapabilities,
};
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
        let _ = self.hwnd(target)?;
        Ok(PresentationTargetCapabilities::new(
            vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm],
            vec![PresentMode::Fifo, PresentMode::Immediate],
            PresentationExtentControl::HostManaged { current: None },
        ))
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
}

/// One acquired DXGI backbuffer. Kept strictly behind FrameAttachment's private
/// backend seam; public RHI code never observes an ID3D12Resource.
pub(crate) struct Dx12FrameAttachment {
    resource: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    swapchain: IDXGISwapChain3,
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
    loss: Arc<Dx12LossState>,
    state: Arc<Mutex<SwapchainState>>,
}

impl Dx12FrameAttachment {
    pub(crate) fn resource(&self) -> &windows::Win32::Graphics::Direct3D12::ID3D12Resource {
        &self.resource
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
                    .Present(1, windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0))
                    .ok()
            } {
                Ok(()) => PresentState::Accepted,
                Err(error) => PresentState::Failed(crate::api::presentation::PresentFailure::new(
                    error.to_string(),
                )),
            },
        };
        self.presents
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(receipt, state);
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .acquired = None;
    }
}

struct Dx12ConfiguredPresentation {
    // The configuration is an independently owned public lease.  Retaining the
    // native device keeps the DXGI objects valid even if the device wrapper is
    // dropped before its frame lease is finished.
    _device: ID3D12Device,
    factory: IDXGIFactory4,
    queue: ID3D12CommandQueue,
    leased: Arc<Mutex<HashSet<ObjectId>>>,
    presents: Arc<Mutex<HashMap<PresentReceiptId, PresentState>>>,
    loss: Arc<Dx12LossState>,
    device: DeviceIdentity,
    target: ObjectId,
    hwnd: isize,
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
        let swapchain = create_swapchain(&factory, &queue, hwnd, config)?;
        Ok(Self {
            _device: device_native,
            factory,
            queue,
            leased,
            presents,
            loss,
            device,
            target,
            hwnd: hwnd.0 as isize,
            state: Arc::new(Mutex::new(SwapchainState {
                swapchain,
                acquired: None,
            })),
            serial: AtomicU64::new(1),
        })
    }
}

impl ConfiguredPresentationBackend for Dx12ConfiguredPresentation {
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities> {
        Ok(PresentationTargetCapabilities::new(
            vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm],
            vec![PresentMode::Fifo, PresentMode::Immediate],
            PresentationExtentControl::HostManaged { current: None },
        ))
    }
    fn reconfigure(&self, config: &PresentationConfiguration) -> RhiResult<()> {
        if self.loss.loss_info().is_some() {
            return Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "the Direct3D 12 device was lost; this presentation lease is terminal",
            ));
        }
        let fresh = create_swapchain(
            &self.factory,
            &self.queue,
            HWND(self.hwnd as *mut _),
            config,
        )?;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.swapchain = fresh;
        state.acquired = None;
        Ok(())
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
        let desc = unsafe { state.swapchain.GetDesc1() }
            .map_err(|e| AcquireError::new(AcquireErrorKind::TargetLost, e.to_string()))?;
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
        .map_err(|e| AcquireError::new(AcquireErrorKind::TargetLost, e.to_string()))?;
        state.acquired = Some(serial);
        Ok(Some(AcquiredSurfaceFrame {
            serial,
            extent: Extent2d {
                width: desc.Width,
                height: desc.Height,
            },
            attachment: Box::new(Dx12FrameAttachment {
                resource,
                swapchain: state.swapchain.clone(),
                presents: Arc::clone(&self.presents),
                loss: Arc::clone(&self.loss),
                state: Arc::clone(&self.state),
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
        // Flip-model DXGI has no release-acquired-image call. Marking it free is safe
        // because no native resource leaves this backend until raster lowering exists.
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

fn create_swapchain(
    factory: &IDXGIFactory4,
    queue: &ID3D12CommandQueue,
    hwnd: HWND,
    config: &PresentationConfiguration,
) -> RhiResult<IDXGISwapChain3> {
    let format = match config.format() {
        TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        _ => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "DXGI swapchains support only the queried 8-bit presentation formats",
            ));
        }
    };
    let (width, height) = match config.extent() {
        PresentationExtent::Exact(extent) => (extent.width, extent.height),
        PresentationExtent::HostManaged => (0, 0),
    };
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
        Flags: 0,
    };
    let chain =
        unsafe { factory.CreateSwapChainForHwnd(queue, hwnd, &desc, None, None) }.map_err(|e| {
            RhiError::new(RhiErrorKind::BackendFailure, e.to_string())
                .at("IDXGIFactory4::CreateSwapChainForHwnd")
        })?;
    chain.cast().map_err(|e| {
        RhiError::new(RhiErrorKind::BackendFailure, e.to_string()).at("IDXGISwapChain1::cast")
    })
}
