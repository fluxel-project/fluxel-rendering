//! Native Vulkan Win32 presentation conformance evidence.
//!
//! This is deliberately a real host-owned `HWND`, WSI surface, swapchain, raster
//! clear and present path.  It does not substitute an off-screen image or a
//! mock presentation backend: those paths cannot prove the ownership and
//! semaphore hand-off that `VK_KHR_swapchain` requires.
//!
//! The host window is the same public raw-window-handle boundary applications
//! use.  The Vulkan implementation itself has no host dependency; this is
//! solely test-host window creation plumbing.

#![cfg(all(windows, feature = "vulkan"))]

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::pin::pin;

use fluxel_host::{Window as HostWindow, WindowConfig};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use crate::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
    RecorderDescriptor, StoreOp,
};
use crate::api::error::RhiErrorKind;
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, PlatformProvider};
use crate::api::presentation::{
    Extent2d, PresentMode, PresentState, PresentationConfiguration, PresentationExtent,
    PresentationExtentControl,
};
use crate::api::submission::{
    CompletionState, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder,
};

use super::platform::VulkanProvider;

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn raster_lane(device: &crate::api::platform::Device) -> SubmissionLaneId {
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(LaneWorkDomains::RASTER))
        .map(|lane| lane.id())
        .expect("the advertised Vulkan graphics lane supports raster work")
}

fn presentation_config(
    capabilities: &crate::api::presentation::PresentationTargetCapabilities,
) -> PresentationConfiguration {
    let format = *capabilities
        .formats()
        .first()
        .expect("a presentable Vulkan target exposes one portable format");
    let config = PresentationConfiguration::new(format).with_present_mode(PresentMode::Fifo);
    match capabilities.extent_control() {
        PresentationExtentControl::HostManaged { .. } => config,
        PresentationExtentControl::Configurable { min, max } => {
            // A hidden test window nevertheless owns a non-zero client area.
            // Clamp the desired test extent to the surface's live legal range.
            config.with_extent(PresentationExtent::Exact(Extent2d {
                width: 64u32.clamp(min.width, max.width),
                height: 64u32.clamp(min.height, max.height),
            }))
        }
    }
}

fn clear_frame_work(
    device: &crate::api::platform::Device,
    frame: &crate::api::presentation::AcquiredFrame,
) -> crate::api::command::RecordedWork {
    let scope = RasterScopeDescriptor::new().with_color(
        crate::api::shader::ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Frame(frame.attachment()),
            load: LoadOp::Clear(ColorClearValue::Float([0.05, 0.2, 0.4, 1.0])),
            store: StoreOp::Store,
            resolve: None,
            depth_slice: None,
        },
    );
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("Vulkan presentation recorder");
    recorder
        .begin_raster(&scope)
        .expect("a configured frame is a final color attachment")
        .end()
        .expect("Vulkan presentation raster end");
    recorder.finish().expect("Vulkan presentation work")
}

#[test]
fn host_win32_surface_acquires_clears_presents_reconfigures_and_abandons() {
    // Some ICDs defer their desktop-presentable surface allocation until the
    // HWND has a mapped client area.  A hand-rolled hidden `WS_OVERLAPPED`
    // window can therefore make `vkCreateSwapchainKHR` fail with the opaque
    // `VK_ERROR_UNKNOWN`, despite a successful surface/support preflight.  Use
    // Fluxel's real host primitive: it creates a shown `WS_OVERLAPPEDWINDOW`
    // with an explicitly non-zero *client* extent and is the boundary clients
    // actually use for a Win32 presentation target.
    let window = HostWindow::new(
        WindowConfig::new("Fluxel Vulkan presentation", 64, 64).expect("valid host config"),
    )
    .expect("host presentation window");
    let raw = window.window_handle().expect("live host HWND").as_raw();
    let RawWindowHandle::Win32(raw) = raw else {
        panic!("Windows host must expose a Win32 raw handle");
    };
    let hinstance = raw
        .hinstance
        .expect("the Win32 host handle retains its module instance");
    let identity = DeviceInstanceId::new(0xA571_0001);
    let native = VulkanProvider::new(identity).expect("Vulkan loader available");
    let target = native
        .register_win32_presentation_target(raw.hwnd.get() as usize, hinstance.get() as usize);
    let provider = PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native));
    let adapters = block_on(provider.enumerate_adapters())
        .expect("Vulkan adapter enumeration")
        .expect("native Vulkan provider lists adapters");
    let Some(adapter) = adapters.iter().find(|adapter| {
        provider
            .supports_presentation(adapter.id(), &target)
            .unwrap_or(false)
    }) else {
        // A loader can be present without a physical device/queue route to this
        // desktop session (for example, a headless CI service).  That is not a
        // false claim of Vulkan WSI support, so this real-hardware evidence has
        // nothing to execute there.
        return;
    };
    let descriptor = DeviceRequestDescriptor::new(
        AdapterSelection::Explicit(adapter.id()),
        DeviceRequirements::new(),
    )
    .require_presentation_target(target.clone());
    let device = block_on(provider.request_device(descriptor))
        .expect("the preflighted Vulkan presentation route creates a device");
    let capabilities = device
        .presentation_capabilities(&target)
        .expect("Vulkan surface facts");
    assert!(capabilities.formats().len() > 0);
    assert!(
        capabilities.present_modes().contains(&PresentMode::Fifo),
        "Vulkan FIFO is the baseline present mode required by this WSI lowering"
    );
    let config = presentation_config(&capabilities);
    let mut surface = block_on(device.configure_presentation(&target, &config))
        .expect("configure Vulkan swapchain");

    // Reconfigure is rejected portably before native swapchain destruction while
    // an image is owned by the frame token.  Abandon then proves that ownership
    // can be terminated without a plan/present and the lease becomes usable.
    let outstanding = block_on(surface.acquire()).expect("first Vulkan acquire");
    let error =
        block_on(surface.reconfigure(&config)).expect_err("outstanding frame refuses reconfigure");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    block_on(outstanding.abandon()).expect("Vulkan frame abandonment");

    let frame = block_on(surface.acquire()).expect("Vulkan frame acquire for clear/present");
    let work = clear_frame_work(&device, &frame);
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(raster_lane(&device), vec![work])
        .expect("presentation raster batch");
    plan.present_after(frame, point)
        .expect("presentation is ordered after its raster use");
    let receipt = block_on(device.submit(plan.build().expect("presentation plan")))
        .expect("Vulkan clear and present submit");
    assert_eq!(receipt.presents().len(), 1);
    assert!(matches!(
        block_on(device.wait_present(receipt.presents()[0].id())),
        Ok(PresentState::Accepted)
    ));
    assert!(matches!(
        block_on(device.wait_completion(receipt.completion())),
        Ok(CompletionState::Complete)
    ));

    // A completed present releases the single outstanding frame claim, so
    // reacquire, reconfigure and abandon must all remain legal on one lease.
    let frame = block_on(surface.acquire()).expect("acquire after accepted present");
    block_on(frame.abandon()).expect("abandon after reacquire");
    block_on(surface.reconfigure(&config)).expect("reconfigure after frame termination");
    let frame = block_on(surface.acquire()).expect("acquire after reconfigure");
    block_on(frame.abandon()).expect("abandon reconfigured frame");
}
