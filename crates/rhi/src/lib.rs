//! Fluxel RHI — the portable GPU execution contract.
//!
//! This crate is the boundary between Fluxel's engine layers and the GPU APIs
//! beneath them. It exposes one portable vocabulary for device ownership,
//! resources, shaders, recording, submission, completion, and presentation, and
//! it owns the rules that decide whether a portable operation is legal without
//! asking a driver.
//!
//! # What this crate is
//!
//! ```text
//! portable GPU execution vocabulary
//! + instance capability facts
//! + strict validation
//! + opaque logical handles
//! + explicit hazard/dependency validation
//! + logical submission/completion/presentation
//! + portable logical statistics / inventory
//! ```
//!
//! # What this crate is not
//!
//! ```text
//! a wrapper over one native API
//! the greatest common denominator of all platforms
//! a native-handle escape hatch
//! ```
//!
//! It is capability-layered rather than levelled down: a backend contributes the
//! facts about what it can do, and the portable vocabulary stays whole. A caller
//! therefore never learns whether it is on DX12, Vulkan, Metal, WebGPU, or the
//! GL family in order to write correct code — but a caller that needs a
//! capability asks the device for the fact rather than probing for a type.
//!
//! # Layering
//!
//! ```text
//! Renderer / material graph / custom pipeline policy
//!     -> RenderGraph declaration and object recipes
//!     -> GraphExecutionPlan
//!     -> RHI RecordedWork + SubmissionPlan
//!     -> backend-private lowering
//!     -> DX12 | Vulkan | Metal | WebGPU | GL family
//! ```
//!
//! RenderGraph owns declarations, versions, dependencies, culling, scheduling,
//! logical lifetime, and presentation intent. This crate owns portable
//! execution, device validation, submission, completion, presentation,
//! retirement, logical observation, and backend lowering. Keeping those apart is
//! what stops the graph from becoming a second source of truth about hazards.
//!
//! # Where the rules live
//!
//! Architecture decisions live under `documents/adr/`; the compact map is
//! `crates/rhi/documents/design-rhi.md`. The public API is specified by this
//! crate's rustdoc and contract tests, not by a duplicate prose specification.
//!
//! # Status
//!
//! The crate is built contract-first. [`api`] holds the public surface, written
//! from the specification with validation and refusal paths fixed before native
//! lowering is admitted. An unavailable lowering must fail structurally before
//! native work is accepted; reachable `todo!()` / `unimplemented!()` paths and
//! dummy success are not valid capability implementations. Backends land under
//! `backend/`; portable defaults and crate-private implementation contracts live
//! beside the public vocabulary in the corresponding `api/` domain.

#![deny(missing_docs)]

pub mod api;

/// Fixture-only hardware evidence reports. This is intentionally not part of
/// the portable RHI vocabulary and exists only when a harness opts in.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support;

// Native lowering, one module per backend. Crate-private for the same reason,
// and feature- and target-gated because a backend that is not being built must
// contribute no code at all.
pub(crate) mod backend;

// This sequence exists only on targets on which a public provider composition
// entry point is available.  Keeping it under the same gate prevents a
// no-backend build from acquiring a misleading, unused process-global state.
#[cfg(any(
    all(feature = "dx12", windows),
    all(feature = "vulkan", not(target_arch = "wasm32")),
    all(feature = "metal", target_vendor = "apple"),
    all(feature = "webgpu", target_arch = "wasm32")
))]
static NEXT_PUBLIC_PROVIDER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0xF1_0000_0000_0000);

/// Opens the native DX12 provider owned by this process.
///
/// This is the host/platform composition seam used by applications, tests and
/// examples. It returns only the portable [`api::platform::PlatformProvider`];
/// DXGI factories, adapters and native devices remain backend-private. A host
/// still supplies any presentation target separately through the presentation
/// API. The function is available only when the DX12 backend is compiled on
/// Windows.
#[cfg(all(feature = "dx12", windows))]
pub fn create_dx12_provider() -> api::error::RhiResult<api::platform::PlatformProvider> {
    use api::identity::DeviceInstanceId;
    use api::platform::provider::{BackendKind, PlatformProvider};
    let instance = DeviceInstanceId::new(
        NEXT_PUBLIC_PROVIDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    );
    let native = backend::dx12::Dx12Provider::new(instance)?;
    Ok(PlatformProvider::new(
        BackendKind::Dx12,
        instance,
        Box::new(native),
    ))
}

/// Opens the native Vulkan provider owned by this process.
///
/// Like [`create_dx12_provider`], this is a composition entry point rather
/// than a native-handle escape hatch. Vulkan instance/device/queue objects stay
/// below the backend seam and callers interact with the returned provider only
/// through the portable API. The provider loads Vulkan dynamically.
#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
pub fn create_vulkan_provider() -> api::error::RhiResult<api::platform::PlatformProvider> {
    use api::identity::DeviceInstanceId;
    use api::platform::provider::{BackendKind, PlatformProvider};
    let instance = DeviceInstanceId::new(
        NEXT_PUBLIC_PROVIDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    );
    let native = backend::vulkan::VulkanProvider::new(instance)?;
    Ok(PlatformProvider::new(
        BackendKind::Vulkan,
        instance,
        Box::new(native),
    ))
}

/// Opens the native Metal provider on an Apple target.
///
/// Surface/layer registration remains a host concern; this function only
/// creates the provider used for adapter/device discovery and returns the same
/// portable handle as the other native composition entry points.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub fn create_metal_provider() -> api::error::RhiResult<api::platform::PlatformProvider> {
    use api::identity::DeviceInstanceId;
    use api::platform::provider::{BackendKind, PlatformProvider};
    let instance = DeviceInstanceId::new(
        NEXT_PUBLIC_PROVIDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    );
    let native = backend::metal::MetalProvider::new(instance);
    Ok(PlatformProvider::new(
        BackendKind::Metal,
        instance,
        Box::new(native),
    ))
}

/// Opens the browser WebGPU provider from the browser's `navigator.gpu` entry
/// point.
///
/// Adapter selection and device creation remain asynchronous browser futures;
/// a browser host supplies a canvas only later when it configures presentation.
/// This function creates only the portable provider handle and does not expose
/// a `GPU`/`GPUDevice` value.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub fn create_webgpu_provider() -> api::error::RhiResult<api::platform::PlatformProvider> {
    use api::identity::DeviceInstanceId;
    use api::platform::provider::{BackendKind, PlatformProvider};
    let instance = DeviceInstanceId::new(
        NEXT_PUBLIC_PROVIDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    );
    let native = backend::webgpu::WebGpuProvider::new(instance);
    Ok(PlatformProvider::new(
        BackendKind::WebGpu,
        instance,
        Box::new(native),
    ))
}

/// Android NativeActivity-only Vulkan WSI integration bridge.
///
/// This is intentionally hidden from the portable RHI contract: it accepts an
/// `ANativeWindow` owned by Android host glue and returns only JSON evidence.
#[cfg(all(feature = "android-wsi-evidence", target_os = "android"))]
#[doc(hidden)]
pub mod android_vulkan_wsi {
    use crate::api::command::{
        ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
        RecorderDescriptor, StoreOp,
    };
    use crate::api::identity::DeviceInstanceId;
    use crate::api::platform::provider::AdapterSelection;
    use crate::api::platform::request::DeviceRequestDescriptor;
    use crate::api::platform::requirements::DeviceRequirements;
    use crate::api::platform::{BackendKind, PlatformProvider};
    use crate::api::presentation::{
        Extent2d, PresentMode, PresentationConfiguration, PresentationExtent,
        PresentationExtentControl,
    };
    use crate::api::submission::{LaneWorkDomains, SubmissionPlanBuilder};
    use crate::backend::vulkan::platform::VulkanProvider;
    use core::future::Future;
    use core::task::{Context, Poll, Waker};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::pin::pin;

    thread_local! {
        // NativeActivity invokes both window callbacks on its main thread. The
        // retained registry entry is therefore thread-local rather than a new
        // cross-thread ownership layer in the portable RHI.
        static ANDROID_TARGETS: RefCell<HashMap<usize, crate::backend::vulkan::presentation::android::AndroidTargetRegistration>> = RefCell::new(HashMap::new());
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    /// Runs the Android WSI evidence workload for one host-owned native window.
    pub fn run(window: *mut core::ffi::c_void) -> Result<String, String> {
        let identity = DeviceInstanceId::new(0xA11D_0001);
        let native = VulkanProvider::new(identity).map_err(|e| e.to_string())?;
        let target = native
            .register_android_presentation_target(window)
            .map_err(|e| e.to_string())?;
        let registration = native.retain_android_presentation_target(target.id());
        let provider = PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native));
        let adapters = block_on(provider.enumerate_adapters())
            .map_err(|e| e.to_string())?
            .ok_or("no Vulkan adapters")?;
        let adapter = adapters
            .iter()
            .find(|a| {
                provider
                    .supports_presentation(a.id(), &target)
                    .unwrap_or(false)
            })
            .ok_or("no Android-presentable Vulkan adapter")?;
        let request = DeviceRequestDescriptor::new(
            AdapterSelection::Explicit(adapter.id()),
            DeviceRequirements::new(),
        )
        .require_presentation_target(target.clone());
        let device = block_on(provider.request_device(request)).map_err(|e| e.to_string())?;
        let caps = device
            .presentation_capabilities(&target)
            .map_err(|e| e.to_string())?;
        let mut config =
            PresentationConfiguration::new(*caps.formats().first().ok_or("no present format")?)
                .with_present_mode(PresentMode::Fifo);
        if let PresentationExtentControl::Configurable { min, max } = caps.extent_control() {
            config = config.with_extent(PresentationExtent::Exact(Extent2d {
                width: 64u32.clamp(min.width, max.width),
                height: 64u32.clamp(min.height, max.height),
            }));
        }
        let mut surface =
            block_on(device.configure_presentation(&target, &config)).map_err(|e| e.to_string())?;
        let frame = block_on(surface.acquire()).map_err(|e| e.to_string())?;
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
            .map_err(|e| e.to_string())?;
        recorder
            .begin_raster(&scope)
            .map_err(|e| e.to_string())?
            .end()
            .map_err(|e| e.to_string())?;
        let work = recorder.finish().map_err(|e| e.to_string())?;
        let lane = device
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|l| l.domains().contains(LaneWorkDomains::RASTER))
            .ok_or("no raster lane")?
            .id();
        let mut plan = SubmissionPlanBuilder::new(&device);
        let point = plan
            .add_batch(lane, vec![work])
            .map_err(|e| e.to_string())?;
        plan.present_after(frame, point)
            .map_err(|e| e.to_string())?;
        let receipt = block_on(device.submit(plan.build().map_err(|e| e.to_string())?))
            .map_err(|e| e.to_string())?;
        let present =
            block_on(device.wait_present(receipt.presents()[0].id())).map_err(|e| e.to_string())?;
        let again = block_on(surface.acquire()).map_err(|e| e.to_string())?;
        block_on(again.abandon()).map_err(|e| e.to_string())?;
        block_on(surface.reconfigure(&config)).map_err(|e| e.to_string())?;
        let final_frame = block_on(surface.acquire()).map_err(|e| e.to_string())?;
        block_on(final_frame.abandon()).map_err(|e| e.to_string())?;
        let evidence = format!(
            "{{\"schema\":\"fluxel.android-vulkan-wsi.v1\",\"acquire\":true,\"raster_clear\":true,\"present\":\"{present:?}\",\"reacquire\":true,\"reconfigure\":true}}"
        );
        // Replace only an old registration for the same native window. The
        // previous guard unregisters before the new one takes ownership.
        ANDROID_TARGETS.with(|targets| {
            targets.borrow_mut().insert(window as usize, registration);
        });
        Ok(evidence)
    }

    /// Called from `ANativeActivityCallbacks::onNativeWindowDestroyed`.
    /// Dropping the registration marks its portable target terminal and pairs
    /// Fluxel's `ANativeWindow_acquire` with `ANativeWindow_release`.
    pub fn on_native_window_destroyed(window: *mut core::ffi::c_void) {
        ANDROID_TARGETS.with(|targets| {
            targets.borrow_mut().remove(&(window as usize));
        });
    }
}
