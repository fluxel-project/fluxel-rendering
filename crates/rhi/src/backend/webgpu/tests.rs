//! Live headed-browser smoke tests for the WebGPU provider.
//!
//! These deliberately prove the bootstrap boundary and portable async request
//! path against a real browser adapter.  Detailed resource and command cases
//! live with the lowering that owns their fixtures.  A software adapter is
//! rejected instead of being allowed to count as GPU evidence.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use js_sys::Reflect;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

use crate::api::format::{TextureFormat, TextureSupportQuery};
use crate::api::identity::DeviceInstanceId;
use crate::api::pipeline::{ComputePipelineDescriptor, PipelineInterfaceDescriptor};
use crate::api::platform::{
    AdapterSelection, BackendKind, DeviceRequestDescriptor, DeviceRequirements, DeviceStatus,
    PlatformProvider,
};
use crate::api::resource::{BufferDescriptor, BufferUsage};
use crate::api::resource::{
    TextureAspects, TextureDescriptor, TextureUsage, TextureViewDescriptor, TextureViewDimension,
};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ComputeWorkgroupSize, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderRequirements, ShaderStage,
};

use super::{WebGpuProvider, js, registry};

wasm_bindgen_test_configure!(run_in_browser);

fn compute_artifact(source: &'static str, hash_byte: u8) -> ShaderArtifact {
    ShaderArtifact::new(
        ShaderStage::Compute,
        "main",
        ShaderCode::Wgsl(Arc::from(source)),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new().with_compute_workgroup_size(ComputeWorkgroupSize::new(1, 1, 1)),
        ShaderRequirements::new(),
        ArtifactHash([hash_byte; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

/// A headed evidence invocation must have selected a hardware WebGPU adapter.
///
/// Chrome redacts adapter diagnostics unless launched with
/// `--enable-unsafe-webgpu`; the release command supplies that flag.  Requiring
/// both a non-fallback adapter and non-redacted diagnostic evidence keeps a
/// SwiftShader/WARP/llvmpipe run from quietly passing as real-GPU evidence.
#[wasm_bindgen_test(async)]
async fn webgpu_adapter_is_not_a_software_fallback() {
    let gpu = js::browser_gpu().expect("navigator.gpu must exist in the Chrome WebGPU run");
    let adapter = JsFuture::from(js::request_adapter(&gpu, None).expect("requestAdapter call"))
        .await
        .expect("requestAdapter promise");
    assert!(
        !adapter.is_null() && !adapter.is_undefined(),
        "browser found no WebGPU adapter"
    );

    // `isFallbackAdapter` is optional in the browser versions we support.
    // When Chrome exposes it, a true answer is immediately disqualifying; when
    // it does not, the non-redacted adapter diagnostics below remain the
    // evidence gate and reject the known software implementations by name.
    if let Some(fallback) = Reflect::get(&adapter, &JsValue::from_str("isFallbackAdapter"))
        .ok()
        .and_then(|value| value.as_bool())
    {
        assert!(
            !fallback,
            "Chrome selected a fallback WebGPU adapter, not a real GPU"
        );
    }

    let info = Reflect::get(&adapter, &JsValue::from_str("info")).expect("adapter.info");
    let description = ["vendor", "architecture", "device", "description"]
        .into_iter()
        .filter_map(|field| Reflect::get(&info, &JsValue::from_str(field)).ok())
        .filter_map(|value| value.as_string())
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" / ");
    assert!(
        !description.is_empty(),
        "adapter diagnostics are redacted; launch Chrome with --enable-unsafe-webgpu"
    );
    let lower = description.to_ascii_lowercase();
    for forbidden in [
        "swiftshader",
        "llvmpipe",
        "software rasterizer",
        "microsoft basic render driver",
        "warp",
    ] {
        assert!(
            !lower.contains(forbidden),
            "software adapter {description:?} is not valid real-GPU evidence"
        );
    }
}

/// Portable device creation must resolve its browser Promises and return a
/// healthy WebGPU execution domain, not merely a raw JavaScript adapter.
#[wasm_bindgen_test(async)]
async fn provider_request_creates_a_healthy_webgpu_device() {
    let instance = DeviceInstanceId::new(0x5747_5055);
    let provider = PlatformProvider::new(
        BackendKind::WebGpu,
        instance,
        Box::new(WebGpuProvider::new(instance)),
    );
    let descriptor =
        DeviceRequestDescriptor::new(AdapterSelection::Default, DeviceRequirements::new());
    let device = provider
        .request_device(descriptor)
        .await
        .expect("portable WebGPU device request");

    assert_eq!(device.backend(), BackendKind::WebGpu);
    assert_eq!(device.status(), DeviceStatus::Active);
    assert!(
        !device.adapter_info().name().trim().is_empty(),
        "WebGPU adapter diagnostics must not be empty"
    );
    let buffer = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::COPY_DST))
        .expect("a live WebGPU device creates a real GPU buffer");
    assert_eq!(buffer.descriptor().size, 16);
    let texture = device
        .create_texture(
            &TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
            )
            .with_view_format(TextureFormat::Rgba8UnormSrgb),
        )
        .expect("WebGPU creates a texture with an explicitly declared alternate view format");
    let view = device
        .create_texture_view(
            &texture,
            &TextureViewDescriptor::new(
                TextureViewDimension::D2,
                TextureAspects::COLOR,
                0,
                1,
                0,
                1,
            )
            .with_format(TextureFormat::Rgba8UnormSrgb),
        )
        .expect("WebGPU creates the declared sRGB view with the color aspect");
    assert_eq!(
        view.descriptor().format,
        Some(TextureFormat::Rgba8UnormSrgb)
    );

    // Pipeline creation must cross the request-only backend seam and await the
    // browser's native Promise. Dropping a pending public future retires its
    // private registry entry; a later JS settlement cannot publish a handle.
    let shader = device
        .create_shader(&compute_artifact(
            "@compute @workgroup_size(1) fn main() {}",
            0x51,
        ))
        .await
        .expect("valid WGSL shader module");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(Vec::new()))
        .expect("empty pipeline interface");
    let pipeline_desc = ComputePipelineDescriptor::new(shader.clone(), interface.clone());
    let baseline_promises = registry::pending_promise_count();
    let mut abandoned = Box::pin(device.create_compute_pipeline(&pipeline_desc));
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    assert!(matches!(
        abandoned.as_mut().poll(&mut context),
        Poll::Pending
    ));
    assert_eq!(registry::pending_promise_count(), baseline_promises + 1);
    drop(abandoned);
    assert_eq!(registry::pending_promise_count(), baseline_promises);

    device
        .create_compute_pipeline(&pipeline_desc)
        .await
        .expect("createComputePipelineAsync must resolve through the RHI future");

    // Native asynchronous validation failures remain structured creation
    // errors and never publish a usable pipeline handle.
    let bad_shader = device
        .create_shader(&compute_artifact(
            "@compute @workgroup_size(1) fn not_main() {}",
            0x52,
        ))
        .await
        .expect("createShaderModule itself accepts the module object");
    let bad_desc = ComputePipelineDescriptor::new(bad_shader, interface);
    assert_eq!(
        device
            .create_compute_pipeline(&bad_desc)
            .await
            .expect_err("missing WGSL entry point must reject the pipeline Promise")
            .kind(),
        crate::api::RhiErrorKind::BackendFailure
    );

    // Compression is an adapter/device feature pair, not a WebGPU baseline.
    // The assertion deliberately follows the published capability answer: it
    // proves an advertised format reaches native creation without requiring a
    // particular desktop GPU or Chrome build to expose BC.
    for format in [
        TextureFormat::Bc1RgbaUnorm,
        TextureFormat::Etc2Rgb8Unorm,
        TextureFormat::Astc4x4Unorm,
    ] {
        let compressed = TextureSupportQuery::new(
            crate::api::resource::TextureDimension::D2,
            format,
            TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
            1,
        );
        if device
            .capabilities()
            .texture_support(&compressed)
            .is_supported()
        {
            device
                .create_texture(&TextureDescriptor::new_2d(
                    4,
                    4,
                    format,
                    TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
                ))
                .expect("a published compressed texture capability must lower to WebGPU creation");
        }
    }
    device.poll().expect("new WebGPU device remains healthy");
}
