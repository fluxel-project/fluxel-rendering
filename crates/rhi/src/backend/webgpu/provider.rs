//! WebGPU provider and asynchronous request bridge.

use crate::api::capability::AvailableCapabilities;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::backend::{DeviceRequestBackend, ProviderBackend, RequestProgress};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::{DeviceRequirements, LimitKey, LimitRequirement, OptionalFeature};
use crate::api::presentation::PresentationTarget;
use crate::api::resource::RouteQuery;

use super::{device::WebGpuDevice, js, registry};

/// Browser WebGPU provider.  It owns no JS value: browser values begin in the
/// TLS registry while executing `request_device` on the owner thread.
pub(crate) struct WebGpuProvider {
    provider: DeviceInstanceId,
}

impl WebGpuProvider {
    pub(crate) fn new(provider: DeviceInstanceId) -> Self {
        Self { provider }
    }
}

impl ProviderBackend for WebGpuProvider {
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<crate::api::platform::AdapterInfo>>> {
        // Browsers intentionally choose adapters asynchronously; publishing a
        // guessed adapter list would violate the optional-enumeration contract.
        Ok(None)
    }

    fn supports_presentation(
        &self,
        _: crate::api::platform::AdapterId,
        _: &PresentationTarget,
    ) -> RhiResult<bool> {
        // There is no adapter token on a no-enumeration provider. Device creation
        // receives the target and the presentation module validates it later.
        Ok(false)
    }

    fn request_device(
        &self,
        descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn DeviceRequestBackend>> {
        let preference = match descriptor.selection() {
            AdapterSelection::Default => None,
            AdapterSelection::PreferHighPerformance => Some("high-performance"),
            AdapterSelection::PreferLowPower => Some("low-power"),
            AdapterSelection::Explicit(_) => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "WebGPU does not expose explicit adapter selection",
                )
                .at("WebGpuProvider::request_device"));
            }
        };
        let gpu = js::browser_gpu().map_err(|error| {
            RhiError::new(RhiErrorKind::Unsupported, js::message(&error))
                .at("WebGpuProvider::request_device")
        })?;
        let adapter = js::request_adapter(&gpu, preference).map_err(|error| {
            RhiError::new(RhiErrorKind::BackendFailure, js::message(&error))
                .at("WebGpuProvider::request_device")
        })?;
        Ok(Box::new(WebGpuRequest {
            provider: self.provider,
            requirements: descriptor.requirements().clone(),
            state: RequestState::Adapter(registry::start_promise(adapter)),
        }))
    }
}

enum RequestState {
    Adapter(registry::WebGpuRequestId),
    Device {
        adapter: wasm_bindgen::JsValue,
        request: registry::WebGpuRequestId,
    },
    Done,
}

struct WebGpuRequest {
    provider: DeviceInstanceId,
    requirements: DeviceRequirements,
    state: RequestState,
}

impl DeviceRequestBackend for WebGpuRequest {
    fn poll_or_register_waker(&mut self, waker: &std::task::Waker) -> RhiResult<RequestProgress> {
        loop {
            match &mut self.state {
                RequestState::Adapter(request) => match registry::poll_promise(*request, waker) {
                    registry::PromisePoll::Pending => return Ok(RequestProgress::Pending),
                    registry::PromisePoll::Failed(message) => {
                        return Err(RhiError::new(RhiErrorKind::BackendFailure, message)
                            .at("WebGpuRequest::poll_or_register_waker"));
                    }
                    registry::PromisePoll::Ready(adapter) => {
                        // A requestAdapter promise resolves to null when no adapter
                        // matches; the device request is deliberately not started.
                        if adapter.is_null() || adapter.is_undefined() {
                            return Err(RhiError::new(
                                RhiErrorKind::Unsupported,
                                "no WebGPU adapter matches this request",
                            )
                            .at("WebGpuRequest::poll_or_register_waker"));
                        }
                        let request = request_plan(&adapter, &self.requirements)?;
                        let promise =
                            js::request_device(&adapter, &request.features, &request.limits)
                                .map_err(|error| {
                                    RhiError::new(RhiErrorKind::BackendFailure, js::message(&error))
                                        .at("WebGpuRequest::poll_or_register_waker")
                                })?;
                        self.state = RequestState::Device {
                            adapter,
                            request: registry::start_promise(promise),
                        };
                    }
                },
                RequestState::Device { adapter, request } => {
                    match registry::poll_promise(*request, waker) {
                        registry::PromisePoll::Pending => return Ok(RequestProgress::Pending),
                        registry::PromisePoll::Failed(message) => {
                            return Err(RhiError::new(RhiErrorKind::BackendFailure, message)
                                .at("WebGpuRequest::poll_or_register_waker"));
                        }
                        registry::PromisePoll::Ready(device_handle) => {
                            let (registration, metadata) =
                                registry::register_resolved_device(adapter.clone(), device_handle)
                                    .map_err(|message| {
                                        RhiError::new(RhiErrorKind::BackendFailure, message)
                                            .at("WebGpuRequest::poll_or_register_waker")
                                    })?;
                            self.state = RequestState::Done;
                            let device = WebGpuDevice::from_registration(
                                registration,
                                self.provider,
                                metadata,
                            )?;
                            validate_requirements(&self.requirements, &device)?;
                            return Ok(RequestProgress::Ready(Box::new(device)));
                        }
                    }
                }
                RequestState::Done => {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "a completed WebGPU device request was polled more than once",
                    )
                    .at("WebGpuRequest::poll_or_register_waker"));
                }
            }
        }
    }
}

/// The finite feature vocabulary this backend understands at device-request
/// time.  A browser advertising a new string cannot change Fluxel facts until a
/// lowering and conformance case explicitly add it here.
const KNOWN_WEBGPU_FEATURES: &[&str] = &[
    "float32-filterable",
    "readonly_and_readwrite_storage_textures",
    "texture-compression-bc",
    "texture-compression-etc2",
    "texture-compression-astc",
    "indirect-first-instance",
];

struct RequestPlan {
    features: Vec<String>,
    limits: Vec<(&'static str, u64)>,
}

/// Turns portable requirements into the narrow subset of WebGPU's
/// `GPUDeviceDescriptor` that has the same meaning.  It intentionally performs
/// no "best guess" request: unknown portable requirements fail before a device
/// is created, while `AtMost` constraints are checked against the actual device
/// after creation because WebGPU cannot request a *smaller* alignment.
fn request_plan(
    adapter: &wasm_bindgen::JsValue,
    requirements: &DeviceRequirements,
) -> RhiResult<RequestPlan> {
    let available = js::supported_features(adapter, KNOWN_WEBGPU_FEATURES);
    let mut features = Vec::new();

    for &feature in requirements.required_features() {
        require_optional_feature(feature, &available, &mut features)?;
    }
    for &feature in requirements.preferred_features() {
        // Preferences are best-effort. A core feature costs no request; an
        // optional browser feature is asked for only if this adapter advertises
        // it, and an unmapped portable preference remains absent from facts.
        let _ = prefer_optional_feature(feature, &available, &mut features);
    }

    for query in requirements.required_texture_support() {
        for format in std::iter::once(query.format()).chain(query.view_formats().iter().copied()) {
            require_compression_feature(format, &available, &mut features)?;
        }
    }
    for query in requirements.required_route_support() {
        for format in route_formats(query) {
            require_compression_feature(format, &available, &mut features)?;
        }
    }

    let mut limits: Vec<(&'static str, u64)> = Vec::new();
    for &requirement in requirements.limit_requirements() {
        match requirement {
            LimitRequirement::AtLeast { key, value } => {
                let name = webgpu_limit(key)
                    .ok_or_else(|| unsupported_requirement("minimum limit", format!("{key:?}")))?;
                // WebIDL numbers are IEEE-754 doubles. Passing a u64 above the
                // exactly representable range would silently request another
                // value, so reject it before constructing JS.
                if value > 9_007_199_254_740_991 {
                    return Err(unsupported_requirement(
                        "minimum limit",
                        format!("{key:?} exceeds the WebGPU integer range"),
                    ));
                }
                if let Some((_, prior)) = limits
                    .iter_mut()
                    .find(|(prior_name, _)| *prior_name == name)
                {
                    *prior = (*prior).max(value);
                } else {
                    limits.push((name, value));
                }
            }
            LimitRequirement::AtMost { key, .. } => {
                if webgpu_limit(key).is_none() {
                    return Err(unsupported_requirement(
                        "maximum alignment limit",
                        format!("{key:?}"),
                    ));
                }
            }
        }
    }
    Ok(RequestPlan { features, limits })
}

fn require_optional_feature(
    feature: OptionalFeature,
    available: &[String],
    selected: &mut Vec<String>,
) -> RhiResult<()> {
    match webgpu_optional_feature(feature) {
        Some(None) => Ok(()),
        Some(Some(name)) if available.iter().any(|candidate| candidate == name) => {
            if !selected.iter().any(|candidate| candidate == name) {
                selected.push(name.to_owned());
            }
            Ok(())
        }
        Some(Some(name)) => Err(unsupported_requirement(
            "required feature",
            format!("{feature:?} requires {name}"),
        )),
        None => Err(unsupported_requirement(
            "required feature",
            format!("{feature:?}"),
        )),
    }
}

fn prefer_optional_feature(
    feature: OptionalFeature,
    available: &[String],
    selected: &mut Vec<String>,
) -> bool {
    let Some(Some(name)) = webgpu_optional_feature(feature) else {
        return webgpu_optional_feature(feature).is_some();
    };
    if !available.iter().any(|candidate| candidate == name) {
        return false;
    }
    if !selected.iter().any(|candidate| candidate == name) {
        selected.push(name.to_owned());
    }
    true
}

/// `Some(None)` is a core WebGPU feature with direct lowering; `Some(name)`
/// requires that precise browser feature string; `None` has no reviewed route.
fn webgpu_optional_feature(feature: OptionalFeature) -> Option<Option<&'static str>> {
    Some(match feature {
        OptionalFeature::Compute
        | OptionalFeature::ComparisonSamplers
        | OptionalFeature::BaseVertex
        | OptionalFeature::ClearBuffer
        | OptionalFeature::IndirectDispatch
        | OptionalFeature::IndependentBlend
        | OptionalFeature::MappablePrimaryBuffers => None,
        OptionalFeature::IndirectDraw | OptionalFeature::IndirectFirstInstance => {
            Some("indirect-first-instance")
        }
        _ => return None,
    })
}

fn require_compression_feature(
    format: TextureFormat,
    available: &[String],
    selected: &mut Vec<String>,
) -> RhiResult<()> {
    if let Some(feature) = compression_feature(format) {
        if !available.iter().any(|candidate| candidate == feature) {
            return Err(unsupported_requirement(
                "compressed texture format",
                format!("{format:?} requires {feature}"),
            ));
        }
        if !selected.iter().any(|candidate| candidate == feature) {
            selected.push(feature.to_owned());
        }
    } else if is_astc_hdr(format) {
        return Err(unsupported_requirement(
            "compressed texture format",
            format!("{format:?}"),
        ));
    }
    Ok(())
}

fn route_formats(query: &RouteQuery) -> Vec<TextureFormat> {
    match query {
        RouteQuery::BufferToBuffer => Vec::new(),
        RouteQuery::BufferToTexture { format, .. }
        | RouteQuery::TextureToBuffer { format, .. }
        | RouteQuery::Resolve { format, .. } => vec![*format],
        RouteQuery::TextureToTexture {
            src_format,
            dst_format,
            ..
        }
        | RouteQuery::Blit {
            src_format,
            dst_format,
            ..
        } => vec![*src_format, *dst_format],
    }
}

fn validate_requirements(
    requirements: &DeviceRequirements,
    device: &WebGpuDevice,
) -> RhiResult<()> {
    let capabilities = AvailableCapabilities::from_facts(
        crate::api::platform::backend::DeviceBackend::capability_facts(device),
    );
    for &feature in requirements.required_features() {
        if !capabilities.supports_feature(feature) {
            return Err(unsupported_requirement(
                "required feature",
                format!("{feature:?}"),
            ));
        }
    }
    for &requirement in requirements.limit_requirements() {
        let actual = capabilities.limit(requirement.key()).ok_or_else(|| {
            unsupported_requirement("device limit", format!("{:?}", requirement.key()))
        })?;
        let satisfied = match requirement {
            LimitRequirement::AtLeast { value, .. } => actual >= value,
            LimitRequirement::AtMost { value, .. } => actual <= value,
        };
        if !satisfied {
            return Err(unsupported_requirement(
                "device limit",
                format!("{:?} requirement was not met", requirement.key()),
            ));
        }
    }
    for query in requirements.required_buffer_support() {
        if !capabilities.buffer_support(query).is_supported() {
            return Err(unsupported_requirement(
                "buffer support",
                format!("{query:?}"),
            ));
        }
    }
    for query in requirements.required_texture_support() {
        if !capabilities.texture_support(query).is_supported() {
            return Err(unsupported_requirement(
                "texture support",
                format!("{query:?}"),
            ));
        }
    }
    for query in requirements.required_binding_support() {
        if !capabilities.binding_support(query).is_supported() {
            return Err(unsupported_requirement(
                "binding support",
                format!("{query:?}"),
            ));
        }
    }
    for query in requirements.required_route_support() {
        if !capabilities.route(query).is_supported() {
            return Err(unsupported_requirement(
                "transfer route",
                format!("{query:?}"),
            ));
        }
    }
    Ok(())
}

fn unsupported_requirement(kind: &'static str, detail: String) -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        format!("WebGPU cannot satisfy {kind}: {detail}"),
    )
    .at("WebGpuProvider::request_device")
}

fn webgpu_limit(key: LimitKey) -> Option<&'static str> {
    Some(match key {
        LimitKey::MaxBufferSize => "maxBufferSize",
        LimitKey::MaxTexture1dDimension => "maxTextureDimension1D",
        LimitKey::MaxTexture2dDimension => "maxTextureDimension2D",
        LimitKey::MaxTexture3dDimension => "maxTextureDimension3D",
        LimitKey::MaxTextureArrayLayers => "maxTextureArrayLayers",
        LimitKey::MaxBindGroups => "maxBindGroups",
        LimitKey::MaxBindingsPerGroup => "maxBindingsPerBindGroup",
        LimitKey::MaxUniformBufferBindingSize => "maxUniformBufferBindingSize",
        LimitKey::MaxStorageBufferBindingSize => "maxStorageBufferBindingSize",
        LimitKey::MaxColorAttachments => "maxColorAttachments",
        LimitKey::MaxVertexBuffers => "maxVertexBuffers",
        LimitKey::MaxVertexAttributes => "maxVertexAttributes",
        LimitKey::MaxVertexBufferArrayStride => "maxVertexBufferArrayStride",
        LimitKey::MaxInterStageShaderVariables => "maxInterStageShaderVariables",
        LimitKey::MaxComputeInvocationsPerWorkgroup => "maxComputeInvocationsPerWorkgroup",
        LimitKey::MaxComputeWorkgroupSizeX => "maxComputeWorkgroupSizeX",
        LimitKey::MaxComputeWorkgroupSizeY => "maxComputeWorkgroupSizeY",
        LimitKey::MaxComputeWorkgroupSizeZ => "maxComputeWorkgroupSizeZ",
        LimitKey::MaxComputeWorkgroupsPerDimension => "maxComputeWorkgroupsPerDimension",
        LimitKey::MaxComputeWorkgroupStorageSize => "maxComputeWorkgroupStorageSize",
        LimitKey::MinUniformBufferOffsetAlignment => "minUniformBufferOffsetAlignment",
        LimitKey::MinStorageBufferOffsetAlignment => "minStorageBufferOffsetAlignment",
        _ => return None,
    })
}

fn compression_feature(format: TextureFormat) -> Option<&'static str> {
    use TextureFormat::*;
    Some(match format {
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm
        | Bc3RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm | Bc5RgUnorm | Bc5RgSnorm | Bc6hRgbUfloat
        | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => "texture-compression-bc",
        Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb
        | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacR11Unorm | EacR11Snorm | EacRg11Unorm
        | EacRg11Snorm => "texture-compression-etc2",
        Astc4x4Unorm | Astc4x4UnormSrgb | Astc5x4Unorm | Astc5x4UnormSrgb | Astc5x5Unorm
        | Astc5x5UnormSrgb | Astc6x5Unorm | Astc6x5UnormSrgb | Astc6x6Unorm | Astc6x6UnormSrgb
        | Astc8x5Unorm | Astc8x5UnormSrgb | Astc8x6Unorm | Astc8x6UnormSrgb | Astc8x8Unorm
        | Astc8x8UnormSrgb | Astc10x5Unorm | Astc10x5UnormSrgb | Astc10x6Unorm
        | Astc10x6UnormSrgb | Astc10x8Unorm | Astc10x8UnormSrgb | Astc10x10Unorm
        | Astc10x10UnormSrgb | Astc12x10Unorm | Astc12x10UnormSrgb | Astc12x12Unorm
        | Astc12x12UnormSrgb => "texture-compression-astc",
        _ => return None,
    })
}

fn is_astc_hdr(format: TextureFormat) -> bool {
    use TextureFormat::*;
    matches!(
        format,
        Astc4x4Hdr
            | Astc5x4Hdr
            | Astc5x5Hdr
            | Astc6x5Hdr
            | Astc6x6Hdr
            | Astc8x5Hdr
            | Astc8x6Hdr
            | Astc8x8Hdr
            | Astc10x5Hdr
            | Astc10x6Hdr
            | Astc10x8Hdr
            | Astc10x10Hdr
            | Astc12x10Hdr
            | Astc12x12Hdr
    )
}
