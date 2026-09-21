//! Metal adapter discovery and logical-device creation.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::MTLCopyAllDevices;
#[cfg(not(target_os = "macos"))]
use objc2_metal::MTLCreateSystemDefaultDevice;
use objc2_metal::MTLDevice;
use objc2_quartz_core::CAMetalLayer;

use crate::api::capability::AvailableCapabilities;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::backend::{DeviceRequestBackend, ProviderBackend, RequestProgress};
use crate::api::platform::requirements::{DeviceRequirements, LimitRequirement};
use crate::api::platform::{
    AdapterId, AdapterInfo, AdapterSelection, BackendKind, DeviceRequestDescriptor,
};
use crate::api::presentation::PresentationTarget;

use super::device::MetalDevice;

struct Candidate {
    serial: u64,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    info: AdapterInfo,
}

/// Provider-local Metal discovery state. Native devices are retained only for
/// the duration of a snapshot/request and never enter the public object model.
pub(crate) struct MetalProvider {
    provider: DeviceInstanceId,
    targets: std::sync::Arc<super::presentation::MetalTargetRegistry>,
}

impl MetalProvider {
    pub(crate) fn new(provider: DeviceInstanceId) -> Self {
        Self {
            provider,
            targets: std::sync::Arc::new(super::presentation::MetalTargetRegistry::new()),
        }
    }

    pub(crate) fn register_layer_target(
        &self,
        layer: Retained<CAMetalLayer>,
    ) -> PresentationTarget {
        self.targets.register_layer_target(layer)
    }

    fn candidates(&self) -> RhiResult<Vec<Candidate>> {
        // Adapter enumeration is a macOS facility.  In particular, do not
        // make iOS/tvOS/visionOS link an enumeration symbol just because all
        // four targets have `target_vendor = "apple"`: those platforms expose
        // one system-default MTLDevice to this RHI provider.
        #[cfg(target_os = "macos")]
        let devices: Vec<Retained<ProtocolObject<dyn MTLDevice>>> =
            MTLCopyAllDevices().into_iter().collect();
        #[cfg(not(target_os = "macos"))]
        let devices: Vec<Retained<ProtocolObject<dyn MTLDevice>>> =
            MTLCreateSystemDefaultDevice().into_iter().collect();

        let mut result = Vec::new();
        for (index, device) in devices.into_iter().enumerate() {
            let serial = u64::try_from(index + 1).map_err(|_| {
                RhiError::new(RhiErrorKind::BackendFailure, "too many Metal devices")
                    .at("MetalProvider::enumerate_adapters")
            })?;
            let facts = super::facts::baseline(&device);
            let name = device.name().to_string();
            result.push(Candidate {
                serial,
                info: AdapterInfo::new(
                    AdapterId::new(self.provider.as_u64(), serial),
                    name,
                    BackendKind::Metal,
                    None,
                    None,
                    AvailableCapabilities::from_facts(facts),
                ),
                device,
            });
        }
        Ok(result)
    }

    fn select(&self, selection: AdapterSelection) -> RhiResult<Candidate> {
        let mut candidates = self.candidates()?;
        match selection {
            AdapterSelection::Explicit(id) => candidates
                .into_iter()
                .find(|candidate| candidate.serial == id.serial())
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::Unsupported,
                        "the explicitly selected Metal adapter is no longer available",
                    )
                    .at("MetalProvider::request_device")
                }),
            AdapterSelection::PreferLowPower => {
                candidates.sort_by_key(|candidate| !candidate.device.isLowPower());
                candidates.into_iter().next().ok_or_else(no_adapter)
            }
            AdapterSelection::PreferHighPerformance => {
                candidates.sort_by_key(|candidate| candidate.device.isLowPower());
                candidates.into_iter().next().ok_or_else(no_adapter)
            }
            _ => candidates.into_iter().next().ok_or_else(no_adapter),
        }
    }
}

fn no_adapter() -> RhiError {
    RhiError::new(RhiErrorKind::Unsupported, "no Metal device is available")
        .at("MetalProvider::request_device")
}

impl ProviderBackend for MetalProvider {
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        Ok(Some(
            self.candidates()?
                .into_iter()
                .map(|item| item.info)
                .collect(),
        ))
    }

    fn supports_presentation(
        &self,
        adapter: AdapterId,
        target: &PresentationTarget,
    ) -> RhiResult<bool> {
        Ok(self.targets.contains(target.id())
            && self
                .candidates()?
                .iter()
                .any(|candidate| candidate.serial == adapter.serial()))
    }

    fn request_device(
        &self,
        descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn DeviceRequestBackend>> {
        let candidate = self.select(descriptor.selection())?;
        if descriptor
            .presentation_targets()
            .iter()
            .any(|target| !self.targets.contains(target.id()))
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "a required Metal presentation target is not registered with this provider",
            )
            .at("MetalProvider::request_device"));
        }
        validate_requirements(
            descriptor.requirements(),
            candidate.info.available_capabilities(),
        )?;
        let device = MetalDevice::new(
            candidate.info,
            candidate.device,
            std::sync::Arc::clone(&self.targets),
        )?;
        Ok(Box::new(MetalRequest {
            device: Some(device),
        }))
    }
}

fn validate_requirements(
    requirements: &DeviceRequirements,
    facts: &AvailableCapabilities,
) -> RhiResult<()> {
    for feature in requirements.required_features() {
        if !facts.supports_feature(*feature) {
            return unsupported_requirement("feature");
        }
    }
    for requirement in requirements.limit_requirements() {
        let Some(actual) = facts.limit(requirement.key()) else {
            return unsupported_requirement("limit");
        };
        let satisfied = match requirement {
            LimitRequirement::AtLeast { value, .. } => actual >= *value,
            LimitRequirement::AtMost { value, .. } => actual <= *value,
        };
        if !satisfied {
            return unsupported_requirement("limit");
        }
    }
    for query in requirements.required_buffer_support() {
        if !facts.buffer_support(query).is_supported() {
            return unsupported_requirement("buffer capability");
        }
    }
    for query in requirements.required_texture_support() {
        if !facts.texture_support(query).is_supported() {
            return unsupported_requirement("texture capability");
        }
    }
    for query in requirements.required_binding_support() {
        if !facts.binding_support(query).is_supported() {
            return unsupported_requirement("binding capability");
        }
    }
    for query in requirements.required_route_support() {
        if !facts.route(query).is_supported() {
            return unsupported_requirement("transfer route");
        }
    }
    Ok(())
}

fn unsupported_requirement(kind: &'static str) -> RhiResult<()> {
    Err(RhiError::new(
        RhiErrorKind::Unsupported,
        format!("selected Metal adapter does not support the requested {kind}"),
    )
    .at("MetalProvider::request_device"))
}

struct MetalRequest {
    device: Option<MetalDevice>,
}

impl DeviceRequestBackend for MetalRequest {
    fn poll_or_register_waker(&mut self, _waker: &std::task::Waker) -> RhiResult<RequestProgress> {
        let device = self.device.take().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a completed Metal device request was polled more than once",
            )
        })?;
        Ok(RequestProgress::Ready(Box::new(device)))
    }
}
