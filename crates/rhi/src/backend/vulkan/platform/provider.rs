//! Vulkan instance ownership, adapter enumeration, and `VkDevice` creation.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the host integration that opens a crate-private Vulkan provider has not landed; platform conformance tests are its current entry point"
    )
)]

use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;

use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::backend::ProviderBackend;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::{DeviceRequirements, LimitRequirement};
use crate::api::platform::{AdapterId, AdapterInfo, BackendKind};
use crate::api::presentation::PresentationTarget;
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};

use crate::backend::vulkan::ffi;

use super::device::VulkanDevice;
use super::facts;
use super::request::VulkanRequest;

/// The instance and loader must outlive all devices made through it.
pub(super) struct VulkanInstance {
    _entry: ash::Entry,
    instance: ash::Instance,
}

impl VulkanInstance {
    fn new() -> RhiResult<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(|error| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                format!("unable to load the Vulkan loader: {error}"),
            )
            .at("VulkanProvider::new")
        })?;
        let application = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
        let create = vk::InstanceCreateInfo::default().application_info(&application);
        let instance = unsafe { entry.create_instance(&create, None) }
            .map_err(|result| ffi::to_rhi(result, "VulkanProvider::new"))?;
        Ok(Self {
            _entry: entry,
            instance,
        })
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

struct Candidate {
    physical: vk::PhysicalDevice,
    serial: u64,
    name: String,
    vendor: u32,
    device: u32,
    discrete: bool,
    graphics_family: u32,
    buffer_ceiling: u64,
    uniform_buffer_ceiling: u64,
    storage_buffer_ceiling: u64,
    non_coherent_atom_size: u64,
    capability_limits: facts::VulkanCapabilityLimits,
}

/// One Vulkan provider owns one loaded instance and may create many devices.
pub(crate) struct VulkanProvider {
    provider: DeviceInstanceId,
    instance: Arc<VulkanInstance>,
}

impl VulkanProvider {
    pub(crate) fn new(provider: DeviceInstanceId) -> RhiResult<Self> {
        Ok(Self {
            provider,
            instance: Arc::new(VulkanInstance::new()?),
        })
    }

    fn candidates(&self) -> RhiResult<Vec<Candidate>> {
        let physical = unsafe { self.instance.instance.enumerate_physical_devices() }
            .map_err(|result| ffi::to_rhi(result, "VulkanProvider::enumerate_adapters"))?;
        let mut candidates = Vec::new();
        for physical in physical {
            let families = unsafe {
                self.instance
                    .instance
                    .get_physical_device_queue_family_properties(physical)
            };
            let Some(graphics_family) = families.iter().position(|family| {
                family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            }) else {
                continue;
            };
            let properties = unsafe {
                self.instance
                    .instance
                    .get_physical_device_properties(physical)
            };
            let memory = unsafe {
                self.instance
                    .instance
                    .get_physical_device_memory_properties(physical)
            };
            let buffer_ceiling = memory.memory_heaps[..memory.memory_heap_count as usize]
                .iter()
                .map(|heap| heap.size)
                .min()
                .unwrap_or(1);
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            candidates.push(Candidate {
                physical,
                // Enumeration order is not identity: drivers may reorder the
                // list as devices appear, disappear, or update. The dispatchable
                // handle is stable for this provider instance and stays private;
                // only its opaque provider-local serial reaches `AdapterId`.
                serial: physical.as_raw(),
                name,
                vendor: properties.vendor_id,
                device: properties.device_id,
                discrete: properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                graphics_family: graphics_family as u32,
                buffer_ceiling,
                uniform_buffer_ceiling: u64::from(properties.limits.max_uniform_buffer_range),
                storage_buffer_ceiling: u64::from(properties.limits.max_storage_buffer_range),
                non_coherent_atom_size: properties.limits.non_coherent_atom_size,
                capability_limits: facts::VulkanCapabilityLimits {
                    max_bindings_per_group: properties.limits.max_per_stage_resources.min(
                        properties
                            .limits
                            .max_descriptor_set_uniform_buffers
                            .saturating_add(properties.limits.max_descriptor_set_storage_buffers),
                    ),
                    max_bound_descriptor_sets: properties.limits.max_bound_descriptor_sets,
                    max_per_stage_uniform_buffers: properties
                        .limits
                        .max_per_stage_descriptor_uniform_buffers,
                    max_per_stage_storage_buffers: properties
                        .limits
                        .max_per_stage_descriptor_storage_buffers,
                    min_uniform_buffer_offset_alignment: properties
                        .limits
                        .min_uniform_buffer_offset_alignment,
                    min_storage_buffer_offset_alignment: properties
                        .limits
                        .min_storage_buffer_offset_alignment,
                    max_compute_work_group_invocations: properties
                        .limits
                        .max_compute_work_group_invocations,
                    max_compute_work_group_size: properties.limits.max_compute_work_group_size,
                    max_compute_work_group_count: properties.limits.max_compute_work_group_count,
                    max_compute_shared_memory_size: properties
                        .limits
                        .max_compute_shared_memory_size,
                },
            });
        }
        Ok(candidates)
    }

    fn select(&self, selection: AdapterSelection) -> RhiResult<Candidate> {
        let mut candidates = self.candidates()?;
        if let AdapterSelection::Explicit(id) = selection {
            return candidates
                .into_iter()
                .find(|candidate| candidate.serial == id.serial())
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::Unsupported,
                        "the explicitly selected Vulkan adapter is no longer available",
                    )
                    .at("VulkanProvider::request_device")
                });
        }
        if matches!(selection, AdapterSelection::PreferHighPerformance) {
            candidates.sort_by_key(|candidate| !candidate.discrete);
        }
        if matches!(selection, AdapterSelection::PreferLowPower) {
            candidates.sort_by_key(|candidate| candidate.discrete);
        }
        candidates.into_iter().next().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "no Vulkan physical device exposes a graphics queue",
            )
            .at("VulkanProvider::request_device")
        })
    }

    fn adapter_info(&self, candidate: &Candidate, facts: CapabilityFacts) -> AdapterInfo {
        AdapterInfo::new(
            AdapterId::new(self.provider.as_u64(), candidate.serial),
            candidate.name.clone(),
            BackendKind::Vulkan,
            Some(candidate.vendor),
            Some(candidate.device),
            AvailableCapabilities::from_facts(facts),
        )
    }

    fn facts(&self, candidate: &Candidate) -> RhiResult<CapabilityFacts> {
        facts::probe(
            &self.instance.instance,
            candidate.physical,
            candidate.buffer_ceiling,
            candidate.uniform_buffer_ceiling,
            candidate.storage_buffer_ceiling,
            candidate.capability_limits,
        )
    }

    fn create_native(&self, descriptor: &DeviceRequestDescriptor) -> RhiResult<VulkanDevice> {
        let candidate = self.select(descriptor.selection())?;
        let facts = self.facts(&candidate)?;
        validate_requirements(
            descriptor.requirements(),
            &AvailableCapabilities::from_facts(facts.clone()),
        )?;
        if !descriptor.presentation_targets().is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "the Vulkan platform slice does not yet enable surface/presentation extensions",
            )
            .at("VulkanProvider::request_device"));
        }
        let priorities = [1.0];
        let queue = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(candidate.graphics_family)
            .queue_priorities(&priorities);
        let create =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue));
        let device = unsafe {
            self.instance
                .instance
                .create_device(candidate.physical, &create, None)
        }
        .map_err(|result| ffi::to_rhi(result, "VulkanProvider::request_device"))?;
        let graphics_queue = unsafe { device.get_device_queue(candidate.graphics_family, 0) };
        let memory_properties = unsafe {
            self.instance
                .instance
                .get_physical_device_memory_properties(candidate.physical)
        };
        let submission = SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::Graphics,
            // v13's base submission invariant requires a graphics lane that
            // carries RASTER|COPY. COMPUTE is now closed by native lowering on
            // this same queue. Raster payload lowering is the next Vulkan
            // vertical slice; until then its required base-domain declaration
            // is guarded by explicit Phase-A Unsupported rather than a no-op.
            LaneWorkDomains::RASTER
                .union(LaneWorkDomains::COMPUTE)
                .union(LaneWorkDomains::COPY),
        )]);
        VulkanDevice::new(
            self.adapter_info(&candidate, facts.clone()),
            Arc::clone(&self.instance),
            device,
            graphics_queue,
            candidate.graphics_family,
            candidate.physical,
            memory_properties,
            candidate.non_coherent_atom_size,
            facts,
            submission,
        )
        .map_err(|failure| failure.into_rhi("VulkanProvider::request_device"))
    }
}

impl ProviderBackend for VulkanProvider {
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        let mut adapters = Vec::new();
        for candidate in self.candidates()? {
            let facts = self.facts(&candidate)?;
            adapters.push(self.adapter_info(&candidate, facts));
        }
        Ok(Some(adapters))
    }

    fn supports_presentation(&self, _: AdapterId, _: &PresentationTarget) -> RhiResult<bool> {
        // A surface extension is intentionally not enabled until the presentation
        // backend owns its complete acquire/present/loss lifecycle.
        Ok(false)
    }

    fn request_device(
        &self,
        descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn crate::api::platform::backend::DeviceRequestBackend>> {
        Ok(Box::new(VulkanRequest::new(
            self.create_native(descriptor)?,
        )))
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
        format!("selected Vulkan adapter does not support the requested {kind}"),
    )
    .at("VulkanProvider::request_device"))
}
