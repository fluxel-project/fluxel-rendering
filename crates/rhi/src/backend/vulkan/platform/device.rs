//! The owned Vulkan execution domain for the platform slice.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::task::Waker;

use ash::vk;

use crate::api::capability::CapabilityFacts;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::backend::DeviceBackend;
use crate::api::platform::{AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::resource::transfer::{ReadbackStatus, ReadbackTicket};
use crate::api::submission::{CompletionState, SubmissionCapabilities};

use crate::backend::vulkan::binding;
use crate::backend::vulkan::command::spine::VulkanCommandSpine;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::pipeline;
use crate::backend::vulkan::resource;
use crate::backend::vulkan::shader;

use super::provider::VulkanInstance;

struct Liveness {
    status: DeviceStatus,
    info: Option<DeviceLossInfo>,
    completion_waiters: BTreeMap<u64, Vec<Waker>>,
    pending_readbacks: Vec<ReadbackTicket>,
}

/// A Vulkan device with exactly one loss authority.
///
/// `VK_ERROR_DEVICE_LOST` may be reported by any device or queue operation.
/// Every such boundary calls `observe_native_failure`, preserving the first
/// reason for the entire `DeviceIdentity`. There is no recovery transition: a
/// fresh `request_device` owns a fresh native `VkDevice` and RHI identity.
pub(crate) struct VulkanDevice {
    adapter: AdapterInfo,
    object: ObjectId,
    shared: std::sync::Arc<VulkanShared>,
    command: VulkanCommandSpine,
    facts: CapabilityFacts,
    submission: SubmissionCapabilities,
}

/// The one shared native ownership domain for a Vulkan RHI device.
///
/// Future native buffers, textures, descriptors, command pools and fences keep
/// one clone of this object.  That makes `VkDevice` outlive every native child,
/// gives all native boundaries one loss authority, and avoids independently
/// reference-counting the device, queue, and liveness state.
pub(crate) struct VulkanShared {
    _instance: std::sync::Arc<VulkanInstance>,
    pub(crate) device: ash::Device,
    /// Retained even though this slice submits no command buffers. It establishes
    /// the physical queue ownership that future command lowering must use rather
    /// than opening a second implicit queue path.
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) graphics_family: u32,
    /// Fixed at device creation. Dedicated allocations choose a compatible
    /// memory type from this snapshot; they never query a possibly unrelated
    /// physical device later.
    #[expect(
        dead_code,
        reason = "retained for format/property probes that accompany later Vulkan capability slices"
    )]
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Required to align flush/invalidate ranges for host-visible memory that
    /// does not advertise HOST_COHERENT.
    pub(crate) non_coherent_atom_size: vk::DeviceSize,
    liveness: Mutex<Liveness>,
}

impl VulkanDevice {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        adapter: AdapterInfo,
        instance: std::sync::Arc<VulkanInstance>,
        device: ash::Device,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        physical_device: vk::PhysicalDevice,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        non_coherent_atom_size: vk::DeviceSize,
        facts: CapabilityFacts,
        submission: SubmissionCapabilities,
    ) -> Result<Self, VulkanFailure> {
        let shared = std::sync::Arc::new(VulkanShared {
            _instance: instance,
            device,
            graphics_queue,
            graphics_family,
            physical_device,
            memory_properties,
            non_coherent_atom_size,
            liveness: Mutex::new(Liveness {
                status: DeviceStatus::Active,
                info: None,
                completion_waiters: BTreeMap::new(),
                pending_readbacks: Vec::new(),
            }),
        });
        let command = VulkanCommandSpine::new(std::sync::Arc::clone(&shared))?;
        Ok(Self {
            adapter,
            object: ObjectId::next(),
            shared,
            command,
            facts,
            submission,
        })
    }

    fn unsupported<T>(&self, what: &'static str) -> RhiResult<T> {
        Err(VulkanFailure::Unsupported {
            what,
            why: "the Vulkan platform slice owns no lowering for this operation",
        }
        .into_rhi("VulkanDevice"))
    }

    /// Converts a device-owned result through the only terminal-loss authority.
    fn observe_failure(&self, error: VulkanFailure) -> RhiError {
        if error.is_terminal() {
            let summary = error.message();
            let diagnostic = error.into_rhi("VulkanDevice");
            self.shared.mark_lost(DeviceLossInfo::new(format!(
                "Vulkan reported VK_ERROR_DEVICE_LOST ({summary}): {diagnostic}",
            )));
            let info = self.loss_info().expect("loss authority records first loss");
            return RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned())
                .at("VulkanDevice");
        }
        error.into_rhi("VulkanDevice")
    }
}

impl VulkanShared {
    fn liveness(&self) -> std::sync::MutexGuard<'_, Liveness> {
        self.liveness
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Publishes the execution domain's first terminal reason and wakes every
    /// completion future. Native failures from resources and queues both reach
    /// this authority, so no pending future depends on which API call happened
    /// to discover loss first.
    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        let (waiters, readbacks) = {
            let mut state = self.liveness();
            if matches!(state.status, DeviceStatus::Lost) {
                return;
            }
            state.status = DeviceStatus::Lost;
            state.info = Some(info);
            (
                std::mem::take(&mut state.completion_waiters),
                std::mem::take(&mut state.pending_readbacks),
            )
        };
        for (_, waiters) in waiters {
            for waker in waiters {
                waker.wake();
            }
        }
        for ticket in readbacks {
            ticket.set_status(ReadbackStatus::DeviceLost);
        }
    }

    pub(crate) fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.liveness().info.clone()
    }

    /// Registers atomically with the device-loss state. `Err` means loss won
    /// the race and the caller must return terminally without retaining a waker.
    pub(crate) fn register_completion_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> Result<(), DeviceLossInfo> {
        let mut state = self.liveness();
        if let Some(info) = &state.info {
            return Err(info.clone());
        }
        let waiters = state.completion_waiters.entry(serial).or_default();
        if !waiters.iter().any(|registered| registered.will_wake(waker)) {
            waiters.push(waker.clone());
        }
        Ok(())
    }

    pub(crate) fn wake_completion(&self, serial: u64) {
        let waiters = self.liveness().completion_waiters.remove(&serial);
        if let Some(waiters) = waiters {
            for waker in waiters {
                waker.wake();
            }
        }
    }

    /// Registers accepted readbacks with the device-wide loss authority. This
    /// is separate from a batch's staging retention: the latter publishes bytes
    /// on success, while this registry guarantees that a loss first observed by
    /// any unrelated native call still terminates every pending `read().await`.
    pub(crate) fn register_readbacks(&self, tickets: &[ReadbackTicket]) {
        let lost = {
            let mut state = self.liveness();
            if state.info.is_some() {
                true
            } else {
                for ticket in tickets {
                    if !state
                        .pending_readbacks
                        .iter()
                        .any(|pending| pending.id() == ticket.id())
                    {
                        state.pending_readbacks.push(ticket.clone());
                    }
                }
                false
            }
        };
        if lost {
            for ticket in tickets {
                ticket.set_status(ReadbackStatus::DeviceLost);
            }
        }
    }

    /// Linearizes successful fence completion against terminal device loss.
    ///
    /// The closure runs while the loss authority is locked. If loss won first,
    /// it is not called; if completion won first, its tickets are removed from
    /// the pending-loss set before loss can be published. This is the boundary
    /// that prevents a completion future or readback from changing terminal
    /// meaning after observers have already seen `DeviceLost`.
    pub(crate) fn commit_completion<R>(
        &self,
        tickets: &[ReadbackTicket],
        commit: impl FnOnce() -> R,
    ) -> Result<R, DeviceLossInfo> {
        let mut state = self.liveness();
        if let Some(info) = &state.info {
            return Err(info.clone());
        }
        let result = commit();
        state.pending_readbacks.retain(|pending| {
            !tickets
                .iter()
                .any(|completed| completed.id() == pending.id())
        });
        Ok(result)
    }
}

impl Drop for VulkanShared {
    fn drop(&mut self) {
        // Every native child retains this one Arc, so reaching the final drop
        // proves all resource, staging and command-spine owners are gone. Future
        // descriptor pools/pipeline caches must join this same ownership domain
        // rather than introducing a second device lifetime registry.
        unsafe { self.device.destroy_device(None) };
    }
}

impl DeviceBackend for VulkanDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }

    fn capability_facts(&self) -> CapabilityFacts {
        self.facts.clone()
    }

    fn submission_capabilities(&self) -> SubmissionCapabilities {
        self.submission.clone()
    }

    fn object_id(&self) -> ObjectId {
        self.object
    }

    fn status(&self) -> DeviceStatus {
        self.shared.liveness().status
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.shared.loss_info()
    }

    fn poll(&self) -> RhiResult<()> {
        self.command.poll();
        if let Some(info) = self.shared.loss_info() {
            return Err(
                RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned())
                    .at("VulkanDevice::poll"),
            );
        }
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        self.command
            .wait_idle()
            .map_err(|failure| self.observe_failure(failure))
    }

    fn create_buffer(
        &self,
        descriptor: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::BufferBackend>> {
        resource::create_buffer(self.shared.clone(), descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::BufferBackend>)
            .map_err(|result| {
                self.observe_failure(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanDevice::create_buffer",
                )))
            })
    }

    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        resource::create_texture(self.shared.clone(), descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::TextureBackend>)
            .map_err(|result| {
                self.observe_failure(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanDevice::create_texture",
                )))
            })
    }

    fn create_texture_view(
        &self,
        texture: &crate::api::resource::Texture,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        let Some(native_texture) = texture
            .native()
            .as_any()
            .downcast_ref::<resource::VulkanTexture>()
        else {
            return self.unsupported("texture-view creation for a non-Vulkan texture");
        };
        resource::create_texture_view(
            self.shared.clone(),
            native_texture,
            texture.descriptor(),
            descriptor,
        )
        .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::TextureViewBackend>)
        .map_err(|result| {
            self.observe_failure(VulkanFailure::Native(ffi::NativeError::new(
                result,
                "VulkanDevice::create_texture_view",
            )))
        })
    }

    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        resource::create_sampler(self.shared.clone(), descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::SamplerBackend>)
            .map_err(|result| {
                self.observe_failure(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanDevice::create_sampler",
                )))
            })
    }

    fn create_shader(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<Box<dyn crate::api::shader::backend::ShaderModuleBackend>> {
        shader::create_shader(self.shared.clone(), artifact)
            .map(|value| {
                Box::new(value) as Box<dyn crate::api::shader::backend::ShaderModuleBackend>
            })
            .map_err(|result| {
                self.observe_failure(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanDevice::create_shader_module",
                )))
            })
    }

    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        binding::create_bind_group(self.shared.clone(), descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::binding::backend::BindGroupBackend>)
            .map_err(|failure| self.observe_failure(failure))
    }

    fn create_compute_pipeline(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>> {
        pipeline::create_compute_pipeline(self.shared.clone(), descriptor)
            .map(|value| {
                Box::new(value) as Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>
            })
            .map_err(|failure| self.observe_failure(failure))
    }

    fn create_raster_pipeline(
        &self,
        _: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::api::pipeline::backend::RasterPipelineBackend>> {
        self.unsupported("raster-pipeline creation")
    }

    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        self.command
            .submit(request)
            .map_err(|failure| self.observe_failure(failure))
    }

    fn completion(&self, serial: u64) -> CompletionState {
        self.command.completion(serial)
    }

    fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &std::task::Waker,
    ) -> CompletionState {
        self.command.completion_or_register_waker(serial, waker)
    }
}
