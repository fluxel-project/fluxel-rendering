//! WebGPU device façade over the owner-thread browser registry.

use std::collections::BTreeSet;

use js_sys::Reflect;
use wasm_bindgen::JsValue;

use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceInstanceId, ObjectId};
use crate::api::platform::backend::DeviceBackend;
use crate::api::platform::{AdapterId, AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::submission::{CompletionState, SubmissionCapabilities};

use super::capabilities::{WebGpuCapabilityInput, WebGpuLimits};
use super::command::WebGpuCommandSpine;
use super::presentation::WebGpuPresentation;
use super::registry::{self, WebGpuAdapterMetadata, WebGpuDriver, WebGpuRegistration};

pub(super) struct WebGpuDevice {
    driver: WebGpuDriver,
    adapter: AdapterInfo,
    facts: CapabilityFacts,
    submission: SubmissionCapabilities,
    object: ObjectId,
    command: WebGpuCommandSpine,
    presentation: WebGpuPresentation,
}

impl WebGpuDevice {
    pub(super) fn from_registration(
        registration: WebGpuRegistration,
        provider: DeviceInstanceId,
        metadata: WebGpuAdapterMetadata,
    ) -> RhiResult<Self> {
        let input = probe_capabilities(registration)?;
        let (facts, submission) = input.into_capabilities();
        let adapter = AdapterInfo::new(
            AdapterId::new(provider.as_u64(), 1),
            adapter_name(&metadata),
            BackendKind::WebGpu,
            None,
            None,
            AvailableCapabilities::from_facts(facts.clone()),
        );
        let driver = WebGpuDriver::new(registration);
        Ok(Self {
            command: WebGpuCommandSpine::new(driver.clone()),
            presentation: WebGpuPresentation::new(driver.clone()),
            driver,
            adapter,
            facts,
            submission,
            object: ObjectId::next(),
        })
    }

    fn lost_error(&self, operation: &'static str) -> RhiError {
        let message = self
            .loss_info()
            .map(|info| info.message().to_owned())
            .unwrap_or_else(|| "the WebGPU device is no longer registered".into());
        RhiError::new(RhiErrorKind::DeviceLost, message).at(operation)
    }
}

impl DeviceBackend for WebGpuDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::WebGpu
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
        registry::device_status(self.driver.registration()).unwrap_or(DeviceStatus::Lost)
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        registry::device_loss(self.driver.registration())
    }

    fn poll(&self) -> RhiResult<()> {
        self.command.poll()?;
        if self.status() == DeviceStatus::Lost {
            return Err(self.lost_error("WebGpuDevice::poll"));
        }
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "WebGPU cannot synchronously block the browser owner thread for queue completion",
        )
        .at("WebGpuDevice::wait_idle"))
    }

    fn create_buffer(
        &self,
        descriptor: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::BufferBackend>> {
        super::resource::create_buffer(&self.driver, descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::BufferBackend>)
    }

    fn map_buffer(
        &self,
        buffer: &crate::api::resource::Buffer,
        mode: crate::api::resource::MapMode,
        range: crate::api::resource::BufferRange,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
        super::resource::map_buffer(&self.driver, buffer, mode, range)
    }

    fn create_query_set(
        &self,
        descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::QuerySetBackend>> {
        super::resource::create_query_set(&self.driver, descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::QuerySetBackend>)
    }

    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        super::resource::create_texture(&self.driver, descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::TextureBackend>)
    }

    fn create_texture_view(
        &self,
        texture: &crate::api::resource::Texture,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        super::resource::create_texture_view(&self.driver, texture, descriptor).map(|value| {
            Box::new(value) as Box<dyn crate::api::resource::backend::TextureViewBackend>
        })
    }

    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        super::resource::create_sampler(&self.driver, descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::SamplerBackend>)
    }

    fn create_shader(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<Box<dyn crate::api::shader::backend::ShaderModuleBackend>> {
        super::shader::create_shader(&self.driver, artifact).map(|value| {
            Box::new(value) as Box<dyn crate::api::shader::backend::ShaderModuleBackend>
        })
    }

    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        super::binding::create_bind_group(&self.driver, descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::binding::backend::BindGroupBackend>)
    }

    fn create_compute_pipeline(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>> {
        super::pipeline::create_compute_pipeline(&self.driver, descriptor).map(|value| {
            Box::new(value) as Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>
        })
    }

    fn create_raster_pipeline(
        &self,
        descriptor: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::api::pipeline::backend::RasterPipelineBackend>> {
        super::pipeline::create_raster_pipeline(&self.driver, descriptor).map(|value| {
            Box::new(value) as Box<dyn crate::api::pipeline::backend::RasterPipelineBackend>
        })
    }

    fn presentation(&self) -> Option<&dyn crate::api::presentation::backend::PresentationBackend> {
        Some(&self.presentation)
    }

    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        self.command.submit(request)
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

fn adapter_name(metadata: &WebGpuAdapterMetadata) -> String {
    let mut parts = vec![metadata.name.clone()];
    for value in [&metadata.vendor, &metadata.architecture, &metadata.device]
        .into_iter()
        .flatten()
    {
        if !value.is_empty() && !parts.iter().any(|part| part == value) {
            parts.push(value.clone());
        }
    }
    parts.join(" / ")
}

fn probe_capabilities(registration: WebGpuRegistration) -> RhiResult<WebGpuCapabilityInput> {
    registry::with_device_handles(registration, |handles| WebGpuCapabilityInput {
        adapter_features: feature_names(&handles.adapter),
        device_features: feature_names(&handles.device),
        limits: limits(&handles.device),
    })
    .ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::DeviceLost,
            "WebGPU device disappeared during capability discovery",
        )
        .at("WebGpuDevice::from_registration")
    })
}

fn feature_names(owner: &JsValue) -> BTreeSet<String> {
    super::js::supported_features(
        owner,
        &[
            "float32-filterable",
            "readonly_and_readwrite_storage_textures",
            "texture-compression-bc",
            "texture-compression-etc2",
            "texture-compression-astc",
            "indirect-first-instance",
        ],
    )
    .into_iter()
    .collect()
}

fn limits(device: &JsValue) -> WebGpuLimits {
    let limits = Reflect::get(device, &JsValue::from_str("limits")).unwrap_or(JsValue::UNDEFINED);
    let u32_value = |name| {
        number(&limits, name)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(0)
    };
    let u64_value = |name| number(&limits, name).unwrap_or(0);
    WebGpuLimits {
        max_buffer_size: u64_value("maxBufferSize"),
        max_texture_dimension_1d: u32_value("maxTextureDimension1D"),
        max_texture_dimension_2d: u32_value("maxTextureDimension2D"),
        max_texture_dimension_3d: u32_value("maxTextureDimension3D"),
        max_texture_array_layers: u32_value("maxTextureArrayLayers"),
        max_bind_groups: u32_value("maxBindGroups"),
        max_bindings_per_bind_group: u32_value("maxBindingsPerBindGroup"),
        max_uniform_buffer_binding_size: u64_value("maxUniformBufferBindingSize"),
        max_storage_buffer_binding_size: u64_value("maxStorageBufferBindingSize"),
        min_uniform_buffer_offset_alignment: u64_value("minUniformBufferOffsetAlignment"),
        min_storage_buffer_offset_alignment: u64_value("minStorageBufferOffsetAlignment"),
        max_color_attachments: u32_value("maxColorAttachments"),
        max_vertex_buffers: u32_value("maxVertexBuffers"),
        max_vertex_attributes: u32_value("maxVertexAttributes"),
        max_vertex_buffer_array_stride: u64_value("maxVertexBufferArrayStride"),
        max_inter_stage_shader_variables: u32_value("maxInterStageShaderVariables"),
        max_compute_invocations_per_workgroup: u32_value("maxComputeInvocationsPerWorkgroup"),
        max_compute_workgroup_size_x: u32_value("maxComputeWorkgroupSizeX"),
        max_compute_workgroup_size_y: u32_value("maxComputeWorkgroupSizeY"),
        max_compute_workgroup_size_z: u32_value("maxComputeWorkgroupSizeZ"),
        max_compute_workgroups_per_dimension: u32_value("maxComputeWorkgroupsPerDimension"),
        max_compute_workgroup_storage_size: u64_value("maxComputeWorkgroupStorageSize"),
    }
}

fn number(value: &JsValue, name: &str) -> Option<u64> {
    let value = Reflect::get(value, &JsValue::from_str(name))
        .ok()?
        .as_f64()?;
    (value.is_finite() && value >= 0.0 && value <= 9_007_199_254_740_991.0).then_some(value as u64)
}
