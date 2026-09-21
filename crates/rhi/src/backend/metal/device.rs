//! Metal execution-domain ownership and the portable backend seam.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};

use crate::api::capability::CapabilityFacts;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::backend::{DeviceBackend, ready_creation_request};
use crate::api::platform::{AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};

use super::command::MetalCommandSpine;

/// One retained ownership domain for the native device and queue. Resources,
/// pipelines and command callbacks retain this object rather than independently
/// reference-counting pieces of the same execution domain.
pub(super) struct MetalShared {
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

unsafe impl Send for MetalShared {}
unsafe impl Sync for MetalShared {}

pub(super) struct MetalDevice {
    adapter: AdapterInfo,
    object: ObjectId,
    shared: Arc<MetalShared>,
    command: MetalCommandSpine,
    presentation: super::presentation::MetalPresentation,
    facts: CapabilityFacts,
    submission: SubmissionCapabilities,
}

impl MetalDevice {
    pub(super) fn new(
        adapter: AdapterInfo,
        native: Retained<ProtocolObject<dyn MTLDevice>>,
        targets: Arc<super::presentation::MetalTargetRegistry>,
    ) -> RhiResult<Self> {
        let queue = native.newCommandQueue().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal returned no command queue",
            )
            .at("MetalDevice::new")
        })?;
        let facts = super::facts::baseline(&native);
        let shared = Arc::new(MetalShared {
            device: native,
            queue,
        });
        let presentation_loss = Arc::new(super::presentation::MetalPresentationLoss::default());
        let command = MetalCommandSpine::new(Arc::clone(&shared), Arc::clone(&presentation_loss))?;
        let presentation = super::presentation::MetalPresentation::new(
            shared.device.clone(),
            targets,
            Arc::clone(&presentation_loss),
        );
        let submission = SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::Graphics,
            LaneWorkDomains::RASTER
                .union(LaneWorkDomains::COMPUTE)
                .union(LaneWorkDomains::COPY),
        )]);
        Ok(Self {
            adapter,
            object: ObjectId::next(),
            shared,
            command,
            presentation,
            facts,
            submission,
        })
    }

    fn unsupported<T>(&self, operation: &'static str) -> RhiResult<T> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!("Metal lowering is unavailable for {operation}"),
        )
        .at("MetalDevice"))
    }
}

impl DeviceBackend for MetalDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Metal
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
        self.command.status()
    }
    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.command.loss_info()
    }
    fn poll(&self) -> RhiResult<()> {
        self.command.poll()
    }
    fn wait_idle(&self) -> RhiResult<()> {
        self.command.wait_idle()
    }

    fn create_buffer(
        &self,
        descriptor: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::BufferBackend>> {
        super::resource::create_buffer(
            &self.shared.device,
            descriptor,
            self.command.mapping_state(),
        )
        .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::BufferBackend>)
    }

    fn map_buffer(
        &self,
        buffer: &crate::api::resource::Buffer,
        mode: crate::api::resource::MapMode,
        range: crate::api::resource::BufferRange,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
        let native = buffer
            .native()
            .as_any()
            .downcast_ref::<super::resource::MetalBuffer>()
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "buffer is not owned by Metal")
            })?;
        super::resource::map_buffer(native, mode, range)
    }

    fn create_query_set(
        &self,
        _descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::QuerySetBackend>> {
        self.unsupported("query sets")
    }

    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        super::resource::create_texture(&self.shared.device, descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::TextureBackend>)
    }

    fn create_texture_view(
        &self,
        texture: &crate::api::resource::Texture,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        let native = texture
            .native()
            .as_any()
            .downcast_ref::<super::resource::MetalTexture>()
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "texture is not owned by Metal")
            })?;
        let format = descriptor.format.unwrap_or(texture.descriptor().format);
        let format = super::format::metal_format(format).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "texture-view format has no Metal lowering",
            )
        })?;
        super::resource::create_texture_view(native, descriptor, format).map(|value| {
            Box::new(value) as Box<dyn crate::api::resource::backend::TextureViewBackend>
        })
    }

    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        super::resource::create_sampler(&self.shared.device, descriptor)
            .map(|value| Box::new(value) as Box<dyn crate::api::resource::backend::SamplerBackend>)
    }

    fn create_shader_request(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::shader::backend::ShaderModuleBackend,
                >,
        >,
    > {
        super::shader::create_shader(Arc::clone(&self.shared), artifact)
            .map(|value| {
                Box::new(value) as Box<dyn crate::api::shader::backend::ShaderModuleBackend>
            })
            .map(ready_creation_request)
    }

    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        Ok(Box::new(super::binding::create_bind_group(descriptor)))
    }

    fn create_compute_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::ComputePipelineBackend,
                >,
        >,
    > {
        super::pipeline::create_compute_pipeline(Arc::clone(&self.shared), descriptor)
            .map(|value| {
                Box::new(value) as Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>
            })
            .map(ready_creation_request)
    }

    fn create_raster_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::RasterPipelineBackend,
                >,
        >,
    > {
        super::pipeline::create_raster_pipeline(Arc::clone(&self.shared), descriptor)
            .map(|value| {
                Box::new(value) as Box<dyn crate::api::pipeline::backend::RasterPipelineBackend>
            })
            .map(ready_creation_request)
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

    fn completion(&self, serial: u64) -> crate::api::submission::CompletionState {
        self.command.completion(serial)
    }

    fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &std::task::Waker,
    ) -> crate::api::submission::CompletionState {
        self.command.completion_or_register_waker(serial, waker)
    }
}
