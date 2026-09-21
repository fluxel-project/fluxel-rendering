use std::any::Any;
use std::sync::{Arc, Mutex};
use std::task::Waker;

use crate::api::capability::CapabilityFacts;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::backend::DeviceBackend;
use crate::api::platform::provider::{AdapterInfo, BackendKind};
use crate::api::platform::{DeviceLossInfo, DeviceStatus};
use crate::api::presentation::PresentationTarget;
use crate::api::submission::{
    CompletionFailure, CompletionState, LaneWorkDomains, SubmissionCapabilities,
    SubmissionLaneClass, SubmissionLaneId, SubmissionLaneInfo,
};

/// An owner-context native object name.  Zero is GL's "no object" sentinel and
/// can never back a published RHI handle.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct GlObjectName(u32);

impl GlObjectName {
    pub(crate) fn new(value: u32, operation: &'static str) -> RhiResult<Self> {
        if value == 0 {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "GL driver returned object name zero for a successful creation",
            )
            .at(operation));
        }
        Ok(Self(value))
    }
    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GlObjectKind {
    Buffer,
    Texture,
    /// A view may be a virtual GL view sharing its texture name. `destroy` must
    /// then release only driver-side view metadata, never delete the texture.
    TextureView,
    Sampler,
    QuerySet,
    Shader,
    /// GL has no descriptor-set object. A driver may use a unique packet id and
    /// release cached binding metadata here; it must not claim a no-op packet is
    /// executable unless its submit lowerer consumes that packet.
    BindGroup,
    ComputePipeline,
    RasterPipeline,
}

/// GL-private references handed to a native/browser lowerer. They carry only
/// validated GL object names, never portable handles or platform context state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlBufferRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlTextureRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlTextureViewRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlSamplerRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlShaderRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlBindGroupRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlComputePipelineRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlRasterPipelineRef {
    pub(crate) name: GlObjectName,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlQuerySetRef {
    pub(crate) name: GlObjectName,
}

/// Pipeline creation packets pair the only native shader inputs with the
/// already-validated portable fixed-state description. Native/browser drivers
/// must consume the refs and never downcast `ShaderModule` themselves.
pub(crate) struct GlComputePipelinePacket<'a> {
    pub(crate) shader: GlShaderRef,
    pub(crate) descriptor: &'a crate::api::pipeline::ComputePipelineDescriptor,
}
pub(crate) struct GlRasterPipelinePacket<'a> {
    pub(crate) vertex: GlShaderRef,
    pub(crate) fragment: Option<GlShaderRef>,
    pub(crate) descriptor: &'a crate::api::pipeline::RasterPipelineDescriptor,
}

/// Canonical, object-name-only packet built before a GL bind-group creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GlBindGroupPacket {
    pub(crate) entries: Vec<GlBindGroupEntry>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GlBindGroupEntry {
    pub(crate) slot: u32,
    pub(crate) resource: GlBindingResource,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GlBindingResource {
    Buffer {
        buffer: GlBufferRef,
        offset: u64,
        size: u64,
    },
    Texture(GlTextureViewRef),
    Sampler(GlSamplerRef),
    BufferArray(Vec<(GlBufferRef, u64, u64)>),
    TextureArray(Vec<GlTextureViewRef>),
    SamplerArray(Vec<GlSamplerRef>),
}

/// Platform-side lowering view. The GL driver receives this rather than the
/// generic submission seam; command detail remains crate-private to this crate.
pub(crate) struct GlSubmissionPlan<'a> {
    pub(crate) batches: Vec<GlSubmissionBatch<'a>>,
}
pub(crate) struct GlSubmissionBatch<'a> {
    pub(crate) point: crate::api::submission::PlanPoint,
    pub(crate) commands: Vec<&'a crate::api::command::record::RecordedCommand>,
}

/// Backend-private one-way notification from a context owner to the v13
/// device liveness authority.  This deliberately carries no context/session
/// object: a GL context loss invalidates one `DeviceIdentity`, full stop.
pub(crate) trait GlLossSink: Send + Sync + 'static {
    fn report_context_loss(&self, info: DeviceLossInfo);
}

/// Owner-context dispatch seam used by native GL and browser implementations.
///
/// A GL context is thread-affine while public RHI handles are `Send + Sync`.
/// Implementors marshal each call to the current WGL/EGL/GLX owner or browser
/// owner thread.  The seam carries no native context, browser session, or token
/// into the common API.  `dispatch` must not report success unless the operation
/// actually reached that owner context.
pub(crate) trait GlExecutionDriver: Send + Sync + 'static {
    /// Marshals `operation` to the context's owner. Implementations perform the
    /// real GL/browser work synchronously before returning.
    fn dispatch(&self, operation: &'static str) -> RhiResult<()>;

    /// Installs the private terminal-loss notification route for this adopted
    /// context. Drivers use it for asynchronous browser/worker observations
    /// which cannot be returned through a synchronous public Device verb.
    fn install_loss_sink(&self, _: Arc<dyn GlLossSink>) {}

    fn create_buffer(
        &self,
        _: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_buffer")
    }
    fn create_texture(
        &self,
        _: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_texture")
    }
    fn create_texture_view(
        &self,
        _: GlTextureRef,
        _: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_texture_view")
    }
    fn create_sampler(
        &self,
        _: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_sampler")
    }
    fn create_query_set(
        &self,
        _: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<GlObjectName> {
        unsupported("create_query_set")
    }
    fn create_shader(&self, _: &crate::api::shader::ShaderArtifact) -> RhiResult<GlObjectName> {
        unsupported("create_shader")
    }
    fn create_bind_group(&self, _: &GlBindGroupPacket) -> RhiResult<GlObjectName> {
        unsupported("create_bind_group")
    }
    fn create_compute_pipeline(&self, _: GlComputePipelinePacket<'_>) -> RhiResult<GlObjectName> {
        unsupported("create_compute_pipeline")
    }
    fn create_raster_pipeline(&self, _: GlRasterPipelinePacket<'_>) -> RhiResult<GlObjectName> {
        unsupported("create_raster_pipeline")
    }
    fn map_buffer(
        &self,
        _: GlBufferRef,
        _: crate::api::resource::MapMode,
        _: crate::api::resource::BufferRange,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
        unsupported("map_buffer")
    }
    fn submit(
        &self,
        _: GlSubmissionPlan<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        unsupported("submit")
    }
    fn completion(&self, serial: u64) -> CompletionState {
        CompletionState::Failed(CompletionFailure::new(format!(
            "GL completion serial {serial} was never accepted"
        )))
    }
    fn completion_or_register_waker(&self, serial: u64, _: &Waker) -> CompletionState {
        self.completion(serial)
    }
    /// Destruction is called by wrapper `Drop`; implementations must schedule it
    /// onto the owner context and tolerate loss, where native deletion is moot.
    fn destroy(&self, _: GlObjectKind, _: GlObjectName) {}

    fn poll(&self) -> RhiResult<()> {
        self.dispatch("GlExecutionDriver::poll")
    }
    fn wait_idle(&self) -> RhiResult<()> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the adopted GL driver has no verified idle wait route",
        ))
    }

    fn supports_presentation(&self, _target: &PresentationTarget) -> RhiResult<bool> {
        Ok(false)
    }
    fn presentation_capabilities(
        &self,
        _: crate::api::identity::ObjectId,
    ) -> RhiResult<crate::api::presentation::PresentationTargetCapabilities> {
        unsupported("presentation_capabilities")
    }
    fn configure_presentation(
        &self,
        _: crate::api::identity::ObjectId,
        _: &crate::api::presentation::PresentationConfiguration,
    ) -> RhiResult<super::presentation::GlPresentationLease> {
        unsupported("configure_presentation")
    }
    fn lease_capabilities(
        &self,
        _: super::presentation::GlPresentationLease,
    ) -> RhiResult<crate::api::presentation::PresentationTargetCapabilities> {
        unsupported("lease_capabilities")
    }
    fn reconfigure_or_register_waker(
        &self,
        _: super::presentation::GlPresentationLease,
        _: &crate::api::presentation::PresentationConfiguration,
        _: &Waker,
    ) -> std::task::Poll<RhiResult<()>> {
        std::task::Poll::Ready(unsupported("reconfigure_presentation"))
    }
    fn try_acquire(
        &self,
        _: super::presentation::GlPresentationLease,
    ) -> Result<
        Option<super::presentation::GlAcquiredFramebuffer>,
        crate::api::presentation::AcquireError,
    > {
        Err(crate::api::presentation::AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "the adopted GL driver has no acquired framebuffer",
        ))
    }
    fn acquire_or_register_waker(
        &self,
        _: super::presentation::GlPresentationLease,
        _: &Waker,
    ) -> std::task::Poll<
        Result<super::presentation::GlAcquiredFramebuffer, crate::api::presentation::AcquireError>,
    > {
        std::task::Poll::Ready(Err(crate::api::presentation::AcquireError::new(
            crate::api::presentation::AcquireErrorKind::NotReady,
            "the adopted GL driver has no acquired framebuffer",
        )))
    }
    fn abandon(
        &self,
        _: super::presentation::GlPresentationLease,
        _: crate::api::presentation::AcquiredFrameId,
    ) -> RhiResult<()> {
        unsupported("abandon_frame")
    }
    fn abandon_no_throw(
        &self,
        _: super::presentation::GlPresentationLease,
        _: crate::api::presentation::AcquiredFrameId,
    ) {
    }
    fn release_presentation(&self, _: super::presentation::GlPresentationLease) {}
    fn present(
        &self,
        _: super::presentation::GlAcquiredFramebuffer,
        _: crate::api::presentation::PresentReceiptId,
    ) {
    }
    fn terminate_present(
        &self,
        _: super::presentation::GlAcquiredFramebuffer,
        _: crate::api::presentation::PresentReceiptId,
        _: crate::api::presentation::PresentState,
    ) {
    }
    fn present_state(
        &self,
        _: crate::api::presentation::PresentReceiptId,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        unsupported("present_state")
    }
    fn present_state_or_register_waker(
        &self,
        _: crate::api::presentation::PresentReceiptId,
        _: &Waker,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        unsupported("present_state_or_register_waker")
    }
    /// Called once by the GL device terminal-loss authority. Implementations wake
    /// all pending acquire and present futures before returning.
    fn device_lost(&self, _: &DeviceLossInfo) {}
}

impl crate::api::presentation::backend::PresentationBackend for GlDevice {
    fn capabilities(
        &self,
        target: ObjectId,
    ) -> RhiResult<crate::api::presentation::PresentationTargetCapabilities> {
        self.active("GlDevice::presentation_capabilities")?;
        self.observe(
            self.driver.presentation_capabilities(target),
            "GlDevice::presentation_capabilities",
        )
    }
    fn configure(
        &self,
        _: crate::api::identity::DeviceIdentity,
        target: ObjectId,
        config: &crate::api::presentation::PresentationConfiguration,
    ) -> RhiResult<Box<dyn crate::api::presentation::backend::ConfiguredPresentationBackend>> {
        self.active("GlDevice::configure_presentation")?;
        let lease = self.observe(
            self.driver.configure_presentation(target, config),
            "GlDevice::configure_presentation",
        )?;
        Ok(super::presentation::configured(
            Arc::clone(&self.driver),
            Arc::clone(&self.loss_sink),
            lease,
        ))
    }
    fn present_state(
        &self,
        receipt: crate::api::presentation::PresentReceiptId,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        self.active("GlDevice::present_state")?;
        self.observe(
            self.driver.present_state(receipt),
            "GlDevice::present_state",
        )
    }
    fn present_state_or_register_waker(
        &self,
        receipt: crate::api::presentation::PresentReceiptId,
        waker: &Waker,
    ) -> RhiResult<crate::api::presentation::PresentState> {
        self.active("GlDevice::present_state_or_register_waker")?;
        self.observe(
            self.driver.present_state_or_register_waker(receipt, waker),
            "GlDevice::present_state_or_register_waker",
        )
    }
}

fn unsupported<T>(operation: &'static str) -> RhiResult<T> {
    Err(RhiError::new(
        RhiErrorKind::Unsupported,
        "the adopted GL driver did not install this v13 lowering",
    )
    .at(operation))
}

macro_rules! native_wrapper {
    ($name:ident, $kind:ident, $trait:path) => {
        struct $name {
            driver: Arc<dyn GlExecutionDriver>,
            name: GlObjectName,
        }
        impl Drop for $name {
            fn drop(&mut self) {
                self.driver.destroy(GlObjectKind::$kind, self.name);
            }
        }
        impl $trait for $name {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
    };
}

native_wrapper!(
    GlBuffer,
    Buffer,
    crate::api::resource::backend::BufferBackend
);
native_wrapper!(
    GlTexture,
    Texture,
    crate::api::resource::backend::TextureBackend
);
native_wrapper!(
    GlTextureView,
    TextureView,
    crate::api::resource::backend::TextureViewBackend
);
native_wrapper!(
    GlSampler,
    Sampler,
    crate::api::resource::backend::SamplerBackend
);
native_wrapper!(
    GlQuerySet,
    QuerySet,
    crate::api::resource::backend::QuerySetBackend
);
native_wrapper!(
    GlShader,
    Shader,
    crate::api::shader::backend::ShaderModuleBackend
);
native_wrapper!(
    GlBindGroup,
    BindGroup,
    crate::api::binding::backend::BindGroupBackend
);
native_wrapper!(
    GlComputePipeline,
    ComputePipeline,
    crate::api::pipeline::backend::ComputePipelineBackend
);
native_wrapper!(
    GlRasterPipeline,
    RasterPipeline,
    crate::api::pipeline::backend::RasterPipelineBackend
);

struct Liveness {
    status: DeviceStatus,
    info: Option<DeviceLossInfo>,
    waiters: Vec<Waker>,
}

impl Liveness {
    fn report(&mut self, info: DeviceLossInfo) -> Option<Vec<Waker>> {
        if matches!(self.status, DeviceStatus::Lost) {
            return None;
        }
        self.status = DeviceStatus::Lost;
        self.info = Some(info);
        Some(core::mem::take(&mut self.waiters))
    }
}

struct GlLossAuthority {
    liveness: Arc<Mutex<Liveness>>,
}

impl GlLossAuthority {
    fn report(&self, info: DeviceLossInfo) -> bool {
        let waiters = self
            .liveness
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .report(info);
        let Some(waiters) = waiters else {
            return false;
        };
        for waker in waiters {
            waker.wake();
        }
        true
    }
}

impl GlLossSink for GlLossAuthority {
    fn report_context_loss(&self, info: DeviceLossInfo) {
        self.report(info);
    }
}

/// Context-owned GL device adapter.  Lowering is intentionally fail-closed
/// until the state-machine command executor is wired through `GlExecutionDriver`.
pub(crate) struct GlDevice {
    backend: BackendKind,
    adapter: AdapterInfo,
    facts: CapabilityFacts,
    object: ObjectId,
    driver: Arc<dyn GlExecutionDriver>,
    liveness: Arc<Mutex<Liveness>>,
    loss_sink: Arc<dyn GlLossSink>,
}

impl GlDevice {
    pub(super) fn new(
        backend: BackendKind,
        adapter: AdapterInfo,
        facts: CapabilityFacts,
        driver: Arc<dyn GlExecutionDriver>,
    ) -> Self {
        let liveness = Arc::new(Mutex::new(Liveness {
            status: DeviceStatus::Active,
            info: None,
            waiters: Vec::new(),
        }));
        let loss_sink: Arc<dyn GlLossSink> = Arc::new(GlLossAuthority {
            liveness: Arc::clone(&liveness),
        });
        driver.install_loss_sink(Arc::clone(&loss_sink));
        Self {
            backend,
            adapter,
            facts,
            object: ObjectId::next(),
            driver,
            liveness,
            loss_sink,
        }
    }

    /// Native/browser glue calls this once on `GL_CONTEXT_LOST`, WebGL
    /// `webglcontextlost`, or an equivalent terminal driver failure.  It wakes
    /// all completion waiters; later public verbs observe the stable loss state.
    pub(crate) fn mark_lost(&self, message: impl Into<String>) {
        let info = DeviceLossInfo::new(message.into());
        let authority = GlLossAuthority {
            liveness: Arc::clone(&self.liveness),
        };
        if authority.report(info.clone()) {
            self.driver.device_lost(&info);
        }
    }

    fn active(&self, operation: &'static str) -> RhiResult<()> {
        if let Some(info) = self.loss_info() {
            return Err(RhiError::new(RhiErrorKind::DeviceLost, info.message()).at(operation));
        }
        Ok(())
    }

    fn observe<T>(&self, result: RhiResult<T>, operation: &'static str) -> RhiResult<T> {
        if let Err(error) = &result {
            if error.kind() == RhiErrorKind::DeviceLost {
                self.mark_lost(error.to_string());
            }
        }
        result.map_err(|error| error.at(operation))
    }

    pub(crate) fn buffer_ref(buffer: &crate::api::resource::Buffer) -> RhiResult<GlBufferRef> {
        buffer
            .native()
            .as_any()
            .downcast_ref::<GlBuffer>()
            .map(|native| GlBufferRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::Unsupported, "buffer has no GL native backing")
            })
    }
    pub(crate) fn texture_ref(texture: &crate::api::resource::Texture) -> RhiResult<GlTextureRef> {
        texture
            .native()
            .as_any()
            .downcast_ref::<GlTexture>()
            .map(|native| GlTextureRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "texture has no GL native backing",
                )
            })
    }
    pub(crate) fn view_ref(
        view: &crate::api::resource::TextureView,
    ) -> RhiResult<GlTextureViewRef> {
        view.native()
            .as_any()
            .downcast_ref::<GlTextureView>()
            .map(|native| GlTextureViewRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "texture view has no GL native backing",
                )
            })
    }
    pub(crate) fn sampler_ref(sampler: &crate::api::resource::Sampler) -> RhiResult<GlSamplerRef> {
        sampler
            .native()
            .as_any()
            .downcast_ref::<GlSampler>()
            .map(|native| GlSamplerRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "sampler has no GL native backing",
                )
            })
    }
    pub(crate) fn shader_ref(shader: &crate::api::shader::ShaderModule) -> RhiResult<GlShaderRef> {
        shader
            .native()
            .as_any()
            .downcast_ref::<GlShader>()
            .map(|native| GlShaderRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "shader module has no GL native backing",
                )
            })
    }
    pub(crate) fn bind_group_ref(
        group: &crate::api::binding::BindGroup,
    ) -> RhiResult<GlBindGroupRef> {
        group
            .native()
            .as_any()
            .downcast_ref::<GlBindGroup>()
            .map(|native| GlBindGroupRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "bind group has no GL native backing",
                )
            })
    }
    pub(crate) fn compute_pipeline_ref(
        pipeline: &crate::api::pipeline::ComputePipeline,
    ) -> RhiResult<GlComputePipelineRef> {
        pipeline
            .native()
            .as_any()
            .downcast_ref::<GlComputePipeline>()
            .map(|native| GlComputePipelineRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "compute pipeline has no GL native backing",
                )
            })
    }
    pub(crate) fn raster_pipeline_ref(
        pipeline: &crate::api::pipeline::RasterPipeline,
    ) -> RhiResult<GlRasterPipelineRef> {
        pipeline
            .native()
            .as_any()
            .downcast_ref::<GlRasterPipeline>()
            .map(|native| GlRasterPipelineRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "raster pipeline has no GL native backing",
                )
            })
    }
    pub(crate) fn query_set_ref(set: &crate::api::query::QuerySet) -> RhiResult<GlQuerySetRef> {
        set.native()
            .as_any()
            .downcast_ref::<GlQuerySet>()
            .map(|native| GlQuerySetRef { name: native.name })
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "query set has no GL native backing",
                )
            })
    }
    fn bind_packet(
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<GlBindGroupPacket> {
        use crate::api::binding::BindingResource;
        let entries = descriptor.entries.iter().map(|entry| {
            let resource = match &entry.resource {
                BindingResource::Buffer(binding) => GlBindingResource::Buffer { buffer: Self::buffer_ref(&binding.buffer)?, offset: binding.range.offset, size: binding.range.size },
                BindingResource::Texture(view) => GlBindingResource::Texture(Self::view_ref(view)?),
                BindingResource::Sampler(sampler) => GlBindingResource::Sampler(Self::sampler_ref(sampler)?),
                BindingResource::BufferArray(bindings) => GlBindingResource::BufferArray(bindings.iter().map(|binding| Ok((Self::buffer_ref(&binding.buffer)?, binding.range.offset, binding.range.size))).collect::<RhiResult<_>>()?),
                BindingResource::TextureArray(views) => GlBindingResource::TextureArray(views.iter().map(Self::view_ref).collect::<RhiResult<_>>()?),
                BindingResource::SamplerArray(samplers) => GlBindingResource::SamplerArray(samplers.iter().map(Self::sampler_ref).collect::<RhiResult<_>>()?),
                BindingResource::AccelerationStructure(_) | BindingResource::AccelerationStructureArray(_) | BindingResource::ExternalTexture(_) => return Err(RhiError::new(RhiErrorKind::Unsupported, "this GL backend does not lower acceleration-structure or external-texture bindings")),
            };
            Ok(GlBindGroupEntry { slot: entry.slot.get(), resource })
        }).collect::<RhiResult<_>>()?;
        Ok(GlBindGroupPacket { entries })
    }
    fn submission_plan<'a>(
        request: &'a crate::api::submission::backend::SubmissionRequest<'a>,
    ) -> GlSubmissionPlan<'a> {
        GlSubmissionPlan {
            batches: request
                .batches
                .iter()
                .map(|batch| GlSubmissionBatch {
                    point: batch.point,
                    commands: batch
                        .work
                        .iter()
                        .flat_map(|work| work.commands().iter())
                        .collect(),
                })
                .collect(),
        }
    }
}

impl DeviceBackend for GlDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }
    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }
    fn capability_facts(&self) -> CapabilityFacts {
        self.facts.clone()
    }
    fn submission_capabilities(&self) -> SubmissionCapabilities {
        // GL has one ordered command stream.  The facts builder must not publish
        // Compute without also supplying a GL compute executor; this baseline
        // therefore admits raster/copy only.
        SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
        )])
    }
    fn object_id(&self) -> ObjectId {
        self.object
    }
    fn status(&self) -> DeviceStatus {
        self.liveness
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .status
    }
    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.liveness
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .info
            .clone()
    }
    fn poll(&self) -> RhiResult<()> {
        self.active("GlDevice::poll")?;
        self.observe(self.driver.poll(), "GlDevice::poll")
    }
    fn wait_idle(&self) -> RhiResult<()> {
        self.active("GlDevice::wait_idle")?;
        self.observe(self.driver.wait_idle(), "GlDevice::wait_idle")
    }
    fn presentation(&self) -> Option<&dyn crate::api::presentation::backend::PresentationBackend> {
        Some(self)
    }

    fn map_buffer(
        &self,
        buffer: &crate::api::resource::Buffer,
        mode: crate::api::resource::MapMode,
        range: crate::api::resource::BufferRange,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
        self.active("GlDevice::map_buffer")?;
        let native = Self::buffer_ref(buffer).map_err(|error| error.at("GlDevice::map_buffer"))?;
        self.observe(
            self.driver.map_buffer(native, mode, range),
            "GlDevice::map_buffer",
        )
    }

    fn create_buffer(
        &self,
        descriptor: &crate::api::resource::buffer::BufferDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::BufferBackend>> {
        self.active("GlDevice::create_buffer")?;
        self.observe(
            self.driver.create_buffer(descriptor),
            "GlDevice::create_buffer",
        )
        .map(|name| {
            Box::new(GlBuffer {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_query_set(
        &self,
        descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::QuerySetBackend>> {
        self.active("GlDevice::create_query_set")?;
        self.observe(
            self.driver.create_query_set(descriptor),
            "GlDevice::create_query_set",
        )
        .map(|name| {
            Box::new(GlQuerySet {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        self.active("GlDevice::create_texture")?;
        self.observe(
            self.driver.create_texture(descriptor),
            "GlDevice::create_texture",
        )
        .map(|name| {
            Box::new(GlTexture {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_texture_view(
        &self,
        texture: &crate::api::resource::Texture,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        self.active("GlDevice::create_texture_view")?;
        let native_texture = Self::texture_ref(texture)
            .map_err(|error| error.at("GlDevice::create_texture_view"))?;
        self.observe(
            self.driver.create_texture_view(native_texture, descriptor),
            "GlDevice::create_texture_view",
        )
        .map(|name| {
            Box::new(GlTextureView {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        self.active("GlDevice::create_sampler")?;
        self.observe(
            self.driver.create_sampler(descriptor),
            "GlDevice::create_sampler",
        )
        .map(|name| {
            Box::new(GlSampler {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_shader(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<Box<dyn crate::api::shader::backend::ShaderModuleBackend>> {
        self.active("GlDevice::create_shader")?;
        self.observe(
            self.driver.create_shader(artifact),
            "GlDevice::create_shader",
        )
        .map(|name| {
            Box::new(GlShader {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        self.active("GlDevice::create_bind_group")?;
        let packet = Self::bind_packet(descriptor)
            .map_err(|error| error.at("GlDevice::create_bind_group"))?;
        self.observe(
            self.driver.create_bind_group(&packet),
            "GlDevice::create_bind_group",
        )
        .map(|name| {
            Box::new(GlBindGroup {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_compute_pipeline(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>> {
        self.active("GlDevice::create_compute_pipeline")?;
        let packet = GlComputePipelinePacket {
            shader: Self::shader_ref(&descriptor.shader)
                .map_err(|error| error.at("GlDevice::create_compute_pipeline"))?,
            descriptor,
        };
        self.observe(
            self.driver.create_compute_pipeline(packet),
            "GlDevice::create_compute_pipeline",
        )
        .map(|name| {
            Box::new(GlComputePipeline {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn create_raster_pipeline(
        &self,
        descriptor: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::api::pipeline::backend::RasterPipelineBackend>> {
        self.active("GlDevice::create_raster_pipeline")?;
        let packet = GlRasterPipelinePacket {
            vertex: Self::shader_ref(&descriptor.vertex)
                .map_err(|error| error.at("GlDevice::create_raster_pipeline"))?,
            fragment: descriptor
                .fragment
                .as_ref()
                .map(Self::shader_ref)
                .transpose()
                .map_err(|error| error.at("GlDevice::create_raster_pipeline"))?,
            descriptor,
        };
        self.observe(
            self.driver.create_raster_pipeline(packet),
            "GlDevice::create_raster_pipeline",
        )
        .map(|name| {
            Box::new(GlRasterPipeline {
                driver: Arc::clone(&self.driver),
                name,
            }) as _
        })
    }
    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        self.active("GlDevice::submit")?;
        self.observe(
            self.driver.submit(Self::submission_plan(request)),
            "GlDevice::submit",
        )
    }
    fn completion(&self, serial: u64) -> CompletionState {
        match self.loss_info() {
            Some(info) => CompletionState::DeviceLost(info),
            None => {
                let state = self.driver.completion(serial);
                if let CompletionState::DeviceLost(info) = &state {
                    self.mark_lost(info.message().to_owned());
                    return CompletionState::DeviceLost(
                        self.loss_info().expect("loss authority stored info"),
                    );
                }
                state
            }
        }
    }
    fn completion_or_register_waker(&self, serial: u64, waker: &Waker) -> CompletionState {
        if let Some(info) = self.loss_info() {
            return CompletionState::DeviceLost(info);
        }
        let state = self.driver.completion_or_register_waker(serial, waker);
        if let CompletionState::DeviceLost(info) = &state {
            self.mark_lost(info.message().to_owned());
            return CompletionState::DeviceLost(
                self.loss_info().expect("loss authority stored info"),
            );
        }
        if matches!(state, CompletionState::Pending) {
            self.liveness
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .waiters
                .push(waker.clone());
        }
        state
    }
}
