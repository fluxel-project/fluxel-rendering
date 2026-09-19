//! The mock device, its adapter snapshot, and its capability database.
//!
//! Two things live here. The capability database is built through the same
//! [`CapabilityDataBuilder`] a hardware backend uses, so every fact the mock
//! answers is a declared fact and every undeclared question falls back to the
//! fail-closed default: an undeclared route is `Unsupported`, an undeclared
//! format is absent, and an undeclared binding shape is `Unsupported`.
//!
//! The `create_*` methods lower what they are given. Descriptor legality,
//! capability checks, and cross-device checks already happened in the `Device`
//! façade, so the mock does not repeat them — a backend that re-decided
//! portable legality would make two backends able to disagree about one
//! contract. The three exceptions are raster, compute, and pipeline-interface
//! creation, which the façade delegates unvalidated; the mock runs the real
//! portable validators there, which is exactly what a hardware backend does.

use std::sync::Arc;

use crate::rhi::binding::{
    BindGroup, BindGroupBackend, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutBackend,
    BindGroupLayoutCompatibilityId, BindGroupLayoutDescriptor, BindingKind, BindingSupportQuery,
    BufferBindingAccess, LayoutFingerprint, PipelineInterfaceCompatibilityId, SamplerKind,
    TextureSampleType,
};
use crate::rhi::capability::{
    AvailableCapabilities, CapabilityData, CapabilityDataBuilder, EnabledCapabilities,
};
use crate::rhi::command::{CommandRecorder, RecorderDescriptor, RecorderLimits};
use crate::rhi::diagnostics::DiagnosticLog;
use crate::rhi::format::{
    BufferCopyLayoutLimits, BufferSupportLimits, LaneDependencyRoute, LaneWorkDomains,
    RouteCapabilities, RouteQuery, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo, TexelCopyLayoutLimits, TextureFormat, TextureSupportLimits,
    TextureSupportQuery,
};
use crate::rhi::pipeline::{
    ComputeLimits, ComputePipeline, ComputePipelineBackend, ComputePipelineDescriptor,
    InterfaceLimits, PipelineInterface, PipelineInterfaceBackend, PipelineInterfaceDescriptor,
    RasterLimits, RasterPipeline, RasterPipelineBackend, RasterPipelineDescriptor,
    ResourceMergeOutcome, validate_compute_descriptor,
    validate_pipeline_interface, validate_raster_descriptor,
};
use crate::rhi::platform::{
    AdapterId, AdapterInfo, BackendKind, DeviceBackend, DeviceIdentity, DeviceLossInfo,
    DeviceStatus, LimitKey, OptionalFeature, RhiError, RhiErrorKind, RhiResult, next_identity,
    next_object_id,
};
use crate::rhi::presentation::{
    ConfiguredPresentation, PresentPlanId, PresentReceipt, PresentReceiptId, PresentState,
    PresentationConfiguration, PresentationTarget, PresentationTargetCapabilities,
};
use crate::rhi::resource::{
    Buffer, BufferBackend, BufferDescriptor, BufferUsage, Extent3d, Sampler, SamplerBackend,
    SamplerDescriptor, Texture, TextureAspect, TextureBackend, TextureDescriptor,
    TextureDimension, TextureUsage, TextureView, TextureViewBackend, TextureViewDescriptor,
    TextureViewDimension,
    UploadDescriptor, UploadJob, UploadJobBackend,
};
use crate::rhi::statistics::DeviceStatistics;
use crate::rhi::shader::{
    ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderModule, ShaderModuleBackend, ShaderStage,
    ShaderStages,
};
use crate::rhi::submission::{
    CompletionPoint, CompletionState, PlanPoint, SubmissionPlan, SubmissionPoint,
    SubmissionReceipt, SubmissionReceiptBackend,
};

use super::frames::{MockConfiguredPresentationBackend, MockTargetPayload, target_capabilities};
use super::pipelines::{MockBindGroup, MockBindGroupLayout, MockComputePipeline, MockPipelineInterface, MockRasterPipeline, MockShaderModule};
use super::recording::MockRecorderBackend;
use super::resources::{
    MockBuffer, MockSampler, MockTexture, MockTextureView, MockUploadJob,
};
use super::MockState;

/// The largest resource the mock declares it can create.
pub(super) const MOCK_MAX_RESOURCE_BYTES: u64 = 1 << 28;

/// The required multiple of a texel copy's buffer offset.
///
/// This is a device fact rather than a portable constant: it is declared here
/// through the route query exactly as a hardware backend declares it, so a
/// caller that has a backend with different limits is not asked to satisfy this
/// backend's.
pub(super) const MOCK_TEXEL_COPY_OFFSET_ALIGNMENT: u64 = 256;

/// The required multiple of a texel copy's `bytes_per_row`.
///
/// The value every target backend imposes on a command-buffer copy, and the one
/// the design names for WebGPU and the D3D12 copy footprint. It is not imposed
/// on an `UploadJob` host layout.
pub(super) const MOCK_TEXEL_COPY_ROW_ALIGNMENT: u32 = 256;

/// The backend half of the mock's device.
pub(super) struct MockDeviceBackend {
    identity: DeviceIdentity,
    adapter: AdapterInfo,
    capabilities: EnabledCapabilities,
    diagnostics: Arc<DiagnosticLog>,
    statistics: DeviceStatistics,
    state: Arc<MockState>,
}

impl MockDeviceBackend {
    /// A device over `capabilities`, with a fresh identity domain.
    pub(super) fn new(capabilities: Arc<CapabilityData>, state: Arc<MockState>) -> Self {
        let identity = next_identity();
        let adapter = AdapterInfo::new(
            AdapterId::new(identity.instance().as_u64(), 0),
            "fluxel-mock",
            BackendKind::Vulkan,
            None,
            None,
            AvailableCapabilities::new(Arc::clone(&capabilities)),
        );
        Self {
            identity,
            adapter,
            capabilities: EnabledCapabilities::new(capabilities),
            // The log belongs to the device, and the device shares it with
            // every backend half that reports into it; see `MockState`.
            diagnostics: Arc::clone(state.diagnostic_log()),
            // The domain belongs to this identity, exactly as a real backend's
            // does: a mock that shared one domain across identities would hide
            // the very scoping rule §47.2 states.
            statistics: DeviceStatistics::new(identity),
            state,
        }
    }

    /// A lease over a new target, configured as requested.
    pub(super) fn lease(
        &self,
        configuration: PresentationConfiguration,
    ) -> ConfiguredPresentation {
        ConfiguredPresentation::new(Arc::new(MockConfiguredPresentationBackend::new(
            self.identity,
            Arc::clone(&self.state),
            configuration,
        )))
    }

    /// The limits a recorder validates against.
    fn recorder_limits(&self) -> RecorderLimits {
        RecorderLimits::from_capabilities(
            &self.capabilities,
            self.capabilities.supports_feature(OptionalFeature::Compute),
        )
    }
}

impl DeviceBackend for MockDeviceBackend {
    fn identity(&self) -> DeviceIdentity {
        self.identity
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }

    fn capabilities(&self) -> &EnabledCapabilities {
        &self.capabilities
    }

    fn status(&self) -> DeviceStatus {
        DeviceStatus::Active
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        None
    }

    fn poll(&self) -> RhiResult<()> {
        self.state.note("device:poll");
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        self.state.note("device:wait_idle");
        Ok(())
    }

    fn presentation_capabilities(
        &self,
        _target: &PresentationTarget,
    ) -> RhiResult<PresentationTargetCapabilities> {
        self.state.note("device:presentation_capabilities");
        Ok(target_capabilities())
    }

    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Buffer> {
        self.state.note("device:create_buffer");
        Ok(Buffer::new(Arc::new(MockBuffer {
            id: next_object_id(),
            device: self.identity,
            descriptor: descriptor.clone(),
        }) as Arc<dyn BufferBackend>))
    }

    fn create_texture(&self, descriptor: &TextureDescriptor) -> RhiResult<Texture> {
        self.state.note("device:create_texture");
        Ok(Texture::new(Arc::new(MockTexture {
            id: next_object_id(),
            device: self.identity,
            descriptor: descriptor.clone(),
        }) as Arc<dyn TextureBackend>))
    }

    fn create_texture_view(
        &self,
        texture: &Texture,
        descriptor: &TextureViewDescriptor,
    ) -> RhiResult<TextureView> {
        self.state.note("device:create_texture_view");
        Ok(TextureView::new(Arc::new(MockTextureView {
            id: next_object_id(),
            device: self.identity,
            texture: texture.clone(),
            descriptor: descriptor.clone(),
        }) as Arc<dyn TextureViewBackend>))
    }

    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> RhiResult<Sampler> {
        self.state.note("device:create_sampler");
        Ok(Sampler::new(Arc::new(MockSampler {
            id: next_object_id(),
            device: self.identity,
            descriptor: descriptor.clone(),
        }) as Arc<dyn SamplerBackend>))
    }

    fn create_upload(&self, descriptor: UploadDescriptor) -> RhiResult<UploadJob> {
        self.state.note("device:create_upload");
        Ok(UploadJob::new(Arc::new(MockUploadJob {
            id: next_object_id(),
            device: self.identity,
            descriptor,
        }) as Arc<dyn UploadJobBackend>))
    }

    fn create_shader(&self, artifact: &ShaderArtifact) -> RhiResult<ShaderModule> {
        self.state.note("device:create_shader");
        Ok(ShaderModule::new(Arc::new(MockShaderModule {
            id: next_object_id(),
            device: self.identity,
            artifact: artifact.clone(),
        }) as Arc<dyn ShaderModuleBackend>))
    }

    fn create_bind_group_layout(
        &self,
        descriptor: &BindGroupLayoutDescriptor,
    ) -> RhiResult<BindGroupLayout> {
        self.state.note("device:create_bind_group_layout");
        let compatibility = BindGroupLayoutCompatibilityId(self.state.intern_layout(descriptor));
        Ok(BindGroupLayout::new(Arc::new(MockBindGroupLayout {
            id: next_object_id(),
            device: self.identity,
            descriptor: descriptor.clone(),
            compatibility,
        }) as Arc<dyn BindGroupLayoutBackend>))
    }

    fn create_bind_group(&self, descriptor: &BindGroupDescriptor) -> RhiResult<BindGroup> {
        self.state.note("device:create_bind_group");
        Ok(BindGroup::new(Arc::new(MockBindGroup {
            id: next_object_id(),
            device: self.identity,
            layout: descriptor.layout.clone(),
            descriptor: descriptor.clone(),
        }) as Arc<dyn BindGroupBackend>))
    }

    fn create_pipeline_interface(
        &self,
        descriptor: &PipelineInterfaceDescriptor,
    ) -> RhiResult<PipelineInterface> {
        self.state.note("device:create_pipeline_interface");
        let limits = InterfaceLimits::from_capabilities(&self.capabilities);
        let binding_limit = |stage, class| self.capabilities.binding_limit(stage, class);
        validate_pipeline_interface(self.identity, descriptor, limits, &binding_limit)?;

        let groups: Vec<u64> = descriptor
            .groups
            .iter()
            .map(|group| group.compatibility_id().as_u64())
            .collect();
        let compatibility =
            PipelineInterfaceCompatibilityId(self.state.intern_interface(&groups));
        Ok(PipelineInterface::new(Arc::new(MockPipelineInterface {
            id: next_object_id(),
            device: self.identity,
            descriptor: descriptor.clone(),
            compatibility,
        }) as Arc<dyn PipelineInterfaceBackend>))
    }

    fn create_raster_pipeline(
        &self,
        descriptor: &RasterPipelineDescriptor,
    ) -> RhiResult<RasterPipeline> {
        self.state.note("device:create_raster_pipeline");
        let limits = RasterLimits::from_capabilities(&self.capabilities);
        let binding_limit = |stage, class| self.capabilities.binding_limit(stage, class);
        let limit = |key| self.capabilities.limit(key);
        let used_bindings: Vec<ResourceMergeOutcome> = validate_raster_descriptor(
            self.identity,
            descriptor,
            limits,
            &binding_limit,
            &limit,
        )?;
        Ok(RasterPipeline::new(Arc::new(MockRasterPipeline {
            id: next_object_id(),
            device: self.identity,
            target_signature: descriptor.target_signature(),
            descriptor: descriptor.clone(),
            used_bindings,
        }) as Arc<dyn RasterPipelineBackend>))
    }

    fn create_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> RhiResult<ComputePipeline> {
        self.state.note("device:create_compute_pipeline");
        let limits = ComputeLimits::from_capabilities(&self.capabilities);
        let used_bindings = validate_compute_descriptor(
            self.identity,
            descriptor,
            limits,
            self.capabilities.supports_feature(OptionalFeature::Compute),
        )?;
        Ok(ComputePipeline::new(Arc::new(MockComputePipeline {
            id: next_object_id(),
            device: self.identity,
            descriptor: descriptor.clone(),
            used_bindings,
        }) as Arc<dyn ComputePipelineBackend>))
    }

    fn create_recorder(&self, descriptor: &RecorderDescriptor) -> RhiResult<CommandRecorder> {
        self.state.note("device:create_recorder");
        let limits = self.recorder_limits();
        let backend = MockRecorderBackend::new(
            self.identity,
            Arc::clone(&self.state),
            descriptor.label.clone(),
        );
        Ok(CommandRecorder::new(Box::new(backend), limits))
    }

    fn submit(&self, plan: SubmissionPlan) -> RhiResult<SubmissionReceipt> {
        self.state.note("device:submit");
        Ok(SubmissionReceipt::new(Arc::new(MockSubmissionReceipt::new(
            self.identity,
            plan,
        )) as Arc<dyn SubmissionReceiptBackend>))
    }

    fn completion_state(&self, point: CompletionPoint) -> RhiResult<CompletionState> {
        self.state.note("device:completion_state");
        debug_assert_eq!(point.device_identity(), self.identity);
        // The mock executes nothing, so it never reports completion. Answering
        // `Complete` would let a test read a terminal state the mock never
        // produced, which is exactly the evidence this backend cannot give.
        Ok(CompletionState::Pending)
    }

    fn configure_presentation(
        &self,
        target: &PresentationTarget,
        configuration: &PresentationConfiguration,
    ) -> RhiResult<ConfiguredPresentation> {
        // A presentation target is a host-owned payload. Only the backend that
        // produced it can recover it, so a target from any other provider is
        // refused here rather than silently accepted and configured with a
        // drawable this backend does not own.
        let owned = target
            .payload()
            .downcast_ref::<MockTargetPayload>()
            .is_some_and(|payload| payload.producer == self.identity);
        if !owned {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the presentation target was not produced by this backend",
            )
            .at("configure_presentation"));
        }
        self.state.note(format!("device:configure_presentation:{}", target.id().as_u64()));
        Ok(self.lease(configuration.clone()))
    }

    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        self.state.note("device:present_state");
        debug_assert_eq!(receipt.device_identity(), self.identity);
        // A present is accepted by a presentation system the mock does not
        // have, so it stays pending forever rather than becoming accepted.
        Ok(PresentState::Pending)
    }

    fn diagnostics(&self) -> &DiagnosticLog {
        &self.diagnostics
    }

    fn statistics(&self) -> &DeviceStatistics {
        &self.statistics
    }
}

/// The receipt of one accepted plan.
///
/// It owns the plan, which is what keeps the plan's frames alive until the
/// receipt is dropped: a submission that accepted a present must not abandon
/// its frame before the presentation system answers.
struct MockSubmissionReceipt {
    device: DeviceIdentity,
    submitted: SubmissionPoint,
    completion: CompletionPoint,
    presents: Vec<PresentReceipt>,
    plan: SubmissionPlan,
}

impl MockSubmissionReceipt {
    fn new(device: DeviceIdentity, plan: SubmissionPlan) -> Self {
        let serial = next_object_id().as_u64();
        let presents = plan
            .presents()
            .iter()
            .enumerate()
            .map(|(index, present)| {
                PresentReceipt::new(
                    PresentReceiptId::new(device, serial.wrapping_add(index as u64)),
                    PresentPlanId {
                        plan: plan.id(),
                        local: present.id().local,
                    },
                )
            })
            .collect();
        Self {
            device,
            submitted: SubmissionPoint::new(device, serial),
            completion: CompletionPoint::new(device, serial),
            presents,
            plan,
        }
    }
}

impl SubmissionReceiptBackend for MockSubmissionReceipt {
    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn submitted(&self) -> SubmissionPoint {
        self.submitted
    }

    fn completion(&self) -> CompletionPoint {
        self.completion
    }

    fn completion_for(&self, point: PlanPoint) -> RhiResult<CompletionPoint> {
        debug_assert_eq!(point.plan(), self.plan.id());
        Ok(self.completion)
    }

    fn presents(&self) -> &[PresentReceipt] {
        &self.presents
    }
}

/// The facts the mock declares.
///
/// The set is deliberately small and explicit: an undeclared fact is a refused
/// fact, which is what makes the mock usable for testing fail-closed behaviour.
///
/// Two lanes carry every work domain and are ordered both ways. Two lanes
/// rather than one is not decoration: a single lane makes every batch pair
/// implicitly ordered by insertion, so neither the unordered-hazard rule nor
/// "a frame use that does not happen-before its present" can be reached at all.
pub(super) fn build_capabilities() -> RhiResult<Arc<CapabilityData>> {
    let domains = LaneWorkDomains::RASTER
        .union(LaneWorkDomains::COMPUTE)
        .union(LaneWorkDomains::COPY);
    let first = SubmissionLaneId::new(0);
    let second = SubmissionLaneId::new(1);
    let submission = SubmissionCapabilities::new(
        vec![
            SubmissionLaneInfo::new(first, SubmissionLaneClass::General, domains),
            SubmissionLaneInfo::new(second, SubmissionLaneClass::General, domains),
        ],
        vec![
            ((first, second), LaneDependencyRoute::Ordered),
            ((second, first), LaneDependencyRoute::Ordered),
        ],
    );

    let mut builder =
        CapabilityDataBuilder::new(BackendKind::Vulkan, ShaderAbiVersion::CURRENT, submission);
    builder.enable_feature(OptionalFeature::Compute);
    builder.enable_feature(OptionalFeature::BindingArrays);

    for (key, value) in [
        (LimitKey::MaxBufferSize, MOCK_MAX_RESOURCE_BYTES),
        (LimitKey::MaxTexture1dDimension, 8192),
        (LimitKey::MaxTexture2dDimension, 8192),
        (LimitKey::MaxTexture3dDimension, 2048),
        (LimitKey::MaxTextureArrayLayers, 256),
        (LimitKey::MaxBindGroups, 4),
        (LimitKey::MaxBindingsPerGroup, 64),
        (LimitKey::MaxBindGroupsPlusVertexBuffers, 24),
        (LimitKey::MaxUniformBufferBindingSize, 1 << 16),
        (LimitKey::MaxStorageBufferBindingSize, 1 << 27),
        (LimitKey::MaxDynamicUniformBuffersPerPipelineLayout, 8),
        (LimitKey::MaxDynamicStorageBuffersPerPipelineLayout, 8),
        (LimitKey::MaxColorAttachments, 4),
        (LimitKey::MaxColorAttachmentBytesPerSample, 32),
        (LimitKey::MaxVertexBuffers, 8),
        (LimitKey::MaxVertexAttributes, 16),
        (LimitKey::MaxVertexBufferArrayStride, 2048),
        (LimitKey::MaxInterStageShaderVariables, 16),
        (LimitKey::MaxComputeInvocationsPerWorkgroup, 256),
        (LimitKey::MaxComputeWorkgroupSizeX, 256),
        (LimitKey::MaxComputeWorkgroupSizeY, 256),
        (LimitKey::MaxComputeWorkgroupSizeZ, 64),
        (LimitKey::MaxComputeWorkgroupsPerDimension, 65535),
        (LimitKey::MaxComputeWorkgroupStorageSize, 16384),
        (LimitKey::MinUniformBufferOffsetAlignment, 256),
        (LimitKey::MinStorageBufferOffsetAlignment, 256),
    ] {
        builder.set_limit(key, value);
    }

    for format in [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Depth32Float,
    ] {
        builder.declare_format(format);
    }
    builder.declare_view_compatibility_group(&[
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
    ]);

    for usage in [
        BufferUsage::COPY_SRC,
        BufferUsage::COPY_DST,
        BufferUsage::VERTEX,
        BufferUsage::INDEX,
        BufferUsage::UNIFORM,
        BufferUsage::STORAGE,
        BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        BufferUsage::INDEX.union(BufferUsage::COPY_DST),
    ] {
        builder.declare_buffer_support(usage, BufferSupportLimits::new(MOCK_MAX_RESOURCE_BYTES));
    }

    for dimension in [
        TextureDimension::D1,
        TextureDimension::D2,
        TextureDimension::D3,
    ] {
        for usage in [
            TextureUsage::COPY_SRC,
            TextureUsage::COPY_DST,
            TextureUsage::SAMPLED,
            TextureUsage::COLOR_ATTACHMENT,
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
            TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
            // A texture that is both a copy source and a copy destination, which
            // is what a resolve or a texture-to-texture copy needs of one
            // texture.
            TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
        ] {
            for format in [
                TextureFormat::Rgba8Unorm,
                // A second color format, so a copy that changes format is
                // expressible and can therefore be refused for the right reason.
                TextureFormat::Rgba8UnormSrgb,
                TextureFormat::Depth32Float,
            ] {
                builder.declare_texture_support(
                    TextureSupportQuery::new(dimension, format, usage, 1),
                    TextureSupportLimits::new(Extent3d::d3(8192, 8192, 2048), 13, 256),
                );
            }
        }
    }

    // The binding shapes the mock can serve. An undeclared shape is
    // `InterfaceUnsupported`, so this list is also what makes a shader that
    // declares a shape outside it a testable refusal rather than an accident.
    for stage in [ShaderStage::Vertex, ShaderStage::Fragment, ShaderStage::Compute] {
        let visibility = ShaderStages::from_stage(stage);
        for kind in [
            BindingKind::UniformBuffer { min_size: 64 },
            BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadOnly,
                min_size: 64,
            },
        ] {
            builder.declare_binding_support(BindingSupportQuery::new(visibility, kind));
        }
        for kind in [
            BindingKind::SampledTexture {
                dimension: TextureViewDimension::D2,
                sample_type: TextureSampleType::Float,
                multisampled: false,
            },
            BindingKind::Sampler {
                kind: SamplerKind::Filtering,
            },
        ] {
            builder.declare_binding_support(BindingSupportQuery::new(visibility, kind));
        }
    }

    // The copy routes the mock can serve, with the alignment facts a backend's
    // copy footprint actually has.
    //
    // Buffer-to-buffer needs the least an implementation can ask for: the
    // design states no portable offset or size alignment for it, so the mock
    // claims none beyond the four-byte word every target can move.
    builder.declare_route(
        RouteQuery::BufferToBuffer,
        RouteCapabilities::buffer_copy(BufferCopyLayoutLimits::new(4, 4)),
    );
    // A texel copy is the case the design does state a number for. The two
    // values below are the ones every target backend imposes on a
    // command-buffer copy, and are deliberately not the values an `UploadJob`
    // host layout may use: section 14 exempts a CPU source layout from these,
    // which is why they belong to the route and not to the buffer.
    // `rows_per_image_alignment` has no field here on purpose; the design
    // removed it, and a texel copy's row-count legality follows from the extent
    // and the format's block geometry.
    for dimension in [
        TextureDimension::D1,
        TextureDimension::D2,
        TextureDimension::D3,
    ] {
        for format in [
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Depth32Float,
        ] {
            for aspect in [TextureAspect::Color, TextureAspect::Depth] {
                let limits = RouteCapabilities::texel_copy(TexelCopyLayoutLimits::new(
                    MOCK_TEXEL_COPY_OFFSET_ALIGNMENT,
                    MOCK_TEXEL_COPY_ROW_ALIGNMENT,
                ));
                builder.declare_route(
                    RouteQuery::BufferToTexture {
                        dimension,
                        format,
                        aspect,
                    },
                    limits,
                );
                builder.declare_route(
                    RouteQuery::TextureToBuffer {
                        dimension,
                        format,
                        aspect,
                    },
                    limits,
                );
            }
        }
    }

    builder.declare_code_format(&ShaderCode::SpirV(Arc::from(Vec::<u32>::new())));

    builder.build()
}

/// The fingerprint the mock reports for a bind group layout.
///
/// The mock keeps the crate's own rule that a fingerprint is a cache key and
/// never a correctness decision, so it derives one canonically and never
/// compares two of them. It is only reachable through `BindGroupLayout`, and
/// the mock's decisions are made from interned canonical values instead.
pub(super) fn fingerprint_of(compatibility: u64) -> LayoutFingerprint {
    use crate::rhi::hash::CanonicalHasher;
    let mut hasher = CanonicalHasher::new();
    hasher.tag(1).u64(compatibility);
    LayoutFingerprint(hasher.finish())
}
