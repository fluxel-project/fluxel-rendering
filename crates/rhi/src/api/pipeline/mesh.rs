//! Mesh/task pipeline descriptors and fail-closed creation façade.

use super::interface::validate_pipeline_interface_descriptor;
use super::resources::{merge_shader_resources, validate_shader_resource_requirements};
use crate::api::binding::{BindingLimitClass, BindingSupportQuery};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, TextureSupportQuery, logical_bytes_per_block};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::pipeline::backend::MeshPipelineBackend;
use crate::api::pipeline::{
    ColorTargetFacts, ColorTargetState, DepthStencilState, MultisampleState, PipelineCache,
    PipelineDeviceFacts, PipelineInterface, PrimitiveState, PrimitiveTopology,
    RenderTargetSignature,
};
use crate::api::platform::Device;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::texture::{TextureDimension, TextureUsage};
use crate::api::shader::{
    ArtifactAcceptance, InterpolationSampling, ShaderArtifact, ShaderLocation, ShaderModule,
    ShaderStage,
};
use core::fmt;
use std::sync::Arc;

/// Mesh/task graphics pipeline descriptor.
#[derive(Clone)]
pub struct MeshPipelineDescriptor {
    /// Diagnostic label.
    pub label: Label,
    /// Optional task/amplification entry point.
    pub task: Option<ShaderModule>,
    /// Required mesh entry point.
    pub mesh: ShaderModule,
    /// Optional fragment entry point.
    pub fragment: Option<ShaderModule>,
    /// Logical binding and immediate-data interface.
    pub interface: PipelineInterface,
    /// Optional native cache from the same device.
    pub cache: Option<PipelineCache>,
    /// Primitive assembly and rasterization state. Mesh output replaces vertex
    /// input only; it does not remove the rest of graphics fixed-function state.
    pub primitive: PrimitiveState,
    /// Optional depth/stencil state.
    pub depth_stencil: Option<DepthStencilState>,
    /// Multisample state shared by every active attachment.
    pub multisample: MultisampleState,
    /// Optional non-zero multiview mask. A contiguous low mask only requires
    /// `Multiview`; a mask with holes additionally requires
    /// `SelectiveMultiview`.
    pub multiview_mask: Option<u32>,
    /// Vector index = fragment output location; holes are preserved.
    pub color_targets: Vec<Option<ColorTargetState>>,
}
impl MeshPipelineDescriptor {
    /// Creates a mesh-only pipeline descriptor.
    pub fn new(
        mesh: ShaderModule,
        interface: PipelineInterface,
        target_signature: RenderTargetSignature,
    ) -> Self {
        Self {
            label: Label::default(),
            task: None,
            mesh,
            fragment: None,
            interface,
            cache: None,
            primitive: PrimitiveState::new(PrimitiveTopology::TriangleList),
            depth_stencil: target_signature
                .depth_stencil_format
                .map(DepthStencilState::new),
            multisample: MultisampleState::new(target_signature.sample_count),
            color_targets: target_signature
                .color_formats
                .into_iter()
                .map(|format| format.map(ColorTargetState::new))
                .collect(),
            multiview_mask: None,
        }
    }
    /// Selects a native pipeline cache owned by the same device.
    pub fn with_cache(mut self, cache: PipelineCache) -> Self {
        self.cache = Some(cache);
        self
    }
    /// Adds a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
    /// Adds an optional task stage.
    pub fn with_task(mut self, task: ShaderModule) -> Self {
        self.task = Some(task);
        self
    }
    /// Adds a fragment stage.
    pub fn with_fragment(mut self, fragment: ShaderModule) -> Self {
        self.fragment = Some(fragment);
        self
    }
    /// Replaces primitive/rasterization state.
    pub fn with_primitive(mut self, state: PrimitiveState) -> Self {
        self.primitive = state;
        self
    }
    /// Adds or replaces depth/stencil state.
    pub fn with_depth_stencil(mut self, state: DepthStencilState) -> Self {
        self.depth_stencil = Some(state);
        self
    }
    /// Replaces multisample state.
    pub fn with_multisample(mut self, state: MultisampleState) -> Self {
        self.multisample = state;
        self
    }
    /// Enables selected multiview layers. Zero is rejected at creation; a
    /// sparse selection additionally requires `SelectiveMultiview`.
    pub fn with_multiview_mask(mut self, mask: u32) -> Self {
        self.multiview_mask = Some(mask);
        self
    }
    /// Sets a color target at its fragment-output location, preserving holes.
    pub fn with_color_target(
        mut self,
        location: crate::api::shader::ShaderLocation,
        target: ColorTargetState,
    ) -> Self {
        let index = location.get() as usize;
        if self.color_targets.len() <= index {
            self.color_targets.resize(index + 1, None);
        }
        self.color_targets[index] = Some(target);
        self
    }
    /// Canonical attachment compatibility signature derived from fixed state.
    pub fn target_signature(&self) -> RenderTargetSignature {
        let mut color_formats: Vec<_> = self
            .color_targets
            .iter()
            .map(|target| target.as_ref().map(|target| target.format))
            .collect();
        while matches!(color_formats.last(), Some(None)) {
            color_formats.pop();
        }
        RenderTargetSignature {
            color_formats,
            depth_stencil_format: self.depth_stencil.as_ref().map(|state| state.format),
            sample_count: self.multisample.count,
        }
    }
}

/// Checks mesh-specific graphics fixed state independently of native lowering.
/// Mesh shaders replace vertex input only; all attachment and multiview state
/// remains a normal portable graphics contract.
pub(crate) fn validate_mesh_fixed_state(
    desc: &MeshPipelineDescriptor,
    feature_supported: impl Fn(OptionalFeature) -> bool,
    limit: impl Fn(crate::api::platform::requirements::LimitKey) -> Option<u64>,
) -> RhiResult<()> {
    if desc.multisample.count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a mesh pipeline multisample count must be non-zero",
        ));
    }
    if matches!(desc.primitive.topology, PrimitiveTopology::PointList)
        && !feature_supported(OptionalFeature::MeshShaderPoints)
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "this device does not enable point primitive output from mesh shaders",
        ));
    }
    for (requested, feature, name) in [
        (
            matches!(
                desc.primitive.polygon_mode,
                crate::api::pipeline::PolygonMode::Line
            ),
            OptionalFeature::PolygonModeLine,
            "line polygon mode",
        ),
        (
            matches!(
                desc.primitive.polygon_mode,
                crate::api::pipeline::PolygonMode::Point
            ),
            OptionalFeature::PolygonModePoint,
            "point polygon mode",
        ),
        (
            desc.primitive.unclipped_depth,
            OptionalFeature::DepthClipControl,
            "unclipped depth",
        ),
        (
            desc.primitive.conservative,
            OptionalFeature::ConservativeRasterization,
            "conservative rasterization",
        ),
    ] {
        if requested && !feature_supported(feature) {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!("{name} is not enabled on this device"),
            ));
        }
    }
    if desc
        .primitive
        .depth_bias
        .is_some_and(|bias| !bias.slope_scale.is_finite() || !bias.clamp.is_finite())
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "depth-bias slope and clamp must be finite",
        ));
    }
    if desc
        .primitive
        .depth_bias
        .is_some_and(|bias| bias.clamp != 0.0)
        && !feature_supported(OptionalFeature::DepthBiasClamp)
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "depth-bias clamp is not enabled on this device",
        ));
    }
    if desc.multisample.mask != u32::MAX && !feature_supported(OptionalFeature::MultisampleMask) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "a non-default multisample mask is not enabled on this device",
        ));
    }
    if let Some(mask) = desc.multiview_mask {
        if mask == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a mesh-pipeline multiview mask must select at least one view",
            ));
        }
        if !feature_supported(OptionalFeature::Multiview) {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not enable multiview rasterization",
            ));
        }
        if !super::is_contiguous_low_multiview_mask(mask)
            && !feature_supported(OptionalFeature::SelectiveMultiview)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "selective multiview rasterization is not enabled on this device",
            ));
        }
        if !feature_supported(OptionalFeature::MeshShaderMultiview) {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not enable mesh-shader multiview",
            ));
        }
        if let Some(max) =
            limit(crate::api::platform::requirements::LimitKey::MaxMultiviewViewCount)
            && u64::from(32 - mask.leading_zeros()) > max
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the mesh-pipeline multiview mask exceeds MaxMultiviewViewCount",
            ));
        }
    }
    Ok(())
}

/// Complete portable validation for mesh graphics pipelines.  This intentionally
/// shares the same resource, attachment and fragment-linkage contract as raster
/// pipelines; only vertex-input and vertex-stage rules are absent.
pub(crate) fn validate_mesh_pipeline_descriptor(
    desc: &MeshPipelineDescriptor,
    facts: PipelineDeviceFacts<'_>,
) -> RhiResult<()> {
    if !(facts.feature_supported)(OptionalFeature::MeshShader) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "mesh shaders are not enabled on this device",
        ));
    }
    let device = desc.interface.device_identity();
    for module in [Some(&desc.mesh), desc.task.as_ref(), desc.fragment.as_ref()]
        .into_iter()
        .flatten()
    {
        if module.device_identity() != device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "a mesh-pipeline shader belongs to a different device than its interface",
            )
            .with_object(module.id()));
        }
    }
    if desc.mesh.stage() != ShaderStage::Mesh
        || desc
            .task
            .as_ref()
            .is_some_and(|module| module.stage() != ShaderStage::Task)
        || desc
            .fragment
            .as_ref()
            .is_some_and(|module| module.stage() != ShaderStage::Fragment)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "mesh pipelines require Mesh, optional Task, and optional Fragment stages",
        ));
    }
    validate_mesh_fixed_state(desc, facts.feature_supported, facts.limit)?;
    validate_pipeline_interface_descriptor(
        desc.interface.descriptor(),
        facts.limit,
        facts.binding_limit,
    )?;

    let mesh_interface = &desc.mesh.artifact().interface;
    let mut stages = vec![(ShaderStage::Mesh, mesh_interface)];
    if let Some(task) = &desc.task {
        stages.push((ShaderStage::Task, &task.artifact().interface));
    }
    if let Some(fragment) = &desc.fragment {
        stages.push((ShaderStage::Fragment, &fragment.artifact().interface));
    }
    let merged = merge_shader_resources(stages)?;
    validate_shader_resource_requirements(&merged, &desc.interface, facts.binding_support)?;
    for module in [Some(&desc.mesh), desc.task.as_ref(), desc.fragment.as_ref()]
        .into_iter()
        .flatten()
    {
        let acceptance = (facts.shader_acceptance)(module.artifact());
        if acceptance != ArtifactAcceptance::Accepted {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device does not accept a mesh-pipeline shader artifact: {acceptance:?}"
                ),
            ));
        }
    }
    // Task-to-mesh payload ABI is intentionally not represented by generic
    // `ShaderLocation` values: native task payloads are not raster varyings.
    // Their code-form-specific compatibility remains part of shader acceptance;
    // only the portable mesh-to-fragment location interface is checked here.
    let fragment_interface = desc
        .fragment
        .as_ref()
        .map(|module| &module.artifact().interface);
    if let Some(fragment) = fragment_interface {
        if fragment.inputs().iter().any(|input| {
            input.interpolation.is_some_and(|interpolation| {
                interpolation.sampling == InterpolationSampling::Sample
            })
        }) && !(facts.feature_supported)(OptionalFeature::MultisampledShading)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "sample-frequency fragment interpolation is not enabled on this device",
            ));
        }
        validate_stage_linkage(mesh_interface, fragment, "mesh", "fragment")?;
    } else if let Some(index) = desc.color_targets.iter().position(Option::is_some) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("a mesh pipeline without a fragment stage cannot have color target {index}"),
        ));
    }
    validate_mesh_targets(desc, fragment_interface, facts)?;
    Ok(())
}

fn validate_stage_linkage(
    producer: &crate::api::shader::ShaderInterface,
    consumer: &crate::api::shader::ShaderInterface,
    producer_name: &str,
    consumer_name: &str,
) -> RhiResult<()> {
    for input in consumer.inputs() {
        let Some(output) = producer
            .outputs()
            .iter()
            .find(|output| output.location == input.location)
        else {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "the {consumer_name} stage reads location {}, which the {producer_name} stage does not write",
                    input.location.get()
                ),
            ));
        };
        if output.numeric_type != input.numeric_type
            || output.components != input.components
            || output.interpolation != input.interpolation
        {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "location {} has incompatible {producer_name}-to-{consumer_name} interface types",
                    input.location.get()
                ),
            ));
        }
    }
    Ok(())
}

fn validate_mesh_targets(
    desc: &MeshPipelineDescriptor,
    fragment: Option<&crate::api::shader::ShaderInterface>,
    facts: PipelineDeviceFacts<'_>,
) -> RhiResult<()> {
    if let Some(state) = &desc.depth_stencil {
        let aspects = crate::api::format::format_aspects(state.format);
        if state.depth.is_some()
            && !aspects.contains(crate::api::resource::subresource::TextureAspects::DEPTH)
            || state.stencil.is_some()
                && !aspects.contains(crate::api::resource::subresource::TextureAspects::STENCIL)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "depth/stencil state selects an aspect absent from its format",
            ));
        }
        let query = TextureSupportQuery::new(
            TextureDimension::D2,
            state.format,
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
            desc.multisample.count,
        );
        if !(facts.texture_support)(&query).is_supported() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "depth-stencil attachment format/sample-count is unsupported",
            ));
        }
    }
    if fragment.is_some_and(|interface| interface.writes_frag_depth())
        && !desc.depth_stencil.as_ref().is_some_and(|state| {
            crate::api::format::format_aspects(state.format)
                .contains(crate::api::resource::subresource::TextureAspects::DEPTH)
        })
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "fragment depth output requires a depth attachment",
        ));
    }
    let mut active = 0u64;
    let mut bytes = 0u64;
    for (index, target) in desc.color_targets.iter().enumerate() {
        let Some(target) = target else { continue };
        active += 1;
        let Some(target_facts) = (facts.color_target_facts)(target.format) else {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "color target format is unsupported",
            ));
        };
        if !target_facts.color_attachment || target.blend.is_some() && !target_facts.blendable {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "color target attachment or blend mode is unsupported",
            ));
        }
        let query = TextureSupportQuery::new(
            TextureDimension::D2,
            target.format,
            TextureUsage::COLOR_ATTACHMENT,
            desc.multisample.count,
        );
        if !(facts.texture_support)(&query).is_supported() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "color attachment format/sample-count is unsupported",
            ));
        }
        if let Some(fragment) = fragment {
            let output = fragment
                .outputs()
                .iter()
                .find(|output| output.location == ShaderLocation::new(index as u32));
            if let Some(output) = output {
                if target_facts.color_output_type != Some(output.numeric_type) {
                    return Err(RhiError::new(
                        RhiErrorKind::Unsupported,
                        "fragment output type is incompatible with its color target",
                    ));
                }
            } else if target.write_mask != crate::api::pipeline::ColorWriteMask::NONE {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a color target without a fragment output must have an empty write mask",
                ));
            }
        }
        bytes = bytes.saturating_add(logical_bytes_per_block(target.format).unwrap_or(0) as u64);
    }
    if let Some(max) = (facts.limit)(LimitKey::MaxColorAttachments)
        && active > max
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "mesh pipeline exceeds MaxColorAttachments",
        ));
    }
    if let Some(max) = (facts.limit)(LimitKey::MaxColorAttachmentBytesPerSample)
        && bytes > max
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "mesh pipeline exceeds MaxColorAttachmentBytesPerSample",
        ));
    }
    if let Some(max) = (facts.limit)(LimitKey::MaxInterStageShaderVariables)
        && fragment.map_or(0, |interface| interface.inputs().len()) as u64 > max
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "mesh pipeline exceeds MaxInterStageShaderVariables",
        ));
    }
    if !desc.primitive.topology.is_strip() && desc.primitive.strip_index_format.is_some() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a non-strip mesh topology cannot specify strip index format",
        ));
    }
    if desc.primitive.depth_bias.is_some()
        && !matches!(
            desc.primitive.topology,
            PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip
        )
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "depth bias requires triangle topology",
        ));
    }
    if desc.multisample.alpha_to_coverage_enabled && desc.multisample.count <= 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "alpha-to-coverage requires multisampling",
        ));
    }
    Ok(())
}

/// Opaque mesh/task pipeline.
#[derive(Clone)]
pub struct MeshPipeline {
    inner: Arc<MeshPipelineInner>,
}
struct MeshPipelineInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: MeshPipelineDescriptor,
    native: Box<dyn MeshPipelineBackend>,
}
impl MeshPipeline {
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: MeshPipelineDescriptor,
        native: Box<dyn MeshPipelineBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(MeshPipelineInner {
                id,
                device,
                descriptor,
                native,
            }),
        }
    }
    /// Process-local identity.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }
    /// Owning device identity.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }
    /// Immutable descriptor.
    pub fn descriptor(&self) -> &MeshPipelineDescriptor {
        &self.inner.descriptor
    }
    pub(crate) fn native(&self) -> &dyn MeshPipelineBackend {
        self.inner.native.as_ref()
    }
}
impl fmt::Debug for MeshPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeshPipeline")
            .field("id", &self.id())
            .field("device", &self.device_identity())
            .finish_non_exhaustive()
    }
}
impl Device {
    /// Creates a mesh/task pipeline. Unsupported adapters reject before native pipeline creation.
    pub async fn create_mesh_pipeline(
        &self,
        desc: &MeshPipelineDescriptor,
    ) -> RhiResult<MeshPipeline> {
        for module in [Some(&desc.mesh), desc.task.as_ref(), desc.fragment.as_ref()]
            .into_iter()
            .flatten()
        {
            if module.device_identity() != self.identity() {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "mesh-pipeline shader belongs to a different device",
                )
                .with_object(module.id())
                .at("Device::create_mesh_pipeline"));
            }
        }
        if desc.interface.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "mesh-pipeline interface belongs to a different device",
            )
            .with_object(desc.interface.id())
            .at("Device::create_mesh_pipeline"));
        }
        if let Some(cache) = &desc.cache {
            if cache.device_identity() != self.identity() {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "mesh-pipeline cache belongs to a different device",
                )
                .with_object(cache.id())
                .at("Device::create_mesh_pipeline"));
            }
        }
        self.require_active()
            .map_err(|e| e.at("Device::create_mesh_pipeline"))?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::MeshShader)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not enable mesh shaders",
            )
            .at("Device::create_mesh_pipeline"));
        }
        if desc.mesh.artifact().stage != ShaderStage::Mesh
            || desc
                .task
                .as_ref()
                .is_some_and(|m| m.artifact().stage != ShaderStage::Task)
            || desc
                .fragment
                .as_ref()
                .is_some_and(|m| m.artifact().stage != ShaderStage::Fragment)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "mesh pipeline modules must have Mesh, optional Task, and optional Fragment stages",
            )
            .at("Device::create_mesh_pipeline"));
        }
        let capabilities = self.capabilities();
        let limit = |key: LimitKey| capabilities.limit(key);
        let binding_support = |query: &BindingSupportQuery| capabilities.binding_support(query);
        let binding_limit =
            |stage: ShaderStage, class: BindingLimitClass| capabilities.binding_limit(stage, class);
        let feature_supported = |feature: OptionalFeature| capabilities.supports_feature(feature);
        let shader_acceptance =
            |artifact: &ShaderArtifact| capabilities.shader_acceptance(artifact);
        let color_target_facts = |format: TextureFormat| {
            capabilities.format(format).map(|facts| {
                ColorTargetFacts::new(
                    facts.color_attachment(),
                    facts.blendable(),
                    facts.has_alpha_channel(),
                    facts.color_output_type(),
                )
            })
        };
        let texture_support = |query: &TextureSupportQuery| capabilities.texture_support(query);
        validate_mesh_pipeline_descriptor(
            desc,
            PipelineDeviceFacts {
                limit: &limit,
                binding_support: &binding_support,
                binding_limit: &binding_limit,
                feature_supported: &feature_supported,
                shader_acceptance: &shader_acceptance,
                color_target_facts: &color_target_facts,
                texture_support: &texture_support,
            },
        )
        .map_err(|error| error.at("Device::create_mesh_pipeline"))?;
        let native = self.native().create_mesh_pipeline(desc)?;
        Ok(MeshPipeline::new(
            ObjectId::next(),
            self.identity(),
            desc.clone(),
            native,
        ))
    }
}
