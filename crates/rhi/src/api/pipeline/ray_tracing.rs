//! Ray-tracing pipeline descriptors.
//!
//! Shader-table byte layout remains a backend lowering detail.  The portable
//! descriptor instead names stable groups, while dispatch supplies checked group
//! data with the device's published alignment limits.

use super::interface::validate_pipeline_interface_descriptor;
use super::resources::{
    merge_shader_resources, validate_shader_immediate_requirements,
    validate_shader_resource_requirements,
};
use crate::api::binding::{BindingLimitClass, BindingSupportQuery};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, TextureSupportQuery};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::pipeline::backend::RayTracingPipelineBackend;
use crate::api::pipeline::{
    ColorTargetFacts, PipelineCache, PipelineDeviceFacts, PipelineInterface,
};
use crate::api::platform::Device;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::shader::{ArtifactAcceptance, ShaderArtifact, ShaderModule, ShaderStage};
use core::fmt;
use std::sync::Arc;

/// A closest-hit, any-hit, and optional intersection-program collection.
#[derive(Clone)]
pub struct RayTracingHitGroup {
    /// Optional closest-hit program.
    pub closest_hit: Option<ShaderModule>,
    /// Optional any-hit program.
    pub any_hit: Option<ShaderModule>,
    /// Optional intersection program for procedural geometry.
    pub intersection: Option<ShaderModule>,
}

/// One shader-table group, preserving caller order as its portable index.
#[derive(Clone)]
pub enum RayTracingShaderGroup {
    /// Ray-generation program; exactly one descriptor member uses this form.
    RayGeneration(ShaderModule),
    /// Miss program.
    Miss(ShaderModule),
    /// Hit-program collection.
    Hit(RayTracingHitGroup),
}

/// Ray-tracing pipeline descriptor.
#[derive(Clone)]
pub struct RayTracingPipelineDescriptor {
    /// Diagnostic label.
    pub label: Label,
    /// Logical binding and immediate-data interface.
    pub interface: PipelineInterface,
    /// Optional native cache from the same device.
    pub cache: Option<PipelineCache>,
    /// Exactly one ray-generation program.
    pub ray_generation: ShaderModule,
    /// Ordered miss programs.
    pub miss: Vec<ShaderModule>,
    /// Ordered hit groups.
    pub hit_groups: Vec<RayTracingHitGroup>,
    /// Requested recursion depth, at least one.
    pub max_recursion_depth: u32,
}
impl RayTracingPipelineDescriptor {
    /// Creates a descriptor with recursion depth one.
    pub fn new(ray_generation: ShaderModule, interface: PipelineInterface) -> Self {
        Self {
            label: Label::default(),
            interface,
            cache: None,
            ray_generation,
            miss: Vec::new(),
            hit_groups: Vec::new(),
            max_recursion_depth: 1,
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
    /// Adds one miss program.
    pub fn with_miss(mut self, miss: ShaderModule) -> Self {
        self.miss.push(miss);
        self
    }
    /// Adds one hit group.
    pub fn with_hit_group(mut self, group: RayTracingHitGroup) -> Self {
        self.hit_groups.push(group);
        self
    }
    /// Selects maximum ray recursion depth.
    pub fn with_max_recursion_depth(mut self, depth: u32) -> Self {
        self.max_recursion_depth = depth;
        self
    }
}

/// Opaque ray-tracing pipeline.
#[derive(Clone)]
pub struct RayTracingPipeline {
    inner: Arc<RayTracingPipelineInner>,
}
struct RayTracingPipelineInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: RayTracingPipelineDescriptor,
    native: Box<dyn RayTracingPipelineBackend>,
}
impl RayTracingPipeline {
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: RayTracingPipelineDescriptor,
        native: Box<dyn RayTracingPipelineBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(RayTracingPipelineInner {
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
    pub fn descriptor(&self) -> &RayTracingPipelineDescriptor {
        &self.inner.descriptor
    }
    pub(crate) fn native(&self) -> &dyn RayTracingPipelineBackend {
        self.inner.native.as_ref()
    }
}
impl fmt::Debug for RayTracingPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RayTracingPipeline")
            .field("id", &self.id())
            .field("device", &self.device_identity())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Creates a ray-tracing pipeline after capability, ownership, stage, and recursion validation.
    pub async fn create_ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDescriptor,
    ) -> RhiResult<RayTracingPipeline> {
        if desc.interface.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "ray-tracing interface belongs to a different device",
            )
            .with_object(desc.interface.id())
            .at("Device::create_ray_tracing_pipeline"));
        }
        if let Some(cache) = &desc.cache {
            if cache.device_identity() != self.identity() {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "ray-tracing pipeline cache belongs to a different device",
                )
                .with_object(cache.id())
                .at("Device::create_ray_tracing_pipeline"));
            }
        }
        self.require_active()
            .map_err(|e| e.at("Device::create_ray_tracing_pipeline"))?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::RayTracingPipeline)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device does not enable ray-tracing pipelines",
            )
            .at("Device::create_ray_tracing_pipeline"));
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
        validate_ray_tracing_pipeline_descriptor(
            desc,
            self.identity(),
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
        .map_err(|error| error.at("Device::create_ray_tracing_pipeline"))?;
        let native = self.native().create_ray_tracing_pipeline(desc)?;
        Ok(RayTracingPipeline::new(
            ObjectId::next(),
            self.identity(),
            desc.clone(),
            native,
        ))
    }
}

/// Validates all ray programs as one interface-bearing pipeline before native
/// compilation. Shader-table layout is backend-private; stage/resource
/// compatibility is not, and therefore cannot be deferred to a driver.
pub(crate) fn validate_ray_tracing_pipeline_descriptor(
    desc: &RayTracingPipelineDescriptor,
    device: DeviceIdentity,
    facts: PipelineDeviceFacts<'_>,
) -> RhiResult<()> {
    let max_depth = (facts.limit)(LimitKey::MaxRayRecursionDepth).unwrap_or(0);
    if desc.max_recursion_depth == 0 || u64::from(desc.max_recursion_depth) > max_depth {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "ray recursion depth is zero or exceeds the device limit",
        ));
    }
    validate_pipeline_interface_descriptor(
        desc.interface.descriptor(),
        facts.limit,
        facts.binding_limit,
    )?;
    let mut modules: Vec<(ShaderStage, &ShaderModule)> =
        vec![(ShaderStage::RayGeneration, &desc.ray_generation)];
    modules.extend(desc.miss.iter().map(|module| (ShaderStage::Miss, module)));
    for group in &desc.hit_groups {
        if group.closest_hit.is_none() && group.any_hit.is_none() && group.intersection.is_none() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a ray-tracing hit group needs at least one hit or intersection program",
            ));
        }
        for (module, stage) in [
            (&group.closest_hit, ShaderStage::ClosestHit),
            (&group.any_hit, ShaderStage::AnyHit),
            (&group.intersection, ShaderStage::Intersection),
        ] {
            if let Some(module) = module {
                modules.push((stage, module));
            }
        }
    }
    for (stage, module) in &modules {
        validate_module(module, device, *stage)?;
        // Ray payload/attribute ABI is code-form specific and deliberately not
        // reinterpreted as raster locations here. The backend's shader-acceptance
        // fact owns that native ABI decision; portable resource requirements are
        // still merged and checked below.
        let acceptance = (facts.shader_acceptance)(module.artifact());
        if acceptance != ArtifactAcceptance::Accepted {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device does not accept a ray-tracing shader artifact: {acceptance:?}"
                ),
            ));
        }
    }
    let merged = merge_shader_resources(
        modules
            .iter()
            .map(|(stage, module)| (*stage, &module.artifact().interface)),
    )?;
    validate_shader_resource_requirements(&merged, &desc.interface, facts.binding_support)?;
    validate_shader_immediate_requirements(
        modules
            .iter()
            .map(|(stage, module)| (*stage, &module.artifact().interface)),
        &desc.interface,
    )
}
fn validate_module(
    module: &ShaderModule,
    device: DeviceIdentity,
    stage: ShaderStage,
) -> RhiResult<()> {
    if module.device_identity() != device {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "ray-tracing shader belongs to a different device",
        )
        .with_object(module.id()));
    }
    if module.artifact().stage != stage {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "ray-tracing shader module has the wrong stage",
        ));
    }
    Ok(())
}
