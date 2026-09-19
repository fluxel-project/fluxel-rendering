//! Pipeline interfaces, raster pipelines, and compute pipelines.
//!
//! This module owns rhi-design sections 23, 27, and 28, plus the fixed-state
//! and vertex-input vocabulary from sections 24 to 26 in [`state`].
//!
//! # What it is
//!
//! A [`PipelineInterface`] is the logical layout contract between shader entry
//! points and the bind group packets they read. A [`RasterPipeline`] and a
//! [`ComputePipeline`] bind that contract to concrete shader modules and fixed
//! state, and carry the [`RenderTargetSignature`] that says which attachments
//! they may be used with.
//!
//! None of these is a `VkPipelineLayout`, a `D3D12` root signature, a Metal
//! argument-buffer layout, or a GL uniform-location table. Those are lowerings.
//!
//! # What it deliberately does not own
//!
//! Inline parameters (push constants, root constants, immediate bytes) are not
//! reserved here. WebGPU's immediate-data facility does not make them base
//! semantics across five backends, so the chapter keeps them as a separately
//! reviewed capability family rather than pre-adding a `reserved_inline_range`
//! that no consumer fills.
//!
//! A persistent pipeline cache file format is not frozen either. What is frozen
//! is that every pipeline can be completely re-described from its shader
//! artifacts, its canonical interface descriptors, its fixed state, and its
//! target signature. A native PSO binary is never a portable correctness
//! source.
//!
//! # Why compatibility is two tokens
//!
//! [`PipelineInterfaceCompatibilityId`] is an exact same-device interning token:
//! two identical canonical descriptors produce the same token, and the token
//! cannot be constructed by a caller. [`LayoutFingerprint`] is a canonical hash
//! for caches, diagnostics, and capture provenance. They are deliberately not
//! interchangeable, and an equal fingerprint never replaces full canonical
//! semantic comparison.

pub mod state;

use std::sync::Arc;

use super::binding::{
    BindGroupIndex, BindGroupLayout, BindingKind, BindingLimitClass, BindingSlotId,
    BufferBindingAccess, LayoutFingerprint, PipelineInterfaceCompatibilityId, StorageAccess,
};
use super::platform::{
    DeviceIdentity, Label, LimitKey, ObjectId, RhiError, RhiErrorKind, RhiResult,
};
use super::shader::{CanonicalShaderInterface, ShaderModule, ShaderStage, ShaderStages};

pub use state::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWriteMask,
    CullMode, DepthBiasState, DepthState, DepthStencilState, FrontFace, IndexFormat,
    MultisampleState, PrimitiveState, PrimitiveTopology, RenderTargetSignature, StencilFaceState,
    StencilOperation, StencilState, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexInputState, VertexStepMode,
};

use state::{
    validate_depth_stencil_state, validate_multisample_state, validate_primitive_state,
    validate_vertex_input, validate_vertex_shader_inputs, VertexInputLimits,
};

/// The ordered group layouts a pipeline's shaders are written against.
///
/// The vector index of `groups` is the [`BindGroupIndex`]. A hole must be
/// filled with an explicit empty layout: needing groups 0 and 2 means declaring
/// three entries, so every backend sees one stable logical numbering instead of
/// inventing its own compaction.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PipelineInterfaceDescriptor {
    /// A diagnostic label.
    pub label: Label,
    /// The group layouts, indexed by bind group index.
    pub groups: Vec<BindGroupLayout>,
}

impl PipelineInterfaceDescriptor {
    /// A descriptor over `groups`.
    pub fn new(groups: Vec<BindGroupLayout>) -> Self {
        Self {
            label: Label::none(),
            groups,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }
}

/// A canonical logical pipeline layout.
#[derive(Clone)]
pub struct PipelineInterface {
    inner: Arc<dyn PipelineInterfaceBackend>,
}

impl core::fmt::Debug for PipelineInterface {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipelineInterface")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl PipelineInterface {
    pub(crate) fn new(inner: Arc<dyn PipelineInterfaceBackend>) -> Self {
        Self { inner }
    }

    /// This interface's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this interface belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The canonical descriptor.
    pub fn descriptor(&self) -> &PipelineInterfaceDescriptor {
        self.inner.descriptor()
    }

    /// The device-scoped exact compatibility token.
    pub fn compatibility_id(&self) -> PipelineInterfaceCompatibilityId {
        self.inner.compatibility_id()
    }

    /// The canonical descriptor fingerprint.
    pub fn fingerprint(&self) -> LayoutFingerprint {
        self.inner.fingerprint()
    }

    /// The layout at `index`, or `None` when the interface has no such group.
    pub fn group(&self, index: BindGroupIndex) -> Option<&BindGroupLayout> {
        self.inner.descriptor().groups.get(index.get() as usize)
    }
}

/// The backend half of a [`PipelineInterface`].
pub(crate) trait PipelineInterfaceBackend: Send + Sync + 'static {
    /// This interface's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this interface belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The canonical descriptor.
    fn descriptor(&self) -> &PipelineInterfaceDescriptor;

    /// The device-scoped exact compatibility token.
    fn compatibility_id(&self) -> PipelineInterfaceCompatibilityId;

    /// The canonical descriptor fingerprint.
    fn fingerprint(&self) -> LayoutFingerprint;
}

/// The portable limits a pipeline interface is measured against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InterfaceLimits {
    /// `LimitKey::MaxBindGroups`.
    pub max_bind_groups: u64,
    /// `LimitKey::MaxDynamicUniformBuffersPerPipelineLayout`, when reported.
    pub max_dynamic_uniform_buffers: Option<u64>,
    /// `LimitKey::MaxDynamicStorageBuffersPerPipelineLayout`, when reported.
    pub max_dynamic_storage_buffers: Option<u64>,
}

/// Validates group count, device identity, and aggregate per-stage limits.
///
/// Per-stage resource counts cannot be decided at the individual layout stage
/// because they span several groups; this is the one place that aggregates
/// them.
pub(crate) fn validate_pipeline_interface(
    device: DeviceIdentity,
    desc: &PipelineInterfaceDescriptor,
    limits: InterfaceLimits,
    binding_limit: &dyn Fn(ShaderStage, BindingLimitClass) -> Option<u32>,
) -> RhiResult<()> {
    if desc.groups.len() as u64 > limits.max_bind_groups {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "pipeline interface declares more groups than MaxBindGroups",
        ));
    }
    for group in &desc.groups {
        if group.device_identity() != device {
            return Err(RhiError::wrong_device(
                "pipeline interface mixes bind group layouts from another device",
            ));
        }
    }

    let mut dynamic_uniform: u64 = 0;
    let mut dynamic_storage: u64 = 0;
    for group in &desc.groups {
        for entry in &group.descriptor().entries {
            if entry.dynamic_offset {
                match entry.kind {
                    BindingKind::UniformBuffer { .. } => dynamic_uniform += 1,
                    BindingKind::StorageBuffer { .. } => dynamic_storage += 1,
                    _ => {}
                }
            }
        }
    }
    if let Some(limit) = limits.max_dynamic_uniform_buffers {
        if dynamic_uniform > limit {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "pipeline interface exceeds MaxDynamicUniformBuffersPerPipelineLayout",
            ));
        }
    }
    if let Some(limit) = limits.max_dynamic_storage_buffers {
        if dynamic_storage > limit {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "pipeline interface exceeds MaxDynamicStorageBuffersPerPipelineLayout",
            ));
        }
    }

    const CLASSES: [BindingLimitClass; 5] = [
        BindingLimitClass::UniformBuffers,
        BindingLimitClass::StorageBuffers,
        BindingLimitClass::SampledTextures,
        BindingLimitClass::StorageTextures,
        BindingLimitClass::Samplers,
    ];
    for stage in [ShaderStage::Vertex, ShaderStage::Fragment, ShaderStage::Compute] {
        let stage_bits = match stage {
            ShaderStage::Vertex => ShaderStages::VERTEX,
            ShaderStage::Fragment => ShaderStages::FRAGMENT,
            ShaderStage::Compute => ShaderStages::COMPUTE,
        };
        for class in CLASSES {
            let Some(limit) = binding_limit(stage, class) else {
                continue;
            };
            let mut total: u64 = 0;
            for group in &desc.groups {
                for entry in &group.descriptor().entries {
                    if !entry.visibility.contains(stage_bits) {
                        continue;
                    }
                    if class_of(&entry.kind) == class {
                        total += u64::from(entry.count.elements());
                    }
                }
            }
            if total > u64::from(limit) {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "pipeline interface exceeds a per-stage binding limit",
                ));
            }
        }
    }
    Ok(())
}

fn class_of(kind: &BindingKind) -> BindingLimitClass {
    match kind {
        BindingKind::UniformBuffer { .. } => BindingLimitClass::UniformBuffers,
        BindingKind::StorageBuffer { .. } => BindingLimitClass::StorageBuffers,
        BindingKind::SampledTexture { .. } => BindingLimitClass::SampledTextures,
        BindingKind::StorageTexture { .. } => BindingLimitClass::StorageTextures,
        BindingKind::Sampler { .. } => BindingLimitClass::Samplers,
    }
}

/// One binding element a pipeline's shaders actually reference.
///
/// This is the merged result across every participating stage, so when the
/// vertex and fragment stages read the same slot with the same kind it appears
/// once. It is what turns "the bind group holds ten resources" into "this draw
/// consumes three".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceMergeOutcome {
    /// The binding kind the merged requirement settles on.
    pub kind: BindingKind,
    /// The slot that carries it.
    pub slot: BindingSlotId,
    /// The group that carries it.
    pub group: BindGroupIndex,
}

/// Checks that one shader's resources are served by the interface.
///
/// `stages` is the stage mask of the entry points actually participating, which
/// is what the layout visibility must cover.
pub(crate) fn validate_shader_resources(
    shaders: &[(ShaderStage, CanonicalShaderInterface<'_>)],
    interface: &PipelineInterface,
) -> RhiResult<Vec<ResourceMergeOutcome>> {
    let mut outcomes: Vec<ResourceMergeOutcome> = Vec::new();
    for (stage, canonical) in shaders {
        let stage_bits = stage_mask(*stage);
        for requirement in canonical.interface().resources() {
            let Some(group) = interface.group(requirement.group) else {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "shader requires group {} but the pipeline interface has no such group",
                        requirement.group.get()
                    ),
                ));
            };
            let Some(entry) = group
                .descriptor()
                .entries
                .iter()
                .find(|entry| entry.slot == requirement.slot)
            else {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "shader requires slot {}:{} but the pipeline interface does not declare it",
                        requirement.group.get(),
                        requirement.slot.get()
                    ),
                ));
            };
            if !entry.visibility.contains(stage_bits) {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "slot {}:{} is not visible to stage {stage:?}",
                        requirement.group.get(),
                        requirement.slot.get()
                    ),
                ));
            }
            if entry.count != requirement.count {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "slot {}:{} count does not match the shader requirement",
                        requirement.group.get(),
                        requirement.slot.get()
                    ),
                ));
            }
            let merged = merge_kind(&entry.kind, &requirement.kind)?;
            if let Some(existing) = outcomes.iter().find(|outcome| {
                outcome.group == requirement.group && outcome.slot == requirement.slot
            }) {
                if existing.kind != merged {
                    return Err(RhiError::new(
                        RhiErrorKind::IncompatibleInterface,
                        format!(
                            "stages disagree about the binding kind at {}:{}",
                            requirement.group.get(),
                            requirement.slot.get()
                        ),
                    ));
                }
                continue;
            }
            outcomes.push(ResourceMergeOutcome {
                kind: merged,
                slot: requirement.slot,
                group: requirement.group,
            });
        }
    }
    Ok(outcomes)
}

fn stage_mask(stage: ShaderStage) -> ShaderStages {
    match stage {
        ShaderStage::Vertex => ShaderStages::VERTEX,
        ShaderStage::Fragment => ShaderStages::FRAGMENT,
        ShaderStage::Compute => ShaderStages::COMPUTE,
    }
}

/// Merges a layout entry's kind with what a shader requires.
///
/// Storage access merges as a small lattice. Read-only plus read-only stays
/// read-only; anything else becomes read-write, and if the device cannot serve
/// read-write the pipeline is refused rather than quietly narrowed.
fn merge_kind(layout: &BindingKind, required: &BindingKind) -> RhiResult<BindingKind> {
    match (layout, required) {
        (BindingKind::UniformBuffer { min_size: layout }, BindingKind::UniformBuffer { min_size }) => {
            if layout < min_size {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    "layout uniform buffer minimum size is below the shader requirement",
                ));
            }
            Ok(required.clone())
        }
        (
            BindingKind::StorageBuffer {
                access: layout_access,
                min_size: layout_min_size,
            },
            BindingKind::StorageBuffer { access, min_size },
        ) => {
            if layout_min_size < min_size {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    "layout storage buffer minimum size is below the shader requirement",
                ));
            }
            Ok(BindingKind::StorageBuffer {
                access: merge_buffer_access(*layout_access, *access),
                min_size: *min_size,
            })
        }
        (BindingKind::SampledTexture { .. }, BindingKind::SampledTexture { .. }) => {
            if layout != required {
                return Err(kind_mismatch());
            }
            Ok(required.clone())
        }
        (BindingKind::StorageTexture { access: layout, .. }, BindingKind::StorageTexture { access, .. }) => {
            if strip_storage_texture_access(layout) != strip_storage_texture_access(access) {
                return Err(kind_mismatch());
            }
            let BindingKind::StorageTexture {
                dimension,
                format,
                ..
            } = required
            else {
                return Err(kind_mismatch());
            };
            Ok(BindingKind::StorageTexture {
                dimension: *dimension,
                format: *format,
                access: merge_storage_access(*layout, *access),
            })
        }
        (BindingKind::Sampler { kind: layout }, BindingKind::Sampler { kind }) => {
            if layout != kind {
                return Err(kind_mismatch());
            }
            Ok(required.clone())
        }
        _ => Err(kind_mismatch()),
    }
}

fn kind_mismatch() -> RhiError {
    RhiError::new(
        RhiErrorKind::IncompatibleInterface,
        "the layout binding kind does not match the shader requirement",
    )
}

fn merge_buffer_access(layout: BufferBindingAccess, required: BufferBindingAccess) -> BufferBindingAccess {
    if layout == BufferBindingAccess::ReadWrite || required == BufferBindingAccess::ReadWrite {
        BufferBindingAccess::ReadWrite
    } else {
        BufferBindingAccess::ReadOnly
    }
}

fn merge_storage_access(layout: StorageAccess, required: StorageAccess) -> StorageAccess {
    if layout == required {
        return layout;
    }
    StorageAccess::ReadWrite
}

fn strip_storage_texture_access(access: &StorageAccess) -> u8 {
    match access {
        StorageAccess::ReadOnly => 0,
        StorageAccess::WriteOnly => 1,
        StorageAccess::ReadWrite => 2,
    }
}

/// The portable limits a raster pipeline's target set is measured against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RasterLimits {
    /// `LimitKey::MaxColorAttachments`.
    pub max_color_attachments: u64,
    /// `LimitKey::MaxInterStageShaderVariables`.
    pub max_inter_stage_variables: u64,
    /// `LimitKey::MaxBindGroupsPlusVertexBuffers`, when reported.
    pub max_bind_groups_plus_vertex_buffers: Option<u64>,
    /// Forwarded to vertex-input validation.
    pub vertex: VertexInputLimits,
}

/// A raster pipeline description.
#[non_exhaustive]
#[derive(Clone)]
pub struct RasterPipelineDescriptor {
    /// A diagnostic label.
    pub label: Label,
    /// The vertex entry point.
    pub vertex: ShaderModule,
    /// The optional fragment entry point.
    pub fragment: Option<ShaderModule>,
    /// The logical layout the shaders are written against.
    pub interface: PipelineInterface,
    /// The vertex fetch layout.
    pub vertex_input: VertexInputState,
    /// Primitive assembly state.
    pub primitive: PrimitiveState,
    /// Optional depth/stencil state.
    pub depth_stencil: Option<DepthStencilState>,
    /// Multisample state.
    pub multisample: MultisampleState,
    /// Vector index is the color output location.
    pub color_targets: Vec<Option<ColorTargetState>>,
}

impl RasterPipelineDescriptor {
    /// A minimal descriptor: vertex only, no vertex input, triangle list, no
    /// depth/stencil, one sample, no color targets.
    pub fn new(vertex: ShaderModule, interface: PipelineInterface) -> Self {
        Self {
            label: Label::none(),
            vertex,
            fragment: None,
            interface,
            vertex_input: VertexInputState::new(),
            primitive: PrimitiveState::new(PrimitiveTopology::TriangleList),
            depth_stencil: None,
            multisample: MultisampleState::new(1),
            color_targets: Vec::new(),
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// Adds a fragment entry point.
    pub fn with_fragment(mut self, fragment: ShaderModule) -> Self {
        self.fragment = Some(fragment);
        self
    }

    /// Sets the vertex input state.
    pub fn with_vertex_input(mut self, state: VertexInputState) -> Self {
        self.vertex_input = state;
        self
    }

    /// Sets the primitive state.
    pub fn with_primitive(mut self, state: PrimitiveState) -> Self {
        self.primitive = state;
        self
    }

    /// Sets the depth/stencil state.
    pub fn with_depth_stencil(mut self, state: DepthStencilState) -> Self {
        self.depth_stencil = Some(state);
        self
    }

    /// Sets the multisample state.
    pub fn with_multisample(mut self, state: MultisampleState) -> Self {
        self.multisample = state;
        self
    }

    /// Sets a color target, extending the vector with empty slots as needed.
    pub fn with_color_target(
        mut self,
        location: super::shader::ShaderLocation,
        target: ColorTargetState,
    ) -> Self {
        let index = location.get() as usize;
        if self.color_targets.len() <= index {
            self.color_targets.resize(index + 1, None);
        }
        self.color_targets[index] = Some(target);
        self
    }

    /// The target set this pipeline is built against.
    pub fn target_signature(&self) -> RenderTargetSignature {
        RenderTargetSignature::new(
            self.color_targets
                .iter()
                .map(|target| target.as_ref().map(|target| target.format))
                .collect(),
            self.depth_stencil.as_ref().map(|state| state.format),
            self.multisample.count,
        )
    }
}

/// A raster pipeline, valid only against its target signature.
#[derive(Clone)]
pub struct RasterPipeline {
    inner: Arc<dyn RasterPipelineBackend>,
}

impl core::fmt::Debug for RasterPipeline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RasterPipeline")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl RasterPipeline {
    pub(crate) fn new(inner: Arc<dyn RasterPipelineBackend>) -> Self {
        Self { inner }
    }

    /// This pipeline's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this pipeline belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The descriptor this pipeline was built from.
    pub fn descriptor(&self) -> &RasterPipelineDescriptor {
        self.inner.descriptor()
    }

    /// The logical layout this pipeline uses.
    pub fn interface(&self) -> &PipelineInterface {
        &self.inner.descriptor().interface
    }

    /// The attachment set this pipeline may be used with.
    pub fn target_signature(&self) -> &RenderTargetSignature {
        self.inner.target_signature()
    }

    /// The binding elements this pipeline's shaders actually reference.
    ///
    /// A bind group may hold resources the shaders never name; only the
    /// elements in this list are consumed when the pipeline draws, so only they
    /// enter a command's actual resource use.
    pub fn used_bindings(&self) -> &[ResourceMergeOutcome] {
        self.inner.used_bindings()
    }
}

/// The backend half of a [`RasterPipeline`].
pub(crate) trait RasterPipelineBackend: Send + Sync + 'static {
    /// This pipeline's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this pipeline belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The descriptor this pipeline was built from.
    fn descriptor(&self) -> &RasterPipelineDescriptor;

    /// The attachment set this pipeline may be used with.
    fn target_signature(&self) -> &RenderTargetSignature;

    /// The binding elements this pipeline's shaders actually reference.
    fn used_bindings(&self) -> &[ResourceMergeOutcome];
}

/// A compute pipeline description.
#[non_exhaustive]
#[derive(Clone)]
pub struct ComputePipelineDescriptor {
    /// A diagnostic label.
    pub label: Label,
    /// The compute entry point.
    pub shader: ShaderModule,
    /// The logical layout the shader is written against.
    pub interface: PipelineInterface,
}

impl ComputePipelineDescriptor {
    /// A descriptor over `shader` and `interface`.
    pub fn new(shader: ShaderModule, interface: PipelineInterface) -> Self {
        Self {
            label: Label::none(),
            shader,
            interface,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }
}

/// A compute pipeline.
#[derive(Clone)]
pub struct ComputePipeline {
    inner: Arc<dyn ComputePipelineBackend>,
}

impl core::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl ComputePipeline {
    pub(crate) fn new(inner: Arc<dyn ComputePipelineBackend>) -> Self {
        Self { inner }
    }

    /// This pipeline's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this pipeline belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The descriptor this pipeline was built from.
    pub fn descriptor(&self) -> &ComputePipelineDescriptor {
        self.inner.descriptor()
    }

    /// The logical layout this pipeline uses.
    pub fn interface(&self) -> &PipelineInterface {
        &self.inner.descriptor().interface
    }

    /// The binding elements this pipeline's shaders actually reference.
    pub fn used_bindings(&self) -> &[ResourceMergeOutcome] {
        self.inner.used_bindings()
    }
}

/// The backend half of a [`ComputePipeline`].
pub(crate) trait ComputePipelineBackend: Send + Sync + 'static {
    /// This pipeline's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this pipeline belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The descriptor this pipeline was built from.
    fn descriptor(&self) -> &ComputePipelineDescriptor;

    /// The binding elements this pipeline's shaders actually reference.
    fn used_bindings(&self) -> &[ResourceMergeOutcome];
}

/// The portable limits a compute pipeline's workgroup shape is measured
/// against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ComputeLimits {
    /// `LimitKey::MaxComputeWorkgroupSizeX`.
    pub max_size_x: u64,
    /// `LimitKey::MaxComputeWorkgroupSizeY`.
    pub max_size_y: u64,
    /// `LimitKey::MaxComputeWorkgroupSizeZ`.
    pub max_size_z: u64,
    /// `LimitKey::MaxComputeInvocationsPerWorkgroup`.
    pub max_invocations: u64,
    /// `LimitKey::MaxComputeWorkgroupStorageSize`.
    pub max_workgroup_storage: u64,
}

/// Validates a compute entry point's workgroup requirements.
///
/// These are the entry point's reflection requirements, not dispatch
/// dimensions; dispatch dimensions are checked separately, at dispatch time.
pub(crate) fn validate_compute_workgroup(
    requirements: &super::shader::ComputeWorkgroupRequirements,
    limits: ComputeLimits,
) -> RhiResult<()> {
    if requirements.x == 0 || requirements.y == 0 || requirements.z == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a compute workgroup dimension must not be zero",
        ));
    }
    let Some(product) = requirements
        .x
        .checked_mul(requirements.y)
        .and_then(|value| value.checked_mul(requirements.z))
    else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "compute workgroup dimensions overflow",
        ));
    };
    if product != requirements.total_invocations {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "compute total invocations does not equal x * y * z",
        ));
    }
    if u64::from(requirements.x) > limits.max_size_x
        || u64::from(requirements.y) > limits.max_size_y
        || u64::from(requirements.z) > limits.max_size_z
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "compute workgroup dimensions exceed the device limits",
        ));
    }
    if u64::from(requirements.total_invocations) > limits.max_invocations {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "compute total invocations exceeds MaxComputeInvocationsPerWorkgroup",
        ));
    }
    if requirements.workgroup_storage_bytes > limits.max_workgroup_storage {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "compute workgroup storage exceeds MaxComputeWorkgroupStorageSize",
        ));
    }
    Ok(())
}

/// Validates everything about a raster descriptor that does not need the
/// backend.
///
/// Returns the participating canonical shader interfaces so a caller does not
/// have to canonicalize them twice.
pub(crate) fn validate_raster_descriptor(
    device: DeviceIdentity,
    desc: &RasterPipelineDescriptor,
    limits: RasterLimits,
    binding_limit: &dyn Fn(ShaderStage, BindingLimitClass) -> Option<u32>,
    limit: &dyn Fn(LimitKey) -> Option<u64>,
) -> RhiResult<Vec<ResourceMergeOutcome>> {
    if desc.vertex.device_identity() != device
        || desc.interface.device_identity() != device
        || desc
            .fragment
            .as_ref()
            .is_some_and(|fragment| fragment.device_identity() != device)
    {
        return Err(RhiError::wrong_device(
            "a raster pipeline mixes objects from another device",
        ));
    }
    if desc.vertex.stage() != ShaderStage::Vertex {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a raster pipeline's vertex module is not a vertex entry point",
        ));
    }
    if let Some(fragment) = &desc.fragment {
        if fragment.stage() != ShaderStage::Fragment {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a raster pipeline's fragment module is not a fragment entry point",
            ));
        }
    }

    validate_primitive_state(&desc.primitive)?;
    validate_multisample_state(&desc.multisample)?;
    validate_vertex_input(&desc.vertex_input, limits.vertex)?;

    let vertex_interface = desc.vertex.artifact().interface.canonical()?;
    let fragment_interface = match &desc.fragment {
        Some(fragment) => Some(fragment.artifact().interface.canonical()?),
        None => None,
    };

    if !vertex_interface.interface().writes_position() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the vertex entry point does not write clip position",
        ));
    }

    let mut shaders = vec![(ShaderStage::Vertex, vertex_interface)];
    if let Some(fragment) = fragment_interface {
        shaders.push((ShaderStage::Fragment, fragment));
    }
    let used_bindings = validate_shader_resources(&shaders, &desc.interface)?;

    validate_vertex_shader_inputs(vertex_interface.interface(), &desc.vertex_input)?;

    // Vertex to fragment linkage: every fragment input must be produced by a
    // vertex output with a compatible shape. Extra vertex outputs are legal.
    if let Some(fragment) = fragment_interface {
        for input in fragment.interface().inputs() {
            let Some(output) = vertex_interface
                .interface()
                .outputs()
                .iter()
                .find(|output| output.location == input.location)
            else {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "fragment input location {} has no vertex output",
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
                        "vertex output and fragment input disagree at location {}",
                        input.location.get()
                    ),
                ));
            }
        }
    }

    let inter_stage_variables = vertex_interface.interface().outputs().len() as u64
        + vertex_interface.interface().inputs().len() as u64
        + fragment_interface
            .map(|fragment| {
                fragment.interface().inputs().len() as u64
                    + fragment.interface().outputs().len() as u64
            })
            .unwrap_or(0);
    if inter_stage_variables > limits.max_inter_stage_variables {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the pipeline exceeds MaxInterStageShaderVariables",
        ));
    }

    // Color targets.
    let active_targets = desc
        .color_targets
        .iter()
        .filter(|target| target.is_some())
        .count() as u64;
    if active_targets > limits.max_color_attachments {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the pipeline declares more color targets than MaxColorAttachments",
        ));
    }
    if desc.fragment.is_none() && active_targets > 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a pipeline with no fragment stage cannot declare color targets",
        ));
    }
    for (location, target) in desc.color_targets.iter().enumerate() {
        let Some(target) = target else {
            continue;
        };
        let facts = super::format::format_facts(target.format);
        if !facts.color_attachment() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!("color target {location} has a format that is not a color attachment"),
            ));
        }
        if let Some(blend) = &target.blend {
            if !facts.blendable() {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    format!("color target {location} blends a format that is not blendable"),
                ));
            }
            if !blend.is_well_formed() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!("color target {location} uses a non-unit factor with Min or Max"),
                ));
            }
        }
        match fragment_interface {
            Some(fragment) => {
                match fragment
                    .interface()
                    .outputs()
                    .iter()
                    .find(|output| output.location.get() as usize == location)
                {
                    Some(output) => {
                        if Some(output.numeric_type) != facts.color_output_type() {
                            return Err(RhiError::new(
                                RhiErrorKind::IncompatibleInterface,
                                format!(
                                    "fragment output at location {location} does not match the target format"
                                ),
                            ));
                        }
                    }
                    None => {
                        if target.write_mask != ColorWriteMask::NONE {
                            return Err(RhiError::new(
                                RhiErrorKind::InvalidUsage,
                                format!(
                                    "color target {location} has no fragment output, so its write mask must be empty"
                                ),
                            ));
                        }
                    }
                }
            }
            None => {}
        }
    }

    // Depth/stencil and fragment depth writes.
    if let Some(state) = &desc.depth_stencil {
        let facts = validate_depth_stencil_state(state)?;
        if let Some(facts) = facts {
            if state.depth.is_some() && !facts.depth_attachment() {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "the depth format is not usable as an attachment",
                ));
            }
        }
    }
    if let Some(fragment) = fragment_interface {
        if fragment.interface().writes_frag_depth() {
            let Some(state) = &desc.depth_stencil else {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the fragment stage writes depth but the pipeline has no depth attachment",
                ));
            };
            if !super::format::format_facts(state.format)
                .aspects()
                .contains(super::resource::TextureAspects::DEPTH)
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the fragment stage writes depth but the pipeline format has no depth aspect",
                ));
            }
        }
    }

    // Alpha-to-coverage has a deliberately narrow contract.
    if desc.multisample.alpha_to_coverage_enabled {
        let Some(fragment) = fragment_interface else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage requires a fragment stage",
            ));
        };
        let Some(Some(target)) = desc.color_targets.first() else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage requires a target at color location 0",
            ));
        };
        let writes_float_rgba = fragment.interface().outputs().iter().any(|output| {
            output.location.get() == 0
                && output.components == 4
                && output.numeric_type == super::shader::ShaderNumericType::Float32
        });
        if !writes_float_rgba {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage requires a Float32 vec4 fragment output at location 0",
            ));
        }
        if !super::format::format_facts(target.format).has_alpha_channel() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage requires a target format with an alpha channel",
            ));
        }
        if fragment.interface().writes_sample_mask() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage and an explicit sample mask output cannot be combined",
            ));
        }
    }

    if let Some(limit) = limits.max_bind_groups_plus_vertex_buffers {
        let total = desc.interface.descriptor().groups.len() as u64
            + desc.vertex_input.buffers.len() as u64;
        if total > limit {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "bind groups plus vertex buffers exceed MaxBindGroupsPlusVertexBuffers",
            ));
        }
    }

    let _ = (limit, binding_limit);
    Ok(used_bindings)
}

/// Validates everything about a compute descriptor that does not need the
/// backend.
pub(crate) fn validate_compute_descriptor(
    device: DeviceIdentity,
    desc: &ComputePipelineDescriptor,
    limits: ComputeLimits,
    compute_enabled: bool,
) -> RhiResult<Vec<ResourceMergeOutcome>> {
    if !compute_enabled {
        return Err(RhiError::unsupported(
            "the device does not enable the compute feature",
        ));
    }
    if desc.shader.device_identity() != device || desc.interface.device_identity() != device {
        return Err(RhiError::wrong_device(
            "a compute pipeline mixes objects from another device",
        ));
    }
    if desc.shader.stage() != ShaderStage::Compute {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a compute pipeline's module is not a compute entry point",
        ));
    }
    let interface = desc.shader.artifact().interface.canonical()?;
    if !interface.interface().inputs().is_empty() || !interface.interface().outputs().is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a compute entry point must declare no stage inputs or outputs",
        ));
    }
    if interface.interface().writes_position()
        || interface.interface().writes_frag_depth()
        || interface.interface().writes_sample_mask()
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a compute entry point must not declare fragment or vertex outputs",
        ));
    }
    let Some(workgroup) = desc.shader.artifact().requirements.compute_workgroup() else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a compute entry point must declare its workgroup requirements",
        ));
    };
    validate_compute_workgroup(&workgroup, limits)?;
    let used_bindings = validate_shader_resources(&[(ShaderStage::Compute, interface)], &desc.interface)?;
    Ok(used_bindings)
}

#[cfg(test)]
mod tests;
