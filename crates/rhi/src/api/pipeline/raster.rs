//! Section 26-27: `RenderTargetSignature`, `RasterPipeline`, and the raster
//! validator.
//!
//! Section 27.3's ten validation blocks, all of which must pass before a backend
//! is touched. The target signature lives here because it is the raster
//! pipeline's own shape, canonicalized by removing trailing `None` targets; it is
//! a value a caller can hold and compare, not a rule.
//!
//! Not owned here: the fixed state the descriptor carries (section 25,
//! `raster_state.rs`), the interface it is checked against (sections 23.1-23.2)
//! and the merged shader requirements (section 23.3).

use core::fmt;
use std::sync::Arc;

use crate::api::binding::{BindingLimitClass, BindingSupportQuery};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, TextureSupportQuery, logical_bytes_per_block};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::pipeline::backend::RasterPipelineBackend;
use crate::api::platform::Device;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::texture::{TextureDimension, TextureUsage};
use crate::api::shader::{
    ShaderArtifact, ShaderLocation, ShaderLocationInterface, ShaderModule, ShaderNumericType,
    ShaderStage,
};

use super::interface::{PipelineInterface, validate_pipeline_interface_descriptor};
use super::raster_state::{
    ColorTargetState, ColorWriteMask, DepthStencilState, MultisampleState, PrimitiveState,
    PrimitiveTopology,
};
use super::resources::{merge_shader_resources, validate_shader_resource_requirements};
use super::vertex_input::{
    VertexInputState, validate_vertex_input_against_interface, validate_vertex_input_state,
};
use super::{ColorTargetFacts, PipelineDeviceFacts};

// ---------------------------------------------------------------------------
// Section 26 - Pipeline target signature
// ---------------------------------------------------------------------------

/// The color and depth-stencil shape a raster pipeline was built for.
///
/// The vector index is the fragment output / color attachment location, and a
/// `None` entry means that location has no color target. Holes are what make this
/// one shape describe both the WebGPU fragment target sequence — which allows
/// `null` slots — and the sparse MRT locations Vulkan, D3D12, and Metal express
/// natively.
///
/// Published so that a target signature can be compared and hashed without a
/// pipeline existing, which is what section 28.1's cache rule needs. It is
/// deliberately *not* `#[non_exhaustive]`: it is a value a caller both reads and
/// constructs, and every field is part of the comparison.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderTargetSignature {
    /// Vector index = fragment output / color attachment location.
    ///
    /// None = this location has no color target.
    pub color_formats: Vec<Option<TextureFormat>>,

    /// The depth and/or stencil attachment format, or `None`.
    pub depth_stencil_format: Option<TextureFormat>,

    /// The sample count every attachment must have.
    pub sample_count: u32,
}

impl RenderTargetSignature {
    /// Removes trailing `None` entries from `color_formats`.
    ///
    /// Section 26's canonicalization, and the reason it exists: `[RGBA8, None,
    /// None]` and `[RGBA8]` describe the same pipeline, and two representations of
    /// one signature would make a cache key or a Capture comparison wrong about
    /// identical pipelines. Interior `None`s are kept — they are real holes.
    ///
    /// The canonical form is what [`RasterPipelineDescriptor::target_signature`]
    /// returns and what a created pipeline stores, so the two answers a caller can
    /// get are always the same string of bytes.
    pub(crate) fn canonicalized(mut self) -> Self {
        while matches!(self.color_formats.last(), Some(None)) {
            self.color_formats.pop();
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Section 27 - RasterPipeline
// ---------------------------------------------------------------------------

/// Everything a caller states about a raster pipeline before it exists.
///
/// The fields are exactly the four things section 28.1 requires a pipeline to be
/// re-describable by — the [`ShaderArtifact`](crate::api::shader::ShaderArtifact)s behind the two modules, the
/// interface's canonical descriptor, the fixed state, and the target signature —
/// plus the vertex input, which is the fourth member of section 27.3's own
/// validation list.
#[non_exhaustive]
#[derive(Clone)]
pub struct RasterPipelineDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// The vertex entry point.
    pub vertex: ShaderModule,
    /// The fragment entry point, when the pipeline has one.
    pub fragment: Option<ShaderModule>,

    /// The logical layout contract bound to these entry points.
    pub interface: PipelineInterface,

    /// The vertex buffer bindings and attributes.
    pub vertex_input: VertexInputState,
    /// The primitive and rasterization state.
    pub primitive: PrimitiveState,

    /// The depth and stencil state, or `None` for a pipeline with no such
    /// attachment.
    pub depth_stencil: Option<DepthStencilState>,
    /// The multisample state.
    pub multisample: MultisampleState,

    /// Vector index = color output location.
    pub color_targets: Vec<Option<ColorTargetState>>,
}

impl RasterPipelineDescriptor {
    /// Creates a minimal graphics descriptor.
    ///
    /// defaults:
    /// - no fragment stage
    /// - empty vertex input
    /// - TriangleList
    /// - no depth/stencil
    /// - sample count 1
    /// - no color targets
    pub fn new(vertex: ShaderModule, interface: PipelineInterface) -> Self {
        Self {
            label: Label::default(),
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

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// Adds the fragment stage.
    pub fn with_fragment(mut self, fragment: ShaderModule) -> Self {
        self.fragment = Some(fragment);
        self
    }

    /// Replaces the vertex input state.
    pub fn with_vertex_input(mut self, state: VertexInputState) -> Self {
        self.vertex_input = state;
        self
    }

    /// Replaces the primitive state.
    pub fn with_primitive(mut self, state: PrimitiveState) -> Self {
        self.primitive = state;
        self
    }

    /// Adds a depth/stencil state.
    pub fn with_depth_stencil(mut self, state: DepthStencilState) -> Self {
        self.depth_stencil = Some(state);
        self
    }

    /// Replaces the multisample state.
    pub fn with_multisample(mut self, state: MultisampleState) -> Self {
        self.multisample = state;
        self
    }

    /// Automatically extends the vector; intermediate locations are filled with None.
    ///
    /// The extension is the point: a caller who wants a target at location 2 says
    /// so, and gets `[None, None, Some(..)]` rather than a target silently landing
    /// at index 0. The holes are visible in [`Self::target_signature`], which is
    /// what keeps a sparse MRT pipeline describable.
    pub fn with_color_target(mut self, location: ShaderLocation, target: ColorTargetState) -> Self {
        let index = location.get() as usize;
        if self.color_targets.len() <= index {
            self.color_targets.resize(index + 1, None);
        }
        self.color_targets[index] = Some(target);
        self
    }

    /// The canonical target signature of this descriptor.
    ///
    /// Canonical, in section 26's sense: trailing `None` entries are removed, so
    /// two descriptors that differ only in unbound high locations produce one
    /// signature. A created [`RasterPipeline`] stores this value rather than
    /// recomputing it, which is what makes
    /// [`RasterPipeline::target_signature`] answer the same thing for the lifetime
    /// of the pipeline.
    pub fn target_signature(&self) -> RenderTargetSignature {
        RenderTargetSignature {
            color_formats: self
                .color_targets
                .iter()
                .map(|target| target.as_ref().map(|target| target.format))
                .collect(),
            depth_stencil_format: self.depth_stencil.as_ref().map(|state| state.format),
            sample_count: self.multisample.count,
        }
        .canonicalized()
    }
}

/// A created raster pipeline.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the [`DeviceIdentity`]
/// that created it. It stores its descriptor and its canonical target signature,
/// because section 28.1 requires a pipeline to be completely re-described by them
/// and because comparing a pipeline against a render pass is a per-frame question
/// that must not re-derive the signature each time.
#[derive(Clone)]
pub struct RasterPipeline {
    inner: Arc<RasterPipelineInner>,
}

/// The one ownership domain of a created raster pipeline.
struct RasterPipelineInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: RasterPipelineDescriptor,
    target_signature: RenderTargetSignature,
    /// Native graphics state retained by the portable handle.  It is only
    /// reachable by crate-private command lowering.
    #[cfg_attr(not(any(feature = "dx12", feature = "vulkan")), allow(dead_code))]
    native: Box<dyn RasterPipelineBackend>,
}

impl RasterPipeline {
    /// Assembles a created pipeline.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only `Device::create_raster_pipeline` may produce one. The signature is the
    /// canonical one, computed from the descriptor it is given.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: RasterPipelineDescriptor,
        native: Box<dyn RasterPipelineBackend>,
    ) -> Self {
        let target_signature = descriptor.target_signature();
        Self {
            inner: Arc::new(RasterPipelineInner {
                id,
                device,
                descriptor,
                target_signature,
                native,
            }),
        }
    }

    /// This pipeline's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }

    /// The device that created this pipeline.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }

    /// The descriptor this pipeline was created from.
    pub fn descriptor(&self) -> &RasterPipelineDescriptor {
        &self.inner.descriptor
    }

    /// The pipeline interface this pipeline was created with.
    pub fn interface(&self) -> &PipelineInterface {
        &self.inner.descriptor.interface
    }

    /// The canonical target signature.
    ///
    /// Stored rather than recomputed: a render pass compares against it, and
    /// section 26 makes one canonical representation of each signature the whole
    /// point.
    pub fn target_signature(&self) -> &RenderTargetSignature {
        &self.inner.target_signature
    }

    /// The backend graphics state used by native command lowering.
    #[cfg_attr(not(any(feature = "dx12", feature = "vulkan")), allow(dead_code))]
    pub(crate) fn native(&self) -> &dyn RasterPipelineBackend {
        self.inner.native.as_ref()
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (defect D6 of the 0.16 plan): section 27.2
/// declares `#[derive(Clone)]` and no `Debug`, and a pipeline is exactly the
/// object a caller needs to name in a log. The descriptor is one call away through
/// [`RasterPipeline::descriptor`].
impl fmt::Debug for RasterPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RasterPipeline")
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
            .finish_non_exhaustive()
    }
}

/// Checks everything about a raster pipeline that does not need a backend.
///
/// Section 27.3's ten blocks, in its own order. Each block below names the
/// sub-heading it implements so that the list can be read against the code
/// without counting:
///
/// ```text
/// Device / stage            identities, and the stage of each module
/// Shader resources          section 23's merge, interface, visibility/kind/count
/// Vertex input              section 24.2's two lists
/// Vertex -> Fragment        every fragment input is provided and compatible
/// Fragment outputs          numeric type against the target, write-mask rule
/// Fragment depth            frag-depth writes require a depth aspect
/// Target facts              attachment/blend/alpha-channel facts per target
/// Limits                    color attachments, bytes per sample, inter-stage vars
/// Strip topology            strip index format only for strips; bias only for triangles
/// ```
///
/// The `MaxBindGroupsPlusVertexBuffers` check lives here rather than in
/// [`validate_pipeline_interface_descriptor`] because it is the one member of
/// section 23.1's aggregate list that needs the vertex input, which only a raster
/// pipeline has; section 27.3's "Limits" block names the same limit, which is why
/// it appears once.
pub(crate) fn validate_raster_pipeline_descriptor(
    desc: &RasterPipelineDescriptor,
    facts: PipelineDeviceFacts<'_>,
) -> RhiResult<()> {
    let limit = facts.limit;

    // --- Device / stage ---------------------------------------------------
    let device = desc.interface.device_identity();
    if desc.vertex.device_identity() != device {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "the vertex shader belongs to a different device than the pipeline interface",
        )
        .with_object(desc.vertex.id()));
    }
    if let Some(fragment) = desc.fragment.as_ref() {
        if fragment.device_identity() != device {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the fragment shader belongs to a different device than the pipeline interface",
            )
            .with_object(fragment.id()));
        }
    }

    if desc.vertex.stage() != ShaderStage::Vertex {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a raster pipeline's vertex entry point has stage {:?}",
                desc.vertex.stage()
            ),
        ));
    }
    if let Some(fragment) = desc.fragment.as_ref() {
        if fragment.stage() != ShaderStage::Fragment {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a raster pipeline's fragment entry point has stage {:?}",
                    fragment.stage()
                ),
            ));
        }
    }

    // --- Shader resources (section 23) -----------------------------------
    validate_pipeline_interface_descriptor(
        desc.interface.descriptor(),
        facts.limit,
        facts.binding_limit,
    )?;

    let vertex_interface = &desc.vertex.artifact().interface;
    let merged = match desc.fragment.as_ref() {
        Some(fragment) => merge_shader_resources([
            (ShaderStage::Vertex, vertex_interface),
            (ShaderStage::Fragment, &fragment.artifact().interface),
        ])?,
        None => merge_shader_resources([(ShaderStage::Vertex, vertex_interface)])?,
    };
    validate_shader_resource_requirements(&merged, &desc.interface, facts.binding_support)?;

    // Section 23.1's raster addendum, which section 27.3's "Limits" block repeats.
    if let Some(max) = limit(LimitKey::MaxBindGroupsPlusVertexBuffers) {
        let total = desc.interface.descriptor().groups.len() + desc.vertex_input.buffers.len();
        if total as u64 > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the pipeline uses {total} bind groups and vertex buffers together, over the \
                     device maximum of {max}"
                ),
            ));
        }
    }

    // --- Vertex input (section 24.2) -------------------------------------
    validate_vertex_input_state(&desc.vertex_input, facts.limit)?;
    validate_vertex_input_against_interface(&desc.vertex_input, vertex_interface)?;

    // --- Vertex -> Fragment inter-stage linkage --------------------------
    let fragment_interface = desc
        .fragment
        .as_ref()
        .map(|fragment| &fragment.artifact().interface);
    if let Some(fragment_interface) = fragment_interface {
        for input in fragment_interface.inputs() {
            let Some(output) = find_location(vertex_interface.outputs(), input.location) else {
                return Err(RhiError::new(
                    RhiErrorKind::IncompatibleInterface,
                    format!(
                        "the fragment stage reads location {}, which the vertex stage does not \
                         write",
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
                        "location {} crosses the stages as {:?}x{} {:?} but is read as {:?}x{} {:?}",
                        input.location.get(),
                        output.numeric_type,
                        output.components,
                        output.interpolation,
                        input.numeric_type,
                        input.components,
                        input.interpolation
                    ),
                ));
            }
        }
    }

    // --- Fragment outputs -------------------------------------------------
    // With no fragment stage, section 27.3's two per-location rules have nothing
    // to compare against, and the block's own closing rule — every color target
    // must be None — is the stronger one, so it is checked first.
    if fragment_interface.is_none() {
        if let Some(index) = desc
            .color_targets
            .iter()
            .position(|target| target.is_some())
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a pipeline with no fragment stage must have no color target, but location \
                     {index} has one"
                ),
            ));
        }
    } else {
        let fragment_outputs = fragment_interface.map(|interface| interface.outputs());
        let outputs: &[ShaderLocationInterface] = fragment_outputs.unwrap_or(&[]);
        for (index, target) in desc.color_targets.iter().enumerate() {
            let Some(target) = target else {
                continue;
            };
            let location = ShaderLocation::new(index as u32);
            match find_location(outputs, location) {
                Some(output) => {
                    let expected = (facts.color_target_facts)(target.format)
                        .and_then(|facts| facts.color_output_type);
                    if expected != Some(output.numeric_type) {
                        return Err(RhiError::new(
                            RhiErrorKind::Unsupported,
                            format!(
                                "color location {index} writes {:?} but {:?} produces {expected:?}",
                                output.numeric_type, target.format
                            ),
                        ));
                    }
                }
                None => {
                    if target.write_mask != ColorWriteMask::NONE {
                        return Err(RhiError::new(
                            RhiErrorKind::InvalidUsage,
                            format!(
                                "color location {index} has no fragment output, so its write mask \
                                 must be ColorWriteMask::NONE"
                            ),
                        ));
                    }
                }
            }
        }
    }

    // --- Fragment depth ---------------------------------------------------
    if let Some(fragment_interface) = fragment_interface {
        if fragment_interface.writes_frag_depth() {
            let carries_depth = desc
                .depth_stencil
                .as_ref()
                .is_some_and(|state| format_has_depth(state.format));
            if !carries_depth {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the fragment stage writes the fragment depth built-in, so the pipeline needs \
                     a depth-stencil state whose format carries the Depth aspect",
                ));
            }
        }
    }

    // --- Target facts -----------------------------------------------------
    // Section 25.3's consistency rule is checked here because this is where the
    // format and the two optional states meet.
    if let Some(state) = desc.depth_stencil.as_ref() {
        let aspects = crate::api::format::format_aspects(state.format);
        if state.depth.is_some()
            && !aspects.contains(crate::api::resource::subresource::TextureAspects::DEPTH)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("{:?} carries no Depth aspect to test", state.format),
            ));
        }
        if state.stencil.is_some()
            && !aspects.contains(crate::api::resource::subresource::TextureAspects::STENCIL)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("{:?} carries no Stencil aspect to test", state.format),
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
                format!(
                    "this device cannot create {:?} as a depth-stencil attachment with sample \
                     count {}",
                    state.format, desc.multisample.count
                ),
            ));
        }
    }

    for (index, target) in desc.color_targets.iter().enumerate() {
        let Some(target) = target else {
            continue;
        };
        let Some(target_facts) = (facts.color_target_facts)(target.format) else {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "{:?} is not a color target format on this device",
                    target.format
                ),
            ));
        };
        if !target_facts.color_attachment {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!("{:?} cannot be used as a color attachment", target.format),
            ));
        }
        if target.blend.is_some() && !target_facts.blendable {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!("{:?} cannot be blended", target.format),
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
                format!(
                    "this device cannot create {:?} as a color attachment at location {index} \
                     with sample count {}",
                    target.format, desc.multisample.count
                ),
            ));
        }
    }

    // --- Limits -----------------------------------------------------------
    let active_targets = desc
        .color_targets
        .iter()
        .filter(|target| target.is_some())
        .count();
    if let Some(max) = limit(LimitKey::MaxColorAttachments) {
        if active_targets as u64 > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the pipeline writes {active_targets} color attachments, over the device \
                     maximum of {max}"
                ),
            ));
        }
    }

    // "MaxColorAttachmentBytesPerSample": the sum of the active color targets'
    // bytes per texel, which is their bytes per sample. A format whose block size
    // is not fixed (the depth formats) cannot be an active color target — the
    // Target facts block refuses it first — so it contributes nothing.
    if let Some(max) = limit(LimitKey::MaxColorAttachmentBytesPerSample) {
        let mut total = 0u64;
        for target in desc.color_targets.iter().flatten() {
            if let Some(bytes) = logical_bytes_per_block(target.format) {
                total += bytes as u64;
            }
        }
        if total > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the active color targets need {total} bytes per sample, over the device \
                     maximum of {max}"
                ),
            ));
        }
    }

    // "MaxInterStageShaderVariables": the specification lists the limit without
    // fixing what is counted. It is counted here as the fragment stage's input
    // locations, which are the consumer side of the variables that actually cross
    // the boundary; the vertex outputs are the producer side of the same
    // variables, and section 27.3 explicitly permits the vertex stage to write
    // more than the fragment stage reads.
    if let Some(max) = limit(LimitKey::MaxInterStageShaderVariables) {
        let count = fragment_interface
            .map(|interface| interface.inputs().len())
            .unwrap_or(0) as u64;
        if count > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the fragment stage reads {count} inter-stage variables, over the device \
                     maximum of {max}"
                ),
            ));
        }
    }

    // --- Strip topology ---------------------------------------------------
    // One rule, not two: a strip index format says which index format restarts a
    // strip, so it is meaningful only where there are strips to restart.
    if !desc.primitive.topology.is_strip() && desc.primitive.strip_index_format.is_some() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "{:?} is not a strip topology, so it cannot declare a strip index format",
                desc.primitive.topology
            ),
        ));
    }

    // Section 25.1: depth bias is a triangle-topology feature in P0, and a
    // non-finite slope scale is not a state any backend can lower.
    if let Some(bias) = desc.primitive.depth_bias.as_ref() {
        if !matches!(
            desc.primitive.topology,
            PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip
        ) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "depth bias is available only for triangle topology in P0, not {:?}",
                    desc.primitive.topology
                ),
            ));
        }
        if !bias.slope_scale.is_finite() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a depth bias slope scale must be finite",
            ));
        }
    }

    // Section 25.4: alpha-to-coverage is a multisampled feature.
    if desc.multisample.alpha_to_coverage_enabled && desc.multisample.count <= 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "alpha-to-coverage is valid only when the sample count is greater than one",
        ));
    }

    // Section 27.3's alpha-to-coverage block, which is about the fragment stage
    // rather than about the multisample state.
    if desc.multisample.alpha_to_coverage_enabled {
        let Some(fragment_interface) = fragment_interface else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage requires a fragment stage",
            ));
        };
        let Some(target) = desc
            .color_targets
            .first()
            .and_then(|target| target.as_ref())
        else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage requires a color target at location 0",
            ));
        };
        let output = find_location(fragment_interface.outputs(), ShaderLocation::new(0));
        let writes_float4 = output.is_some_and(|output| {
            output.numeric_type == ShaderNumericType::Float32 && output.components == 4
        });
        if !writes_float4 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage requires the fragment stage to write a Float32 vec4 at \
                 location 0",
            ));
        }
        let has_alpha =
            (facts.color_target_facts)(target.format).is_some_and(|facts| facts.has_alpha_channel);
        if !has_alpha {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "alpha-to-coverage requires a location 0 target with an alpha channel, and \
                     {:?} has none",
                    target.format
                ),
            ));
        }
        if fragment_interface.writes_sample_mask() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "alpha-to-coverage and the sample-mask built-in cannot be used together",
            ));
        }
    }

    Ok(())
}

/// The location entry with this logical location, if any.
fn find_location(
    list: &[ShaderLocationInterface],
    location: ShaderLocation,
) -> Option<&ShaderLocationInterface> {
    list.iter().find(|entry| entry.location == location)
}

/// Whether a format carries the Depth aspect.
///
/// Asked through [`crate::api::format::format_aspects`] rather than through a
/// `FormatFacts`, because at this point in validation there may be no facts for
/// the format at all, and "does this format have a depth aspect" is a fact of the
/// format name.
fn format_has_depth(format: TextureFormat) -> bool {
    crate::api::format::format_aspects(format)
        .contains(crate::api::resource::subresource::TextureAspects::DEPTH)
}

/// Section 27.2's creation verb, defined in the chapter that owns the type it
/// produces.
///
/// The placement is the specification's own: section 27.2 writes this verb in an
/// `impl Device` in its own chapter, so the definition site is the owner.
impl Device {
    /// Creates a raster pipeline on this device from a descriptor.
    ///
    /// Section 3.1's O(1) identity step comes first, over the three device-owned
    /// objects the descriptor names: the interface and the two shader modules.
    /// Section 27.3's device and stage block compares the modules against the
    /// *interface's* device, so proving the interface is this device's is the
    /// façade's half of the rule; without it, a descriptor whose parts all agree
    /// with each other but belong to another device would validate and say nothing
    /// about this one.
    ///
    /// Section 27.3's ten validation blocks then run, through
    /// `validate_raster_pipeline_descriptor`, against the seven device answers
    /// the descriptor-bag carries. Those answers are read from this device rather
    /// than passed in by the caller, because section 7.2 makes the device's own
    /// answers — not the adapter's snapshot — the ones that decide legality.
    ///
    /// Panics until a backend port exists. Both steps above still run first,
    /// because each refusal they produce is a statement about the descriptor that
    /// a caller can act on without any device object having been allocated.
    pub async fn create_raster_pipeline(
        &self,
        desc: &RasterPipelineDescriptor,
    ) -> RhiResult<RasterPipeline> {
        let identity = self.identity();
        if desc.interface.device_identity() != identity {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the pipeline interface belongs to a different device",
            )
            .with_object(desc.interface.id()));
        }
        if desc.vertex.device_identity() != identity {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the vertex shader belongs to a different device",
            )
            .with_object(desc.vertex.id()));
        }
        if let Some(fragment) = desc.fragment.as_ref() {
            if fragment.device_identity() != identity {
                return Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "the fragment shader belongs to a different device",
                )
                .with_object(fragment.id()));
            }
        }

        // Section 6.5's liveness verdict, after every ownership comparison above
        // and before the device's seven answers are read. A descriptor naming a
        // foreign interface or shader is `WrongDevice` even on a lost device:
        // section 3.1 puts those comparisons first.
        self.require_active()?;

        let capabilities = self.capabilities();

        // Each closure is one of the seven questions `PipelineDeviceFacts` names,
        // answered by the method of the same name on `EnabledCapabilities`. They
        // are locals rather than inline struct-literal fields because the bag
        // holds `&dyn Fn` references, and a reference needs a binding to point at.
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

        validate_raster_pipeline_descriptor(
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
        )?;

        let native = self.native().create_raster_pipeline(desc)?;
        Ok(RasterPipeline::new(
            ObjectId::next(),
            identity,
            desc.clone(),
            native,
        ))
    }
}
