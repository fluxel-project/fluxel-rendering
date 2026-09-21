//! Sections 19.5-19.7: what one entry point requires.
//!
//! The resource requirements in the binding vocabulary, the location interface,
//! and the optional features and device limits the entry point needs. This is the
//! caller's statement; it is deliberately not a
//! verdict about any device.
//!
//! Not owned here: the vocabulary the requirements are written in (19.1-19.4, in
//! `vocabulary.rs`) and the rules that check the statement (19.6-19.7, in
//! `validation.rs`). Section 19.5 reuses `BindingKind` and `BindingCount`
//! directly rather than defining shader-side copies, so the two cannot drift.

use crate::api::binding::{
    BindGroupIndex, BindingCount, BindingKind, BindingSlotId, BindingSupportQuery,
};
use crate::api::platform::requirements::{LimitRequirement, OptionalFeature};

use super::vocabulary::{ShaderLocationInterface, ShaderStage, stage_mask};

/// Inclusive subgroup-size interval reported by a device.
///
/// The values are grouped instead of exposed as two unrelated limits so an
/// invalid `min > max` fact cannot be represented by a published capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubgroupSizeRange {
    /// Smallest supported subgroup width.
    pub min: u32,
    /// Largest supported subgroup width.
    pub max: u32,
}

/// The fixed three-dimensional local size of a compute entry point.
///
/// This is entry-point metadata, rather than a dispatch parameter: a dispatch
/// selects how many workgroups run, while shader code fixes how many invocations
/// each workgroup contains.  It deliberately remains representable even when an
/// axis is zero so [`ShaderArtifact`](super::ShaderArtifact) validation can report
/// the producer error instead of a builder silently repairing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeWorkgroupSize {
    /// Local invocations on the X axis.
    pub x: u32,
    /// Local invocations on the Y axis.
    pub y: u32,
    /// Local invocations on the Z axis.
    pub z: u32,
}

impl ComputeWorkgroupSize {
    /// Describes the local size declared by the compute entry point.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub(crate) fn invocation_count(self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }
}

impl SubgroupSizeRange {
    /// Creates a non-empty inclusive range.
    pub fn new(min: u32, max: u32) -> Option<Self> {
        (min != 0 && min <= max).then_some(Self { min, max })
    }

    /// Whether `size` is in the reported interval.
    pub const fn contains(self, size: u32) -> bool {
        self.min <= size && size <= self.max
    }

    pub(crate) fn encode_into(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.min.to_le_bytes());
        out.extend_from_slice(&self.max.to_le_bytes());
    }
}

/// A shader builtin whose portable semantics must be available to an entry
/// point. Native lowering remains private; unsupported builtins are rejected by
/// the corresponding required feature before shader compilation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderBuiltin {
    /// Draw ordinal within a multi-draw command.
    DrawIndex,
    /// Raster primitive ordinal.
    PrimitiveIndex,
    /// Per-vertex invocation data.
    PerVertex,
    /// Fragment barycentric coordinates.
    Barycentrics,
    /// Clip-distance output.
    ClipDistance,
    /// Vertex positions returned by a committed ray-query intersection.
    ///
    /// This is separate from the acceleration-structure binding and ordinary
    /// ray-query operation: native APIs gate fetching hit-triangle vertex data
    /// behind an additional feature.
    RayHitVertexPosition,
}

/// One cooperative-matrix operation required by a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CooperativeMatrixRequirement {
    /// Matrix rows.
    pub rows: u32,
    /// Matrix columns.
    pub columns: u32,
    /// Contracting dimension.
    pub depth: u32,
    /// Stages which execute the operation.
    pub stages: crate::api::shader::ShaderStages,
}

/// Scalar component type used by a cooperative matrix operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CooperativeMatrixComponentType {
    /// IEEE binary16.
    Float16,
    /// IEEE binary32.
    Float32,
    /// Signed 8-bit integer.
    Sint8,
    /// Unsigned 8-bit integer.
    Uint8,
    /// Signed 16-bit integer.
    Sint16,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Signed 32-bit integer.
    Sint32,
    /// Unsigned 32-bit integer.
    Uint32,
}

/// Execution granularity of a cooperative matrix operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CooperativeMatrixScope {
    /// One subgroup executes the operation cooperatively.
    Subgroup,
    /// One workgroup executes the operation cooperatively.
    Workgroup,
}

/// One adapter-probed cooperative-matrix shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CooperativeMatrixProperties {
    /// Matrix rows.
    pub rows: u32,
    /// Matrix columns.
    pub columns: u32,
    /// Contracting dimension.
    pub depth: u32,
    /// Operand component type.
    pub component_type: CooperativeMatrixComponentType,
    /// Result component type.
    pub result_type: CooperativeMatrixComponentType,
    /// Supported shader stages.
    pub stages: crate::api::shader::ShaderStages,
    /// Execution scope.
    pub scope: CooperativeMatrixScope,
}

impl CooperativeMatrixProperties {
    /// Returns whether this probed native shape satisfies the portable requirement.
    pub fn satisfies(self, requirement: CooperativeMatrixRequirement) -> bool {
        self.rows == requirement.rows
            && self.columns == requirement.columns
            && self.depth == requirement.depth
            && self.stages.contains(requirement.stages)
    }

    pub(crate) fn encode_into(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.columns.to_le_bytes());
        out.extend_from_slice(&self.depth.to_le_bytes());
        out.push(self.component_type as u8);
        out.push(self.result_type as u8);
        self.stages.encode_into(out);
        out.push(self.scope as u8);
    }
}

/// One resource an entry point requires, in the RHI binding vocabulary.
///
/// Section 19.5 reuses [`BindingKind`] and [`BindingCount`] directly rather than
/// defining a shader-side vocabulary for them, so that reflection and
/// `BindGroupLayout` cannot slowly diverge into two systems that disagree about
/// what a storage texture is.
///
/// `dynamic_offset` is deliberately absent: it is not a shader semantic. Whether
/// a dynamic offset is used is decided by the layout, and the shader sees only the
/// resolved resource (section 19.5).
#[derive(Clone, Debug)]
pub struct ShaderResourceRequirement {
    /// The logical group the resource lives in.
    pub group: BindGroupIndex,
    /// The slot within that group.
    pub slot: BindingSlotId,

    /// The resource semantics this entry point actually requires.
    pub kind: BindingKind,

    /// The fixed resource count of this logical binding in shader code.
    pub count: BindingCount,
}

impl ShaderResourceRequirement {
    /// This requirement as the capability query it implies, for an entry point at
    /// `stage`.
    ///
    /// One place rather than two, because section 19.7 makes both askers ask the
    /// same thing: `validate_shader_artifact` asks before the backend is touched,
    /// and `acceptance::decide` asks as part of the device's verdict. A requirement
    /// is answered by asking [`BindingSupportQuery`] about it — never by repeating
    /// capability in [`ShaderRequirements`] — so building the query twice is how the
    /// two answers start to differ.
    ///
    /// `dynamic_offset` is false because it is a layout fact, not a shader semantic
    /// (section 19.5): whether a dynamic offset is used is decided by the layout,
    /// and the shader sees only the resolved resource.
    ///
    /// Crate-private: a caller's own spelling of this query is two lines, and
    /// publishing a builder for it would declare a convenience the specification
    /// does not have.
    pub(crate) fn binding_query(&self, stage: ShaderStage) -> BindingSupportQuery {
        BindingSupportQuery {
            visibility: stage_mask(stage),
            kind: self.kind.clone(),
            count: self.count,
            dynamic_offset: false,
        }
    }
}

/// The portable semantics of one entry point.
///
/// A canonical, duplicate-free description. The builder methods below collect
/// entries; they do not sort or deduplicate them, because section 19.6 makes a
/// duplicate or non-canonical interface a rejection rather than something the RHI
/// may silently repair:
///
/// ```text
/// resources  (group, slot) unique, ordered lexicographically by (group, slot)
/// inputs     location unique, ascending
/// outputs    location unique, ascending
/// ```
///
/// Input and output locations are separate namespaces: a vertex input at location
/// 0 and a vertex output at location 0 are both legal and are not a collision.
///
/// Built-ins (`vertex_index`, `front_facing`, `global_invocation_id`, and the
/// rest of section 19.6's list) do not occupy a location and do not appear here.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ShaderInterface {
    resources: Vec<ShaderResourceRequirement>,
    inputs: Vec<ShaderLocationInterface>,
    outputs: Vec<ShaderLocationInterface>,
    writes_position: bool,
    writes_frag_depth: bool,
    writes_sample_mask: bool,
    compute_workgroup_size: Option<ComputeWorkgroupSize>,
}

impl ShaderInterface {
    /// An empty interface.
    ///
    /// Legal as a starting point. A complete compute entry point must additionally
    /// declare its local size with [`Self::with_compute_workgroup_size`], although
    /// it may declare no locations at all.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one resource requirement.
    pub fn with_resource(mut self, requirement: ShaderResourceRequirement) -> Self {
        self.resources.push(requirement);
        self
    }

    /// Adds one input location.
    pub fn with_input(mut self, input: ShaderLocationInterface) -> Self {
        self.inputs.push(input);
        self
    }

    /// Adds one output location.
    pub fn with_output(mut self, output: ShaderLocationInterface) -> Self {
        self.outputs.push(output);
        self
    }

    /// Declares whether this entry point writes the position built-in.
    ///
    /// Must be true for a vertex entry point; a vertex shader that does not write
    /// the position cannot produce geometry at all.
    pub fn with_writes_position(mut self, value: bool) -> Self {
        self.writes_position = value;
        self
    }

    /// Declares whether this entry point writes the fragment depth built-in.
    ///
    /// Optional for a fragment entry point. Writing it requires the pipeline to
    /// carry a depth-stencil state whose format has a depth aspect (section 27.3).
    pub fn with_writes_frag_depth(mut self, value: bool) -> Self {
        self.writes_frag_depth = value;
        self
    }

    /// Declares whether this entry point writes the sample-mask built-in.
    pub fn with_writes_sample_mask(mut self, value: bool) -> Self {
        self.writes_sample_mask = value;
        self
    }

    /// Declares the fixed local invocation shape of a compute entry point.
    ///
    /// Exactly one shape is required for [`ShaderStage::Compute`]; it is forbidden
    /// on every other stage.  The artifact validator checks that every axis is
    /// non-zero, and device acceptance compares all three axes and their product
    /// against the enabled compute limits.
    pub fn with_compute_workgroup_size(mut self, size: ComputeWorkgroupSize) -> Self {
        self.compute_workgroup_size = Some(size);
        self
    }

    /// The resource requirements, in the order they were added.
    ///
    /// A valid artifact has them already canonical; this accessor does not sort.
    pub fn resources(&self) -> &[ShaderResourceRequirement] {
        &self.resources
    }

    /// The input locations, in the order they were added.
    pub fn inputs(&self) -> &[ShaderLocationInterface] {
        &self.inputs
    }

    /// The output locations, in the order they were added.
    pub fn outputs(&self) -> &[ShaderLocationInterface] {
        &self.outputs
    }

    /// Whether this entry point writes the position built-in.
    pub fn writes_position(&self) -> bool {
        self.writes_position
    }

    /// Whether this entry point writes the fragment depth built-in.
    pub fn writes_frag_depth(&self) -> bool {
        self.writes_frag_depth
    }

    /// Whether this entry point writes the sample-mask built-in.
    pub fn writes_sample_mask(&self) -> bool {
        self.writes_sample_mask
    }

    /// The declared compute local size, if this is a compute entry point.
    pub fn compute_workgroup_size(&self) -> Option<ComputeWorkgroupSize> {
        self.compute_workgroup_size
    }
}

/// What an entry point requires of the device beyond its interface.
///
/// Section 19.7 keeps binding capability *out* of this type: a resource
/// requirement is answered by asking
/// [`BindingSupportQuery`](crate::api::binding::BindingSupportQuery) about
/// [`ShaderInterface::resources`], and duplicating the answer here is how two
/// sources of truth for one fact start.
///
/// A valid artifact's collections are canonical: `required_features` unique and
/// sorted by discriminant, `limit_requirements` duplicate-free and sorted by
/// `(LimitKey, variant, value)`. See `validate_shader_artifact`.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ShaderRequirements {
    required_features: Vec<OptionalFeature>,
    limit_requirements: Vec<LimitRequirement>,
    builtins: Vec<ShaderBuiltin>,
    cooperative_matrices: Vec<CooperativeMatrixRequirement>,
}

impl ShaderRequirements {
    /// No stated requirements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requires an optional feature.
    pub fn require_feature(mut self, feature: OptionalFeature) -> Self {
        self.required_features.push(feature);
        self
    }

    /// Requires a device limit.
    pub fn require_limit(mut self, requirement: LimitRequirement) -> Self {
        self.limit_requirements.push(requirement);
        self
    }

    /// Requires a portable shader builtin.
    pub fn require_builtin(mut self, builtin: ShaderBuiltin) -> Self {
        self.builtins.push(builtin);
        self
    }

    /// Requires one cooperative-matrix configuration.
    pub fn require_cooperative_matrix(mut self, matrix: CooperativeMatrixRequirement) -> Self {
        self.cooperative_matrices.push(matrix);
        self
    }

    /// The required optional features, in the order they were added.
    pub fn required_features(&self) -> &[OptionalFeature] {
        &self.required_features
    }

    /// The required device limits, in the order they were added.
    pub fn limit_requirements(&self) -> &[LimitRequirement] {
        &self.limit_requirements
    }

    /// Builtins used by the entry point.
    pub fn builtins(&self) -> &[ShaderBuiltin] {
        &self.builtins
    }

    /// Cooperative-matrix configurations used by the entry point.
    pub fn cooperative_matrices(&self) -> &[CooperativeMatrixRequirement] {
        &self.cooperative_matrices
    }

    /// Feature implied by a builtin, owned here so acceptance and validation do
    /// not grow separate tables.
    pub(crate) fn builtin_feature(builtin: ShaderBuiltin) -> OptionalFeature {
        match builtin {
            ShaderBuiltin::DrawIndex => OptionalFeature::ShaderDrawIndex,
            ShaderBuiltin::PrimitiveIndex => OptionalFeature::PrimitiveIndex,
            ShaderBuiltin::PerVertex => OptionalFeature::ShaderPerVertex,
            ShaderBuiltin::Barycentrics => OptionalFeature::ShaderBarycentrics,
            ShaderBuiltin::ClipDistance => OptionalFeature::ClipDistances,
            ShaderBuiltin::RayHitVertexPosition => OptionalFeature::RayHitVertexReturn,
        }
    }
}
