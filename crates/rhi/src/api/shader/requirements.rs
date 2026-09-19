//! Sections 19.5-19.7: what one entry point requires.
//!
//! The resource requirements in the binding vocabulary, the location interface,
//! the compute workgroup shape, and the optional features and device limits the
//! entry point needs. This is the caller's statement; it is deliberately not a
//! verdict about any device.
//!
//! Not owned here: the vocabulary the requirements are written in (19.1-19.4, in
//! `vocabulary.rs`) and the rules that check the statement (19.6-19.7, in
//! `validation.rs`). Section 19.5 reuses `BindingKind` and `BindingCount`
//! directly rather than defining shader-side copies, so the two cannot drift.

use crate::api::binding::{BindGroupIndex, BindingCount, BindingKind, BindingSlotId};
use crate::api::platform::requirements::{LimitRequirement, OptionalFeature};

use super::vocabulary::ShaderLocationInterface;

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
}

impl ShaderInterface {
    /// An empty interface.
    ///
    /// Legal as a starting point and legal as the whole thing for a compute entry
    /// point, which may declare no locations at all.
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
}

/// What one compute entry point requires of a workgroup.
///
/// Section 19.7 requires this to be reflection output of the *entry point*, not a
/// dispatch decision: it says what the shader was compiled to need. The dispatch
/// dimensions a caller later passes to `dispatch` are validated separately, and
/// conflating the two would let a pipeline that cannot run be created.
///
/// The public constructor is deliberate: this is prober/producer data that the
/// artifact itself carries, not a device fact, so there is nothing to forge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeWorkgroupRequirements {
    /// Workgroup size on X.
    pub x: u32,
    /// Workgroup size on Y.
    pub y: u32,
    /// Workgroup size on Z.
    pub z: u32,

    /// Total invocations per workgroup.
    ///
    /// Must equal `x * y * z` without overflow: the two spellings of the same
    /// number are both carried because a backend needs each of them, and a
    /// mismatch means the reflection is internally inconsistent.
    pub total_invocations: u32,

    /// Workgroup/shared-memory bytes this entry point requires.
    pub workgroup_storage_bytes: u64,
}

impl ComputeWorkgroupRequirements {
    /// Assembles one entry point's workgroup requirements.
    pub fn new(
        x: u32,
        y: u32,
        z: u32,
        total_invocations: u32,
        workgroup_storage_bytes: u64,
    ) -> Self {
        Self {
            x,
            y,
            z,
            total_invocations,
            workgroup_storage_bytes,
        }
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
    compute_workgroup: Option<ComputeWorkgroupRequirements>,
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

    /// Declares the workgroup requirements of a compute entry point.
    pub fn with_compute_workgroup(mut self, requirements: ComputeWorkgroupRequirements) -> Self {
        self.compute_workgroup = Some(requirements);
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

    /// The workgroup requirements, for a compute entry point.
    ///
    /// Required to be `Some` for compute and `None` for vertex and fragment; the
    /// two halves of that rule are checked by `validate_shader_artifact`.
    pub fn compute_workgroup(&self) -> Option<ComputeWorkgroupRequirements> {
        self.compute_workgroup
    }
}
