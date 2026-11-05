//! Portable WGSL composition, validation, and reflection.
#![deny(missing_docs)]
mod capabilities;
pub use capabilities::NagaCapability;
use capabilities::required_naga_capabilities;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    sync::Arc,
};
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Caller-provided identity for one source contributor.
pub struct SourceUnitId(pub String);
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// A named WGSL source contributor.
pub struct SourceUnit {
    /// Identity used in source maps and diagnostics.
    pub id: SourceUnitId,
    /// WGSL text contributed by this unit.
    pub source: String,
}
impl SourceUnit {
    /// Creates a source unit with a caller-provided identity.
    pub fn new(id: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id: SourceUnitId(id.into()),
            source: source.into(),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Programmable stage of a selected entry point.
pub enum ShaderStage {
    /// Vertex stage.
    Vertex,
    /// Fragment stage.
    Fragment,
    /// Compute stage.
    Compute,
}
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Name and stage of a complete program entry point.
pub struct EntryPoint {
    /// WGSL entry-point name.
    pub name: String,
    /// Stage declared in WGSL.
    pub stage: ShaderStage,
}
impl EntryPoint {
    /// Creates an entry-point requirement.
    pub fn new(name: impl Into<String>, stage: ShaderStage) -> Self {
        Self {
            name: name.into(),
            stage,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Broad resource class, refined by an optional binding shape.
pub enum BindingKind {
    /// Uniform buffer.
    Uniform,
    /// Read-only storage buffer.
    StorageRead,
    /// Read-write storage buffer.
    StorageReadWrite,
    /// Sampler resource.
    Sampler,
    /// Sampled or depth texture.
    Texture,
    /// Storage texture.
    StorageTexture,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// A consumer declaration checked against entry-reachable resources.
pub struct BindingDeclaration {
    /// Bind-group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Broad resource class.
    pub kind: BindingKind,
    /// Optional exact shape; absence leaves that property unconstrained.
    pub shape: Option<BindingShape>,
    /// Optional exact set of stages that use this binding.
    pub visibility: Option<BTreeSet<ShaderStage>>,
    /// Optional exact entry-reachable access flags.
    pub access: Option<BindingAccess>,
}
impl BindingDeclaration {
    /// Declares a binding and its broad resource class.
    pub const fn new(group: u32, binding: u32, kind: BindingKind) -> Self {
        Self {
            group,
            binding,
            kind,
            shape: None,
            visibility: None,
            access: None,
        }
    }
    /// Requires a particular reflected resource shape.
    pub fn with_shape(mut self, shape: BindingShape) -> Self {
        self.shape = Some(shape);
        self
    }
    /// Requires the exact set of entry stages using the binding.
    pub fn with_visibility(mut self, stages: impl IntoIterator<Item = ShaderStage>) -> Self {
        self.visibility = Some(stages.into_iter().collect());
        self
    }
    /// Requires exact reflected access flags.
    pub fn with_access(mut self, access: BindingAccess) -> Self {
        self.access = Some(access);
        self
    }
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// Consumer geometry semantic mapped to a typed shader location.
pub struct VertexInputDeclaration {
    /// Consumer-defined semantic identifier.
    pub semantic: String,
    /// WGSL input location.
    pub location: u32,
    /// Required shader value type.
    pub ty: VertexType,
}
impl VertexInputDeclaration {
    /// Creates a semantic-to-location requirement.
    pub fn new(semantic: impl Into<String>, location: u32, ty: VertexType) -> Self {
        Self {
            semantic: semantic.into(),
            location,
            ty,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Supported scalar classes for logical input and callable types.
pub enum ScalarType {
    /// Half-precision floating point.
    F16,
    /// Single-precision floating point.
    F32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Portable scalar or vector type, independent of buffer packing.
pub enum VertexType {
    /// One scalar.
    Scalar(ScalarType),
    /// Two scalars.
    Vec2(ScalarType),
    /// Three scalars.
    Vec3(ScalarType),
    /// Four scalars.
    Vec4(ScalarType),
}
/// Scalar or vector type used by a declared callable.
pub type ValueType = VertexType;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Supported exact resource-shape requirements.
pub enum BindingShape {
    /// Uniform buffer with a scalar or vector root type.
    Uniform(ValueType),
    /// Non-arrayed, non-multisampled float 2D texture.
    Texture2dF32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Entry-reachable use flags reported by the logical validator.
pub struct BindingAccess {
    /// Data is read.
    pub read: bool,
    /// Data is written.
    pub write: bool,
    /// Resource metadata is queried.
    pub query: bool,
    /// Naga reports image-atomic access; buffer atomics may report read/write.
    pub atomic: bool,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// A function signature checked against the composed WGSL module.
pub struct CallableDeclaration {
    /// WGSL function name.
    pub name: String,
    /// Argument types in declaration order.
    pub arguments: Vec<ValueType>,
    /// Return type, or none for a function without a return value.
    pub result: Option<ValueType>,
}
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
/// Portable feature requirements declared by the consumer or inferred from WGSL.
pub struct PipelineRequirements {
    /// Names to be resolved by the selected execution layer.
    pub names: BTreeSet<String>,
    /// Naga capabilities required by the full validated WGSL module.
    ///
    /// This set is sufficient to validate the module again. It can include
    /// capabilities with no Fluxel RHI feature-name mapping, which the selected
    /// consumer must either map from queried support or reject explicitly.
    pub naga_capabilities: BTreeSet<NagaCapability>,
}
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
/// Complete logical input; concrete device resources are deliberately absent.
pub struct ShaderCompositionRequest {
    /// Ordered WGSL contributors.
    pub source_units: Vec<SourceUnit>,
    /// Selected program entry points.
    pub entry_points: Vec<EntryPoint>,
    /// Complete declarations of resources reachable from selected entries.
    pub bindings: Vec<BindingDeclaration>,
    /// Optional complete semantic mapping for vertex inputs.
    pub vertex_inputs: Vec<VertexInputDeclaration>,
    /// Function signatures requiring agreement with actual WGSL.
    pub callables: Vec<CallableDeclaration>,
    /// Caller-applied static choices affecting logical identity.
    pub static_options: BTreeMap<String, String>,
    /// Portable requirements, later checked by the actual device.
    pub requirements: PipelineRequirements,
}
impl ShaderCompositionRequest {
    /// Starts a request with ordered source contributors.
    pub fn new(source_units: Vec<SourceUnit>) -> Self {
        Self {
            source_units,
            ..Self::default()
        }
    }
    /// Adds a required entry point.
    pub fn with_entry_point(mut self, x: EntryPoint) -> Self {
        self.entry_points.push(x);
        self
    }
    /// Adds a resource binding declaration.
    pub fn with_binding(mut self, x: BindingDeclaration) -> Self {
        self.bindings.push(x);
        self
    }
    /// Adds a logical vertex-input mapping.
    pub fn with_vertex_input(mut self, x: VertexInputDeclaration) -> Self {
        self.vertex_inputs.push(x);
        self
    }
    /// Adds a function signature contract.
    pub fn with_callable(mut self, callable: CallableDeclaration) -> Self {
        self.callables.push(callable);
        self
    }
    /// Records a static choice already applied by the source contributor.
    pub fn with_static_option(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.static_options.insert(k.into(), v.into());
        self
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
/// Resource used by one or more selected program entries.
pub struct ReflectedBinding {
    /// Bind-group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Broad resource class.
    pub kind: BindingKind,
    /// Recognized exact shape, when supported by this version.
    pub shape: Option<BindingShape>,
    /// Selected stages that actually use the binding.
    pub visibility: BTreeSet<ShaderStage>,
    /// Exact per-entry usage; aggregate flags below are the union of these uses.
    pub entry_access: BTreeMap<EntryPoint, BindingAccess>,
    /// At least one selected entry reads data.
    pub read: bool,
    /// At least one selected entry writes data.
    pub write: bool,
    /// At least one selected entry queries resource metadata.
    pub query: bool,
    /// Naga reports image-atomic use; buffer atomics report read/write.
    pub atomic: bool,
}
#[derive(Clone, Debug, Eq, PartialEq)]
/// Reflected shader input, independent of physical stream layout.
pub struct ReflectedVertexInput {
    /// WGSL input location.
    pub location: u32,
    /// Scalar or vector value type at that location.
    pub ty: VertexType,
}
#[derive(Clone, Debug)]
/// Logical interface of the selected program entries.
pub struct ShaderInterface {
    entries: Vec<EntryPoint>,
    bindings: Vec<ReflectedBinding>,
    vertex_inputs: Vec<ReflectedVertexInput>,
}
impl ShaderInterface {
    /// Returns the selected entry points.
    pub fn entry_points(&self) -> &[EntryPoint] {
        &self.entries
    }
    /// Returns entry-reachable bindings and their usage.
    pub fn bindings(&self) -> &[ReflectedBinding] {
        &self.bindings
    }
    /// Returns vertex input locations and logical types.
    pub fn vertex_inputs(&self) -> &[ReflectedVertexInput] {
        &self.vertex_inputs
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
/// Structured source, validation, or declared-interface failure.
pub enum ShaderError {
    /// A request contained no source units.
    EmptySource,
    /// A request selected no entry point.
    EmptyEntries,
    /// The same entry was selected more than once.
    DuplicateEntry(EntryPoint),
    /// More than one vertex entry was selected for one shared vertex interface.
    MultipleVertexEntries,
    /// Two contributors supplied the same identity.
    DuplicateSource(String),
    /// WGSL parsing failed, optionally mapped to its source contributor.
    Parse {
        /// Source contributor if the parser supplied a usable span.
        source: Option<SourceUnitId>,
        /// One-based source line, when available.
        line: Option<u32>,
        /// One-based source column, when available.
        column: Option<u32>,
        /// Validator-provided diagnostic text.
        message: String,
    },
    /// WGSL validation failed, optionally mapped to its source contributor.
    Validate {
        /// Source contributor if the validator supplied a usable span.
        source: Option<SourceUnitId>,
        /// One-based source line, when available.
        line: Option<u32>,
        /// One-based source column, when available.
        column: Option<u32>,
        /// Validator-provided diagnostic text.
        message: String,
    },
    /// A requested entry is absent or has a different stage.
    MissingEntry(EntryPoint),
    /// Declared and reflected resource bindings disagree.
    BindingMismatch {
        /// Bind-group index.
        group: u32,
        /// Binding index.
        binding: u32,
    },
    /// A consumer geometry mapping disagrees with reflected input.
    VertexMismatch {
        /// Supplied semantic, or empty when a reflected location is missing.
        semantic: String,
        /// Input location involved in the mismatch.
        location: u32,
    },
    /// A reachable binding class is not modeled by this implementation.
    UnsupportedBinding,
    /// A vertex or callable value type is not modeled by this implementation.
    UnsupportedVertexType,
    /// Declared callable metadata differs from actual WGSL.
    CallableMismatch {
        /// Function name involved in the mismatch.
        name: String,
    },
}
impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ShaderError {}
#[derive(Clone, Debug)]
/// Validated portable source and its logical interface, not a native binary.
pub struct ShaderArtifact {
    source: String,
    source_map: Vec<SourceSpan>,
    interface: ShaderInterface,
    requirements: PipelineRequirements,
    key: ShaderArtifactKey,
}
#[derive(Clone, Debug, Eq, PartialEq)]
/// Byte range occupied by one contributor in the composed WGSL source.
pub struct SourceSpan {
    /// Contributor identity.
    pub source: SourceUnitId,
    /// Half-open byte range in the composed source.
    pub bytes: std::ops::Range<usize>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// Collision-safe logical identity scoped to this implementation's request.
pub struct ShaderArtifactKey(ShaderCompositionRequest);
impl ShaderArtifact {
    /// Returns the composed WGSL source.
    pub fn source(&self) -> &str {
        &self.source
    }
    /// Returns contributor ranges within the composed source.
    pub fn source_map(&self) -> &[SourceSpan] {
        &self.source_map
    }
    /// Returns the reflected logical program interface.
    pub fn interface(&self) -> &ShaderInterface {
        &self.interface
    }
    /// Returns the first selected entry for a stage, if present.
    pub fn entry_point(&self, s: ShaderStage) -> Option<&EntryPoint> {
        self.interface.entries.iter().find(|x| x.stage == s)
    }
    /// Returns consumer-supplied requirements for device admission.
    pub fn requirements(&self) -> &PipelineRequirements {
        &self.requirements
    }
    /// Returns the complete logical request identity.
    pub fn cache_key(&self) -> &ShaderArtifactKey {
        &self.key
    }
}
#[derive(Default)]
/// Reuses validated logical artifacts for equal complete requests.
pub struct ShaderComposer {
    cache: HashMap<ShaderCompositionRequest, Arc<ShaderArtifact>>,
}

#[cfg(test)]
mod tests;

impl ShaderComposer {
    /// Creates an empty logical artifact cache.
    pub fn new() -> Self {
        Self::default()
    }
    /// Composes, validates and reflects a complete portable shader program.
    pub fn compose(
        &mut self,
        r: &ShaderCompositionRequest,
    ) -> Result<Arc<ShaderArtifact>, ShaderError> {
        if let Some(x) = self.cache.get(r) {
            return Ok(x.clone());
        }
        let x = Arc::new(build(r, ShaderArtifactKey(r.clone()))?);
        self.cache.insert(r.clone(), x.clone());
        Ok(x)
    }
}

mod composition;
use composition::build;
