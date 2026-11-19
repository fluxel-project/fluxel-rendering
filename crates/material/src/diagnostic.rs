//! Material composition and instance validation failures.

use std::fmt;

use crate::{
    BindingLocation, MaterialAssetRef, MaterialNodeId, MaterialNodeOutput, MaterialResourceKind,
    MaterialType, SurfaceOutput,
};

/// Material composition and instance validation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialError {
    /// A callable declared an input-port name more than once.
    DuplicateCallableInput {
        /// Callable name.
        callable: String,
        /// Repeated input-port name.
        input: String,
    },
    /// Author source or metadata referenced a generated node wrapper.
    GeneratedNodeReference {
        /// Author fragment or Code Node identity.
        source: String,
        /// Generated node-wrapper name.
        generated: String,
    },
    /// A node identity occurred more than once.
    DuplicateNode {
        /// Repeated node identity.
        node: MaterialNodeId,
    },
    /// A graph named a node that it did not contain.
    MissingNode {
        /// Referenced node identity.
        node: MaterialNodeId,
    },
    /// A node connected the same input more than once.
    DuplicateNodeInput {
        /// Receiving node identity.
        node: MaterialNodeId,
        /// Repeated input-port name.
        input: String,
    },
    /// A node named an input absent from its callable.
    UnknownNodeInput {
        /// Receiving node identity.
        node: MaterialNodeId,
        /// Unknown input-port name.
        input: String,
    },
    /// An upstream node cannot expose this port from its declared result type.
    NodeOutputUnavailable {
        /// Upstream node identity.
        node: MaterialNodeId,
        /// Requested output port.
        output: MaterialNodeOutput,
        /// Declared return type of the upstream callable.
        actual: MaterialType,
    },
    /// A callable input had no source node.
    MissingNodeInput {
        /// Receiving node identity.
        node: MaterialNodeId,
        /// Required input-port name.
        input: String,
    },
    /// An upstream output did not match the target input type.
    NodeInputTypeMismatch {
        /// Receiving node identity.
        node: MaterialNodeId,
        /// Input-port name.
        input: String,
        /// Callable input type.
        expected: MaterialType,
        /// Source node result type.
        actual: MaterialType,
    },
    /// A node requested a context value unavailable to this surface evaluator.
    NodeContextUnavailable {
        /// Receiving node identity.
        node: MaterialNodeId,
        /// Input-port name.
        input: String,
    },
    /// The surface has no vertex UV0 evaluation entry.
    VertexUv0Unsupported {
        /// Selected surface output.
        output: SurfaceOutput,
    },
    /// A vertex UV0 callable does not take and return a UV vector.
    VertexUv0Signature {
        /// Selected callable.
        callable: String,
    },
    /// A vertex UV0 graph output does not produce a UV vector.
    VertexUv0GraphOutput {
        /// Node selected as the vertex output.
        node: MaterialNodeId,
        /// Declared result type of that node.
        actual: MaterialType,
    },
    /// Material vertex varyings require a selected typed vertex graph.
    VertexVaryingsRequireGraph,
    /// More than one material vertex varying used the same identifier.
    DuplicateVertexVarying {
        /// Repeated varying identifier.
        id: u32,
    },
    /// A material vertex varying selected something other than a node result.
    InvalidVertexVaryingSource {
        /// Varying identifier with the invalid source.
        id: u32,
    },
    /// A reachable node graph cycle cannot lower to WGSL wrappers.
    NodeCycle {
        /// Node encountered a second time on the active DFS path.
        node: MaterialNodeId,
    },
    /// An asset-bound variant was used with another asset or content generation.
    VariantMaterialMismatch {
        /// Asset generation recorded by the variant.
        expected: MaterialAssetRef,
        /// Asset generation supplied for the instance.
        actual: MaterialAssetRef,
    },
    /// A fragment name occurred more than once.
    DuplicateFragment {
        /// Duplicate name.
        name: String,
    },
    /// A Code Node source unit had no diagnostic identity.
    EmptyCodeNodeSourceId {
        /// Code Node whose source identity was empty.
        node: String,
    },
    /// Two Code Nodes claimed the same diagnostic source identity.
    DuplicateCodeNodeSourceId {
        /// Duplicate source-unit identity.
        source_id: String,
    },
    /// A parameter name occurred more than once.
    DuplicateParameter {
        /// Duplicate name.
        name: String,
    },
    /// A resource name occurred more than once.
    DuplicateResource {
        /// Duplicate name.
        name: String,
    },
    /// A Code Node and definition-level declaration disagreed about one resource.
    ConflictingResourceDeclaration {
        /// Resource name with incompatible declarations.
        name: String,
    },
    /// A parameter named a resource that the material did not declare.
    ParameterResourceMissing {
        /// Dynamic parameter name.
        parameter: String,
        /// Missing resource name.
        resource: String,
    },
    /// A parameter targeted a resource that cannot receive its value.
    ParameterResourceTypeMismatch {
        /// Dynamic parameter name.
        parameter: String,
        /// Declared parameter type.
        parameter_type: MaterialType,
        /// Declared resource kind.
        resource_kind: MaterialResourceKind,
    },
    /// More than one parameter attempted to write the same uniform resource.
    ParameterResourceConflict {
        /// Conflicting resource name.
        resource: String,
    },
    /// A parameter WGSL name collided with another top-level material declaration.
    ParameterNameConflict {
        /// Dynamic parameter name.
        parameter: String,
    },
    /// Two emitted top-level WGSL declarations used the same name.
    TopLevelNameConflict {
        /// Conflicting WGSL name.
        name: String,
    },
    /// The top-level WGSL declaration scan could not complete safely.
    TopLevelWgslUnparseable,
    /// Two resources claimed the same shader-visible location.
    BindingConflict {
        /// Conflicting location.
        location: BindingLocation,
    },
    /// More than one transfer-encoding requirement named the same resource.
    DuplicateTextureEncodingRequirement {
        /// Texture resource name.
        resource: String,
    },
    /// A transfer-encoding requirement named no material resource.
    TextureEncodingResourceMissing {
        /// Texture resource name.
        resource: String,
    },
    /// A transfer-encoding requirement named a non-texture resource.
    TextureEncodingResourceTypeMismatch {
        /// Texture resource name.
        resource: String,
        /// Declared resource kind.
        actual: MaterialResourceKind,
    },
    /// The selected root fragment was not defined.
    MissingFragment {
        /// Requested fragment name.
        name: String,
    },
    /// A declared fragment call did not target a defined fragment.
    InvalidCall {
        /// Calling fragment.
        caller: String,
        /// Missing callee.
        callee: String,
    },
    /// A declared call's argument types differ from its callee inputs.
    CallTypeMismatch {
        /// Calling fragment.
        caller: String,
        /// Called fragment.
        callee: String,
    },
    /// The fragment call graph contains a cycle.
    CallCycle {
        /// One fragment in the cycle.
        fragment: String,
    },
    /// The selected surface output is not provided by the root fragment.
    UnsupportedOutput {
        /// Selected output.
        output: SurfaceOutput,
    },
    /// The selected root needs arguments that its evaluation ABI cannot supply.
    RootInputsUnsupported {
        /// Root fragment name.
        fragment: String,
    },
    /// A graph metallic-roughness surface requires a selected node graph.
    PbrGraphRequired,
    /// A graph PBR surface selected both a tangent-space normal callable and
    /// a world-space graph normal source.
    PbrGraphNormalConflict,
    /// A graph metallic-roughness property must select a node or node output.
    PbrGraphPropertySource {
        /// Property being selected.
        property: &'static str,
    },
    /// A graph PBR property did not resolve to its required type: `F32` for
    /// metallic/roughness/occlusion or `Vec3` for `world_normal`.
    PbrGraphPropertyType {
        /// Property being selected.
        property: &'static str,
        /// Resolved source type.
        actual: MaterialType,
    },
    /// The metallic-roughness surface domain named no property callable.
    MissingSurfaceProperties {
        /// Requested property callable name.
        properties: String,
    },
    /// The metallic-roughness property callable returned the wrong type.
    SurfacePropertiesTypeMismatch {
        /// Property callable name.
        properties: String,
        /// Declared callable result type.
        actual: MaterialType,
    },
    /// The metallic-roughness property callable needs arguments its ABI cannot supply.
    SurfacePropertiesInputsUnsupported {
        /// Property callable name.
        properties: String,
    },
    /// The tangent-space normal surface domain named no normal callable.
    MissingSurfaceNormal {
        /// Requested normal callable name.
        normal: String,
    },
    /// The tangent-space normal callable returned the wrong type.
    SurfaceNormalTypeMismatch {
        /// Normal callable name.
        normal: String,
        /// Declared callable result type.
        actual: MaterialType,
    },
    /// The tangent-space normal callable needs arguments its ABI cannot supply.
    SurfaceNormalInputsUnsupported {
        /// Normal callable name.
        normal: String,
    },
    /// Optional surface channels were selected for an incompatible surface output.
    SurfaceExtensionsUnsupported {
        /// Selected output.
        output: SurfaceOutput,
    },
    /// An optional surface channel named no callable.
    MissingSurfaceExtension {
        /// Requested callable name.
        name: String,
    },
    /// An optional surface channel returned the wrong type.
    SurfaceExtensionTypeMismatch {
        /// Requested callable name.
        name: String,
        /// Declared callable result type.
        actual: MaterialType,
    },
    /// An optional surface channel needs arguments its ABI cannot supply.
    SurfaceExtensionInputsUnsupported {
        /// Requested callable name.
        name: String,
    },
    /// A custom callable and the standard emissive configuration both tried to
    /// generate the emissive evaluator.
    EmissiveConfigurationConflict,
    /// An emissive control named no declared material parameter.
    EmissiveParameterMissing {
        /// Emissive control field.
        control: &'static str,
        /// Missing material parameter name.
        parameter: String,
    },
    /// An emissive control named a parameter with an incompatible type.
    EmissiveParameterTypeMismatch {
        /// Emissive control field.
        control: &'static str,
        /// Material parameter name.
        parameter: String,
        /// Required parameter type.
        expected: MaterialType,
        /// Declared parameter type.
        actual: MaterialType,
    },
    /// The emissive configuration named no declared resource.
    EmissiveResourceMissing {
        /// Emissive configuration field.
        field: &'static str,
        /// Missing resource name.
        resource: String,
    },
    /// The emissive configuration named a resource with an incompatible kind.
    EmissiveResourceTypeMismatch {
        /// Emissive configuration field.
        field: &'static str,
        /// Resource name.
        resource: String,
        /// Required resource kind.
        expected: MaterialResourceKind,
        /// Declared resource kind.
        actual: MaterialResourceKind,
    },
    /// Explicit texture encoding metadata contradicts the emissive texture.
    EmissiveTextureEncodingConflict {
        /// Texture resource name.
        resource: String,
    },
    /// PBR factors require UV0 and packed occlusion also requires the normal-surface interface.
    PbrSurfaceFactorsUnsupported {
        /// Selected output.
        output: SurfaceOutput,
    },
    /// A standard PBR factor named no declared material parameter.
    PbrSurfaceFactorParameterMissing {
        /// Configuration field.
        field: &'static str,
        /// Missing parameter name.
        parameter: String,
    },
    /// A standard PBR factor named a parameter with an incompatible type.
    PbrSurfaceFactorParameterTypeMismatch {
        /// Configuration field.
        field: &'static str,
        /// Parameter name.
        parameter: String,
        /// Required type.
        expected: MaterialType,
        /// Actual type.
        actual: MaterialType,
    },
    /// A standard PBR texture named no declared material resource.
    PbrSurfaceFactorResourceMissing {
        /// Configuration field.
        field: &'static str,
        /// Missing resource name.
        resource: String,
    },
    /// A standard PBR texture named a resource with an incompatible kind.
    PbrSurfaceFactorResourceTypeMismatch {
        /// Configuration field.
        field: &'static str,
        /// Resource name.
        resource: String,
        /// Required kind.
        expected: MaterialResourceKind,
        /// Actual kind.
        actual: MaterialResourceKind,
    },
    /// An occlusion strength was selected without a packed occlusion channel.
    PbrOcclusionStrengthWithoutPackedChannel,
    /// Explicit texture encoding metadata contradicts a standard PBR texture role.
    PbrSurfaceFactorTextureEncodingConflict {
        /// Texture resource name.
        resource: String,
    },
    /// A surface alpha cutoff must be finite before it can become WGSL source.
    SurfaceAlphaCutoffNonFinite,
    /// Transmission controls require a UV0 normal PBR surface.
    TransmissionExtensionsUnsupported {
        /// Selected output.
        output: SurfaceOutput,
    },
    /// A requested transmission callable was not declared.
    MissingTransmissionExtension {
        /// Requested callable name.
        name: String,
    },
    /// A requested transmission callable returned the wrong type.
    TransmissionExtensionTypeMismatch {
        /// Requested callable name.
        name: String,
        /// Declared callable result type.
        actual: MaterialType,
    },
    /// A requested transmission callable needs arguments unavailable to UV0 evaluation.
    TransmissionExtensionInputsUnsupported {
        /// Requested callable name.
        name: String,
    },
    /// A static transmission value would make refraction or Beer-Lambert invalid.
    InvalidTransmissionParameter {
        /// Parameter rejected before shader generation.
        parameter: &'static str,
    },
    /// A material-local refraction environment requires transmission controls.
    RefractionEnvironmentTransmissionRequired,
    /// A refraction environment named no declared material resource.
    RefractionEnvironmentResourceMissing {
        /// Configuration field.
        field: &'static str,
        /// Missing resource name.
        resource: String,
    },
    /// A refraction environment resource has an incompatible kind.
    RefractionEnvironmentResourceTypeMismatch {
        /// Configuration field.
        field: &'static str,
        /// Resource name.
        resource: String,
        /// Required kind.
        expected: MaterialResourceKind,
        /// Declared kind.
        actual: MaterialResourceKind,
    },
    /// A refraction environment resource must be an ordinary material binding.
    RefractionEnvironmentResourceGroup {
        /// Configuration field.
        field: &'static str,
        /// Resource name.
        resource: String,
        /// Declared bind group.
        group: u32,
    },
    /// A static refraction environment value was non-finite.
    InvalidRefractionEnvironmentParameter {
        /// Parameter rejected before shader generation.
        parameter: &'static str,
    },
    /// Explicit texture encoding metadata contradicts a refraction environment.
    RefractionEnvironmentTextureEncodingConflict {
        /// Texture resource name.
        resource: String,
    },
    /// A dynamic parameter was not declared by the variant.
    UnknownParameter {
        /// Parameter name.
        name: String,
    },
    /// A required dynamic parameter was absent.
    MissingParameter {
        /// Parameter name.
        name: String,
    },
    /// A supplied value did not match the declared type.
    ParameterTypeMismatch {
        /// Parameter name.
        name: String,
        /// Declared type.
        expected: MaterialType,
        /// Supplied type.
        actual: MaterialType,
    },
    /// A dynamic resource was not declared by the variant.
    UnknownResource {
        /// Resource name.
        name: String,
    },
    /// A required dynamic resource was absent.
    MissingResource {
        /// Resource name.
        name: String,
    },
}

impl fmt::Display for MaterialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MaterialError {}
