//! Material composition and instance validation failures.

use super::{
    BindingLocation, MaterialAssetRef, MaterialResourceKind, MaterialType, SurfaceOutput, fmt,
};

/// Material composition and instance validation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialError {
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
    /// The selected root needs arguments that the minimal evaluation ABI cannot supply.
    RootInputsUnsupported {
        /// Root fragment name.
        fragment: String,
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
