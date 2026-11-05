//! Typed material composition and generic WGSL contributions.
//!
//! Material owns authoring validation, static variants, instance values, and a
//! private normalized representation. It emits only generic WGSL source and
//! interface declarations; a Shader consumer supplies pass templates and entry
//! points before composing a complete program.

#![deny(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use fluxel_assets::{AssetId, AssetKind, AssetSnapshot, ContentGeneration};
use fluxel_shader::{ScalarType, ValueType};

#[path = "code_node.rs"]
mod code_node;
#[path = "compiler.rs"]
mod compiler;
#[path = "contribution.rs"]
mod contribution;
#[path = "errors.rs"]
mod errors;
#[path = "names.rs"]
mod names;
#[path = "surface.rs"]
mod surface;
#[path = "validation.rs"]
mod validation;
pub use code_node::MaterialCodeNode;
pub use contribution::MaterialShaderContribution;
pub use errors::MaterialError;
use validation::NormalizedMaterialIr;

use names::wgsl_top_level_names;

/// Marker for material assets stored through `fluxel-assets`.
pub struct MaterialAsset;

impl AssetKind for MaterialAsset {}

/// Opaque logical identity for one material asset.
pub type MaterialAssetId = AssetId<MaterialAsset>;

/// The durable identity of one material content generation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MaterialAssetRef {
    id: MaterialAssetId,
    generation: ContentGeneration,
}

impl MaterialAssetRef {
    /// Captures identity and content generation from an immutable asset snapshot.
    #[must_use]
    pub fn from_snapshot<V>(snapshot: &AssetSnapshot<MaterialAsset, V>) -> Self {
        Self {
            id: snapshot.id(),
            generation: snapshot.generation(),
        }
    }

    /// Returns the logical asset identity.
    #[must_use]
    pub const fn id(self) -> MaterialAssetId {
        self.id
    }

    /// Returns the immutable content generation represented by this reference.
    #[must_use]
    pub const fn generation(self) -> ContentGeneration {
        self.generation
    }
}

/// Types exposed by the minimal material function boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaterialType {
    /// A scalar `f32` value.
    F32,
    /// A `vec2<f32>` value.
    Vec2,
    /// A `vec3<f32>` value.
    Vec3,
    /// A `vec4<f32>` value.
    Vec4,
}

impl MaterialType {
    /// Returns the WGSL spelling for this type.
    #[must_use]
    pub const fn wgsl(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
        }
    }

    fn shader_type(self) -> ValueType {
        match self {
            Self::F32 => ValueType::Scalar(ScalarType::F32),
            Self::Vec2 => ValueType::Vec2(ScalarType::F32),
            Self::Vec3 => ValueType::Vec3(ScalarType::F32),
            Self::Vec4 => ValueType::Vec4(ScalarType::F32),
        }
    }
}

/// A typed material parameter supplied per instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialParameter {
    /// Stable parameter name used by authoring and WGSL fragments.
    pub name: String,
    /// Parameter value type.
    pub ty: MaterialType,
    /// Logical resource that receives this parameter's per-instance value.
    pub resource: String,
}

/// A runtime value for a dynamic material parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicValue {
    /// Scalar parameter value.
    F32(f32),
    /// Two-component parameter value.
    Vec2([f32; 2]),
    /// Three-component parameter value.
    Vec3([f32; 3]),
    /// Four-component parameter value.
    Vec4([f32; 4]),
}

impl DynamicValue {
    /// Returns this value's declared material type.
    #[must_use]
    pub const fn ty(&self) -> MaterialType {
        match self {
            Self::F32(_) => MaterialType::F32,
            Self::Vec2(_) => MaterialType::Vec2,
            Self::Vec3(_) => MaterialType::Vec3,
            Self::Vec4(_) => MaterialType::Vec4,
        }
    }
}

/// A bind-group location declared by a material resource.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BindingLocation {
    /// Bind group index.
    pub group: u32,
    /// Binding index within the bind group.
    pub binding: u32,
}

/// Resource kinds supported by the initial material core.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaterialResourceKind {
    /// A uniform `vec4<f32>` binding.
    UniformVec4,
    /// A read-only `texture_2d<f32>` binding.
    Texture2dF32,
}

impl MaterialResourceKind {
    /// Returns the WGSL type spelling for this resource.
    #[must_use]
    pub const fn wgsl(self) -> &'static str {
        match self {
            Self::UniformVec4 => "vec4<f32>",
            Self::Texture2dF32 => "texture_2d<f32>",
        }
    }
}

/// A named material resource and its shader-visible binding location.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialResource {
    /// Stable resource name used by fragments.
    pub name: String,
    /// Generic resource kind.
    pub kind: MaterialResourceKind,
    /// Shader-visible bind-group location.
    pub location: BindingLocation,
}

/// A statically selected material option.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StaticValue {
    /// Boolean static value.
    Bool(bool),
    /// Unsigned integer static value.
    U32(u32),
}

impl StaticValue {
    fn canonical(&self) -> String {
        match self {
            Self::Bool(value) => format!("bool:{value}"),
            Self::U32(value) => format!("u32:{value}"),
        }
    }
}

/// A typed input expected by a material fragment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FragmentInput {
    /// Input name for diagnostics.
    pub name: String,
    /// Input type.
    pub ty: MaterialType,
}

/// An explicit call made by a fragment to another fragment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FragmentCall {
    /// Called fragment name.
    pub fragment: String,
    /// Argument types in call order.
    pub arguments: Vec<MaterialType>,
}

/// Rust-defined material function metadata plus its WGSL implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialFragment {
    /// Stable fragment name.
    pub name: String,
    /// Input metadata used to validate explicit material calls.
    pub inputs: Vec<FragmentInput>,
    /// Fragment result type.
    pub output: MaterialType,
    /// Explicit outgoing calls used for deterministic graph validation.
    pub calls: Vec<FragmentCall>,
    /// WGSL declarations and function implementation for this fragment.
    pub wgsl: String,
}

/// The selected surface output for the minimal material contribution.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SurfaceOutput {
    /// A fragment-color `vec4<f32>` output.
    Color,
    /// A base color plus metallic and perceptual-roughness property callable.
    MetallicRoughness {
        /// Zero-argument `vec2<f32>` callable yielding metallic then roughness.
        properties: String,
    },
}

/// One material's validated, author-facing definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialDefinition {
    fragments: Vec<MaterialFragment>,
    code_nodes: Vec<MaterialCodeNode>,
    root_fragment: String,
    parameters: Vec<MaterialParameter>,
    resources: Vec<MaterialResource>,
    static_options: BTreeMap<String, StaticValue>,
    output: SurfaceOutput,
}

impl MaterialDefinition {
    /// Creates a material definition without compiling it.
    #[must_use]
    pub fn new(
        fragments: Vec<MaterialFragment>,
        root_fragment: impl Into<String>,
        parameters: Vec<MaterialParameter>,
        resources: Vec<MaterialResource>,
        static_options: BTreeMap<String, StaticValue>,
        output: SurfaceOutput,
    ) -> Self {
        Self {
            fragments,
            code_nodes: Vec::new(),
            root_fragment: root_fragment.into(),
            parameters,
            resources,
            static_options,
            output,
        }
    }

    /// Adds one first-class author WGSL node to this material definition.
    #[must_use]
    pub fn with_code_node(mut self, node: MaterialCodeNode) -> Self {
        self.code_nodes.push(node);
        self
    }

    /// Returns the selected surface domain.
    #[must_use]
    pub fn output(&self) -> &SurfaceOutput {
        &self.output
    }
}

/// A stable static-variant identity derived from canonical material input.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MaterialVariantKey {
    canonical: String,
    material: Option<MaterialAssetRef>,
}

impl MaterialVariantKey {
    /// Returns the canonical variant representation for diagnostics and caches.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.canonical
    }
}

/// A generic shader binding declaration emitted by Material.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialBindingDeclaration {
    /// Resource name.
    pub name: String,
    /// Bind-group location.
    pub location: BindingLocation,
    /// Declared resource kind.
    pub kind: MaterialResourceKind,
}

/// The concrete uniform binding that receives one dynamic material parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialParameterBinding {
    /// Dynamic parameter name accepted by [`MaterialInstance`].
    pub parameter: String,
    /// Logical resource name used to resolve the concrete binding owner.
    pub resource: String,
    /// Shader-visible bind-group location of the uniform.
    pub location: BindingLocation,
    /// Type expected from the dynamic parameter value.
    pub ty: MaterialType,
}

/// A compiled static material variant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledMaterialVariant {
    material: Option<MaterialAssetRef>,
    variant_key: MaterialVariantKey,
    contribution: MaterialShaderContribution,
    parameters: Vec<MaterialParameter>,
    resources: Vec<MaterialResource>,
    output: SurfaceOutput,
}

impl CompiledMaterialVariant {
    /// Returns the asset generation this variant represents, when it was compiled from an asset.
    #[must_use]
    pub const fn material(&self) -> Option<MaterialAssetRef> {
        self.material
    }
    /// Returns the variant's static identity.
    #[must_use]
    pub fn key(&self) -> &MaterialVariantKey {
        &self.variant_key
    }

    /// Returns the generic contribution consumed by Shader.
    #[must_use]
    pub fn contribution(&self) -> &MaterialShaderContribution {
        &self.contribution
    }

    /// Returns the selected surface domain for this compiled variant.
    #[must_use]
    pub fn output(&self) -> &SurfaceOutput {
        &self.output
    }
}

/// Per-instance dynamic material data over a compiled static variant.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialInstance {
    material: MaterialAssetRef,
    variant: MaterialVariantKey,
    parameters: BTreeMap<String, DynamicValue>,
    resources: BTreeMap<String, String>,
}

impl MaterialInstance {
    /// Validates dynamic values against a compiled variant and creates an instance.
    pub fn new(
        material: MaterialAssetRef,
        variant: &CompiledMaterialVariant,
        parameters: BTreeMap<String, DynamicValue>,
        resources: BTreeMap<String, String>,
    ) -> Result<Self, MaterialError> {
        if let Some(expected) = variant.material {
            if expected != material {
                return Err(MaterialError::VariantMaterialMismatch {
                    expected,
                    actual: material,
                });
            }
        }
        let expected_parameters = variant
            .parameters
            .iter()
            .map(|parameter| (parameter.name.as_str(), parameter.ty))
            .collect::<BTreeMap<_, _>>();
        for (name, value) in &parameters {
            let expected = expected_parameters
                .get(name.as_str())
                .ok_or_else(|| MaterialError::UnknownParameter { name: name.clone() })?;
            if *expected != value.ty() {
                return Err(MaterialError::ParameterTypeMismatch {
                    name: name.clone(),
                    expected: *expected,
                    actual: value.ty(),
                });
            }
        }
        for parameter in &variant.parameters {
            if !parameters.contains_key(&parameter.name) {
                return Err(MaterialError::MissingParameter {
                    name: parameter.name.clone(),
                });
            }
        }
        let expected_resources = variant
            .resources
            .iter()
            .map(|resource| resource.name.as_str())
            .collect::<BTreeSet<_>>();
        for name in resources.keys() {
            if !expected_resources.contains(name.as_str()) {
                return Err(MaterialError::UnknownResource { name: name.clone() });
            }
        }
        for resource in &variant.resources {
            if !resources.contains_key(&resource.name) {
                return Err(MaterialError::MissingResource {
                    name: resource.name.clone(),
                });
            }
        }
        Ok(Self {
            material,
            variant: variant.variant_key.clone(),
            parameters,
            resources,
        })
    }

    /// Returns the asset identity and content generation this instance represents.
    #[must_use]
    pub const fn material(&self) -> MaterialAssetRef {
        self.material
    }

    /// Returns the selected static variant key.
    #[must_use]
    pub fn variant(&self) -> &MaterialVariantKey {
        &self.variant
    }

    /// Returns the validated dynamic parameter values for concrete uniform upload.
    #[must_use]
    pub fn parameters(&self) -> &BTreeMap<String, DynamicValue> {
        &self.parameters
    }

    /// Returns one validated dynamic parameter by name.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&DynamicValue> {
        self.parameters.get(name)
    }

    /// Returns the validated logical resource values for concrete binding resolution.
    #[must_use]
    pub fn resources(&self) -> &BTreeMap<String, String> {
        &self.resources
    }

    /// Returns one validated logical resource value by name.
    #[must_use]
    pub fn resource(&self, name: &str) -> Option<&str> {
        self.resources.get(name).map(String::as_str)
    }
}

/// Compiles a validated authoring definition into one static material variant.
pub struct MaterialCompiler;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
