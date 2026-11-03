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
use fluxel_shader::{
    BindingDeclaration, BindingKind, BindingShape, CallableDeclaration, ScalarType, SourceUnit,
    ValueType,
};
use sha2::{Digest, Sha256};

#[path = "compiler.rs"]
mod compiler;
#[path = "names.rs"]
mod names;

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
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SurfaceOutput {
    /// A fragment-color `vec4<f32>` output.
    Color,
}

/// One material's validated, author-facing definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialDefinition {
    fragments: Vec<MaterialFragment>,
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
            root_fragment: root_fragment.into(),
            parameters,
            resources,
            static_options,
            output,
        }
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

/// Generic source and interface data for a Shader composition request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialShaderContribution {
    /// Short deterministic source-unit identifier for diagnostics and source maps.
    ///
    /// This is not the material cache key: the complete canonical variant key and
    /// complete shader composition request remain the collision-safe identities.
    pub source_id: String,
    /// WGSL declarations containing `evaluate_material`.
    pub source: String,
    /// Resource declarations Shader must compare against reflection.
    pub bindings: Vec<MaterialBindingDeclaration>,
    /// Dynamic parameter-to-uniform associations consumed by the renderer.
    pub parameter_bindings: Vec<MaterialParameterBinding>,
    /// Function signatures that Shader must validate against actual WGSL.
    pub callables: Vec<CallableDeclaration>,
    /// Canonical static choices participating in composition identity.
    pub static_options: BTreeMap<String, StaticValue>,
}

impl MaterialShaderContribution {
    /// Returns Shader-owned callable metadata for WGSL signature validation.
    #[must_use]
    pub fn callable_declarations(&self) -> Vec<CallableDeclaration> {
        self.callables.clone()
    }
    /// Converts this contribution into the Shader-owned source unit.
    #[must_use]
    pub fn source_unit(&self) -> SourceUnit {
        SourceUnit::new(self.source_id.clone(), self.source.clone())
    }

    /// Converts resource metadata into Shader-owned binding declarations.
    #[must_use]
    pub fn binding_declarations(&self) -> Vec<BindingDeclaration> {
        self.bindings
            .iter()
            .map(|binding| {
                BindingDeclaration::new(
                    binding.location.group,
                    binding.location.binding,
                    match binding.kind {
                        MaterialResourceKind::UniformVec4 => BindingKind::Uniform,
                        MaterialResourceKind::Texture2dF32 => BindingKind::Texture,
                    },
                )
                .with_shape(match binding.kind {
                    MaterialResourceKind::UniformVec4 => {
                        BindingShape::Uniform(ValueType::Vec4(ScalarType::F32))
                    }
                    MaterialResourceKind::Texture2dF32 => BindingShape::Texture2dF32,
                })
            })
            .collect()
    }

    /// Converts static material choices into Shader composition option values.
    #[must_use]
    pub fn shader_static_options(&self) -> BTreeMap<String, String> {
        self.static_options
            .iter()
            .map(|(name, value)| (name.clone(), value.canonical()))
            .collect()
    }
}

/// A compiled static material variant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledMaterialVariant {
    material: Option<MaterialAssetRef>,
    variant_key: MaterialVariantKey,
    contribution: MaterialShaderContribution,
    parameters: Vec<MaterialParameter>,
    resources: Vec<MaterialResource>,
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

/// Compiles a validated authoring definition into one static material variant.
pub struct MaterialCompiler;

#[derive(Clone, Debug)]
struct NormalizedMaterialIr {
    fragments: Vec<MaterialFragment>,
    root_fragment: String,
    parameters: Vec<MaterialParameter>,
    resources: Vec<MaterialResource>,
    static_options: BTreeMap<String, StaticValue>,
}

impl NormalizedMaterialIr {
    fn from_definition(definition: &MaterialDefinition) -> Result<Self, MaterialError> {
        let mut fragments = definition.fragments.clone();
        fragments.sort_by(|left, right| left.name.cmp(&right.name));
        let mut names = BTreeSet::new();
        for fragment in &fragments {
            if !names.insert(fragment.name.as_str()) {
                return Err(MaterialError::DuplicateFragment {
                    name: fragment.name.clone(),
                });
            }
        }
        let fragment_map = fragments
            .iter()
            .map(|fragment| (fragment.name.as_str(), fragment))
            .collect::<BTreeMap<_, _>>();
        let root = fragment_map
            .get(definition.root_fragment.as_str())
            .ok_or_else(|| MaterialError::MissingFragment {
                name: definition.root_fragment.clone(),
            })?;
        if definition.output == SurfaceOutput::Color && root.output != MaterialType::Vec4 {
            return Err(MaterialError::UnsupportedOutput {
                output: definition.output,
            });
        }
        if !root.inputs.is_empty() {
            return Err(MaterialError::RootInputsUnsupported {
                fragment: root.name.clone(),
            });
        }
        let mut parameter_names = BTreeSet::new();
        for parameter in &definition.parameters {
            if !parameter_names.insert(parameter.name.as_str()) {
                return Err(MaterialError::DuplicateParameter {
                    name: parameter.name.clone(),
                });
            }
        }
        let mut resource_names = BTreeSet::new();
        let mut locations = BTreeSet::new();
        for resource in &definition.resources {
            if !resource_names.insert(resource.name.as_str()) {
                return Err(MaterialError::DuplicateResource {
                    name: resource.name.clone(),
                });
            }
            if !locations.insert(resource.location) {
                return Err(MaterialError::BindingConflict {
                    location: resource.location,
                });
            }
        }
        let resources = definition
            .resources
            .iter()
            .map(|resource| (resource.name.as_str(), resource))
            .collect::<BTreeMap<_, _>>();
        let mut parameter_resources = BTreeSet::new();
        for parameter in &definition.parameters {
            let conflicts_with_resource = resources
                .get(parameter.name.as_str())
                .is_some_and(|resource| resource.name != parameter.resource);
            if conflicts_with_resource
                || fragment_map.contains_key(parameter.name.as_str())
                || definition
                    .static_options
                    .contains_key(parameter.name.as_str())
                || parameter.name == "evaluate_material"
            {
                return Err(MaterialError::ParameterNameConflict {
                    parameter: parameter.name.clone(),
                });
            }
            let resource = resources.get(parameter.resource.as_str()).ok_or_else(|| {
                MaterialError::ParameterResourceMissing {
                    parameter: parameter.name.clone(),
                    resource: parameter.resource.clone(),
                }
            })?;
            if !parameter_resources.insert(resource.name.as_str()) {
                return Err(MaterialError::ParameterResourceConflict {
                    resource: resource.name.clone(),
                });
            }
            if parameter.ty != MaterialType::Vec4
                || resource.kind != MaterialResourceKind::UniformVec4
            {
                return Err(MaterialError::ParameterResourceTypeMismatch {
                    parameter: parameter.name.clone(),
                    parameter_type: parameter.ty,
                    resource_kind: resource.kind,
                });
            }
        }
        validate_top_level_names(
            &fragments,
            &definition.resources,
            &definition.parameters,
            &definition.static_options,
        )?;
        for fragment in &fragments {
            for call in &fragment.calls {
                let callee = fragment_map.get(call.fragment.as_str()).ok_or_else(|| {
                    MaterialError::InvalidCall {
                        caller: fragment.name.clone(),
                        callee: call.fragment.clone(),
                    }
                })?;
                if callee
                    .inputs
                    .iter()
                    .map(|input| input.ty)
                    .ne(call.arguments.iter().copied())
                {
                    return Err(MaterialError::CallTypeMismatch {
                        caller: fragment.name.clone(),
                        callee: call.fragment.clone(),
                    });
                }
            }
        }
        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        for fragment in &fragments {
            visit(&fragment.name, &fragment_map, &mut visiting, &mut visited)?;
        }
        Ok(Self {
            fragments,
            root_fragment: definition.root_fragment.clone(),
            parameters: sorted_by_name(definition.parameters.clone(), |value| &value.name),
            resources: sorted_by_name(definition.resources.clone(), |value| &value.name),
            static_options: definition.static_options.clone(),
        })
    }

    fn variant_key(&self) -> MaterialVariantKey {
        let mut canonical = String::new();
        canonical_piece(&mut canonical, "root", &self.root_fragment);
        for fragment in &self.fragments {
            canonical_piece(&mut canonical, "fragment", &fragment.name);
            canonical_piece(&mut canonical, "output", fragment.output.wgsl());
            canonical_piece(&mut canonical, "wgsl", &fragment.wgsl);
            for input in &fragment.inputs {
                canonical_piece(&mut canonical, "input-name", &input.name);
                canonical_piece(&mut canonical, "input-type", input.ty.wgsl());
            }
            for call in &fragment.calls {
                canonical_piece(&mut canonical, "call", &call.fragment);
                for argument in &call.arguments {
                    canonical_piece(&mut canonical, "argument", argument.wgsl());
                }
            }
        }
        for parameter in &self.parameters {
            canonical_piece(&mut canonical, "parameter", &parameter.name);
            canonical_piece(&mut canonical, "parameter-type", parameter.ty.wgsl());
            canonical_piece(&mut canonical, "parameter-resource", &parameter.resource);
        }
        for resource in &self.resources {
            canonical_piece(&mut canonical, "resource", &resource.name);
            canonical_piece(&mut canonical, "resource-kind", resource.kind.wgsl());
            canonical_piece(
                &mut canonical,
                "resource-group",
                &resource.location.group.to_string(),
            );
            canonical_piece(
                &mut canonical,
                "resource-binding",
                &resource.location.binding.to_string(),
            );
        }
        for (name, value) in &self.static_options {
            canonical_piece(&mut canonical, "static-name", name);
            canonical_piece(&mut canonical, "static-value", &value.canonical());
        }
        MaterialVariantKey {
            canonical,
            material: None,
        }
    }

    fn contribution(&self) -> MaterialShaderContribution {
        let mut source = String::from("// Generated by fluxel-material.\n");
        for (name, value) in &self.static_options {
            match value {
                StaticValue::Bool(value) => {
                    source.push_str(&format!("const {name}: bool = {value};\n"))
                }
                StaticValue::U32(value) => {
                    source.push_str(&format!("const {name}: u32 = {value}u;\n"))
                }
            }
        }
        for resource in &self.resources {
            let shader_name = self
                .parameters
                .iter()
                .find(|parameter| parameter.resource == resource.name)
                .map_or(resource.name.as_str(), |parameter| parameter.name.as_str());
            let prefix = match resource.kind {
                MaterialResourceKind::UniformVec4 => "var<uniform>",
                MaterialResourceKind::Texture2dF32 => "var",
            };
            source.push_str(&format!(
                "@group({}) @binding({}) {prefix} {}: {};\n",
                resource.location.group,
                resource.location.binding,
                shader_name,
                resource.kind.wgsl()
            ));
        }
        for fragment in &self.fragments {
            source.push_str(&fragment.wgsl);
            source.push('\n');
        }
        source.push_str(&format!(
            "fn evaluate_material() -> vec4<f32> {{\n    return {}();\n}}\n",
            self.root_fragment
        ));
        let variant_key = self.variant_key();
        MaterialShaderContribution {
            source_id: source_unit_id(variant_key.as_str()),
            source,
            bindings: self
                .resources
                .iter()
                .map(|resource| MaterialBindingDeclaration {
                    name: resource.name.clone(),
                    location: resource.location,
                    kind: resource.kind,
                })
                .collect(),
            parameter_bindings: self
                .parameters
                .iter()
                .map(|parameter| {
                    let resource = self
                        .resources
                        .iter()
                        .find(|resource| resource.name == parameter.resource)
                        .expect("validated parameter resource");
                    MaterialParameterBinding {
                        parameter: parameter.name.clone(),
                        resource: resource.name.clone(),
                        location: resource.location,
                        ty: parameter.ty,
                    }
                })
                .collect(),
            callables: self
                .fragments
                .iter()
                .map(|fragment| CallableDeclaration {
                    name: fragment.name.clone(),
                    arguments: fragment
                        .inputs
                        .iter()
                        .map(|input| input.ty.shader_type())
                        .collect(),
                    result: Some(fragment.output.shader_type()),
                })
                .collect(),
            static_options: self.static_options.clone(),
        }
    }
}

fn validate_top_level_names(
    fragments: &[MaterialFragment],
    resources: &[MaterialResource],
    parameters: &[MaterialParameter],
    static_options: &BTreeMap<String, StaticValue>,
) -> Result<(), MaterialError> {
    let mut names = BTreeSet::from([String::from("evaluate_material")]);
    let mut insert = |name: &str| {
        if names.insert(name.to_owned()) {
            Ok(())
        } else {
            Err(MaterialError::TopLevelNameConflict {
                name: name.to_owned(),
            })
        }
    };
    for name in static_options.keys() {
        insert(name)?;
    }
    for resource in resources {
        let shader_name = parameters
            .iter()
            .find(|parameter| parameter.resource == resource.name)
            .map_or(resource.name.as_str(), |parameter| parameter.name.as_str());
        insert(shader_name)?;
    }
    for fragment in fragments {
        for name in wgsl_top_level_names(&fragment.wgsl)? {
            insert(&name)?;
        }
    }
    Ok(())
}

fn source_unit_id(canonical: &str) -> String {
    format!("material:{:x}", Sha256::digest(canonical.as_bytes()))
}

fn visit(
    name: &str,
    fragments: &BTreeMap<&str, &MaterialFragment>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<(), MaterialError> {
    if visited.contains(name) {
        return Ok(());
    }
    if !visiting.insert(name.to_owned()) {
        return Err(MaterialError::CallCycle {
            fragment: name.to_owned(),
        });
    }
    let fragment = fragments.get(name).expect("validated root and calls");
    for call in &fragment.calls {
        visit(&call.fragment, fragments, visiting, visited)?;
    }
    visiting.remove(name);
    visited.insert(name.to_owned());
    Ok(())
}

fn sorted_by_name<T>(mut values: Vec<T>, name: impl Fn(&T) -> &String) -> Vec<T> {
    values.sort_by(|left, right| name(left).cmp(name(right)));
    values
}

fn canonical_piece(output: &mut String, label: &str, value: &str) {
    output.push_str(label);
    output.push(':');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
