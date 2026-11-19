//! Generic Shader contribution assembled from normalized Material inputs.

use std::collections::BTreeMap;

use fluxel_shader::{
    BindingDeclaration, BindingKind, BindingShape, CallableDeclaration, SourceUnit,
};

use crate::{BindingLocation, MaterialResourceKind, MaterialType, StaticValue};

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
    /// Dynamic parameter name accepted by `MaterialInstance`.
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
    source_units: Vec<SourceUnit>,
    /// Resource declarations Shader must compare against reflection.
    pub bindings: Vec<MaterialBindingDeclaration>,
    /// Dynamic parameter-to-uniform associations consumed by the renderer.
    pub parameter_bindings: Vec<MaterialParameterBinding>,
    /// Function signatures that Shader must validate against actual WGSL.
    pub callables: Vec<CallableDeclaration>,
    /// Canonical static choices participating in composition identity.
    pub static_options: BTreeMap<String, StaticValue>,
    vertex_varyings: Vec<(u32, MaterialType)>,
    requires_texture_uv1: bool,
    requires_surface_texture_uv1: bool,
}

impl MaterialShaderContribution {
    pub(crate) fn new(
        source_id: String,
        generated_source: String,
        code_node_units: Vec<SourceUnit>,
        bindings: Vec<MaterialBindingDeclaration>,
        parameter_bindings: Vec<MaterialParameterBinding>,
        callables: Vec<CallableDeclaration>,
        static_options: BTreeMap<String, StaticValue>,
    ) -> Self {
        let mut source_units = Vec::with_capacity(code_node_units.len() + 1);
        source_units.push(SourceUnit::new(source_id.clone(), generated_source));
        source_units.extend(code_node_units);
        Self {
            source_units,
            bindings,
            parameter_bindings,
            callables,
            static_options,
            vertex_varyings: Vec::new(),
            requires_texture_uv1: false,
            requires_surface_texture_uv1: false,
        }
    }

    /// Returns the ordered source units consumed by Shader composition.
    ///
    /// The generated glue is first; each following Code Node is the exact
    /// source unit supplied by its author. This is the authoritative source
    /// representation for diagnostics and composition.
    #[must_use]
    pub fn source_units(&self) -> Vec<SourceUnit> {
        self.source_units.clone()
    }

    /// Returns the stable identity of the generated Material glue unit.
    #[must_use]
    pub fn source_id(&self) -> &str {
        &self.source_units[0].id.0
    }

    /// Returns a valid aggregate source snapshot for legacy consumers.
    ///
    /// New composition must use [`Self::source_units`] to preserve Code Node
    /// attribution. The aggregate is derived on demand, so it cannot drift from
    /// the stored source units.
    #[must_use]
    pub fn source(&self) -> String {
        self.source_units
            .iter()
            .map(|unit| unit.source.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns a single aggregate source unit for legacy one-unit composition.
    #[must_use]
    pub fn source_unit(&self) -> SourceUnit {
        SourceUnit::new(self.source_id(), self.source())
    }

    /// Returns Shader-owned callable metadata for WGSL signature validation.
    #[must_use]
    pub fn callable_declarations(&self) -> Vec<CallableDeclaration> {
        self.callables.clone()
    }

    /// Returns material-local vertex varying identifiers and their values'
    /// Shader-visible types in interpolation declaration order.
    #[must_use]
    pub fn vertex_varyings(&self) -> &[(u32, MaterialType)] {
        &self.vertex_varyings
    }

    pub(super) fn with_vertex_varyings(
        mut self,
        vertex_varyings: Vec<(u32, MaterialType)>,
    ) -> Self {
        self.vertex_varyings = vertex_varyings;
        self
    }

    pub(super) fn with_texture_uv_requirements(
        mut self,
        requires_texture_uv1: bool,
        requires_surface_texture_uv1: bool,
    ) -> Self {
        self.requires_texture_uv1 = requires_texture_uv1;
        self.requires_surface_texture_uv1 = requires_surface_texture_uv1;
        self
    }

    /// Returns whether any generated texture evaluator requires the UV1 input.
    #[must_use]
    pub const fn requires_texture_uv1(&self) -> bool {
        self.requires_texture_uv1
    }

    /// Returns whether generated base or ORM surface sampling requires UV1.
    #[must_use]
    pub const fn requires_surface_texture_uv1(&self) -> bool {
        self.requires_surface_texture_uv1
    }

    /// Converts resource metadata into Shader-owned binding declarations.
    #[must_use]
    pub fn binding_declarations(&self) -> Vec<BindingDeclaration> {
        self.bindings
            .iter()
            .map(|binding| {
                let (kind, shape) = match binding.kind {
                    MaterialResourceKind::Texture2dF32 => {
                        (BindingKind::Texture, BindingShape::Texture2dF32)
                    }
                    MaterialResourceKind::Texture2dDepth => {
                        (BindingKind::Texture, BindingShape::Texture2dDepth)
                    }
                    MaterialResourceKind::Texture2dArrayDepth => {
                        (BindingKind::Texture, BindingShape::Texture2dArrayDepth)
                    }
                    MaterialResourceKind::StorageTexture2dRgba8UnormRead => (
                        BindingKind::StorageTexture,
                        BindingShape::StorageTexture2dRgba8UnormRead,
                    ),
                    MaterialResourceKind::Texture2dArrayF32 => {
                        (BindingKind::Texture, BindingShape::Texture2dArrayF32)
                    }
                    MaterialResourceKind::TextureCubeF32 => {
                        (BindingKind::Texture, BindingShape::TextureCubeF32)
                    }
                    MaterialResourceKind::Texture3dF32 => {
                        (BindingKind::Texture, BindingShape::Texture3dF32)
                    }
                    MaterialResourceKind::Sampler { comparison } => {
                        (BindingKind::Sampler, BindingShape::Sampler { comparison })
                    }
                    uniform => {
                        let ty = uniform
                            .uniform_material_type()
                            .expect("all remaining resource kinds are uniforms");
                        (
                            BindingKind::Uniform,
                            BindingShape::Uniform(ty.shader_type()),
                        )
                    }
                };
                BindingDeclaration::new(binding.location.group, binding.location.binding, kind)
                    .with_shape(shape)
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
