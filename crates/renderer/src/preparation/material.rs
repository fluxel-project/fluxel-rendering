//! Material, Shader, and geometry preparation for the first Forward consumer.
//!
//! This module validates logical products only. It has no device object, Graph
//! declaration, binding allocation, or submission authority.

use std::{fmt, sync::Arc};

use fluxel_assets::AssetSnapshot;
use fluxel_material::{
    CompiledMaterialVariant, MaterialAsset, MaterialCompiler, MaterialDefinition, MaterialError,
};
use fluxel_shader::{
    ShaderArtifact, ShaderComposer, ShaderCompositionRequest, ShaderError, VertexInputDeclaration,
};

use super::{GeometryError, GeometryLayout, PreparedGeometry};

/// The mutually validated logical products needed by a later Forward draw.
#[derive(Clone, Debug)]
pub struct PreparedMaterial {
    material: CompiledMaterialVariant,
    shader: Arc<ShaderArtifact>,
    geometry: PreparedGeometry,
}

impl PreparedMaterial {
    /// Returns the compiled static material variant.
    #[must_use]
    pub fn material(&self) -> &CompiledMaterialVariant {
        &self.material
    }

    /// Returns the validated logical Shader artifact.
    #[must_use]
    pub fn shader(&self) -> &Arc<ShaderArtifact> {
        &self.shader
    }

    /// Returns RHI vertex layouts resolved from the selected Shader interface.
    #[must_use]
    pub fn geometry(&self) -> &PreparedGeometry {
        &self.geometry
    }
}

/// A refusal while combining one Material, Shader request, and geometry layout.
#[derive(Debug)]
pub enum MaterialPreparationError {
    /// Material validation or normalization failed.
    Material(MaterialError),
    /// The complete logical Shader request failed validation or reflection.
    Shader(ShaderError),
    /// Geometry could not satisfy the selected logical vertex interface.
    Geometry(GeometryError),
    /// The pass request and Material supplied incompatible static choices.
    StaticOptionConflict {
        /// Static option name.
        name: String,
    },
    /// A reflected vertex input has no caller-supplied semantic mapping.
    MissingVertexSemantic {
        /// Reflected Shader location.
        location: u32,
    },
}

impl fmt::Display for MaterialPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Material(error) => error.fmt(formatter),
            Self::Shader(error) => error.fmt(formatter),
            Self::Geometry(error) => error.fmt(formatter),
            Self::StaticOptionConflict { name } => {
                write!(
                    formatter,
                    "pass and material disagree on static option {name}"
                )
            }
            Self::MissingVertexSemantic { location } => {
                write!(
                    formatter,
                    "Shader vertex location {location} has no geometry semantic"
                )
            }
        }
    }
}

impl std::error::Error for MaterialPreparationError {}

/// Compiles and validates one Material generation against a complete pass request.
///
/// The caller supplies the pass template, entry points, semantic mappings, and
/// device-independent requirements. This function adds the exact Material
/// contribution, validates the complete Shader request, then maps the Shader's
/// already-validated semantic declarations to physical geometry streams.
pub fn prepare_material(
    snapshot: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    geometry: &GeometryLayout,
    composer: &mut ShaderComposer,
    mut request: ShaderCompositionRequest,
) -> Result<PreparedMaterial, MaterialPreparationError> {
    let material =
        MaterialCompiler::compile_for(snapshot).map_err(MaterialPreparationError::Material)?;
    let contribution = material.contribution();
    for (name, value) in contribution.shader_static_options() {
        if let Some(existing) = request.static_options.get(&name) {
            if existing != &value {
                return Err(MaterialPreparationError::StaticOptionConflict { name });
            }
        } else {
            request.static_options.insert(name, value);
        }
    }
    request.source_units.push(contribution.source_unit());
    request.bindings.extend(contribution.binding_declarations());
    request
        .callables
        .extend(contribution.callable_declarations());
    let shader = composer
        .compose(&request)
        .map_err(MaterialPreparationError::Shader)?;
    validate_vertex_semantics(&request.vertex_inputs, shader.interface().vertex_inputs())?;
    let geometry = geometry
        .prepare(&request.vertex_inputs)
        .map_err(MaterialPreparationError::Geometry)?;
    Ok(PreparedMaterial {
        material,
        shader,
        geometry,
    })
}

fn validate_vertex_semantics(
    declarations: &[VertexInputDeclaration],
    reflected: &[fluxel_shader::ReflectedVertexInput],
) -> Result<(), MaterialPreparationError> {
    for input in reflected {
        if !declarations
            .iter()
            .any(|declaration| declaration.location == input.location && declaration.ty == input.ty)
        {
            return Err(MaterialPreparationError::MissingVertexSemantic {
                location: input.location,
            });
        }
    }
    Ok(())
}
