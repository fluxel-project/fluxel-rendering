//! Material, Shader, and geometry preparation for the first Forward consumer.
//!
//! This module validates logical products only. It has no device object, Graph
//! declaration, binding allocation, or submission authority.

use std::{fmt, sync::Arc};

use fluxel_assets::AssetSnapshot;
use fluxel_material::{
    CompiledMaterialVariant, MaterialAsset, MaterialCompiler, MaterialDefinition, MaterialError,
    SurfaceAlphaMode,
};
use fluxel_shader::{
    BindingKind, BindingShape, ShaderArtifact, ShaderComposer, ShaderCompositionRequest,
    ShaderError, ShaderStage, VertexInputDeclaration,
};

use super::{
    GeometryError, GeometryLayout, PreparedGeometry,
    forward_fragment::{OitDepthPrecision, OitDrawStage},
    forward_schema::MaterialBindingSchema,
};

/// The mutually validated logical products needed by a later Forward draw.
#[derive(Clone, Debug)]
pub struct PreparedMaterial {
    material: CompiledMaterialVariant,
    automatic_alpha_mode: Option<SurfaceAlphaMode>,
    shader: Arc<ShaderArtifact>,
    geometry: PreparedGeometry,
    environment_irradiance: bool,
    specular_ibl: bool,
    transmission: bool,
    scene_refraction: bool,
    emissive_disabled: bool,
    oit_fragment: Option<(OitDrawStage, OitDepthPrecision)>,
    binding_schema: Option<MaterialBindingSchema>,
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

    pub(crate) const fn automatic_alpha_mode(&self) -> Option<SurfaceAlphaMode> {
        self.automatic_alpha_mode
    }

    pub(crate) fn uses_specular_ibl(&self) -> bool {
        self.specular_ibl
    }

    pub(crate) fn uses_transmission(&self) -> bool {
        self.transmission
    }

    pub(crate) const fn uses_scene_refraction(&self) -> bool {
        self.scene_refraction
    }

    /// Returns the renderer-owned OIT shader stage and its admitted extrema
    /// representation when this artifact was prepared for dual depth peeling.
    pub(crate) fn oit_fragment(&self) -> Option<(OitDrawStage, OitDepthPrecision)> {
        self.oit_fragment
    }

    pub(crate) fn uses_environment_irradiance(&self) -> bool {
        self.environment_irradiance
    }

    pub(crate) fn emissive_disabled(&self) -> bool {
        self.emissive_disabled
    }

    pub(crate) fn binding_schema(&self) -> Option<&MaterialBindingSchema> {
        self.binding_schema.as_ref()
    }

    pub(crate) fn with_binding_schema(mut self, schema: MaterialBindingSchema) -> Self {
        self.binding_schema = Some(schema);
        self
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
    /// The selected environment route does not match the reflected sampling bindings.
    InvalidEnvironmentRoute,
    /// Renderer-owned OIT metadata does not match the reflected texture ABI.
    InvalidOitFragmentRoute,
    /// The selected renderer route cannot carry Material vertex varyings.
    InvalidVertexVaryingRoute {
        /// The selected material context, if the request supplied one.
        context: Option<String>,
    },
    /// A PBR template did not expose exactly one marker for the selected varying role.
    MissingVertexVaryingMarker {
        /// The missing marker's role.
        marker: &'static str,
    },
    /// Dynamic varyings would exceed the WGSL location address space.
    VertexVaryingLocationOverflow,
    /// The secondary texture coordinate must advance with mesh vertices.
    TextureUv1RequiresVertexStep,
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
            Self::InvalidEnvironmentRoute => {
                formatter.write_str("invalid renderer environment route")
            }
            Self::InvalidOitFragmentRoute => {
                formatter.write_str("invalid renderer OIT fragment route")
            }
            Self::InvalidVertexVaryingRoute { context } => write!(
                formatter,
                "renderer material context cannot carry vertex varyings: {}",
                context.as_deref().unwrap_or("missing"),
            ),
            Self::MissingVertexVaryingMarker { marker } => {
                write!(
                    formatter,
                    "PBR template requires one vertex-varying {marker} marker"
                )
            }
            Self::TextureUv1RequiresVertexStep => {
                write!(
                    formatter,
                    "material texture coordinate UV1 requires a vertex-rate stream"
                )
            }
            Self::VertexVaryingLocationOverflow => {
                formatter.write_str("vertex varying locations overflow u32")
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
    request: ShaderCompositionRequest,
) -> Result<PreparedMaterial, MaterialPreparationError> {
    prepare_material_for_alpha(snapshot, geometry, composer, request, None)
}

pub(crate) fn prepare_material_for_alpha(
    snapshot: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    geometry: &GeometryLayout,
    composer: &mut ShaderComposer,
    request: ShaderCompositionRequest,
    automatic_alpha_mode: Option<SurfaceAlphaMode>,
) -> Result<PreparedMaterial, MaterialPreparationError> {
    let material =
        MaterialCompiler::compile_for(snapshot).map_err(MaterialPreparationError::Material)?;
    prepare_compiled_material(material, geometry, composer, request, automatic_alpha_mode)
}

pub(super) fn prepare_compiled_material(
    material: CompiledMaterialVariant,
    geometry: &GeometryLayout,
    composer: &mut ShaderComposer,
    mut request: ShaderCompositionRequest,
    automatic_alpha_mode: Option<SurfaceAlphaMode>,
) -> Result<PreparedMaterial, MaterialPreparationError> {
    let emissive_disabled = request
        .static_options
        .get("renderer.emissive")
        .is_some_and(|value| value == "disabled");
    if material
        .surface_policy()
        .blend_mode
        .uses_premultiplied_rgb()
    {
        super::forward_fragment::use_premultiplied_rgb(&mut request);
    }
    crate::pbr::specialize_normal_basis(&mut request, material.normal_map_settings());
    crate::pbr::specialize_diffuse_roughness(&mut request, material.uses_diffuse_roughness());
    crate::pbr::specialize_ambient_occlusion(&mut request, material.uses_ambient_occlusion_color());
    let contribution = material.contribution();
    let uv1_argument = contribution.requires_texture_uv1();
    // A view-disabled emissive sampler cannot reach the supplied secondary UV.
    let uv1_input =
        uv1_argument && (!emissive_disabled || contribution.requires_surface_texture_uv1());
    if uv1_input
        && geometry.streams.iter().any(|stream| {
            matches!(stream.step_mode, fluxel_rhi::api::VertexStepMode::Instance)
                && stream
                    .attributes
                    .iter()
                    .any(|attribute| attribute.semantic == "UV1")
        })
    {
        return Err(MaterialPreparationError::TextureUv1RequiresVertexStep);
    }
    super::vertex_varyings::specialize(
        &mut request,
        contribution.vertex_varyings(),
        uv1_argument,
        uv1_input,
    )?;
    for (name, value) in contribution.shader_static_options() {
        if let Some(existing) = request.static_options.get(&name) {
            if existing != &value {
                return Err(MaterialPreparationError::StaticOptionConflict { name });
            }
        } else {
            request.static_options.insert(name, value);
        }
    }
    request
        .source_units
        .extend(contribution.source_units_with_alpha_mode(automatic_alpha_mode));
    request.bindings.extend(
        contribution
            .binding_declarations()
            .into_iter()
            .map(|binding| {
                if emissive_disabled && binding.group == 0 {
                    binding.with_optional_use()
                } else {
                    binding
                }
            }),
    );
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
    let environment_irradiance = request
        .static_options
        .get("renderer.environment")
        .is_some_and(|value| value == "latlong-diffuse-irradiance-v1");
    if environment_irradiance {
        let bindings = shader.interface().bindings();
        let valid = [
            (
                0,
                fluxel_shader::BindingKind::Texture,
                fluxel_shader::BindingShape::Texture2dF32,
            ),
            (
                1,
                fluxel_shader::BindingKind::Sampler,
                fluxel_shader::BindingShape::Sampler { comparison: false },
            ),
        ]
        .into_iter()
        .all(|(binding, kind, shape)| {
            bindings.iter().any(|actual| {
                actual.group == 2
                    && actual.binding == binding
                    && actual.kind == kind
                    && actual.shape == Some(shape)
                    && actual.read
                    && !actual.write
            })
        });
        if !valid {
            return Err(MaterialPreparationError::InvalidEnvironmentRoute);
        }
    }
    let specular_ibl = request
        .static_options
        .get("renderer.specular_ibl")
        .is_some_and(|value| value == "latlong-linear-roughness-lod-height-correlated-v1");
    let transmission_route = request
        .static_options
        .get("renderer.transmission")
        .map(String::as_str);
    let scene_refraction = transmission_route == Some("scene-color-projected-v1");
    let scene_environment_refraction =
        transmission_route == Some("environment-refraction-beer-lambert-v1");
    let transmission = scene_refraction
        || scene_environment_refraction
        || transmission_route == Some("material-environment-refraction-v1");
    if (scene_refraction || scene_environment_refraction) && !specular_ibl {
        return Err(MaterialPreparationError::InvalidEnvironmentRoute);
    }
    if specular_ibl {
        use fluxel_shader::{BindingKind, BindingShape};
        for (binding, kind, shape) in [
            (2, BindingKind::Texture, Some(BindingShape::Texture2dF32)),
            (
                3,
                BindingKind::Sampler,
                Some(BindingShape::Sampler { comparison: false }),
            ),
            (4, BindingKind::Texture, Some(BindingShape::Texture2dF32)),
            (
                5,
                BindingKind::Sampler,
                Some(BindingShape::Sampler { comparison: false }),
            ),
            (6, BindingKind::Uniform, None),
        ] {
            if !shader.interface().bindings().iter().any(|actual| {
                actual.group == 2
                    && actual.binding == binding
                    && actual.kind == kind
                    && shape.is_none_or(|shape| actual.shape == Some(shape))
                    && actual.read
                    && !actual.write
            }) {
                return Err(MaterialPreparationError::InvalidEnvironmentRoute);
            }
        }
    }
    let oit_fragment = oit_fragment_route(&request, shader.interface().bindings())?;
    Ok(PreparedMaterial {
        material,
        automatic_alpha_mode,
        shader,
        geometry,
        environment_irradiance,
        specular_ibl,
        transmission,
        scene_refraction,
        emissive_disabled,
        oit_fragment,
        binding_schema: None,
    })
}

fn oit_fragment_route(
    request: &ShaderCompositionRequest,
    bindings: &[fluxel_shader::ReflectedBinding],
) -> Result<Option<(OitDrawStage, OitDepthPrecision)>, MaterialPreparationError> {
    let stage = match request
        .static_options
        .get("renderer.forward_fragment_stage")
        .map(String::as_str)
    {
        Some("ordinary") | None => return Ok(None),
        Some("oit-initial-depth") => OitDrawStage::InitialDepth,
        Some("oit-peel") => OitDrawStage::Peel,
        Some(_) => return Err(MaterialPreparationError::InvalidOitFragmentRoute),
    };
    let precision = match request
        .static_options
        .get("renderer.oit_depth_precision")
        .map(String::as_str)
    {
        Some("rg32float") => OitDepthPrecision::Rg32Float,
        Some("rg16float") => OitDepthPrecision::Rg16Float,
        _ => return Err(MaterialPreparationError::InvalidOitFragmentRoute),
    };
    let expected = match stage {
        OitDrawStage::InitialDepth => &[0][..],
        OitDrawStage::Peel => &[0, 1][..],
    };
    let valid = bindings.iter().filter(|actual| actual.group == 3).count() == expected.len()
        && expected.iter().all(|binding| {
            bindings.iter().any(|actual| {
                actual.group == 3
                    && actual.binding == *binding
                    && actual.kind == BindingKind::Texture
                    && actual.shape == Some(BindingShape::Texture2dF32)
                    && actual.visibility.contains(&ShaderStage::Fragment)
                    && actual.read
                    && !actual.write
            })
        });
    if !valid {
        return Err(MaterialPreparationError::InvalidOitFragmentRoute);
    }
    Ok(Some((stage, precision)))
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
