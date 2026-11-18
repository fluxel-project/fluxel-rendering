//! Consumer-owned binding facts derived from one prepared material.

use fluxel_material::{MaterialResourceKind, MaterialTextureEncoding, MaterialType};
use fluxel_shader::{BindingKind, BindingShape, ShaderStage};

use super::{ForwardDrawSource, ForwardPreparationError, PreparedMaterial};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaterialBindingSchema {
    pub(crate) uniforms: Vec<MaterialUniformBinding>,
    pub(crate) textures: Vec<MaterialTextureBinding>,
    pub(crate) sampler: Option<u32>,
    pub(crate) sampler_name: Option<String>,
    pub(crate) unused_textures: Vec<String>,
    pub(crate) unused_samplers: Vec<String>,
    pub(crate) legacy_tint_compatible: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaterialUniformBinding {
    pub(crate) binding: u32,
    pub(crate) parameter: String,
    pub(crate) ty: MaterialType,
    pub(crate) size: u64,
}

/// One material-declared texture location, ordered by its actual binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaterialTextureBinding {
    pub(crate) binding: u32,
    pub(crate) name: String,
    /// Optional material-declared asset encoding requirement.
    pub(crate) encoding: Option<MaterialTextureEncoding>,
}

pub(crate) fn material_binding_schema(
    prepared: &PreparedMaterial,
    source: ForwardDrawSource,
) -> Result<MaterialBindingSchema, ForwardPreparationError> {
    let invalid = || ForwardPreparationError::MaterialBindingSchema { source };
    let contribution = prepared.material().contribution();
    let reflected = prepared.shader().interface().bindings();
    let mut schema = MaterialBindingSchema {
        uniforms: Vec::new(),
        textures: Vec::new(),
        sampler: None,
        sampler_name: None,
        unused_textures: Vec::new(),
        unused_samplers: Vec::new(),
        legacy_tint_compatible: false,
    };
    let mut active_count = 0;
    let mut original_uniforms = 0;
    let mut original_vec4 = false;
    let mut original_textures = 0;
    let mut original_samplers = 0;
    for binding in &contribution.bindings {
        if binding.location.group != 0 {
            return Err(invalid());
        }
        let location = binding.location.binding;
        let actual = reflected
            .iter()
            .find(|actual| actual.group == 0 && actual.binding == location);
        match binding.kind {
            MaterialResourceKind::Texture2dF32 => original_textures += 1,
            MaterialResourceKind::Sampler { comparison: false } => original_samplers += 1,
            kind if kind.uniform_material_type().is_some() => {
                original_uniforms += 1;
                original_vec4 |= kind.uniform_material_type() == Some(MaterialType::Vec4);
            }
            _ => return Err(invalid()),
        }
        let Some(actual) = actual else {
            if !prepared.emissive_disabled() {
                return Err(invalid());
            }
            match binding.kind {
                MaterialResourceKind::Texture2dF32 => {
                    schema.unused_textures.push(binding.name.clone())
                }
                MaterialResourceKind::Sampler { .. } => {
                    schema.unused_samplers.push(binding.name.clone())
                }
                _ => {}
            }
            continue;
        };
        active_count += 1;
        if !(actual.read || actual.query)
            || actual.write
            || actual.visibility.len() != 1
            || !actual.visibility.contains(&ShaderStage::Fragment)
        {
            return Err(invalid());
        }

        match binding.kind {
            MaterialResourceKind::Texture2dF32
                if matches!(actual.kind, BindingKind::Texture)
                    && actual.shape == Some(BindingShape::Texture2dF32) =>
            {
                schema.textures.push(MaterialTextureBinding {
                    binding: location,
                    name: binding.name.clone(),
                    encoding: prepared
                        .material()
                        .texture_encoding_requirements()
                        .iter()
                        .find(|requirement| requirement.resource == binding.name)
                        .map(|requirement| requirement.encoding),
                });
            }
            MaterialResourceKind::Sampler { comparison: false }
                if matches!(actual.kind, BindingKind::Sampler)
                    && actual.shape == Some(BindingShape::Sampler { comparison: false })
                    && schema.sampler.is_none() =>
            {
                schema.sampler = Some(location);
                schema.sampler_name = Some(binding.name.clone());
            }
            kind if matches!(actual.kind, BindingKind::Uniform) => {
                let ty = kind.uniform_material_type().ok_or_else(invalid)?;
                let size = kind.uniform_min_binding_size().ok_or_else(invalid)?;
                if actual.shape != Some(BindingShape::Uniform(ty.shader_type())) {
                    return Err(invalid());
                }
                let parameter = contribution
                    .parameter_bindings
                    .iter()
                    .find(|parameter| {
                        parameter.location == binding.location
                            && parameter.resource == binding.name
                            && parameter.ty == ty
                    })
                    .ok_or_else(invalid)?;
                schema.uniforms.push(MaterialUniformBinding {
                    binding: location,
                    parameter: parameter.parameter.clone(),
                    ty,
                    size,
                });
            }
            _ => return Err(invalid()),
        }
    }
    schema
        .uniforms
        .sort_unstable_by_key(|uniform| uniform.binding);
    schema
        .textures
        .sort_unstable_by_key(|texture| texture.binding);
    if reflected
        .iter()
        .filter(|binding| binding.group == 0)
        .count()
        != active_count
    {
        return Err(invalid());
    }
    schema.legacy_tint_compatible =
        original_uniforms == 1 && original_vec4 && original_textures != 0 && original_samplers <= 1;
    Ok(schema)
}
