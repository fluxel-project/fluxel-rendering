//! Consumer-owned binding facts derived from one prepared material.

use fluxel_material::{MaterialResourceKind, MaterialType};
use fluxel_shader::{BindingKind, BindingShape, ShaderStage};

use super::{ForwardDrawSource, ForwardPreparationError, PreparedMaterial};

#[derive(Clone, Debug)]
pub(crate) struct MaterialBindingSchema {
    pub(crate) tint: u32,
    pub(crate) texture: u32,
    pub(crate) texture_name: String,
    pub(crate) sampler: Option<u32>,
}

pub(crate) fn material_binding_schema(
    prepared: &PreparedMaterial,
    source: ForwardDrawSource,
) -> Result<MaterialBindingSchema, ForwardPreparationError> {
    let contribution = prepared.material().contribution();
    let bindings: Vec<_> = contribution
        .bindings
        .iter()
        .filter(|binding| binding.location.group == 0)
        .collect();
    let tint: Vec<_> = bindings
        .iter()
        .filter(|binding| binding.kind == MaterialResourceKind::UniformVec4)
        .collect();
    let texture: Vec<_> = bindings
        .iter()
        .filter(|binding| binding.kind == MaterialResourceKind::Texture2dF32)
        .collect();
    let sampler: Vec<_> = bindings
        .iter()
        .filter(|binding| binding.kind == MaterialResourceKind::Sampler { comparison: false })
        .collect();
    let valid_parameter = tint.len() == 1
        && contribution
            .parameter_bindings
            .iter()
            .filter(|parameter| {
                parameter.ty == MaterialType::Vec4
                    && parameter.resource == tint[0].name
                    && parameter.location == tint[0].location
            })
            .count()
            == 1;
    if !valid_parameter
        || texture.len() != 1
        || sampler.len() > 1
        || bindings.len() != contribution.bindings.len()
        || bindings.len() != 2 + sampler.len()
    {
        return Err(ForwardPreparationError::MaterialBindingSchema { source });
    }
    if prepared
        .shader()
        .interface()
        .bindings()
        .iter()
        .any(|actual| {
            actual.group == 0
                && !bindings.iter().any(|binding| {
                    binding.location.binding == actual.binding
                        && (actual.read || actual.query)
                        && !actual.write
                        && actual.visibility.contains(&ShaderStage::Fragment)
                        && actual
                            .visibility
                            .iter()
                            .all(|stage| *stage == ShaderStage::Fragment)
                        && matches!(
                            (
                                binding.kind.clone(),
                                actual.kind.clone(),
                                actual.shape.clone()
                            ),
                            (
                                MaterialResourceKind::UniformVec4,
                                BindingKind::Uniform,
                                Some(BindingShape::Uniform(fluxel_shader::ValueType::Vec4(
                                    fluxel_shader::ScalarType::F32
                                )))
                            ) | (
                                MaterialResourceKind::Texture2dF32,
                                BindingKind::Texture,
                                Some(BindingShape::Texture2dF32)
                            ) | (
                                MaterialResourceKind::Sampler { comparison: false },
                                BindingKind::Sampler,
                                Some(BindingShape::Sampler { comparison: false })
                            )
                        )
                })
        })
    {
        return Err(ForwardPreparationError::MaterialBindingSchema { source });
    }
    Ok(MaterialBindingSchema {
        tint: tint[0].location.binding,
        texture: texture[0].location.binding,
        texture_name: texture[0].name.clone(),
        sampler: sampler.first().map(|binding| binding.location.binding),
    })
}
