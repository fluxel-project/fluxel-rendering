//! Resolves per-draw values against the selected material and shader ABI.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use fluxel_material::{
    CompiledMaterialVariant, DynamicValue, MaterialInstance, MaterialResourceKind, MaterialType,
};

use super::{ForwardDrawSource, ForwardPreparationError, ForwardResources};
use crate::preparation::MaterialBindingSchema;

pub(super) fn resolve_resource_aliases<'a>(
    resources: &'a ForwardResources,
    instance: Option<&MaterialInstance>,
    variant: &CompiledMaterialVariant,
    source: ForwardDrawSource,
) -> Result<Cow<'a, ForwardResources>, ForwardPreparationError> {
    let Some(instance) = instance else {
        return Ok(Cow::Borrowed(resources));
    };
    if variant.material() != Some(instance.material()) || variant.key() != instance.variant() {
        return Err(ForwardPreparationError::MaterialInstanceVariant { source });
    }
    let mut provided = HashMap::with_capacity(resources.textures.len());
    for texture in &resources.textures {
        if provided
            .insert(texture.resource.as_str(), texture)
            .is_some()
        {
            return Err(ForwardPreparationError::DuplicateTextureResource {
                source,
                resource: texture.resource.clone(),
            });
        }
    }
    let mut used = HashSet::with_capacity(provided.len());
    let mut mapped = ForwardResources {
        geometry: resources.geometry,
        material: resources.material,
        textures: Vec::new(),
        sampler: resources.sampler.clone(),
        tint: resources.tint,
    };
    for binding in &variant.contribution().bindings {
        match binding.kind {
            MaterialResourceKind::Texture2dF32 => {
                let alias = instance.resource(&binding.name).ok_or_else(|| {
                    ForwardPreparationError::MissingTextureResource {
                        source,
                        resource: binding.name.clone(),
                    }
                })?;
                let mut texture = (*provided.get(alias).ok_or_else(|| {
                    ForwardPreparationError::MissingTextureResource {
                        source,
                        resource: alias.to_owned(),
                    }
                })?)
                .clone();
                texture.resource.clone_from(&binding.name);
                mapped.textures.push(texture);
                used.insert(alias);
            }
            MaterialResourceKind::Sampler { comparison: false } => {
                if let Some(sampler) = &mut mapped.sampler {
                    let alias = instance.resource(&binding.name);
                    if alias != Some(sampler.resource.as_str()) {
                        return Err(ForwardPreparationError::SamplerResourceMismatch {
                            source,
                            expected: alias.unwrap_or(&binding.name).to_owned(),
                            actual: sampler.resource.clone(),
                        });
                    }
                    sampler.resource.clone_from(&binding.name);
                }
            }
            _ => {}
        }
    }
    if let Some(texture) = resources
        .textures
        .iter()
        .find(|texture| !used.contains(texture.resource.as_str()))
    {
        return Err(ForwardPreparationError::UnexpectedTextureResource {
            source,
            resource: texture.resource.clone(),
        });
    }
    Ok(Cow::Owned(mapped))
}

pub(super) fn resolve_uniform_values(
    instance: Option<&MaterialInstance>,
    schema: &MaterialBindingSchema,
    tint: [f32; 4],
    source: ForwardDrawSource,
) -> Result<Vec<DynamicValue>, ForwardPreparationError> {
    let Some(instance) = instance else {
        if schema.uniforms.len() != 1
            || schema.uniforms[0].ty != MaterialType::Vec4
            || schema.textures.is_empty()
        {
            return Err(ForwardPreparationError::MaterialBindingSchema { source });
        }
        return Ok(vec![DynamicValue::Vec4(tint)]);
    };
    schema
        .uniforms
        .iter()
        .map(|uniform| {
            let invalid = || ForwardPreparationError::InvalidMaterialParameter {
                source,
                parameter: uniform.parameter.clone(),
            };
            let value = instance.parameter(&uniform.parameter).ok_or_else(invalid)?;
            if value.ty() != uniform.ty || value.components().iter().any(|value| !value.is_finite())
            {
                return Err(invalid());
            }
            Ok(value.clone())
        })
        .collect()
}
