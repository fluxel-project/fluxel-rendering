//! Resolves per-draw values against the selected material and shader ABI.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use fluxel_material::{CompiledMaterialVariant, DynamicValue, MaterialInstance};

use super::{ForwardDrawSource, ForwardPreparationError, ForwardResources};
use crate::preparation::MaterialBindingSchema;

pub(super) fn resolve_resource_aliases<'a>(
    resources: &'a ForwardResources,
    instance: Option<&MaterialInstance>,
    variant: &CompiledMaterialVariant,
    schema: &MaterialBindingSchema,
    source: ForwardDrawSource,
) -> Result<Cow<'a, ForwardResources>, ForwardPreparationError> {
    let Some(instance) = instance else {
        if schema.unused_textures.is_empty() && schema.unused_samplers.is_empty() {
            return Ok(Cow::Borrowed(resources));
        }
        let mut active = resources.clone();
        active
            .textures
            .retain(|binding| !schema.unused_textures.contains(&binding.resource));
        if active
            .sampler
            .as_ref()
            .is_some_and(|binding| schema.unused_samplers.contains(&binding.resource))
        {
            active.sampler = None;
        }
        return Ok(Cow::Owned(active));
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
    let mut used: HashSet<_> = schema
        .unused_textures
        .iter()
        .filter_map(|name| instance.resource(name))
        .collect();
    let mut mapped = ForwardResources {
        geometry: resources.geometry,
        material: resources.material,
        textures: Vec::with_capacity(schema.textures.len()),
        sampler: resources.sampler.clone(),
        tint: resources.tint,
    };
    for binding in &schema.textures {
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
    if let Some(sampler) = &mut mapped.sampler {
        let active = schema.sampler_name.as_deref();
        let active_alias = active.and_then(|name| instance.resource(name));
        if let Some(name) = active.filter(|_| active_alias == Some(sampler.resource.as_str())) {
            sampler.resource = name.to_owned();
        } else if schema
            .unused_samplers
            .iter()
            .any(|name| instance.resource(name) == Some(sampler.resource.as_str()))
        {
            mapped.sampler = None;
        } else if let Some(name) = active {
            return Err(ForwardPreparationError::SamplerResourceMismatch {
                source,
                expected: active_alias.unwrap_or(name).to_owned(),
                actual: sampler.resource.clone(),
            });
        } else {
            return Err(ForwardPreparationError::UnexpectedSamplerResource {
                source,
                resource: sampler.resource.clone(),
            });
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
        if !schema.legacy_tint_compatible {
            return Err(ForwardPreparationError::MaterialBindingSchema { source });
        }
        if schema.uniforms.is_empty() {
            return Ok(Vec::new());
        }
        if tint.iter().any(|value| !value.is_finite()) {
            return Err(ForwardPreparationError::InvalidTint { source });
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
