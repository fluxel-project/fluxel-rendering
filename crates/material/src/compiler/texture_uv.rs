//! Validation and canonicalization of generated-texture UV metadata.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    DiffuseRoughnessConfiguration, EmissiveConfiguration, MaterialError, MaterialTextureUv,
    PbrSurfaceFactors, TextureUvSet,
};

use super::canonical::piece as canonical_piece;

pub(super) fn validate(
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
    factors: Option<&PbrSurfaceFactors>,
    emissive: Option<&EmissiveConfiguration>,
    diffuse_roughness: Option<&DiffuseRoughnessConfiguration>,
) -> Result<BTreeMap<String, MaterialTextureUv>, MaterialError> {
    let mut generated_texture_names = BTreeSet::new();
    if let Some(factors) = factors {
        if let Some(base) = &factors.base_texture {
            generated_texture_names.insert(base.texture.as_str());
        }
        if let Some(packed) = &factors.packed_metallic_roughness {
            generated_texture_names.insert(packed.texture.as_str());
        }
    }
    if let Some(texture) = emissive.and_then(|configuration| configuration.texture.as_ref()) {
        generated_texture_names.insert(texture.texture.as_str());
    }
    if let Some(texture) =
        diffuse_roughness.and_then(|configuration| configuration.texture.as_ref())
    {
        generated_texture_names.insert(texture.texture.as_str());
    }
    for resource in texture_uv.keys() {
        if !generated_texture_names.contains(resource.as_str()) {
            return Err(MaterialError::TextureUvResourceUnused {
                resource: resource.clone(),
            });
        }
    }
    Ok(texture_uv.clone())
}

pub(super) fn canonical(texture_uv: &BTreeMap<String, MaterialTextureUv>, target: &mut String) {
    for (resource, coordinates) in texture_uv {
        canonical_piece(target, "texture-uv-resource", resource);
        canonical_piece(
            target,
            "texture-uv-set",
            match coordinates.set() {
                TextureUvSet::Uv0 => "uv0",
                TextureUvSet::Uv1 => "uv1",
            },
        );
        for (column, values) in coordinates.bits().into_iter().enumerate() {
            for (row, value) in values.into_iter().enumerate() {
                canonical_piece(
                    target,
                    "texture-uv-coefficient",
                    &format!("{column}:{row}:{value}"),
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Requirements {
    pub(super) any: bool,
    pub(super) surface: bool,
}

pub(super) fn requirements(
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
    factors: Option<&PbrSurfaceFactors>,
    diffuse_roughness: Option<&DiffuseRoughnessConfiguration>,
) -> Requirements {
    let mut surface_texture_names = BTreeSet::new();
    if let Some(factors) = factors {
        if let Some(base) = &factors.base_texture {
            surface_texture_names.insert(base.texture.as_str());
        }
        if let Some(packed) = &factors.packed_metallic_roughness {
            surface_texture_names.insert(packed.texture.as_str());
        }
    }
    if let Some(texture) =
        diffuse_roughness.and_then(|configuration| configuration.texture.as_ref())
    {
        surface_texture_names.insert(texture.texture.as_str());
    }
    let surface = texture_uv.iter().any(|(resource, coordinates)| {
        coordinates.set() == TextureUvSet::Uv1 && surface_texture_names.contains(resource.as_str())
    });
    let any = texture_uv
        .values()
        .any(|coordinates| coordinates.set() == TextureUvSet::Uv1);
    Requirements { any, surface }
}
