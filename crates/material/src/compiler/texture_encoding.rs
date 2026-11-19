//! Validation for optional material texture transfer-encoding requirements.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    MaterialDefinition, MaterialError, MaterialResource, MaterialResourceKind,
    MaterialTextureEncoding, MaterialTextureEncodingRequirement,
};

use super::canonical::piece;

pub(super) fn validate(
    requirements: &[MaterialTextureEncodingRequirement],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let resources = resources
        .iter()
        .map(|resource| (resource.name.as_str(), resource))
        .collect::<BTreeMap<_, _>>();
    let mut named = BTreeSet::new();
    for requirement in requirements {
        if !named.insert(requirement.resource.as_str()) {
            return Err(MaterialError::DuplicateTextureEncodingRequirement {
                resource: requirement.resource.clone(),
            });
        }
        let resource = resources
            .get(requirement.resource.as_str())
            .ok_or_else(|| MaterialError::TextureEncodingResourceMissing {
                resource: requirement.resource.clone(),
            })?;
        if !matches!(
            resource.kind,
            MaterialResourceKind::Texture2dF32
                | MaterialResourceKind::Texture2dArrayF32
                | MaterialResourceKind::TextureCubeF32
                | MaterialResourceKind::Texture3dF32
        ) {
            return Err(MaterialError::TextureEncodingResourceTypeMismatch {
                resource: requirement.resource.clone(),
                actual: resource.kind,
            });
        }
    }
    Ok(())
}

pub(super) fn canonical(requirements: &[MaterialTextureEncodingRequirement], target: &mut String) {
    let mut requirements = requirements.to_vec();
    requirements.sort_by(|left, right| left.resource.cmp(&right.resource));
    for requirement in requirements {
        piece(target, "texture-encoding-resource", &requirement.resource);
        piece(
            target,
            "texture-encoding",
            match requirement.encoding {
                MaterialTextureEncoding::Linear => "linear",
                MaterialTextureEncoding::Srgb => "srgb",
                MaterialTextureEncoding::SrgbEncodedUnorm => "srgb-encoded-unorm",
            },
        );
    }
}

pub(super) fn effective_requirements(
    definition: &MaterialDefinition,
) -> Result<Vec<MaterialTextureEncodingRequirement>, MaterialError> {
    let mut requirements = definition.texture_encoding_requirements().to_vec();
    if let Some(texture) = definition
        .emissive_configuration()
        .and_then(|configuration| configuration.texture.as_ref())
    {
        match requirements
            .iter()
            .find(|requirement| requirement.resource == texture.texture)
        {
            Some(requirement) if requirement.encoding != texture.encoding => {
                return Err(MaterialError::EmissiveTextureEncodingConflict {
                    resource: texture.texture.clone(),
                });
            }
            Some(_) => {}
            None => requirements.push(MaterialTextureEncodingRequirement {
                resource: texture.texture.clone(),
                encoding: texture.encoding,
            }),
        }
    }
    let requirements = super::pbr_factors::effective_texture_encodings(
        definition.pbr_surface_factors(),
        requirements,
    )?;
    let requirements = super::diffuse_roughness::effective_texture_encodings(
        definition.diffuse_roughness_configuration(),
        requirements,
    )?;
    let requirements = super::ambient_occlusion::effective_texture_encodings(
        definition.ambient_occlusion_configuration(),
        requirements,
    )?;
    super::transmission::effective_texture_encodings(
        definition.refraction_environment(),
        requirements,
    )
}
