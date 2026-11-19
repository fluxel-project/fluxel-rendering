//! Validation for optional material texture transfer-encoding requirements.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    MaterialError, MaterialResource, MaterialResourceKind, MaterialTextureEncoding,
    MaterialTextureEncodingRequirement,
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
            MaterialResourceKind::Texture2dF32 | MaterialResourceKind::Texture2dArrayF32
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
