//! Immutable material texture resolution and source encoding admission.

use std::collections::HashMap;

use super::forward_validation::{
    validate_cube_texture_array, validate_generated_cube_texture_array,
    validate_generated_texture_array, validate_texture, validate_texture_array,
};
use super::*;
use crate::preparation::{MaterialBindingSchema, MaterialTextureBinding};

pub(super) fn ordered_texture_sources<'a>(
    provided: &'a [ForwardTextureBinding],
    expected: &[MaterialTextureBinding],
    source: ForwardDrawSource,
) -> Result<Vec<&'a ForwardTextureBinding>, ForwardPreparationError> {
    let mut by_name = HashMap::with_capacity(provided.len());
    for texture in provided {
        if by_name.insert(texture.resource.as_str(), texture).is_some() {
            return Err(ForwardPreparationError::DuplicateTextureResource {
                source,
                resource: texture.resource.clone(),
            });
        }
    }
    let mut ordered = Vec::with_capacity(expected.len());
    for texture in expected {
        let Some(provided) = by_name.get(texture.name.as_str()) else {
            return Err(ForwardPreparationError::MissingTextureResource {
                source,
                resource: texture.name.clone(),
            });
        };
        ordered.push(*provided);
    }
    for texture in provided {
        if !expected
            .iter()
            .any(|expected| expected.name == texture.resource)
        {
            return Err(ForwardPreparationError::UnexpectedTextureResource {
                source,
                resource: texture.resource.clone(),
            });
        }
    }
    Ok(ordered)
}

pub(super) fn resolve_asset_textures(
    resources: &ForwardResources,
    schema: &MaterialBindingSchema,
    assets: &ForwardAssets,
    source: ForwardDrawSource,
) -> Result<Vec<ResolvedForwardTexture>, ForwardPreparationError> {
    ordered_texture_sources(&resources.textures, &schema.textures, source)?
        .into_iter()
        .zip(&schema.textures)
        .map(|(texture, expected)| match &texture.source {
            ForwardTextureSource::CaptureSlot(_) => {
                Err(ForwardPreparationError::CaptureTextureOutsideReceiver {
                    source,
                    resource: texture.resource.clone(),
                })
            }
            asset => resolve_texture_asset(asset, expected, assets, source),
        })
        .collect()
}

pub(super) fn resolve_texture_asset(
    binding: &ForwardTextureSource,
    expected: &MaterialTextureBinding,
    assets: &ForwardAssets,
    source: ForwardDrawSource,
) -> Result<ResolvedForwardTexture, ForwardPreparationError> {
    match binding {
        ForwardTextureSource::Asset(reference) => {
            require_texture_dimension(expected, fluxel_rhi::api::TextureViewDimension::D2, source)?;
            let asset = resolve_asset(&assets.textures, *reference, "texture", source)?;
            validate_texture(asset.value(), source)?;
            validate_texture_encoding(expected, asset.value().encoding, source)?;
            Ok(ResolvedForwardTexture::Asset(asset))
        }
        ForwardTextureSource::ArrayAsset(asset) => {
            require_texture_dimension(
                expected,
                fluxel_rhi::api::TextureViewDimension::D2Array,
                source,
            )?;
            validate_texture_array(asset.value(), source)?;
            validate_texture_encoding(expected, asset.value().encoding, source)?;
            Ok(ResolvedForwardTexture::ArrayAsset(asset.clone()))
        }
        ForwardTextureSource::ArrayAssetWithGeneratedMips { asset, mip_levels } => {
            require_texture_dimension(
                expected,
                fluxel_rhi::api::TextureViewDimension::D2Array,
                source,
            )?;
            validate_generated_texture_array(asset.value(), *mip_levels, source)?;
            validate_texture_encoding(expected, asset.value().encoding, source)?;
            Ok(ResolvedForwardTexture::GeneratedArray {
                asset: asset.clone(),
                mip_levels: *mip_levels,
            })
        }
        ForwardTextureSource::CubeAsset {
            asset,
            generated_mips,
        } => {
            require_texture_dimension(
                expected,
                fluxel_rhi::api::TextureViewDimension::Cube,
                source,
            )?;
            match generated_mips {
                Some(mip_levels) => {
                    validate_generated_cube_texture_array(asset.value(), *mip_levels, source)?;
                }
                None => validate_cube_texture_array(asset.value(), source)?,
            }
            validate_texture_encoding(expected, asset.value().encoding, source)?;
            Ok(ResolvedForwardTexture::CubeAsset {
                asset: asset.clone(),
                generated_mips: *generated_mips,
            })
        }
        ForwardTextureSource::RefractionRadiance(asset) => {
            require_texture_dimension(expected, fluxel_rhi::api::TextureViewDimension::D2, source)?;
            validate_texture_encoding(expected, TextureEncoding::Linear, source)?;
            Ok(ResolvedForwardTexture::RefractionRadiance(asset.clone()))
        }
        ForwardTextureSource::CaptureSlot(slot) => {
            Err(ForwardPreparationError::CaptureTextureOutsideReceiver {
                source,
                resource: slot.clone(),
            })
        }
    }
}

fn require_texture_dimension(
    expected: &MaterialTextureBinding,
    actual: fluxel_rhi::api::TextureViewDimension,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    if expected.dimension == actual {
        Ok(())
    } else {
        Err(ForwardPreparationError::TextureResourceDimensionMismatch {
            source,
            resource: expected.name.clone(),
            expected: expected.dimension,
            actual,
        })
    }
}

pub(super) fn validate_texture_encoding(
    expected: &MaterialTextureBinding,
    actual: TextureEncoding,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let Some(required_encoding) = expected.encoding else {
        return Ok(());
    };
    let required_encoding = match required_encoding {
        fluxel_material::MaterialTextureEncoding::Linear => TextureEncoding::Linear,
        fluxel_material::MaterialTextureEncoding::Srgb => TextureEncoding::Srgb,
        fluxel_material::MaterialTextureEncoding::SrgbEncodedUnorm => {
            TextureEncoding::SrgbEncodedUnorm
        }
    };
    if actual != required_encoding {
        return Err(ForwardPreparationError::TextureEncodingMismatch {
            source,
            resource: expected.name.clone(),
            expected: required_encoding,
            actual,
        });
    }
    Ok(())
}
