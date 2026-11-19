//! PBR opacity validation and lowering.
use super::canonical::piece;
use crate::{
    MaterialError, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialTextureEncoding, MaterialTextureEncodingRequirement, MaterialTextureUv, MaterialType,
    PbrOpacityChannels, PbrOpacityConfiguration, SurfaceOutput,
};
use std::collections::BTreeMap;
pub(super) fn validate(
    configuration: Option<&PbrOpacityConfiguration>,
    output: &SurfaceOutput,
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let Some(configuration) = configuration else {
        return Ok(());
    };
    if !super::surface::supplies_pbr_context(output) {
        return Err(MaterialError::PbrSurfaceFactorsUnsupported {
            output: output.clone(),
        });
    }
    for name in [
        configuration.alpha_parameter.as_deref(),
        configuration
            .texture
            .as_ref()
            .and_then(|value| value.level_parameter.as_deref()),
    ]
    .into_iter()
    .flatten()
    {
        let parameter = parameters
            .iter()
            .find(|parameter| parameter.name == name)
            .ok_or_else(|| MaterialError::PbrSurfaceFactorParameterMissing {
                field: "opacity",
                parameter: name.into(),
            })?;
        if parameter.ty != MaterialType::F32 {
            return Err(MaterialError::PbrSurfaceFactorParameterTypeMismatch {
                field: "opacity",
                parameter: name.into(),
                expected: MaterialType::F32,
                actual: parameter.ty,
            });
        }
    }
    if let Some(texture) = &configuration.texture {
        for (name, expected) in [
            (&texture.texture, MaterialResourceKind::Texture2dF32),
            (
                &texture.sampler,
                MaterialResourceKind::Sampler { comparison: false },
            ),
        ] {
            let resource = resources
                .iter()
                .find(|resource| resource.name == *name)
                .ok_or_else(|| MaterialError::PbrSurfaceFactorResourceMissing {
                    field: "opacity",
                    resource: name.clone(),
                })?;
            if resource.kind != expected {
                return Err(MaterialError::PbrSurfaceFactorResourceTypeMismatch {
                    field: "opacity",
                    resource: name.clone(),
                    expected,
                    actual: resource.kind,
                });
            }
        }
    }
    Ok(())
}
pub(super) fn canonical(configuration: Option<&PbrOpacityConfiguration>, target: &mut String) {
    let Some(value) = configuration else {
        return;
    };
    piece(target, "pbr-opacity", "enabled");
    if let Some(name) = &value.alpha_parameter {
        piece(target, "pbr-opacity-alpha", name);
    }
    if let Some(texture) = &value.texture {
        piece(target, "pbr-opacity-texture", &texture.texture);
        piece(target, "pbr-opacity-sampler", &texture.sampler);
        piece(
            target,
            "pbr-opacity-channels",
            match texture.channels {
                PbrOpacityChannels::Alpha => "alpha",
                PbrOpacityChannels::Luminance => "luminance",
            },
        );
        if let Some(name) = &texture.level_parameter {
            piece(target, "pbr-opacity-level", name);
        }
    }
}
pub(super) fn effective_encoding(
    configuration: Option<&PbrOpacityConfiguration>,
    mut requirements: Vec<MaterialTextureEncodingRequirement>,
) -> Result<Vec<MaterialTextureEncodingRequirement>, MaterialError> {
    let Some(texture) = configuration.and_then(|value| value.texture.as_ref()) else {
        return Ok(requirements);
    };
    if matches!(texture.channels, PbrOpacityChannels::Alpha) {
        return Ok(requirements);
    }
    match requirements
        .iter()
        .find(|value| value.resource == texture.texture)
    {
        Some(value) if value.encoding != MaterialTextureEncoding::Linear => {
            Err(MaterialError::PbrSurfaceFactorTextureEncodingConflict {
                resource: texture.texture.clone(),
            })
        }
        Some(_) => Ok(requirements),
        None => {
            requirements.push(MaterialTextureEncodingRequirement {
                resource: texture.texture.clone(),
                encoding: MaterialTextureEncoding::Linear,
            });
            Ok(requirements)
        }
    }
}
pub(super) fn append(
    generated: &mut String,
    configuration: &PbrOpacityConfiguration,
    value: &str,
    base_local: &str,
    alpha_local: &str,
    uv: &str,
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
    uv1: Option<&str>,
) {
    let alpha = configuration.alpha_parameter.as_deref().unwrap_or("1.0");
    generated.push_str(&format!(
        "    let {base_local} = {value};\n    var {alpha_local} = {base_local}.a * {alpha};\n"
    ));
    if let Some(map) = &configuration.texture {
        let sample_uv = super::pbr_factors::sample_uv(texture_uv, &map.texture, uv, uv1);
        let sample = format!(
            "textureSample({}, {}, {sample_uv})",
            map.texture, map.sampler
        );
        let operation = match map.channels {
            PbrOpacityChannels::Alpha => format!("*= {sample}.a"),
            PbrOpacityChannels::Luminance => {
                format!("= clamp(dot({sample}.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0)")
            }
        };
        let level = map.level_parameter.as_deref().unwrap_or("1.0");
        generated.push_str(&format!(
            "    {alpha_local} {operation};\n    {alpha_local} *= {level};\n"
        ));
    }
}
