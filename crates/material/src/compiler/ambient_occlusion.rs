//! Babylon-style colored ambient-occlusion lowering.

use std::collections::BTreeMap;

use crate::{
    AmbientOcclusionConfiguration, AmbientOcclusionMapChannels, MaterialError, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialTextureEncoding,
    MaterialTextureEncodingRequirement, MaterialTextureUv, MaterialType, PbrPackedChannel,
    PbrSurfaceFactors, SurfaceExtensions, SurfaceOutput,
};

use super::canonical::piece;

pub(super) fn validate(
    configuration: Option<&AmbientOcclusionConfiguration>,
    output: &SurfaceOutput,
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let Some(configuration) = configuration else {
        return Ok(());
    };
    if !super::surface::supplies_pbr_context(output) {
        return Err(MaterialError::AmbientOcclusionUnsupported {
            output: output.clone(),
        });
    }
    parameter(
        configuration.map.level_parameter.as_deref(),
        "map.level_parameter",
        parameters,
    )?;
    parameter(
        configuration.strength_parameter.as_deref(),
        "strength_parameter",
        parameters,
    )?;
    parameter(
        configuration.direct_weight_parameter.as_deref(),
        "direct_weight_parameter",
        parameters,
    )?;
    resource(
        "map.texture",
        &configuration.map.texture,
        MaterialResourceKind::Texture2dF32,
        resources,
    )?;
    resource(
        "map.sampler",
        &configuration.map.sampler,
        MaterialResourceKind::Sampler { comparison: false },
        resources,
    )
}

pub(super) fn effective_texture_encodings(
    configuration: Option<&AmbientOcclusionConfiguration>,
    mut requirements: Vec<MaterialTextureEncodingRequirement>,
) -> Result<Vec<MaterialTextureEncodingRequirement>, MaterialError> {
    let Some(configuration) = configuration else {
        return Ok(requirements);
    };
    match requirements
        .iter()
        .find(|requirement| requirement.resource == configuration.map.texture)
    {
        Some(requirement) if requirement.encoding != MaterialTextureEncoding::Linear => {
            Err(MaterialError::PbrSurfaceFactorTextureEncodingConflict {
                resource: configuration.map.texture.clone(),
            })
        }
        Some(_) => Ok(requirements),
        None => {
            requirements.push(MaterialTextureEncodingRequirement {
                resource: configuration.map.texture.clone(),
                encoding: MaterialTextureEncoding::Linear,
            });
            Ok(requirements)
        }
    }
}

pub(super) fn canonical(
    configuration: Option<&AmbientOcclusionConfiguration>,
    target: &mut String,
) {
    let Some(configuration) = configuration else {
        return;
    };
    piece(target, "ambient-occlusion", "enabled");
    piece(target, "ambient-occlusion-map", &configuration.map.texture);
    piece(
        target,
        "ambient-occlusion-sampler",
        &configuration.map.sampler,
    );
    optional(
        target,
        "ambient-occlusion-level",
        configuration.map.level_parameter.as_deref(),
    );
    optional(
        target,
        "ambient-occlusion-strength",
        configuration.strength_parameter.as_deref(),
    );
    optional(
        target,
        "ambient-occlusion-direct-weight",
        configuration.direct_weight_parameter.as_deref(),
    );
    piece(
        target,
        "ambient-occlusion-map-channels",
        match configuration.map.channels {
            AmbientOcclusionMapChannels::Red => "red",
            AmbientOcclusionMapChannels::Rgb => "rgb",
        },
    );
    piece(target, "ambient-occlusion-encoding", "linear");
}

pub(super) fn append_evaluator(
    generated: &mut String,
    configuration: Option<&AmbientOcclusionConfiguration>,
    extensions: &SurfaceExtensions,
    factors: Option<&PbrSurfaceFactors>,
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
) {
    let Some(configuration) = configuration else {
        return;
    };
    let level = configuration
        .map
        .level_parameter
        .as_deref()
        .unwrap_or("1.0");
    let strength = configuration.strength_parameter.as_deref().unwrap_or("1.0");
    let direct_weight = configuration
        .direct_weight_parameter
        .as_deref()
        .unwrap_or("0.0");
    let mut bindings = vec![
        configuration.map.texture.as_str(),
        configuration.map.sampler.as_str(),
        level,
        strength,
        direct_weight,
    ];
    let packed_occlusion = factors.and_then(|factors| {
        factors
            .packed_metallic_roughness
            .as_ref()
            .and_then(|packed| {
                packed.occlusion.map(|channel| {
                    (
                        packed,
                        channel,
                        factors
                            .occlusion_strength_parameter
                            .as_deref()
                            .unwrap_or(strength),
                    )
                })
            })
    });
    if let Some((packed, _, packed_strength)) = packed_occlusion {
        bindings.extend([
            packed.texture.as_str(),
            packed.sampler.as_str(),
            packed_strength,
        ]);
    }
    if let Some(legacy) = extensions.occlusion.as_deref() {
        bindings.push(legacy);
    }
    let authored_parameter = unique_local("authored_occlusion", &bindings);
    bindings.push(authored_parameter.as_str());
    let uv = unique_local("ambient_occlusion_uv", &bindings);
    let uv1 = super::pbr_factors::uv1_name(texture_uv, bindings.iter().copied());
    let prefix = super::pbr_factors::uv1_parameter(uv1.as_deref());
    let map_uv =
        super::pbr_factors::sample_uv(texture_uv, &configuration.map.texture, &uv, uv1.as_deref());
    let sampled = match configuration.map.channels {
        AmbientOcclusionMapChannels::Red => format!(
            "vec3<f32>(textureSample({}, {}, {map_uv}).r)",
            configuration.map.texture, configuration.map.sampler
        ),
        AmbientOcclusionMapChannels::Rgb => format!(
            "textureSample({}, {}, {map_uv}).rgb",
            configuration.map.texture, configuration.map.sampler
        ),
    };
    let authored = extensions.occlusion.as_deref().map_or_else(
        || authored_parameter.clone(),
        |legacy| format!("{authored_parameter} * {legacy}({uv})"),
    );
    let mut color = format!("mix(vec3<f32>({authored}), {sampled} * {level}, {strength})");
    if let Some((packed, channel, packed_strength)) = packed_occlusion {
        let packed_uv =
            super::pbr_factors::sample_uv(texture_uv, &packed.texture, &uv, uv1.as_deref());
        color = format!(
            "mix({color}, vec3<f32>(textureSample({}, {}, {packed_uv}).{}), {packed_strength})",
            packed.texture,
            packed.sampler,
            channel_expression(channel)
        );
    }
    let monochrome = match configuration.map.channels {
        AmbientOcclusionMapChannels::Red => "ao_rgb.r".to_owned(),
        AmbientOcclusionMapChannels::Rgb => {
            "clamp(dot(ao_rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0)".to_owned()
        }
    };
    generated.push_str(&format!("fn evaluate_ambient_occlusion_uv0({prefix}{uv}: vec2<f32>, {authored_parameter}: f32) -> vec4<f32> {{\n    return vec4<f32>({color}, {direct_weight});\n}}\nfn evaluate_ambient_occlusion_monochrome(ao_rgb: vec3<f32>) -> f32 {{ return {monochrome}; }}\n"));
}

fn parameter(
    name: Option<&str>,
    field: &'static str,
    parameters: &[MaterialParameter],
) -> Result<(), MaterialError> {
    let Some(name) = name else {
        return Ok(());
    };
    let parameter = parameters
        .iter()
        .find(|parameter| parameter.name == name)
        .ok_or_else(|| MaterialError::PbrSurfaceFactorParameterMissing {
            field,
            parameter: name.into(),
        })?;
    if parameter.ty != MaterialType::F32 {
        return Err(MaterialError::PbrSurfaceFactorParameterTypeMismatch {
            field,
            parameter: name.into(),
            expected: MaterialType::F32,
            actual: parameter.ty,
        });
    }
    Ok(())
}

fn resource(
    field: &'static str,
    name: &str,
    expected: MaterialResourceKind,
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let resource = resources
        .iter()
        .find(|resource| resource.name == name)
        .ok_or_else(|| MaterialError::PbrSurfaceFactorResourceMissing {
            field,
            resource: name.into(),
        })?;
    if resource.kind != expected {
        return Err(MaterialError::PbrSurfaceFactorResourceTypeMismatch {
            field,
            resource: name.into(),
            expected,
            actual: resource.kind,
        });
    }
    Ok(())
}

fn channel_expression(channel: PbrPackedChannel) -> &'static str {
    match channel {
        PbrPackedChannel::Red => "r",
        PbrPackedChannel::Green => "g",
        PbrPackedChannel::Blue => "b",
        PbrPackedChannel::Alpha => "a",
    }
}
fn optional(target: &mut String, label: &str, value: Option<&str>) {
    if let Some(value) = value {
        piece(target, label, value);
    }
}
fn unique_local(seed: &str, bindings: &[&str]) -> String {
    let mut name = seed.to_owned();
    while bindings.contains(&name.as_str()) {
        name.push('_');
    }
    name
}
