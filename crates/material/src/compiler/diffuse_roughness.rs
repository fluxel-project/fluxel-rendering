//! Babylon-style independent diffuse-roughness lowering.

use std::collections::BTreeMap;

use crate::{
    DiffuseRoughnessConfiguration, MaterialError, MaterialParameter, MaterialResource,
    MaterialResourceKind, MaterialTextureEncoding, MaterialTextureEncodingRequirement,
    MaterialTextureUv, MaterialType, SurfaceOutput,
};

use super::canonical::piece;

pub(super) fn validate(
    configuration: Option<&DiffuseRoughnessConfiguration>,
    output: &SurfaceOutput,
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let Some(configuration) = configuration else {
        return Ok(());
    };
    if !is_pbr(output) || (configuration.texture.is_some() && !super::surface::uses_uv0(output)) {
        return Err(MaterialError::DiffuseRoughnessUnsupported {
            output: output.clone(),
        });
    }
    let Some(texture) = &configuration.texture else {
        return Ok(());
    };
    parameter(
        texture.level_parameter.as_deref(),
        "texture.level_parameter",
        parameters,
    )?;
    resource(
        "texture.texture",
        &texture.texture,
        MaterialResourceKind::Texture2dF32,
        resources,
    )?;
    resource(
        "texture.sampler",
        &texture.sampler,
        MaterialResourceKind::Sampler { comparison: false },
        resources,
    )
}

pub(super) fn effective_texture_encodings(
    configuration: Option<&DiffuseRoughnessConfiguration>,
    mut requirements: Vec<MaterialTextureEncodingRequirement>,
) -> Result<Vec<MaterialTextureEncodingRequirement>, MaterialError> {
    let Some(texture) = configuration.and_then(|configuration| configuration.texture.as_ref())
    else {
        return Ok(requirements);
    };
    match requirements
        .iter()
        .find(|requirement| requirement.resource == texture.texture)
    {
        Some(requirement) if requirement.encoding != MaterialTextureEncoding::Linear => {
            Err(MaterialError::DiffuseRoughnessTextureEncodingConflict {
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

pub(super) fn canonical(
    configuration: Option<&DiffuseRoughnessConfiguration>,
    target: &mut String,
) {
    let Some(configuration) = configuration else {
        return;
    };
    piece(target, "diffuse-roughness", "enabled");
    piece(
        target,
        "diffuse-roughness-base-bits",
        &configuration.base_roughness().to_bits().to_string(),
    );
    if let Some(texture) = &configuration.texture {
        piece(target, "diffuse-roughness-texture", &texture.texture);
        piece(target, "diffuse-roughness-sampler", &texture.sampler);
        if let Some(level) = &texture.level_parameter {
            piece(target, "diffuse-roughness-level", level);
        }
        piece(target, "diffuse-roughness-encoding", "linear");
    }
}

pub(super) fn append_evaluator(
    generated: &mut String,
    configuration: Option<&DiffuseRoughnessConfiguration>,
    output: &SurfaceOutput,
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
) {
    let Some(configuration) = configuration else {
        return;
    };
    let base = configuration.base_roughness();
    let Some(texture) = &configuration.texture else {
        if super::surface::uses_uv0(output) {
            let uv1 = super::pbr_factors::uv1_name(texture_uv, []);
            let prefix = super::pbr_factors::uv1_parameter(uv1.as_deref());
            generated.push_str(&format!(
                "fn evaluate_diffuse_roughness_uv0({prefix}uv: vec2<f32>) -> f32 {{ return {base:?}; }}\n"
            ));
        } else {
            generated.push_str(&format!(
                "fn evaluate_diffuse_roughness() -> f32 {{ return {base:?}; }}\n"
            ));
        }
        return;
    };
    debug_assert!(super::surface::uses_uv0(output));
    let bindings = [
        texture.texture.as_str(),
        texture.sampler.as_str(),
        texture.level_parameter.as_deref().unwrap_or("1.0"),
    ];
    let uv = unique_local("diffuse_roughness_uv", &bindings);
    let uv1 = super::pbr_factors::uv1_name(texture_uv, bindings);
    let prefix = super::pbr_factors::uv1_parameter(uv1.as_deref());
    let level = texture.level_parameter.as_deref().unwrap_or("1.0");
    let sample_uv =
        super::pbr_factors::sample_uv(texture_uv, &texture.texture, &uv, uv1.as_deref());
    generated.push_str(&format!(
        "fn evaluate_diffuse_roughness_uv0({prefix}{uv}: vec2<f32>) -> f32 {{\n    return {base:?} * (textureSample({}, {}, {sample_uv}).r * {level});\n}}\n",
        texture.texture, texture.sampler
    ));
}

fn is_pbr(output: &SurfaceOutput) -> bool {
    matches!(
        output,
        SurfaceOutput::MetallicRoughness { .. }
            | SurfaceOutput::MetallicRoughnessUv0 { .. }
            | SurfaceOutput::MetallicRoughnessUv0Normal { .. }
            | SurfaceOutput::MetallicRoughnessUv0Graph { .. }
    )
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
        .ok_or_else(|| MaterialError::DiffuseRoughnessParameterMissing {
            field,
            parameter: name.into(),
        })?;
    if parameter.ty != MaterialType::F32 {
        return Err(MaterialError::DiffuseRoughnessParameterTypeMismatch {
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
        .ok_or_else(|| MaterialError::DiffuseRoughnessResourceMissing {
            field,
            resource: name.into(),
        })?;
    if resource.kind != expected {
        return Err(MaterialError::DiffuseRoughnessResourceTypeMismatch {
            field,
            resource: name.into(),
            expected,
            actual: resource.kind,
        });
    }
    Ok(())
}

fn unique_local(seed: &str, bindings: &[&str]) -> String {
    let mut name = seed.to_owned();
    while bindings.contains(&name.as_str()) {
        name.push('_');
    }
    name
}
