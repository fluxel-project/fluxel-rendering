//! Opt-in Babylon-style PBR factor and packed-data lowering.

use crate::{
    MaterialError, MaterialNodeInputSource, MaterialParameter, MaterialResource,
    MaterialResourceKind, MaterialTextureEncoding, MaterialTextureEncodingRequirement,
    MaterialType, PbrPackedChannel, PbrSurfaceFactors, SurfaceOutput,
};

use super::{canonical::piece as canonical_piece, node_graph::PBR_CONTEXT_INPUTS};

pub(super) fn validate(
    factors: Option<&PbrSurfaceFactors>,
    output: &SurfaceOutput,
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let Some(factors) = factors else {
        return Ok(());
    };
    let packed_occlusion = factors
        .packed_metallic_roughness
        .as_ref()
        .is_some_and(|packed| packed.occlusion.is_some());
    if !matches!(output, SurfaceOutput::MetallicRoughnessUv0Normal { .. })
        && (!matches!(output, SurfaceOutput::MetallicRoughnessUv0 { .. }) || packed_occlusion)
    {
        return Err(MaterialError::PbrSurfaceFactorsUnsupported {
            output: output.clone(),
        });
    }
    parameter(
        factors.base_factor_parameter.as_deref(),
        "base_factor_parameter",
        MaterialType::Vec4,
        parameters,
    )?;
    parameter(
        factors.metallic_factor_parameter.as_deref(),
        "metallic_factor_parameter",
        MaterialType::F32,
        parameters,
    )?;
    parameter(
        factors.roughness_factor_parameter.as_deref(),
        "roughness_factor_parameter",
        MaterialType::F32,
        parameters,
    )?;
    if let Some(texture) = &factors.base_texture {
        resource(
            "base_texture.texture",
            &texture.texture,
            MaterialResourceKind::Texture2dF32,
            resources,
        )?;
        resource(
            "base_texture.sampler",
            &texture.sampler,
            MaterialResourceKind::Sampler { comparison: false },
            resources,
        )?;
    }
    if let Some(packed) = &factors.packed_metallic_roughness {
        resource(
            "packed_metallic_roughness.texture",
            &packed.texture,
            MaterialResourceKind::Texture2dF32,
            resources,
        )?;
        resource(
            "packed_metallic_roughness.sampler",
            &packed.sampler,
            MaterialResourceKind::Sampler { comparison: false },
            resources,
        )?;
    }
    if factors.occlusion_strength_parameter.is_some() && !packed_occlusion {
        return Err(MaterialError::PbrOcclusionStrengthWithoutPackedChannel);
    }
    parameter(
        factors.occlusion_strength_parameter.as_deref(),
        "occlusion_strength_parameter",
        MaterialType::F32,
        parameters,
    )
}

pub(super) fn effective_texture_encodings(
    factors: Option<&PbrSurfaceFactors>,
    mut requirements: Vec<MaterialTextureEncodingRequirement>,
) -> Result<Vec<MaterialTextureEncodingRequirement>, MaterialError> {
    let Some(factors) = factors else {
        return Ok(requirements);
    };
    if let Some(base) = &factors.base_texture {
        require_encoding(
            &mut requirements,
            &base.texture,
            MaterialTextureEncoding::Srgb,
        )?;
    }
    if let Some(packed) = &factors.packed_metallic_roughness {
        require_encoding(
            &mut requirements,
            &packed.texture,
            MaterialTextureEncoding::Linear,
        )?;
    }
    Ok(requirements)
}

pub(super) fn canonical(factors: Option<&PbrSurfaceFactors>, target: &mut String) {
    let Some(factors) = factors else {
        return;
    };
    canonical_piece(target, "pbr-surface-factors", "enabled");
    optional(
        target,
        "pbr-base-factor",
        factors.base_factor_parameter.as_deref(),
    );
    optional(
        target,
        "pbr-metallic-factor",
        factors.metallic_factor_parameter.as_deref(),
    );
    optional(
        target,
        "pbr-roughness-factor",
        factors.roughness_factor_parameter.as_deref(),
    );
    optional(
        target,
        "pbr-occlusion-strength",
        factors.occlusion_strength_parameter.as_deref(),
    );
    if let Some(base) = &factors.base_texture {
        canonical_piece(target, "pbr-base-texture", &base.texture);
        canonical_piece(target, "pbr-base-sampler", &base.sampler);
        canonical_piece(target, "pbr-base-encoding", "srgb");
    }
    if let Some(packed) = &factors.packed_metallic_roughness {
        canonical_piece(target, "pbr-packed-texture", &packed.texture);
        canonical_piece(target, "pbr-packed-sampler", &packed.sampler);
        canonical_piece(target, "pbr-packed-encoding", "linear");
        canonical_piece(target, "pbr-packed-metallic", channel(packed.metallic));
        canonical_piece(target, "pbr-packed-roughness", channel(packed.roughness));
        if let Some(occlusion) = packed.occlusion {
            canonical_piece(target, "pbr-packed-occlusion", channel(occlusion));
        }
    }
}

pub(super) fn append_uv0_evaluators(
    generated: &mut String,
    root: &str,
    root_pbr_inputs: &[MaterialNodeInputSource],
    properties: &str,
    factors: Option<&PbrSurfaceFactors>,
) {
    let forbidden = factors.map_or_else(|| vec![root], |factors| factor_symbols(factors, [root]));
    let uv = if factors.is_some() {
        unique_local("pbr_uv", forbidden.iter().copied())
    } else {
        unique_local("uv", forbidden.iter().copied())
    };
    let mut parameters = format!("{uv}: vec2<f32>");
    let mut arguments = uv.clone();
    let mut context_parameters = String::new();
    for (input, seed, ty) in PBR_CONTEXT_INPUTS {
        let name = unique_local(seed, forbidden.iter().copied());
        let parameter = format!(", {name}: {}", ty.wgsl());
        context_parameters.push_str(&parameter);
        if root_pbr_inputs.contains(&input) {
            arguments.push_str(&format!(", {name}"));
        }
    }
    let name = if root_pbr_inputs.is_empty() {
        "evaluate_material_uv0"
    } else {
        parameters.push_str(&context_parameters);
        "evaluate_pbr_material_uv0"
    };
    let value = format!("{root}({arguments})");
    let value = if let Some(factors) = factors {
        generated.push_str(&uv0_properties_evaluator(properties, factors));
        let factor = factors
            .base_factor_parameter
            .as_deref()
            .unwrap_or("vec4<f32>(1.0)");
        let sample = factors.base_texture.as_ref().map_or_else(
            || String::from("vec4<f32>(1.0)"),
            |texture| {
                format!(
                    "textureSample({}, {}, {uv})",
                    texture.texture, texture.sampler
                )
            },
        );
        format!("{value} * {factor} * {sample}")
    } else {
        generated.push_str(&format!(
            "fn evaluate_metallic_roughness_uv0(uv: vec2<f32>) -> vec2<f32> {{\n    return {properties}(uv);\n}}\n"
        ));
        value
    };
    generated.push_str(&format!(
        "fn {name}({parameters}) -> vec4<f32> {{\n    return {value};\n}}\n"
    ));
    if root_pbr_inputs.is_empty() {
        generated.push_str(&format!(
            "fn evaluate_pbr_material_uv0({parameters}{context_parameters}) -> vec4<f32> {{\n    return evaluate_material_uv0({uv});\n}}\n"
        ));
    }
}

pub(super) fn uv0_properties_evaluator(properties: &str, factors: &PbrSurfaceFactors) -> String {
    let uv = unique_local("pbr_uv", factor_symbols(factors, [properties]));
    let metallic = factors
        .metallic_factor_parameter
        .as_deref()
        .unwrap_or("1.0");
    let roughness = factors
        .roughness_factor_parameter
        .as_deref()
        .unwrap_or("1.0");
    let Some(texture) = factors.packed_metallic_roughness.as_ref() else {
        return format!(
            "fn evaluate_metallic_roughness_uv0({uv}: vec2<f32>) -> vec2<f32> {{\n    return {properties}({uv}) * vec2<f32>({metallic}, {roughness});\n}}\n"
        );
    };
    let sample = unique_local(
        "pbr_packed_sample",
        factor_symbols(factors, [properties, uv.as_str()]),
    );
    format!(
        "fn evaluate_metallic_roughness_uv0({uv}: vec2<f32>) -> vec2<f32> {{\n    let {sample} = textureSample({}, {}, {uv});\n    return {properties}({uv}) * vec2<f32>({metallic}, {roughness}) * vec2<f32>({sample}.{}, {sample}.{});\n}}\n",
        texture.texture,
        texture.sampler,
        channel_expression(texture.metallic),
        channel_expression(texture.roughness),
    )
}

pub(super) fn uv0_occlusion_evaluator(
    authored: Option<&str>,
    factors: Option<&PbrSurfaceFactors>,
) -> String {
    let uv = factors.map_or_else(
        || unique_local("pbr_uv", authored.into_iter()),
        |factors| unique_local("pbr_uv", factor_symbols(factors, authored.into_iter())),
    );
    let previous = authored.map_or_else(|| String::from("1.0"), |name| format!("{name}({uv})"));
    let Some(packed) = factors.and_then(|factors| factors.packed_metallic_roughness.as_ref())
    else {
        return format!(
            "fn evaluate_occlusion_uv0({uv}: vec2<f32>) -> f32 {{\n    return {previous};\n}}\n"
        );
    };
    let Some(channel) = packed.occlusion else {
        return format!(
            "fn evaluate_occlusion_uv0({uv}: vec2<f32>) -> f32 {{\n    return {previous};\n}}\n"
        );
    };
    let strength = factors
        .and_then(|factors| factors.occlusion_strength_parameter.as_deref())
        .unwrap_or("1.0");
    format!(
        "fn evaluate_occlusion_uv0({uv}: vec2<f32>) -> f32 {{\n    return mix({previous}, textureSample({}, {}, {uv}).{}, {strength});\n}}\n",
        packed.texture,
        packed.sampler,
        channel_expression(channel),
    )
}

fn factor_symbols<'a>(
    factors: &'a PbrSurfaceFactors,
    extra: impl IntoIterator<Item = &'a str>,
) -> Vec<&'a str> {
    let mut symbols = extra.into_iter().collect::<Vec<_>>();
    symbols.extend(
        [
            factors.base_factor_parameter.as_deref(),
            factors.metallic_factor_parameter.as_deref(),
            factors.roughness_factor_parameter.as_deref(),
            factors.occlusion_strength_parameter.as_deref(),
        ]
        .into_iter()
        .flatten(),
    );
    if let Some(texture) = &factors.base_texture {
        symbols.extend([texture.texture.as_str(), texture.sampler.as_str()]);
    }
    if let Some(texture) = &factors.packed_metallic_roughness {
        symbols.extend([texture.texture.as_str(), texture.sampler.as_str()]);
    }
    symbols
}

fn unique_local<'a>(seed: &str, symbols: impl IntoIterator<Item = &'a str>) -> String {
    let symbols = symbols.into_iter().collect::<Vec<_>>();
    let mut name = seed.to_owned();
    while symbols.contains(&name.as_str()) {
        name.push('_');
    }
    name
}

fn parameter(
    name: Option<&str>,
    field: &'static str,
    expected: MaterialType,
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
    if parameter.ty != expected {
        return Err(MaterialError::PbrSurfaceFactorParameterTypeMismatch {
            field,
            parameter: name.into(),
            expected,
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

fn require_encoding(
    requirements: &mut Vec<MaterialTextureEncodingRequirement>,
    resource: &str,
    encoding: MaterialTextureEncoding,
) -> Result<(), MaterialError> {
    match requirements
        .iter()
        .find(|requirement| requirement.resource == resource)
    {
        Some(requirement) if requirement.encoding != encoding => {
            Err(MaterialError::PbrSurfaceFactorTextureEncodingConflict {
                resource: resource.into(),
            })
        }
        Some(_) => Ok(()),
        None => {
            requirements.push(MaterialTextureEncodingRequirement {
                resource: resource.into(),
                encoding,
            });
            Ok(())
        }
    }
}

fn optional(target: &mut String, label: &str, value: Option<&str>) {
    if let Some(value) = value {
        canonical_piece(target, label, value);
    }
}

fn channel(channel: PbrPackedChannel) -> &'static str {
    match channel {
        PbrPackedChannel::Red => "red",
        PbrPackedChannel::Green => "green",
        PbrPackedChannel::Blue => "blue",
        PbrPackedChannel::Alpha => "alpha",
    }
}

fn channel_expression(channel: PbrPackedChannel) -> &'static str {
    match channel {
        PbrPackedChannel::Red => "r",
        PbrPackedChannel::Green => "g",
        PbrPackedChannel::Blue => "b",
        PbrPackedChannel::Alpha => "a",
    }
}
