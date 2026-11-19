//! Opt-in Babylon-style PBR factor and packed-data lowering.

use std::collections::BTreeMap;

use crate::{
    MaterialError, MaterialNodeInputSource, MaterialParameter, MaterialResource,
    MaterialResourceKind, MaterialTextureEncoding, MaterialTextureEncodingRequirement,
    MaterialTextureUv, MaterialType, PbrPackedChannel, PbrSurfaceFactors, SurfaceOutput,
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
    if !super::surface::supplies_pbr_context(output)
        || (packed_occlusion && !super::surface::supports_surface_effects(output))
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
    root_pbr_inputs: &[(MaterialNodeInputSource, MaterialType)],
    properties: Option<&str>,
    factors: Option<&PbrSurfaceFactors>,
    opacity: Option<&crate::PbrOpacityConfiguration>,
    vertex_varyings: &[(u32, MaterialType)],
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
) {
    let mut forbidden =
        factors.map_or_else(|| vec![root], |factors| factor_symbols(factors, [root]));
    if let Some(opacity) = opacity {
        forbidden.extend(opacity.alpha_parameter.as_deref());
        if let Some(map) = &opacity.texture {
            forbidden.extend([map.texture.as_str(), map.sampler.as_str()]);
            forbidden.extend(map.level_parameter.as_deref());
        }
    }
    let uv = if factors.is_some() {
        unique_local("pbr_uv", forbidden.iter().copied())
    } else {
        unique_local("uv", forbidden.iter().copied())
    };
    let uv1 = uv1_name(texture_uv, forbidden.iter().copied());
    let prefix = uv1_parameter(uv1.as_deref());
    let mut parameters = format!("{prefix}{uv}: vec2<f32>");
    let mut arguments = uv.clone();
    let mut full_arguments = format!("{}{uv}", uv1_argument(uv1.as_deref()));
    for (input, seed, ty) in PBR_CONTEXT_INPUTS {
        let name = unique_local(seed, forbidden.iter().copied());
        parameters.push_str(&format!(", {name}: {}", ty.wgsl()));
        full_arguments.push_str(&format!(", {name}"));
        if root_pbr_inputs.iter().any(|(source, _)| *source == input) {
            arguments.push_str(&format!(", {name}"));
        }
    }
    for &(id, ty) in vertex_varyings {
        let source = MaterialNodeInputSource::VertexVarying(id);
        let name = unique_local(&format!("varying_{id}"), forbidden.iter().copied());
        parameters.push_str(&format!(", {name}: {}", ty.wgsl()));
        full_arguments.push_str(&format!(", {name}"));
        if root_pbr_inputs.iter().any(|(input, _)| *input == source) {
            arguments.push_str(&format!(", {name}"));
        }
    }

    let raw = format!("{root}({arguments})");
    if let Some(properties) = properties {
        append_legacy_uv0(
            generated,
            &raw,
            &uv,
            &parameters,
            root_pbr_inputs.is_empty(),
            properties,
            factors,
            texture_uv,
            uv1.as_deref(),
        );
    }
    generated.push_str("struct FluxelPbrSurface { base: vec4<f32>, properties: vec2<f32>, world_normal: vec3<f32>, has_world_normal: bool, occlusion: f32 };\n");
    generated.push_str(&format!(
        "fn evaluate_pbr_surface_uv0({parameters}) -> FluxelPbrSurface {{\n"
    ));
    let (base, properties, world_normal, has_world_normal, occlusion) = if properties.is_some() {
        (
            format!("evaluate_pbr_material_uv0({full_arguments})"),
            format!(
                "evaluate_metallic_roughness_uv0({}{uv})",
                uv1_argument(uv1.as_deref())
            ),
            "vec3<f32>(0.0)".into(),
            "false".into(),
            "1.0".into(),
        )
    } else {
        let values = unique_local("pbr_graph_values", forbidden.iter().copied());
        generated.push_str(&format!("    let {values} = {raw};\n"));
        let base = apply_base_factors(
            &format!("{values}.base"),
            &uv,
            factors,
            texture_uv,
            uv1.as_deref(),
        );
        let properties = apply_property_factors(
            generated,
            format!("{values}.properties"),
            &uv,
            factors,
            forbidden.iter().copied(),
            texture_uv,
            uv1.as_deref(),
        );
        (
            base,
            properties,
            format!("{values}.world_normal"),
            format!("{values}.has_world_normal"),
            format!("{values}.occlusion"),
        )
    };
    let base = if let Some(opacity) = opacity {
        let base_local = unique_local("pbr_opacity_base", forbidden.iter().copied());
        let alpha_local = unique_local("pbr_opacity_alpha", forbidden.iter().copied());
        super::opacity::append(
            generated,
            opacity,
            &base,
            &base_local,
            &alpha_local,
            &uv,
            texture_uv,
            uv1.as_deref(),
        );
        format!("vec4<f32>({base_local}.rgb, {alpha_local})")
    } else {
        base
    };
    generated.push_str(&format!(
        "    return FluxelPbrSurface({base}, {properties}, {world_normal}, {has_world_normal}, {occlusion});\n}}\n"
    ));
}

fn append_legacy_uv0(
    generated: &mut String,
    raw: &str,
    uv: &str,
    parameters: &str,
    is_uv_only: bool,
    properties: &str,
    factors: Option<&PbrSurfaceFactors>,
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
    base_uv1: Option<&str>,
) {
    let property_symbols = factors.map_or_else(
        || vec![properties],
        |factors| factor_symbols(factors, [properties].into_iter()),
    );
    let property_uv = unique_local("uv", property_symbols.iter().copied());
    let property_uv1 = uv1_name(texture_uv, property_symbols.iter().copied());
    let property_prefix = uv1_parameter(property_uv1.as_deref());
    generated.push_str(&format!(
        "fn evaluate_metallic_roughness_uv0({property_prefix}{property_uv}: vec2<f32>) -> vec2<f32> {{\n"
    ));
    let properties = apply_property_factors(
        generated,
        format!("{properties}({property_uv})"),
        &property_uv,
        factors,
        property_symbols.iter().copied(),
        texture_uv,
        property_uv1.as_deref(),
    );
    generated.push_str(&format!("    return {properties};\n}}\n"));

    let base = apply_base_factors(raw, uv, factors, texture_uv, base_uv1);
    let base_prefix = uv1_parameter(base_uv1);
    if is_uv_only {
        generated.push_str(&format!(
            "fn evaluate_material_uv0({base_prefix}{uv}: vec2<f32>) -> vec4<f32> {{\n    return {base};\n}}\n"
        ));
    }
    let base = if is_uv_only {
        format!("evaluate_material_uv0({}{uv})", uv1_argument(base_uv1))
    } else {
        base
    };
    generated.push_str(&format!(
        "fn evaluate_pbr_material_uv0({parameters}) -> vec4<f32> {{\n    return {base};\n}}\n"
    ));
}

fn apply_base_factors(
    value: &str,
    uv: &str,
    factors: Option<&PbrSurfaceFactors>,
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
    uv1: Option<&str>,
) -> String {
    let Some(factors) = factors else {
        return value.into();
    };
    let factor = factors
        .base_factor_parameter
        .as_deref()
        .unwrap_or("vec4<f32>(1.0)");
    let sample = factors.base_texture.as_ref().map_or_else(
        || String::from("vec4<f32>(1.0)"),
        |texture| {
            let uv = sample_uv(texture_uv, &texture.texture, uv, uv1);
            format!(
                "textureSample({}, {}, {uv})",
                texture.texture, texture.sampler
            )
        },
    );
    format!("{value} * {factor} * {sample}")
}

fn apply_property_factors<'a>(
    generated: &mut String,
    value: String,
    uv: &str,
    factors: Option<&PbrSurfaceFactors>,
    forbidden: impl IntoIterator<Item = &'a str>,
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
    uv1: Option<&str>,
) -> String {
    let Some(factors) = factors else {
        return value;
    };
    let metallic = factors
        .metallic_factor_parameter
        .as_deref()
        .unwrap_or("1.0");
    let roughness = factors
        .roughness_factor_parameter
        .as_deref()
        .unwrap_or("1.0");
    let mut value = format!("{value} * vec2<f32>({metallic}, {roughness})");
    if let Some(texture) = &factors.packed_metallic_roughness {
        let sample = unique_local("pbr_packed_sample", forbidden);
        let uv = sample_uv(texture_uv, &texture.texture, uv, uv1);
        generated.push_str(&format!(
            "    let {sample} = textureSample({}, {}, {uv});\n",
            texture.texture, texture.sampler
        ));
        value.push_str(&format!(
            " * vec2<f32>({sample}.{}, {sample}.{})",
            channel_expression(texture.metallic),
            channel_expression(texture.roughness)
        ));
    }
    value
}

pub(super) fn uv0_occlusion_evaluator(
    authored: Option<&str>,
    factors: Option<&PbrSurfaceFactors>,
    texture_uv: &BTreeMap<String, MaterialTextureUv>,
) -> String {
    let names = factors.map_or_else(
        || authored.into_iter().collect(),
        |factors| factor_symbols(factors, authored.into_iter()),
    );
    let uv = unique_local("pbr_uv", names.iter().copied());
    let uv1 = uv1_name(texture_uv, names.iter().copied());
    let prefix = uv1_parameter(uv1.as_deref());
    let previous = authored.map_or_else(|| String::from("1.0"), |name| format!("{name}({uv})"));
    let Some(packed) = factors.and_then(|factors| factors.packed_metallic_roughness.as_ref())
    else {
        return format!(
            "fn evaluate_occlusion_uv0({prefix}{uv}: vec2<f32>) -> f32 {{\n    return {previous};\n}}\n"
        );
    };
    let Some(channel) = packed.occlusion else {
        return format!(
            "fn evaluate_occlusion_uv0({prefix}{uv}: vec2<f32>) -> f32 {{\n    return {previous};\n}}\n"
        );
    };
    let strength = factors
        .and_then(|factors| factors.occlusion_strength_parameter.as_deref())
        .unwrap_or("1.0");
    let sample_uv = sample_uv(texture_uv, &packed.texture, &uv, uv1.as_deref());
    format!(
        "fn evaluate_occlusion_uv0({prefix}{uv}: vec2<f32>) -> f32 {{\n    return mix({previous}, textureSample({}, {}, {sample_uv}).{}, {strength});\n}}\n",
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

pub(super) fn uv1_name<'a>(
    coordinates: &BTreeMap<String, MaterialTextureUv>,
    symbols: impl IntoIterator<Item = &'a str>,
) -> Option<String> {
    coordinates
        .values()
        .any(|value| value.set() == crate::TextureUvSet::Uv1)
        .then(|| unique_local("fluxel_uv1", symbols))
}

pub(super) fn uv1_parameter(name: Option<&str>) -> String {
    name.map_or_else(String::new, |name| format!("{name}: vec2<f32>, "))
}

fn uv1_argument(name: Option<&str>) -> String {
    name.map_or_else(String::new, |name| format!("{name}, "))
}

/// Babylon multiplies the selected UV and a unit affine coordinate by the
/// texture matrix. These columns are the XY projection of that operation.
pub(super) fn sample_uv(
    coordinates: &BTreeMap<String, MaterialTextureUv>,
    resource: &str,
    uv0: &str,
    uv1: Option<&str>,
) -> String {
    let Some(coordinates) = coordinates.get(resource) else {
        return uv0.into();
    };
    let uv = match coordinates.set() {
        crate::TextureUvSet::Uv0 => uv0,
        crate::TextureUvSet::Uv1 => uv1.expect("UV1 sampling has a corresponding parameter"),
    };
    let [[xu, xv], [yu, yv], [tu, tv]] = coordinates.columns();
    format!(
        "(mat3x2<f32>(vec2<f32>({xu:?}, {xv:?}), vec2<f32>({yu:?}, {yv:?}), vec2<f32>({tu:?}, {tv:?})) * vec3<f32>({uv}, 1.0))"
    )
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
