//! Surface-domain validation and generated evaluation entry wrappers.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    validation::FunctionSignature, EmissiveConfiguration, MaterialDefinition, MaterialError,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, PbrSurfaceFactors,
    SurfaceAlphaMode, SurfaceExtensions, SurfaceOutput, SurfacePolicy,
};

/// Validates all surface-owned extensions after the root callable is resolved.
pub(super) fn validate_definition(
    definition: &MaterialDefinition,
    root: &FunctionSignature,
    functions: &BTreeMap<String, FunctionSignature>,
    root_name: &str,
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    validate(
        definition.output(),
        definition.surface_extensions(),
        definition.emissive_configuration(),
        definition.pbr_surface_factors(),
        root,
        functions,
        root_name,
        parameters,
        resources,
    )?;
    super::transmission::validate(
        definition.output(),
        definition.transmission_extensions(),
        definition.refraction_environment(),
        functions,
        resources,
    )
}

pub(super) fn validate(
    output: &SurfaceOutput,
    extensions: &SurfaceExtensions,
    emissive_configuration: Option<&EmissiveConfiguration>,
    pbr_surface_factors: Option<&PbrSurfaceFactors>,
    root: &FunctionSignature,
    functions: &BTreeMap<String, FunctionSignature>,
    root_name: &str,
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    if (extensions.occlusion.is_some()
        || extensions.emissive.is_some()
        || emissive_configuration.is_some())
        && !matches!(output, SurfaceOutput::MetallicRoughnessUv0Normal { .. })
    {
        return Err(MaterialError::SurfaceExtensionsUnsupported {
            output: output.clone(),
        });
    }
    super::pbr_factors::validate(pbr_surface_factors, output, parameters, resources)?;
    if root.output != MaterialType::Vec4 {
        return Err(MaterialError::UnsupportedOutput {
            output: output.clone(),
        });
    }
    let expects_uv0 = uses_uv0(output);
    if !has_evaluator_inputs(root, expects_uv0) {
        return Err(MaterialError::RootInputsUnsupported {
            fragment: root_name.into(),
        });
    }
    let (properties, properties_expect_uv0, normal) = match output {
        SurfaceOutput::Color | SurfaceOutput::ColorUv0 => return Ok(()),
        SurfaceOutput::MetallicRoughness { properties } => (properties, false, None),
        SurfaceOutput::MetallicRoughnessUv0 { properties } => (properties, true, None),
        SurfaceOutput::MetallicRoughnessUv0Normal { properties, normal } => {
            (properties, true, Some(normal))
        }
    };
    let reserved = reserved_names(output);
    for name in [root_name, properties.as_str()]
        .into_iter()
        .chain(normal.iter().map(|name| name.as_str()))
    {
        if reserved.contains(name) {
            return Err(MaterialError::TopLevelNameConflict { name: name.into() });
        }
    }
    let properties_signature =
        functions
            .get(properties)
            .ok_or_else(|| MaterialError::MissingSurfaceProperties {
                properties: properties.clone(),
            })?;
    if properties_signature.output != MaterialType::Vec2 {
        return Err(MaterialError::SurfacePropertiesTypeMismatch {
            properties: properties.clone(),
            actual: properties_signature.output,
        });
    }
    if !has_evaluator_inputs(properties_signature, properties_expect_uv0) {
        return Err(MaterialError::SurfacePropertiesInputsUnsupported {
            properties: properties.clone(),
        });
    }
    if let Some(normal) = normal {
        let normal_signature =
            functions
                .get(normal)
                .ok_or_else(|| MaterialError::MissingSurfaceNormal {
                    normal: normal.clone(),
                })?;
        if normal_signature.output != MaterialType::Vec3 {
            return Err(MaterialError::SurfaceNormalTypeMismatch {
                normal: normal.clone(),
                actual: normal_signature.output,
            });
        }
        if !has_evaluator_inputs(normal_signature, true) {
            return Err(MaterialError::SurfaceNormalInputsUnsupported {
                normal: normal.clone(),
            });
        }
    }
    validate_extension(
        extensions.occlusion.as_deref(),
        MaterialType::F32,
        &reserved,
        functions,
    )?;
    validate_extension(
        extensions.emissive.as_deref(),
        MaterialType::Vec3,
        &reserved,
        functions,
    )?;
    if extensions.emissive.is_some() && emissive_configuration.is_some() {
        return Err(MaterialError::EmissiveConfigurationConflict);
    }
    if let Some(configuration) = emissive_configuration {
        validate_emissive_configuration(configuration, parameters, resources)?;
    }
    Ok(())
}

fn validate_emissive_configuration(
    configuration: &EmissiveConfiguration,
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    validate_emissive_parameter(
        configuration.color_parameter.as_deref(),
        "color_parameter",
        MaterialType::Vec3,
        parameters,
    )?;
    validate_emissive_parameter(
        configuration.intensity_parameter.as_deref(),
        "intensity_parameter",
        MaterialType::F32,
        parameters,
    )?;
    let Some(texture) = &configuration.texture else {
        return Ok(());
    };
    validate_emissive_parameter(
        texture.level_parameter.as_deref(),
        "texture.level_parameter",
        MaterialType::F32,
        parameters,
    )?;
    validate_emissive_resource(
        "texture.texture",
        &texture.texture,
        MaterialResourceKind::Texture2dF32,
        resources,
    )?;
    validate_emissive_resource(
        "texture.sampler",
        &texture.sampler,
        MaterialResourceKind::Sampler { comparison: false },
        resources,
    )
}

fn validate_emissive_parameter(
    name: Option<&str>,
    control: &'static str,
    expected: MaterialType,
    parameters: &[MaterialParameter],
) -> Result<(), MaterialError> {
    let Some(name) = name else {
        return Ok(());
    };
    let parameter = parameters
        .iter()
        .find(|parameter| parameter.name == name)
        .ok_or_else(|| MaterialError::EmissiveParameterMissing {
            control,
            parameter: name.into(),
        })?;
    if parameter.ty != expected {
        return Err(MaterialError::EmissiveParameterTypeMismatch {
            control,
            parameter: name.into(),
            expected,
            actual: parameter.ty,
        });
    }
    Ok(())
}

fn validate_emissive_resource(
    field: &'static str,
    name: &str,
    expected: MaterialResourceKind,
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let resource = resources
        .iter()
        .find(|resource| resource.name == name)
        .ok_or_else(|| MaterialError::EmissiveResourceMissing {
            field,
            resource: name.into(),
        })?;
    if resource.kind != expected {
        return Err(MaterialError::EmissiveResourceTypeMismatch {
            field,
            resource: name.into(),
            expected,
            actual: resource.kind,
        });
    }
    Ok(())
}

fn validate_extension(
    name: Option<&str>,
    expected: MaterialType,
    reserved: &BTreeSet<String>,
    functions: &BTreeMap<String, FunctionSignature>,
) -> Result<(), MaterialError> {
    let Some(name) = name else {
        return Ok(());
    };
    if reserved.contains(name) {
        return Err(MaterialError::TopLevelNameConflict { name: name.into() });
    }
    let signature = functions
        .get(name)
        .ok_or_else(|| MaterialError::MissingSurfaceExtension { name: name.into() })?;
    if signature.output != expected {
        return Err(MaterialError::SurfaceExtensionTypeMismatch {
            name: name.into(),
            actual: signature.output,
        });
    }
    if !has_evaluator_inputs(signature, true) {
        return Err(MaterialError::SurfaceExtensionInputsUnsupported { name: name.into() });
    }
    Ok(())
}

pub(super) fn uses_uv0(output: &SurfaceOutput) -> bool {
    matches!(
        output,
        SurfaceOutput::ColorUv0
            | SurfaceOutput::MetallicRoughnessUv0 { .. }
            | SurfaceOutput::MetallicRoughnessUv0Normal { .. }
    )
}

fn has_evaluator_inputs(signature: &FunctionSignature, expects_uv0: bool) -> bool {
    match (expects_uv0, signature.inputs.as_slice()) {
        (false, []) => true,
        (true, [input]) => input.ty == MaterialType::Vec2,
        _ => false,
    }
}

pub(super) fn reserved_names(output: &SurfaceOutput) -> BTreeSet<String> {
    let mut names = match output {
        SurfaceOutput::Color => BTreeSet::from([String::from("evaluate_material")]),
        SurfaceOutput::ColorUv0 => BTreeSet::from([String::from("evaluate_material_uv0")]),
        SurfaceOutput::MetallicRoughness { .. } => BTreeSet::from([
            String::from("evaluate_material"),
            String::from("evaluate_metallic_roughness"),
        ]),
        SurfaceOutput::MetallicRoughnessUv0 { .. } => BTreeSet::from([
            String::from("evaluate_material_uv0"),
            String::from("evaluate_metallic_roughness_uv0"),
        ]),
        SurfaceOutput::MetallicRoughnessUv0Normal { .. } => BTreeSet::from([
            String::from("evaluate_material_uv0"),
            String::from("evaluate_metallic_roughness_uv0"),
            String::from("evaluate_normal_uv0"),
            String::from("evaluate_occlusion_uv0"),
            String::from("evaluate_emissive_uv0"),
        ]),
    };
    names.extend([
        String::from("material_alpha_rejected"),
        String::from("material_output_alpha"),
        String::from("material_two_sided_lighting"),
        String::from("evaluate_transmission_uv0"),
        String::from("evaluate_thickness_uv0"),
        String::from("material_surface_ior"),
        String::from("material_volume_ior"),
        String::from("material_refraction_has_thickness"),
        String::from("material_attenuation_color"),
        String::from("material_attenuation_distance"),
    ]);
    names
}

pub(super) fn canonical_extensions(
    extensions: &SurfaceExtensions,
    emissive_configuration: Option<&EmissiveConfiguration>,
    pbr_surface_factors: Option<&PbrSurfaceFactors>,
    target: &mut String,
) {
    if let Some(occlusion) = &extensions.occlusion {
        canonical_piece(target, "surface-occlusion", occlusion);
    }
    super::pbr_factors::canonical(pbr_surface_factors, target);
    if let Some(emissive) = &extensions.emissive {
        canonical_piece(target, "surface-emissive", emissive);
    }
    if let Some(configuration) = emissive_configuration {
        canonical_piece(target, "surface-emissive-configuration", "enabled");
        if let Some(color) = &configuration.color_parameter {
            canonical_piece(target, "surface-emissive-color", color);
        }
        if let Some(intensity) = &configuration.intensity_parameter {
            canonical_piece(target, "surface-emissive-intensity", intensity);
        }
        if let Some(texture) = &configuration.texture {
            canonical_piece(target, "surface-emissive-texture", &texture.texture);
            canonical_piece(target, "surface-emissive-sampler", &texture.sampler);
            if let Some(level) = &texture.level_parameter {
                canonical_piece(target, "surface-emissive-level", level);
            }
            canonical_piece(
                target,
                "surface-emissive-texture-encoding",
                match texture.encoding {
                    crate::MaterialTextureEncoding::Linear => "linear",
                    crate::MaterialTextureEncoding::Srgb => "srgb",
                    crate::MaterialTextureEncoding::SrgbEncodedUnorm => "srgb-encoded-unorm",
                },
            );
            if matches!(
                texture.encoding,
                crate::MaterialTextureEncoding::SrgbEncodedUnorm
            ) {
                canonical_piece(
                    target,
                    "surface-emissive-srgb-encoded-unorm-decode",
                    match configuration.srgb_encoded_unorm_decode {
                        crate::EmissiveSrgbDecode::BabylonApproximate => "babylon-approximate",
                        crate::EmissiveSrgbDecode::Exact => "exact",
                    },
                );
            }
        }
    }
}

pub(super) fn canonical_policy(policy: SurfacePolicy, target: &mut String) {
    let alpha_mode = match policy.alpha_mode {
        SurfaceAlphaMode::Opaque => "opaque",
        SurfaceAlphaMode::Mask => "mask",
        SurfaceAlphaMode::Blend => "blend",
        SurfaceAlphaMode::MaskBlend => "mask-blend",
    };
    canonical_piece(target, "surface-alpha-mode", alpha_mode);
    let blend_mode = match policy.blend_mode {
        crate::SurfaceBlendMode::Combine => None,
        crate::SurfaceBlendMode::Premultiplied => Some("premultiplied"),
        crate::SurfaceBlendMode::PremultipliedPorterDuff => Some("premultiplied-porter-duff"),
    };
    if let Some(blend_mode) = blend_mode {
        canonical_piece(target, "surface-blend-mode", blend_mode);
    }
    canonical_piece(
        target,
        "surface-alpha-cutoff-bits",
        &policy.alpha_cutoff.to_bits().to_string(),
    );
    canonical_piece(
        target,
        "surface-back-face-culling",
        &policy.back_face_culling.to_string(),
    );
    if let Some(orientation) = policy.side_orientation {
        let orientation = match orientation {
            crate::SurfaceSideOrientation::CounterClockwise => "counter-clockwise",
            crate::SurfaceSideOrientation::Clockwise => "clockwise",
        };
        canonical_piece(target, "surface-side-orientation", orientation);
    }
    if policy.separate_culling_pass {
        canonical_piece(target, "surface-separate-culling-pass", "enabled");
    }
    canonical_piece(
        target,
        "surface-two-sided-lighting",
        &policy.two_sided_lighting.to_string(),
    );
}

pub(super) fn canonical(output: &SurfaceOutput, target: &mut String) {
    match output {
        SurfaceOutput::Color => {}
        SurfaceOutput::ColorUv0 => canonical_piece(target, "surface-output", "color-uv0"),
        SurfaceOutput::MetallicRoughness { properties } => {
            canonical_piece(target, "surface-output", "metallic-roughness");
            canonical_piece(target, "surface-properties", properties);
        }
        SurfaceOutput::MetallicRoughnessUv0 { properties } => {
            canonical_piece(target, "surface-output", "metallic-roughness-uv0");
            canonical_piece(target, "surface-properties", properties);
        }
        SurfaceOutput::MetallicRoughnessUv0Normal { properties, normal } => {
            canonical_piece(target, "surface-output", "metallic-roughness-uv0-normal");
            canonical_piece(target, "surface-properties", properties);
            canonical_piece(target, "surface-normal", normal);
        }
    }
}

pub(super) fn append_evaluators(
    generated: &mut String,
    root_fragment: &str,
    output: &SurfaceOutput,
    extensions: &SurfaceExtensions,
    emissive_configuration: Option<&EmissiveConfiguration>,
    pbr_surface_factors: Option<&PbrSurfaceFactors>,
    policy: SurfacePolicy,
) {
    match output {
        SurfaceOutput::Color => generated.push_str(&format!(
            "fn evaluate_material() -> vec4<f32> {{\n    return {root_fragment}();\n}}\n"
        )),
        SurfaceOutput::ColorUv0 => generated.push_str(&format!(
            "fn evaluate_material_uv0(uv: vec2<f32>) -> vec4<f32> {{\n    return {root_fragment}(uv);\n}}\n"
        )),
        SurfaceOutput::MetallicRoughness { properties } => {
            generated.push_str(&format!(
                "fn evaluate_material() -> vec4<f32> {{\n    return {root_fragment}();\n}}\n"
            ));
            generated.push_str(&format!(
                "fn evaluate_metallic_roughness() -> vec2<f32> {{\n    return {properties}();\n}}\n"
            ));
        }
        SurfaceOutput::MetallicRoughnessUv0 { properties } => {
            append_uv0_pbr_evaluators(generated, root_fragment, properties, pbr_surface_factors);
        }
        SurfaceOutput::MetallicRoughnessUv0Normal { properties, normal } => {
            append_uv0_pbr_evaluators(generated, root_fragment, properties, pbr_surface_factors);
            generated.push_str(&format!("fn evaluate_normal_uv0(uv: vec2<f32>) -> vec3<f32> {{\n    return {normal}(uv);\n}}\n"));
            generated.push_str(&super::pbr_factors::uv0_occlusion_evaluator(
                extensions.occlusion.as_deref(),
                pbr_surface_factors,
            ));
            match (&extensions.emissive, emissive_configuration) {
                (Some(_), Some(_)) => unreachable!("validated emissive configuration conflict"),
                (Some(emissive), None) => generated.push_str(&format!("fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> {{\n    return {emissive}(uv);\n}}\n")),
                (None, Some(configuration)) => append_emissive_configuration(generated, configuration),
                (None, None) => generated.push_str("fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0); }\n"),
            }
        }
    }
    append_policy_helpers(generated, policy);
}

fn append_uv0_pbr_evaluators(
    generated: &mut String,
    root: &str,
    properties: &str,
    factors: Option<&PbrSurfaceFactors>,
) {
    if let Some(factors) = factors {
        generated.push_str(&super::pbr_factors::uv0_base_evaluator(root, factors));
        generated.push_str(&super::pbr_factors::uv0_properties_evaluator(
            properties, factors,
        ));
    } else {
        generated.push_str(&format!(
            "fn evaluate_material_uv0(uv: vec2<f32>) -> vec4<f32> {{
    return {root}(uv);
}}
"
        ));
        generated.push_str(&format!(
            "fn evaluate_metallic_roughness_uv0(uv: vec2<f32>) -> vec2<f32> {{
    return {properties}(uv);
}}
"
        ));
    }
}

fn append_emissive_configuration(generated: &mut String, configuration: &EmissiveConfiguration) {
    let color = configuration
        .color_parameter
        .as_deref()
        .unwrap_or("vec3<f32>(0.0)");
    let intensity = configuration
        .intensity_parameter
        .as_deref()
        .unwrap_or("1.0");
    let texture = configuration.texture.as_ref();
    let level = texture
        .and_then(|texture| texture.level_parameter.as_deref())
        .unwrap_or("1.0");
    let mut bindings = [color, intensity, level, "", ""];
    if let Some(texture) = texture {
        bindings[3] = &texture.texture;
        bindings[4] = &texture.sampler;
    }
    let uv = unique_emissive_local("emissive_uv", &bindings);
    generated.push_str(&format!(
        "fn evaluate_emissive_uv0({uv}: vec2<f32>) -> vec3<f32> {{\n"
    ));

    if let Some(texture) = texture {
        let sample = format!(
            "textureSample({}, {}, {uv}).rgb",
            texture.texture, texture.sampler
        );
        let linear = match texture.encoding {
            crate::MaterialTextureEncoding::Linear | crate::MaterialTextureEncoding::Srgb => sample,
            crate::MaterialTextureEncoding::SrgbEncodedUnorm => {
                let name = unique_emissive_local("emissive_sample", &bindings);
                generated.push_str(&format!("    let {name} = {sample};\n"));
                match configuration.srgb_encoded_unorm_decode {
                    crate::EmissiveSrgbDecode::BabylonApproximate => {
                        format!("pow({name}, vec3<f32>(2.2))")
                    }
                    crate::EmissiveSrgbDecode::Exact => format!(
                        "select(\n        pow(0.947867299 * ({name} + vec3<f32>(0.055)), vec3<f32>(2.4)),\n        0.0773993808 * {name},\n        {name} <= vec3<f32>(0.04045)\n    )"
                    ),
                }
            }
        };
        generated.push_str(&format!(
            "    return {color} * {linear} * {level} * {intensity};\n"
        ));
    } else {
        generated.push_str(&format!("    return {color} * {intensity};\n"));
    }
    generated.push_str("}\n");
}

fn unique_emissive_local(seed: &str, bindings: &[&str]) -> String {
    let mut name = seed.to_owned();
    while bindings.contains(&name.as_str()) {
        name.push('_');
    }
    name
}

fn append_policy_helpers(generated: &mut String, policy: SurfacePolicy) {
    let rejects_alpha = matches!(
        policy.alpha_mode,
        SurfaceAlphaMode::Mask | SurfaceAlphaMode::MaskBlend
    );
    let preserves_alpha = matches!(
        policy.alpha_mode,
        SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
    );
    generated.push_str(&format!(
        "fn material_alpha_rejected(alpha: f32) -> bool {{ return {rejects_alpha} && alpha < {}; }}\n",
        policy.alpha_cutoff,
    ));
    generated.push_str(&format!(
        "fn material_output_alpha(alpha: f32) -> f32 {{ if ({preserves_alpha}) {{ return alpha; }} return 1.0; }}\n"
    ));
    let two_sided_lighting = !policy.back_face_culling && policy.two_sided_lighting;
    generated.push_str(&format!(
        "fn material_two_sided_lighting() -> bool {{ return {}; }}\n",
        two_sided_lighting,
    ));
}

fn canonical_piece(output: &mut String, label: &str, value: &str) {
    output.push_str(label);
    output.push(':');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}
