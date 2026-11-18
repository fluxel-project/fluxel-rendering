//! Surface-domain validation and generated evaluation entry wrappers.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    MaterialDefinition, MaterialError, MaterialType, SurfaceAlphaMode, SurfaceExtensions,
    SurfaceOutput, SurfacePolicy, validation::FunctionSignature,
};

/// Validates all surface-owned extensions after the root callable is resolved.
pub(super) fn validate_definition(
    definition: &MaterialDefinition,
    root: &FunctionSignature,
    functions: &BTreeMap<String, FunctionSignature>,
    root_name: &str,
) -> Result<(), MaterialError> {
    validate(
        definition.output(),
        definition.surface_extensions(),
        root,
        functions,
        root_name,
    )?;
    super::transmission::validate(
        definition.output(),
        definition.transmission_extensions(),
        functions,
    )
}

pub(super) fn validate(
    output: &SurfaceOutput,
    extensions: &SurfaceExtensions,
    root: &FunctionSignature,
    functions: &BTreeMap<String, FunctionSignature>,
    root_name: &str,
) -> Result<(), MaterialError> {
    if (!extensions.occlusion.is_none() || !extensions.emissive.is_none())
        && !matches!(output, SurfaceOutput::MetallicRoughnessUv0Normal { .. })
    {
        return Err(MaterialError::SurfaceExtensionsUnsupported {
            output: output.clone(),
        });
    }
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

pub(super) fn canonical_extensions(extensions: &SurfaceExtensions, target: &mut String) {
    if let Some(occlusion) = &extensions.occlusion {
        canonical_piece(target, "surface-occlusion", occlusion);
    }
    if let Some(emissive) = &extensions.emissive {
        canonical_piece(target, "surface-emissive", emissive);
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
            generated.push_str(&format!("fn evaluate_material_uv0(uv: vec2<f32>) -> vec4<f32> {{\n    return {root_fragment}(uv);\n}}\n"));
            generated.push_str(&format!("fn evaluate_metallic_roughness_uv0(uv: vec2<f32>) -> vec2<f32> {{\n    return {properties}(uv);\n}}\n"));
        }
        SurfaceOutput::MetallicRoughnessUv0Normal { properties, normal } => {
            generated.push_str(&format!("fn evaluate_material_uv0(uv: vec2<f32>) -> vec4<f32> {{\n    return {root_fragment}(uv);\n}}\n"));
            generated.push_str(&format!("fn evaluate_metallic_roughness_uv0(uv: vec2<f32>) -> vec2<f32> {{\n    return {properties}(uv);\n}}\n"));
            generated.push_str(&format!("fn evaluate_normal_uv0(uv: vec2<f32>) -> vec3<f32> {{\n    return {normal}(uv);\n}}\n"));
            match &extensions.occlusion {
                Some(occlusion) => generated.push_str(&format!("fn evaluate_occlusion_uv0(uv: vec2<f32>) -> f32 {{\n    return {occlusion}(uv);\n}}\n")),
                None => generated.push_str("fn evaluate_occlusion_uv0(uv: vec2<f32>) -> f32 { return 1.0; }\n"),
            }
            match &extensions.emissive {
                Some(emissive) => generated.push_str(&format!("fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> {{\n    return {emissive}(uv);\n}}\n")),
                None => generated.push_str("fn evaluate_emissive_uv0(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0); }\n"),
            }
        }
    }
    append_policy_helpers(generated, policy);
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
