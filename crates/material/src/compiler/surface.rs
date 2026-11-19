//! Surface-domain validation and generated evaluation entry wrappers.

use std::collections::{BTreeMap, BTreeSet};

use crate::MaterialNodeInputSource;

use super::{
    AmbientOcclusionConfiguration, DiffuseRoughnessConfiguration, EmissiveConfiguration,
    MaterialDefinition, MaterialError, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialType, PbrSurfaceFactors, SurfaceAlphaMode, SurfaceExtensions, SurfaceOutput,
    SurfacePolicy, validation::FunctionSignature,
};

#[path = "surface_emission.rs"]
mod surface_emission;
pub(super) use surface_emission::append_evaluators;

/// Validates all surface-owned extensions after the root callable is resolved.
pub(super) fn validate_definition(
    definition: &MaterialDefinition,
    root: &FunctionSignature,
    functions: &BTreeMap<String, FunctionSignature>,
    root_name: &str,
    root_pbr_inputs: &[(MaterialNodeInputSource, MaterialType)],
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    validate(
        definition.output(),
        definition.surface_extensions(),
        definition.emissive_configuration(),
        definition.pbr_surface_factors(),
        definition.diffuse_roughness_configuration(),
        definition.ambient_occlusion_configuration(),
        root,
        functions,
        root_name,
        root_pbr_inputs,
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

/// Whether this output supplies a tangent-space normal callable.
pub(super) fn has_tangent_space_normal(output: &SurfaceOutput) -> bool {
    matches!(output, SurfaceOutput::MetallicRoughnessUv0Normal { .. })
        || matches!(
            output,
            SurfaceOutput::MetallicRoughnessUv0Graph {
                normal: Some(_),
                ..
            }
        )
}

/// Adds normal-map metadata to the static variant identity.
pub(super) fn canonical_normal_map_settings(
    settings: Option<crate::NormalMapSettings>,
    canonical: &mut String,
) {
    use super::canonical::piece;
    let Some(settings) =
        settings.filter(|settings| *settings != crate::NormalMapSettings::default())
    else {
        return;
    };
    piece(
        canonical,
        "normal-map-strength",
        &settings.strength().to_bits().to_string(),
    );
    piece(
        canonical,
        "normal-map-invert-x",
        &settings.invert_x.to_string(),
    );
    piece(
        canonical,
        "normal-map-invert-y",
        &settings.invert_y.to_string(),
    );
}

pub(super) fn validate(
    output: &SurfaceOutput,
    extensions: &SurfaceExtensions,
    emissive_configuration: Option<&EmissiveConfiguration>,
    pbr_surface_factors: Option<&PbrSurfaceFactors>,
    diffuse_roughness_configuration: Option<&DiffuseRoughnessConfiguration>,
    ambient_occlusion_configuration: Option<&AmbientOcclusionConfiguration>,
    root: &FunctionSignature,
    functions: &BTreeMap<String, FunctionSignature>,
    root_name: &str,
    root_pbr_inputs: &[(MaterialNodeInputSource, MaterialType)],
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    if matches!(
        output,
        SurfaceOutput::MetallicRoughnessUv0Graph {
            normal: Some(_),
            world_normal: Some(_),
            ..
        }
    ) {
        return Err(MaterialError::PbrGraphNormalConflict);
    }
    if (extensions.occlusion.is_some()
        || extensions.emissive.is_some()
        || emissive_configuration.is_some())
        && !supports_surface_effects(output)
    {
        return Err(MaterialError::SurfaceExtensionsUnsupported {
            output: output.clone(),
        });
    }
    super::pbr_factors::validate(pbr_surface_factors, output, parameters, resources)?;
    super::diffuse_roughness::validate(
        diffuse_roughness_configuration,
        output,
        parameters,
        resources,
    )?;
    super::ambient_occlusion::validate(
        ambient_occlusion_configuration,
        output,
        parameters,
        resources,
    )?;
    if root.output != MaterialType::Vec4 {
        return Err(MaterialError::UnsupportedOutput {
            output: output.clone(),
        });
    }
    let expects_uv0 = uses_uv0(output);
    let valid_inputs = if root_pbr_inputs.is_empty() {
        has_evaluator_inputs(root, expects_uv0)
    } else {
        matches!(root.inputs.as_slice(), [uv, context @ ..]
            if uv.ty == MaterialType::Vec2 && context.len() == root_pbr_inputs.len()
            && context.iter().map(|input| input.ty)
                .eq(root_pbr_inputs.iter().map(|(_, ty)| *ty)))
    };
    if !valid_inputs {
        return Err(MaterialError::RootInputsUnsupported {
            fragment: root_name.into(),
        });
    }
    let (properties, properties_expect_uv0, normal) = match output {
        SurfaceOutput::Color | SurfaceOutput::ColorUv0 => return Ok(()),
        SurfaceOutput::MetallicRoughness { properties } => (Some(properties.as_str()), false, None),
        SurfaceOutput::MetallicRoughnessUv0 { properties } => {
            (Some(properties.as_str()), true, None)
        }
        SurfaceOutput::MetallicRoughnessUv0Normal { properties, normal } => {
            (Some(properties.as_str()), true, Some(normal.as_str()))
        }
        SurfaceOutput::MetallicRoughnessUv0Graph { normal, .. } => (None, true, normal.as_deref()),
    };
    let reserved = reserved_names(
        output,
        diffuse_roughness_configuration.is_some(),
        ambient_occlusion_configuration.is_some(),
    );
    for name in std::iter::once(root_name).chain(properties).chain(normal) {
        if reserved.contains(name) {
            return Err(MaterialError::TopLevelNameConflict { name: name.into() });
        }
    }
    if let Some(properties) = properties {
        let properties_signature =
            functions
                .get(properties)
                .ok_or_else(|| MaterialError::MissingSurfaceProperties {
                    properties: properties.into(),
                })?;
        if properties_signature.output != MaterialType::Vec2 {
            return Err(MaterialError::SurfacePropertiesTypeMismatch {
                properties: properties.into(),
                actual: properties_signature.output,
            });
        }
        if !has_evaluator_inputs(properties_signature, properties_expect_uv0) {
            return Err(MaterialError::SurfacePropertiesInputsUnsupported {
                properties: properties.into(),
            });
        }
    }
    if let Some(normal) = normal {
        let normal_signature =
            functions
                .get(normal)
                .ok_or_else(|| MaterialError::MissingSurfaceNormal {
                    normal: normal.into(),
                })?;
        if normal_signature.output != MaterialType::Vec3 {
            return Err(MaterialError::SurfaceNormalTypeMismatch {
                normal: normal.into(),
                actual: normal_signature.output,
            });
        }
        if !has_evaluator_inputs(normal_signature, true) {
            return Err(MaterialError::SurfaceNormalInputsUnsupported {
                normal: normal.into(),
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
            | SurfaceOutput::MetallicRoughnessUv0Graph { .. }
    )
}

pub(super) fn supplies_pbr_context(output: &SurfaceOutput) -> bool {
    matches!(
        output,
        SurfaceOutput::MetallicRoughnessUv0 { .. }
            | SurfaceOutput::MetallicRoughnessUv0Normal { .. }
            | SurfaceOutput::MetallicRoughnessUv0Graph { .. }
    )
}

pub(super) fn supports_surface_effects(output: &SurfaceOutput) -> bool {
    matches!(output, SurfaceOutput::MetallicRoughnessUv0Normal { .. })
        || matches!(
            output,
            SurfaceOutput::MetallicRoughnessUv0Graph {
                normal: Some(_),
                ..
            }
        )
}

fn has_evaluator_inputs(signature: &FunctionSignature, expects_uv0: bool) -> bool {
    match (expects_uv0, signature.inputs.as_slice()) {
        (false, []) => true,
        (true, [input]) => input.ty == MaterialType::Vec2,
        _ => false,
    }
}

pub(super) fn reserved_names(
    output: &SurfaceOutput,
    uses_diffuse_roughness: bool,
    uses_ambient_occlusion_color: bool,
) -> BTreeSet<String> {
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
        SurfaceOutput::MetallicRoughnessUv0Graph { normal, .. } => {
            let mut names = BTreeSet::from([
                String::from("FluxelPbrGraphValues"),
                String::from("evaluate_occlusion_uv0"),
                String::from("evaluate_emissive_uv0"),
            ]);
            if normal.is_some() {
                names.extend([String::from("evaluate_normal_uv0")]);
            }
            names
        }
    };
    if supplies_pbr_context(output) {
        names.insert(String::from("evaluate_pbr_material_uv0"));
        names.insert(String::from("evaluate_pbr_surface_uv0"));
        names.insert(String::from("FluxelPbrSurface"));
        names.insert(String::from("FluxelVertexVaryings"));
        names.insert(String::from("evaluate_pbr_vertex_varyings"));
        names.insert(String::from(super::vertex_uv0::EVALUATOR));
    }
    if uses_diffuse_roughness {
        names.insert(if uses_uv0(output) {
            String::from("evaluate_diffuse_roughness_uv0")
        } else {
            String::from("evaluate_diffuse_roughness")
        });
    }
    if uses_ambient_occlusion_color {
        names.extend([
            String::from("evaluate_ambient_occlusion_uv0"),
            String::from("evaluate_ambient_occlusion_monochrome"),
        ]);
    }
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
        crate::SurfaceBlendMode::Add => Some("add"),
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
        SurfaceOutput::MetallicRoughnessUv0Graph { normal, .. } => {
            canonical_piece(target, "surface-output", "metallic-roughness-uv0-graph");
            if let Some(normal) = normal {
                canonical_piece(target, "surface-normal", normal);
            }
        }
    }
}

fn canonical_piece(output: &mut String, label: &str, value: &str) {
    output.push_str(label);
    output.push(':');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}
