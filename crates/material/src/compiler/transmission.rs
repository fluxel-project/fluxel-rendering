//! Transmission extension validation and generated WGSL helpers.

use std::collections::BTreeMap;

use crate::MaterialTextureEncoding;

use super::{
    MaterialError, MaterialResource, MaterialResourceKind, MaterialTextureEncodingRequirement,
    MaterialType, RefractionEnvironment, SurfaceOutput, TransmissionExtensions, canonical::piece,
    validation::FunctionSignature,
};

const SURFACE_IOR_DEFAULT: f32 = 1.5;
const ATTENUATION_DISTANCE_DEFAULT: f32 = 1.0;

pub(super) fn validate(
    output: &SurfaceOutput,
    extensions: Option<&TransmissionExtensions>,
    environment: Option<&RefractionEnvironment>,
    functions: &BTreeMap<String, FunctionSignature>,
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let Some(extensions) = extensions else {
        return if environment.is_some() {
            Err(MaterialError::RefractionEnvironmentTransmissionRequired)
        } else {
            Ok(())
        };
    };
    if !matches!(output, SurfaceOutput::MetallicRoughnessUv0Normal { .. }) {
        return Err(MaterialError::TransmissionExtensionsUnsupported {
            output: output.clone(),
        });
    }
    validate_callable(extensions.intensity.as_deref(), functions)?;
    validate_callable(extensions.thickness.as_deref(), functions)?;
    validate_static(extensions)?;
    validate_environment(environment, resources)
}

fn validate_environment(
    environment: Option<&RefractionEnvironment>,
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let Some(environment) = environment else {
        return Ok(());
    };
    let texture = resources
        .iter()
        .find(|resource| resource.name == environment.texture)
        .ok_or_else(|| MaterialError::RefractionEnvironmentResourceMissing {
            field: "texture",
            resource: environment.texture.clone(),
        })?;
    validate_environment_resource("texture", texture, MaterialResourceKind::Texture2dF32)?;
    let sampler = resources
        .iter()
        .find(|resource| resource.name == environment.sampler)
        .ok_or_else(|| MaterialError::RefractionEnvironmentResourceMissing {
            field: "sampler",
            resource: environment.sampler.clone(),
        })?;
    validate_environment_resource(
        "sampler",
        sampler,
        MaterialResourceKind::Sampler { comparison: false },
    )?;
    if !environment.lod_scale.is_finite() {
        return Err(MaterialError::InvalidRefractionEnvironmentParameter {
            parameter: "lod_scale",
        });
    }
    if !environment.lod_offset.is_finite() {
        return Err(MaterialError::InvalidRefractionEnvironmentParameter {
            parameter: "lod_offset",
        });
    }
    if environment
        .direction_from_world
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(MaterialError::InvalidRefractionEnvironmentParameter {
            parameter: "direction_from_world",
        });
    }
    Ok(())
}

fn validate_environment_resource(
    field: &'static str,
    resource: &MaterialResource,
    expected: MaterialResourceKind,
) -> Result<(), MaterialError> {
    if resource.location.group != 0 {
        return Err(MaterialError::RefractionEnvironmentResourceGroup {
            field,
            resource: resource.name.clone(),
            group: resource.location.group,
        });
    }
    if resource.kind != expected {
        return Err(MaterialError::RefractionEnvironmentResourceTypeMismatch {
            field,
            resource: resource.name.clone(),
            expected,
            actual: resource.kind,
        });
    }
    Ok(())
}

fn validate_callable(
    name: Option<&str>,
    functions: &BTreeMap<String, FunctionSignature>,
) -> Result<(), MaterialError> {
    let Some(name) = name else {
        return Ok(());
    };
    let signature = functions
        .get(name)
        .ok_or_else(|| MaterialError::MissingTransmissionExtension { name: name.into() })?;
    if signature.output != MaterialType::F32 {
        return Err(MaterialError::TransmissionExtensionTypeMismatch {
            name: name.into(),
            actual: signature.output,
        });
    }
    let accepts_uv0 =
        matches!(signature.inputs.as_slice(), [input] if input.ty == MaterialType::Vec2);
    if !accepts_uv0 {
        return Err(MaterialError::TransmissionExtensionInputsUnsupported { name: name.into() });
    }
    Ok(())
}

fn validate_static(extensions: &TransmissionExtensions) -> Result<(), MaterialError> {
    if !extensions.index_of_refraction.is_finite() || extensions.index_of_refraction < 1.0 {
        return Err(MaterialError::InvalidTransmissionParameter {
            parameter: "index_of_refraction",
        });
    }
    if let Some(volume_ior) = extensions.volume_index_of_refraction {
        if !volume_ior.is_finite() || volume_ior < 1.0 {
            return Err(MaterialError::InvalidTransmissionParameter {
                parameter: "volume_index_of_refraction",
            });
        }
    }
    if extensions
        .attenuation_color
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(MaterialError::InvalidTransmissionParameter {
            parameter: "attenuation_color",
        });
    }
    if !extensions.attenuation_distance.is_finite() || extensions.attenuation_distance <= 0.0 {
        return Err(MaterialError::InvalidTransmissionParameter {
            parameter: "attenuation_distance",
        });
    }
    if !extensions.minimum_thickness.is_finite() || extensions.minimum_thickness < 0.0 {
        return Err(MaterialError::InvalidTransmissionParameter {
            parameter: "minimum_thickness",
        });
    }
    if !extensions.maximum_thickness.is_finite()
        || extensions.maximum_thickness < extensions.minimum_thickness
    {
        return Err(MaterialError::InvalidTransmissionParameter {
            parameter: "maximum_thickness",
        });
    }
    Ok(())
}

pub(super) fn canonical(extensions: Option<&TransmissionExtensions>, target: &mut String) {
    let Some(extensions) = extensions else {
        piece(target, "transmission", "absent");
        return;
    };
    piece(target, "transmission", "present");
    canonical_optional(
        target,
        "transmission-intensity",
        extensions.intensity.as_deref(),
    );
    canonical_optional(
        target,
        "transmission-thickness",
        extensions.thickness.as_deref(),
    );
    canonical_f32(
        target,
        "transmission-surface-ior",
        extensions.index_of_refraction,
    );
    match extensions.volume_index_of_refraction {
        Some(value) => canonical_f32(target, "transmission-volume-ior", value),
        None => piece(target, "transmission-volume-ior", "surface"),
    }
    for (index, value) in extensions.attenuation_color.iter().enumerate() {
        canonical_f32(
            target,
            &format!("transmission-attenuation-color-{index}"),
            *value,
        );
    }
    canonical_f32(
        target,
        "transmission-attenuation-distance",
        extensions.attenuation_distance,
    );
    canonical_f32(
        target,
        "transmission-minimum-thickness",
        extensions.minimum_thickness,
    );
    canonical_f32(
        target,
        "transmission-maximum-thickness",
        extensions.maximum_thickness,
    );
}

pub(super) fn canonical_environment(
    environment: Option<&RefractionEnvironment>,
    target: &mut String,
) {
    let Some(environment) = environment else {
        piece(target, "refraction-environment", "absent");
        return;
    };
    piece(target, "refraction-environment", "present");
    piece(
        target,
        "refraction-environment-texture",
        &environment.texture,
    );
    piece(
        target,
        "refraction-environment-sampler",
        &environment.sampler,
    );
    canonical_f32(
        target,
        "refraction-environment-lod-scale",
        environment.lod_scale,
    );
    canonical_f32(
        target,
        "refraction-environment-lod-offset",
        environment.lod_offset,
    );
    for (row, values) in environment.direction_from_world.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            canonical_f32(
                target,
                &format!("refraction-environment-direction-{row}-{column}"),
                *value,
            );
        }
    }
}

pub(super) fn effective_texture_encodings(
    environment: Option<&RefractionEnvironment>,
    mut requirements: Vec<MaterialTextureEncodingRequirement>,
) -> Result<Vec<MaterialTextureEncodingRequirement>, MaterialError> {
    let Some(environment) = environment else {
        return Ok(requirements);
    };
    match requirements
        .iter()
        .find(|requirement| requirement.resource == environment.texture)
    {
        Some(requirement) if requirement.encoding != MaterialTextureEncoding::Linear => {
            return Err(
                MaterialError::RefractionEnvironmentTextureEncodingConflict {
                    resource: environment.texture.clone(),
                },
            );
        }
        Some(_) => {}
        None => requirements.push(MaterialTextureEncodingRequirement {
            resource: environment.texture.clone(),
            encoding: MaterialTextureEncoding::Linear,
        }),
    }
    Ok(requirements)
}

fn canonical_optional(target: &mut String, label: &str, value: Option<&str>) {
    match value {
        Some(value) => {
            piece(target, label, "present");
            piece(target, &format!("{label}-name"), value);
        }
        None => piece(target, label, "absent"),
    }
}

fn canonical_f32(target: &mut String, label: &str, value: f32) {
    piece(target, label, &value.to_bits().to_string());
}

pub(super) fn append_evaluators(
    generated: &mut String,
    extensions: Option<&TransmissionExtensions>,
) {
    let intensity = extensions
        .and_then(|extensions| extensions.intensity.as_deref())
        .map_or_else(
            || default_intensity(extensions.is_some()),
            |name| format!("{name}(uv)"),
        );
    let thickness = extensions.map_or_else(
        || "0.0".to_owned(),
        |extensions| match extensions.thickness.as_deref() {
            Some(name) => format!(
                "{} + ({} - {}) * {name}(uv)",
                extensions.minimum_thickness,
                extensions.maximum_thickness,
                extensions.minimum_thickness,
            ),
            None => format!(
                "{} - {}",
                extensions.maximum_thickness, extensions.minimum_thickness,
            ),
        },
    );
    let surface_ior = extensions.map_or(SURFACE_IOR_DEFAULT, |value| value.index_of_refraction);
    let volume_ior = extensions.map_or(SURFACE_IOR_DEFAULT, |value| {
        value
            .volume_index_of_refraction
            .unwrap_or(value.index_of_refraction)
    });
    let attenuation_color = extensions.map_or([1.0; 3], |value| value.attenuation_color);
    let attenuation_distance = extensions.map_or(ATTENUATION_DISTANCE_DEFAULT, |value| {
        value.attenuation_distance
    });
    generated.push_str(&format!(
        "fn evaluate_transmission_uv0(uv: vec2<f32>) -> f32 {{ return {intensity}; }}\n"
    ));
    generated.push_str(&format!(
        "fn evaluate_thickness_uv0(uv: vec2<f32>) -> f32 {{ return {thickness}; }}\n"
    ));
    generated.push_str(&format!(
        "fn material_surface_ior() -> f32 {{ return {surface_ior}; }}\n"
    ));
    generated.push_str(&format!(
        "fn material_volume_ior() -> f32 {{ return {volume_ior}; }}\n"
    ));
    let has_thickness =
        extensions.is_some_and(|value| value.maximum_thickness != value.minimum_thickness);
    generated.push_str(&format!(
        "fn material_refraction_has_thickness() -> bool {{ return {has_thickness}; }}\n"
    ));
    generated.push_str(&format!(
        "fn material_attenuation_color() -> vec3<f32> {{ return vec3<f32>({}, {}, {}); }}\n",
        attenuation_color[0], attenuation_color[1], attenuation_color[2],
    ));
    generated.push_str(&format!(
        "fn material_attenuation_distance() -> f32 {{ return {attenuation_distance}; }}\n"
    ));
}

fn default_intensity(extensions_present: bool) -> String {
    if extensions_present {
        "1.0".into()
    } else {
        "0.0".into()
    }
}
