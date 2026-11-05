//! Surface-domain validation and generated evaluation entry wrappers.

use std::collections::{BTreeMap, BTreeSet};

use super::{validation::FunctionSignature, MaterialError, MaterialType, SurfaceOutput};

/// Validates the root callables required by one selected surface domain.
pub(super) fn validate(
    output: &SurfaceOutput,
    root: &FunctionSignature,
    functions: &BTreeMap<String, FunctionSignature>,
    root_name: &str,
) -> Result<(), MaterialError> {
    if root.output != MaterialType::Vec4 {
        return Err(MaterialError::UnsupportedOutput {
            output: output.clone(),
        });
    }
    if !root.inputs.is_empty() {
        return Err(MaterialError::RootInputsUnsupported {
            fragment: root_name.into(),
        });
    }
    let SurfaceOutput::MetallicRoughness { properties } = output else {
        return Ok(());
    };
    let reserved = reserved_names(output);
    for name in [root_name, properties.as_str()] {
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
    if !properties_signature.inputs.is_empty() {
        return Err(MaterialError::SurfacePropertiesInputsUnsupported {
            properties: properties.clone(),
        });
    }
    Ok(())
}

/// Returns the generated function names reserved by the selected domain.
pub(super) fn reserved_names(output: &SurfaceOutput) -> BTreeSet<String> {
    let mut names = BTreeSet::from([String::from("evaluate_material")]);
    if matches!(output, SurfaceOutput::MetallicRoughness { .. }) {
        names.insert(String::from("evaluate_metallic_roughness"));
    }
    names
}

/// Adds the domain-specific canonical representation without altering Color.
pub(super) fn canonical(output: &SurfaceOutput, target: &mut String) {
    if let SurfaceOutput::MetallicRoughness { properties } = output {
        canonical_piece(target, "surface-output", "metallic-roughness");
        canonical_piece(target, "surface-properties", properties);
    }
}

/// Adds generated material evaluation wrappers for the selected domain.
pub(super) fn append_evaluators(
    generated: &mut String,
    root_fragment: &str,
    output: &SurfaceOutput,
) {
    generated.push_str(&format!(
        "fn evaluate_material() -> vec4<f32> {{\n    return {root_fragment}();\n}}\n"
    ));
    if let SurfaceOutput::MetallicRoughness { properties } = output {
        generated.push_str(&format!(
            "fn evaluate_metallic_roughness() -> vec2<f32> {{\n    return {properties}();\n}}\n"
        ));
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
