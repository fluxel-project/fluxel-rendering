//! Vertex UV0 evaluator shared by the UV0 PBR geometry entries.

use std::collections::BTreeMap;

use super::{MaterialDefinition, MaterialError, MaterialType, SurfaceOutput};
use super::{canonical::piece, surface::supplies_pbr_context, validation::FunctionSignature};

pub(super) const EVALUATOR: &str = "evaluate_pbr_vertex_uv0";

pub(super) fn validate(
    definition: &MaterialDefinition,
    functions: &BTreeMap<String, FunctionSignature>,
) -> Result<Option<String>, MaterialError> {
    let Some(callable) = definition.vertex_uv0() else {
        return Ok(None);
    };
    if !supplies_pbr_context(definition.output()) {
        return Err(MaterialError::VertexUv0Unsupported {
            output: definition.output().clone(),
        });
    }
    let signature = functions
        .get(callable)
        .ok_or_else(|| MaterialError::MissingFragment {
            name: callable.into(),
        })?;
    if !matches!(signature.inputs.as_slice(), [uv] if uv.ty == MaterialType::Vec2)
        || signature.output != MaterialType::Vec2
    {
        return Err(MaterialError::VertexUv0Signature {
            callable: callable.into(),
        });
    }
    Ok(Some(callable.into()))
}

pub(super) fn canonical(output: &SurfaceOutput, callable: Option<&str>, canonical: &mut String) {
    if supplies_pbr_context(output) {
        piece(
            canonical,
            "surface-evaluator-abi",
            "pbr-uv0-world-context-with-vertex-uv",
        );
    }
    if let Some(callable) = callable {
        piece(canonical, "vertex-uv0", callable);
    }
}

pub(super) fn append_evaluator(
    output: &SurfaceOutput,
    callable: Option<&str>,
    target: &mut String,
) {
    if supplies_pbr_context(output) {
        let uv = if callable == Some("uv") {
            "vertex_uv"
        } else {
            "uv"
        };
        let value = callable.map_or_else(|| uv.into(), |callable| format!("{callable}({uv})"));
        target.push_str(&format!(
            "fn {EVALUATOR}({uv}: vec2<f32>) -> vec2<f32> {{\n    return {value};\n}}\n"
        ));
    }
}
