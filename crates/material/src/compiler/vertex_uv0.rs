//! Vertex UV0 evaluator shared by the UV0 PBR geometry entries.

use std::collections::BTreeMap;

use crate::authoring::VertexUv0;

use super::{
    MaterialDefinition, MaterialError, MaterialType, SurfaceOutput,
    canonical::piece,
    node_graph::{NormalizedNodeGraph, normalize_vertex_uv0},
    surface::supplies_pbr_context,
    validation::FunctionSignature,
};

pub(super) const EVALUATOR: &str = "evaluate_pbr_vertex_uv0";
pub(super) const VARYING_EVALUATOR: &str = "evaluate_pbr_vertex_varyings";

#[derive(Clone, Debug)]
pub(super) enum NormalizedVertexUv0 {
    Callable(String),
    Graph(NormalizedNodeGraph),
}

impl NormalizedVertexUv0 {
    pub(super) fn graph(&self) -> Option<&NormalizedNodeGraph> {
        match self {
            Self::Callable(_) => None,
            Self::Graph(graph) => Some(graph),
        }
    }

    fn callable(&self) -> &str {
        match self {
            Self::Callable(callable) => callable,
            Self::Graph(graph) => &graph.root,
        }
    }
}

pub(super) fn varying_types(vertex_uv0: Option<&NormalizedVertexUv0>) -> &[(u32, MaterialType)] {
    vertex_uv0
        .and_then(NormalizedVertexUv0::graph)
        .map_or(&[], |graph| graph.vertex_varyings.as_slice())
}

pub(super) fn validate(
    definition: &MaterialDefinition,
    functions: &BTreeMap<String, FunctionSignature>,
) -> Result<Option<NormalizedVertexUv0>, MaterialError> {
    let selected = definition.vertex_uv0();
    let varyings = definition.vertex_varyings();
    if !varyings.is_empty() && !matches!(selected, Some(VertexUv0::Graph(_))) {
        return Err(MaterialError::VertexVaryingsRequireGraph);
    }
    let Some(vertex_uv0) = selected else {
        return Ok(None);
    };
    if !supplies_pbr_context(definition.output()) {
        return Err(MaterialError::VertexUv0Unsupported {
            output: definition.output().clone(),
        });
    }
    let normalized = match vertex_uv0 {
        VertexUv0::Callable(callable) => {
            let signature =
                functions
                    .get(callable)
                    .ok_or_else(|| MaterialError::MissingFragment {
                        name: callable.clone(),
                    })?;
            if !matches!(signature.inputs.as_slice(), [uv] if uv.ty == MaterialType::Vec2)
                || signature.output != MaterialType::Vec2
            {
                return Err(MaterialError::VertexUv0Signature {
                    callable: callable.clone(),
                });
            }
            NormalizedVertexUv0::Callable(callable.clone())
        }
        VertexUv0::Graph(graph) => {
            let normalized = normalize_vertex_uv0(graph, functions, varyings)?;
            if normalized.output != MaterialType::Vec2 {
                return Err(MaterialError::VertexUv0GraphOutput {
                    node: graph.output,
                    actual: normalized.output,
                });
            }
            NormalizedVertexUv0::Graph(normalized)
        }
    };
    Ok(Some(normalized))
}

pub(super) fn canonical(
    output: &SurfaceOutput,
    vertex_uv0: Option<&NormalizedVertexUv0>,
    canonical: &mut String,
) {
    if supplies_pbr_context(output) {
        piece(
            canonical,
            "surface-evaluator-abi",
            "pbr-uv0-surface-value-v4",
        );
    }
    match vertex_uv0 {
        Some(NormalizedVertexUv0::Callable(callable)) => piece(canonical, "vertex-uv0", callable),
        Some(NormalizedVertexUv0::Graph(graph)) => {
            piece(canonical, "vertex-uv0-graph", &graph.source)
        }
        None => {}
    }
}

pub(super) fn append_evaluator(
    output: &SurfaceOutput,
    vertex_uv0: Option<&NormalizedVertexUv0>,
    target: &mut String,
) {
    if !supplies_pbr_context(output) {
        return;
    }
    if let Some(graph) = vertex_uv0.and_then(NormalizedVertexUv0::graph) {
        target.push_str(&graph.source);
    }
    let callable = vertex_uv0.map(NormalizedVertexUv0::callable);
    let uv = if callable == Some("uv") {
        "vertex_uv"
    } else {
        "uv"
    };
    let mut value = callable.map_or_else(|| uv.into(), |callable| format!("{callable}({uv})"));
    if !varying_types(vertex_uv0).is_empty() {
        target.push_str(&format!(
            "fn {VARYING_EVALUATOR}({uv}: vec2<f32>) -> FluxelVertexVaryings {{\n    return {value};\n}}\n"
        ));
        value = format!("{VARYING_EVALUATOR}({uv}).uv0");
    }
    target.push_str(&format!(
        "fn {EVALUATOR}({uv}: vec2<f32>) -> vec2<f32> {{\n    return {value};\n}}\n"
    ));
}
