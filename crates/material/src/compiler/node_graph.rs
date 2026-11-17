//! Typed runtime material-node topology lowered to WGSL call wrappers.

use std::collections::{BTreeMap, BTreeSet};

use fluxel_shader::CallableDeclaration;

use crate::{
    MaterialError, MaterialNodeGraph, MaterialNodeId, MaterialNodeInputSource, MaterialType,
};

use super::validation::FunctionSignature;

/// Validated generated glue for a node-authored material graph.
#[derive(Clone, Debug)]
pub(super) struct NormalizedNodeGraph {
    pub(super) root: String,
    pub(super) output: MaterialType,
    pub(super) source: String,
    pub(super) callables: Vec<CallableDeclaration>,
    pub(super) generated_names: Vec<String>,
    pub(super) accepts_uv0: bool,
}

pub(super) fn normalize(
    graph: &MaterialNodeGraph,
    functions: &BTreeMap<String, FunctionSignature>,
    accepts_uv0: bool,
) -> Result<NormalizedNodeGraph, MaterialError> {
    let mut nodes = BTreeMap::new();
    for node in &graph.nodes {
        if nodes.insert(node.id, node).is_some() {
            return Err(MaterialError::DuplicateNode { node: node.id });
        }
    }
    let output = nodes
        .get(&graph.output)
        .ok_or(MaterialError::MissingNode { node: graph.output })?;
    for node in nodes.values() {
        validate_node(
            node.id,
            &node.callable,
            &node.inputs,
            &nodes,
            functions,
            accepts_uv0,
        )?;
    }
    // Every declared node is emitted into the generated source.  Validate all
    // components rather than only the selected output closure so an unreachable
    // recursive pair cannot leak invalid WGSL into an otherwise valid material.
    for node in nodes.keys().copied() {
        validate_cycles(node, &nodes)?;
    }

    let mut source = String::from("// Generated material node graph.\n");
    let mut callables = Vec::with_capacity(nodes.len());
    for node in nodes.values() {
        let signature = functions
            .get(&node.callable)
            .expect("validated material callable");
        source.push_str(&format!(
            "fn {}({}) -> {} {{ return {}(",
            node_name(node.id),
            evaluator_argument(accepts_uv0),
            signature.output.wgsl(),
            node.callable
        ));
        for (index, input) in signature.inputs.iter().enumerate() {
            if index != 0 {
                source.push_str(", ");
            }
            let source_node = node
                .inputs
                .iter()
                .find(|connection| connection.input == input.name)
                .expect("validated node input");
            match source_node.source {
                MaterialNodeInputSource::Node(source_node) => {
                    source.push_str(&format!(
                        "{}({})",
                        node_name(source_node),
                        argument_name(accepts_uv0)
                    ));
                }
                MaterialNodeInputSource::Uv0 => source.push_str("uv"),
            }
        }
        source.push_str("); }\n");
        callables.push(CallableDeclaration {
            name: node_name(node.id),
            arguments: evaluator_arguments(accepts_uv0),
            result: Some(signature.output.shader_type()),
        });
    }
    Ok(NormalizedNodeGraph {
        root: node_name(output.id),
        output: functions
            .get(&output.callable)
            .expect("validated output callable")
            .output,
        source,
        generated_names: callables
            .iter()
            .map(|callable| callable.name.clone())
            .collect(),
        callables,
        accepts_uv0,
    })
}

fn validate_node(
    node: MaterialNodeId,
    callable: &str,
    inputs: &[crate::MaterialNodeInput],
    nodes: &BTreeMap<MaterialNodeId, &crate::MaterialNode>,
    functions: &BTreeMap<String, FunctionSignature>,
    accepts_uv0: bool,
) -> Result<(), MaterialError> {
    let signature = functions
        .get(callable)
        .ok_or_else(|| MaterialError::MissingFragment {
            name: callable.into(),
        })?;
    let mut connected = BTreeSet::new();
    for connection in inputs {
        if !connected.insert(connection.input.as_str()) {
            return Err(MaterialError::DuplicateNodeInput {
                node,
                input: connection.input.clone(),
            });
        }
        let expected = signature
            .inputs
            .iter()
            .find(|input| input.name == connection.input)
            .ok_or_else(|| MaterialError::UnknownNodeInput {
                node,
                input: connection.input.clone(),
            })?;
        let actual = match connection.source {
            MaterialNodeInputSource::Node(source_node) => {
                let source = nodes
                    .get(&source_node)
                    .ok_or(MaterialError::MissingNode { node: source_node })?;
                functions
                    .get(&source.callable)
                    .ok_or_else(|| MaterialError::MissingFragment {
                        name: source.callable.clone(),
                    })?
                    .output
            }
            MaterialNodeInputSource::Uv0 => {
                if !accepts_uv0 {
                    return Err(MaterialError::NodeContextUnavailable {
                        node,
                        input: connection.input.clone(),
                    });
                }
                MaterialType::Vec2
            }
        };
        if expected.ty != actual {
            return Err(MaterialError::NodeInputTypeMismatch {
                node,
                input: connection.input.clone(),
                expected: expected.ty,
                actual,
            });
        }
    }
    if let Some(input) = signature
        .inputs
        .iter()
        .find(|input| !connected.contains(input.name.as_str()))
    {
        return Err(MaterialError::MissingNodeInput {
            node,
            input: input.name.clone(),
        });
    }
    Ok(())
}

fn validate_cycles(
    node: MaterialNodeId,
    nodes: &BTreeMap<MaterialNodeId, &crate::MaterialNode>,
) -> Result<(), MaterialError> {
    fn visit(
        node: MaterialNodeId,
        nodes: &BTreeMap<MaterialNodeId, &crate::MaterialNode>,
        visiting: &mut BTreeSet<MaterialNodeId>,
        visited: &mut BTreeSet<MaterialNodeId>,
    ) -> Result<(), MaterialError> {
        if visited.contains(&node) {
            return Ok(());
        }
        if !visiting.insert(node) {
            return Err(MaterialError::NodeCycle { node });
        }
        for input in &nodes.get(&node).expect("checked node").inputs {
            if let MaterialNodeInputSource::Node(source) = input.source {
                visit(source, nodes, visiting, visited)?;
            }
        }
        visiting.remove(&node);
        visited.insert(node);
        Ok(())
    }
    visit(node, nodes, &mut BTreeSet::new(), &mut BTreeSet::new())
}

fn evaluator_argument(accepts_uv0: bool) -> &'static str {
    if accepts_uv0 { "uv: vec2<f32>" } else { "" }
}

fn argument_name(accepts_uv0: bool) -> &'static str {
    if accepts_uv0 { "uv" } else { "" }
}

fn evaluator_arguments(accepts_uv0: bool) -> Vec<fluxel_shader::ValueType> {
    accepts_uv0
        .then_some(fluxel_shader::ValueType::Vec2(
            fluxel_shader::ScalarType::F32,
        ))
        .into_iter()
        .collect()
}

pub(super) fn node_name(node: MaterialNodeId) -> String {
    // WGSL reserves identifiers beginning with a double underscore.
    format!("fluxel_material_node_{}", node.0)
}
