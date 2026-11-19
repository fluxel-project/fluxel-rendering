//! Typed runtime material-node topology lowered to a WGSL evaluator.

use std::collections::{BTreeMap, BTreeSet};

use fluxel_shader::CallableDeclaration;

use crate::{
    MaterialError, MaterialNodeGraph, MaterialNodeId, MaterialNodeInputSource, MaterialNodeOutput,
    MaterialType, SurfaceOutput,
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
    pub(super) inputs: Vec<crate::FragmentInput>,
    pub(super) pbr_inputs: Vec<MaterialNodeInputSource>,
}

pub(super) const PBR_CONTEXT_INPUTS: [(MaterialNodeInputSource, &str, MaterialType); 4] = [
    (
        MaterialNodeInputSource::WorldPosition,
        "world_position",
        MaterialType::Vec3,
    ),
    (
        MaterialNodeInputSource::WorldNormal,
        "world_normal",
        MaterialType::Vec3,
    ),
    (
        MaterialNodeInputSource::CameraPosition,
        "camera_position",
        MaterialType::Vec3,
    ),
    (
        MaterialNodeInputSource::VertexUv0,
        "vertex_uv0",
        MaterialType::Vec2,
    ),
];

/// Distinguishes graph wrappers and locals in a material that declares both stages.
#[derive(Clone, Copy)]
enum NodeGraphStage {
    Surface,
    VertexUv0,
}

impl NodeGraphStage {
    fn node_name(self, node: MaterialNodeId) -> String {
        match self {
            Self::Surface => format!("fluxel_material_node_{}", node.0),
            Self::VertexUv0 => format!("fluxel_material_vertex_node_{}", node.0),
        }
    }
}

pub(super) fn normalize_surface(
    graph: &MaterialNodeGraph,
    functions: &BTreeMap<String, FunctionSignature>,
    surface_output: &SurfaceOutput,
    has_vertex_uv0: bool,
) -> Result<NormalizedNodeGraph, MaterialError> {
    let context = PBR_CONTEXT_INPUTS
        .into_iter()
        .filter(|(source, _, _)| {
            super::surface::supplies_pbr_context(surface_output)
                && (*source != MaterialNodeInputSource::VertexUv0 || has_vertex_uv0)
        })
        .collect::<Vec<_>>();
    normalize(
        graph,
        functions,
        NodeGraphStage::Surface,
        super::surface::uses_uv0(surface_output),
        &context,
    )
}

pub(super) fn normalize_vertex_uv0(
    graph: &MaterialNodeGraph,
    functions: &BTreeMap<String, FunctionSignature>,
) -> Result<NormalizedNodeGraph, MaterialError> {
    normalize(graph, functions, NodeGraphStage::VertexUv0, true, &[])
}

fn normalize(
    graph: &MaterialNodeGraph,
    functions: &BTreeMap<String, FunctionSignature>,
    stage: NodeGraphStage,
    accepts_uv0: bool,
    available_context: &[(MaterialNodeInputSource, &str, MaterialType)],
) -> Result<NormalizedNodeGraph, MaterialError> {
    let mut context_types = Vec::new();
    if accepts_uv0 {
        context_types.push((MaterialNodeInputSource::Uv0, MaterialType::Vec2));
    }
    context_types.extend(
        available_context
            .iter()
            .map(|&(source, _, ty)| (source, ty)),
    );
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
            &context_types,
        )?;
    }
    let order = evaluation_order(graph.output, &nodes)?;

    let uv = context_name("uv", functions);
    let context = available_context
        .iter()
        .copied()
        .filter(|(source, _, _)| {
            nodes
                .values()
                .any(|node| node.inputs.iter().any(|input| input.source == *source))
        })
        .map(|(source, name, ty)| (source, context_name(name, functions), ty))
        .collect::<Vec<_>>();
    let mut inputs = Vec::new();
    if accepts_uv0 {
        inputs.push(crate::FragmentInput {
            name: uv.clone(),
            ty: MaterialType::Vec2,
        });
    }
    inputs.extend(context.iter().map(|(_, name, ty)| crate::FragmentInput {
        name: name.clone(),
        ty: *ty,
    }));
    let parameters = inputs
        .iter()
        .map(|input| format!("{}: {}", input.name, input.ty.wgsl()))
        .collect::<Vec<_>>()
        .join(", ");
    let root = stage.node_name(output.id);
    let output = functions[&output.callable].output;
    let locals = order
        .iter()
        .map(|id| {
            (
                *id,
                context_name(&format!("{}_value", stage.node_name(*id)), functions),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut source = format!(
        "// Generated material node graph.\nfn {root}({parameters}) -> {} {{\n",
        output.wgsl(),
    );
    for id in order {
        let node = nodes[&id];
        let signature = &functions[&node.callable];
        let local = &locals[&id];
        source.push_str(&format!(
            "    let {local}: {} = {}(",
            signature.output.wgsl(),
            node.callable
        ));
        for (index, input) in signature.inputs.iter().enumerate() {
            if index != 0 {
                source.push_str(", ");
            }
            let connection = node
                .inputs
                .iter()
                .find(|connection| connection.input == input.name)
                .expect("validated node input");
            match connection.source {
                MaterialNodeInputSource::Node(source_node) => {
                    source.push_str(&locals[&source_node])
                }
                MaterialNodeInputSource::NodeOutput { node, output } => {
                    source.push_str(&locals[&node]);
                    source.push('.');
                    source.push_str(output.projection().0);
                }
                MaterialNodeInputSource::Uv0 => source.push_str(&uv),
                pbr_input => {
                    let (_, name, _) = context
                        .iter()
                        .find(|(kind, _, _)| *kind == pbr_input)
                        .expect("validated PBR context input");
                    source.push_str(name);
                }
            }
        }
        source.push_str(");\n");
    }
    source.push_str(&format!("    return {};\n}}\n", locals[&graph.output]));
    let callable = CallableDeclaration {
        name: root.clone(),
        arguments: inputs.iter().map(|input| input.ty.shader_type()).collect(),
        result: Some(output.shader_type()),
    };
    Ok(NormalizedNodeGraph {
        root,
        output,
        source,
        generated_names: nodes.keys().map(|id| stage.node_name(*id)).collect(),
        callables: vec![callable],
        inputs,
        pbr_inputs: context.into_iter().map(|(source, _, _)| source).collect(),
    })
}

fn validate_node(
    node: MaterialNodeId,
    callable: &str,
    inputs: &[crate::MaterialNodeInput],
    nodes: &BTreeMap<MaterialNodeId, &crate::MaterialNode>,
    functions: &BTreeMap<String, FunctionSignature>,
    context_types: &[(MaterialNodeInputSource, MaterialType)],
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
            MaterialNodeInputSource::Node(node) => upstream_type(node, nodes, functions)?,
            MaterialNodeInputSource::NodeOutput { node, output } => {
                let actual = upstream_type(node, nodes, functions)?;
                output
                    .ty(actual)
                    .ok_or(MaterialError::NodeOutputUnavailable {
                        node,
                        output,
                        actual,
                    })?
            }
            context => context_types
                .iter()
                .find_map(|(source, ty)| (*source == context).then_some(*ty))
                .ok_or_else(|| MaterialError::NodeContextUnavailable {
                    node,
                    input: connection.input.clone(),
                })?,
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

fn upstream_type(
    node: MaterialNodeId,
    nodes: &BTreeMap<MaterialNodeId, &crate::MaterialNode>,
    functions: &BTreeMap<String, FunctionSignature>,
) -> Result<MaterialType, MaterialError> {
    let source = nodes
        .get(&node)
        .ok_or(MaterialError::MissingNode { node })?;
    functions
        .get(&source.callable)
        .map(|signature| signature.output)
        .ok_or_else(|| MaterialError::MissingFragment {
            name: source.callable.clone(),
        })
}

fn evaluation_order(
    root: MaterialNodeId,
    nodes: &BTreeMap<MaterialNodeId, &crate::MaterialNode>,
) -> Result<Vec<MaterialNodeId>, MaterialError> {
    fn visit(
        node: MaterialNodeId,
        nodes: &BTreeMap<MaterialNodeId, &crate::MaterialNode>,
        visiting: &mut BTreeSet<MaterialNodeId>,
        visited: &mut BTreeSet<MaterialNodeId>,
        order: &mut Vec<MaterialNodeId>,
    ) -> Result<(), MaterialError> {
        if visited.contains(&node) {
            return Ok(());
        }
        if !visiting.insert(node) {
            return Err(MaterialError::NodeCycle { node });
        }
        let dependencies: BTreeSet<_> = nodes[&node]
            .inputs
            .iter()
            .filter_map(|input| match input.source {
                MaterialNodeInputSource::Node(source)
                | MaterialNodeInputSource::NodeOutput { node: source, .. } => Some(source),
                _ => None,
            })
            .collect();
        for source in dependencies {
            visit(source, nodes, visiting, visited, order)?;
        }
        visiting.remove(&node);
        visited.insert(node);
        order.push(node);
        Ok(())
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut order = Vec::with_capacity(nodes.len());
    visit(root, nodes, &mut visiting, &mut visited, &mut order)?;
    let reachable = order.len();
    // Unused nodes still participate in authoring validation.
    for node in nodes.keys().copied() {
        visit(node, nodes, &mut visiting, &mut visited, &mut order)?;
    }
    order.truncate(reachable);
    Ok(order)
}

impl MaterialNodeOutput {
    fn projection(self) -> (&'static str, usize, MaterialType) {
        use MaterialType::{F32, Vec2, Vec3, Vec4};
        match self {
            Self::X => ("x", 1, F32),
            Self::Y => ("y", 2, F32),
            Self::Z => ("z", 3, F32),
            Self::W => ("w", 4, F32),
            Self::Xy => ("xy", 2, Vec2),
            Self::Zw => ("zw", 4, Vec2),
            Self::Xyz => ("xyz", 3, Vec3),
            Self::R => ("r", 1, F32),
            Self::G => ("g", 2, F32),
            Self::B => ("b", 3, F32),
            Self::A => ("a", 4, F32),
            Self::Rgb => ("rgb", 3, Vec3),
            Self::Rgba => ("rgba", 4, Vec4),
        }
    }

    fn ty(self, source: MaterialType) -> Option<MaterialType> {
        let (_, minimum_width, result) = self.projection();
        let width = match source {
            MaterialType::Vec2 => 2,
            MaterialType::Vec3 => 3,
            MaterialType::Vec4 => 4,
            _ => return None,
        };
        (width >= minimum_width).then_some(result)
    }
}

fn context_name(seed: &str, functions: &BTreeMap<String, FunctionSignature>) -> String {
    let mut name = seed.to_owned();
    while functions.contains_key(&name) {
        name.push('_');
    }
    name
}
