//! Resolves and emits the typed output interface of a vertex node graph.

use super::{PropertyRoot, node_value_type, property_expression};
use crate::{
    MaterialError, MaterialNode, MaterialNodeId, MaterialNodeInputSource, MaterialType,
    MaterialVertexVarying,
};
use std::{collections::BTreeMap, fmt::Write};

use super::FunctionSignature;

pub(super) struct ResolvedVarying {
    pub(super) id: u32,
    pub(super) root: PropertyRoot,
    pub(super) ty: MaterialType,
}

pub(super) fn resolve(
    varyings: &[MaterialVertexVarying],
    nodes: &BTreeMap<MaterialNodeId, &MaterialNode>,
    functions: &BTreeMap<String, FunctionSignature>,
) -> Result<Vec<ResolvedVarying>, MaterialError> {
    let mut resolved = BTreeMap::new();
    for varying in varyings {
        let id = varying.id;
        let root = match varying.source {
            MaterialNodeInputSource::Node(node) => (node, None),
            MaterialNodeInputSource::NodeOutput { node, output } => (node, Some(output)),
            _ => return Err(MaterialError::InvalidVertexVaryingSource { id }),
        };
        let ty = node_value_type(root, nodes, functions)?;
        if resolved
            .insert(id, ResolvedVarying { id, root, ty })
            .is_some()
        {
            return Err(MaterialError::DuplicateVertexVarying { id });
        }
    }
    Ok(resolved.into_values().collect())
}

pub(super) fn append_type(source: &mut String, outputs: &[ResolvedVarying]) -> &'static str {
    source.push_str("struct FluxelVertexVaryings { uv0: vec2<f32>,\n");
    for output in outputs {
        writeln!(source, "    varying_{}: {},", output.id, output.ty.wgsl())
            .expect("write to string");
    }
    source.push_str("};\n");
    "FluxelVertexVaryings"
}

pub(super) fn append_result(
    source: &mut String,
    outputs: &[ResolvedVarying],
    uv0: &str,
    locals: &BTreeMap<MaterialNodeId, String>,
) {
    write!(source, "    return FluxelVertexVaryings({uv0}").expect("write to string");
    for output in outputs {
        write!(source, ", {}", property_expression(output.root, locals)).expect("write to string");
    }
    source.push_str(");\n}\n");
}
