//! Code-node source identity validation.

use std::collections::{BTreeMap, BTreeSet};

use crate::{FragmentCall, FragmentInput, MaterialCodeNode, MaterialError, MaterialType};

#[derive(Clone, Debug)]
pub(super) struct FunctionSignature {
    pub(super) inputs: Vec<FragmentInput>,
    pub(super) output: MaterialType,
    pub(super) calls: Vec<FragmentCall>,
}

pub(super) fn validate_code_node_sources(nodes: &[MaterialCodeNode]) -> Result<(), MaterialError> {
    let mut ids = BTreeSet::new();
    for node in nodes {
        if node.source.id.0.is_empty() {
            return Err(MaterialError::EmptyCodeNodeSourceId {
                node: node.name.clone(),
            });
        }
        if !ids.insert(node.source.id.0.as_str()) {
            return Err(MaterialError::DuplicateCodeNodeSourceId {
                source_id: node.source.id.0.clone(),
            });
        }
    }
    Ok(())
}

pub(super) fn visit(
    name: &str,
    functions: &BTreeMap<String, FunctionSignature>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<(), MaterialError> {
    if visited.contains(name) {
        return Ok(());
    }
    if !visiting.insert(name.to_owned()) {
        return Err(MaterialError::CallCycle {
            fragment: name.to_owned(),
        });
    }
    for call in &functions.get(name).expect("validated function name").calls {
        visit(&call.fragment, functions, visiting, visited)?;
    }
    visiting.remove(name);
    visited.insert(name.to_owned());
    Ok(())
}
