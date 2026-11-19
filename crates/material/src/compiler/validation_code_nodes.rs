//! Code-node source identity validation.

use std::collections::BTreeSet;

use crate::{MaterialCodeNode, MaterialError};

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
