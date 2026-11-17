//! First-class author-provided WGSL material nodes.

use crate::{FragmentCall, FragmentInput, MaterialResource, MaterialType};

/// One typed material function supplied as an independently attributed WGSL unit.
///
/// The node remains part of the same Material call graph as legacy fragments.
/// Its source is passed unchanged to Shader composition so diagnostics retain
/// the author's source-unit identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialCodeNode {
    /// Material-callable name implemented by this node.
    pub name: String,
    /// Typed arguments required by the callable.
    pub inputs: Vec<FragmentInput>,
    /// Result type returned by the callable.
    pub output: MaterialType,
    /// Explicit material-call edges used for deterministic validation.
    pub calls: Vec<FragmentCall>,
    /// Exact author WGSL and its diagnostic identity.
    pub source: fluxel_shader::SourceUnit,
    /// Resources declared by this node in addition to definition-level resources.
    pub resources: Vec<MaterialResource>,
}
