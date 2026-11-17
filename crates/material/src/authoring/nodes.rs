//! Runtime material node identities and connections.

/// Stable identity of one runtime node in a material authoring graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaterialNodeId(pub u32);

/// The value supplied to one material-node input port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialNodeInputSource {
    /// Output of another node in the same graph.
    Node(MaterialNodeId),
    /// The mesh UV0 coordinate supplied by the UV0 surface evaluator.
    Uv0,
}

/// One connection from an upstream value to one named callable input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialNodeInput {
    /// Input port name on the receiving node's callable.
    pub input: String,
    /// Value that supplies that port.
    pub source: MaterialNodeInputSource,
}

/// One callable invocation in a material node graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialNode {
    /// Stable identity within this graph.
    pub id: MaterialNodeId,
    /// Existing Material fragment or Code Node callable invoked by this node.
    pub callable: String,
    /// Named incoming connections for every callable argument.
    pub inputs: Vec<MaterialNodeInput>,
}

/// Typed runtime topology for one material evaluation.
///
/// Nodes invoke existing material callables. Compilation validates each named
/// connection, then generates wrappers for the selected evaluator context; it
/// does not interpret this graph during a frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialNodeGraph {
    /// Callable-invocation nodes.
    pub nodes: Vec<MaterialNode>,
    /// Node whose result supplies the selected surface output.
    pub output: MaterialNodeId,
}
