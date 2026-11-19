//! Runtime material node identities and connections.

/// Stable identity of one runtime node in a material authoring graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaterialNodeId(pub u32);

/// A named output port projected from a float vector returned by a node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialNodeOutput {
    /// First vector component.
    X,
    /// Second vector component.
    Y,
    /// Third vector component.
    Z,
    /// Fourth vector component.
    W,
    /// First two vector components.
    Xy,
    /// Last two components of a four-component vector.
    Zw,
    /// First three vector components.
    Xyz,
    /// Red component.
    R,
    /// Green component.
    G,
    /// Blue component.
    B,
    /// Alpha component.
    A,
    /// Red, green and blue components.
    Rgb,
    /// Red, green, blue and alpha components.
    Rgba,
}

/// The value supplied to one material-node input port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialNodeInputSource {
    /// Output of another node in the same graph.
    Node(MaterialNodeId),
    /// Named output of a vector-valued node in the same graph.
    NodeOutput {
        /// Node which produces the vector.
        node: MaterialNodeId,
        /// Output port selected from that vector.
        output: MaterialNodeOutput,
    },
    /// The mesh UV0 coordinate supplied by the UV0 surface evaluator.
    Uv0,
    /// Interpolated XYZ in Fluxel world space, supplied by the UV0 PBR evaluator.
    WorldPosition,
    /// Normalized, interpolated geometric mesh normal in Fluxel world space.
    /// The UV0 PBR evaluator provides it before sidedness and normal mapping.
    WorldNormal,
    /// Actual camera eye position in Fluxel world space, supplied by the UV0 PBR evaluator.
    CameraPosition,
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
/// connection, then emits the output's dependencies in evaluation order. All
/// ports of the same upstream node use the result of one invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialNodeGraph {
    /// Callable-invocation nodes.
    pub nodes: Vec<MaterialNode>,
    /// Node whose result supplies the selected surface output.
    pub output: MaterialNodeId,
}
