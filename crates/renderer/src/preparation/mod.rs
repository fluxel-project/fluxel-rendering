//! Visibility, ordering, batching, draw preparation, and concrete bindings.
//!
//! Preparation resolves Material and Shader products without taking ownership
//! of their semantics. Concrete buffer and texture uses must reach Graph.

mod forward;
mod forward_sources;
mod geometry;
mod geometry_key;
pub use forward::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardDrawInput, ForwardDrawRequest,
    ForwardPreparationError, ForwardResources, GeometryAsset, GeometryData, prepare_forward,
};
pub(crate) use forward::{
    ForwardBatchStats, ForwardCompatibilityKey, batch_forward_draws, prepare_feature_draws,
};
pub use forward_sources::{ForwardDrawOrder, ForwardDrawSource};
pub(crate) use geometry_key::{VertexLayoutKey, vertex_layout_key};
mod material;
pub use geometry::{
    GeometryAttribute, GeometryBufferOwner, GeometryDraw, GeometryError, GeometryLayout,
    GeometryStream, GeometryStreamBinding, PreparedGeometry, PreparedGeometryBindings,
};
pub use material::{MaterialPreparationError, PreparedMaterial, prepare_material};
