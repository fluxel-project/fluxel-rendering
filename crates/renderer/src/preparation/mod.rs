//! Visibility, ordering, batching, draw preparation, and concrete bindings.
//!
//! Preparation resolves Material and Shader products without taking ownership
//! of their semantics. Concrete buffer and texture uses must reach Graph.

mod environment;
mod forward;
mod forward_sources;
mod geometry;
mod geometry_key;
pub use environment::{EnvironmentIrradianceAsset, EnvironmentIrradianceData};
#[cfg(test)]
pub(crate) use forward::prepare_feature_draws;
pub use forward::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCaptureResources, ForwardDrawInput,
    ForwardDrawRequest, ForwardPreparationError, ForwardResources, ForwardTextureSource,
    GeometryAsset, GeometryData, PreparedCaptureReceiver, prepare_forward,
    prepare_forward_capture_receiver,
};
pub(crate) use forward::{
    ForwardBatchStats, ForwardCompatibilityKey, batch_forward_draws,
    prepare_feature_draws_with_environment, prepare_forward_with_environment,
};
pub use forward_sources::{ForwardDrawOrder, ForwardDrawSource};
pub(crate) use geometry_key::{VertexLayoutKey, vertex_layout_key};
mod material;
pub use geometry::{
    GeometryAttribute, GeometryBufferOwner, GeometryDraw, GeometryError, GeometryLayout,
    GeometryStream, GeometryStreamBinding, PreparedGeometry, PreparedGeometryBindings,
};
pub use material::{MaterialPreparationError, PreparedMaterial, prepare_material};

#[cfg(test)]
#[path = "../tests/preparation/forward_capture.rs"]
mod capture_tests;
