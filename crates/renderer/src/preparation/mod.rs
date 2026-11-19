//! Visibility, ordering, batching, draw preparation, and concrete bindings.
//!
//! Preparation resolves Material and Shader products without taking ownership
//! of their semantics. Concrete buffer and texture uses must reach Graph.

mod environment;
mod specular_environment;
pub use specular_environment::{
    EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, LinearSpecularLod,
    RefractionRadianceAsset, RefractionRadianceData, SpecularIblImage, SpecularIblMip,
};
mod forward;
pub(crate) mod forward_fragment;
mod forward_instances;
pub(crate) mod particle_material;
pub(crate) use forward_instances::{mesh_vertex_layout, with_instance_transform};
mod forward_sampler;
mod forward_schema;
pub(crate) mod forward_templates;
pub(crate) use forward_schema::{
    MaterialBindingSchema, MaterialTextureAccess, MaterialTextureBinding, material_binding_schema,
};
mod forward_sources;
mod geometry;
mod geometry_key;
pub use environment::{EnvironmentIrradianceAsset, EnvironmentIrradianceData};
pub(crate) use forward::ResolvedForwardTexture;
#[cfg(test)]
pub(crate) use forward::prepare_feature_draws;
pub(crate) use forward::prepare_particle_draw;
pub use forward::{
    ColorTextureArrayAsset, ColorTextureArrayData, ColorTextureAsset, ColorTextureData,
    ColorTextureVolumeAsset, ColorTextureVolumeData, DepthTextureArrayAsset, DepthTextureArrayData,
    DepthTextureAsset, DepthTextureData, ForwardAssets, ForwardCaptureResources, ForwardDrawInput,
    ForwardDrawRequest, ForwardPreparationError, ForwardResources, ForwardSamplerBinding,
    ForwardTextureBinding, ForwardTextureSource, GeometryAsset, GeometryData,
    PreparedCaptureReceiver, TextureEncoding, prepare_forward, prepare_forward_capture_receiver,
    prepare_forward_capture_receiver_with_instance, prepare_forward_scene_refraction_receiver,
    prepare_forward_scene_refraction_receiver_with_sampler,
};
pub(crate) use forward::{
    ForwardBatchStats, ForwardCompatibilityKey, OitDrawPartition, batch_forward_draws,
    partition_oit_draws, prepare_feature_draws_with_lighting, prepare_forward_with_lighting,
    prepare_oit_material,
};
pub use forward_sources::{ForwardDrawOrder, ForwardDrawSource};
pub(crate) use geometry_key::{VertexLayoutKey, vertex_layout_key};
mod material;
mod vertex_varyings;
pub use geometry::{
    GeometryAttribute, GeometryBufferOwner, GeometryDraw, GeometryError, GeometryLayout,
    GeometryStream, GeometryStreamBinding, PreparedGeometry, PreparedGeometryBindings,
};
pub(crate) use material::prepare_material_for_alpha;
pub use material::{MaterialPreparationError, PreparedMaterial, prepare_material};

#[cfg(test)]
#[path = "../tests/preparation/forward_capture.rs"]
mod capture_tests;
