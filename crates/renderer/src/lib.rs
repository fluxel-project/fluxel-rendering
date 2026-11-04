//! Scene data, preparation, frame policy, and the private RenderGraph.
//!
//! This crate is being rebuilt from the current workspace architecture. The
//! initial Copy, offscreen and Surface Raster, and supported Compute fixtures
//! exercise asset retention and RHI completion. Complete rendering and extension contracts
//! remain under development.

#![deny(missing_docs)]

mod feature;
mod frame_data;
mod graph;
mod pipeline;
mod preparation;
mod render_scene;
mod renderer;
mod residency;

pub use feature::{
    CpuFeature, CpuPhase, CpuPhaseContribution, CpuPhaseContributions, FeatureContext,
    FeatureError, FeatureHandle, FeatureIssuer, TypedFeatureSnapshot, TypedFeatureStorage,
    validate_context,
};
pub use pipeline::{
    BoundsVisibility, ClearForwardPass, ForwardCpuPipeline, ForwardPhaseContext, FramePipeline,
    FrameTopology, OpaqueForwardPass, PipelineError, PreparedFrame, VisibilityProvider,
};
pub use preparation::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardDrawInput, ForwardPreparationError,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryBufferOwner, GeometryData,
    GeometryDraw, GeometryError, GeometryLayout, GeometryStream, GeometryStreamBinding,
    MaterialPreparationError, PreparedGeometry, PreparedGeometryBindings, PreparedMaterial,
    prepare_forward, prepare_material,
};
pub use render_scene::{
    ChangeReceipt, DirtyRange, FrameSnapshot, RenderBounds, RenderChange, RenderChangeSet,
    RenderId, RenderRecord, RenderResources, RenderScene, RenderTransform, RenderView,
    RenderViewId, SceneAssetRef, SceneCapacity, SceneDirty, SceneError,
};
pub use renderer::{
    ForwardCacheStats, Renderer, RendererError, ShaderAdmissionError, ShutdownReport, Submission,
};
