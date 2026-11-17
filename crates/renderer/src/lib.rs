//! Scene data, preparation, frame policy, and the private RenderGraph.
//!
//! This crate is being rebuilt from the current workspace architecture. The
//! initial Copy, offscreen and Surface Raster, and supported Compute fixtures
//! exercise asset retention and RHI completion. Complete rendering and extension contracts
//! remain under development.

#![deny(missing_docs)]

mod atomic;
mod feature;
mod frame_data;
mod graph;
mod pbr;
mod pipeline;
mod preparation;
mod render_scene;
mod renderer;
mod residency;

pub use feature::{
    CpuFeature, CpuPhase, CpuPhaseContribution, CpuPhaseContributions, DeferredPositionDeformer,
    DeformationBinding, FeatureBinding, FeatureContext, FeatureError, FeatureHandle, FeatureIssuer,
    PositionDeformation, RenderFeature, TypedFeatureSnapshot, TypedFeatureStorage,
    validate_context,
};
pub use pipeline::{
    BoundsVisibility, ClearForwardPass, DeferredCpuPipeline, DeferredDeformationPass,
    DeferredGeometryPass, DeferredLightingPass, DeferredPhaseContext, ForwardCompositionContext,
    ForwardCompositionPass, ForwardCpuPipeline, ForwardDepthMode, ForwardFeaturePass,
    ForwardPhaseContext, FramePipeline, FrameTopology, OpaqueForwardPass, PipelineError,
    PreparedFrame, VisibilityProvider,
};
pub use preparation::{
    ColorTextureAsset, ColorTextureData, EnvironmentIrradianceAsset, EnvironmentIrradianceData,
    EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, ForwardAssets,
    ForwardCaptureResources, ForwardDrawInput, ForwardDrawOrder, ForwardDrawRequest,
    ForwardDrawSource, ForwardPreparationError, ForwardResources, ForwardTextureSource,
    GeometryAsset, GeometryAttribute, GeometryBufferOwner, GeometryData, GeometryDraw,
    GeometryError, GeometryLayout, GeometryStream, GeometryStreamBinding, LinearSpecularLod,
    MaterialPreparationError, PreparedCaptureReceiver, PreparedGeometry, PreparedGeometryBindings,
    PreparedMaterial, SpecularIblImage, SpecularIblMip, prepare_forward,
    prepare_forward_capture_receiver, prepare_material,
};
pub use render_scene::{
    ChangeReceipt, DirtyRange, FrameSnapshot, RenderBounds, RenderChange, RenderChangeSet,
    RenderId, RenderRecord, RenderResources, RenderScene, RenderTransform, RenderView,
    RenderViewId, SceneAssetRef, SceneCapacity, SceneDirty, SceneError, ShadingCamera,
};
pub use renderer::{
    CaptureTextureSlot, DeferredCacheStats, DeferredProbeFacts, ForwardCacheStats, Renderer,
    RendererError, ShaderAdmissionError, ShutdownReport, Submission,
};
