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
    GpuIndirectDraw, PositionDeformation, RenderFeature, TypedFeatureSnapshot, TypedFeatureStorage,
    validate_context,
};
pub use pipeline::{
    BoundsVisibility, ClearForwardPass, DeferredCpuPipeline, DeferredDeformationPass,
    DeferredGeometryPass, DeferredLightingPass, DeferredPhaseContext, DualDepthPeelingConfig,
    ForwardCompositionContext, ForwardCompositionPass, ForwardCpuPipeline, ForwardDepthMode,
    ForwardFeaturePass, ForwardPairCompositionContext, ForwardPairCompositionPass,
    ForwardPhaseContext, FramePipeline, FrameTopology, OpaqueForwardPass, PipelineError,
    PreparedFrame, SortedForwardPass, VisibilityProvider,
};
pub use preparation::{
    ColorTextureArrayAsset, ColorTextureArrayData, ColorTextureAsset, ColorTextureData,
    ColorTextureVolumeAsset, ColorTextureVolumeData, DepthTextureArrayAsset, DepthTextureArrayData,
    DepthTextureAsset, DepthTextureData, EnvironmentIrradianceAsset, EnvironmentIrradianceData,
    EnvironmentSpecularIblAsset, EnvironmentSpecularIblData, ForwardAssets,
    ForwardCaptureResources, ForwardDrawInput, ForwardDrawOrder, ForwardDrawRequest,
    ForwardDrawSource, ForwardPreparationError, ForwardResources, ForwardSamplerBinding,
    ForwardTextureBinding, ForwardTextureSource, GeometryAsset, GeometryAttribute,
    GeometryBufferOwner, GeometryData, GeometryDraw, GeometryError, GeometryLayout, GeometryStream,
    GeometryStreamBinding, LinearSpecularLod, MaterialPreparationError, PreparedCaptureReceiver,
    PreparedGeometry, PreparedGeometryBindings, PreparedMaterial, RefractionRadianceAsset,
    RefractionRadianceData, SpecularIblImage, SpecularIblMip, TextureEncoding, prepare_forward,
    prepare_forward_capture_receiver, prepare_forward_capture_receiver_with_instance,
    prepare_forward_scene_refraction_receiver,
    prepare_forward_scene_refraction_receiver_with_sampler, prepare_material,
};
pub use render_scene::{
    ChangeReceipt, DirectionalLight, DirtyRange, FrameSnapshot, MeshVisibility, RenderBounds,
    RenderChange, RenderChangeSet, RenderId, RenderRecord, RenderResources, RenderScene,
    RenderTransform, RenderView, RenderViewId, SceneAssetRef, SceneCapacity, SceneDirty,
    SceneError, ShadingCamera,
};
pub use renderer::{
    CaptureTextureSlot, DeferredCacheStats, DeferredProbeFacts, ForwardCacheStats,
    OutputProcessing, OutputTransfer, Renderer, RendererError, ShaderAdmissionError,
    ShutdownReport, Submission,
};
