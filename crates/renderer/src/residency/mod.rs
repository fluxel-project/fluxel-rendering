//! GPU residency kept alive until RHI completion evidence proves retirement safe.

//! Per-submission GPU buffers retained through RHI completion.

mod deferred;
mod deformation;
mod environment;
mod oit;
mod specular_environment;
pub(crate) use deformation::{DeferredDeformation, DeformationAllocationPlan};
pub(crate) use specular_environment::{SpecularEnvironmentFrameOwner, SpecularIblCopyFootprint};
mod forward;
#[cfg(test)]
pub(crate) use deferred::allocate_deferred_scene_color;
pub(crate) use deferred::{
    DeferredDraw, DeferredFrame, DeferredPublication, DeferredSurfaceFrame,
    allocate_deferred_scene_color_desc, deferred_albedo_format,
    deferred_scene_color_desc_with_format, plan_deferred_gbuffer_with_effects,
};
pub(crate) use environment::EnvironmentFrameOwner;
pub(crate) use oit::{OitDepthPrecision, OitOwners, OitPingPongOwners, OitPlan, OitTextureOwner};
mod forward_cache;
pub(crate) use forward_cache::{
    ForwardGeometryCacheKey, ForwardResidentCache, ForwardTextureCacheKey,
};
mod upload;
pub(crate) use forward::{
    ForwardBindGroup, ForwardDepthAttachment, ForwardDraw, ForwardFrame, ForwardFrameBuffers,
    ForwardResidentBuffers, ForwardResidentUpload, ForwardSampler, ForwardStreamInput,
    ForwardTexture, ForwardTextureInput, ForwardTextureUpload, ForwardUniformUpload,
    ForwardVertexStream, SurfaceForwardFrame, SurfaceForwardFrameBuffers, plan_forward_frame,
    plan_forward_hdr_frame, plan_forward_resident, plan_surface_forward_frame,
};

pub(crate) use upload::{
    ComputeBuffers, RasterBuffers, ReadSetRasterBuffers, SurfaceRasterBuffers, UploadBuffers,
};
