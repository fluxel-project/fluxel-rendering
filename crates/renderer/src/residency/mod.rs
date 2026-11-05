//! GPU residency kept alive until RHI completion evidence proves retirement safe.

//! Per-submission GPU buffers retained through RHI completion.

mod deferred;
mod deformation;
pub(crate) use deformation::{DeferredDeformation, DeformationAllocationPlan};
mod forward;
pub(crate) use deferred::{
    DeferredDraw, DeferredFrame, DeferredSurfaceFrame, plan_deferred_gbuffer,
};
mod forward_cache;
pub(crate) use forward_cache::{
    ForwardGeometryCacheKey, ForwardResidentCache, ForwardTextureCacheKey,
};
mod upload;
pub(crate) use forward::{
    ForwardBindGroup, ForwardDraw, ForwardFrame, ForwardFrameBuffers, ForwardResidentBuffers,
    ForwardResidentUpload, ForwardStreamInput, ForwardTexture, ForwardTextureInput,
    ForwardTextureUpload, ForwardUniformUpload, ForwardVertexStream, SurfaceForwardFrame,
    SurfaceForwardFrameBuffers, plan_forward_frame, plan_forward_resident,
    plan_surface_forward_frame,
};

pub(crate) use upload::{
    ComputeBuffers, RasterBuffers, ReadSetRasterBuffers, SurfaceRasterBuffers, UploadBuffers,
};
