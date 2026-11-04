//! GPU residency kept alive until RHI completion evidence proves retirement safe.

//! Per-submission GPU buffers retained through RHI completion.

mod forward;
mod forward_cache;
pub(crate) use forward_cache::{
    ForwardGeometryCacheKey, ForwardResidentCache, ForwardTextureCacheKey,
};
mod upload;
pub(crate) use forward::{
    ForwardBindGroup, ForwardDraw, ForwardFrame, ForwardFrameBuffers, ForwardResidentBuffers,
    ForwardStreamInput, ForwardTexture, ForwardTextureInput, ForwardTextureUpload,
    ForwardUniformUpload, ForwardVertexStream, SurfaceForwardFrame, SurfaceForwardFrameBuffers,
    plan_forward_frame, plan_forward_resident, plan_surface_forward_frame,
};

pub(crate) use upload::{
    ComputeBuffers, RasterBuffers, ReadSetRasterBuffers, SurfaceRasterBuffers, UploadBuffers,
};
