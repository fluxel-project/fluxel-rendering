//! GPU residency kept alive until RHI completion evidence proves retirement safe.

//! Per-submission GPU buffers retained through RHI completion.

mod upload;

pub(crate) use upload::{
    ComputeBuffers, RasterBuffers, ReadSetRasterBuffers, SurfaceRasterBuffers, UploadBuffers,
};
