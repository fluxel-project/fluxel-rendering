//! Resource allocation on Direct3D 12.
//!
//! Buffer allocation, readback mapping, and transient allocation facts. Textures,
//! texture views and samplers join as their own files once their native lowering
//! exists; no public RHI handle exposes an ID3D12 resource while that work is
//! pending.
//!
//! The re-exports below are the chapter's inside face: `dx12` is a private module
//! of the crate, so `pub(crate)` here is narrower than it looks and reaches only
//! this backend's other chapters. They exist so that a caller says
//! `resource::create_buffer` rather than naming the file, which is what lets
//! [`buffer`] be split further without touching its callers.

mod buffer;
mod query;
mod readback;
mod sampler;
mod texture;
mod transient;
mod view;

pub(crate) use buffer::{
    Dx12Buffer, Dx12BufferHeap, StagingHeap, create_buffer, create_staging, map_buffer,
    map_resource,
};
pub(crate) use query::{Dx12QuerySet, create_query_set};
pub(crate) use readback::readback_bytes;
pub(crate) use sampler::{Dx12Sampler, create_sampler};
pub(crate) use texture::{Dx12Texture, create_texture};
pub(crate) use transient::transient_capabilities;
pub(crate) use view::{Dx12TextureView, create_texture_view};
