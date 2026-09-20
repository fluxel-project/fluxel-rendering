//! Pipelines on Direct3D 12: root signatures and state objects.
//!
//! # Why both portable pipeline verbs do their native work here
//!
//! Direct3D 12 has neither a bind-group-layout object nor a pipeline-interface
//! object. The analogue of both is the *root signature*, and a root signature is
//! a property of a pipeline: it is built from the whole ordered group sequence at
//! once, serialized, and attached to a state object. So
//! `create_pipeline_interface` makes no backend call at all — it interns a
//! portable description — and everything native happens the first time a
//! pipeline is created from it.
//!
//! # What the driver is asked, and when
//!
//! `CreateComputePipelineState` is where the driver's shader compiler runs, and
//! therefore where a malformed or unsupported program is refused. That is why
//! [`crate::backend::dx12::shader`]'s `create_shader` proves nothing about
//! bytecode: this chapter is the first place a driver's opinion of the DXIL is
//! available, and a failure from here is the first in the crate that can be about
//! the *program* rather than about a descriptor. It is reported with its native
//! kind and never folded into `InvalidUsage`.
//!
//! Two Direct3D 12 behaviours shape what this chapter must do rather than what it
//! may skip. First, `D3D12SerializeRootSignature` and `CreateRootSignature` can
//! each fail, and the failure carries the serializer's own diagnostic blob, which
//! is the only description of *why* a root signature was refused. Second, when a
//! shader carries no embedded root signature, D3D12 does **not** check the state
//! object against the root signature bound at command-list time — a mismatch is
//! silent and shows up as a wrong read at dispatch. The mapping between the group
//! sequence and the descriptor tables is therefore the one thing this chapter and
//! [`crate::backend::dx12::binding`] must agree on exactly, which is why the table
//! shape is computed by one function that both call.
//!
//! The compute lowering owns the only currently frozen native pipeline seam.
//! Raster pipelines do not yet have a backend seam in `api::pipeline::backend`, so this
//! module deliberately does not invent a DX12-only raster handle for them.

mod compute;
mod interface;
mod raster;

pub(crate) use compute::{Dx12ComputePipeline, create_compute_pipeline};
pub(crate) use raster::{Dx12RasterPipeline, create_raster_pipeline};
