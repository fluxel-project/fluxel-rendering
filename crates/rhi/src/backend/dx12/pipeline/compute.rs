//! DX12 compute-pipeline seam.
//!
//! The portable API owns validation and the opaque pipeline handle. This module
//! owns only the native state that a future DX12 command lowerer will downcast.
//! The creation path is deliberately present now so its ownership does not drift
//! into `platform`, but actual root-signature/PSO lowering is not implemented.

use std::any::Any;

use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use crate::api::pipeline::ComputePipelineDescriptor;
use crate::api::pipeline::backend::ComputePipelineBackend;
use crate::backend::dx12::failure::Dx12Failure;

/// Native state behind a portable compute pipeline.
///
/// Fields are added only when the lowering can create and command submission can
/// consume the corresponding DX12 objects; an empty placeholder must not claim a
/// successfully created PSO.
pub(crate) struct Dx12ComputePipeline {
    _private: (),
}

impl ComputePipelineBackend for Dx12ComputePipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Lowers a validated portable compute descriptor to DX12.
///
/// This is intentionally not a compatibility fallback: no native object is
/// fabricated while root-signature and PSO lowering are absent.
pub(crate) fn create_compute_pipeline(
    device: &ID3D12Device,
    descriptor: &ComputePipelineDescriptor,
) -> Result<Dx12ComputePipeline, Dx12Failure> {
    let _root_signature =
        super::interface::build_root_signature(device, &descriptor.interface.descriptor().groups)?;
    unimplemented!("DX12 compute pipeline lowering is not implemented")
}
