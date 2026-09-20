//! DX12 root-signature lowering for one portable pipeline interface.
//!
//! A root signature is backend-private state assembled at pipeline creation; it
//! is not a public `PipelineInterface` object. The frozen portable descriptor
//! already guarantees device identity and logical compatibility before this seam
//! is reached.

use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12RootSignature};

use crate::api::binding::BindGroupLayout;
use crate::backend::dx12::failure::Dx12Failure;

/// Lowers the ordered portable group layouts into a DX12 root signature.
///
/// An empty group sequence is valid: it lowers to a root signature with no
/// parameters when this implementation is completed.
pub(crate) fn build_root_signature(
    _device: &ID3D12Device,
    _groups: &[BindGroupLayout],
) -> Result<ID3D12RootSignature, Dx12Failure> {
    unimplemented!("DX12 root-signature lowering is not implemented")
}
