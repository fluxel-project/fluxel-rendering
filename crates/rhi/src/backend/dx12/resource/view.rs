//! DX12 sampled-texture descriptors.
//!
//! A portable `TextureView` does not encode *how* it will be used. DX12 has a
//! different descriptor type for SRV/UAV/RTV/DSV, so this first native object is
//! deliberately an SRV descriptor for sampled bindings; attachment and storage
//! descriptors are emitted by the binding/attachment lowering where that use is
//! known.

use std::any::Any;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_SHADER_RESOURCE_VIEW_DESC,
    D3D12_SHADER_RESOURCE_VIEW_DESC_0, D3D12_SRV_DIMENSION_TEXTURE1D,
    D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
    D3D12_SRV_DIMENSION_TEXTURE2DMS, D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
    D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_SRV_DIMENSION_TEXTURECUBE,
    D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, D3D12_TEX1D_SRV, D3D12_TEX2D_ARRAY_SRV, D3D12_TEX2D_SRV,
    D3D12_TEX2DMS_ARRAY_SRV, D3D12_TEX2DMS_SRV, D3D12_TEX3D_SRV, D3D12_TEXCUBE_ARRAY_SRV,
    D3D12_TEXCUBE_SRV, ID3D12DescriptorHeap, ID3D12Device,
};

use super::Dx12Texture;
use crate::api::resource::backend::TextureViewBackend;
use crate::api::resource::texture::TextureDescriptor;
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::backend::dx12::{ffi, platform::facts::dxgi_format};

/// An owned, CPU-visible SRV descriptor. The heap is intentionally retained:
/// D3D12 descriptor handles are addresses, not references.
pub(crate) struct Dx12TextureView {
    _heap: ID3D12DescriptorHeap,
    cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
}

impl Dx12TextureView {
    pub(crate) fn cpu(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.cpu
    }
}

impl TextureViewBackend for Dx12TextureView {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn create_texture_view(
    device: &ID3D12Device,
    texture: &Dx12Texture,
    base: &TextureDescriptor,
    descriptor: &TextureViewDescriptor,
) -> Result<Dx12TextureView, ffi::NativeError> {
    let format = dxgi_format(descriptor.format.unwrap_or(base.format)).ok_or_else(|| {
        ffi::NativeError::driver_contract_violation(
            "DX12 has no exact DXGI representation for this texture view format",
            "Device::create_texture_view",
        )
    })?;
    let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the heap description is a live local and the returned COM object
    // owns the descriptor storage until the view drops.
    let heap = unsafe { device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&heap_desc) }
        .map_err(|error| ffi::NativeError::new(&error, "Device::create_texture_view"))?;
    // SAFETY: this getter only reads the descriptor heap's immutable start address.
    let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let srv = srv_desc(format, base, descriptor);
    // SAFETY: `texture` and `srv` live across the call, and `cpu` points at the
    // sole slot of the retained CPU-visible CBV/SRV/UAV heap.
    unsafe { device.CreateShaderResourceView(texture.resource(), Some(&srv), cpu) };
    Ok(Dx12TextureView { _heap: heap, cpu })
}

fn srv_desc(
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    base: &TextureDescriptor,
    view: &TextureViewDescriptor,
) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut result = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV::default(),
        },
    };
    result.Anonymous = match view.dimension {
        TextureViewDimension::D1 => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture1D: D3D12_TEX1D_SRV {
                MostDetailedMip: view.base_mip,
                MipLevels: view.mip_count,
                ResourceMinLODClamp: 0.0,
            },
        },
        TextureViewDimension::D2 if base.sample_count == 1 => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: view.base_mip,
                MipLevels: view.mip_count,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
        TextureViewDimension::D2 => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DMS: D3D12_TEX2DMS_SRV {
                UnusedField_NothingToDefine: 0,
            },
        },
        TextureViewDimension::D2Array if base.sample_count == 1 => {
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: view.base_mip,
                    MipLevels: view.mip_count,
                    FirstArraySlice: view.base_layer,
                    ArraySize: view.layer_count,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            }
        }
        TextureViewDimension::D2Array => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DMSArray: D3D12_TEX2DMS_ARRAY_SRV {
                FirstArraySlice: view.base_layer,
                ArraySize: view.layer_count,
            },
        },
        TextureViewDimension::Cube => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCube: D3D12_TEXCUBE_SRV {
                MostDetailedMip: view.base_mip,
                MipLevels: view.mip_count,
                ResourceMinLODClamp: 0.0,
            },
        },
        TextureViewDimension::CubeArray => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                MostDetailedMip: view.base_mip,
                MipLevels: view.mip_count,
                First2DArrayFace: view.base_layer,
                NumCubes: view.layer_count / 6,
                ResourceMinLODClamp: 0.0,
            },
        },
        TextureViewDimension::D3 => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_SRV {
                MostDetailedMip: view.base_mip,
                MipLevels: view.mip_count,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    result.ViewDimension = match view.dimension {
        TextureViewDimension::D1 => D3D12_SRV_DIMENSION_TEXTURE1D,
        TextureViewDimension::D2 if base.sample_count == 1 => D3D12_SRV_DIMENSION_TEXTURE2D,
        TextureViewDimension::D2 => D3D12_SRV_DIMENSION_TEXTURE2DMS,
        TextureViewDimension::D2Array if base.sample_count == 1 => {
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY
        }
        TextureViewDimension::D2Array => D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
        TextureViewDimension::Cube => D3D12_SRV_DIMENSION_TEXTURECUBE,
        TextureViewDimension::CubeArray => D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
        TextureViewDimension::D3 => D3D12_SRV_DIMENSION_TEXTURE3D,
    };
    result
}
