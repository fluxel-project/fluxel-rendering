//! Native D3D12 texture allocation.

use std::any::Any;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE1D, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_DIMENSION_TEXTURE3D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_FLAGS, D3D12_RESOURCE_STATE_COMMON,
    D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::api::resource::backend::TextureBackend;
use crate::api::resource::texture::{TextureDescriptor, TextureDimension, TextureUsage};
use crate::backend::dx12::{ffi, platform::facts::dxgi_format};

/// The committed resource behind one portable texture.
pub(crate) struct Dx12Texture {
    resource: ID3D12Resource,
}

impl Dx12Texture {
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }
}

impl TextureBackend for Dx12Texture {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Creates a default-heap texture in COMMON state.
pub(crate) fn create_texture(
    device: &ID3D12Device,
    descriptor: &TextureDescriptor,
) -> Result<Dx12Texture, ffi::NativeError> {
    let format = dxgi_format(descriptor.format).ok_or_else(|| {
        ffi::NativeError::driver_contract_violation(
            "DX12 has no exact DXGI representation for the accepted portable texture format",
            "Device::create_texture",
        )
    })?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let native = D3D12_RESOURCE_DESC {
        Dimension: match descriptor.dimension {
            TextureDimension::D1 => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            TextureDimension::D2 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            TextureDimension::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        },
        Alignment: 0,
        Width: descriptor.extent.width as u64,
        Height: descriptor.extent.height,
        DepthOrArraySize: match descriptor.dimension {
            TextureDimension::D3 => descriptor.extent.depth as u16,
            TextureDimension::D1 | TextureDimension::D2 => descriptor.array_layers as u16,
        },
        MipLevels: descriptor.mip_levels as u16,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: descriptor.sample_count,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: resource_flags(descriptor.usage),
    };
    let mut resource = None;
    // SAFETY: all descriptors are local and outlive the call; no optimized clear
    // value is supplied because the public descriptor has no clear-value contract.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &native,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
            .map_err(|error| ffi::NativeError::new(&error, "Device::create_texture"))?;
    }
    let resource = resource.ok_or_else(|| {
        ffi::NativeError::driver_contract_violation(
            "CreateCommittedResource reported success without producing a texture",
            "Device::create_texture",
        )
    })?;
    Ok(Dx12Texture { resource })
}

fn resource_flags(usage: TextureUsage) -> D3D12_RESOURCE_FLAGS {
    let mut flags = D3D12_RESOURCE_FLAG_NONE;
    if usage.contains(TextureUsage::STORAGE) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if usage.contains(TextureUsage::COLOR_ATTACHMENT) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::format::TextureFormat;

    #[test]
    fn attachment_and_storage_usage_become_creation_flags() {
        let flags = resource_flags(TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::STORAGE));
        assert_ne!(
            flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_FLAG_NONE
        );
        assert_ne!(
            flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_NONE
        );
    }

    #[test]
    fn exact_format_mapping_rejects_abstract_depth_format() {
        assert!(dxgi_format(TextureFormat::Depth24Plus).is_none());
    }
}
