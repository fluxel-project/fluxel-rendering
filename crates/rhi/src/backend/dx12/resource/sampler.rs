//! Native D3D12 sampler descriptors.

use std::any::Any;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMPARISON_FUNC, D3D12_COMPARISON_FUNC_ALWAYS, D3D12_COMPARISON_FUNC_EQUAL,
    D3D12_COMPARISON_FUNC_GREATER, D3D12_COMPARISON_FUNC_GREATER_EQUAL, D3D12_COMPARISON_FUNC_LESS,
    D3D12_COMPARISON_FUNC_LESS_EQUAL, D3D12_COMPARISON_FUNC_NEVER, D3D12_COMPARISON_FUNC_NONE,
    D3D12_COMPARISON_FUNC_NOT_EQUAL, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_FILTER,
    D3D12_SAMPLER_DESC, D3D12_TEXTURE_ADDRESS_MODE, D3D12_TEXTURE_ADDRESS_MODE_BORDER,
    D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
    D3D12_TEXTURE_ADDRESS_MODE_WRAP, ID3D12DescriptorHeap, ID3D12Device,
};

use crate::api::resource::backend::SamplerBackend;
use crate::api::resource::sampler::{
    AddressMode, CompareFunction, FilterMode, SamplerBorderColor, SamplerDescriptor,
};
use crate::backend::dx12::ffi;

/// One CPU-visible immutable sampler descriptor.
pub(crate) struct Dx12Sampler {
    _heap: ID3D12DescriptorHeap,
    cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
}

impl Dx12Sampler {
    pub(crate) fn cpu(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.cpu
    }
}
impl SamplerBackend for Dx12Sampler {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn create_sampler(
    device: &ID3D12Device,
    descriptor: &SamplerDescriptor,
) -> Result<Dx12Sampler, ffi::NativeError> {
    let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the descriptor is a valid local and the returned heap owns its slot.
    let heap = unsafe { device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&heap_desc) }
        .map_err(|error| ffi::NativeError::new(&error, "Device::create_sampler"))?;
    // SAFETY: this getter only reads the descriptor heap's immutable start address.
    let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let native = D3D12_SAMPLER_DESC {
        Filter: filter(descriptor),
        AddressU: address(descriptor.address_u),
        AddressV: address(descriptor.address_v),
        AddressW: address(descriptor.address_w),
        MipLODBias: 0.0,
        MaxAnisotropy: descriptor.max_anisotropy as u32,
        ComparisonFunc: comparison(descriptor.compare),
        BorderColor: border_color(descriptor.border_color),
        MinLOD: descriptor.lod_min,
        MaxLOD: descriptor.lod_max,
    };
    // SAFETY: `cpu` is the only slot of the retained sampler heap, and native is
    // a fully initialized D3D12 sampler descriptor.
    unsafe { device.CreateSampler(&native, cpu) };
    Ok(Dx12Sampler { _heap: heap, cpu })
}

fn address(mode: AddressMode) -> D3D12_TEXTURE_ADDRESS_MODE {
    match mode {
        AddressMode::ClampToEdge => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressMode::Repeat => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressMode::MirrorRepeat => D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        AddressMode::ClampToBorder => D3D12_TEXTURE_ADDRESS_MODE_BORDER,
    }
}
fn border_color(value: SamplerBorderColor) -> [f32; 4] {
    match value {
        SamplerBorderColor::TransparentBlack | SamplerBorderColor::Zero => [0.0; 4],
        SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
        SamplerBorderColor::OpaqueWhite => [1.0; 4],
    }
}

fn comparison(compare: Option<CompareFunction>) -> D3D12_COMPARISON_FUNC {
    match compare {
        None => D3D12_COMPARISON_FUNC_NONE,
        Some(CompareFunction::Never) => D3D12_COMPARISON_FUNC_NEVER,
        Some(CompareFunction::Less) => D3D12_COMPARISON_FUNC_LESS,
        Some(CompareFunction::Equal) => D3D12_COMPARISON_FUNC_EQUAL,
        Some(CompareFunction::LessEqual) => D3D12_COMPARISON_FUNC_LESS_EQUAL,
        Some(CompareFunction::Greater) => D3D12_COMPARISON_FUNC_GREATER,
        Some(CompareFunction::NotEqual) => D3D12_COMPARISON_FUNC_NOT_EQUAL,
        Some(CompareFunction::GreaterEqual) => D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        Some(CompareFunction::Always) => D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

fn filter(desc: &SamplerDescriptor) -> D3D12_FILTER {
    // D3D12's basic filter encoding uses bits 2, 4 and 0 for min, mag and mip;
    // comparison adds bit 7. Anisotropic is its dedicated encoding.
    if desc.max_anisotropy > 1 {
        return D3D12_FILTER(if desc.compare.is_some() { 213 } else { 85 });
    }
    let nearest = |mode| matches!(mode, FilterMode::Nearest) as i32;
    let base = (1 - nearest(desc.min_filter)) * 4
        + (1 - nearest(desc.mag_filter)) * 16
        + (1 - nearest(desc.mip_filter));
    D3D12_FILTER(base + if desc.compare.is_some() { 128 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::resource::sampler::SamplerDescriptor;

    #[test]
    fn default_sampler_is_point_filter() {
        assert_eq!(filter(&SamplerDescriptor::new()).0, 0);
    }

    #[test]
    fn comparison_anisotropy_uses_the_dedicated_filter() {
        let desc = SamplerDescriptor::new()
            .with_compare(CompareFunction::Less)
            .with_max_anisotropy(2);
        assert_eq!(filter(&desc).0, 213);
    }
}
