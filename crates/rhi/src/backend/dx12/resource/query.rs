//! Direct3D 12 query-heap ownership and the native query type mapping.
//!
//! Only occlusion is published at present.  D3D12 has timestamp and pipeline
//! statistics heaps too, but publishing either needs their complete portable
//! conversion/result-layout contract.  A heap kind is not, by itself, that
//! contract.

use std::any::Any;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_QUERY_HEAP_DESC, D3D12_QUERY_HEAP_TYPE_OCCLUSION, D3D12_QUERY_TYPE,
    D3D12_QUERY_TYPE_OCCLUSION, ID3D12Device, ID3D12QueryHeap,
};

use crate::api::query::{QuerySetDescriptor, QueryType};
use crate::api::resource::backend::QuerySetBackend;
use crate::backend::dx12::ffi;

/// Native storage behind an occlusion query set.
pub(crate) struct Dx12QuerySet {
    heap: ID3D12QueryHeap,
}

impl Dx12QuerySet {
    pub(crate) fn heap(&self) -> &ID3D12QueryHeap {
        &self.heap
    }
    pub(crate) fn query_type(&self) -> D3D12_QUERY_TYPE {
        D3D12_QUERY_TYPE_OCCLUSION
    }
}

impl QuerySetBackend for Dx12QuerySet {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Allocates one native query heap after portable capability validation.
pub(crate) fn create_query_set(
    device: &ID3D12Device,
    descriptor: &QuerySetDescriptor,
) -> Result<Dx12QuerySet, ffi::NativeError> {
    if !matches!(descriptor.ty, QueryType::Occlusion) {
        return Err(ffi::NativeError::driver_contract_violation(
            "DX12 only publishes occlusion-query heap lowering",
            "Dx12Device::create_query_set",
        ));
    }
    let desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION,
        Count: descriptor.count,
        NodeMask: 0,
    };
    let mut heap = None;
    unsafe {
        device
            .CreateQueryHeap(&desc, &mut heap)
            .map_err(|error| ffi::NativeError::new(&error, "ID3D12Device::CreateQueryHeap"))?;
    }
    let heap = heap.ok_or_else(|| {
        ffi::NativeError::driver_contract_violation(
            "CreateQueryHeap reported success without producing a query heap",
            "Dx12Device::create_query_set",
        )
    })?;
    Ok(Dx12QuerySet { heap })
}
