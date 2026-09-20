//! D3D12 occlusion query encoding.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, ID3D12GraphicsCommandList,
};

use crate::api::command::record::QueryResolve;
use crate::api::query::QuerySet;
use crate::backend::dx12::failure::Dx12Failure;
use crate::backend::dx12::resource::Dx12QuerySet;

use super::dx12_buffer;
use super::transfer::CommittedBatch;
use super::transition::Transitions;

fn native(set: &QuerySet) -> Result<&Dx12QuerySet, Dx12Failure> {
    set.native()
        .as_any()
        .downcast_ref::<Dx12QuerySet>()
        .ok_or(Dx12Failure::Unsupported {
            what: "a query set this DX12 device did not allocate",
            why: "its native query heap belongs to another backend",
        })
}

pub(super) fn begin(
    list: &ID3D12GraphicsCommandList,
    set: &QuerySet,
    index: u32,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let native = native(set)?;
    unsafe { list.BeginQuery(native.heap(), native.query_type(), index) };
    committed.query_sets.push(set.clone());
    Ok(())
}

pub(super) fn end(
    list: &ID3D12GraphicsCommandList,
    set: &QuerySet,
    index: u32,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let native = native(set)?;
    unsafe { list.EndQuery(native.heap(), native.query_type(), index) };
    committed.query_sets.push(set.clone());
    Ok(())
}

pub(super) fn resolve(
    list: &ID3D12GraphicsCommandList,
    query: &QueryResolve,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let set = native(&query.set)?;
    let destination = dx12_buffer(&query.destination)?;
    let mut entering = Transitions::default();
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);
    unsafe {
        list.ResolveQueryData(
            set.heap(),
            set.query_type(),
            query.first_query,
            query.query_count,
            destination.resource(),
            query.destination_offset,
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    committed.query_sets.push(query.set.clone());
    committed.indirect_buffers.push(query.destination.clone());
    Ok(())
}
