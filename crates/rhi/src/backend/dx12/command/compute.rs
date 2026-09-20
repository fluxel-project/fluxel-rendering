//! Lowering a validated compute dispatch onto a Direct3D 12 command list.

use std::collections::HashMap;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
    D3D12_RESOURCE_STATES, ID3D12GraphicsCommandList,
};

use crate::api::command::record::ComputeDispatch;
use crate::api::command::{AccessMask, ResourceUse};
use crate::api::resource::Buffer;
use crate::backend::dx12::binding::Dx12BindGroup;
use crate::backend::dx12::failure::Dx12Failure;
use crate::backend::dx12::pipeline::Dx12ComputePipeline;

use super::dx12_buffer;
use super::transfer::CommittedBatch;
use super::transition::Transitions;

/// Records one dispatch and retains every native object it references until the
/// batch's completion fence passes.
pub(super) fn lower_compute_dispatch(
    list: &ID3D12GraphicsCommandList,
    dispatch: &ComputeDispatch,
    uses: &[ResourceUse],
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let pipeline = dispatch
        .pipeline
        .native()
        .as_any()
        .downcast_ref::<Dx12ComputePipeline>()
        .ok_or(Dx12Failure::Unsupported {
            what: "a compute pipeline this device did not create",
            why: "its native state belongs to another backend",
        })?;

    let mut native_groups = Vec::with_capacity(dispatch.groups.len());
    for bound in &dispatch.groups {
        if !bound.dynamic_offsets.is_empty() {
            return Err(Dx12Failure::Unsupported {
                what: "a compute bind group with dynamic offsets",
                why: "the DX12 root-signature lowering currently exposes descriptor tables only",
            });
        }
        let native = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<Dx12BindGroup>()
            .ok_or(Dx12Failure::Unsupported {
                what: "a bind group this device did not create",
                why: "its descriptor table belongs to another backend",
            })?;
        native_groups.push((bound, native));
    }

    let mut buffers: HashMap<_, (Buffer, AccessMask)> = HashMap::new();
    for resource_use in uses {
        let ResourceUse::Buffer(buffer_use) = resource_use else {
            return Err(Dx12Failure::Unsupported {
                what: "a compute dispatch that touches a texture or presentation frame",
                why: "the DX12 texture and presentation resource lowering is not implemented",
            });
        };
        buffers
            .entry(buffer_use.buffer.id())
            .and_modify(|(_, access)| *access = access.union(buffer_use.access))
            .or_insert_with(|| (buffer_use.buffer.clone(), buffer_use.access));
    }

    let mut entering = Transitions::default();
    let mut leaving = Transitions::default();
    for (buffer, access) in buffers.values() {
        let native = dx12_buffer(buffer)?;
        let state = shader_state(*access);
        entering.push(native.resource(), D3D12_RESOURCE_STATE_COMMON, state);
        leaving.push(native.resource(), state, D3D12_RESOURCE_STATE_COMMON);
    }
    entering.record(list);

    // Every DX12 bind group from one device allocates from the device's single
    // shader-visible CBV/SRV/UAV heap. Bind it once before setting table roots.
    if let Some((_, first)) = native_groups.first() {
        unsafe {
            list.SetDescriptorHeaps(&[
                Some(first.view_heap().clone()),
                Some(first.sampler_heap().clone()),
            ])
        };
    }
    unsafe {
        list.SetComputeRootSignature(pipeline.root_signature());
        list.SetPipelineState(pipeline.pipeline_state());
        for (bound, native) in &native_groups {
            if let Some(parameter) = pipeline.view_root_parameter(bound.index.get()) {
                list.SetComputeRootDescriptorTable(parameter, native.view_table());
            }
            if let Some(parameter) = pipeline.sampler_root_parameter(bound.index.get()) {
                list.SetComputeRootDescriptorTable(parameter, native.sampler_table());
            }
        }
        let (x, y, z) = dispatch.workgroups;
        list.Dispatch(x, y, z);
    }
    leaving.record(list);

    committed.compute_pipelines.push(dispatch.pipeline.clone());
    committed
        .bind_groups
        .extend(dispatch.groups.iter().map(|bound| bound.group.clone()));
    Ok(())
}

fn shader_state(access: AccessMask) -> D3D12_RESOURCE_STATES {
    if access.contains(AccessMask::SHADER_WRITE) {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
    }
}
