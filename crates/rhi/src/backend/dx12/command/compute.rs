//! Lowering a validated compute dispatch onto a Direct3D 12 command list.
//!
//! TODO(perf): Dispatch lowering currently rebinds the root signature, PSO,
//! descriptor heaps and every root table for every dispatch. A future
//! command-list-local state cache may skip unchanged native bindings, but must be
//! invalidated on command-list reset and remain entirely backend-private.

use std::collections::HashMap;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_SIGNATURE_DESC, D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_DESC_0,
    D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
    D3D12_RESOURCE_STATES, ID3D12CommandSignature, ID3D12Device, ID3D12GraphicsCommandList,
    ID3D12Resource,
};

use crate::api::command::record::{ComputeDispatch, ComputeIndirect};
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
        for write in &dispatch.immediates {
            let (parameter, destination) = pipeline
                .immediate_root_parameter(write.offset, write.bytes.len() as u32)
                .ok_or(Dx12Failure::Unsupported {
                    what: "an immediate write outside the DX12 root-constant layout",
                    why: "portable validation must keep writes within declared ranges",
                })?;
            let values: Vec<u32> = write
                .bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte immediate word")))
                .collect();
            list.SetComputeRoot32BitConstants(
                parameter,
                values.len() as u32,
                values.as_ptr().cast(),
                destination,
            );
        }
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

/// Lowers a native `Dispatch` command signature.  The API packet carries no
/// root constants, so the signature contains exactly the twelve-byte dispatch
/// argument and may use the already-bound root signature unchanged.
pub(super) fn lower_compute_indirect(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    dispatch: &ComputeIndirect,
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
                why: "DX12 root-descriptor dynamic-offset lowering is not implemented",
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
                what: "an indirect compute dispatch that touches a texture or presentation frame",
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
    if let Some((_, first)) = native_groups.first() {
        unsafe {
            list.SetDescriptorHeaps(&[
                Some(first.view_heap().clone()),
                Some(first.sampler_heap().clone()),
            ])
        };
    }
    let argument = dx12_buffer(&dispatch.arguments)?;
    let argument_desc = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
        Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
    };
    let signature_desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: 12,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument_desc,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    unsafe {
        device
            .CreateCommandSignature(
                &signature_desc,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12RootSignature>,
                &mut signature,
            )
            .map_err(|error| {
                Dx12Failure::Native(crate::backend::dx12::ffi::NativeError::new(
                    &error,
                    "ID3D12Device::CreateCommandSignature",
                ))
            })?;
    }
    let signature = signature.ok_or(Dx12Failure::Unsupported {
        what: "a missing compute command signature",
        why: "CreateCommandSignature returned success without a signature",
    })?;
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
        list.ExecuteIndirect(
            &signature,
            1,
            argument.resource(),
            dispatch.arguments_offset,
            None::<&ID3D12Resource>,
            0,
        );
    }
    leaving.record(list);
    committed.compute_pipelines.push(dispatch.pipeline.clone());
    committed
        .bind_groups
        .extend(dispatch.groups.iter().map(|bound| bound.group.clone()));
    committed.indirect_buffers.push(dispatch.arguments.clone());
    committed.command_signatures.push(signature);
    Ok(())
}

fn shader_state(access: AccessMask) -> D3D12_RESOURCE_STATES {
    if access.contains(AccessMask::INDIRECT_READ) {
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    } else if access.contains(AccessMask::SHADER_WRITE) {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
    }
}
