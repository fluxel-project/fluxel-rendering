//! DX12 compute-pipeline seam.
//!
//! The portable API owns validation and the opaque pipeline handle. This module
//! owns the root signature and PSO consumed by the DX12 command lowerer.

use std::any::Any;

use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_SHADER_BYTECODE, ID3D12Device, ID3D12PipelineState,
    ID3D12RootSignature,
};
use windows::core::PCWSTR;

use crate::api::pipeline::ComputePipelineDescriptor;
use crate::api::pipeline::backend::ComputePipelineBackend;
use crate::backend::dx12::failure::Dx12Failure;

/// Native state behind a portable compute pipeline.
///
/// Both objects are retained for command submission.
pub(crate) struct Dx12ComputePipeline {
    /// Kept beside the state object because command lowering must bind the exact
    /// root signature the PSO was created with.
    root_signature: super::interface::Dx12RootSignature,
    /// The driver's compiled compute state.
    state: ID3D12PipelineState,
}

impl Dx12ComputePipeline {
    /// The state object command lowering binds before dispatching.
    pub(crate) fn pipeline_state(&self) -> &ID3D12PipelineState {
        &self.state
    }

    /// The root signature that defines this PSO's descriptor-table ABI.
    pub(crate) fn root_signature(&self) -> &ID3D12RootSignature {
        self.root_signature.handle()
    }

    /// The compact root-parameter index for a logical group's view table.
    ///
    /// An empty group has no descriptor-table parameter, and sampler tables are
    /// not lowerable yet, so command lowering must treat `None` as "nothing to
    /// bind" rather than applying the old arithmetic mapping.
    pub(crate) fn view_root_parameter(&self, group: u32) -> Option<u32> {
        self.root_signature.view_parameter(group)
    }

    pub(crate) fn sampler_root_parameter(&self, group: u32) -> Option<u32> {
        self.root_signature.sampler_parameter(group)
    }
    pub(crate) fn immediate_root_parameter(&self, offset: u32, size: u32) -> Option<(u32, u32)> {
        self.root_signature.immediate_parameter(offset, size)
    }
}

impl ComputePipelineBackend for Dx12ComputePipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Lowers a validated portable compute descriptor to DX12.
///
/// This is intentionally not a compatibility fallback: it requires DXIL and
/// creates the real driver objects.
pub(crate) fn create_compute_pipeline(
    device: &ID3D12Device,
    descriptor: &ComputePipelineDescriptor,
) -> Result<Dx12ComputePipeline, Dx12Failure> {
    let root_signature = super::interface::build_root_signature(
        device,
        &descriptor.interface.descriptor().groups,
        &descriptor.interface.descriptor().immediate_ranges,
        false,
    )?;
    let Some(shader) = descriptor
        .shader
        .native()
        .as_any()
        .downcast_ref::<crate::backend::dx12::shader::Dx12ShaderModule>()
    else {
        return Err(Dx12Failure::Unsupported {
            what: "a compute shader from another backend",
            why: "a DX12 pipeline state needs DXIL held by Dx12ShaderModule",
        });
    };
    let Some(dxil) = shader.dxil() else {
        return Err(Dx12Failure::Unsupported {
            what: "a non-DXIL compute shader",
            why: "Direct3D 12 compute pipeline states consume DXIL bytecode",
        });
    };
    let mut description = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // D3D12 reads the root signature during this call. The projection wraps
        // this field in ManuallyDrop, so the clone is released explicitly after
        // the synchronous call rather than leaked with the descriptor struct.
        pRootSignature: ManuallyDrop::new(Some(root_signature.handle().clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr().cast(),
            BytecodeLength: dxil.len(),
        },
        ..Default::default()
    };
    // The DXIL slice is borrowed from the shader handle held by `descriptor`,
    // and both it and the root-signature clone remain live until the synchronous
    // driver call returns.
    let state: Result<ID3D12PipelineState, Dx12Failure> =
        match super::cache::native_cache(descriptor.cache.as_ref())? {
            Some(cache) => {
                let name = super::cache::compute_name(descriptor);
                let library = cache.library();
                match unsafe {
                    library.LoadComputePipeline::<_, ID3D12PipelineState>(
                        PCWSTR(name.as_ptr()),
                        &description,
                    )
                } {
                    Ok(state) => Ok(state),
                    // A name miss and a non-terminal invalidated stored entry both
                    // fall back to authoritative driver creation, then replace the
                    // entry. A terminal Load failure must not be hidden by a
                    // successful creation attempt: its outer device boundary owns
                    // the loss transition.
                    Err(error) => {
                        let failure = crate::backend::dx12::ffi::NativeError::new(
                            &error,
                            "ID3D12PipelineLibrary::LoadComputePipeline",
                        );
                        if failure.failure().is_terminal() {
                            Err(Dx12Failure::Native(failure))
                        } else {
                            (|| -> Result<ID3D12PipelineState, Dx12Failure> {
                                let state = unsafe {
                                    device.CreateComputePipelineState::<ID3D12PipelineState>(
                                        &description,
                                    )
                                }
                                .map_err(|error| {
                                    Dx12Failure::Native(
                                        crate::backend::dx12::ffi::NativeError::new(
                                            &error,
                                            "Dx12Device::create_compute_pipeline",
                                        ),
                                    )
                                })?;
                                unsafe { library.StorePipeline(PCWSTR(name.as_ptr()), &state) }
                                    .map_err(|error| {
                                        Dx12Failure::Native(
                                            crate::backend::dx12::ffi::NativeError::new(
                                                &error,
                                                "ID3D12PipelineLibrary::StorePipeline",
                                            ),
                                        )
                                    })?;
                                Ok(state)
                            })()
                        }
                    }
                }
            }
            None => {
                unsafe { device.CreateComputePipelineState::<ID3D12PipelineState>(&description) }
                    .map_err(|error| {
                        Dx12Failure::Native(crate::backend::dx12::ffi::NativeError::new(
                            &error,
                            "Dx12Device::create_compute_pipeline",
                        ))
                    })
            }
        };
    // SAFETY: this is the one release of the clone placed in pRootSignature
    // above, after the native call has finished reading the descriptor.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    let state = state?;

    Ok(Dx12ComputePipeline {
        root_signature,
        state,
    })
}
