//! DX12 graphics-PSO lowering.
//!
//! The portable raster descriptor has already passed all capability and
//! cross-object validation when it reaches this module.  This file only turns
//! that frozen description into the D3D12 objects consumed by a command list.

use std::any::Any;
use std::mem::ManuallyDrop;

use windows::Win32::Foundation::{FALSE, TRUE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::api::format::TextureFormat;
use crate::api::pipeline::backend::RasterPipelineBackend;
use crate::api::pipeline::{
    PrimitiveTopology, RasterPipelineDescriptor, VertexFormat, VertexStepMode,
};
use crate::backend::dx12::failure::Dx12Failure;

/// Native state retained by a portable raster pipeline.
pub(crate) struct Dx12RasterPipeline {
    root_signature: super::interface::Dx12RootSignature,
    state: ID3D12PipelineState,
}

impl Dx12RasterPipeline {
    pub(crate) fn pipeline_state(&self) -> &ID3D12PipelineState {
        &self.state
    }
    pub(crate) fn root_signature(&self) -> &ID3D12RootSignature {
        self.root_signature.handle()
    }
    pub(crate) fn view_root_parameter(&self, group: u32) -> Option<u32> {
        self.root_signature.view_parameter(group)
    }
    /// Samplers live in D3D12's separate shader-visible heap.  Graphics and
    /// compute use the same root-signature mapping; keeping this accessor here
    /// prevents a raster path from accidentally binding only CBV/SRV/UAV tables.
    pub(crate) fn sampler_root_parameter(&self, group: u32) -> Option<u32> {
        self.root_signature.sampler_parameter(group)
    }
}

impl RasterPipelineBackend for Dx12RasterPipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn create_raster_pipeline(
    device: &ID3D12Device,
    descriptor: &RasterPipelineDescriptor,
) -> Result<Dx12RasterPipeline, Dx12Failure> {
    let root_signature =
        super::interface::build_root_signature(device, &descriptor.interface.descriptor().groups)?;
    let vertex = dxil(&descriptor.vertex, "vertex")?;
    let fragment = match descriptor.fragment.as_ref() {
        Some(shader) => Some(dxil(shader, "fragment")?),
        None => None,
    };

    // Input element descriptors borrow the semantic name byte strings and this
    // vector until CreateGraphicsPipelineState has synchronously consumed them.
    // Shader locations lower to the conventional LOCATION<n> semantic ABI.
    let semantic_names: Vec<Vec<u8>> = descriptor
        .vertex_input
        .buffers
        .iter()
        .flat_map(|buffer| buffer.attributes.iter())
        .map(|attribute| format!("LOCATION{}\0", attribute.location.get()).into_bytes())
        .collect();
    let mut semantic_index = 0usize;
    let input_elements: Vec<D3D12_INPUT_ELEMENT_DESC> = descriptor
        .vertex_input
        .buffers
        .iter()
        .enumerate()
        .flat_map(|(slot, buffer)| {
            buffer
                .attributes
                .iter()
                .map(move |attribute| (slot, buffer, attribute))
        })
        .map(|(slot, buffer, attribute)| {
            let semantic = &semantic_names[semantic_index];
            semantic_index += 1;
            Ok(D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(semantic.as_ptr()),
                SemanticIndex: 0,
                Format: vertex_format(attribute.format)?,
                InputSlot: slot as u32,
                AlignedByteOffset: u32::try_from(attribute.offset)
                    .map_err(|_| unsupported("vertex attribute offset larger than u32"))?,
                InputSlotClass: match buffer.step_mode {
                    VertexStepMode::Vertex => D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    VertexStepMode::Instance => D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                },
                InstanceDataStepRate: match buffer.step_mode {
                    VertexStepMode::Vertex => 0,
                    VertexStepMode::Instance => 1,
                },
            })
        })
        .collect::<Result<_, Dx12Failure>>()?;

    let mut rtv = [DXGI_FORMAT_UNKNOWN; 8];
    for (index, target) in descriptor.color_targets.iter().enumerate() {
        if let Some(target) = target {
            rtv[index] = texture_format(target.format)?;
        }
    }
    let dsv = descriptor
        .depth_stencil
        .as_ref()
        .map(|state| texture_format(state.format))
        .transpose()?
        .unwrap_or(DXGI_FORMAT_UNKNOWN);
    let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_signature.handle().clone())),
        VS: bytecode(vertex),
        PS: fragment.map(bytecode).unwrap_or_default(),
        BlendState: blend_state(descriptor),
        SampleMask: descriptor.multisample.mask,
        RasterizerState: rasterizer_state(descriptor),
        DepthStencilState: depth_stencil_state(descriptor),
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_elements.as_ptr(),
            NumElements: input_elements.len() as u32,
        },
        PrimitiveTopologyType: topology(descriptor.primitive.topology),
        NumRenderTargets: descriptor.color_targets.len() as u32,
        RTVFormats: rtv,
        DSVFormat: dsv,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: descriptor.multisample.count,
            Quality: 0,
        },
        ..Default::default()
    };
    let state = unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&description) };
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    let state = state.map_err(|error| {
        Dx12Failure::Native(crate::backend::dx12::ffi::NativeError::new(
            &error,
            "Dx12Device::create_raster_pipeline",
        ))
    })?;
    Ok(Dx12RasterPipeline {
        root_signature,
        state,
    })
}

fn dxil<'a>(
    shader: &'a crate::api::shader::ShaderModule,
    stage: &'static str,
) -> Result<&'a [u8], Dx12Failure> {
    let native = shader
        .native()
        .as_any()
        .downcast_ref::<crate::backend::dx12::shader::Dx12ShaderModule>()
        .ok_or_else(|| unsupported("a shader from another backend"))?;
    native.dxil().ok_or(Dx12Failure::Unsupported {
        what: "a non-DXIL raster shader",
        why: stage,
    })
}
fn unsupported(what: &'static str) -> Dx12Failure {
    Dx12Failure::Unsupported {
        what,
        why: "the DX12 graphics pipeline lowerer cannot represent it",
    }
}
fn bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}
fn topology(value: PrimitiveTopology) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match value {
        PrimitiveTopology::PointList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
        PrimitiveTopology::LineList | PrimitiveTopology::LineStrip => {
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
        }
        PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip => {
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
        }
    }
}
fn vertex_format(value: VertexFormat) -> Result<DXGI_FORMAT, Dx12Failure> {
    Ok(match value {
        VertexFormat::Float32 => DXGI_FORMAT_R32_FLOAT,
        VertexFormat::Float32x2 => DXGI_FORMAT_R32G32_FLOAT,
        VertexFormat::Float32x3 => DXGI_FORMAT_R32G32B32_FLOAT,
        VertexFormat::Float32x4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        VertexFormat::Uint32 => DXGI_FORMAT_R32_UINT,
        VertexFormat::Uint32x2 => DXGI_FORMAT_R32G32_UINT,
        VertexFormat::Uint32x3 => DXGI_FORMAT_R32G32B32_UINT,
        VertexFormat::Uint32x4 => DXGI_FORMAT_R32G32B32A32_UINT,
        VertexFormat::Sint32 => DXGI_FORMAT_R32_SINT,
        VertexFormat::Sint32x2 => DXGI_FORMAT_R32G32_SINT,
        VertexFormat::Sint32x3 => DXGI_FORMAT_R32G32B32_SINT,
        VertexFormat::Sint32x4 => DXGI_FORMAT_R32G32B32A32_SINT,
        VertexFormat::Unorm8x2 => DXGI_FORMAT_R8G8_UNORM,
        VertexFormat::Unorm8x4 => DXGI_FORMAT_R8G8B8A8_UNORM,
    })
}
fn texture_format(value: TextureFormat) -> Result<DXGI_FORMAT, Dx12Failure> {
    Ok(match value {
        TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        TextureFormat::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        TextureFormat::Depth16Unorm => DXGI_FORMAT_D16_UNORM,
        TextureFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        TextureFormat::Depth32FloatStencil8 => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        _ => return Err(unsupported("this texture format")),
    })
}
fn rasterizer_state(desc: &RasterPipelineDescriptor) -> D3D12_RASTERIZER_DESC {
    D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: match desc.primitive.cull_mode {
            crate::api::pipeline::CullMode::None => D3D12_CULL_MODE_NONE,
            crate::api::pipeline::CullMode::Front => D3D12_CULL_MODE_FRONT,
            crate::api::pipeline::CullMode::Back => D3D12_CULL_MODE_BACK,
        },
        FrontCounterClockwise: if matches!(
            desc.primitive.front_face,
            crate::api::pipeline::FrontFace::Ccw
        ) {
            TRUE
        } else {
            FALSE
        },
        DepthBias: desc.primitive.depth_bias.map_or(0, |v| v.constant),
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: desc.primitive.depth_bias.map_or(0.0, |v| v.slope_scale),
        DepthClipEnable: TRUE,
        MultisampleEnable: if desc.multisample.count > 1 {
            TRUE
        } else {
            FALSE
        },
        AntialiasedLineEnable: FALSE,
        ForcedSampleCount: 0,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    }
}
fn blend_state(desc: &RasterPipelineDescriptor) -> D3D12_BLEND_DESC {
    let mut result = D3D12_BLEND_DESC {
        AlphaToCoverageEnable: if desc.multisample.alpha_to_coverage_enabled {
            TRUE
        } else {
            FALSE
        },
        IndependentBlendEnable: TRUE,
        ..Default::default()
    };
    for (index, target) in desc.color_targets.iter().enumerate() {
        if let Some(target) = target {
            let blend = target.blend;
            result.RenderTarget[index] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: if blend.is_some() { TRUE } else { FALSE },
                LogicOpEnable: FALSE,
                SrcBlend: blend.map_or(D3D12_BLEND_ONE, |b| blend_factor(b.color.src_factor)),
                DestBlend: blend.map_or(D3D12_BLEND_ZERO, |b| blend_factor(b.color.dst_factor)),
                BlendOp: blend.map_or(D3D12_BLEND_OP_ADD, |b| blend_op(b.color.operation)),
                SrcBlendAlpha: blend.map_or(D3D12_BLEND_ONE, |b| blend_factor(b.alpha.src_factor)),
                DestBlendAlpha: blend
                    .map_or(D3D12_BLEND_ZERO, |b| blend_factor(b.alpha.dst_factor)),
                BlendOpAlpha: blend.map_or(D3D12_BLEND_OP_ADD, |b| blend_op(b.alpha.operation)),
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: color_write_mask(target.write_mask),
            };
        }
    }
    result
}
fn color_write_mask(value: crate::api::pipeline::ColorWriteMask) -> u8 {
    let mut result = 0;
    if value.contains(crate::api::pipeline::ColorWriteMask::RED) {
        result |= 1;
    }
    if value.contains(crate::api::pipeline::ColorWriteMask::GREEN) {
        result |= 2;
    }
    if value.contains(crate::api::pipeline::ColorWriteMask::BLUE) {
        result |= 4;
    }
    if value.contains(crate::api::pipeline::ColorWriteMask::ALPHA) {
        result |= 8;
    }
    result
}
fn blend_factor(value: crate::api::pipeline::BlendFactor) -> D3D12_BLEND {
    use crate::api::pipeline::BlendFactor::*;
    match value {
        Zero => D3D12_BLEND_ZERO,
        One => D3D12_BLEND_ONE,
        Src => D3D12_BLEND_SRC_COLOR,
        OneMinusSrc => D3D12_BLEND_INV_SRC_COLOR,
        SrcAlpha => D3D12_BLEND_SRC_ALPHA,
        OneMinusSrcAlpha => D3D12_BLEND_INV_SRC_ALPHA,
        Dst => D3D12_BLEND_DEST_COLOR,
        OneMinusDst => D3D12_BLEND_INV_DEST_COLOR,
        DstAlpha => D3D12_BLEND_DEST_ALPHA,
        OneMinusDstAlpha => D3D12_BLEND_INV_DEST_ALPHA,
        SrcAlphaSaturated => D3D12_BLEND_SRC_ALPHA_SAT,
        Constant => D3D12_BLEND_BLEND_FACTOR,
        OneMinusConstant => D3D12_BLEND_INV_BLEND_FACTOR,
    }
}
fn blend_op(value: crate::api::pipeline::BlendOperation) -> D3D12_BLEND_OP {
    use crate::api::pipeline::BlendOperation::*;
    match value {
        Add => D3D12_BLEND_OP_ADD,
        Subtract => D3D12_BLEND_OP_SUBTRACT,
        ReverseSubtract => D3D12_BLEND_OP_REV_SUBTRACT,
        Min => D3D12_BLEND_OP_MIN,
        Max => D3D12_BLEND_OP_MAX,
    }
}
fn depth_stencil_state(desc: &RasterPipelineDescriptor) -> D3D12_DEPTH_STENCIL_DESC {
    let depth = desc.depth_stencil.as_ref().and_then(|v| v.depth);
    let stencil = desc.depth_stencil.as_ref().and_then(|v| v.stencil);
    D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: if depth.is_some() { TRUE } else { FALSE },
        DepthWriteMask: if depth.is_some_and(|v| v.write_enabled) {
            D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: depth.map_or(D3D12_COMPARISON_FUNC_ALWAYS, |v| compare(v.compare)),
        StencilEnable: if stencil.is_some() { TRUE } else { FALSE },
        StencilReadMask: stencil
            .map_or(D3D12_DEFAULT_STENCIL_READ_MASK as u8, |v| v.read_mask as u8),
        StencilWriteMask: stencil.map_or(D3D12_DEFAULT_STENCIL_WRITE_MASK as u8, |v| {
            v.write_mask as u8
        }),
        FrontFace: stencil.map_or(default_stencil_face(), |v| stencil_face(v.front)),
        BackFace: stencil.map_or(default_stencil_face(), |v| stencil_face(v.back)),
    }
}
fn compare(value: crate::api::resource::sampler::CompareFunction) -> D3D12_COMPARISON_FUNC {
    use crate::api::resource::sampler::CompareFunction::*;
    match value {
        Never => D3D12_COMPARISON_FUNC_NEVER,
        Less => D3D12_COMPARISON_FUNC_LESS,
        Equal => D3D12_COMPARISON_FUNC_EQUAL,
        LessEqual => D3D12_COMPARISON_FUNC_LESS_EQUAL,
        Greater => D3D12_COMPARISON_FUNC_GREATER,
        NotEqual => D3D12_COMPARISON_FUNC_NOT_EQUAL,
        GreaterEqual => D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        Always => D3D12_COMPARISON_FUNC_ALWAYS,
    }
}
fn default_stencil_face() -> D3D12_DEPTH_STENCILOP_DESC {
    D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    }
}
fn stencil_face(value: crate::api::pipeline::StencilFaceState) -> D3D12_DEPTH_STENCILOP_DESC {
    D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: stencil_op(value.fail_op),
        StencilDepthFailOp: stencil_op(value.depth_fail_op),
        StencilPassOp: stencil_op(value.pass_op),
        StencilFunc: compare(value.compare),
    }
}
fn stencil_op(value: crate::api::pipeline::StencilOperation) -> D3D12_STENCIL_OP {
    use crate::api::pipeline::StencilOperation::*;
    match value {
        Keep => D3D12_STENCIL_OP_KEEP,
        Zero => D3D12_STENCIL_OP_ZERO,
        Replace => D3D12_STENCIL_OP_REPLACE,
        Invert => D3D12_STENCIL_OP_INVERT,
        IncrementClamp => D3D12_STENCIL_OP_INCR_SAT,
        DecrementClamp => D3D12_STENCIL_OP_DECR_SAT,
        IncrementWrap => D3D12_STENCIL_OP_INCR,
        DecrementWrap => D3D12_STENCIL_OP_DECR,
    }
}
