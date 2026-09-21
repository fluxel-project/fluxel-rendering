//! Metal compute and raster pipeline-state lowering.
//!
//! Pipeline descriptors arrive here after portable validation has checked
//! identities, shader stages, target signatures and capability facts. The
//! remaining work is a direct, fallible Metal translation; no alternate shader
//! or cross-device object is accepted.

use std::any::Any;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCompareFunction,
    MTLComputePipelineState, MTLDepthClipMode, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStencilDescriptor,
    MTLStencilOperation, MTLVertexAmplificationViewMapping, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::backend::{ComputePipelineBackend, RasterPipelineBackend};
use crate::api::pipeline::{
    BlendFactor, BlendOperation, ColorWriteMask, ComputePipelineDescriptor,
    RasterPipelineDescriptor, VertexFormat, VertexStepMode,
};

use super::binding::MetalBindingAbi;
use super::device::MetalShared;
use super::shader::MetalShader;

/// Native compute state. The shared domain holds the MTLDevice alive for this
/// pipeline's entire portable lifetime.
pub(super) struct MetalComputePipeline {
    _shared: Arc<MetalShared>,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The MSL entry's fixed local size, validated by the portable pipeline
    /// constructor. Dispatch lowering uses this as Metal's `threadsPerThreadgroup`;
    /// workgroup counts are a separate recorded command value.
    workgroup_size: crate::api::shader::ComputeWorkgroupSize,
    /// Stable native direct-argument mapping derived from the pipeline interface.
    abi: MetalBindingAbi,
}

unsafe impl Send for MetalComputePipeline {}
unsafe impl Sync for MetalComputePipeline {}

impl MetalComputePipeline {
    pub(super) fn state(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }
    pub(super) const fn workgroup_size(&self) -> crate::api::shader::ComputeWorkgroupSize {
        self.workgroup_size
    }
    pub(super) fn binding_abi(&self) -> &MetalBindingAbi {
        &self.abi
    }
}

impl ComputePipelineBackend for MetalComputePipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Native render state. Metal keeps depth/stencil comparison in a separate,
/// immutable `MTLDepthStencilState`; it is built alongside the render PSO and
/// selected by raster command lowering before each draw.
pub(super) struct MetalRasterPipeline {
    _shared: Arc<MetalShared>,
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth_stencil: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    /// Metal depth clipping is render-encoder state, not part of the PSO. Keep
    /// the pipeline-selected value here so command lowering can apply it whenever
    /// this pipeline becomes current without rereading a portable descriptor.
    depth_clip_mode: Option<MTLDepthClipMode>,
    /// One mapping per selected portable multiview bit. Sparse masks preserve
    /// their original layer index instead of compacting selected layers.
    vertex_amplification_mappings: Option<Vec<MTLVertexAmplificationViewMapping>>,
    /// Stable native direct-argument mapping derived from the pipeline interface.
    abi: MetalBindingAbi,
}

unsafe impl Send for MetalRasterPipeline {}
unsafe impl Sync for MetalRasterPipeline {}

impl MetalRasterPipeline {
    pub(super) fn state(&self) -> &ProtocolObject<dyn MTLRenderPipelineState> {
        &self.state
    }
    pub(super) fn binding_abi(&self) -> &MetalBindingAbi {
        &self.abi
    }
    pub(super) fn depth_stencil(&self) -> Option<&ProtocolObject<dyn MTLDepthStencilState>> {
        self.depth_stencil.as_deref()
    }
    /// The render-encoder depth clip mode selected from `PrimitiveState`.
    pub(super) const fn depth_clip_mode(&self) -> Option<MTLDepthClipMode> {
        self.depth_clip_mode
    }
    /// The view mappings command lowering installs for a multiview draw.
    pub(super) fn vertex_amplification_mappings(
        &self,
    ) -> Option<&[MTLVertexAmplificationViewMapping]> {
        self.vertex_amplification_mappings.as_deref()
    }
}

impl RasterPipelineBackend for MetalRasterPipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) fn create_compute_pipeline(
    shared: Arc<MetalShared>,
    descriptor: &ComputePipelineDescriptor,
) -> RhiResult<MetalComputePipeline> {
    let shader = downcast_shader(&descriptor.shader, "compute")?;
    let state = shared
        .device
        .newComputePipelineStateWithFunction_error(shader.function())
        .map_err(|error| native_error("create compute pipeline", &error))?;
    let workgroup_size = descriptor
        .shader
        .artifact()
        .interface
        .compute_workgroup_size()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal compute pipeline shader has no validated workgroup size",
            )
        })?;
    let abi = MetalBindingAbi::from_compute_interface(
        &descriptor.interface,
        &descriptor.shader.artifact().interface,
    )?;
    Ok(MetalComputePipeline {
        _shared: shared,
        state,
        workgroup_size,
        abi,
    })
}

pub(super) fn create_raster_pipeline(
    shared: Arc<MetalShared>,
    descriptor: &RasterPipelineDescriptor,
) -> RhiResult<MetalRasterPipeline> {
    let vertex = downcast_shader(&descriptor.vertex, "vertex")?;
    let fragment = descriptor
        .fragment
        .as_ref()
        .map(|module| downcast_shader(module, "fragment"))
        .transpose()?;
    let native = MTLRenderPipelineDescriptor::new();
    native.setVertexFunction(Some(vertex.function()));
    native.setFragmentFunction(fragment.map(|shader| shader.function()));
    native.setRasterSampleCount(descriptor.multisample.count as usize);
    native.setAlphaToCoverageEnabled(descriptor.multisample.alpha_to_coverage_enabled);
    let vertex_amplification_mappings = multiview_mappings(descriptor.multiview_mask);
    if let Some(mappings) = &vertex_amplification_mappings {
        // Metal does not bounds-check the unsafe PSO setter. Public validation
        // has already refused a zero mask; keep an exact native count check here
        // as a defence against stale capability facts.
        if !shared
            .device
            .supportsVertexAmplificationCount(mappings.len())
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal does not support this multiview amplification count",
            )
            .at("MetalDevice::create_raster_pipeline"));
        }
        unsafe {
            native.setMaxVertexAmplificationCount(mappings.len());
        }
    }
    if let Some(label) = descriptor.label.as_deref() {
        native.setLabel(Some(&NSString::from_str(label)));
    }
    configure_vertex_input(&native, descriptor)?;
    configure_targets(&native, descriptor)?;
    let depth_stencil = descriptor
        .depth_stencil
        .as_ref()
        .map(|state| create_depth_stencil_state(&shared.device, state))
        .transpose()?;
    // `unclipped_depth` is validated against DepthClipControl before this
    // backend is reached. Metal exposes the corresponding setting on the render
    // encoder (rather than on MTLRenderPipelineDescriptor), so retain the exact
    // native choice with the immutable pipeline for command lowering.
    // On supporting devices retain both Clamp and the explicit Clip reset: the
    // mode is encoder state and would otherwise leak across pipeline switches.
    // Older families retain `None` and never receive the optional selector.
    let depth_clip_supported = super::facts::supports_depth_clip_control(&shared.device);
    if descriptor.primitive.unclipped_depth && !depth_clip_supported {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal depth-clip clamp is unavailable on this device",
        )
        .at("MetalDevice::create_raster_pipeline"));
    }
    let depth_clip_mode = depth_clip_supported.then_some(if descriptor.primitive.unclipped_depth {
        MTLDepthClipMode::Clamp
    } else {
        MTLDepthClipMode::Clip
    });
    let abi = MetalBindingAbi::from_raster_interfaces(
        &descriptor.interface,
        &descriptor.vertex.artifact().interface,
        descriptor
            .fragment
            .as_ref()
            .map(|shader| &shader.artifact().interface),
    )?;
    let state = shared
        .device
        .newRenderPipelineStateWithDescriptor_error(&native)
        .map_err(|error| native_error("create render pipeline", &error))?;
    Ok(MetalRasterPipeline {
        _shared: shared,
        state,
        depth_stencil,
        depth_clip_mode,
        vertex_amplification_mappings,
        abi,
    })
}

/// Preserves each selected logical view's index as both native offsets. Thus a
/// SelectiveMultiview mask such as `0b101` renders into layers zero and two,
/// rather than incorrectly compacting its second view into layer one.
fn multiview_mappings(mask: Option<u32>) -> Option<Vec<MTLVertexAmplificationViewMapping>> {
    mask.map(|mask| {
        (0..u32::BITS)
            .filter(|bit| mask & (1 << bit) != 0)
            .map(|bit| MTLVertexAmplificationViewMapping {
                viewportArrayIndexOffset: bit,
                renderTargetArrayIndexOffset: bit,
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selective_multiview_keeps_sparse_layer_indices() {
        let mappings = multiview_mappings(Some(0b101)).expect("a mask has mappings");
        assert_eq!(mappings.len(), 2);
        assert_eq!(mappings[0].viewportArrayIndexOffset, 0);
        assert_eq!(mappings[0].renderTargetArrayIndexOffset, 0);
        assert_eq!(mappings[1].viewportArrayIndexOffset, 2);
        assert_eq!(mappings[1].renderTargetArrayIndexOffset, 2);
    }

    #[test]
    fn ordinary_raster_has_no_amplification_mapping() {
        assert!(multiview_mappings(None).is_none());
    }
}

fn create_depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
    state: &crate::api::pipeline::DepthStencilState,
) -> RhiResult<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
    let descriptor = MTLDepthStencilDescriptor::new();
    if let Some(depth) = state.depth {
        descriptor.setDepthCompareFunction(compare_function(depth.compare));
        descriptor.setDepthWriteEnabled(depth.write_enabled);
    } else {
        // Explicitly disabling the test makes a declared attachment with no
        // depth operation behave the same across drivers, instead of relying on
        // the native descriptor's defaults.
        descriptor.setDepthCompareFunction(MTLCompareFunction::Always);
        descriptor.setDepthWriteEnabled(false);
    }
    if let Some(stencil) = state.stencil {
        let front = stencil_descriptor(stencil.front, stencil.read_mask, stencil.write_mask);
        let back = stencil_descriptor(stencil.back, stencil.read_mask, stencil.write_mask);
        descriptor.setFrontFaceStencil(Some(&front));
        descriptor.setBackFaceStencil(Some(&back));
    }
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal failed to create depth/stencil state",
            )
            .at("MetalDevice::create_raster_pipeline")
        })
}

fn stencil_descriptor(
    face: crate::api::pipeline::StencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> Retained<MTLStencilDescriptor> {
    let descriptor = MTLStencilDescriptor::new();
    descriptor.setStencilCompareFunction(compare_function(face.compare));
    descriptor.setReadMask(read_mask);
    descriptor.setWriteMask(write_mask);
    descriptor.setStencilFailureOperation(stencil_operation(face.fail_op));
    descriptor.setDepthFailureOperation(stencil_operation(face.depth_fail_op));
    descriptor.setDepthStencilPassOperation(stencil_operation(face.pass_op));
    descriptor
}

fn compare_function(value: crate::api::resource::sampler::CompareFunction) -> MTLCompareFunction {
    use crate::api::resource::sampler::CompareFunction as F;
    match value {
        F::Never => MTLCompareFunction::Never,
        F::Less => MTLCompareFunction::Less,
        F::Equal => MTLCompareFunction::Equal,
        F::LessEqual => MTLCompareFunction::LessEqual,
        F::Greater => MTLCompareFunction::Greater,
        F::NotEqual => MTLCompareFunction::NotEqual,
        F::GreaterEqual => MTLCompareFunction::GreaterEqual,
        F::Always => MTLCompareFunction::Always,
    }
}

fn stencil_operation(value: crate::api::pipeline::StencilOperation) -> MTLStencilOperation {
    use crate::api::pipeline::StencilOperation as O;
    match value {
        O::Keep => MTLStencilOperation::Keep,
        O::Zero => MTLStencilOperation::Zero,
        O::Replace => MTLStencilOperation::Replace,
        O::Invert => MTLStencilOperation::Invert,
        O::IncrementClamp => MTLStencilOperation::IncrementClamp,
        O::DecrementClamp => MTLStencilOperation::DecrementClamp,
        O::IncrementWrap => MTLStencilOperation::IncrementWrap,
        O::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

fn downcast_shader<'a>(
    module: &'a crate::api::shader::ShaderModule,
    stage: &'static str,
) -> RhiResult<&'a MetalShader> {
    module
        .native()
        .as_any()
        .downcast_ref::<MetalShader>()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "pipeline shader does not belong to this Metal backend",
            )
            .at(stage)
        })
}

fn configure_vertex_input(
    native: &MTLRenderPipelineDescriptor,
    descriptor: &RasterPipelineDescriptor,
) -> RhiResult<()> {
    let vertex = MTLVertexDescriptor::new();
    let layouts = vertex.layouts();
    let attributes = vertex.attributes();
    for (buffer_index, buffer) in descriptor.vertex_input.buffers.iter().enumerate() {
        let layout = unsafe { layouts.objectAtIndexedSubscript(buffer_index) };
        unsafe { layout.setStride(buffer.stride as usize) };
        layout.setStepFunction(match buffer.step_mode {
            VertexStepMode::Vertex => MTLVertexStepFunction::PerVertex,
            VertexStepMode::Instance => MTLVertexStepFunction::PerInstance,
        });
        for attribute in &buffer.attributes {
            let location = attribute.location.get() as usize;
            let native_attribute = unsafe { attributes.objectAtIndexedSubscript(location) };
            native_attribute.setFormat(vertex_format(attribute.format)?);
            unsafe {
                native_attribute.setOffset(attribute.offset as usize);
                native_attribute.setBufferIndex(buffer_index);
            }
        }
    }
    native.setVertexDescriptor(Some(&vertex));
    Ok(())
}

fn configure_targets(
    native: &MTLRenderPipelineDescriptor,
    descriptor: &RasterPipelineDescriptor,
) -> RhiResult<()> {
    let colors = native.colorAttachments();
    for (index, target) in descriptor.color_targets.iter().enumerate() {
        let Some(target) = target else { continue };
        let color = unsafe { colors.objectAtIndexedSubscript(index) };
        color.setPixelFormat(super::format::metal_format(target.format).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "color target format has no Metal mapping",
            )
        })?);
        color.setWriteMask(color_write_mask(target.write_mask));
        if let Some(blend) = target.blend {
            color.setBlendingEnabled(true);
            color.setSourceRGBBlendFactor(blend_factor(blend.color.src_factor));
            color.setDestinationRGBBlendFactor(blend_factor(blend.color.dst_factor));
            color.setRgbBlendOperation(blend_operation(blend.color.operation));
            color.setSourceAlphaBlendFactor(blend_factor(blend.alpha.src_factor));
            color.setDestinationAlphaBlendFactor(blend_factor(blend.alpha.dst_factor));
            color.setAlphaBlendOperation(blend_operation(blend.alpha.operation));
        }
    }
    if let Some(depth) = &descriptor.depth_stencil {
        let format = super::format::metal_format(depth.format).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "depth/stencil format has no Metal mapping",
            )
        })?;
        let aspects = crate::api::format::format_aspects(depth.format);
        if aspects.contains(crate::api::resource::TextureAspects::DEPTH) {
            native.setDepthAttachmentPixelFormat(format);
        }
        if aspects.contains(crate::api::resource::TextureAspects::STENCIL) {
            native.setStencilAttachmentPixelFormat(format);
        }
    }
    Ok(())
}

fn vertex_format(format: VertexFormat) -> RhiResult<MTLVertexFormat> {
    use MTLVertexFormat as M;
    use VertexFormat as F;
    Ok(match format {
        F::Uint8 => M::UChar,
        F::Uint8x2 => M::UChar2,
        F::Uint8x4 => M::UChar4,
        F::Sint8 => M::Char,
        F::Sint8x2 => M::Char2,
        F::Sint8x4 => M::Char4,
        F::Unorm8 => M::UCharNormalized,
        F::Unorm8x2 => M::UChar2Normalized,
        F::Unorm8x4 => M::UChar4Normalized,
        F::Unorm8x4Bgra => M::UChar4Normalized_BGRA,
        F::Snorm8 => M::CharNormalized,
        F::Snorm8x2 => M::Char2Normalized,
        F::Snorm8x4 => M::Char4Normalized,
        F::Uint16 => M::UShort,
        F::Uint16x2 => M::UShort2,
        F::Uint16x4 => M::UShort4,
        F::Sint16 => M::Short,
        F::Sint16x2 => M::Short2,
        F::Sint16x4 => M::Short4,
        F::Unorm16 => M::UShortNormalized,
        F::Unorm16x2 => M::UShort2Normalized,
        F::Unorm16x4 => M::UShort4Normalized,
        F::Snorm16 => M::ShortNormalized,
        F::Snorm16x2 => M::Short2Normalized,
        F::Snorm16x4 => M::Short4Normalized,
        F::Float16 => M::Half,
        F::Float16x2 => M::Half2,
        F::Float16x4 => M::Half4,
        F::Float32 => M::Float,
        F::Float32x2 => M::Float2,
        F::Float32x3 => M::Float3,
        F::Float32x4 => M::Float4,
        F::Uint32 => M::UInt,
        F::Uint32x2 => M::UInt2,
        F::Uint32x3 => M::UInt3,
        F::Uint32x4 => M::UInt4,
        F::Sint32 => M::Int,
        F::Sint32x2 => M::Int2,
        F::Sint32x3 => M::Int3,
        F::Sint32x4 => M::Int4,
        F::Unorm10_10_10_2 => M::UInt1010102Normalized,
        F::Float64 | F::Float64x2 | F::Float64x3 | F::Float64x4 => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal vertex fetch does not support f64 vertex formats",
            ));
        }
    })
}

fn color_write_mask(mask: ColorWriteMask) -> MTLColorWriteMask {
    let mut out = MTLColorWriteMask(0);
    if mask.contains(ColorWriteMask::RED) {
        out.0 |= MTLColorWriteMask::Red.0;
    }
    if mask.contains(ColorWriteMask::GREEN) {
        out.0 |= MTLColorWriteMask::Green.0;
    }
    if mask.contains(ColorWriteMask::BLUE) {
        out.0 |= MTLColorWriteMask::Blue.0;
    }
    if mask.contains(ColorWriteMask::ALPHA) {
        out.0 |= MTLColorWriteMask::Alpha.0;
    }
    out
}

fn blend_factor(value: BlendFactor) -> MTLBlendFactor {
    use BlendFactor as F;
    use MTLBlendFactor as M;
    match value {
        F::Zero => M::Zero,
        F::One => M::One,
        F::Src => M::SourceColor,
        F::OneMinusSrc => M::OneMinusSourceColor,
        F::SrcAlpha => M::SourceAlpha,
        F::OneMinusSrcAlpha => M::OneMinusSourceAlpha,
        F::Dst => M::DestinationColor,
        F::OneMinusDst => M::OneMinusDestinationColor,
        F::DstAlpha => M::DestinationAlpha,
        F::OneMinusDstAlpha => M::OneMinusDestinationAlpha,
        F::SrcAlphaSaturated => M::SourceAlphaSaturated,
        F::Constant => M::BlendColor,
        F::OneMinusConstant => M::OneMinusBlendColor,
        F::Src1 => M::Source1Color,
        F::OneMinusSrc1 => M::OneMinusSource1Color,
        F::Src1Alpha => M::Source1Alpha,
        F::OneMinusSrc1Alpha => M::OneMinusSource1Alpha,
    }
}

fn blend_operation(value: BlendOperation) -> MTLBlendOperation {
    match value {
        BlendOperation::Add => MTLBlendOperation::Add,
        BlendOperation::Subtract => MTLBlendOperation::Subtract,
        BlendOperation::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOperation::Min => MTLBlendOperation::Min,
        BlendOperation::Max => MTLBlendOperation::Max,
    }
}

fn native_error(operation: &'static str, _error: &NSError) -> RhiError {
    RhiError::new(RhiErrorKind::BackendFailure, operation).at("MetalDevice::create_pipeline")
}
