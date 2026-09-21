//! Standalone colour-MSAA resolve lowering.
//!
//! Metal exposes hardware resolve only as a render-pass store action.  That
//! operation cannot express the public `TextureResolve` command: the command
//! permits independent source/destination origins, a destination mip, and an
//! arbitrary consecutive array-layer range.  This module therefore uses one
//! small, backend-private compute kernel.  Four fixed native signatures cover
//! 2D/2D-array source and destination combinations.  They address the selected
//! mip and layer directly, avoiding an unsupported attempt to create a view of
//! a multisample texture.
//!
//! `float` texture access deliberately covers the published non-integer colour
//! formats.  Metal performs the normal unorm/snorm conversion, and sRGB reads
//! and writes use its standard linear decode/encode path. Integer resolves are
//! not published: averaging integer samples would invent a semantic which the
//! portable command does not specify.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLLibrary, MTLSize,
};

use crate::api::command::copy::TextureResolve;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};

use super::command::metal_texture;

const THREAD_WIDTH: usize = 8;
const THREAD_HEIGHT: usize = 8;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ResolveParams {
    uint2 source_origin;
    uint2 destination_origin;
    uint2 extent;
    uint source_layer;
    uint destination_layer;
    uint destination_mip;
    uint sample_count;
};

kernel void fluxel_resolve_msaa_00(
    texture2d_ms<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant ResolveParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (any(gid >= params.extent)) {
        return;
    }
    uint2 source_coord = params.source_origin + gid;
    float4 total = float4(0.0);
    for (uint sample = 0; sample < params.sample_count; ++sample) {
        total += source.read(source_coord, sample);
    }
    destination.write(total / float(params.sample_count), params.destination_origin + gid, params.destination_mip);
}

kernel void fluxel_resolve_msaa_01(
    texture2d_ms<float, access::read> source [[texture(0)]],
    texture2d_array<float, access::write> destination [[texture(1)]],
    constant ResolveParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    if (any(gid >= params.extent)) return;
    float4 total = float4(0.0);
    for (uint sample = 0; sample < params.sample_count; ++sample)
        total += source.read(params.source_origin + gid, sample);
    destination.write(total / float(params.sample_count), params.destination_origin + gid,
                      params.destination_layer, params.destination_mip);
}

kernel void fluxel_resolve_msaa_10(
    texture2d_ms_array<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant ResolveParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    if (any(gid >= params.extent)) return;
    float4 total = float4(0.0);
    for (uint sample = 0; sample < params.sample_count; ++sample)
        total += source.read(params.source_origin + gid, params.source_layer, sample);
    destination.write(total / float(params.sample_count), params.destination_origin + gid, params.destination_mip);
}

kernel void fluxel_resolve_msaa_11(
    texture2d_ms_array<float, access::read> source [[texture(0)]],
    texture2d_array<float, access::write> destination [[texture(1)]],
    constant ResolveParams& params [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    if (any(gid >= params.extent)) return;
    float4 total = float4(0.0);
    for (uint sample = 0; sample < params.sample_count; ++sample)
        total += source.read(params.source_origin + gid, params.source_layer, sample);
    destination.write(total / float(params.sample_count), params.destination_origin + gid,
                      params.destination_layer, params.destination_mip);
}
"#;

/// Lazily-created immutable state shared by every standalone resolve on one
/// device. It is owned by `MetalShared`; command recording only borrows it.
pub(super) struct MetalResolvePipeline {
    _library: Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>,
    states: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 4],
}

unsafe impl Send for MetalResolvePipeline {}
unsafe impl Sync for MetalResolvePipeline {}

pub(super) fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
) -> RhiResult<MetalResolvePipeline> {
    let source = NSString::from_str(SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|_| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "compile Metal standalone resolve kernel",
            )
            .at("MetalResolve")
        })?;
    let states = [
        pipeline_state(device, &library, "00")?,
        pipeline_state(device, &library, "01")?,
        pipeline_state(device, &library, "10")?,
        pipeline_state(device, &library, "11")?,
    ];
    Ok(MetalResolvePipeline {
        _library: library,
        states,
    })
}

/// Encode a complete public direct resolve after public validation has checked
/// dimensions, bounds, aspect, sample counts, and non-overlap.  The explicit
/// checked conversions below are still necessary because Objective-C uses
/// `NSUInteger` and this backend supports 32-bit Apple hosts.
pub(super) fn encode(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &MetalResolvePipeline,
    resolve: &TextureResolve,
) -> RhiResult<()> {
    let source = metal_texture(&resolve.src)?;
    let destination = metal_texture(&resolve.dst)?;
    let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::BackendFailure,
            "Metal failed to create standalone resolve compute encoder",
        )
        .at("MetalResolve")
    })?;
    let source_array = resolve.src.descriptor().array_layers > 1;
    let destination_array = resolve.dst.descriptor().array_layers > 1;
    let state_index = (source_array as usize) << 1 | destination_array as usize;
    encoder.setComputePipelineState(&pipeline.states[state_index]);

    let threads = MTLSize {
        width: THREAD_WIDTH,
        height: THREAD_HEIGHT,
        depth: 1,
    };
    let groups = MTLSize {
        width: div_ceil(resolve.extent.width, THREAD_WIDTH as u32)?,
        height: div_ceil(resolve.extent.height, THREAD_HEIGHT as u32)?,
        depth: 1,
    };
    for layer in 0..resolve.src_subresource.layer_count {
        let params = ResolveParams {
            source_origin: [resolve.src_origin.x, resolve.src_origin.y],
            destination_origin: [resolve.dst_origin.x, resolve.dst_origin.y],
            extent: [resolve.extent.width, resolve.extent.height],
            source_layer: resolve
                .src_subresource
                .base_layer
                .checked_add(layer)
                .ok_or_else(|| overflow("source layer"))?,
            destination_layer: resolve
                .dst_subresource
                .base_layer
                .checked_add(layer)
                .ok_or_else(|| overflow("destination layer"))?,
            destination_mip: resolve.dst_subresource.mip_level,
            sample_count: source.sample_count,
        };
        let params_ptr = NonNull::from(&params).cast();
        unsafe {
            encoder.setTexture_atIndex(Some(&source.raw), 0);
            encoder.setTexture_atIndex(Some(&destination.raw), 1);
            encoder.setBytes_length_atIndex(params_ptr, core::mem::size_of::<ResolveParams>(), 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }
    encoder.endEncoding();
    Ok(())
}

fn pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
    suffix: &str,
) -> RhiResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let name = NSString::from_str(&format!("fluxel_resolve_msaa_{suffix}"));
    let function = library.newFunctionWithName(&name).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::BackendFailure,
            "Metal standalone resolve kernel entry is absent",
        )
        .at("MetalResolve")
    })?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "create Metal standalone resolve pipeline",
            )
            .at("MetalResolve")
        })
}

#[repr(C)]
struct ResolveParams {
    source_origin: [u32; 2],
    destination_origin: [u32; 2],
    extent: [u32; 2],
    source_layer: u32,
    destination_layer: u32,
    destination_mip: u32,
    sample_count: u32,
}

fn div_ceil(value: u32, divisor: u32) -> RhiResult<usize> {
    usize::try_from(value.div_ceil(divisor)).map_err(|_| overflow("dispatch grid"))
}

fn overflow(what: &'static str) -> RhiError {
    RhiError::new(
        RhiErrorKind::InvalidUsage,
        format!("Metal standalone resolve {what} exceeds host address space"),
    )
    .at("MetalResolve")
}
