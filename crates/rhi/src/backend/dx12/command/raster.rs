//! Direct3D 12 lowering for a recorded raster scope.
//!
//! The portable recorder has already validated attachment compatibility and
//! draw bounds. This module owns only native state: attachment descriptors,
//! COMMON-to-render transitions, IA state, and draw calls.

use std::collections::HashMap;

use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT};

use crate::api::command::attachment::{
    ColorAttachmentView, DepthAttachmentMode, StencilAttachmentMode,
};
use crate::api::command::geometry::{ColorClearValue, LoadOp};
use crate::api::command::record::{RasterBegin, RasterDraw};
use crate::api::command::{AccessMask, IndexFormat, ResourceUse, TextureUseIntent};
use crate::api::identity::ObjectId;
use crate::api::pipeline::PrimitiveTopology;
use crate::api::resource::buffer::Buffer;
use crate::api::resource::texture::{Texture, TextureDimension};
use crate::api::resource::view::{TextureView, TextureViewDimension};
use crate::backend::dx12::binding::Dx12BindGroup;
use crate::backend::dx12::failure::{Dx12Failure, ref_native};
use crate::backend::dx12::pipeline::Dx12RasterPipeline;
use crate::backend::dx12::platform::facts::dxgi_format;
use crate::backend::dx12::presentation::Dx12FrameAttachment;

use super::transfer::CommittedBatch;
use super::transition::Transitions;
use super::{dx12_buffer, dx12_texture};

pub(super) struct RasterScopeState {
    color_views: Vec<TextureView>,
    depth_view: Option<TextureView>,
    color_heaps: Vec<ID3D12DescriptorHeap>,
    depth_heap: Option<ID3D12DescriptorHeap>,
    depth_read_only: bool,
}

pub(super) fn lower_raster_begin(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    begin: &RasterBegin,
    _committed: &mut CommittedBatch,
) -> Result<RasterScopeState, Dx12Failure> {
    // OMSetRenderTargets needs a dense handle array. Sparse portable MRT is
    // valid, but needs explicit null descriptors; retain an honest refusal until
    // that descriptor form is represented here.
    for (expected, (location, _)) in begin.colors.iter().enumerate() {
        if *location != expected as u32 {
            return Err(unsupported(
                "a sparse color-attachment set",
                "DX12 lowering has not yet allocated null RTV descriptors for MRT holes",
            ));
        }
    }
    let mut entering = Transitions::default();
    let mut colors = Vec::with_capacity(begin.colors.len());
    let color_views = Vec::with_capacity(begin.colors.len());
    let mut color_heaps = Vec::with_capacity(begin.colors.len());
    for (_, color) in &begin.colors {
        if color.resolve.is_some() {
            return Err(unsupported(
                "a raster attachment resolve",
                "ResolveSubresource lowering is not implemented yet",
            ));
        }
        let (heap, handle) = match &color.view {
            ColorAttachmentView::Texture(view) => {
                let native = dx12_texture(view.texture())?;
                entering.push(
                    native.resource(),
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                );
                create_rtv(device, view)?
            }
            ColorAttachmentView::Frame(frame) => {
                let native = frame
                    .native()
                    .as_any()
                    .downcast_ref::<Dx12FrameAttachment>()
                    .ok_or_else(|| {
                        unsupported(
                            "a frame attachment this device did not acquire",
                            "its native drawable belongs to another backend",
                        )
                    })?;
                create_frame_rtv(device, native.resource(), frame.format())?
            }
        };
        if let ColorAttachmentView::Frame(frame) = &color.view {
            _committed.raster_frames.push(frame.clone());
        }
        colors.push(handle);
        color_heaps.push(heap);
    }
    let (depth_stencil, depth_view, depth_heap, depth_read_only) =
        if let Some(depth) = &begin.depth_stencil {
            let native = dx12_texture(depth.view.texture())?;
            let read_only = matches!(depth.depth, Some(DepthAttachmentMode::ReadOnly) | None)
                && matches!(depth.stencil, Some(StencilAttachmentMode::ReadOnly) | None);
            entering.push(
                native.resource(),
                D3D12_RESOURCE_STATE_COMMON,
                if read_only {
                    D3D12_RESOURCE_STATE_DEPTH_READ
                } else {
                    D3D12_RESOURCE_STATE_DEPTH_WRITE
                },
            );
            let (heap, handle) = create_dsv(device, &depth.view, depth.depth, depth.stencil)?;
            (
                Some(handle),
                Some(depth.view.clone()),
                Some(heap),
                read_only,
            )
        } else {
            (None, None, None, false)
        };
    entering.record(list);
    unsafe {
        list.OMSetRenderTargets(
            colors.len() as u32,
            Some(colors.as_ptr()),
            false,
            depth_stencil.as_ref().map(std::ptr::from_ref),
        );
    }
    for ((_, attachment), handle) in begin.colors.iter().zip(&colors) {
        if let LoadOp::Clear(value) = attachment.load {
            unsafe { list.ClearRenderTargetView(*handle, &clear_color(value), None) };
        }
    }
    if let (Some(depth), Some(handle)) = (&begin.depth_stencil, depth_stencil) {
        let mut flags = D3D12_CLEAR_FLAGS(0);
        let mut clear_depth = 1.0;
        let mut clear_stencil = 0u8;
        if let Some(DepthAttachmentMode::ReadWrite {
            load: LoadOp::Clear(value),
            ..
        }) = depth.depth
        {
            flags |= D3D12_CLEAR_FLAG_DEPTH;
            clear_depth = value;
        }
        if let Some(StencilAttachmentMode::ReadWrite {
            load: LoadOp::Clear(value),
            ..
        }) = depth.stencil
        {
            flags |= D3D12_CLEAR_FLAG_STENCIL;
            clear_stencil = value as u8;
        }
        if flags != D3D12_CLEAR_FLAGS(0) {
            unsafe { list.ClearDepthStencilView(handle, flags, clear_depth, clear_stencil, None) };
        }
    }
    Ok(RasterScopeState {
        color_views,
        depth_view,
        color_heaps,
        depth_heap,
        depth_read_only,
    })
}

fn create_frame_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    format: crate::api::format::TextureFormat,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let format = dxgi_format(format).ok_or_else(|| {
        unsupported(
            "a presentation format without DXGI mapping",
            "DX12 cannot create its RTV",
        )
    })?;
    let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_RTV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };
    unsafe { device.CreateRenderTargetView(resource, Some(&desc), handle) };
    Ok((heap, handle))
}

pub(super) fn lower_raster_draw(
    list: &ID3D12GraphicsCommandList,
    draw: &RasterDraw,
    uses: &[ResourceUse],
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let pipeline = draw
        .pipeline
        .native()
        .as_any()
        .downcast_ref::<Dx12RasterPipeline>()
        .ok_or_else(|| {
            unsupported(
                "a raster pipeline this device did not create",
                "its native state belongs to another backend",
            )
        })?;
    let mut groups = Vec::with_capacity(draw.groups.len());
    for bound in &draw.groups {
        if !bound.dynamic_offsets.is_empty() {
            return Err(unsupported(
                "a raster bind group with dynamic offsets",
                "DX12 root-descriptor dynamic-offset lowering is not implemented",
            ));
        }
        let group = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<Dx12BindGroup>()
            .ok_or_else(|| {
                unsupported(
                    "a bind group this device did not create",
                    "its descriptor table belongs to another backend",
                )
            })?;
        groups.push((bound, group));
    }
    let mut entering = Transitions::default();
    let mut leaving = Transitions::default();
    let mut buffers = HashMap::<ObjectId, (Buffer, D3D12_RESOURCE_STATES)>::new();
    let mut textures = HashMap::<ObjectId, (Texture, D3D12_RESOURCE_STATES)>::new();
    for resource_use in uses {
        match resource_use {
            ResourceUse::Buffer(use_) => {
                let state = buffer_state(use_.access);
                buffers
                    .entry(use_.buffer.id())
                    .and_modify(|(_, prior)| *prior |= state)
                    .or_insert_with(|| (use_.buffer.clone(), state));
            }
            ResourceUse::Texture(use_) => match use_.intent {
                TextureUseIntent::ColorAttachment
                | TextureUseIntent::DepthStencilRead
                | TextureUseIntent::DepthStencilWrite => {}
                TextureUseIntent::ShaderRead | TextureUseIntent::ShaderReadWrite => {
                    let state = texture_state(use_.access);
                    textures
                        .entry(use_.texture.id())
                        .and_modify(|(_, prior)| *prior |= state)
                        .or_insert_with(|| (use_.texture.clone(), state));
                }
                _ => {
                    return Err(unsupported(
                        "a raster draw texture use outside shader bindings",
                        "copy and resolve uses have separate lowerings",
                    ));
                }
            },
            ResourceUse::Frame(_) => {
                return Err(unsupported(
                    "a presentation frame used by a raster draw",
                    "presentation attachment lowering is not implemented",
                ));
            }
        }
    }
    for (buffer, state) in buffers.values() {
        let native = dx12_buffer(buffer)?;
        entering.push(native.resource(), D3D12_RESOURCE_STATE_COMMON, *state);
        leaving.push(native.resource(), *state, D3D12_RESOURCE_STATE_COMMON);
        committed.raster_buffers.push(buffer.clone());
    }
    for (texture, state) in textures.values() {
        let native = dx12_texture(texture)?;
        entering.push(native.resource(), D3D12_RESOURCE_STATE_COMMON, *state);
        leaving.push(native.resource(), *state, D3D12_RESOURCE_STATE_COMMON);
        committed.raster_textures.push(texture.clone());
    }
    let mut vertex_views = Vec::with_capacity(draw.vertex_buffers.len());
    for (slot, binding) in &draw.vertex_buffers {
        let buffer = dx12_buffer(&binding.buffer)?;
        let stride = draw
            .pipeline
            .descriptor()
            .vertex_input
            .buffers
            .get(*slot as usize)
            .ok_or_else(|| {
                unsupported(
                    "a vertex buffer slot absent from the pipeline",
                    "portable validation should have refused it",
                )
            })?
            .stride;
        vertex_views.push((
            *slot,
            D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: unsafe { buffer.resource().GetGPUVirtualAddress() }
                    + binding.range.offset,
                SizeInBytes: u32::try_from(binding.range.size).map_err(|_| {
                    unsupported(
                        "a vertex buffer range larger than 4 GiB",
                        "D3D12 IA views use u32 sizes",
                    )
                })?,
                StrideInBytes: u32::try_from(stride).map_err(|_| {
                    unsupported(
                        "a vertex stride larger than u32",
                        "D3D12 IA views use u32 strides",
                    )
                })?,
            },
        ));
    }
    entering.record(list);
    unsafe {
        list.SetGraphicsRootSignature(pipeline.root_signature());
        list.SetPipelineState(pipeline.pipeline_state());
        if let Some((_, group)) = groups.first() {
            list.SetDescriptorHeaps(&[Some(group.view_heap().clone())]);
        }
        for (bound, group) in &groups {
            if let Some(parameter) = pipeline.view_root_parameter(bound.index.get()) {
                list.SetGraphicsRootDescriptorTable(parameter, group.view_table());
            }
        }
        list.IASetPrimitiveTopology(primitive_topology(
            draw.pipeline.descriptor().primitive.topology,
        ));
        // Bind each slot independently: the portable raster scope may bind slot
        // 3 while slots 0..2 are intentionally absent, and an IA call starting
        // at zero would silently reinterpret that view as slot zero.
        for (slot, view) in &vertex_views {
            list.IASetVertexBuffers(*slot, Some(std::slice::from_ref(view)));
        }
        let viewport = draw.viewport.unwrap_or_else(|| default_viewport(scope));
        list.RSSetViewports(&[D3D12_VIEWPORT {
            TopLeftX: viewport.x,
            TopLeftY: viewport.y,
            Width: viewport.width,
            Height: viewport.height,
            MinDepth: viewport.min_depth,
            MaxDepth: viewport.max_depth,
        }]);
        let scissor = draw.scissor.unwrap_or_else(|| default_scissor(scope));
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
            left: scissor.x as i32,
            top: scissor.y as i32,
            right: scissor.right().unwrap_or(u32::MAX) as i32,
            bottom: scissor.bottom().unwrap_or(u32::MAX) as i32,
        }]);
        list.OMSetBlendFactor(Some(&[
            draw.blend_constant.r,
            draw.blend_constant.g,
            draw.blend_constant.b,
            draw.blend_constant.a,
        ]));
        list.OMSetStencilRef(draw.stencil_reference);
        if let Some(index) = &draw.index {
            let buffer = dx12_buffer(&index.binding.buffer)?;
            let format = match index.format {
                IndexFormat::Uint16 => DXGI_FORMAT_R16_UINT,
                IndexFormat::Uint32 => DXGI_FORMAT_R32_UINT,
            };
            list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: buffer.resource().GetGPUVirtualAddress()
                    + index.binding.range.offset,
                SizeInBytes: u32::try_from(index.binding.range.size).map_err(|_| {
                    unsupported(
                        "an index buffer range larger than 4 GiB",
                        "D3D12 IA views use u32 sizes",
                    )
                })?,
                Format: format,
            }));
            list.DrawIndexedInstanced(
                draw.range.end - draw.range.start,
                draw.instances.end - draw.instances.start,
                draw.range.start,
                draw.base_vertex,
                draw.instances.start,
            );
        } else {
            list.DrawInstanced(
                draw.range.end - draw.range.start,
                draw.instances.end - draw.instances.start,
                draw.range.start,
                draw.instances.start,
            );
        }
    }
    leaving.record(list);
    committed.raster_pipelines.push(draw.pipeline.clone());
    committed
        .bind_groups
        .extend(draw.groups.iter().map(|group| group.group.clone()));
    Ok(())
}

pub(super) fn lower_raster_end(
    list: &ID3D12GraphicsCommandList,
    scope: RasterScopeState,
    committed: &mut CommittedBatch,
) {
    let mut leaving = Transitions::default();
    for view in &scope.color_views {
        if let Ok(texture) = dx12_texture(view.texture()) {
            leaving.push(
                texture.resource(),
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
    }
    if let Some(view) = &scope.depth_view {
        if let Ok(texture) = dx12_texture(view.texture()) {
            leaving.push(
                texture.resource(),
                if scope.depth_read_only {
                    D3D12_RESOURCE_STATE_DEPTH_READ
                } else {
                    D3D12_RESOURCE_STATE_DEPTH_WRITE
                },
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
    }
    leaving.record(list);
    committed.raster_views.extend(scope.color_views);
    if let Some(view) = scope.depth_view {
        committed.raster_views.push(view);
    }
    committed.raster_descriptor_heaps.extend(scope.color_heaps);
    if let Some(heap) = scope.depth_heap {
        committed.raster_descriptor_heaps.push(heap);
    }
}

fn create_rtv(
    device: &ID3D12Device,
    view: &TextureView,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let texture = dx12_texture(view.texture())?;
    let descriptor = view.descriptor();
    if !matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        || !matches!(descriptor.dimension, TextureViewDimension::D2)
        || view.texture().descriptor().sample_count != 1
    {
        return Err(unsupported(
            "a non-single-sample 2D color attachment view",
            "DX12 raster lowering currently implements 2D RTV descriptors",
        ));
    }
    let format = dxgi_format(view.format()).ok_or_else(|| {
        unsupported(
            "a color attachment format without DXGI mapping",
            "DX12 cannot create its RTV",
        )
    })?;
    let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_RTV {
                MipSlice: descriptor.base_mip,
                PlaneSlice: 0,
            },
        },
    };
    unsafe { device.CreateRenderTargetView(texture.resource(), Some(&desc), handle) };
    Ok((heap, handle))
}

fn create_dsv(
    device: &ID3D12Device,
    view: &TextureView,
    depth: Option<DepthAttachmentMode>,
    stencil: Option<StencilAttachmentMode>,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let texture = dx12_texture(view.texture())?;
    let descriptor = view.descriptor();
    if !matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        || !matches!(descriptor.dimension, TextureViewDimension::D2)
        || view.texture().descriptor().sample_count != 1
    {
        return Err(unsupported(
            "a non-single-sample 2D depth attachment view",
            "DX12 raster lowering currently implements 2D DSV descriptors",
        ));
    }
    let format = dxgi_format(view.format()).ok_or_else(|| {
        unsupported(
            "a depth attachment format without DXGI mapping",
            "DX12 cannot create its DSV",
        )
    })?;
    let mut flags = D3D12_DSV_FLAG_NONE;
    if matches!(depth, Some(DepthAttachmentMode::ReadOnly) | None) {
        flags |= D3D12_DSV_FLAG_READ_ONLY_DEPTH;
    }
    if matches!(stencil, Some(StencilAttachmentMode::ReadOnly) | None) {
        flags |= D3D12_DSV_FLAG_READ_ONLY_STENCIL;
    }
    let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
        Flags: flags,
        Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_DSV {
                MipSlice: descriptor.base_mip,
            },
        },
    };
    unsafe { device.CreateDepthStencilView(texture.resource(), Some(&desc), handle) };
    Ok((heap, handle))
}

fn cpu_heap(
    device: &ID3D12Device,
    ty: D3D12_DESCRIPTOR_HEAP_TYPE,
) -> Result<ID3D12DescriptorHeap, Dx12Failure> {
    unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: ty,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .map_err(|error| ref_native(&error))
}

fn default_viewport(scope: &RasterScopeState) -> crate::api::command::Viewport {
    let extent = scope
        .color_views
        .first()
        .map(|v| v.extent())
        .or_else(|| scope.depth_view.as_ref().map(|v| v.extent()))
        .expect("validated raster scope has attachment");
    crate::api::command::Viewport::new(
        0.0,
        0.0,
        extent.width as f32,
        extent.height as f32,
        0.0,
        1.0,
    )
}
fn default_scissor(scope: &RasterScopeState) -> crate::api::command::Rect {
    let extent = scope
        .color_views
        .first()
        .map(|v| v.extent())
        .or_else(|| scope.depth_view.as_ref().map(|v| v.extent()))
        .expect("validated raster scope has attachment");
    crate::api::command::Rect::new(0, 0, extent.width, extent.height)
}
fn clear_color(value: ColorClearValue) -> [f32; 4] {
    match value {
        ColorClearValue::Float(v) => v,
        ColorClearValue::Sint(v) => v.map(|x| x as f32),
        ColorClearValue::Uint(v) => v.map(|x| x as f32),
    }
}
fn primitive_topology(
    topology: PrimitiveTopology,
) -> windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY {
    match topology {
        PrimitiveTopology::PointList => D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        PrimitiveTopology::LineList => D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        PrimitiveTopology::LineStrip => D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        PrimitiveTopology::TriangleList => D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        PrimitiveTopology::TriangleStrip => D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    }
}
fn unsupported(what: &'static str, why: &'static str) -> Dx12Failure {
    Dx12Failure::Unsupported { what, why }
}

fn buffer_state(access: AccessMask) -> D3D12_RESOURCE_STATES {
    if access.contains(AccessMask::SHADER_WRITE) {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
            | D3D12_RESOURCE_STATE_INDEX_BUFFER
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    }
}

fn texture_state(access: AccessMask) -> D3D12_RESOURCE_STATES {
    if access.contains(AccessMask::SHADER_WRITE) {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    }
}
