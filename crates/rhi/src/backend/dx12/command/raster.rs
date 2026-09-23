//! Direct3D 12 lowering for a recorded raster scope.
//!
//! The portable recorder has already validated attachment compatibility and
//! draw bounds. This module owns only native state: attachment descriptors,
//! resource-to-render transitions, IA state, and draw calls.
//!
//! TODO(perf): Raster lowering currently creates one CPU-only RTV/DSV heap per
//! attachment use and retains those heaps in `CommittedBatch` until the batch
//! fence completes. This is a correct baseline: the CPU descriptor handles stay
//! valid for command-list execution and cannot be recycled while the list may
//! still reference them. Replace it with persistent RTV/DSV allocators only with
//! fence-keyed descriptor retirement, and cache/reuse views only when resource,
//! subresource range, format, and read-only flags match exactly. Attachment
//! ownership and `CompletionPoint` already provide the required RHI semantics;
//! no native heap or descriptor handle belongs in the public API.
//!
//! TODO(perf): Draw lowering currently rebinds the root signature, PSO, both
//! descriptor heaps and every root table for every draw. Add a command-list-local
//! state cache keyed by those native identities and invalidate it on list reset;
//! only changed slots need native calls. This is backend encoder state, not a
//! reason to add pipeline/binding state to the public recorder.

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
use crate::api::command::record::{RasterBegin, RasterDraw, RasterIndirect};
use crate::api::command::{AccessMask, IndexFormat, ResourceUse, TextureUseIntent};
use crate::api::identity::ObjectId;
use crate::api::pipeline::PrimitiveTopology;
use crate::api::presentation::FrameAttachment;
use crate::api::resource::buffer::Buffer;
use crate::api::resource::texture::{Extent3d, Texture, TextureDimension};
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
    colors: Vec<Dx12RenderAttachment>,
    /// CPU RTV heaps backing the null descriptors that fill sparse-MRT holes.
    /// A null RTV has no resource, so these are the only thing keeping the
    /// descriptor alive through the scope; dropped with the scope.  Never read —
    /// it is an ownership field: the handles it owns are referenced by the
    /// `color_handles` span handed to `OMSetRenderTargets`, and releasing the
    /// heap while that span is live would free the descriptors mid-draw.
    #[allow(dead_code)]
    null_rtv_heaps: Vec<ID3D12DescriptorHeap>,
    /// Multisample→single-sample color resolves to issue at scope end, one per
    /// attachment that declared `resolve`. Each carries the source (MSAA) and
    /// destination (single-sample) native resources and the destination view's
    /// subresource index. Resolved lazily in `lower_raster_end` so the source is
    /// still `RESOLVE_SOURCE` when the resolve fires.
    resolves: Vec<Dx12ColorResolve>,
    /// Parallel to `colors`: whether that color attachment is the source of a
    /// resolve. A resolve source leaves in `RESOLVE_SOURCE` (not `RENDER_TARGET`)
    /// once the resolve block has transitioned it, so `lower_raster_end` picks
    /// the leaving start state from this rather than assuming every color is a
    /// plain render target.
    color_resolves: Vec<bool>,
    depth_view: Option<TextureView>,
    depth_heap: Option<ID3D12DescriptorHeap>,
    depth_read_only: bool,
    attachment_extent: Extent3d,
}

/// One color-attachment resolve: source MSAA texture into a single-sample target.
struct Dx12ColorResolve {
    src: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    dst: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    dst_subresource: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
}

/// The DX12-native shape shared by ordinary texture attachments and swapchain
/// frames. The portable distinction is consumed once, at construction; command
/// encoding below works only with a resource, its state round trip and a native
/// descriptor. Retention deliberately keeps the portable owner as well as the
/// COM resource because presentation ownership and texture-view lifetime remain
/// distinct contracts even though both lower to an `ID3D12Resource`.
struct Dx12RenderAttachment {
    resource: ID3D12Resource,
    handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    heap: ID3D12DescriptorHeap,
    enter_state: D3D12_RESOURCE_STATES,
    leave_state: D3D12_RESOURCE_STATES,
    retention: AttachmentRetention,
}

enum AttachmentRetention {
    Texture(TextureView),
    Frame(FrameAttachment),
}

pub(super) fn lower_raster_begin(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    begin: &RasterBegin,
    _committed: &mut CommittedBatch,
) -> Result<RasterScopeState, Dx12Failure> {
    let mut entering = Transitions::default();
    let mut color_attachments = Vec::with_capacity(begin.colors.len());
    let mut resolves = Vec::new();
    let mut color_resolves = Vec::with_capacity(begin.colors.len());
    for (_, color) in &begin.colors {
        color_resolves.push(color.resolve.is_some());
        let attachment = lower_color_attachment(device, &color.view, color.depth_slice)?;
        // A resolve target is part of the attachment (section 31.1): it is a
        // single-sampled texture the multisampled source resolves into at scope
        // end. Resolve is only meaningful alongside a multisampled source.
        if let Some(target) = &color.resolve {
            let src = attachment.resource.clone();
            let ColorAttachmentView::Texture(target_view) = target else {
                return Err(unsupported(
                    "a resolve target that is not a texture view",
                    "only texture resolves are lowered on this backend",
                ));
            };
            let dst = dx12_texture(target_view.texture())?.resource().clone();
            let format = dxgi_format(target_view.format()).ok_or_else(|| {
                unsupported(
                    "a resolve target whose format has no DXGI mapping",
                    "DX12 cannot resolve into it",
                )
            })?;
            // Whole-subresource resolve for the common single-layer, base-mip
            // single-sample target. A layered/resolved-mip target would need a
            // subresource index derived from the view.
            resolves.push(Dx12ColorResolve {
                src,
                dst,
                dst_subresource: 0,
                format,
            });
        }
        entering.push(
            &attachment.resource,
            attachment.enter_state,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        color_attachments.push(attachment);
    }

    // Sparse portable MRT is a legal scope: `begin.colors` carries each
    // attachment's own `location`, so locations need not be dense. D3D12
    // `OMSetRenderTargets` takes a dense array indexed by attachment slot, so a
    // hole (a location with no attachment) must be filled with an explicit null
    // RTV descriptor. The null descriptor's format is taken from the first real
    // attachment; writes to a null slot are discarded by the fixed-function
    // stage, which is exactly what "no attachment at this slot" means.
    let max_location = begin
        .colors
        .iter()
        .map(|(location, _)| *location)
        .max()
        .unwrap_or(0);
    let null_format = begin
        .colors
        .first()
        .and_then(|(_, color)| dxgi_format(color.view.format()));
    let mut handle_by_location: Vec<Option<D3D12_CPU_DESCRIPTOR_HANDLE>> =
        vec![None; (max_location + 1) as usize];
    for ((location, _), attachment) in begin.colors.iter().zip(&color_attachments) {
        handle_by_location[*location as usize] = Some(attachment.handle);
    }
    // One CPU RTV heap per hole, created eagerly. A null RTV has no backing
    // resource; it lives exactly as long as the scope, matching the real
    // attachments' per-descriptor heap lifetime.
    let mut null_rtv_heaps = Vec::new();
    if let Some(format) = null_format {
        for slot in handle_by_location.iter_mut() {
            if slot.is_none() {
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
                // `pResource = None` creates an RTV with no backing resource;
                // writes to this slot are discarded rather than reaching any
                // memory. This is the D3D12-native spelling of a sparse MRT hole.
                unsafe { device.CreateRenderTargetView(None, Some(&desc), handle) };
                *slot = Some(handle);
                null_rtv_heaps.push(heap);
            }
        }
    }
    let color_handles = handle_by_location
        .iter()
        .map(|handle| {
            handle.expect("every MRT slot is either a real or a null RTV descriptor")
        })
        .collect::<Vec<_>>();
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
            color_handles.len() as u32,
            Some(color_handles.as_ptr()),
            false,
            depth_stencil.as_ref().map(std::ptr::from_ref),
        );
    }
    // Clear each real attachment at its own location. `color_handles` is indexed
    // by attachment slot (sparse, with null RTVs at holes), so a real attachment
    // may sit at a higher index than its position in `begin.colors`; the zip
    // below must therefore look each attachment up by stored location rather
    // than by enumeration order.
    for (location, attachment) in &begin.colors {
        let handle = color_handles[*location as usize];
        if let LoadOp::Clear(value) = attachment.load {
            unsafe { list.ClearRenderTargetView(handle, &clear_color(value), None) };
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
    let attachment_extent = begin
        .colors
        .first()
        .map(|(_, color)| color.view.extent())
        .or_else(|| {
            begin
                .depth_stencil
                .as_ref()
                .map(|depth| depth.view.extent())
        })
        .ok_or_else(|| {
            unsupported(
                "a raster scope without attachments",
                "portable validation must reject it before DX12 lowering",
            )
        })?;
    Ok(RasterScopeState {
        colors: color_attachments,
        null_rtv_heaps,
        resolves,
        color_resolves,
        depth_view,
        depth_heap,
        depth_read_only,
        attachment_extent,
    })
}

fn lower_color_attachment(
    device: &ID3D12Device,
    view: &ColorAttachmentView,
    depth_slice: Option<u32>,
) -> Result<Dx12RenderAttachment, Dx12Failure> {
    match view {
        ColorAttachmentView::Texture(view) => {
            let resource = dx12_texture(view.texture())?.resource().clone();
            let (heap, handle) = create_rtv(device, view, depth_slice)?;
            Ok(Dx12RenderAttachment {
                resource,
                handle,
                heap,
                enter_state: D3D12_RESOURCE_STATE_COMMON,
                leave_state: D3D12_RESOURCE_STATE_COMMON,
                retention: AttachmentRetention::Texture(view.clone()),
            })
        }
        ColorAttachmentView::Frame(frame) => {
            if depth_slice.is_some() {
                return Err(unsupported(
                    "a depth slice on a presentation attachment",
                    "a frame attachment is not a 3D texture",
                ));
            }
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
            let resource = native.resource().clone();
            let (heap, handle) = create_frame_rtv(device, &resource, frame.format())?;
            Ok(Dx12RenderAttachment {
                resource,
                handle,
                heap,
                // PRESENT is numerically COMMON, but keeping the symbolic state
                // here protects the swapchain ownership invariant from being
                // erased by the native unification.
                enter_state: D3D12_RESOURCE_STATE_PRESENT,
                leave_state: D3D12_RESOURCE_STATE_PRESENT,
                retention: AttachmentRetention::Frame(frame.clone()),
            })
        }
    }
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
            ResourceUse::AccelerationStructure(_) => {
                return Err(unsupported(
                    "an acceleration structure used by a raster draw",
                    "DX12 acceleration-structure binding lowering is not enabled",
                ));
            }
            ResourceUse::Query(_) => {}
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
        for write in &draw.immediates {
            let (parameter, destination) = pipeline
                .immediate_root_parameter(write.offset, write.bytes.len() as u32)
                .ok_or_else(|| {
                    unsupported(
                        "an immediate write outside the DX12 root-constant layout",
                        "portable validation must keep writes within declared ranges",
                    )
                })?;
            let values: Vec<u32> = write
                .bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte immediate word")))
                .collect();
            list.SetGraphicsRoot32BitConstants(
                parameter,
                values.len() as u32,
                values.as_ptr().cast(),
                destination,
            );
        }
        if let Some((_, group)) = groups.first() {
            // D3D12 permits precisely one CBV/SRV/UAV and one sampler heap to
            // be active. A group's table addresses are offsets in these shared
            // device heaps, so bind both before setting graphics roots.
            list.SetDescriptorHeaps(&[
                Some(group.view_heap().clone()),
                Some(group.sampler_heap().clone()),
            ]);
        }
        for (bound, group) in &groups {
            if let Some(parameter) = pipeline.view_root_parameter(bound.index.get()) {
                list.SetGraphicsRootDescriptorTable(parameter, group.view_table());
            }
            if let Some(parameter) = pipeline.sampler_root_parameter(bound.index.get()) {
                list.SetGraphicsRootDescriptorTable(parameter, group.sampler_table());
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

/// Executes one native draw signature after reusing the ordinary raster state
/// encoder.  The zero-count setup draw emits no primitives; it exists solely to
/// keep direct and indirect binding/IA/dynamic-state lowering identical.
pub(super) fn lower_raster_indirect(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    indirect: &RasterIndirect,
    uses: &[ResourceUse],
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let setup = RasterDraw {
        pipeline: indirect.pipeline.clone(),
        groups: indirect.groups.clone(),
        vertex_buffers: indirect.vertex_buffers.clone(),
        index: indirect.index.clone(),
        viewport: indirect.viewport,
        scissor: indirect.scissor,
        blend_constant: indirect.blend_constant,
        stencil_reference: indirect.stencil_reference,
        range: 0..0,
        instances: 0..0,
        base_vertex: 0,
        immediates: Vec::new(),
    };
    lower_raster_draw(list, &setup, uses, scope, committed)?;
    let indexed = indirect.index.is_some();
    let kind = if indexed {
        D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED
    } else {
        D3D12_INDIRECT_ARGUMENT_TYPE_DRAW
    };
    let desc = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: kind,
        Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
    };
    let signature_desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: indirect.stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &desc,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    unsafe {
        device
            .CreateCommandSignature(
                &signature_desc,
                None::<&ID3D12RootSignature>,
                &mut signature,
            )
            .map_err(|error| {
                Dx12Failure::Native(crate::backend::dx12::ffi::NativeError::new(
                    &error,
                    "ID3D12Device::CreateCommandSignature",
                ))
            })?;
    }
    let signature = signature.ok_or_else(|| {
        unsupported(
            "a missing raster command signature",
            "CreateCommandSignature returned success without a signature",
        )
    })?;
    let arguments = dx12_buffer(&indirect.arguments)?;
    let count = match indirect.count.as_ref() {
        Some((buffer, offset, max)) => Some((dx12_buffer(buffer)?, *offset, *max)),
        None => None,
    };
    let mut entering = Transitions::default();
    entering.push(
        arguments.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    if let Some((buffer, _, _)) = &count {
        entering.push(
            buffer.resource(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        );
    }
    entering.record(list);
    unsafe {
        list.ExecuteIndirect(
            &signature,
            count
                .as_ref()
                .map_or(indirect.draw_count, |(_, _, max)| *max),
            arguments.resource(),
            indirect.arguments_offset,
            count.as_ref().map(|(buffer, _, _)| buffer.resource()),
            count.as_ref().map_or(0, |(_, offset, _)| *offset),
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        arguments.resource(),
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COMMON,
    );
    if let Some((buffer, _, _)) = &count {
        leaving.push(
            buffer.resource(),
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
    }
    leaving.record(list);
    committed.indirect_buffers.push(indirect.arguments.clone());
    if let Some((buffer, _, _)) = &indirect.count {
        committed.indirect_buffers.push(buffer.clone());
    }
    committed.command_signatures.push(signature);
    Ok(())
}

pub(super) fn lower_raster_end(
    list: &ID3D12GraphicsCommandList,
    scope: RasterScopeState,
    committed: &mut CommittedBatch,
) {
    let RasterScopeState {
        colors,
        null_rtv_heaps: _,
        resolves,
        color_resolves,
        depth_view,
        depth_heap,
        depth_read_only,
        attachment_extent: _,
    } = scope;

    // Multisample color resolve, before the source leaves RENDER_TARGET. Each
    // attachment's resolve target receives the resolved single-sample result.
    if !resolves.is_empty() {
        let mut resolve_transitions = Transitions::default();
        for op in &resolves {
            resolve_transitions.push(
                &op.src,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            );
            resolve_transitions.push(
                &op.dst,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
            );
        }
        resolve_transitions.record(list);
        for op in &resolves {
            // SAFETY: both resources are live DXGI allocations owned by the
            // attached texture views, and the resolve is issued while the source
            // is still `RESOLVE_SOURCE`.
            unsafe {
                list.ResolveSubresource(
                    &op.dst,
                    op.dst_subresource,
                    &op.src,
                    0,
                    op.format,
                )
            };
        }
    }

    let mut leaving = Transitions::default();
    // A resolve source leaves from `RESOLVE_SOURCE` (it was transitioned there
    // by the resolve block above), not from `RENDER_TARGET`. Transitioning a
    // resolve source from `RENDER_TARGET` to COMMON would name a start state it
    // is no longer in, which D3D12 treats as a resource-state violation.
    for (index, attachment) in colors.iter().enumerate() {
        let from = if color_resolves.get(index).copied().unwrap_or(false) {
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE
        } else {
            D3D12_RESOURCE_STATE_RENDER_TARGET
        };
        leaving.push(&attachment.resource, from, attachment.leave_state);
    }
    for op in &resolves {
        // The resolved destination returns to COMMON with the source's leaving
        // phase, so a later copy/read is legal.
        leaving.push(
            &op.dst,
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        );
    }
    if let Some(view) = &depth_view {
        if let Ok(texture) = dx12_texture(view.texture()) {
            leaving.push(
                texture.resource(),
                if depth_read_only {
                    D3D12_RESOURCE_STATE_DEPTH_READ
                } else {
                    D3D12_RESOURCE_STATE_DEPTH_WRITE
                },
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
    }
    leaving.record(list);
    for attachment in colors {
        committed.raster_descriptor_heaps.push(attachment.heap);
        match attachment.retention {
            AttachmentRetention::Texture(view) => committed.raster_views.push(view),
            AttachmentRetention::Frame(frame) => committed.raster_frames.push(frame),
        }
    }
    if let Some(view) = depth_view {
        committed.raster_views.push(view);
    }
    if let Some(heap) = depth_heap {
        committed.raster_descriptor_heaps.push(heap);
    }
}

fn create_rtv(
    device: &ID3D12Device,
    view: &TextureView,
    depth_slice: Option<u32>,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let texture = dx12_texture(view.texture())?;
    let descriptor = view.descriptor();
    // A multisampled color attachment is a legitimate MSAA render target: the
    // resource carries `SampleDesc.Count = sample_count`, and an RTV on it is
    // dimensioned by D3D12 to match. The color RTV form is identical regardless
    // of sample count; only resolve-aware layout differs, handled in scope end.
    let format = dxgi_format(view.format()).ok_or_else(|| {
        unsupported(
            "a color attachment format without DXGI mapping",
            "DX12 cannot create its RTV",
        )
    })?;
    let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    // A portable D2 view selects one array layer. D3D12 requires the ARRAY
    // descriptor form whenever the resource has multiple layers; TEXTURE2D
    // would silently target layer zero and ignore `base_layer`.
    let desc = if matches!(view.texture().descriptor().dimension, TextureDimension::D3)
        && matches!(descriptor.dimension, TextureViewDimension::D3)
    {
        let slice = depth_slice.ok_or_else(|| {
            unsupported(
                "a 3D color attachment without a depth slice",
                "the portable attachment requires an explicit 3D slice",
            )
        })?;
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_RTV {
                    MipSlice: descriptor.base_mip,
                    FirstWSlice: slice,
                    WSize: 1,
                },
            },
        }
    } else if matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        && matches!(
            descriptor.dimension,
            TextureViewDimension::D2 | TextureViewDimension::D2Array
        )
        && (view.texture().descriptor().array_layers > 1
            || matches!(descriptor.dimension, TextureViewDimension::D2Array))
    {
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                    MipSlice: descriptor.base_mip,
                    FirstArraySlice: descriptor.base_layer,
                    ArraySize: 1,
                    PlaneSlice: 0,
                },
            },
        }
    } else if matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        && matches!(descriptor.dimension, TextureViewDimension::D2)
    {
        if depth_slice.is_some() {
            return Err(unsupported(
                "a depth slice on a 2D color attachment",
                "only a 3D RTV accepts a depth slice",
            ));
        }
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV {
                    MipSlice: descriptor.base_mip,
                    PlaneSlice: 0,
                },
            },
        }
    } else {
        return Err(unsupported(
            "a color attachment view dimension",
            "DX12 supports this backend's 2D and explicit-slice 3D RTV paths only",
        ));
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
        || !matches!(
            descriptor.dimension,
            TextureViewDimension::D2 | TextureViewDimension::D2Array
        )
    {
        // A multisampled 2D depth attachment is legitimate MSAA; the DSV form is
        // the same 2D descriptor regardless of sample count (D3D12 reads the
        // sample count from the resource). Only 2D (and 2D-array) depth is
        // lowered here.
        return Err(unsupported(
            "a non-2D depth attachment view",
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
    let desc = if view.texture().descriptor().array_layers > 1
        || matches!(descriptor.dimension, TextureViewDimension::D2Array)
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
            Flags: flags,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: descriptor.base_mip,
                    FirstArraySlice: descriptor.base_layer,
                    ArraySize: 1,
                },
            },
        }
    } else {
        D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            Flags: flags,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_DSV {
                    MipSlice: descriptor.base_mip,
                },
            },
        }
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
    let extent = scope.attachment_extent;
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
    let extent = scope.attachment_extent;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A presentation attachment has no `TextureView`. Keep the default dynamic
    /// state source separate from texture-view retention so frame-only scopes do
    /// not regress into the historical `.first().expect()` panic.
    #[test]
    fn default_dynamic_state_uses_the_recorded_attachment_extent() {
        let scope = RasterScopeState {
            colors: Vec::new(),
            null_rtv_heaps: Vec::new(),
            resolves: Vec::new(),
            color_resolves: Vec::new(),
            depth_view: None,
            depth_heap: None,
            depth_read_only: false,
            attachment_extent: Extent3d::d2(640, 480),
        };

        let viewport = default_viewport(&scope);
        assert_eq!((viewport.width, viewport.height), (640.0, 480.0));
        let scissor = default_scissor(&scope);
        assert_eq!((scissor.width, scissor.height), (640, 480));
    }
}
