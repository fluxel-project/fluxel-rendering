//! Lowering the two transfer directions: host bytes into a buffer, and a buffer
//! back out to the host.
//!
//! The two are mirror images that share a staging heap choice and almost nothing
//! else, which is why they live together and why the retention type below carries
//! both — the difference between them is the difference between a `Vec<Dx12Buffer>`
//! and a ticket, and a reader comparing the pair should see that in one place.
//!
//! # Why the staging outlives the recording
//!
//! `ExecuteCommandLists` is asynchronous, so a staging allocation whose last
//! reference is dropped when recording returns would be freed while the GPU is
//! still copying through it. [`CommittedBatch`] is what holds it until the fence
//! reports the batch finished, and it is the only thing in this chapter that
//! exists for that reason.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BOX, D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_FLAG_STENCIL, D3D12_CLEAR_FLAGS,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DEPTH_STENCIL_VIEW_DESC, D3D12_DEPTH_STENCIL_VIEW_DESC_0,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_DIMENSION_TEXTURE2DARRAY, D3D12_DSV_FLAG_NONE,
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_TEX2D_ARRAY_DSV, D3D12_TEX2D_DSV,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    ID3D12CommandSignature, ID3D12DescriptorHeap, ID3D12Device, ID3D12GraphicsCommandList,
    ID3D12Resource,
};

use crate::api::binding::BindGroup;
use crate::api::command::ResourceUse;
use crate::api::command::copy::{BufferTextureCopy, TextureCopy};
use crate::api::format::{block_extent, logical_bytes_per_block};
use crate::api::pipeline::{ComputePipeline, RasterPipeline};
use crate::api::presentation::FrameAttachment;
use crate::api::query::QuerySet;
use crate::api::resource::buffer::Buffer;
use crate::api::resource::subresource::{
    TextureAspect, TextureAspects, TextureSubresourceLayers, TextureSubresourceRange,
};
use crate::api::resource::texture::{Texture, TextureDimension, mip_extent};
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTexelLayout, ReadbackTicket, UploadDescriptor,
    UploadJob,
};
use crate::api::resource::view::TextureView;
use crate::backend::dx12::ffi;
use crate::backend::dx12::platform::facts::dxgi_format;
use crate::backend::dx12::resource::{Dx12Buffer, StagingHeap, create_staging, readback_bytes};

use super::transition::Transitions;
use super::{dx12_buffer, dx12_texture};
use crate::backend::dx12::failure::{Dx12Failure, ref_native};

/// A batch that has been committed, and the host-visible memory its command list
/// reads or writes.
///
/// Both halves are needed for the same reason: `ExecuteCommandLists` is
/// asynchronous, so a staging allocation whose last reference is dropped when
/// recording returns would be freed while the GPU is still copying out of it.
/// Retaining it here until the fence reports the batch finished is what keeps the
/// list's resource references valid for the list's whole life.
///
/// TODO(perf): This batch retention is the current resource-retirement authority:
/// native staging, CPU attachment heaps, and portable handles are released only
/// after its serial completes. A general deferred-destruction queue may replace
/// these per-kind vectors, but every native object must be retired after the
/// maximum completion serial of all batches that reference it. Device loss is an
/// exception: staging remains conservatively retained through teardown because a
/// lost fence is not proof native DMA stopped. This changes no public ownership
/// rule; `CompletionPoint` is already the portable retirement boundary.
pub(super) struct CommittedBatch {
    /// The serial that reports this batch's completion.
    pub(super) serial: u64,
    /// Upload staging: written by the CPU before the commit, read by the GPU
    /// during it, and of no use afterwards.
    pub(super) staging: Vec<Dx12Buffer>,
    /// Readback staging: written by the GPU, read by the CPU once the serial is
    /// reached, and then published to the ticket that asked for it.
    pub(super) readbacks: Vec<ReadbackRetention>,
    /// Compute objects whose native descriptor tables and PSOs are referenced by
    /// this batch's command list. They remain alive until the batch fence passes.
    pub(super) compute_pipelines: Vec<ComputePipeline>,
    /// Raster state and the attachment views referenced by a graphics command
    /// list. D3D12 descriptors are addresses, so their owners must outlive GPU
    /// execution just like upload staging does.
    pub(super) raster_pipelines: Vec<RasterPipeline>,
    pub(super) raster_buffers: Vec<Buffer>,
    pub(super) raster_views: Vec<TextureView>,
    pub(super) raster_frames: Vec<FrameAttachment>,
    pub(super) raster_textures: Vec<Texture>,
    pub(super) raster_descriptor_heaps: Vec<ID3D12DescriptorHeap>,
    pub(super) bind_groups: Vec<BindGroup>,
    /// Query heaps and indirect-argument buffers referenced by native commands.
    /// D3D12 command lists retain neither COM query heaps nor portable buffer
    /// owners; these handles therefore live through the batch fence.
    pub(super) query_sets: Vec<QuerySet>,
    pub(super) indirect_buffers: Vec<Buffer>,
    /// `ExecuteIndirect` only borrows its command signature. Retain the COM
    /// object until the submission fence reports completion.
    pub(super) command_signatures: Vec<ID3D12CommandSignature>,
    /// Every portable resource use named by the accepted work.
    ///
    /// D3D12 command lists do not make the application's portable resource
    /// ownership live until execution completes.  In particular, a transfer
    /// recording can be the last owner of an otherwise unbound texture or
    /// buffer; `SubmissionPlan` is consumed immediately after Phase B, while
    /// the queue is still reading that native object.  Retaining the recorded
    /// uses here makes the fence the single retirement boundary for *all*
    /// command paths, instead of relying on individual lowerings to remember
    /// an ad-hoc keep-alive vector.
    ///
    /// This intentionally retains the portable `ResourceUse`, not raw COM
    /// pointers: it covers buffers, textures, query sets and future resource
    /// categories with their normal Device identity/lifetime semantics.
    pub(super) resource_uses: Vec<ResourceUse>,
}

/// A readback's staging buffer and the ticket waiting on it.
pub(super) struct ReadbackRetention {
    /// The `READBACK` heap allocation the GPU copies into.
    pub(super) staging: Dx12Buffer,
    /// The ticket whose bytes these are.
    pub(super) ticket: ReadbackTicket,
    /// How many bytes were copied, which is the range's size rather than the
    /// buffer's.
    pub(super) size: u64,
    /// The native packed-footprint layout, when the source was a texture.
    pub(super) layout: Option<ReadbackTexelLayout>,
}

/// Lowers a buffer upload: staging copy from caller bytes, then GPU copy.
///
/// # Why the CPU write happens here and not at `create_buffer_upload`
///
/// The upload heap allocation could have been made and filled when the job
/// was created, since the bytes are already retained and immutable
/// (section 17.2). It is made here instead because the allocation must
/// outlive the *commit*, not the job: a staging buffer created at job
/// creation would be held by the job, which the caller may keep for as long
/// as it likes, and a job encoded into several plans would need one staging
/// buffer per plan anyway. Creating it inside the recording ties its life to
/// the batch that reads it, which is exactly the lifetime the fence reports.
pub(super) fn lower_upload(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    job: &UploadJob,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let UploadDescriptor::Buffer(descriptor) = job.descriptor() else {
        return lower_texture_upload(device, list, job, committed);
    };
    let destination = dx12_buffer(&descriptor.dst)?;

    let length = descriptor.bytes.len();
    let staging =
        create_staging(device, length as u64, StagingHeap::Upload).map_err(Dx12Failure::Native)?;

    let mut pointer: *mut core::ffi::c_void = std::ptr::null_mut();
    // SAFETY: `Map` on an `UPLOAD` heap resource makes the whole allocation
    // CPU-writable and writes the address into `pointer`; the null read range
    // is what Direct3D 12 requires for a write-only heap. The mapping stays
    // live until the `Unmap` below.
    unsafe {
        staging
            .resource()
            .Map(0, None, Some(&mut pointer))
            .map_err(|error| ref_native(&error))?;
    }
    let Some(pointer) = std::ptr::NonNull::new(pointer.cast::<u8>()) else {
        return Err(Dx12Failure::Native(
            ffi::NativeError::driver_contract_violation(
                "Map reported success without producing a pointer",
                "Dx12Device::submit",
            ),
        ));
    };
    // SAFETY: the mapping covers `length` bytes because that is the resource's
    // own width, which is what `create_staging` was asked for. The source is
    // the job's retained `Arc<[u8]>`, alive for the whole recording, and host
    // memory and a GPU allocation cannot overlap. `Unmap` follows the copy and
    // is the single matching call for the single mapping above.
    unsafe {
        std::ptr::copy_nonoverlapping(descriptor.bytes.as_ptr(), pointer.as_ptr(), length);
        staging.resource().Unmap(0, None);
    }

    let mut entering = Transitions::default();
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);
    // SAFETY: the staging buffer was created in `GENERIC_READ` and stays
    // there — Direct3D 12 permits no transition in an upload heap — which is
    // a state the copy's source half may be in. The destination was put in
    // `COPY_DEST` by the barrier above, and `dst_offset` plus `length` was
    // validated against the destination's size at job creation (section 17.3).
    unsafe {
        list.CopyBufferRegion(
            destination.resource(),
            descriptor.dst_offset,
            staging.resource(),
            0,
            length as u64,
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);

    committed.staging.push(staging);
    Ok(())
}

/// Zeroes a validated buffer range without requiring public `STORAGE` usage.
///
/// D3D12 has a UAV clear, but that would turn a copy-destination clear into a
/// different public resource contract.  A zero-filled upload allocation plus
/// `CopyBufferRegion` preserves `COPY_DST` semantics and works for every buffer
/// the portable `ClearBuffer` command admits.
pub(super) fn lower_buffer_clear(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    buffer: &Buffer,
    range: crate::api::resource::BufferRange,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let destination = dx12_buffer(buffer)?;
    let host_len = usize::try_from(range.size).map_err(|_| Dx12Failure::Unsupported {
        what: "a buffer clear too large for this host address space",
        why: "the zero-upload fallback must materialize every cleared byte before GPU execution",
    })?;
    let staging =
        create_staging(device, range.size, StagingHeap::Upload).map_err(Dx12Failure::Native)?;
    let mut pointer: *mut core::ffi::c_void = std::ptr::null_mut();
    unsafe {
        staging
            .resource()
            .Map(0, None, Some(&mut pointer))
            .map_err(|error| ref_native(&error))?;
    }
    let pointer = std::ptr::NonNull::new(pointer.cast::<u8>()).ok_or_else(|| {
        Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
            "Map reported success without producing a pointer",
            "Dx12Device::submit",
        ))
    })?;
    unsafe {
        std::ptr::write_bytes(pointer.as_ptr(), 0, host_len);
        staging.resource().Unmap(0, None);
    }
    let mut entering = Transitions::default();
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);
    unsafe {
        list.CopyBufferRegion(
            destination.resource(),
            range.offset,
            staging.resource(),
            0,
            range.size,
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    committed.staging.push(staging);
    committed.indirect_buffers.push(buffer.clone());
    Ok(())
}

/// Clears complete color subresources by copying an explicitly zeroed upload
/// footprint into each one.
///
/// D3D12 has no format-independent equivalent of WebGPU's texture-clear
/// command.  A `COPY_DST` upload is nevertheless a real native route, not a
/// shader fallback: it preserves the public command's zero-value semantics for
/// every color format with a fixed block layout, including BC/ETC/ASTC. Depth
/// and stencil use the DSV route below; planar formats are not created by this
/// backend and therefore cannot reach either path.
pub(super) fn lower_texture_clear(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &Texture,
    range: TextureSubresourceRange,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    if range.aspects != TextureAspects::COLOR {
        return lower_depth_stencil_clear(device, list, texture, range, committed);
    }
    let destination = dx12_texture(texture)?;
    let descriptor = texture.descriptor();
    let mut copies = Vec::new();
    let mut next_offset = 0_u64;
    for mip_level in range.base_mip..range.base_mip + range.mip_count {
        let extent = mip_extent(descriptor.extent, descriptor.dimension, mip_level);
        let images = if descriptor.dimension == TextureDimension::D3 {
            1
        } else {
            range.layer_count
        };
        for layer in 0..images {
            let layers = TextureSubresourceLayers {
                aspect: TextureAspect::Color,
                mip_level,
                base_layer: range.base_layer,
                layer_count: range.layer_count,
            };
            let subresource = texture_subresource(texture, layers, layer)?;
            let (footprint, _, _) =
                region_footprint(device, texture, subresource, extent, next_offset)?;
            let bytes = u64::from(footprint.Footprint.RowPitch)
                .checked_mul(u64::from(block_rows(descriptor.format, extent.height)))
                .and_then(|value| value.checked_mul(u64::from(extent.depth)))
                .ok_or_else(|| {
                    Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                        "texture clear upload footprint overflowed",
                        "Dx12Device::submit",
                    ))
                })?;
            next_offset = align_up(
                next_offset.checked_add(bytes).ok_or_else(|| {
                    Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                        "texture clear staging allocation overflowed",
                        "Dx12Device::submit",
                    ))
                })?,
                512,
            )?;
            copies.push((subresource, footprint));
        }
    }
    let staging =
        create_staging(device, next_offset, StagingHeap::Upload).map_err(Dx12Failure::Native)?;
    let mut pointer: *mut core::ffi::c_void = std::ptr::null_mut();
    unsafe {
        staging
            .resource()
            .Map(0, None, Some(&mut pointer))
            .map_err(|error| ref_native(&error))?;
    }
    let clear_result = (|| {
        let pointer = std::ptr::NonNull::new(pointer.cast::<u8>()).ok_or_else(|| {
            Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                "Map reported success without producing a pointer",
                "Dx12Device::submit",
            ))
        })?;
        let length = usize::try_from(next_offset).map_err(|_| Dx12Failure::Unsupported {
            what: "a texture clear staging allocation too large for this host address space",
            why: "the native zero-upload route must materialize all cleared texels before execution",
        })?;
        unsafe { std::ptr::write_bytes(pointer.as_ptr(), 0, length) };
        Ok(())
    })();
    unsafe { staging.resource().Unmap(0, None) };
    clear_result?;

    let mut entering = Transitions::default();
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);
    for (subresource, footprint) in copies {
        let mut source = footprint_location(staging.resource(), footprint);
        let mut target = texture_location(destination.resource(), subresource);
        unsafe {
            list.CopyTextureRegion(&target, 0, 0, 0, &source, None);
            drop_copy_location(&mut source);
            drop_copy_location(&mut target);
        }
    }
    let mut leaving = Transitions::default();
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    committed.staging.push(staging);
    // A command list only borrows the resource; retain the portable owner to
    // the completion serial just like attachment textures.
    committed.raster_textures.push(texture.clone());
    Ok(())
}

/// Clears selected depth/stencil subresources through temporary CPU-only DSVs.
///
/// A DSV describes one mip/layer, hence every selected array layer receives its
/// own descriptor. The descriptors and portable texture owner are retained in
/// `CommittedBatch` through the fence just as raster attachments are.  D3D12's
/// depth/stencil clear values are explicitly zero here, matching ClearTexture's
/// backend-defined zero contract rather than a raster pass's caller-supplied
/// load value.
fn lower_depth_stencil_clear(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &Texture,
    range: TextureSubresourceRange,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    if range.aspects.contains(TextureAspects::COLOR)
        || !(range.aspects.contains(TextureAspects::DEPTH)
            || range.aspects.contains(TextureAspects::STENCIL))
    {
        return Err(Dx12Failure::Unsupported {
            what: "a mixed or non-depth/stencil texture clear",
            why: "a D3D12 DSV clear selects depth/stencil planes and cannot be combined with a color upload clear",
        });
    }
    let native = dx12_texture(texture)?;
    let descriptor = texture.descriptor();
    if descriptor.dimension != TextureDimension::D2 || descriptor.sample_count != 1 {
        return Err(Dx12Failure::Unsupported {
            what: "a non-single-sample 2D depth/stencil texture clear",
            why: "the DX12 ClearTexture DSV route lowers the same 2D DSV shape as raster attachment lowering",
        });
    }
    let format = dxgi_format(descriptor.format).ok_or(Dx12Failure::Unsupported {
        what: "a depth/stencil clear format without an exact DXGI format",
        why: "DX12 must create a typed DSV for the texture's declared format",
    })?;
    let mut flags = D3D12_CLEAR_FLAGS(0);
    if range.aspects.contains(TextureAspects::DEPTH) {
        flags |= D3D12_CLEAR_FLAG_DEPTH;
    }
    if range.aspects.contains(TextureAspects::STENCIL) {
        flags |= D3D12_CLEAR_FLAG_STENCIL;
    }
    let mut entering = Transitions::default();
    entering.push(
        native.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
    );
    entering.record(list);
    for mip in range.base_mip..range.base_mip + range.mip_count {
        for layer in range.base_layer..range.base_layer + range.layer_count {
            let heap = unsafe {
                device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
            }
            .map_err(|error| ref_native(&error))?;
            let handle: D3D12_CPU_DESCRIPTOR_HANDLE =
                unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
            let dsv = if descriptor.array_layers > 1 {
                D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                    Flags: D3D12_DSV_FLAG_NONE,
                    Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                            MipSlice: mip,
                            FirstArraySlice: layer,
                            ArraySize: 1,
                        },
                    },
                }
            } else {
                D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_NONE,
                    Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_DSV { MipSlice: mip },
                    },
                }
            };
            unsafe {
                device.CreateDepthStencilView(native.resource(), Some(&dsv), handle);
                list.ClearDepthStencilView(handle, flags, 0.0, 0, None);
            }
            committed.raster_descriptor_heaps.push(heap);
        }
    }
    let mut leaving = Transitions::default();
    leaving.push(
        native.resource(),
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    committed.raster_textures.push(texture.clone());
    Ok(())
}

/// Lowers a buffer readback: GPU copy into staging, then a ticket the drain
/// publishes from.
///
/// The reverse of [`lower_upload`] in every respect, including which
/// way the staging is retained: upload staging is dead the moment the batch
/// finishes, while readback staging is the thing the batch's completion is
/// *for*.
pub(super) fn lower_readback(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    ticket: &ReadbackTicket,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let ReadbackRequest::Buffer { src, range, .. } = ticket.request() else {
        return lower_texture_readback(device, list, ticket, committed);
    };
    let source = dx12_buffer(src)?;

    let staging =
        create_staging(device, range.size, StagingHeap::Readback).map_err(Dx12Failure::Native)?;

    let mut entering = Transitions::default();
    entering.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    entering.record(list);
    // SAFETY: the staging buffer was created in `COPY_DEST` and stays there —
    // Direct3D 12 permits no transition in a readback heap — which is a state
    // the copy's destination half may be in. The source was put in
    // `COPY_SOURCE` by the barrier above, and the range was validated against
    // the source's size at record time (section 18.1).
    unsafe {
        list.CopyBufferRegion(
            staging.resource(),
            0,
            source.resource(),
            range.offset,
            range.size,
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);

    committed.readbacks.push(ReadbackRetention {
        staging,
        ticket: ticket.clone(),
        size: range.size,
        layout: None,
    });
    Ok(())
}

/// Copies one finished readback's bytes off the GPU and hands them to its ticket.
///
/// A non-terminal mapping refusal becomes [`ReadbackStatus::Failed`]. A removal,
/// reset, hang, or driver-internal failure becomes `DeviceLost` and returns its
/// diagnosis to the caller so it can update the shared loss authority after
/// releasing the command-spine lock.
pub(super) fn publish_readback(
    retention: &ReadbackRetention,
) -> Option<crate::api::platform::DeviceLossInfo> {
    match readback_bytes(&retention.staging, retention.size) {
        // A buffer range is tightly packed by definition, so there is no texel
        // layout to publish beside its bytes.
        Ok(bytes) => {
            retention.ticket.publish(bytes, retention.layout);
            None
        }
        Err(failure) if failure.failure().is_terminal() => {
            let info = crate::api::platform::DeviceLossInfo::new(format!(
                "Direct3D 12 reported a terminal failure while mapping completed readback data: {}",
                failure.as_error()
            ));
            retention.ticket.set_status(ReadbackStatus::DeviceLost);
            Some(info)
        }
        Err(_) => {
            retention.ticket.set_status(ReadbackStatus::Failed);
            None
        }
    }
}

/// Aligns an offset for D3D12 placed-footprint placement.
fn align_up(value: u64, alignment: u64) -> Result<u64, Dx12Failure> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| {
            Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                "texture transfer staging allocation overflowed",
                "Dx12Device::submit",
            ))
        })
}

/// Number of physical block rows occupied by a logical texel height.
///
/// Copy descriptors are expressed in logical texels, while D3D12's placed
/// footprints are laid out in format blocks.  Keeping this conversion beside
/// the DX12 footprint lowering prevents a compressed upload/readback from
/// accidentally treating its 4x4 blocks as sixteen independent rows.
fn block_rows(format: crate::api::format::TextureFormat, height: u32) -> u32 {
    let (_, block_height) = block_extent(format);
    height.div_ceil(block_height)
}

/// Number of bytes in one physical block row of a logical texel width.
fn block_row_bytes(
    format: crate::api::format::TextureFormat,
    width: u32,
) -> Result<u64, Dx12Failure> {
    let (block_width, _) = block_extent(format);
    let blocks = width.div_ceil(block_width);
    let bytes = logical_bytes_per_block(format).ok_or(Dx12Failure::Unsupported {
        what: "a texture transfer with an abstract texel byte size",
        why: "the portable format has no fixed host byte layout for staging repacking",
    })?;
    Ok(u64::from(blocks) * u64::from(bytes))
}

fn texture_subresource(
    texture: &Texture,
    layers: TextureSubresourceLayers,
    layer: u32,
) -> Result<u32, Dx12Failure> {
    let descriptor = texture.descriptor();
    let plane = match layers.aspect {
        TextureAspect::Color | TextureAspect::Depth => 0,
        // P0 has no stencil-only texture copy route in the DX12 facts.  Do not
        // guess a DXGI plane mapping: it differs between depth-stencil formats.
        TextureAspect::Stencil => {
            return Err(Dx12Failure::Unsupported {
                what: "a stencil-plane texture transfer",
                why: "the DX12 lowering currently exposes only plane-zero color/depth copies",
            });
        }
        // Multi-planar formats require per-format DXGI plane arithmetic and
        // plane-specific footprint validation.  The DX12 fact table keeps their
        // transfer routes disabled until that entire path is implemented.
        TextureAspect::Plane0 | TextureAspect::Plane1 | TextureAspect::Plane2 => {
            return Err(Dx12Failure::Unsupported {
                what: "a multi-planar texture transfer",
                why: "DX12 plane-specific transfer lowering is not enabled",
            });
        }
    };
    let array = match descriptor.dimension {
        crate::api::resource::texture::TextureDimension::D3 => 0,
        _ => layers.base_layer + layer,
    };
    Ok(layers.mip_level
        + array * descriptor.mip_levels
        + plane * descriptor.mip_levels * descriptor.array_layers)
}

fn region_footprint(
    device: &ID3D12Device,
    texture: &Texture,
    subresource: u32,
    extent: crate::api::resource::texture::Extent3d,
    offset: u64,
) -> Result<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64), Dx12Failure> {
    let native = dx12_texture(texture)?;
    // SAFETY: GetDesc only reads the immutable resource descriptor.
    let description = unsafe { native.resource().GetDesc() };
    let mut placed = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut rows = 0;
    let mut row_size = 0;
    let mut ignored_total = 0;
    // SAFETY: all output pointers name initialized locals and the resource
    // description remains valid throughout the call.
    unsafe {
        device.GetCopyableFootprints(
            &description,
            subresource,
            1,
            offset,
            Some(&mut placed),
            Some(&mut rows),
            Some(&mut row_size),
            Some(&mut ignored_total),
        );
    }
    // The driver supplied the format and the required 256-byte pitch.  Only the
    // copied box dimensions differ from the whole-mip footprint it queried.
    placed.Offset = offset;
    placed.Footprint.Width = extent.width;
    placed.Footprint.Height = extent.height;
    placed.Footprint.Depth = extent.depth;
    let row_bytes = block_row_bytes(texture.descriptor().format, extent.width)?;
    if row_bytes > u64::from(placed.Footprint.RowPitch) {
        return Err(Dx12Failure::Native(
            ffi::NativeError::driver_contract_violation(
                "GetCopyableFootprints returned a row pitch smaller than the copied texel row",
                "Dx12Device::submit",
            ),
        ));
    }
    Ok((placed, rows, row_bytes))
}

fn texture_location(resource: &ID3D12Resource, subresource: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: subresource,
        },
    }
}

fn footprint_location(
    resource: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    }
}

/// Releases the cloned COM reference held in a temporary copy location.
unsafe fn drop_copy_location(location: &mut D3D12_TEXTURE_COPY_LOCATION) {
    // SAFETY: every location above initializes `pResource` from one clone, and
    // this runs exactly once after `CopyTextureRegion` has consumed the struct.
    unsafe { core::mem::ManuallyDrop::drop(&mut location.pResource) };
}

fn lower_texture_upload(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    job: &UploadJob,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let UploadDescriptor::Texture(descriptor) = job.descriptor() else {
        unreachable!()
    };
    let destination = dx12_texture(&descriptor.dst)?;
    let images = match descriptor.dst.descriptor().dimension {
        crate::api::resource::texture::TextureDimension::D3 => descriptor.extent.depth,
        _ => descriptor.subresource.layer_count,
    };
    let first_subresource = texture_subresource(&descriptor.dst, descriptor.subresource, 0)?;
    let (first_footprint, _, row_bytes) = region_footprint(
        device,
        &descriptor.dst,
        first_subresource,
        descriptor.extent,
        0,
    )?;
    let row_pitch = u64::from(first_footprint.Footprint.RowPitch);
    let block_rows = block_rows(descriptor.dst.descriptor().format, descriptor.extent.height);
    let slice_bytes = row_pitch
        .checked_mul(u64::from(block_rows))
        .ok_or_else(|| {
            Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                "texture upload footprint overflowed",
                "Dx12Device::submit",
            ))
        })?;
    let image_stride = align_up(slice_bytes, 512)?;
    let staging_size = if descriptor.dst.descriptor().dimension
        == crate::api::resource::texture::TextureDimension::D3
    {
        slice_bytes.checked_mul(u64::from(images))
    } else {
        image_stride.checked_mul(u64::from(images))
    }
    .ok_or_else(|| {
        Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
            "texture upload staging size overflowed",
            "Dx12Device::submit",
        ))
    })?;
    let staging =
        create_staging(device, staging_size, StagingHeap::Upload).map_err(Dx12Failure::Native)?;
    write_texture_upload(
        &staging, descriptor, row_pitch, row_bytes, block_rows, images,
    )?;

    let mut entering = Transitions::default();
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);
    for image in 0..images {
        let offset = if descriptor.dst.descriptor().dimension
            == crate::api::resource::texture::TextureDimension::D3
        {
            0
        } else {
            image_stride * u64::from(image)
        };
        let mut extent = descriptor.extent;
        if descriptor.dst.descriptor().dimension
            != crate::api::resource::texture::TextureDimension::D3
        {
            extent.depth = 1;
        }
        let subresource = texture_subresource(&descriptor.dst, descriptor.subresource, image)?;
        let (footprint, _, _) =
            region_footprint(device, &descriptor.dst, subresource, extent, offset)?;
        let mut source = footprint_location(staging.resource(), footprint);
        let mut target = texture_location(destination.resource(), subresource);
        // SAFETY: the staging footprint was populated with the required pitch,
        // the native texture is in COPY_DEST, and portable validation checked the region.
        unsafe {
            list.CopyTextureRegion(
                &target,
                descriptor.origin.x,
                descriptor.origin.y,
                descriptor.origin.z,
                &source,
                None,
            );
            drop_copy_location(&mut source);
            drop_copy_location(&mut target);
        }
    }
    let mut leaving = Transitions::default();
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    committed.staging.push(staging);
    Ok(())
}

fn write_texture_upload(
    staging: &Dx12Buffer,
    descriptor: &crate::api::resource::transfer::TextureUploadDescriptor,
    destination_row_pitch: u64,
    row_bytes: u64,
    block_rows: u32,
    images: u32,
) -> Result<(), Dx12Failure> {
    let mut pointer: *mut core::ffi::c_void = std::ptr::null_mut();
    unsafe {
        staging
            .resource()
            .Map(0, None, Some(&mut pointer))
            .map_err(|error| ref_native(&error))?;
    }
    let result = (|| {
        let destination = std::ptr::NonNull::new(pointer.cast::<u8>()).ok_or_else(|| {
            Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                "Map reported success without producing a pointer",
                "Dx12Device::submit",
            ))
        })?;
        let source_row_pitch = u64::from(descriptor.source_layout.bytes_per_row);
        let source_image_pitch =
            source_row_pitch * u64::from(descriptor.source_layout.rows_per_image);
        for image in 0..images {
            let source_image = u64::from(image) * source_image_pitch;
            let destination_image = if descriptor.dst.descriptor().dimension
                == crate::api::resource::texture::TextureDimension::D3
            {
                u64::from(image) * destination_row_pitch * u64::from(block_rows)
            } else {
                u64::from(image) * align_up(destination_row_pitch * u64::from(block_rows), 512)?
            };
            for row in 0..block_rows {
                let source_offset = source_image + u64::from(row) * source_row_pitch;
                let destination_offset = destination_image + u64::from(row) * destination_row_pitch;
                let source_end = source_offset.checked_add(row_bytes).ok_or_else(|| {
                    Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                        "texture upload source offset overflowed",
                        "Dx12Device::submit",
                    ))
                })?;
                if source_end > descriptor.bytes.len() as u64 {
                    return Err(Dx12Failure::Native(
                        ffi::NativeError::driver_contract_violation(
                            "validated texture upload bytes did not cover its declared region",
                            "Dx12Device::submit",
                        ),
                    ));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        descriptor.bytes.as_ptr().add(source_offset as usize),
                        destination.as_ptr().add(destination_offset as usize),
                        row_bytes as usize,
                    );
                }
            }
        }
        Ok(())
    })();
    unsafe {
        staging.resource().Unmap(0, None);
    }
    result
}

fn lower_texture_readback(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    ticket: &ReadbackTicket,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let ReadbackRequest::Texture {
        src,
        subresource,
        origin,
        extent,
        ..
    } = ticket.request()
    else {
        unreachable!()
    };
    let source = dx12_texture(src)?;
    let images = match src.descriptor().dimension {
        crate::api::resource::texture::TextureDimension::D3 => extent.depth,
        _ => subresource.layer_count,
    };
    let first = texture_subresource(src, *subresource, 0)?;
    let (footprint, _, _) = region_footprint(device, src, first, *extent, 0)?;
    let row_pitch = u64::from(footprint.Footprint.RowPitch);
    let block_rows = block_rows(src.descriptor().format, extent.height);
    let rows_per_image =
        if src.descriptor().dimension == crate::api::resource::texture::TextureDimension::D3 {
            block_rows
        } else {
            (align_up(row_pitch * u64::from(block_rows), 512)? / row_pitch) as u32
        };
    let total_size = row_pitch
        .checked_mul(u64::from(rows_per_image))
        .and_then(|size| size.checked_mul(u64::from(images)))
        .ok_or_else(|| {
            Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                "texture readback footprint overflowed",
                "Dx12Device::submit",
            ))
        })?;
    let staging =
        create_staging(device, total_size, StagingHeap::Readback).map_err(Dx12Failure::Native)?;
    let mut entering = Transitions::default();
    entering.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    entering.record(list);
    for image in 0..images {
        let offset =
            if src.descriptor().dimension == crate::api::resource::texture::TextureDimension::D3 {
                0
            } else {
                row_pitch * u64::from(rows_per_image) * u64::from(image)
            };
        let mut copy_extent = *extent;
        if src.descriptor().dimension != crate::api::resource::texture::TextureDimension::D3 {
            copy_extent.depth = 1;
        }
        let native_subresource = texture_subresource(src, *subresource, image)?;
        let (footprint, _, _) =
            region_footprint(device, src, native_subresource, copy_extent, offset)?;
        let mut target = footprint_location(staging.resource(), footprint);
        let mut source_location = texture_location(source.resource(), native_subresource);
        let box_ = D3D12_BOX {
            left: origin.x,
            top: origin.y,
            front: origin.z,
            right: origin.x + copy_extent.width,
            bottom: origin.y + copy_extent.height,
            back: origin.z + copy_extent.depth,
        };
        unsafe {
            list.CopyTextureRegion(&target, 0, 0, 0, &source_location, Some(&box_));
            drop_copy_location(&mut target);
            drop_copy_location(&mut source_location);
        }
    }
    let mut leaving = Transitions::default();
    leaving.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    committed.readbacks.push(ReadbackRetention {
        staging,
        ticket: ticket.clone(),
        size: total_size,
        layout: Some(ReadbackTexelLayout {
            bytes_per_row: row_pitch as u32,
            rows_per_image,
            total_size,
        }),
    });
    Ok(())
}

/// Lowers an exact texture-to-texture region copy.  Resolves and filtered blits
/// deliberately remain separate unsupported commands; neither is a raw copy.
pub(super) fn lower_texture_copy(
    list: &ID3D12GraphicsCommandList,
    copy: &TextureCopy,
) -> Result<(), Dx12Failure> {
    let source = dx12_texture(&copy.src)?;
    let destination = dx12_texture(&copy.dst)?;
    let images = match copy.src.descriptor().dimension {
        crate::api::resource::texture::TextureDimension::D3 => copy.extent.depth,
        _ => copy.src_subresource.layer_count,
    };
    let mut entering = Transitions::default();
    entering.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    entering.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    entering.record(list);
    for image in 0..images {
        let src_subresource = texture_subresource(&copy.src, copy.src_subresource, image)?;
        let dst_subresource = texture_subresource(&copy.dst, copy.dst_subresource, image)?;
        let mut source_location = texture_location(source.resource(), src_subresource);
        let mut target = texture_location(destination.resource(), dst_subresource);
        let mut extent = copy.extent;
        if copy.src.descriptor().dimension != crate::api::resource::texture::TextureDimension::D3 {
            extent.depth = 1;
        }
        let box_ = D3D12_BOX {
            left: copy.src_origin.x,
            top: copy.src_origin.y,
            front: copy.src_origin.z,
            right: copy.src_origin.x + extent.width,
            bottom: copy.src_origin.y + extent.height,
            back: copy.src_origin.z + extent.depth,
        };
        unsafe {
            list.CopyTextureRegion(
                &target,
                copy.dst_origin.x,
                copy.dst_origin.y,
                copy.dst_origin.z,
                &source_location,
                Some(&box_),
            );
            drop_copy_location(&mut source_location);
            drop_copy_location(&mut target);
        }
    }
    let mut leaving = Transitions::default();
    leaving.push(
        source.resource(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.push(
        destination.resource(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    Ok(())
}

/// Lowers the public buffer-to-texture and texture-to-buffer commands without
/// repacking: unlike upload/readback, their buffer layout is already the native
/// GPU copy layout validated at record time.
pub(super) fn lower_buffer_texture_copy(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    copy: &BufferTextureCopy,
    to_texture: bool,
) -> Result<(), Dx12Failure> {
    let buffer = dx12_buffer(&copy.buffer)?;
    let texture = dx12_texture(&copy.texture)?;
    let is_3d =
        copy.texture.descriptor().dimension == crate::api::resource::texture::TextureDimension::D3;
    let logical_block_rows = block_rows(copy.texture.descriptor().format, copy.extent.height);
    // A D3D12 placed footprint exposes RowPitch but no independent slice-pitch
    // field: its slice stride is the physical row count times RowPitch. Array
    // layers can use separate placed footprints below, but one 3D subresource
    // cannot directly express extra rows between depth slices. Refuse that
    // uncommon layout during transactional Phase A instead of silently ignoring
    // rows_per_image and addressing different bytes than ResourceUse reports.
    // Upload/readback do not have this restriction because they repack through
    // private staging; a future GPU-copy staging path can close this case without
    // changing the public API.
    if is_3d && copy.rows_per_image != logical_block_rows {
        return Err(Dx12Failure::Unsupported {
            what: "a padded 3D buffer-texture image pitch",
            why: "D3D12 placed footprints have no independent slice pitch; the current direct-copy path requires rows_per_image to equal the copied block-row count",
        });
    }
    let images = if is_3d {
        1
    } else {
        copy.texture_subresource.layer_count
    };
    let image_stride = u64::from(copy.bytes_per_row)
        .checked_mul(u64::from(copy.rows_per_image))
        .ok_or_else(|| {
            Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                "buffer-texture copy image stride overflowed",
                "Dx12Device::submit",
            ))
        })?;
    let minimum_row_bytes = block_row_bytes(copy.texture.descriptor().format, copy.extent.width)?;
    if u64::from(copy.bytes_per_row) < minimum_row_bytes {
        return Err(Dx12Failure::Native(
            ffi::NativeError::driver_contract_violation(
                "portable copy validation accepted a row pitch smaller than its texel row",
                "Dx12Device::submit",
            ),
        ));
    }
    let mut entering = Transitions::default();
    entering.push(
        buffer.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        if to_texture {
            D3D12_RESOURCE_STATE_COPY_SOURCE
        } else {
            D3D12_RESOURCE_STATE_COPY_DEST
        },
    );
    entering.push(
        texture.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        if to_texture {
            D3D12_RESOURCE_STATE_COPY_DEST
        } else {
            D3D12_RESOURCE_STATE_COPY_SOURCE
        },
    );
    entering.record(list);

    for image in 0..images {
        let offset = copy
            .buffer_offset
            .checked_add(image_stride * u64::from(image))
            .ok_or_else(|| {
                Dx12Failure::Native(ffi::NativeError::driver_contract_violation(
                    "buffer-texture copy offset overflowed",
                    "Dx12Device::submit",
                ))
            })?;
        // D3D12 requires every placed footprint offset to be 512-byte aligned.
        // The public route records the same requirement for the first image; a
        // multi-layer layout whose caller supplied image stride cannot represent
        // later unaligned images without an unadvertised intermediate copy.
        if !offset.is_multiple_of(512) {
            return Err(Dx12Failure::Unsupported {
                what: "a multi-layer buffer-texture copy with an unaligned image stride",
                why: "each D3D12 placed footprint must start at a 512-byte offset; this backend does not insert a hidden staging copy",
            });
        }
        let subresource = texture_subresource(&copy.texture, copy.texture_subresource, image)?;
        let mut extent = copy.extent;
        if !is_3d {
            extent.depth = 1;
        }
        let (mut footprint, _, _) =
            region_footprint(device, &copy.texture, subresource, extent, offset)?;
        footprint.Footprint.RowPitch = copy.bytes_per_row;
        let mut buffer_location = footprint_location(buffer.resource(), footprint);
        let mut texture_location = texture_location(texture.resource(), subresource);
        let box_ = D3D12_BOX {
            left: copy.texture_origin.x,
            top: copy.texture_origin.y,
            front: copy.texture_origin.z,
            right: copy.texture_origin.x + extent.width,
            bottom: copy.texture_origin.y + extent.height,
            back: copy.texture_origin.z + extent.depth,
        };
        unsafe {
            if to_texture {
                list.CopyTextureRegion(
                    &texture_location,
                    copy.texture_origin.x,
                    copy.texture_origin.y,
                    copy.texture_origin.z,
                    &buffer_location,
                    None,
                );
            } else {
                list.CopyTextureRegion(&buffer_location, 0, 0, 0, &texture_location, Some(&box_));
            }
            drop_copy_location(&mut buffer_location);
            drop_copy_location(&mut texture_location);
        }
    }
    let mut leaving = Transitions::default();
    leaving.push(
        buffer.resource(),
        if to_texture {
            D3D12_RESOURCE_STATE_COPY_SOURCE
        } else {
            D3D12_RESOURCE_STATE_COPY_DEST
        },
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.push(
        texture.resource(),
        if to_texture {
            D3D12_RESOURCE_STATE_COPY_DEST
        } else {
            D3D12_RESOURCE_STATE_COPY_SOURCE
        },
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{block_row_bytes, block_rows};
    use crate::api::format::TextureFormat;

    #[test]
    fn compressed_footprints_count_blocks_not_texels() {
        assert_eq!(block_rows(TextureFormat::Bc1RgbaUnorm, 5), 2);
        assert_eq!(
            block_row_bytes(TextureFormat::Bc1RgbaUnorm, 5).ok(),
            Some(16)
        );
        assert_eq!(
            block_row_bytes(TextureFormat::Bc7RgbaUnorm, 4).ok(),
            Some(16)
        );
    }
}
