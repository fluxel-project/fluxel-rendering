//! The native descriptor packet behind one portable bind group.
//!
//! A group is a *range* of the device's one shader-visible heap, filled with view
//! descriptors. Creating one therefore has exactly three jobs: claim a run long
//! enough for the layout, write each entry's view into the slot its layout named,
//! and keep alive everything an address in that run points at.
//!
//! # Why the group owns the resources
//!
//! A native descriptor holds a *pointer* — a GPU virtual address, or a heap
//! address — and Direct3D 12 has no notion of a descriptor referring to a
//! resource. Section 22.2 makes the bind group the owner of everything it binds,
//! and on this backend that is literal: drop the last portable [`Buffer`] handle
//! while a group still points into it and the driver reads freed memory, which
//! surfaces as corruption and not as an error. So [`Dx12BindGroup`] holds a
//! [`Buffer`] for every entry it wrote an address for, and the descriptors stay
//! meaningful exactly as long as the group does.
//!
//! That is also why the *range* is released in `Drop` rather than by the device:
//! the slots and the reasons for holding them have the same lifetime.
//!
//! # The byte-address rule, and the constant-buffer one
//!
//! A raw buffer view addresses DWORDs, so a storage range becomes
//! `FirstElement = offset / 4` and `NumElements = len / 4`, and an offset that is
//! not a multiple of four cannot be expressed at all. A constant-buffer view's
//! `SizeInBytes` must be a multiple of 256, so a shorter range is *widened* to the
//! next multiple. Both are narrowings the portable layer does not state — section
//! 18.1 lets a caller bind any range the device's alignment limits permit — and
//! neither is rounded silently: a range that cannot be expressed is refused
//! rather than moved, because moving it would change which bytes the shader reads.
//!
//! Widening a constant buffer is the one asymmetry, and it is safe in the
//! direction that matters: the view is what the shader *may* read, the shader's
//! declared need is inside the caller's range, so a longer view still contains
//! everything the contract promised. Narrowing would not.

use std::any::Any;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUFFER_SRV, D3D12_BUFFER_SRV_FLAG_RAW, D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_RAW,
    D3D12_CONSTANT_BUFFER_VIEW_DESC, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_GPU_DESCRIPTOR_HANDLE,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0, D3D12_SRV_DIMENSION_BUFFER,
    D3D12_UAV_DIMENSION_BUFFER, D3D12_UNORDERED_ACCESS_VIEW_DESC,
    D3D12_UNORDERED_ACCESS_VIEW_DESC_0, ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;

use crate::api::binding::backend::BindGroupBackend;
use crate::api::binding::{BindGroupDescriptor, BindingKind, BindingResource};
use crate::api::resource::buffer::{Buffer, BufferBinding};
use crate::backend::dx12::failure::Dx12Failure;
use crate::backend::dx12::resource::Dx12Buffer;

use super::heap::DescriptorHeap;
use super::layout::{RangePlan, TablePlan};
use super::vocabulary::RegisterClass;

/// The constant-buffer view's size granularity.
///
/// `D3D12_CONSTANT_BUFFER_VIEW_DESC::SizeInBytes` must be a multiple of 256, which
/// is a Direct3D 12 rule about the *view* and not about the caller's range: a
/// layout's `min_size` and a [`BufferRange`](crate::api::resource::buffer::BufferRange)
/// are held only to the shader's own need and the device's binding limits.
const CONSTANT_BUFFER_ALIGNMENT: u64 = 256;

/// The byte alignment a raw buffer view addresses.
const RAW_VIEW_ALIGNMENT: u64 = 4;

/// One portable bind group's descriptors, living in a range of the view heap.
pub(crate) struct Dx12BindGroup {
    /// The run this group owns, released in `Drop`.
    start: u32,
    /// How many descriptors the run is, which the release needs and nothing else
    /// does.
    count: u32,
    /// The heap the run is in.
    ///
    /// An `Arc` rather than a borrow, because the group outlives the call that
    /// made it and must be able to give its slots back on its own.
    heap: Arc<DescriptorHeap>,
    /// Every buffer an address in the run points at.
    ///
    /// Textures and samplers will join this list when they are lowered; today a
    /// buffer is the only thing a group can bind, and a field holding an enum
    /// with one inhabitant would be an abstraction with no second case to justify
    /// it. Held for the reason the module doc gives, and never read — which is
    /// what an ownership field is.
    buffers: Vec<Buffer>,
}

impl Dx12BindGroup {
    /// The GPU address of this group's view table, for the dispatch lowering.
    ///
    /// The address of the run's *first* descriptor, which is what
    /// `SetComputeRootDescriptorTable` takes: the root signature holds the offsets
    /// *within* the table, so the table is named from its start.
    pub(crate) fn view_table(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.heap.gpu(self.start)
    }
}

impl BindGroupBackend for Dx12BindGroup {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for Dx12BindGroup {
    fn drop(&mut self) {
        self.heap.release(self.start, self.count);
    }
}

/// Writes one group's descriptors into the heap.
///
/// # Errors
///
/// [`Dx12Failure::Unsupported`] when the layout's plan cannot be built (an
/// unwritten lowering), when the heap has no run long enough (this backend's
/// fixed capacity, not the request's fault), and when an entry's range cannot be
/// expressed as a Direct3D 12 view (a storage range that is not DWORD-aligned, or
/// a uniform range whose padded view would run past its buffer). Every one of
/// those is `Unsupported` rather than `InvalidUsage`: nothing about the portable
/// request is illegal in any of them.
pub(crate) fn create_bind_group(
    device: &ID3D12Device,
    heap: &Arc<DescriptorHeap>,
    descriptor: &BindGroupDescriptor,
) -> Result<Dx12BindGroup, Dx12Failure> {
    let plan = TablePlan::of(descriptor.layout.descriptor())?;
    let count = plan.view_descriptors();
    let start = if count == 0 {
        // Empty layouts are meaningful placeholders in a PipelineInterface and
        // an empty group for one owns no native descriptor range.
        0
    } else {
        heap.allocate(count).ok_or(Dx12Failure::Unsupported {
            what: "a bind group larger than this device's free descriptor slots",
            why: "this backend's descriptor heap is a fixed 65536 descriptors, and every \
                  live bind group holds its range until the last handle to it is dropped",
        })?
    };

    let mut buffers = Vec::with_capacity(descriptor.entries.len());
    // The run is released on a failure below rather than leaked: the
    // `Dx12BindGroup` that would have owned it is never built, so nothing else
    // can give it back.
    if let Err(failure) = write_entries(device, heap, start, &plan, descriptor, &mut buffers) {
        heap.release(start, count);
        return Err(failure);
    }

    Ok(Dx12BindGroup {
        start,
        count,
        heap: Arc::clone(heap),
        buffers,
    })
}

/// Writes every entry's view, in the order the plan laid the slots out.
fn write_entries(
    device: &ID3D12Device,
    heap: &DescriptorHeap,
    start: u32,
    plan: &TablePlan,
    descriptor: &BindGroupDescriptor,
    buffers: &mut Vec<Buffer>,
) -> Result<(), Dx12Failure> {
    for entry in &descriptor.entries {
        // A lookup rather than an index by position: the plan's order is the
        // layout's slot order and the entries' order is not guaranteed to be the
        // same even though canonicalization usually makes them agree, so
        // positionally indexing would be right by accident.
        //
        // The `None` arm is unreachable — section 22.3 refuses a group whose
        // entries do not fill its layout exactly, and the plan was built from
        // that same layout — but it returns an error rather than unwrapping
        // because the alternative to a checked lookup is a panic in a library.
        let Some(range) = plan.range_for(entry.slot) else {
            return Err(Dx12Failure::Unsupported {
                what: "a bind group entry whose layout does not declare its slot",
                why: "the portable layer refuses that shape before this backend is \
                      reached, so this is a total function over an empty case",
            });
        };
        match &entry.resource {
            BindingResource::Buffer(binding) => {
                write_element(device, heap, start + range.first, range, binding)?;
                buffers.push(binding.buffer.clone());
            }
            BindingResource::BufferArray(bindings) => {
                if bindings.len() as u32 != range.count {
                    return Err(Dx12Failure::Unsupported {
                        what: "a bind group array whose length its layout does not declare",
                        why: "the portable layer refuses that shape before this backend \
                              is reached, so this is a total function over an empty case",
                    });
                }
                for (element, binding) in bindings.iter().enumerate() {
                    write_element(
                        device,
                        heap,
                        start + range.first + element as u32,
                        range,
                        binding,
                    )?;
                    buffers.push(binding.buffer.clone());
                }
            }
            // The view types below have no lowering in this backend at all, and
            // the refusal names which one arrived rather than reporting them as
            // one gap. A texture reaching here is a *portable* gap as well —
            // `Device::create_texture` stops with `unimplemented!()` — while a
            // sampler reaching here is only this backend's, which is why the two
            // messages differ.
            BindingResource::Texture(_) | BindingResource::TextureArray(_) => {
                return Err(Dx12Failure::Unsupported {
                    what: "a texture bound to a group",
                    why: "this backend has no texture resource, no texture view and no \
                          format mapping, and no caller can reach them anyway: \
                          Device::create_texture stops with unimplemented!()",
                });
            }
            BindingResource::Sampler(_) | BindingResource::SamplerArray(_) => {
                return Err(Dx12Failure::Unsupported {
                    what: "a sampler bound to a group",
                    why: "the sampler heap and the per-entry sampler write are not \
                          written, and no caller can reach them anyway: \
                          Device::create_sampler stops with unimplemented!(), so no \
                          portable Sampler value exists to bind",
                });
            }
        }
    }
    Ok(())
}

/// Writes one descriptor: one buffer binding, into one slot.
fn write_element(
    device: &ID3D12Device,
    heap: &DescriptorHeap,
    slot: u32,
    range: &RangePlan,
    binding: &BufferBinding,
) -> Result<(), Dx12Failure> {
    let Some(native) = binding
        .buffer
        .native()
        .as_any()
        .downcast_ref::<Dx12Buffer>()
    else {
        return Err(Dx12Failure::Unsupported {
            what: "a buffer this device did not allocate",
            why: "its native allocation belongs to another backend, and section 3.3 \
                  makes that a refusal rather than a migration",
        });
    };
    let size = native.size();
    let destination = heap.cpu(slot);

    // The view is chosen from the *layout's* kind rather than from the resource
    // variant, because the kind is what the shader will read through and a
    // mismatch has to be a refusal: writing a buffer SRV into a slot the layout
    // declared as a sampled texture would leave the driver reading a raw view
    // through a typed binding, which Direct3D 12 does not check at dispatch.
    //
    // `destination` is inside this group's own allocated run and the resource is
    // retained by the group. The leaf writers contain the narrowly scoped native
    // calls that rely on those invariants.
    match &range.kind {
        BindingKind::UniformBuffer { .. } => {
            write_constant_buffer(device, destination, native, binding, size)
        }
        BindingKind::StorageBuffer { .. } => match range.class {
            RegisterClass::ShaderResource => write_raw_view(
                device,
                destination,
                native,
                binding,
                size,
                RawView::ShaderResource,
            ),
            RegisterClass::UnorderedAccess => write_raw_view(
                device,
                destination,
                native,
                binding,
                size,
                RawView::UnorderedAccess,
            ),
            RegisterClass::ConstantBuffer | RegisterClass::Sampler => {
                Err(Dx12Failure::Unsupported {
                    what: "a storage buffer in a register class its access does not select",
                    why: "the class is derived from the kind by one function, so a \
                              disagreement here means that function and this match have \
                              drifted apart",
                })
            }
        },
        BindingKind::SampledTexture { .. }
        | BindingKind::StorageTexture { .. }
        | BindingKind::Sampler { .. } => Err(Dx12Failure::Unsupported {
            what: "a buffer bound to a slot whose layout declares a texture or a sampler",
            why: "the portable layer refuses that shape before this backend is \
                      reached, so this is a total function over an empty case",
        }),
    }
}

/// Writes a constant-buffer view, widening the range to the view's granularity.
fn write_constant_buffer(
    device: &ID3D12Device,
    destination: D3D12_CPU_DESCRIPTOR_HANDLE,
    native: &Dx12Buffer,
    binding: &BufferBinding,
    size: u64,
) -> Result<(), Dx12Failure> {
    let offset = binding.range.offset;
    let padded = binding.range.size.div_ceil(CONSTANT_BUFFER_ALIGNMENT) * CONSTANT_BUFFER_ALIGNMENT;
    // Both bounds are Direct3D 12's and neither is the caller's: the view must
    // stay inside the allocation, and `SizeInBytes` is a `u32` on the native side.
    if offset + padded > size || padded > u32::MAX as u64 {
        return Err(Dx12Failure::Unsupported {
            what: "a uniform buffer range with no constant-buffer view it fits in",
            why: "Direct3D 12 requires a constant-buffer view's SizeInBytes to be a \
                  multiple of 256, so a range shorter than that is widened — and this \
                  one cannot be widened without running past the end of its buffer",
        });
    }
    let description = D3D12_CONSTANT_BUFFER_VIEW_DESC {
        // SAFETY: `GetGPUVirtualAddress` takes no argument and returns the
        // resource's base address. The offset added is inside the allocation
        // because of the bound checked above.
        BufferLocation: unsafe { native.resource().GetGPUVirtualAddress() } + offset,
        SizeInBytes: padded as u32,
    };
    // SAFETY: the description is a local that outlives the call and the
    // destination is a writable descriptor in this backend's own heap.
    unsafe { device.CreateConstantBufferView(Some(&description), destination) };
    Ok(())
}

/// Which raw buffer view to write.
///
/// Two variants rather than a class, because the two native calls take different
/// descriptor types and different flag constants, and the SRV half additionally
/// carries a component mapping the UAV half has no field for.
#[derive(Clone, Copy)]
enum RawView {
    /// A read-only view: `t` in the shader, `ByteAddressBuffer` in HLSL.
    ShaderResource,
    /// A writable view: `u` in the shader, `RWByteAddressBuffer` in HLSL.
    UnorderedAccess,
}

/// Writes a raw buffer view over `binding`'s range.
///
/// The DWORD arithmetic and its two refusals are shared by both halves, which is
/// why they are computed once here rather than in each arm below.
fn write_raw_view(
    device: &ID3D12Device,
    destination: D3D12_CPU_DESCRIPTOR_HANDLE,
    native: &Dx12Buffer,
    binding: &BufferBinding,
    size: u64,
    view: RawView,
) -> Result<(), Dx12Failure> {
    let offset = binding.range.offset;
    if offset % RAW_VIEW_ALIGNMENT != 0 {
        return Err(Dx12Failure::Unsupported {
            what: "a storage buffer range that is not four-byte aligned",
            why: "a raw buffer view addresses DWORDs, so FirstElement is a count of \
                  them and an offset that is not a multiple of four has no view",
        });
    }
    // The requested length rounds *up* to whole DWORDs, then clamps to what the
    // allocation has left. Both halves are needed: rounding up alone could name
    // bytes past the end, and clamping alone could hand the shader fewer bytes
    // than the caller's range promised.
    let wanted = binding.range.size.div_ceil(RAW_VIEW_ALIGNMENT);
    let available = (size - offset) / RAW_VIEW_ALIGNMENT;
    let elements = wanted.min(available);
    if elements < wanted || elements > u32::MAX as u64 {
        return Err(Dx12Failure::Unsupported {
            what: "a storage buffer range with no raw view that covers it",
            why: "a raw buffer view is a count of DWORDs, so a range whose length is \
                  not a whole number of DWORDs cannot be covered by one — and this \
                  one cannot be rounded up without running past the end of its \
                  buffer",
        });
    }
    let first = offset / RAW_VIEW_ALIGNMENT;
    let elements = elements as u32;
    let resource: &ID3D12Resource = native.resource();

    match view {
        RawView::ShaderResource => {
            let description = D3D12_SHADER_RESOURCE_VIEW_DESC {
                // `R32_TYPELESS` and the raw flag are one decision: Direct3D 12
                // requires exactly this format for a raw view, and a typed format
                // would make the view an ordinary typed buffer read.
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                // Direct3D 12's own identity mapping, which is what an
                // untyped-by-design view wants.
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: first,
                        NumElements: elements,
                        // Zero means "raw", not "stride zero": a nonzero stride is
                        // the structured form, which this ABI cannot describe.
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            // SAFETY: the description is a local that outlives the call, the
            // resource is the one the descriptor names, and the destination is a
            // writable descriptor in this backend's own heap. The bounds were
            // checked above.
            unsafe {
                device.CreateShaderResourceView(resource, Some(&description), destination);
            }
        }
        RawView::UnorderedAccess => {
            let description = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: first,
                        NumElements: elements,
                        StructureByteStride: 0,
                        // No counter, because a raw view is not an append/consume
                        // buffer: v13 has no counter resource to bind.
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    },
                },
            };
            // SAFETY: as above, and the null counter resource is the documented
            // "this view has no counter".
            unsafe {
                device.CreateUnorderedAccessView(
                    resource,
                    None::<&ID3D12Resource>,
                    Some(&description),
                    destination,
                );
            }
        }
    }
    Ok(())
}
