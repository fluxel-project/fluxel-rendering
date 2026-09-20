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
//! Widening a constant buffer is the one asymmetry.  `Dx12Buffer` privately
//! pads a uniform-capable native allocation, so the widened tail is backed
//! without changing the portable buffer's logical size.  Raw views receive no
//! equivalent widening: their DWORD tail must be exact, because that descriptor
//! would otherwise expose logical bytes outside the caller's `BufferRange`.

use std::any::Any;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUFFER_SRV, D3D12_BUFFER_SRV_FLAG_RAW, D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_RAW,
    D3D12_CONSTANT_BUFFER_VIEW_DESC, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_GPU_DESCRIPTOR_HANDLE,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0, D3D12_SRV_DIMENSION_BUFFER,
    D3D12_TEX1D_UAV, D3D12_TEX2D_ARRAY_UAV, D3D12_TEX2D_UAV, D3D12_TEX3D_UAV,
    D3D12_UAV_DIMENSION_BUFFER, D3D12_UAV_DIMENSION_TEXTURE1D, D3D12_UAV_DIMENSION_TEXTURE2D,
    D3D12_UAV_DIMENSION_TEXTURE2DARRAY, D3D12_UAV_DIMENSION_TEXTURE3D,
    D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_UNORDERED_ACCESS_VIEW_DESC_0, ID3D12DescriptorHeap,
    ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;

use crate::api::binding::backend::BindGroupBackend;
use crate::api::binding::{BindGroupDescriptor, BindingKind, BindingResource};
use crate::api::resource::buffer::{Buffer, BufferBinding};
use crate::api::resource::{Sampler, TextureView};
use crate::backend::dx12::failure::Dx12Failure;
use crate::backend::dx12::resource::{Dx12Buffer, Dx12Sampler, Dx12Texture, Dx12TextureView};

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
    sampler_start: u32,
    sampler_count: u32,
    sampler_heap: Arc<DescriptorHeap>,
    /// Every buffer an address in the run points at.
    ///
    /// Textures and samplers will join this list when they are lowered; today a
    /// buffer is the only thing a group can bind, and a field holding an enum
    /// with one inhabitant would be an abstraction with no second case to justify
    /// it. Held for the reason the module doc gives, and never read — which is
    /// what an ownership field is.
    _buffers: Vec<Buffer>,
    _textures: Vec<TextureView>,
    _samplers: Vec<Sampler>,
}

/// An uncommitted descriptor run.
///
/// A group needs one range from each of D3D12's two shader-visible heap types.
/// Those claims are one logical operation: publishing only one would strand
/// descriptors whenever the second allocation or a subsequent descriptor write
/// fails.  Keeping the rollback in this small RAII value makes every early
/// return transactional, including future descriptor kinds added below.
struct DescriptorReservation {
    start: u32,
    count: u32,
    heap: Arc<DescriptorHeap>,
    committed: bool,
}

impl DescriptorReservation {
    fn claim(
        heap: &Arc<DescriptorHeap>,
        count: u32,
        what: Dx12Failure,
    ) -> Result<Self, Dx12Failure> {
        let start = if count == 0 {
            0
        } else {
            heap.allocate(count).ok_or(what)?
        };
        Ok(Self {
            start,
            count,
            heap: Arc::clone(heap),
            committed: false,
        })
    }

    fn start(&self) -> u32 {
        self.start
    }

    fn commit(mut self) -> (u32, u32, Arc<DescriptorHeap>) {
        self.committed = true;
        (self.start, self.count, Arc::clone(&self.heap))
    }
}

impl Drop for DescriptorReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.heap.release(self.start, self.count);
        }
    }
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

    /// The shader-visible heap containing this group's view table.
    pub(crate) fn view_heap(&self) -> &ID3D12DescriptorHeap {
        self.heap.handle()
    }

    pub(crate) fn sampler_table(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.sampler_heap.gpu(self.sampler_start)
    }

    pub(crate) fn sampler_heap(&self) -> &ID3D12DescriptorHeap {
        self.sampler_heap.handle()
    }
}

impl BindGroupBackend for Dx12BindGroup {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for Dx12BindGroup {
    fn drop(&mut self) {
        // TODO(perf): Immediate release is correct while these immutable tables
        // are owned for their full bind-group lifetime. If a descriptor cache
        // later recycles ranges earlier, it must queue retirement behind every
        // submitted list that bound the range. This is DX12-private: BindGroup
        // ownership plus CompletionPoint already form the public lifetime seam.
        self.heap.release(self.start, self.count);
        self.sampler_heap
            .release(self.sampler_start, self.sampler_count);
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
    sampler_heap: &Arc<DescriptorHeap>,
    descriptor: &BindGroupDescriptor,
) -> Result<Dx12BindGroup, Dx12Failure> {
    let plan = TablePlan::of(descriptor.layout.descriptor())?;
    let count = plan.view_descriptors();
    let sampler_count = plan.sampler_descriptors();
    let views = DescriptorReservation::claim(
        heap,
        count,
        Dx12Failure::Unsupported {
            what: "a bind group larger than this device's free descriptor slots",
            why: "this backend's descriptor heap is a fixed 65536 descriptors, and every \
                  live bind group holds its range until the last handle to it is dropped",
        },
    )?;
    let samplers_reservation = DescriptorReservation::claim(
        sampler_heap,
        sampler_count,
        Dx12Failure::Unsupported {
            what: "a bind group's samplers larger than this device's free descriptor slots",
            why: "the DX12 sampler heap is fixed-size and every live bind group retains its range",
        },
    )?;

    let mut buffers = Vec::with_capacity(descriptor.entries.len());
    let mut textures = Vec::with_capacity(descriptor.entries.len());
    let mut samplers = Vec::with_capacity(descriptor.entries.len());
    write_entries(
        device,
        heap,
        views.start(),
        sampler_heap,
        samplers_reservation.start(),
        &plan,
        descriptor,
        &mut buffers,
        &mut textures,
        &mut samplers,
    )?;

    // Committing is deliberately the final fallible-operation boundary.  Until
    // here either reservation's Drop returns its range, so a failed group never
    // consumes descriptor capacity visible to the next group.
    let (start, count, heap) = views.commit();
    let (sampler_start, sampler_count, sampler_heap) = samplers_reservation.commit();

    Ok(Dx12BindGroup {
        start,
        count,
        heap,
        sampler_start,
        sampler_count,
        sampler_heap,
        _buffers: buffers,
        _textures: textures,
        _samplers: samplers,
    })
}

/// Writes every entry's view, in the order the plan laid the slots out.
fn write_entries(
    device: &ID3D12Device,
    heap: &DescriptorHeap,
    start: u32,
    sampler_heap: &DescriptorHeap,
    sampler_start: u32,
    plan: &TablePlan,
    descriptor: &BindGroupDescriptor,
    buffers: &mut Vec<Buffer>,
    textures: &mut Vec<TextureView>,
    samplers: &mut Vec<Sampler>,
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
            BindingResource::Texture(view) => {
                write_texture(device, heap.cpu(start + range.first), range, view)?;
                textures.push(view.clone());
            }
            BindingResource::TextureArray(views) => {
                for (element, view) in views.iter().enumerate() {
                    write_texture(
                        device,
                        heap.cpu(start + range.first + element as u32),
                        range,
                        view,
                    )?;
                    textures.push(view.clone());
                }
            }
            BindingResource::Sampler(sampler) => {
                write_sampler(
                    device,
                    sampler_heap.cpu(sampler_start + range.first),
                    sampler,
                )?;
                samplers.push(sampler.clone());
            }
            BindingResource::SamplerArray(values) => {
                for (element, sampler) in values.iter().enumerate() {
                    write_sampler(
                        device,
                        sampler_heap.cpu(sampler_start + range.first + element as u32),
                        sampler,
                    )?;
                    samplers.push(sampler.clone());
                }
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
            write_constant_buffer(device, destination, native, binding)
        }
        BindingKind::StorageBuffer { .. } => match range.class {
            RegisterClass::ShaderResource => write_raw_view(
                device,
                destination,
                native,
                binding,
                RawView::ShaderResource,
            ),
            RegisterClass::UnorderedAccess => write_raw_view(
                device,
                destination,
                native,
                binding,
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

/// Copies a texture SRV into the group's shader-visible view heap.
///
/// The view owns its CPU-only descriptor, while the group owns the portable
/// view. Copying keeps descriptor storage separate from resource lifetime and
/// is the D3D12-prescribed way to populate a shader-visible heap.
fn write_texture(
    device: &ID3D12Device,
    destination: D3D12_CPU_DESCRIPTOR_HANDLE,
    range: &RangePlan,
    view: &TextureView,
) -> Result<(), Dx12Failure> {
    let native = view
        .native()
        .as_any()
        .downcast_ref::<Dx12TextureView>()
        .ok_or(Dx12Failure::Unsupported {
            what: "a texture view this device did not create",
            why: "its native descriptor belongs to another backend",
        })?;
    match &range.kind {
        BindingKind::SampledTexture { .. }
        | BindingKind::StorageTexture {
            access: crate::api::binding::StorageAccess::ReadOnly,
            ..
        } => unsafe {
            device.CopyDescriptorsSimple(
                1,
                destination,
                native.cpu(),
                windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
            Ok(())
        },
        BindingKind::StorageTexture { .. } => write_texture_uav(device, destination, view),
        _ => Err(Dx12Failure::Unsupported {
            what: "a texture bound to a non-texture slot",
            why: "portable validation normally rejects this shape before backend lowering",
        }),
    }
}

fn write_texture_uav(
    device: &ID3D12Device,
    destination: D3D12_CPU_DESCRIPTOR_HANDLE,
    view: &TextureView,
) -> Result<(), Dx12Failure> {
    let texture = view.texture();
    let native = texture
        .native()
        .as_any()
        .downcast_ref::<Dx12Texture>()
        .ok_or(Dx12Failure::Unsupported {
            what: "a texture this device did not create",
            why: "its native allocation belongs to another backend",
        })?;
    let format = crate::backend::dx12::platform::facts::dxgi_format(view.format()).ok_or(
        Dx12Failure::Unsupported {
            what: "a storage texture format with no DXGI UAV representation",
            why: "the public format is accepted only where the device capability table says it is usable",
        },
    )?;
    use crate::api::resource::view::TextureViewDimension;
    let descriptor = match view.descriptor().dimension {
        TextureViewDimension::D1 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture1D: D3D12_TEX1D_UAV {
                    MipSlice: view.descriptor().base_mip,
                },
            },
        },
        TextureViewDimension::D2 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: view.descriptor().base_mip,
                    PlaneSlice: 0,
                },
            },
        },
        TextureViewDimension::D2Array => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: view.descriptor().base_mip,
                    FirstArraySlice: view.descriptor().base_layer,
                    ArraySize: view.descriptor().layer_count,
                    PlaneSlice: 0,
                },
            },
        },
        TextureViewDimension::D3 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV {
                    MipSlice: view.descriptor().base_mip,
                    FirstWSlice: 0,
                    WSize: u32::MAX,
                },
            },
        },
        TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
            return Err(Dx12Failure::Unsupported {
                what: "a cube storage texture binding",
                why: "D3D12 has no cube UAV descriptor dimension",
            });
        }
    };
    unsafe {
        device.CreateUnorderedAccessView(
            native.resource(),
            None::<&ID3D12Resource>,
            Some(&descriptor),
            destination,
        );
    }
    Ok(())
}

/// Copies a sampler descriptor into the group's shader-visible sampler heap.
fn write_sampler(
    device: &ID3D12Device,
    destination: D3D12_CPU_DESCRIPTOR_HANDLE,
    sampler: &Sampler,
) -> Result<(), Dx12Failure> {
    let native = sampler
        .native()
        .as_any()
        .downcast_ref::<Dx12Sampler>()
        .ok_or(Dx12Failure::Unsupported {
            what: "a sampler this device did not create",
            why: "its native descriptor belongs to another backend",
        })?;
    unsafe {
        device.CopyDescriptorsSimple(
            1,
            destination,
            native.cpu(),
            windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        );
    }
    Ok(())
}

/// Writes a constant-buffer view, widening the range to the view's granularity.
fn write_constant_buffer(
    device: &ID3D12Device,
    destination: D3D12_CPU_DESCRIPTOR_HANDLE,
    native: &Dx12Buffer,
    binding: &BufferBinding,
) -> Result<(), Dx12Failure> {
    let offset = binding.range.offset;
    if offset
        .checked_add(binding.range.size)
        .is_none_or(|end| end > native.size())
    {
        return Err(Dx12Failure::Unsupported {
            what: "a uniform buffer range outside its logical buffer",
            why: "native CBV tail padding is private backing and cannot make an out-of-range portable BufferRange valid",
        });
    }
    let padded = aligned_view_size(binding.range.size, CONSTANT_BUFFER_ALIGNMENT).ok_or(
        Dx12Failure::Unsupported {
            what: "a uniform buffer range whose constant-buffer view size overflows",
            why: "Direct3D 12 requires a 256-byte-multiple view size, and rounding this range would exceed the representable native size",
        },
    )?;
    // Both bounds are Direct3D 12's and neither is the caller's: the view must
    // stay inside the allocation, and `SizeInBytes` is a `u32` on the native side.
    if offset
        .checked_add(padded)
        .is_none_or(|end| end > native.allocation_size())
        || padded > u32::MAX as u64
    {
        return Err(Dx12Failure::Unsupported {
            what: "a uniform buffer range with no constant-buffer view it fits in",
            why: "Direct3D 12 requires a constant-buffer view's SizeInBytes to be a \
                  multiple of 256, so a range shorter than that is widened — and this \
                  one cannot be widened without running past the native allocation",
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
    view: RawView,
) -> Result<(), Dx12Failure> {
    let offset = binding.range.offset;
    if offset
        .checked_add(binding.range.size)
        .is_none_or(|end| end > native.size())
    {
        return Err(Dx12Failure::Unsupported {
            what: "a storage buffer range outside its logical buffer",
            why: "private native allocation padding cannot make an out-of-range portable BufferRange valid",
        });
    }
    if offset % RAW_VIEW_ALIGNMENT != 0 {
        return Err(Dx12Failure::Unsupported {
            what: "a storage buffer range that is not four-byte aligned",
            why: "a raw buffer view addresses DWORDs, so FirstElement is a count of \
                  them and an offset that is not a multiple of four has no view",
        });
    }
    // A raw descriptor's end is a DWORD boundary.  Rounding the requested range
    // up would make shader loads observe bytes outside the portable
    // `BufferRange` whenever the range ends before the resource does.  Native
    // allocation padding is intentionally *not* permission to expose those
    // logical bytes, so this lowering refuses an inexact tail.
    if binding.range.size % RAW_VIEW_ALIGNMENT != 0 {
        return Err(Dx12Failure::Unsupported {
            what: "a storage buffer range whose size is not four-byte aligned",
            why: "a raw buffer view has whole DWORD elements; rounding its tail up would expose bytes outside the portable BufferRange",
        });
    }
    let elements = binding.range.size / RAW_VIEW_ALIGNMENT;
    if elements > u32::MAX as u64 {
        return Err(Dx12Failure::Unsupported {
            what: "a storage buffer range with no raw view that covers it",
            why: "a raw buffer view is a count of DWORDs, so a range whose length is \
                  not a whole number of DWORDs cannot be represented exactly, and \
                  one raw view cannot name more than u32::MAX DWORDs",
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

/// Rounds a native view extent without relying on an overflowing `size + mask`.
fn aligned_view_size(size: u64, alignment: u64) -> Option<u64> {
    let quotient = size / alignment;
    let remainder = size % alignment;
    quotient
        .checked_add(u64::from(remainder != 0))?
        .checked_mul(alignment)
}

#[cfg(test)]
mod tests {
    use super::{RAW_VIEW_ALIGNMENT, aligned_view_size};

    #[test]
    fn constant_buffer_rounding_is_exact_at_boundaries_and_never_wraps() {
        assert_eq!(aligned_view_size(1, 256), Some(256));
        assert_eq!(aligned_view_size(256, 256), Some(256));
        assert_eq!(aligned_view_size(257, 256), Some(512));
        assert_eq!(aligned_view_size(u64::MAX, 256), None);
    }

    #[test]
    fn raw_views_require_an_exact_dword_tail() {
        assert_eq!(16 % RAW_VIEW_ALIGNMENT, 0);
        assert_ne!(17 % RAW_VIEW_ALIGNMENT, 0);
    }
}
