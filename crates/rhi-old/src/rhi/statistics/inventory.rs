//! Live inventory and the descriptor-based logical memory estimate
//! (§47.14 - §47.18).
//!
//! The live inventory is an object lifecycle table, not a counter: an object is
//! present from creation until its backing is reclaimed, and a clone of a handle
//! is invisible to it. That is what makes `buffers == 1` true for one buffer
//! cloned a hundred times.
//!
//! The memory estimate is portable arithmetic over canonical descriptors and
//! format facts. It is never a VRAM, residency, or physical-byte number
//! (§47.17), and it is deliberately named `logical_estimated_bytes`.

use std::collections::{HashMap, HashSet};

use crate::rhi::format::format_facts;
use crate::rhi::platform::ObjectId;
use crate::rhi::presentation::AcquiredFrameId;
use crate::rhi::resource::{BufferDescriptor, TextureDescriptor, TextureUsage};

use super::interval::UsedObject;

/// The kind of logical object a lifecycle event is about.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ObjectKind {
    /// A buffer.
    Buffer,
    /// A texture.
    Texture,
    /// A texture view.
    TextureView,
    /// A sampler.
    Sampler,
    /// A shader module.
    ShaderModule,
    /// A bind group layout.
    BindGroupLayout,
    /// A bind group.
    BindGroup,
    /// A pipeline interface.
    PipelineInterface,
    /// A raster pipeline.
    RasterPipeline,
    /// A compute pipeline.
    ComputePipeline,
}

/// A logical object entering the live inventory.
///
/// The descriptors are borrowed: the inventory keeps the one fact the memory
/// estimate needs, not a copy of the creation descriptor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub(crate) enum CreatedObject<'a> {
    /// A buffer with its creation descriptor.
    Buffer {
        /// The buffer's object id.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: &'a BufferDescriptor,
    },
    /// A texture with its creation descriptor.
    Texture {
        /// The texture's object id.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: &'a TextureDescriptor,
    },
    /// A texture view.
    TextureView(ObjectId),
    /// A sampler.
    Sampler(ObjectId),
    /// A shader module.
    ShaderModule(ObjectId),
    /// A bind group layout.
    BindGroupLayout(ObjectId),
    /// A bind group.
    BindGroup(ObjectId),
    /// A pipeline interface.
    PipelineInterface(ObjectId),
    /// A raster pipeline.
    RasterPipeline(ObjectId),
    /// A compute pipeline.
    ComputePipeline(ObjectId),
}

impl CreatedObject<'_> {
    /// This object's kind.
    pub(crate) fn kind(&self) -> ObjectKind {
        match self {
            Self::Buffer { .. } => ObjectKind::Buffer,
            Self::Texture { .. } => ObjectKind::Texture,
            Self::TextureView(_) => ObjectKind::TextureView,
            Self::Sampler(_) => ObjectKind::Sampler,
            Self::ShaderModule(_) => ObjectKind::ShaderModule,
            Self::BindGroupLayout(_) => ObjectKind::BindGroupLayout,
            Self::BindGroup(_) => ObjectKind::BindGroup,
            Self::PipelineInterface(_) => ObjectKind::PipelineInterface,
            Self::RasterPipeline(_) => ObjectKind::RasterPipeline,
            Self::ComputePipeline(_) => ObjectKind::ComputePipeline,
        }
    }

    /// This object's id.
    pub(crate) fn id(&self) -> ObjectId {
        match self {
            Self::Buffer { id, .. }
            | Self::Texture { id, .. }
            | Self::TextureView(id)
            | Self::Sampler(id)
            | Self::ShaderModule(id)
            | Self::BindGroupLayout(id)
            | Self::BindGroup(id)
            | Self::PipelineInterface(id)
            | Self::RasterPipeline(id)
            | Self::ComputePipeline(id) => *id,
        }
    }
}

/// The working-set key of an object kind, for kinds the working set tracks.
pub(crate) fn used_object(kind: ObjectKind, id: ObjectId) -> Option<UsedObject> {
    match kind {
        ObjectKind::Buffer => Some(UsedObject::Buffer(id)),
        ObjectKind::Texture => Some(UsedObject::Texture(id)),
        ObjectKind::Sampler => Some(UsedObject::Sampler(id)),
        ObjectKind::BindGroup => Some(UsedObject::BindGroup(id)),
        ObjectKind::ShaderModule => Some(UsedObject::ShaderModule(id)),
        ObjectKind::RasterPipeline => Some(UsedObject::RasterPipeline(id)),
        ObjectKind::ComputePipeline => Some(UsedObject::ComputePipeline(id)),
        ObjectKind::TextureView | ObjectKind::BindGroupLayout | ObjectKind::PipelineInterface => None,
    }
}

/// How good a memory estimate is (§47.15).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEstimateQuality {
    /// The estimate follows from canonical descriptors and declared format
    /// facts.
    LogicalEstimate,
    /// No estimate is available, for example because the format has
    /// implementation-defined backing.
    Unknown,
}

/// A logical resource byte estimate (§47.15).
///
/// The value is `Option` because "the format does not define a portable size" is
/// a real answer, and a zero would be a wrong one. The name is frozen: this is
/// never `vram_bytes`, `gpu_memory_bytes`, or `physical_bytes`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct MemoryEstimate {
    /// The descriptor-based logical bytes, absent when no estimate exists.
    pub logical_estimated_bytes: Option<u64>,
    /// How the value was produced.
    pub quality: MemoryEstimateQuality,
}

impl MemoryEstimate {
    /// An absent estimate.
    pub const fn unknown() -> Self {
        Self {
            logical_estimated_bytes: None,
            quality: MemoryEstimateQuality::Unknown,
        }
    }

    /// The logical estimate `bytes`.
    pub const fn logical(bytes: u64) -> Self {
        Self {
            logical_estimated_bytes: Some(bytes),
            quality: MemoryEstimateQuality::LogicalEstimate,
        }
    }
}

impl Default for MemoryEstimate {
    fn default() -> Self {
        Self::unknown()
    }
}

/// The memory estimate of the live inventory, split by class (§47.15).
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ResourceMemoryStatistics {
    /// Live buffers. A non-overlapping top-level classification.
    pub buffers: MemoryEstimate,
    /// Live textures. A non-overlapping top-level classification.
    pub textures: MemoryEstimate,
    /// `buffers` plus `textures`.
    pub total_resources: MemoryEstimate,

    /// The live textures whose usage allows a render target. An overlapping
    /// analytical subset of `textures`; do not add it to `textures` again.
    pub render_target_textures: MemoryEstimate,
    /// The live textures whose usage allows a color attachment. An overlapping
    /// analytical subset of `textures`.
    pub color_attachment_textures: MemoryEstimate,
    /// The live textures whose usage allows a depth/stencil attachment. An
    /// overlapping analytical subset of `textures`.
    pub depth_stencil_textures: MemoryEstimate,
}

/// The distinct logical objects currently in RHI inventory (§47.14).
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct LiveObjectCounts {
    /// Live buffers.
    pub buffers: u64,

    /// Live textures.
    pub textures: u64,

    /// Live textures usable as a render target. An overlapping subset of
    /// `textures`.
    pub render_target_textures: u64,
    /// Live textures usable as a color attachment. An overlapping subset of
    /// `textures`.
    pub color_attachment_textures: u64,
    /// Live textures usable as a depth/stencil attachment. An overlapping
    /// subset of `textures`.
    pub depth_stencil_textures: u64,

    /// Live texture views.
    pub texture_views: u64,
    /// Live samplers.
    pub samplers: u64,

    /// Live shader modules.
    pub shader_modules: u64,

    /// Live bind group layouts.
    pub bind_group_layouts: u64,
    /// Live bind groups.
    pub bind_groups: u64,

    /// Live pipeline interfaces.
    pub pipeline_interfaces: u64,
    /// Live raster pipelines.
    pub raster_pipelines: u64,
    /// Live compute pipelines.
    pub compute_pipelines: u64,

    /// Frames currently in `Acquired` or `PlannedForPresent`.
    pub outstanding_frames: u64,
}

/// The live inventory of one device identity (§47.15).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct InventoryStatistics {
    /// The device identity this inventory belongs to.
    pub device: crate::rhi::platform::DeviceIdentity,
    /// The distinct live objects.
    pub objects: LiveObjectCounts,
    /// The logical memory estimate of those objects.
    pub memory: ResourceMemoryStatistics,
}

/// The per-texture facts the inventory keeps for its memory estimate.
#[derive(Clone, Copy, Debug)]
struct TextureRecord {
    estimate: MemoryEstimate,
    render_target: bool,
    color_attachment: bool,
    depth_stencil: bool,
}

impl TextureRecord {
    fn new(descriptor: &TextureDescriptor) -> Self {
        let usage = descriptor.usage;
        let color_attachment = usage.contains(TextureUsage::COLOR_ATTACHMENT);
        let depth_stencil = usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT);
        Self {
            estimate: estimate_texture_bytes(descriptor),
            render_target: color_attachment || depth_stencil,
            color_attachment,
            depth_stencil,
        }
    }
}

/// The object lifecycle table behind the live inventory.
///
/// It survives a new collection epoch on purpose: `configure()` restarts event
/// counters, because those counts belong to a collection rule, but the objects
/// that exist do not change because a caller asked for different statistics
/// (§47.3).
#[derive(Debug, Default)]
pub(crate) struct InventoryTable {
    buffers: HashMap<ObjectId, u64>,
    textures: HashMap<ObjectId, TextureRecord>,
    texture_views: HashSet<ObjectId>,
    samplers: HashSet<ObjectId>,
    shader_modules: HashSet<ObjectId>,
    bind_group_layouts: HashSet<ObjectId>,
    bind_groups: HashSet<ObjectId>,
    pipeline_interfaces: HashSet<ObjectId>,
    raster_pipelines: HashSet<ObjectId>,
    compute_pipelines: HashSet<ObjectId>,
    frames: HashSet<AcquiredFrameId>,
}

impl InventoryTable {
    /// Adds `object` to the live inventory.
    pub(crate) fn insert(&mut self, object: &CreatedObject<'_>) {
        match object {
            CreatedObject::Buffer { id, descriptor } => {
                self.buffers.insert(*id, descriptor.size);
            }
            CreatedObject::Texture { id, descriptor } => {
                self.textures.insert(*id, TextureRecord::new(descriptor));
            }
            CreatedObject::TextureView(id) => {
                self.texture_views.insert(*id);
            }
            CreatedObject::Sampler(id) => {
                self.samplers.insert(*id);
            }
            CreatedObject::ShaderModule(id) => {
                self.shader_modules.insert(*id);
            }
            CreatedObject::BindGroupLayout(id) => {
                self.bind_group_layouts.insert(*id);
            }
            CreatedObject::BindGroup(id) => {
                self.bind_groups.insert(*id);
            }
            CreatedObject::PipelineInterface(id) => {
                self.pipeline_interfaces.insert(*id);
            }
            CreatedObject::RasterPipeline(id) => {
                self.raster_pipelines.insert(*id);
            }
            CreatedObject::ComputePipeline(id) => {
                self.compute_pipelines.insert(*id);
            }
        }
    }

    /// Removes a reclaimed object from the live inventory.
    pub(crate) fn remove(&mut self, kind: ObjectKind, id: ObjectId) {
        match kind {
            ObjectKind::Buffer => {
                self.buffers.remove(&id);
            }
            ObjectKind::Texture => {
                self.textures.remove(&id);
            }
            ObjectKind::TextureView => {
                self.texture_views.remove(&id);
            }
            ObjectKind::Sampler => {
                self.samplers.remove(&id);
            }
            ObjectKind::ShaderModule => {
                self.shader_modules.remove(&id);
            }
            ObjectKind::BindGroupLayout => {
                self.bind_group_layouts.remove(&id);
            }
            ObjectKind::BindGroup => {
                self.bind_groups.remove(&id);
            }
            ObjectKind::PipelineInterface => {
                self.pipeline_interfaces.remove(&id);
            }
            ObjectKind::RasterPipeline => {
                self.raster_pipelines.remove(&id);
            }
            ObjectKind::ComputePipeline => {
                self.compute_pipelines.remove(&id);
            }
        }
    }

    /// Records that an acquired frame is outstanding.
    pub(crate) fn insert_frame(&mut self, frame: AcquiredFrameId) {
        self.frames.insert(frame);
    }

    /// Records that an acquired frame reached a terminal state.
    pub(crate) fn remove_frame(&mut self, frame: AcquiredFrameId) {
        self.frames.remove(&frame);
    }

    /// The distinct live objects.
    pub(crate) fn counts(&self) -> LiveObjectCounts {
        LiveObjectCounts {
            buffers: count_of(self.buffers.len()),
            textures: count_of(self.textures.len()),
            render_target_textures: count_of(
                self.textures.values().filter(|r| r.render_target).count(),
            ),
            color_attachment_textures: count_of(
                self.textures.values().filter(|r| r.color_attachment).count(),
            ),
            depth_stencil_textures: count_of(
                self.textures.values().filter(|r| r.depth_stencil).count(),
            ),
            texture_views: count_of(self.texture_views.len()),
            samplers: count_of(self.samplers.len()),
            shader_modules: count_of(self.shader_modules.len()),
            bind_group_layouts: count_of(self.bind_group_layouts.len()),
            bind_groups: count_of(self.bind_groups.len()),
            pipeline_interfaces: count_of(self.pipeline_interfaces.len()),
            raster_pipelines: count_of(self.raster_pipelines.len()),
            compute_pipelines: count_of(self.compute_pipelines.len()),
            outstanding_frames: count_of(self.frames.len()),
        }
    }

    /// The logical memory estimate of the live objects.
    pub(crate) fn memory(&self) -> ResourceMemoryStatistics {
        let buffers = sum_estimates(self.buffers.values().copied().map(MemoryEstimate::logical));
        let textures = sum_estimates(self.textures.values().map(|record| record.estimate));
        let total_resources = sum_estimates([buffers, textures].into_iter());
        ResourceMemoryStatistics {
            buffers,
            textures,
            total_resources,
            render_target_textures: sum_estimates(
                self.textures
                    .values()
                    .filter(|record| record.render_target)
                    .map(|record| record.estimate),
            ),
            color_attachment_textures: sum_estimates(
                self.textures
                    .values()
                    .filter(|record| record.color_attachment)
                    .map(|record| record.estimate),
            ),
            depth_stencil_textures: sum_estimates(
                self.textures
                    .values()
                    .filter(|record| record.depth_stencil)
                    .map(|record| record.estimate),
            ),
        }
    }
}

/// The logical estimate of a buffer: its descriptor size (§47.16).
///
/// Native allocation padding, page rounding, and metadata are excluded, which is
/// why the result is an estimate of the descriptor rather than of an allocation.
pub(crate) fn estimate_buffer_bytes(descriptor: &BufferDescriptor) -> MemoryEstimate {
    MemoryEstimate::logical(descriptor.size)
}

/// A `u32` extent shifted right `mip` times, clamped to at least one texel.
fn mip_extent(value: u32, mip: u32) -> u64 {
    // `checked_shr` returns `None` past 31, which is exactly the point where a
    // `u32` extent is empty and the clamp to one applies.
    u64::from(value.checked_shr(mip).unwrap_or(0).max(1))
}

/// A `u32` extent shifted right 32 times is zero, so every mip level from the
/// 32nd on covers one block. The tail is summed instead of iterated, which keeps
/// the estimate O(1) in the declared mip count.
const MAX_ITERATED_MIPS: u32 = 32;

/// The logical estimate of a texture (§47.17).
///
/// The estimate excludes tiling, row alignment, driver metadata, compression,
/// the mip tail, heap fragmentation, aliasing, and residency. A format whose
/// backing is implementation-defined - `Depth24Plus` is the portable example -
/// has no bytes-per-block fact, so its estimate is `Unknown` rather than an
/// invented number.
pub(crate) fn estimate_texture_bytes(descriptor: &TextureDescriptor) -> MemoryEstimate {
    let facts = format_facts(descriptor.format);
    let Some(bytes_per_block) = facts.logical_bytes_per_block() else {
        return MemoryEstimate::unknown();
    };
    let bytes_per_block = u64::from(bytes_per_block);
    let block_width = u64::from(facts.block_width());
    let block_height = u64::from(facts.block_height());
    if block_width == 0 || block_height == 0 {
        return MemoryEstimate::unknown();
    }

    let layers = u64::from(descriptor.array_layers);
    let samples = u64::from(descriptor.sample_count);

    let mut total: u64 = 0;
    let mut capped_level_bytes: u64 = 0;
    let head = descriptor.mip_levels.min(MAX_ITERATED_MIPS);
    for mip in 0..head {
        let blocks_x = mip_extent(descriptor.extent.width, mip).div_ceil(block_width);
        let blocks_y = mip_extent(descriptor.extent.height, mip).div_ceil(block_height);
        let blocks_z = mip_extent(descriptor.extent.depth, mip);
        let level = blocks_x
            .checked_mul(blocks_y)
            .and_then(|value| value.checked_mul(blocks_z))
            .and_then(|value| value.checked_mul(bytes_per_block))
            .and_then(|value| value.checked_mul(layers))
            .and_then(|value| value.checked_mul(samples));
        let Some(level) = level else {
            return MemoryEstimate::unknown();
        };
        let Some(sum) = total.checked_add(level) else {
            return MemoryEstimate::unknown();
        };
        total = sum;
        // Only read when a tail exists, and a tail implies `head` was the full
        // 32 levels, so this was assigned at least once.
        capped_level_bytes = level;
    }
    if descriptor.mip_levels > head {
        let tail_levels = u64::from(descriptor.mip_levels - head);
        let Some(tail) = tail_levels.checked_mul(capped_level_bytes) else {
            return MemoryEstimate::unknown();
        };
        let Some(sum) = total.checked_add(tail) else {
            return MemoryEstimate::unknown();
        };
        total = sum;
    }

    MemoryEstimate::logical(total)
}

/// An aggregate estimate: `Unknown` if any part is unknown or if the sum
/// overflows.
///
/// An unknown part cannot be treated as zero, because a total that silently
/// ignores a class of live objects is worse than no total.
fn sum_estimates(estimates: impl Iterator<Item = MemoryEstimate>) -> MemoryEstimate {
    let mut total: u64 = 0;
    for estimate in estimates {
        let Some(bytes) = estimate.logical_estimated_bytes else {
            return MemoryEstimate::unknown();
        };
        let Some(sum) = total.checked_add(bytes) else {
            return MemoryEstimate::unknown();
        };
        total = sum;
    }
    MemoryEstimate::logical(total)
}

/// A set length as a `u64` counter, saturating on a 32-bit host.
fn count_of(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}
