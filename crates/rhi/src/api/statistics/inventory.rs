//! Live inventory and the logical memory estimate (specification 47.14–47.18).
//!
//! Two questions that share a module because they are asked together: "what
//! exists right now" and "roughly how many bytes is it". Neither is a history —
//! the inventory is a present-tense fact, unaffected by a collection epoch, and
//! the estimate is computed from a descriptor rather than measured from a heap.
//!
//! # The estimate is logical, and the name is load-bearing
//!
//! Section 47.15 freezes the field name as `logical_estimated_bytes` and section
//! 47.17 lists what it excludes:
//!
//! ```text
//! tiling / swizzle         heap fragmentation
//! row alignment            alias reuse
//! driver metadata          residency
//! compression / mip tail
//! ```
//!
//! and names three things it must never be called: `vram_bytes`,
//! `gpu_memory_bytes`, `physical_bytes`. A caller that wants to know how much
//! memory a frame actually costs on a device is asking an allocator question,
//! and section 47.1 leaves that to allocator telemetry rather than answering it
//! here with a number that would be wrong by an unknown factor and would differ
//! per driver.
//!
//! # Where a number comes from, and where it does not
//!
//! `estimate_texture_bytes` implements section 47.17's arithmetic in full, and
//! is driven directly by the contract tests because the rule needs only a
//! descriptor and a format's facts. It answers [`MemoryEstimate::unknown`] in
//! exactly two cases, both of them stated by the specification rather than
//! chosen here: any intermediate overflow, and a format whose
//! `logical_bytes_per_block()` is `None` because the format name fixes a
//! semantic rather than a layout.

use crate::api::format::FormatFacts;
use crate::api::identity::DeviceIdentity;
use crate::api::resource::texture::TextureDescriptor;

/// How much a logical memory estimate can be trusted.
///
/// Two states, and the smaller one is the important one. `Unknown` is not a
/// failure and not an error: it is the answer for a texture whose backing the
/// format does not fix, and a caller that treats it as zero will under-report
/// and a caller that treats it as an error will refuse legal work.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEstimateQuality {
    /// The number is the descriptor-and-format estimate section 47.17 defines.
    ///
    /// Exact for what it estimates and not a measurement of anything: the
    /// exclusion list above is why "exact" is not the word for it.
    LogicalEstimate,
    /// No estimate can be given from the descriptor and format facts.
    Unknown,
}

/// A descriptor-based logical resource byte estimate.
///
/// The two fields travel together rather than as an `Option<u64>` because a
/// quality without a number and a number without a quality are both incomplete:
/// a caller deciding whether to warn about memory wants both, and a caller
/// summing five estimates has to know whether one of them was `Unknown` before
/// it can say anything about the total.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct MemoryEstimate {
    /// The estimated byte count, or `None` when no estimate is available.
    ///
    /// Never zero as a substitute for "unknown": zero is a real answer for a
    /// descriptor that sizes nothing.
    pub logical_estimated_bytes: Option<u64>,
    /// How much the number above can be trusted.
    pub quality: MemoryEstimateQuality,
}

impl MemoryEstimate {
    /// No estimate is available.
    pub const fn unknown() -> Self {
        Self {
            logical_estimated_bytes: None,
            quality: MemoryEstimateQuality::Unknown,
        }
    }

    /// The descriptor-based logical estimate.
    pub const fn logical(bytes: u64) -> Self {
        Self {
            logical_estimated_bytes: Some(bytes),
            quality: MemoryEstimateQuality::LogicalEstimate,
        }
    }
}

impl Default for MemoryEstimate {
    /// `Unknown`, not `logical(0)`.
    ///
    /// Section 47.15 declares `Default` without saying which value it is. It is
    /// `Unknown` here because the other candidate would be a false statement:
    /// `logical(0)` claims a resource of exactly zero bytes with the quality of
    /// a real estimate, and [`ResourceMemoryStatistics`] derives `Default`, so a
    /// defaulted total would assert that an unmeasured inventory occupies
    /// nothing. `Unknown` asserts only that nothing was estimated, which is what
    /// a default value actually knows.
    fn default() -> Self {
        Self::unknown()
    }
}

/// How many distinct logical objects exist right now.
///
/// Section 47.14's definition, which is what makes the counts comparable to the
/// lifecycle counters: a unique logical object in the RHI inventory that has not
/// yet been finally reclaimed or become terminal. Cloning counts once — a buffer
/// handle cloned a hundred times is one buffer — and cloning a frame attachment
/// does not increase `outstanding_frames`, for the same reason.
///
/// The four texture fields deliberately overlap. `textures` counts every
/// texture, and the three attachment fields each count a subset, so a render
/// target that is also a color attachment is counted in both subsets and once in
/// the total. Adding them to `textures` would double-count, which is why each of
/// them says "overlapping subset" in its own documentation.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct LiveObjectCounts {
    /// Buffers that exist.
    pub buffers: u64,

    /// Textures that exist.
    pub textures: u64,

    /// An overlapping subset of textures: those created with the render-target
    /// usage bit.
    pub render_target_textures: u64,
    /// An overlapping subset of textures: those usable as a color attachment.
    pub color_attachment_textures: u64,
    /// An overlapping subset of textures: those usable as a depth or stencil
    /// attachment.
    pub depth_stencil_textures: u64,

    /// Texture views that exist.
    pub texture_views: u64,
    /// Samplers that exist.
    pub samplers: u64,

    /// Shader modules that exist.
    pub shader_modules: u64,

    /// Bind group layouts that exist.
    pub bind_group_layouts: u64,
    /// Bind groups that exist.
    pub bind_groups: u64,

    /// Pipeline interfaces that exist.
    pub pipeline_interfaces: u64,
    /// Raster pipelines that exist.
    pub raster_pipelines: u64,
    /// Compute pipelines that exist.
    pub compute_pipelines: u64,

    /// Frames currently in the acquired or planned-for-present state.
    ///
    /// Not a count of frame attachments a caller is holding: cloning a frame
    /// attachment does not increment it, and dropping one does not decrement it.
    /// The count is of frames the device has actually handed out and not yet
    /// retired.
    pub outstanding_frames: u64,
}

/// The logical byte estimate for each inventory class.
///
/// The three top-level fields are non-overlapping and the three below them are
/// not. Section 47.15 writes that boundary explicitly — "overlapping analytical
/// subsets of Texture; do not add them to `textures` again" — and it is the one
/// arithmetic mistake this record invites: `buffers + textures` is a meaningful
/// total and `textures + color_attachment_textures` is not.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ResourceMemoryStatistics {
    /// Non-overlapping: all live buffers.
    pub buffers: MemoryEstimate,
    /// Non-overlapping: all live textures, including every attachment below.
    pub textures: MemoryEstimate,
    /// Non-overlapping: all live resources, equal to `buffers + textures` where
    /// both estimates are known.
    pub total_resources: MemoryEstimate,

    /// Overlapping subset of `textures`: render-target textures.
    pub render_target_textures: MemoryEstimate,
    /// Overlapping subset of `textures`: color-attachment textures.
    pub color_attachment_textures: MemoryEstimate,
    /// Overlapping subset of `textures`: depth-stencil textures.
    pub depth_stencil_textures: MemoryEstimate,
}

/// The live inventory and its logical memory estimate.
///
/// Returned by [`super::DeviceStatistics::inventory`]. Present-tense, and so
/// unaffected by a collection epoch: a reconfigure restarts the cumulative
/// counters and leaves what exists alone.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct InventoryStatistics {
    /// The device this inventory observed.
    pub device: DeviceIdentity,
    /// How many of each object class exist.
    pub objects: LiveObjectCounts,
    /// The logical byte estimate for each class.
    pub memory: ResourceMemoryStatistics,
}

/// Section 47.17's texture byte estimate, over a descriptor and format facts.
///
/// The algorithm, transcribed in the specification's own order:
///
/// ```text
/// mip_width  = max(1, width  >> mip)
/// mip_height = max(1, height >> mip)
/// mip_depth  = max(1, depth  >> mip)
///
/// blocks_x = ceil(mip_width  / block_width)
/// blocks_y = ceil(mip_height / block_height)
///
/// mip_bytes =
///     blocks_x * blocks_y * mip_depth * bytes_per_block * array_layers * sample_count
/// ```
///
/// summed over every mip level, with checked arithmetic on every intermediate
/// multiplication and addition. Any overflow yields [`MemoryEstimate::unknown`],
/// as does a format whose `logical_bytes_per_block()` is `None` — which section
/// 47.17 requires for implementation-defined backing such as `Depth24Plus`,
/// where the driver chooses the layout and a computed number would be a guess
/// indistinguishable from a measurement.
///
/// The facts are a parameter rather than read from a device, for the reason the
/// resource validators give: a rule that takes the capability answer it must
/// respect stays portable and testable, and the device is left responsible only
/// for producing the facts.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "driven directly by the contract tests, as the module note states; nothing else reads it yet"
    )
)]
pub(crate) fn estimate_texture_bytes(
    desc: &TextureDescriptor,
    facts: &FormatFacts,
) -> MemoryEstimate {
    let Some(bytes_per_block) = facts.logical_bytes_per_block() else {
        return MemoryEstimate::unknown();
    };
    let block_width = u64::from(facts.block_width().max(1));
    let block_height = u64::from(facts.block_height().max(1));
    let layers = u64::from(desc.array_layers);
    let samples = u64::from(desc.sample_count);

    let total = (0..desc.mip_levels).try_fold(0u64, |total, mip| {
        let mip_width = mip_dimension(desc.extent.width, mip);
        let mip_height = mip_dimension(desc.extent.height, mip);
        let mip_depth = mip_dimension(desc.extent.depth, mip);

        let blocks_x = mip_width.checked_add(block_width - 1)? / block_width;
        let blocks_y = mip_height.checked_add(block_height - 1)? / block_height;

        let mip_bytes = blocks_x
            .checked_mul(blocks_y)?
            .checked_mul(mip_depth)?
            .checked_mul(u64::from(bytes_per_block))?
            .checked_mul(layers)?
            .checked_mul(samples)?;

        total.checked_add(mip_bytes)
    });

    match total {
        Some(bytes) => MemoryEstimate::logical(bytes),
        None => MemoryEstimate::unknown(),
    }
}

/// One dimension of a mip level, under the specification's `max(1, dim >> mip)`
/// rule.
///
/// The `max(1, ..)` is what makes a mip chain of a non-power-of-two texture
/// terminate at one texel rather than at zero, and the `checked_shr` is what
/// keeps a caller-supplied `mip_levels` larger than the width of the shift from
/// being a panic: shifting a `u32` past its width is zero by the specification's
/// arithmetic, hence one by its rule.
fn mip_dimension(base: u32, mip: u32) -> u64 {
    match base.checked_shr(mip) {
        Some(value) => u64::from(value.max(1)),
        None => 1,
    }
}
