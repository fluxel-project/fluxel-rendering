//! Format facts, route facts, and submission-lane facts for RHI API v1.
//!
//! This module owns rhi-design section 8 (texture formats and their facts),
//! section 9 (route facts), and section 10 (submission capabilities).
//!
//! # What it is
//!
//! Every question here is answered about a *shape*, not about a global boolean:
//! format facts depend only on the format, texture support depends on
//! `(dimension, format, usage, sample_count, view intent)`, and route support
//! depends on the endpoint shapes that actually change native legality. A
//! backend that answers `Supported` here has committed to executing the
//! corresponding command directly.
//!
//! # What it deliberately does not own
//!
//! There is no `supports_real_overlap`, no `supports_gpu_lane_dependencies`
//! boolean, no `queue_family_index`, and no `native_queue_count`. Multiple
//! logical lanes are not evidence of hardware parallelism, multiple native
//! queues are not a guarantee of overlap, and one native queue does not forbid
//! several logical scheduling lanes. Real overlap is a profiler observation, so
//! it is not a correctness fact and has no place in this vocabulary.
//!
//! Fallback is also not owned here: `RouteSupport::Unsupported` means the RHI
//! command returns [`crate::rhi::RhiErrorKind::Unsupported`]. A backend may never
//! lower a blit into a fullscreen shader, a copy into a staging CPU round trip,
//! or a resolve into a compute shader without the caller asking for it, because
//! that would silently rewrite capture, statistics, and the performance model.

use std::collections::BTreeMap;

use super::binding::{StorageAccess, TextureSampleType};
use super::resource::{
    Extent3d, TextureAspect, TextureAspects, TextureDimension, TextureUsage,
};
use super::shader::ShaderNumericType;

/// The formats RHI API v1 freezes for the current renderer main path.
///
/// Compressed, planar, and video formats are not in P0. Future additions extend
/// this enum and [`format_facts`]; they never add a second compressed-format
/// entry point.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureFormat {
    /// One unorm red channel.
    R8Unorm,
    /// One snorm red channel.
    R8Snorm,
    /// One unsigned integer red channel.
    R8Uint,
    /// One signed integer red channel.
    R8Sint,

    /// Two unorm channels.
    Rg8Unorm,
    /// Two snorm channels.
    Rg8Snorm,
    /// Two unsigned integer channels.
    Rg8Uint,
    /// Two signed integer channels.
    Rg8Sint,

    /// Four unorm channels.
    Rgba8Unorm,
    /// Four unorm channels in the sRGB transfer function.
    Rgba8UnormSrgb,
    /// Four snorm channels.
    Rgba8Snorm,
    /// Four unsigned integer channels.
    Rgba8Uint,
    /// Four signed integer channels.
    Rgba8Sint,

    /// Four unorm channels in BGRA order.
    Bgra8Unorm,
    /// Four unorm channels in BGRA order, sRGB transfer function.
    Bgra8UnormSrgb,

    /// One unsigned integer 16-bit channel.
    R16Uint,
    /// One signed integer 16-bit channel.
    R16Sint,
    /// One half-float channel.
    R16Float,

    /// Two unsigned integer 16-bit channels.
    Rg16Uint,
    /// Two signed integer 16-bit channels.
    Rg16Sint,
    /// Two half-float channels.
    Rg16Float,

    /// Four unsigned integer 16-bit channels.
    Rgba16Uint,
    /// Four signed integer 16-bit channels.
    Rgba16Sint,
    /// Four half-float channels.
    Rgba16Float,

    /// One unsigned integer 32-bit channel.
    R32Uint,
    /// One signed integer 32-bit channel.
    R32Sint,
    /// One 32-bit float channel.
    R32Float,

    /// Two unsigned integer 32-bit channels.
    Rg32Uint,
    /// Two signed integer 32-bit channels.
    Rg32Sint,
    /// Two 32-bit float channels.
    Rg32Float,

    /// Four unsigned integer 32-bit channels.
    Rgba32Uint,
    /// Four signed integer 32-bit channels.
    Rgba32Sint,
    /// Four 32-bit float channels.
    Rgba32Float,

    /// 16-bit unorm depth.
    Depth16Unorm,

    /// Portable depth semantic.
    ///
    /// The backend may choose the actual backing precision and layout that
    /// satisfies the contract, so this promises no fixed bytes per block and
    /// cannot be used for bit-exact VRAM estimates.
    Depth24Plus,

    /// Portable depth plus 8-bit stencil.
    Depth24PlusStencil8,
    /// 32-bit float depth.
    Depth32Float,
    /// 32-bit float depth plus 8-bit stencil.
    Depth32FloatStencil8,
}

/// What a sampled texture binding may legally be filtered with.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlitFilter {
    /// No filtering; texels are selected.
    Nearest,
    /// Linear filtering.
    Linear,
}

/// Whether a format is readable, writable, or both as a storage texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageAccessSupport {
    read_only: bool,
    write_only: bool,
    read_write: bool,
}

impl StorageAccessSupport {
    const NONE: Self = Self {
        read_only: false,
        write_only: false,
        read_write: false,
    };

    const fn read_write_only() -> Self {
        Self {
            read_only: true,
            write_only: true,
            read_write: false,
        }
    }

    pub(crate) const fn new(read_only: bool, write_only: bool, read_write: bool) -> Self {
        Self {
            read_only,
            write_only,
            read_write,
        }
    }

    /// Whether this format supports `access` as a storage texture.
    pub fn supports(&self, access: StorageAccess) -> bool {
        match access {
            StorageAccess::ReadOnly => self.read_only,
            StorageAccess::WriteOnly => self.write_only,
            StorageAccess::ReadWrite => self.read_write,
        }
    }

    /// Whether any storage access is available.
    pub fn is_any(&self) -> bool {
        self.read_only || self.write_only || self.read_write
    }
}

/// Format facts under the current device/adapter contract.
///
/// Fields are opaque so that adding a fact later is not a public struct-literal
/// breaking change.
///
/// `PartialEq`/`Eq`/`Hash` exist so a capability snapshot can be compared
/// exactly and interned; they are not a statement that two equal `FormatFacts`
/// come from the same device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatFacts {
    aspects: u8,
    sample_type: Option<TextureSampleType>,
    storage: StorageAccessSupport,
    color_attachment: bool,
    depth_attachment: bool,
    stencil_attachment: bool,
    blendable: bool,
    has_alpha_channel: bool,
    color_output_type: Option<ShaderNumericType>,
    block_width: u32,
    block_height: u32,
    logical_bytes_per_block: Option<u32>,
}

impl FormatFacts {
    /// The set of color/depth/stencil aspects this format addresses.
    pub fn aspects(&self) -> TextureAspects {
        TextureAspects::from_bits(self.aspects)
    }

    /// The portable sample type for a shader sampled binding.
    ///
    /// `None` means the format cannot be used as a sampled texture at all.
    /// `Float` means a filtering sampler is legal, `UnfilterableFloat` means
    /// only a non-filtering sampler is, and `Sint`/`Uint`/`Depth` carry their
    /// own binding semantics. `filterable` alone cannot replace this, because an
    /// integer or unfilterable-float format can still be shader-read.
    pub fn sample_type(&self) -> Option<TextureSampleType> {
        self.sample_type
    }

    /// Which storage-texture accesses this format supports.
    pub fn storage_access(&self) -> StorageAccessSupport {
        self.storage
    }

    /// Whether this format is legal as a color attachment.
    pub fn color_attachment(&self) -> bool {
        self.color_attachment
    }

    /// Whether this format is legal as a depth attachment.
    pub fn depth_attachment(&self) -> bool {
        self.depth_attachment
    }

    /// Whether this format is legal as a stencil attachment.
    pub fn stencil_attachment(&self) -> bool {
        self.stencil_attachment
    }

    /// Whether this format may be blended. True only for color-attachment
    /// formats.
    pub fn blendable(&self) -> bool {
        self.blendable
    }

    /// Whether the color format has an alpha component.
    pub fn has_alpha_channel(&self) -> bool {
        self.has_alpha_channel
    }

    /// The numeric class a fragment shader must write when this format is a
    /// color attachment. `None` for depth/stencil formats.
    pub fn color_output_type(&self) -> Option<ShaderNumericType> {
        self.color_output_type
    }

    /// How many texels one addressable texel or block covers horizontally.
    pub fn block_width(&self) -> u32 {
        self.block_width
    }

    /// How many texels one addressable texel or block covers vertically.
    pub fn block_height(&self) -> u32 {
        self.block_height
    }

    /// Bytes per block usable for a descriptor-based logical memory estimate.
    ///
    /// Implementation-defined backing such as [`TextureFormat::Depth24Plus`]
    /// returns `None` rather than inventing a number.
    pub fn logical_bytes_per_block(&self) -> Option<u32> {
        self.logical_bytes_per_block
    }

    pub(crate) const fn new(
        aspects: u8,
        sample_type: Option<TextureSampleType>,
        storage: StorageAccessSupport,
        color_attachment: bool,
        depth_attachment: bool,
        stencil_attachment: bool,
        blendable: bool,
        has_alpha_channel: bool,
        color_output_type: Option<ShaderNumericType>,
        logical_bytes_per_block: Option<u32>,
    ) -> Self {
        Self {
            aspects,
            sample_type,
            storage,
            color_attachment,
            depth_attachment,
            stencil_attachment,
            blendable,
            has_alpha_channel,
            color_output_type,
            block_width: 1,
            block_height: 1,
            logical_bytes_per_block,
        }
    }
}

/// The format facts of `format`.
///
/// This is a pure function of the format on purpose: a fact that could differ
/// per device belongs in [`TextureSupportQuery`] or [`RouteQuery`] instead.
pub fn format_facts(format: TextureFormat) -> FormatFacts {
    use ShaderNumericType::{Float32, Sint32, Uint32};
    use TextureFormat as F;

    const COLOR: u8 = TextureAspects::COLOR_BITS;
    const DEPTH: u8 = TextureAspects::DEPTH_BITS;
    const DEPTH_STENCIL: u8 = TextureAspects::DEPTH_BITS | TextureAspects::STENCIL_BITS;

    let rw = StorageAccessSupport::read_write_only();
    let none = StorageAccessSupport::NONE;

    // A normalized or float color format.
    let color = |sample, storage, blendable, alpha, output, bytes| {
        FormatFacts::new(
            COLOR,
            Some(sample),
            storage,
            true,
            false,
            false,
            blendable,
            alpha,
            Some(output),
            Some(bytes),
        )
    };
    // A depth-only format.
    let depth_only = |sample, bytes| {
        FormatFacts::new(
            DEPTH,
            Some(sample),
            none,
            false,
            true,
            false,
            false,
            false,
            None,
            bytes,
        )
    };
    // A combined depth/stencil format.
    let depth_stencil = |bytes| {
        FormatFacts::new(
            DEPTH_STENCIL,
            Some(TextureSampleType::Depth),
            none,
            false,
            true,
            true,
            false,
            false,
            None,
            bytes,
        )
    };

    match format {
        F::R8Unorm => color(TextureSampleType::Float, rw, true, false, Float32, 1),
        F::R8Snorm => color(TextureSampleType::Float, rw, true, false, Float32, 1),
        F::R8Uint => color(TextureSampleType::Uint, rw, false, false, Uint32, 1),
        F::R8Sint => color(TextureSampleType::Sint, rw, false, false, Sint32, 1),

        F::Rg8Unorm => color(TextureSampleType::Float, rw, true, false, Float32, 2),
        F::Rg8Snorm => color(TextureSampleType::Float, rw, true, false, Float32, 2),
        F::Rg8Uint => color(TextureSampleType::Uint, rw, false, false, Uint32, 2),
        F::Rg8Sint => color(TextureSampleType::Sint, rw, false, false, Sint32, 2),

        F::Rgba8Unorm => color(TextureSampleType::Float, rw, true, true, Float32, 4),
        F::Rgba8UnormSrgb => color(TextureSampleType::Float, rw, true, true, Float32, 4),
        F::Rgba8Snorm => color(TextureSampleType::Float, rw, true, true, Float32, 4),
        F::Rgba8Uint => color(TextureSampleType::Uint, rw, false, true, Uint32, 4),
        F::Rgba8Sint => color(TextureSampleType::Sint, rw, false, true, Sint32, 4),

        F::Bgra8Unorm => color(TextureSampleType::Float, rw, true, true, Float32, 4),
        F::Bgra8UnormSrgb => color(TextureSampleType::Float, rw, true, true, Float32, 4),

        F::R16Uint => color(TextureSampleType::Uint, rw, false, false, Uint32, 2),
        F::R16Sint => color(TextureSampleType::Sint, rw, false, false, Sint32, 2),
        F::R16Float => color(TextureSampleType::Float, rw, true, false, Float32, 2),

        F::Rg16Uint => color(TextureSampleType::Uint, rw, false, false, Uint32, 4),
        F::Rg16Sint => color(TextureSampleType::Sint, rw, false, false, Sint32, 4),
        F::Rg16Float => color(TextureSampleType::Float, rw, true, false, Float32, 4),

        F::Rgba16Uint => color(TextureSampleType::Uint, rw, false, true, Uint32, 8),
        F::Rgba16Sint => color(TextureSampleType::Sint, rw, false, true, Sint32, 8),
        F::Rgba16Float => color(TextureSampleType::Float, rw, true, true, Float32, 8),

        F::R32Uint => color(TextureSampleType::Uint, rw, false, false, Uint32, 4),
        F::R32Sint => color(TextureSampleType::Sint, rw, false, false, Sint32, 4),
        F::R32Float => color(TextureSampleType::Float, rw, false, false, Float32, 4),

        F::Rg32Uint => color(TextureSampleType::Uint, rw, false, false, Uint32, 8),
        F::Rg32Sint => color(TextureSampleType::Sint, rw, false, false, Sint32, 8),
        F::Rg32Float => color(TextureSampleType::Float, rw, false, false, Float32, 8),

        F::Rgba32Uint => color(TextureSampleType::Uint, rw, false, true, Uint32, 16),
        F::Rgba32Sint => color(TextureSampleType::Sint, rw, false, true, Sint32, 16),
        F::Rgba32Float => color(TextureSampleType::Float, rw, false, true, Float32, 16),

        F::Depth16Unorm => depth_only(TextureSampleType::Depth, Some(2)),
        F::Depth24Plus => depth_only(TextureSampleType::Depth, None),
        F::Depth24PlusStencil8 => depth_stencil(None),
        F::Depth32Float => depth_only(TextureSampleType::Depth, Some(4)),
        F::Depth32FloatStencil8 => depth_stencil(None),
    }
}

/// A capability question about one texture shape.
///
/// The query deliberately does not contain extent, mip levels, or array layers:
/// those are returned as limits by [`TextureSupport::Supported`], so one query
/// answers "is this kind of texture legal, and how large may it be".
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureSupportQuery {
    dimension: TextureDimension,
    format: TextureFormat,
    usage: TextureUsage,
    sample_count: u32,
    view_formats: Vec<TextureFormat>,
    view_compatibility: TextureViewCompatibility,
}

impl TextureSupportQuery {
    /// A query for the base shape and usage.
    pub fn new(
        dimension: TextureDimension,
        format: TextureFormat,
        usage: TextureUsage,
        sample_count: u32,
    ) -> Self {
        Self {
            dimension,
            format,
            usage,
            sample_count,
            view_formats: Vec::new(),
            view_compatibility: TextureViewCompatibility::NONE,
        }
    }

    /// Adds one permitted alternate view format.
    ///
    /// Formats are canonicalized as a set on construction so that the query
    /// fingerprint does not depend on declaration order or duplicates.
    pub fn with_view_format(mut self, format: TextureFormat) -> Self {
        self.view_formats.push(format);
        self.view_formats.sort_unstable();
        self.view_formats.dedup();
        self
    }

    /// Declares the view intent that must be fixed at texture creation.
    pub fn with_view_compatibility(mut self, compatibility: TextureViewCompatibility) -> Self {
        self.view_compatibility = compatibility;
        self
    }

    /// The queried dimension.
    pub fn dimension(&self) -> TextureDimension {
        self.dimension
    }

    /// The queried base format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// The queried usage set.
    pub fn usage(&self) -> TextureUsage {
        self.usage
    }

    /// The queried sample count.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// The permitted alternate view formats, sorted and deduplicated.
    pub fn view_formats(&self) -> &[TextureFormat] {
        &self.view_formats
    }

    /// The queried view compatibility intent.
    pub fn view_compatibility(&self) -> TextureViewCompatibility {
        self.view_compatibility
    }
}

/// The size limits that apply when a texture shape is supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureSupportLimits {
    max_extent: Extent3d,
    max_mip_levels: u32,
    max_array_layers: u32,
}

impl TextureSupportLimits {
    pub(crate) const fn new(max_extent: Extent3d, max_mip_levels: u32, max_array_layers: u32) -> Self {
        Self {
            max_extent,
            max_mip_levels,
            max_array_layers,
        }
    }

    /// The largest permitted extent for this shape.
    pub fn max_extent(&self) -> Extent3d {
        self.max_extent
    }

    /// The largest permitted mip level count for this shape.
    pub fn max_mip_levels(&self) -> u32 {
        self.max_mip_levels
    }

    /// The largest permitted array layer count for this shape.
    pub fn max_array_layers(&self) -> u32 {
        self.max_array_layers
    }
}

/// Whether a texture shape is legal, and within which limits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureSupport {
    /// The shape is not creatable.
    Unsupported,
    /// The shape is creatable within the returned limits.
    Supported(TextureSupportLimits),
}

impl TextureSupport {
    /// Whether the shape is creatable.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// The limits, present exactly when the shape is supported.
    pub fn limits(&self) -> Option<&TextureSupportLimits> {
        match self {
            Self::Unsupported => None,
            Self::Supported(limits) => Some(limits),
        }
    }
}

/// A capability question about one buffer usage set.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferSupportQuery {
    usage: super::resource::BufferUsage,
}

impl BufferSupportQuery {
    /// A query for `usage`.
    pub fn new(usage: super::resource::BufferUsage) -> Self {
        Self { usage }
    }

    /// The queried usage set.
    pub fn usage(&self) -> super::resource::BufferUsage {
        self.usage
    }
}

/// The limits that apply when a buffer usage set is supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferSupportLimits {
    max_size: u64,
}

impl BufferSupportLimits {
    pub(crate) const fn new(max_size: u64) -> Self {
        Self { max_size }
    }

    /// The largest permitted buffer size for this usage set.
    pub fn max_size(&self) -> u64 {
        self.max_size
    }
}

/// Whether a buffer usage set is legal, and within which limits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferSupport {
    /// The usage set is not creatable.
    Unsupported,
    /// The usage set is creatable within the returned limits.
    Supported(BufferSupportLimits),
}

impl BufferSupport {
    /// Whether the usage set is creatable.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// The limits, present exactly when the usage set is supported.
    pub fn limits(&self) -> Option<&BufferSupportLimits> {
        match self {
            Self::Unsupported => None,
            Self::Supported(limits) => Some(limits),
        }
    }
}

/// View semantics that must be declared when a texture is created.
///
/// Vulkan requires a cube-compatible image creation flag, so a cube view cannot
/// be added later by [`super::TextureView`] creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureViewCompatibility(u32);

impl TextureViewCompatibility {
    /// No extra view intent.
    pub const NONE: Self = Self(0);

    /// The texture permits Cube and CubeArray views.
    ///
    /// Requires `D2`, `width == height`, `array_layers >= 6`, and
    /// `sample_count == 1`.
    pub const CUBE: Self = Self(1 << 0);

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two view intents.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u32 {
        self.0
    }
}

/// A question about whether one portable operation has a legal direct route.
///
/// The key carries the texture shape and sample facts that change native
/// legality. Without them a blit or copy would report `Supported` and then fail
/// on the real descriptor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RouteQuery {
    /// Buffer to buffer copy.
    BufferToBuffer,

    /// Buffer to texture copy.
    BufferToTexture {
        /// The destination texture dimension.
        dimension: TextureDimension,
        /// The destination texture format.
        format: TextureFormat,
        /// The destination aspect.
        aspect: TextureAspect,
    },

    /// Texture to buffer copy.
    TextureToBuffer {
        /// The source texture dimension.
        dimension: TextureDimension,
        /// The source texture format.
        format: TextureFormat,
        /// The source aspect.
        aspect: TextureAspect,
    },

    /// Texture to texture copy.
    TextureToTexture {
        /// The source dimension.
        src_dimension: TextureDimension,
        /// The source format.
        src_format: TextureFormat,
        /// The source aspect.
        src_aspect: TextureAspect,
        /// The source sample count.
        src_sample_count: u32,

        /// The destination dimension.
        dst_dimension: TextureDimension,
        /// The destination format.
        dst_format: TextureFormat,
        /// The destination aspect.
        dst_aspect: TextureAspect,
        /// The destination sample count.
        dst_sample_count: u32,
    },

    /// Multisample resolve.
    Resolve {
        /// The resolve format.
        format: TextureFormat,
        /// The source sample count.
        src_sample_count: u32,
    },

    /// Filtered or unfiltered blit.
    Blit {
        /// The source dimension.
        src_dimension: TextureDimension,
        /// The source format.
        src_format: TextureFormat,

        /// The destination dimension.
        dst_dimension: TextureDimension,
        /// The destination format.
        dst_format: TextureFormat,

        /// The requested filter.
        filter: BlitFilter,
    },
}

/// Alignment limits for a buffer-to-buffer copy route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferCopyLayoutLimits {
    offset_alignment: u64,
    size_alignment: u64,
}

impl BufferCopyLayoutLimits {
    pub(crate) const fn new(offset_alignment: u64, size_alignment: u64) -> Self {
        Self {
            offset_alignment,
            size_alignment,
        }
    }

    /// The required multiple of the copy offset.
    pub fn offset_alignment(&self) -> u64 {
        self.offset_alignment
    }

    /// The required multiple of the copy size.
    pub fn size_alignment(&self) -> u64 {
        self.size_alignment
    }
}

/// Alignment limits for a texel copy route involving a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TexelCopyLayoutLimits {
    buffer_offset_alignment: u64,
    bytes_per_row_alignment: u32,
}

impl TexelCopyLayoutLimits {
    pub(crate) const fn new(buffer_offset_alignment: u64, bytes_per_row_alignment: u32) -> Self {
        Self {
            buffer_offset_alignment,
            bytes_per_row_alignment,
        }
    }

    /// The required multiple of the buffer offset.
    pub fn buffer_offset_alignment(&self) -> u64 {
        self.buffer_offset_alignment
    }

    /// The required multiple of `bytes_per_row`.
    pub fn bytes_per_row_alignment(&self) -> u32 {
        self.bytes_per_row_alignment
    }
}

/// The facts attached to a supported route.
///
/// `rows_per_image` deliberately has no additional alignment field here: its
/// legality follows from extent, format block geometry, and the route rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RouteCapabilities {
    buffer_copy_layout: Option<BufferCopyLayoutLimits>,
    texel_copy_layout: Option<TexelCopyLayoutLimits>,
}

impl RouteCapabilities {
    pub(crate) const fn buffer_copy(limits: BufferCopyLayoutLimits) -> Self {
        Self {
            buffer_copy_layout: Some(limits),
            texel_copy_layout: None,
        }
    }

    pub(crate) const fn texel_copy(limits: TexelCopyLayoutLimits) -> Self {
        Self {
            buffer_copy_layout: None,
            texel_copy_layout: Some(limits),
        }
    }

    pub(crate) const fn unconstrained() -> Self {
        Self {
            buffer_copy_layout: None,
            texel_copy_layout: None,
        }
    }

    /// The buffer copy alignment limits, when this route copies between buffers.
    pub fn buffer_copy_layout(&self) -> Option<BufferCopyLayoutLimits> {
        self.buffer_copy_layout
    }

    /// The texel copy alignment limits, when this route copies to or from a
    /// texture.
    pub fn texel_copy_layout(&self) -> Option<TexelCopyLayoutLimits> {
        self.texel_copy_layout
    }
}

/// Whether a route exists, and with which facts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteSupport {
    /// No legal direct route exists.
    Unsupported,
    /// A legal direct route exists.
    Supported(RouteCapabilities),
}

impl RouteSupport {
    /// Whether a legal direct route exists.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// The route facts, present exactly when the route is supported.
    pub fn capabilities(&self) -> Option<&RouteCapabilities> {
        match self {
            Self::Unsupported => None,
            Self::Supported(capabilities) => Some(capabilities),
        }
    }
}

/// The opaque logical identity of one submission lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionLaneId(u16);

impl SubmissionLaneId {
    pub(crate) const fn new(value: u16) -> Self {
        Self(value)
    }

    /// The underlying value, for diagnostics and canonical ordering.
    pub fn as_u16(self) -> u16 {
        self.0
    }
}

/// The scheduling classification of a submission lane.
///
/// The class is for scheduling and diagnostics only. Command legality follows
/// from [`SubmissionLaneInfo::domains`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SubmissionLaneClass {
    /// A lane that accepts every domain the device supports.
    General,
    /// A lane classified for graphics work.
    Graphics,
    /// A lane classified for compute work.
    Compute,
    /// A lane classified for copy work.
    Transfer,
}

/// Which recorded-work domains may be submitted to a lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaneWorkDomains(u8);

impl LaneWorkDomains {
    /// Raster work.
    pub const RASTER: Self = Self(1 << 0);
    /// Compute work.
    pub const COMPUTE: Self = Self(1 << 1);
    /// Copy work.
    pub const COPY: Self = Self(1 << 2);

    pub(crate) const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two domain sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no domain is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u8 {
        self.0
    }
}

/// One submission lane's identity, class, and executable domains.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionLaneInfo {
    id: SubmissionLaneId,
    class: SubmissionLaneClass,
    domains: LaneWorkDomains,
}

impl SubmissionLaneInfo {
    pub(crate) const fn new(
        id: SubmissionLaneId,
        class: SubmissionLaneClass,
        domains: LaneWorkDomains,
    ) -> Self {
        Self { id, class, domains }
    }

    /// This lane's identity.
    pub fn id(&self) -> SubmissionLaneId {
        self.id
    }

    /// This lane's scheduling classification.
    pub fn class(&self) -> SubmissionLaneClass {
        self.class
    }

    /// The recorded-work domains that may be submitted here.
    ///
    /// This is the correctness fact; `class` is only a label.
    pub fn domains(&self) -> LaneWorkDomains {
        self.domains
    }
}

/// How a happens-before relation between two lanes can be established.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LaneDependencyRoute {
    /// Producer and consumer are already on the same ordered lane, so lane
    /// order itself satisfies happens-before.
    Ordered,

    /// The lanes differ and a GPU-side dependency can be established.
    Gpu,

    /// The two logical lanes must be collapsed into one ordered execution
    /// domain during lowering.
    Collapse,

    /// No portable route establishes the dependency.
    Unsupported,
}

/// The device's lanes and the pairwise relations between them.
///
/// Equality is exact structural equality, used by capability interning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionCapabilities {
    lanes: Vec<SubmissionLaneInfo>,
    routes: BTreeMap<(SubmissionLaneId, SubmissionLaneId), LaneDependencyRoute>,
}

impl SubmissionCapabilities {
    /// A capability set with one base lane accepting the given domains.
    ///
    /// Every device has at least one lane whose domains include `RASTER` and
    /// `COPY`. When `OptionalFeature::Compute` is enabled, at least one lane's
    /// domains include `COMPUTE`; it may be this same base lane.
    pub(crate) fn new(
        mut lanes: Vec<SubmissionLaneInfo>,
        mut routes: Vec<((SubmissionLaneId, SubmissionLaneId), LaneDependencyRoute)>,
    ) -> Self {
        // Both containers are canonicalized so that two backends declaring the
        // same lanes and relations in a different order produce the same
        // capability contract rather than two interned ids.
        lanes.sort_unstable();
        routes.sort_by_key(|(pair, _)| *pair);
        Self {
            lanes,
            routes: routes.into_iter().collect(),
        }
    }

    /// The device's lanes in canonical order.
    pub fn lanes(&self) -> &[SubmissionLaneInfo] {
        &self.lanes
    }

    /// The lane with this identity, if the device exposes one.
    pub fn lane(&self, id: SubmissionLaneId) -> Option<&SubmissionLaneInfo> {
        self.lanes.iter().find(|lane| lane.id == id)
    }

    /// How `to` can be ordered after `from`.
    ///
    /// The same lane is always [`LaneDependencyRoute::Ordered`], because one
    /// lane is one logically ordered submission domain. A pair the backend did
    /// not prove resolves to [`LaneDependencyRoute::Unsupported`], so an
    /// unproved relation never silently becomes a correctness claim.
    pub fn dependency_route(
        &self,
        from: SubmissionLaneId,
        to: SubmissionLaneId,
    ) -> LaneDependencyRoute {
        if from == to {
            return LaneDependencyRoute::Ordered;
        }
        self.routes
            .get(&(from, to))
            .copied()
            .unwrap_or(LaneDependencyRoute::Unsupported)
    }

    /// Whether every id in `this` is a lane of this device.
    pub(crate) fn has_lane(&self, id: SubmissionLaneId) -> bool {
        self.lane(id).is_some()
    }

    /// Whether this lane set satisfies the base guarantee for `compute_enabled`.
    pub(crate) fn satisfies_base_guarantee(&self, compute_enabled: bool) -> bool {
        let base = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
        let has_base = self.lanes.iter().any(|lane| lane.domains.contains(base));
        let has_compute = !compute_enabled
            || self
                .lanes
                .iter()
                .any(|lane| lane.domains.contains(LaneWorkDomains::COMPUTE));
        has_base && has_compute
    }
}

impl core::fmt::Display for LaneDependencyRoute {
    /// The stable spelling used by diagnostics and capture metadata.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Ordered => "ordered",
            Self::Gpu => "gpu",
            Self::Collapse => "collapse",
            Self::Unsupported => "unsupported",
        })
    }
}
