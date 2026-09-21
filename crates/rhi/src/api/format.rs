//! The format vocabulary, per-format facts, and the texture-support query
//! (specification section 8).
//!
//! Section 8 splits what looks like one question into two, and the split is the
//! reason this module exists:
//!
//! ```text
//! FormatFacts           what can this format itself do under the current
//!                       Device/Adapter contract?
//! TextureSupportQuery   can a texture with this dimension + format + usage +
//!                       sample_count be created?
//! ```
//!
//! Merging them again is the mistake the chapter names. A format fact is
//! answerable without naming a descriptor — a depth format carries a depth
//! aspect whether or not anyone creates one — while a support question is not,
//! because on Vulkan and DX12 image-format support depends on the image type,
//! the usage mask, and the sample count *together*. A single "is this format
//! supported" boolean would have to answer a question nobody can ask: section
//! 8.4 gives the worked counter-example, where `Rgba16Float` is a legal 2D color
//! attachment and an illegal 3D one.
//!
//! # What this module owns
//!
//! - [`TextureFormat`], the frozen P0 format set.
//! - [`FormatFacts`], the per-format fact surface, including
//!   `logical_bytes_per_block` — section 8.2 defines it and the audit note in
//!   `0.16-plan.md` (A9) settles that the method stays with its type, because a
//!   method cannot live in a module its receiver is not defined in.
//! - [`TextureSupportQuery`], [`TextureSupportLimits`], and [`TextureSupport`],
//!   which together decide whether a *descriptor* is creatable.
//!
//! # What this module deliberately does not own
//!
//! - *Binding* support — whether a stage-visibility/kind/count/dynamic-offset
//!   combination can be expressed at all — is `BindingSupportQuery` (section
//!   20.4). Section 8 opens by naming the two queries and forbidding their
//!   merger.
//! - Surface presentability is a presentation fact (module 05): one format can
//!   be presentable on one surface and not on another, so a format cannot carry
//!   the answer.
//! - Copy row-pitch alignment is a route fact ([`super::resource::route`]).
//! - Whether a `TextureDescriptor::view_formats` list is *legal* is answered by
//!   texture-view validation, because only that code knows the aspect, dimension,
//!   and subresource range the view will actually use.
//!
//! Those four exclusions are section 8.2's own list of what must not be a format
//! fact, and they are the boundary of this module.
//!
//! # The one part of section 8 that is not here
//!
//! Section 8.5 declares a method on `EnabledCapabilities`, a type owned by
//! module 01:
//!
//! ```text
//! EnabledCapabilities::texture_view_format_compatible(base_format, view_format)
//!     -> bool
//! ```
//!
//! It is the only part of section 8 that belongs to another module's receiver,
//! so it is written where that receiver is defined
//! ([`crate::api::capability::EnabledCapabilities::texture_view_format_compatible`])
//! rather than here: an inherent impl needs its type, and a second definition of
//! a section-8.5 interface inside a section-8 module is what the root
//! specification forbids.

use crate::api::binding::vocabulary::{StorageAccess, TextureSampleType};
use crate::api::resource::subresource::TextureAspects;
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};
use crate::api::shader::vocabulary::ShaderNumericType;

/// The portable texture formats.
///
/// Compressed formats deliberately live in this same vocabulary.  Their support
/// is still queried per concrete format through [`FormatFacts`] and
/// [`TextureSupportQuery`]; there is no coarse `CompressedTextureApi` switch.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// One 8-bit unsigned normalized red channel.
    R8Unorm,
    /// One 8-bit signed normalized red channel.
    R8Snorm,
    /// One 8-bit unsigned integer red channel.
    R8Uint,
    /// One 8-bit signed integer red channel.
    R8Sint,

    /// Two 8-bit unsigned normalized channels.
    Rg8Unorm,
    /// Two 8-bit signed normalized channels.
    Rg8Snorm,
    /// Two 8-bit unsigned integer channels.
    Rg8Uint,
    /// Two 8-bit signed integer channels.
    Rg8Sint,

    /// Four 8-bit unsigned normalized channels.
    Rgba8Unorm,
    /// Four 8-bit unsigned normalized channels, sRGB encoded.
    Rgba8UnormSrgb,
    /// Four 8-bit signed normalized channels.
    Rgba8Snorm,
    /// Four 8-bit unsigned integer channels.
    Rgba8Uint,
    /// Four 8-bit signed integer channels.
    Rgba8Sint,

    // Block-compressed families. Each variant remains independently probeable:
    // support for ETC2 is not evidence for BC or ASTC support (and vice versa).
    /// BC1/DXT1 RGBA, linear.
    Bc1RgbaUnorm,
    /// BC1/DXT1 RGBA, sRGB.
    Bc1RgbaUnormSrgb,
    /// BC2/DXT3 RGBA, linear.
    Bc2RgbaUnorm,
    /// BC2/DXT3 RGBA, sRGB.
    Bc2RgbaUnormSrgb,
    /// BC3/DXT5 RGBA, linear.
    Bc3RgbaUnorm,
    /// BC3/DXT5 RGBA, sRGB.
    Bc3RgbaUnormSrgb,
    /// BC4 single-channel unsigned normalized.
    Bc4RUnorm,
    /// BC4 single-channel signed normalized.
    Bc4RSnorm,
    /// BC5 two-channel unsigned normalized.
    Bc5RgUnorm,
    /// BC5 two-channel signed normalized.
    Bc5RgSnorm,
    /// BC6H RGB unsigned floating point.
    Bc6hRgbUfloat,
    /// BC6H RGB signed floating point.
    Bc6hRgbFloat,
    /// BC7 RGBA, linear.
    Bc7RgbaUnorm,
    /// BC7 RGBA, sRGB.
    Bc7RgbaUnormSrgb,

    /// ETC2 RGB8, linear.
    Etc2Rgb8Unorm,
    /// ETC2 RGB8, sRGB.
    Etc2Rgb8UnormSrgb,
    /// ETC2 RGB8 with one-bit alpha, linear.
    Etc2Rgb8A1Unorm,
    /// ETC2 RGB8 with one-bit alpha, sRGB.
    Etc2Rgb8A1UnormSrgb,
    /// ETC2 RGBA8, linear.
    Etc2Rgba8Unorm,
    /// ETC2 RGBA8, sRGB.
    Etc2Rgba8UnormSrgb,
    /// EAC single-channel unsigned normalized.
    EacR11Unorm,
    /// EAC single-channel signed normalized.
    EacR11Snorm,
    /// EAC two-channel unsigned normalized.
    EacRg11Unorm,
    /// EAC two-channel signed normalized.
    EacRg11Snorm,

    /// ASTC 4x4, linear.
    Astc4x4Unorm,
    /// ASTC 4x4, sRGB.
    Astc4x4UnormSrgb,
    /// ASTC 4x4 HDR. Availability is an exact per-format device fact.
    Astc4x4Hdr,
    /// ASTC 5x4, linear.
    Astc5x4Unorm,
    /// ASTC 5x4, sRGB.
    Astc5x4UnormSrgb,
    /// ASTC 5x4 HDR.
    Astc5x4Hdr,
    /// ASTC 5x5, linear.
    Astc5x5Unorm,
    /// ASTC 5x5, sRGB.
    Astc5x5UnormSrgb,
    /// ASTC 5x5 HDR.
    Astc5x5Hdr,
    /// ASTC 6x5, linear.
    Astc6x5Unorm,
    /// ASTC 6x5, sRGB.
    Astc6x5UnormSrgb,
    /// ASTC 6x5 HDR.
    Astc6x5Hdr,
    /// ASTC 6x6, linear.
    Astc6x6Unorm,
    /// ASTC 6x6, sRGB.
    Astc6x6UnormSrgb,
    /// ASTC 6x6 HDR.
    Astc6x6Hdr,
    /// ASTC 8x5, linear.
    Astc8x5Unorm,
    /// ASTC 8x5, sRGB.
    Astc8x5UnormSrgb,
    /// ASTC 8x5 HDR.
    Astc8x5Hdr,
    /// ASTC 8x6, linear.
    Astc8x6Unorm,
    /// ASTC 8x6, sRGB.
    Astc8x6UnormSrgb,
    /// ASTC 8x6 HDR.
    Astc8x6Hdr,
    /// ASTC 8x8, linear.
    Astc8x8Unorm,
    /// ASTC 8x8, sRGB.
    Astc8x8UnormSrgb,
    /// ASTC 8x8 HDR.
    Astc8x8Hdr,
    /// ASTC 10x5, linear.
    Astc10x5Unorm,
    /// ASTC 10x5, sRGB.
    Astc10x5UnormSrgb,
    /// ASTC 10x5 HDR.
    Astc10x5Hdr,
    /// ASTC 10x6, linear.
    Astc10x6Unorm,
    /// ASTC 10x6, sRGB.
    Astc10x6UnormSrgb,
    /// ASTC 10x6 HDR.
    Astc10x6Hdr,
    /// ASTC 10x8, linear.
    Astc10x8Unorm,
    /// ASTC 10x8, sRGB.
    Astc10x8UnormSrgb,
    /// ASTC 10x8 HDR.
    Astc10x8Hdr,
    /// ASTC 10x10, linear.
    Astc10x10Unorm,
    /// ASTC 10x10, sRGB.
    Astc10x10UnormSrgb,
    /// ASTC 10x10 HDR.
    Astc10x10Hdr,
    /// ASTC 12x10, linear.
    Astc12x10Unorm,
    /// ASTC 12x10, sRGB.
    Astc12x10UnormSrgb,
    /// ASTC 12x10 HDR.
    Astc12x10Hdr,
    /// ASTC 12x12, linear.
    Astc12x12Unorm,
    /// ASTC 12x12, sRGB.
    Astc12x12UnormSrgb,
    /// ASTC 12x12 HDR.
    Astc12x12Hdr,

    /// Four 8-bit unsigned normalized channels in BGRA order.
    Bgra8Unorm,
    /// Four 8-bit unsigned normalized channels in BGRA order, sRGB encoded.
    Bgra8UnormSrgb,

    /// One 16-bit unsigned integer red channel.
    R16Uint,
    /// One 16-bit signed integer red channel.
    R16Sint,
    /// One 16-bit float red channel.
    R16Float,
    /// One 16-bit unsigned normalized red channel.
    R16Unorm,
    /// One 16-bit signed normalized red channel.
    R16Snorm,

    /// Two 16-bit unsigned integer channels.
    Rg16Uint,
    /// Two 16-bit signed integer channels.
    Rg16Sint,
    /// Two 16-bit float channels.
    Rg16Float,
    /// Two 16-bit unsigned normalized channels.
    Rg16Unorm,
    /// Two 16-bit signed normalized channels.
    Rg16Snorm,

    /// Four 16-bit unsigned integer channels.
    Rgba16Uint,
    /// Four 16-bit signed integer channels.
    Rgba16Sint,
    /// Four 16-bit float channels.
    Rgba16Float,
    /// Four 16-bit unsigned normalized channels.
    Rgba16Unorm,
    /// Four 16-bit signed normalized channels.
    Rgba16Snorm,

    /// Shared-exponent RGB 9e5 floating-point color.
    Rgb9e5Ufloat,
    /// Packed RGB 10:10:10 plus alpha 2 unsigned integer.
    Rgb10a2Uint,
    /// Packed RGB 10:10:10 plus alpha 2 normalized color.
    Rgb10a2Unorm,
    /// Packed 11:11:10 unsigned floating-point RGB color.
    Rg11b10Ufloat,

    /// One 32-bit unsigned integer red channel.
    R32Uint,
    /// One 32-bit signed integer red channel.
    R32Sint,
    /// One 32-bit float red channel.
    R32Float,
    /// One 64-bit unsigned integer channel.
    R64Uint,

    /// Two 32-bit unsigned integer channels.
    Rg32Uint,
    /// Two 32-bit signed integer channels.
    Rg32Sint,
    /// Two 32-bit float channels.
    Rg32Float,

    /// Four 32-bit unsigned integer channels.
    Rgba32Uint,
    /// Four 32-bit signed integer channels.
    Rgba32Sint,
    /// Four 32-bit float channels.
    Rgba32Float,

    /// A 16-bit unsigned normalized depth channel.
    Depth16Unorm,

    /// Portable depth semantic.
    ///
    /// The backend may choose the actual backing precision/layout implementation
    /// that satisfies the contract. Therefore this does not promise fixed
    /// bytes-per-block and cannot be used for bit-exact VRAM estimates.
    Depth24Plus,

    /// At least 24 bits of depth plus 8 bits of stencil.
    Depth24PlusStencil8,
    /// A 32-bit float depth channel.
    Depth32Float,
    /// A 32-bit float depth channel plus 8 bits of stencil.
    Depth32FloatStencil8,
    /// Stencil-only 8-bit format.
    Stencil8,
    /// Two-plane 8-bit 4:2:0 video format.
    Nv12,
    /// Two-plane 10-bit 4:2:0 video format stored in 16-bit words.
    P010,
}

/// Which of the single-aspect storage accesses one format supports.
///
/// Separate booleans rather than a bitset because the three answers are read
/// independently and are not independent facts: a read-write format is normally
/// also read-only and write-only, but a backend may legalize a read-write usage
/// that it refuses to expose as a dedicated read-only or write-only one, so
/// collapsing them into a bit set would invite exactly the wrong inference.
///
/// Section 8.2 introduces this type because `filterable` alone cannot stand in
/// for "sampleable": an integer or unfilterable-float format is readable by a
/// shader and still cannot be sampled through a filtering sampler.
///
/// The three fields are read through [`Self::supports`], section 8.2's only
/// accessor for them: [`FormatFacts::storage_access`] hands the whole record out
/// and `supports` answers about the one access a caller is asking after. They are
/// not dead code — both accessors are public, so the compiler reaches them — and
/// a `#[expect(dead_code)]` here would therefore be an unfulfilled expectation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageAccessSupport {
    read_only: bool,
    write_only: bool,
    read_write: bool,
}

impl StorageAccessSupport {
    /// Records the three storage-access answers for one format.
    ///
    /// Crate-private: these are probed facts about a device, and a caller-built
    /// answer would describe hardware that was never asked.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller until Vulkan publishes storage access facts"
        )
    )]
    pub(crate) fn new(read_only: bool, write_only: bool, read_write: bool) -> Self {
        Self {
            read_only,
            write_only,
            read_write,
        }
    }

    /// Whether this format supports one storage access.
    ///
    /// Section 8.2 declares this the type's only accessor, so it is how a caller
    /// reads the one answer it is asking after instead of interpreting the whole
    /// record. The three booleans stay separate for the reason the type documents:
    /// a backend may legalize a read-write usage that it refuses to expose as a
    /// dedicated read-only or write-only one, so a neighbouring `true` is not
    /// evidence for the access a caller actually wants.
    ///
    /// A `false` here is a refusal by this device's format facts, not an absence
    /// of an answer: the record is the probed fact, and section 8.2's exclusion
    /// list keeps "can this format be read/written from a shader" out of
    /// [`TextureSupportQuery`] because it is a property of the format under the
    /// current contract rather than of any one descriptor.
    pub fn supports(&self, access: StorageAccess) -> bool {
        match access {
            StorageAccess::ReadOnly => self.read_only,
            StorageAccess::WriteOnly => self.write_only,
            StorageAccess::ReadWrite => self.read_write,
        }
    }
}

/// Format facts under the current Device/Adapter contract.
///
/// Fields are opaque, so adding a format fact in the future does not cause a
/// public struct-literal breaking change — which is also why this type has no
/// public constructor: a caller-built `FormatFacts` would claim facts about a
/// device that was never queried.
///
/// The facts held here are the ones the *format alone* decides. Anything that
/// depends on how the format is used — 3D creation, MSAA, usage combinations,
/// row-pitch alignment — is not a format fact and is answered by
/// [`TextureSupport`] or by a route fact (section 8.2's exclusion list).
#[derive(Clone, Copy, Debug)]
pub struct FormatFacts {
    format: TextureFormat,
    storage_access: StorageAccessSupport,
    color_attachment: bool,
    depth_attachment: bool,
    stencil_attachment: bool,
    blendable: bool,
    filterable: bool,
    storage_atomic: bool,
}

impl FormatFacts {
    /// Records the probed facts for one format.
    ///
    /// Crate-private: section 8.2 scopes these facts to "the current
    /// Device/Adapter contract", so only the device that probed them may
    /// assemble them.
    #[cfg_attr(
        all(
            not(test),
            not(any(feature = "dx12", feature = "vulkan", feature = "gl-family"))
        ),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan capability ports construct probed texture limits; without either backend this is test-only"
        )
    )]
    pub(crate) fn new(
        format: TextureFormat,
        storage_access: StorageAccessSupport,
        color_attachment: bool,
        depth_attachment: bool,
        stencil_attachment: bool,
        blendable: bool,
    ) -> Self {
        Self {
            format,
            storage_access,
            color_attachment,
            depth_attachment,
            stencil_attachment,
            blendable,
            // A backend must opt in after probing its native format feature
            // bits. In particular this keeps R32Float fail-closed instead of
            // claiming filterability from the enum spelling.
            filterable: false,
            storage_atomic: false,
        }
    }

    /// Adds device-probed sampling and atomic-storage answers to a fact record.
    /// This is crate-private because callers must not manufacture hardware facts.
    pub(crate) fn with_sampling_and_atomic(
        mut self,
        filterable: bool,
        storage_atomic: bool,
    ) -> Self {
        self.filterable = filterable;
        self.storage_atomic = storage_atomic;
        self
    }

    /// Set of Color / Depth / Stencil aspects.
    ///
    /// Decided by the format name alone: no backend can give a depth format a
    /// color aspect, and a depth-stencil format without both would not be one.
    pub fn aspects(&self) -> TextureAspects {
        format_aspects(self.format)
    }

    /// Portable sample type for a shader sampled binding.
    ///
    /// `None` means the format cannot be used as a sampled texture at all
    /// (section 8.2). It is not an error and not "unknown": the caller has asked a
    /// question with a negative answer, and the answer is about the format rather
    /// than about this device. Every currently named format is sampleable when
    /// the device's precise format/use query supports it; BC, ETC2/EAC and ASTC
    /// therefore answer the floating-point sample class. `None` remains in the
    /// contract for future formats such as non-sampleable planar representations.
    ///
    /// The two float variants are the reason this is not a `bool` "filterable":
    /// [`TextureSampleType::Float`] permits a filtering sampler and
    /// [`TextureSampleType::UnfilterableFloat`] permits only a non-filtering one,
    /// which is the distinction section 8.2 opens with — integer and
    /// unfilterable-float formats "can be shader read but cannot be filtered". A
    /// filterability boolean would answer a different question for the integer
    /// and depth classes, whose sample types carry the binding's semantics
    /// instead.
    pub fn sample_type(&self) -> Option<TextureSampleType> {
        match sample_type(self.format) {
            Some(TextureSampleType::UnfilterableFloat) if self.filterable => {
                Some(TextureSampleType::Float)
            }
            other => other,
        }
    }

    /// Whether this device permits linear filtering for this concrete format.
    /// It is intentionally a probed answer rather than a static format table.
    pub fn filterable(&self) -> bool {
        self.filterable
    }

    /// Whether storage-image atomic operations are supported for this format.
    pub fn storage_atomic(&self) -> bool {
        self.storage_atomic
    }

    /// Which storage accesses this format supports.
    ///
    /// Plain readback of the probed answer. Note the asymmetry with
    /// [`Self::aspects`]: whether a format *has* a color aspect is a fact about
    /// the format, while whether it may be written from a shader is a fact about
    /// the device, which is why this one is probed rather than derived.
    pub fn storage_access(&self) -> StorageAccessSupport {
        self.storage_access
    }

    /// Whether the format may be used as a color attachment.
    ///
    /// Probed rather than derived: section 8.2's exclusion list sends "whether a
    /// particular usage combination can be created" to [`TextureSupport`], and a
    /// device that refuses a format as a render target is making exactly that
    /// statement. Deriving it from [`Self::aspects`] would report a permission
    /// the device never granted.
    pub fn color_attachment(&self) -> bool {
        self.color_attachment
    }

    /// Whether the format may be used as a depth attachment.
    ///
    /// Probed for the same reason as [`Self::color_attachment`].
    pub fn depth_attachment(&self) -> bool {
        self.depth_attachment
    }

    /// Whether the format may be used as a stencil attachment.
    ///
    /// Probed for the same reason as [`Self::color_attachment`].
    pub fn stencil_attachment(&self) -> bool {
        self.stencil_attachment
    }

    /// Can be true only for color-attachment formats.
    ///
    /// Probed, and the section states the one implication this pulls rather than
    /// the fact itself: blending is a property of the attachment, so `true` here
    /// is not a claim that the format is a color attachment, only that it is not
    /// disqualified as one.
    pub fn blendable(&self) -> bool {
        self.blendable
    }

    /// Whether the color format has an alpha component.
    ///
    /// Decided by the format name: `Bgra8Unorm` has an alpha channel even though
    /// its components are stored in the opposite order from `Rgba8Unorm`, and
    /// the one- and two-channel families do not.
    pub fn has_alpha_channel(&self) -> bool {
        has_alpha_channel(self.format)
    }

    /// Numeric class required when a fragment shader writes this color attachment.
    ///
    /// Section 8.2's mapping, read off the format name: a normalized or float
    /// format writes `Float32`, a signed integer format `Sint32`, and an unsigned
    /// integer format `Uint32`.
    ///
    /// `None` for compressed, depth and depth/stencil formats. They are not color
    /// attachments, so they have no class to state. Command clear validation
    /// translates this same answer instead of maintaining a second format table.
    pub fn color_output_type(&self) -> Option<ShaderNumericType> {
        color_output_type(self.format)
    }

    /// How many texels one addressable texel/block covers, horizontally.
    ///
    /// One for uncompressed formats; the codec block width for BC, ETC2/EAC and
    /// ASTC. Keeping this in the central format table makes transfer validation,
    /// backend footprints and logical estimates agree.
    pub fn block_width(&self) -> u32 {
        block_extent(self.format).0
    }

    /// How many texels one addressable texel/block covers, vertically.
    ///
    /// One for uncompressed formats; the codec block height for BC, ETC2/EAC and
    /// ASTC.
    pub fn block_height(&self) -> u32 {
        block_extent(self.format).1
    }

    /// Bytes/block that may be used for a descriptor-based logical memory
    /// estimate.
    ///
    /// `None` for the formats whose byte count the format name does not fix, so
    /// that an estimate built from this cannot look exact where it is not
    /// (section 18.8 forbids a logical estimate from pretending to be a real
    /// allocation). See `logical_bytes_per_block` for which formats those are
    /// and why.
    pub fn logical_bytes_per_block(&self) -> Option<u32> {
        logical_bytes_per_block(self.format)
    }
}

/// Bytes per addressable block for the formats whose entry size the name fixes.
///
/// `None` for the three depth formats that name a *semantic* rather than a
/// layout: section 8.1 says the backend chooses the backing for `Depth24Plus`,
/// and the two stencil-bearing depth formats inherit that freedom — a driver may
/// store `Depth32FloatStencil8` as one packed 5-byte texel or as a depth plane
/// plus a separate stencil plane. Returning a number for them would turn a
/// settled estimate into a guess that a caller cannot tell apart from a
/// measurement.
pub(crate) fn logical_bytes_per_block(format: TextureFormat) -> Option<u32> {
    Some(match format {
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint => 1,
        TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg8Uint
        | TextureFormat::Rg8Sint
        | TextureFormat::R16Uint
        | TextureFormat::R16Sint
        | TextureFormat::R16Float
        | TextureFormat::R16Unorm
        | TextureFormat::R16Snorm
        | TextureFormat::Depth16Unorm => 2,
        TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rgba8Sint
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::Rg16Uint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rg16Float
        | TextureFormat::Rg16Unorm
        | TextureFormat::Rg16Snorm
        | TextureFormat::R32Uint
        | TextureFormat::R32Sint
        | TextureFormat::R32Float
        | TextureFormat::Rgb9e5Ufloat
        | TextureFormat::Rgb10a2Uint
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rg11b10Ufloat
        | TextureFormat::Depth32Float => 4,
        TextureFormat::Rgba16Uint
        | TextureFormat::Rgba16Sint
        | TextureFormat::Rgba16Float
        | TextureFormat::Rgba16Unorm
        | TextureFormat::Rgba16Snorm
        | TextureFormat::Rg32Uint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rg32Float => 8,
        TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint | TextureFormat::Rgba32Float => 16,
        TextureFormat::R64Uint => 8,
        TextureFormat::Stencil8 => 1,
        TextureFormat::Bc1RgbaUnorm
        | TextureFormat::Bc1RgbaUnormSrgb
        | TextureFormat::Bc4RUnorm
        | TextureFormat::Bc4RSnorm
        | TextureFormat::Etc2Rgb8Unorm
        | TextureFormat::Etc2Rgb8UnormSrgb
        | TextureFormat::Etc2Rgb8A1Unorm
        | TextureFormat::Etc2Rgb8A1UnormSrgb
        | TextureFormat::EacR11Unorm
        | TextureFormat::EacR11Snorm => 8,
        TextureFormat::Bc2RgbaUnorm
        | TextureFormat::Bc2RgbaUnormSrgb
        | TextureFormat::Bc3RgbaUnorm
        | TextureFormat::Bc3RgbaUnormSrgb
        | TextureFormat::Bc5RgUnorm
        | TextureFormat::Bc5RgSnorm
        | TextureFormat::Bc6hRgbUfloat
        | TextureFormat::Bc6hRgbFloat
        | TextureFormat::Bc7RgbaUnorm
        | TextureFormat::Bc7RgbaUnormSrgb
        | TextureFormat::Etc2Rgba8Unorm
        | TextureFormat::Etc2Rgba8UnormSrgb
        | TextureFormat::EacRg11Unorm
        | TextureFormat::EacRg11Snorm
        | TextureFormat::Astc4x4Unorm
        | TextureFormat::Astc4x4UnormSrgb
        | TextureFormat::Astc4x4Hdr
        | TextureFormat::Astc5x4Unorm
        | TextureFormat::Astc5x4UnormSrgb
        | TextureFormat::Astc5x4Hdr
        | TextureFormat::Astc5x5Unorm
        | TextureFormat::Astc5x5UnormSrgb
        | TextureFormat::Astc5x5Hdr
        | TextureFormat::Astc6x5Unorm
        | TextureFormat::Astc6x5UnormSrgb
        | TextureFormat::Astc6x5Hdr
        | TextureFormat::Astc6x6Unorm
        | TextureFormat::Astc6x6UnormSrgb
        | TextureFormat::Astc6x6Hdr
        | TextureFormat::Astc8x5Unorm
        | TextureFormat::Astc8x5UnormSrgb
        | TextureFormat::Astc8x5Hdr
        | TextureFormat::Astc8x6Unorm
        | TextureFormat::Astc8x6UnormSrgb
        | TextureFormat::Astc8x6Hdr
        | TextureFormat::Astc8x8Unorm
        | TextureFormat::Astc8x8UnormSrgb
        | TextureFormat::Astc8x8Hdr
        | TextureFormat::Astc10x5Unorm
        | TextureFormat::Astc10x5UnormSrgb
        | TextureFormat::Astc10x5Hdr
        | TextureFormat::Astc10x6Unorm
        | TextureFormat::Astc10x6UnormSrgb
        | TextureFormat::Astc10x6Hdr
        | TextureFormat::Astc10x8Unorm
        | TextureFormat::Astc10x8UnormSrgb
        | TextureFormat::Astc10x8Hdr
        | TextureFormat::Astc10x10Unorm
        | TextureFormat::Astc10x10UnormSrgb
        | TextureFormat::Astc10x10Hdr
        | TextureFormat::Astc12x10Unorm
        | TextureFormat::Astc12x10UnormSrgb
        | TextureFormat::Astc12x10Hdr
        | TextureFormat::Astc12x12Unorm
        | TextureFormat::Astc12x12UnormSrgb
        | TextureFormat::Astc12x12Hdr => 16,
        TextureFormat::Depth24Plus
        | TextureFormat::Depth24PlusStencil8
        | TextureFormat::Depth32FloatStencil8
        // Multi-planar formats do not have one byte size: each selected plane
        // has its own footprint and is queried through its aspect.
        | TextureFormat::Nv12
        | TextureFormat::P010 => return None,
    })
}

/// Texel extent represented by one addressable block.  Kept next to the byte
/// table so upload, readback and estimates cannot diverge on a new format.
pub(crate) fn block_extent(format: TextureFormat) -> (u32, u32) {
    match format {
        TextureFormat::Astc4x4Unorm
        | TextureFormat::Astc4x4UnormSrgb
        | TextureFormat::Astc4x4Hdr => (4, 4),
        TextureFormat::Astc5x4Unorm
        | TextureFormat::Astc5x4UnormSrgb
        | TextureFormat::Astc5x4Hdr => (5, 4),
        TextureFormat::Astc5x5Unorm
        | TextureFormat::Astc5x5UnormSrgb
        | TextureFormat::Astc5x5Hdr => (5, 5),
        TextureFormat::Astc6x5Unorm
        | TextureFormat::Astc6x5UnormSrgb
        | TextureFormat::Astc6x5Hdr => (6, 5),
        TextureFormat::Astc6x6Unorm
        | TextureFormat::Astc6x6UnormSrgb
        | TextureFormat::Astc6x6Hdr => (6, 6),
        TextureFormat::Astc8x5Unorm
        | TextureFormat::Astc8x5UnormSrgb
        | TextureFormat::Astc8x5Hdr => (8, 5),
        TextureFormat::Astc8x6Unorm
        | TextureFormat::Astc8x6UnormSrgb
        | TextureFormat::Astc8x6Hdr => (8, 6),
        TextureFormat::Astc8x8Unorm
        | TextureFormat::Astc8x8UnormSrgb
        | TextureFormat::Astc8x8Hdr => (8, 8),
        TextureFormat::Astc10x5Unorm
        | TextureFormat::Astc10x5UnormSrgb
        | TextureFormat::Astc10x5Hdr => (10, 5),
        TextureFormat::Astc10x6Unorm
        | TextureFormat::Astc10x6UnormSrgb
        | TextureFormat::Astc10x6Hdr => (10, 6),
        TextureFormat::Astc10x8Unorm
        | TextureFormat::Astc10x8UnormSrgb
        | TextureFormat::Astc10x8Hdr => (10, 8),
        TextureFormat::Astc10x10Unorm
        | TextureFormat::Astc10x10UnormSrgb
        | TextureFormat::Astc10x10Hdr => (10, 10),
        TextureFormat::Astc12x10Unorm
        | TextureFormat::Astc12x10UnormSrgb
        | TextureFormat::Astc12x10Hdr => (12, 10),
        TextureFormat::Astc12x12Unorm
        | TextureFormat::Astc12x12UnormSrgb
        | TextureFormat::Astc12x12Hdr => (12, 12),
        TextureFormat::Bc1RgbaUnorm
        | TextureFormat::Bc1RgbaUnormSrgb
        | TextureFormat::Bc2RgbaUnorm
        | TextureFormat::Bc2RgbaUnormSrgb
        | TextureFormat::Bc3RgbaUnorm
        | TextureFormat::Bc3RgbaUnormSrgb
        | TextureFormat::Bc4RUnorm
        | TextureFormat::Bc4RSnorm
        | TextureFormat::Bc5RgUnorm
        | TextureFormat::Bc5RgSnorm
        | TextureFormat::Bc6hRgbUfloat
        | TextureFormat::Bc6hRgbFloat
        | TextureFormat::Bc7RgbaUnorm
        | TextureFormat::Bc7RgbaUnormSrgb
        | TextureFormat::Etc2Rgb8Unorm
        | TextureFormat::Etc2Rgb8UnormSrgb
        | TextureFormat::Etc2Rgb8A1Unorm
        | TextureFormat::Etc2Rgb8A1UnormSrgb
        | TextureFormat::Etc2Rgba8Unorm
        | TextureFormat::Etc2Rgba8UnormSrgb
        | TextureFormat::EacR11Unorm
        | TextureFormat::EacR11Snorm
        | TextureFormat::EacRg11Unorm
        | TextureFormat::EacRg11Snorm => (4, 4),
        _ => (1, 1),
    }
}

/// The aspects one format covers.
///
/// Kept as a free function rather than a method so that texture-view validation —
/// which must decide an aspect rule before any `FormatFacts` exists — reads the
/// same table this module's accessor reads. Two copies of this mapping would
/// disagree first on the depth-stencil formats.
pub(crate) fn format_aspects(format: TextureFormat) -> TextureAspects {
    match format {
        TextureFormat::Depth16Unorm | TextureFormat::Depth24Plus | TextureFormat::Depth32Float => {
            TextureAspects::DEPTH
        }
        TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8 => {
            TextureAspects::DEPTH.union(TextureAspects::STENCIL)
        }
        TextureFormat::Stencil8 => TextureAspects::STENCIL,
        TextureFormat::Nv12 | TextureFormat::P010 => {
            TextureAspects::PLANE0.union(TextureAspects::PLANE1)
        }
        _ => TextureAspects::COLOR,
    }
}

/// Whether one format carries an alpha channel.
pub(crate) fn has_alpha_channel(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Rgba8Snorm
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rgba8Sint
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba16Sint
            | TextureFormat::Rgba16Float
            | TextureFormat::Rgba16Unorm
            | TextureFormat::Rgba16Snorm
            | TextureFormat::Rgb10a2Uint
            | TextureFormat::Rgb10a2Unorm
            | TextureFormat::Rgba32Uint
            | TextureFormat::Rgba32Sint
            | TextureFormat::Rgba32Float
            | TextureFormat::Bc1RgbaUnorm
            | TextureFormat::Bc1RgbaUnormSrgb
            | TextureFormat::Bc2RgbaUnorm
            | TextureFormat::Bc2RgbaUnormSrgb
            | TextureFormat::Bc3RgbaUnorm
            | TextureFormat::Bc3RgbaUnormSrgb
            | TextureFormat::Bc7RgbaUnorm
            | TextureFormat::Bc7RgbaUnormSrgb
            | TextureFormat::Etc2Rgb8A1Unorm
            | TextureFormat::Etc2Rgb8A1UnormSrgb
            | TextureFormat::Etc2Rgba8Unorm
            | TextureFormat::Etc2Rgba8UnormSrgb
            // ASTC's portable vocabulary is RGBA for every block footprint
            // and channel encoding, including the SFLOAT/HDR forms.
            | TextureFormat::Astc4x4Unorm
            | TextureFormat::Astc4x4UnormSrgb
            | TextureFormat::Astc4x4Hdr
            | TextureFormat::Astc5x4Unorm
            | TextureFormat::Astc5x4UnormSrgb
            | TextureFormat::Astc5x4Hdr
            | TextureFormat::Astc5x5Unorm
            | TextureFormat::Astc5x5UnormSrgb
            | TextureFormat::Astc5x5Hdr
            | TextureFormat::Astc6x5Unorm
            | TextureFormat::Astc6x5UnormSrgb
            | TextureFormat::Astc6x5Hdr
            | TextureFormat::Astc6x6Unorm
            | TextureFormat::Astc6x6UnormSrgb
            | TextureFormat::Astc6x6Hdr
            | TextureFormat::Astc8x5Unorm
            | TextureFormat::Astc8x5UnormSrgb
            | TextureFormat::Astc8x5Hdr
            | TextureFormat::Astc8x6Unorm
            | TextureFormat::Astc8x6UnormSrgb
            | TextureFormat::Astc8x6Hdr
            | TextureFormat::Astc8x8Unorm
            | TextureFormat::Astc8x8UnormSrgb
            | TextureFormat::Astc8x8Hdr
            | TextureFormat::Astc10x5Unorm
            | TextureFormat::Astc10x5UnormSrgb
            | TextureFormat::Astc10x5Hdr
            | TextureFormat::Astc10x6Unorm
            | TextureFormat::Astc10x6UnormSrgb
            | TextureFormat::Astc10x6Hdr
            | TextureFormat::Astc10x8Unorm
            | TextureFormat::Astc10x8UnormSrgb
            | TextureFormat::Astc10x8Hdr
            | TextureFormat::Astc10x10Unorm
            | TextureFormat::Astc10x10UnormSrgb
            | TextureFormat::Astc10x10Hdr
            | TextureFormat::Astc12x10Unorm
            | TextureFormat::Astc12x10UnormSrgb
            | TextureFormat::Astc12x10Hdr
            | TextureFormat::Astc12x12Unorm
            | TextureFormat::Astc12x12UnormSrgb
            | TextureFormat::Astc12x12Hdr
    )
}

/// The portable sample type of one format (section 8.2).
///
/// Kept as a free function rather than a method, for the reason given on
/// [`format_aspects`]: the answer is a property of the format name, so a rule that
/// must decide it before — or without — any `FormatFacts` reads this same table
/// rather than keeping a copy, and two copies would disagree first on the integer
/// classes.
///
/// The classes come from the format name alone:
///
/// ```text
/// Uint   every unsigned integer format
/// Sint   every signed integer format
/// Depth  the five depth and depth/stencil formats
/// Float  the normalized and 16-bit float formats
/// UnfilterableFloat  the 32-bit float formats
/// ```
///
/// The 32-bit float formats are the unfilterable ones because filterability is a
/// property of the *format*, not of the sampler that reads it: `R16Float`,
/// `Rg16Float`, and `Rgba16Float` may be read through a filtering sampler, while
/// `R32Float`, `Rg32Float`, and `Rgba32Float` may only be read through a
/// non-filtering one. Answering `Float` for them would tell a caller that a
/// filtering sampler is legal where the platform refuses one, and section 8.2
/// names exactly this class — "can be shader read but cannot be filtered" — as
/// the reason [`TextureSampleType`] is a sample type rather than a `filterable`
/// boolean.
///
/// `Option` is the contract the chapter freezes (`None` = "cannot be used as a
/// sampled texture"). Every currently named format is sampleable when its
/// per-device facts support the requested use: BC, ETC2/EAC and ASTC decode to
/// the floating-point sample class, while depth-stencil formats answer
/// [`TextureSampleType::Depth`]. The view's aspect choice is checked separately.
pub(crate) fn sample_type(format: TextureFormat) -> Option<TextureSampleType> {
    if is_compressed(format) {
        // BC/ETC2/EAC/ASTC are sampled floating-point representations. Exact
        // filterability remains a per-format device fact, not an enum claim.
        return Some(TextureSampleType::Float);
    }
    Some(match format {
        TextureFormat::R8Uint
        | TextureFormat::Rg8Uint
        | TextureFormat::Rgba8Uint
        | TextureFormat::R16Uint
        | TextureFormat::Rg16Uint
        | TextureFormat::Rgba16Uint
        | TextureFormat::R32Uint
        | TextureFormat::Rg32Uint
        | TextureFormat::Rgba32Uint
        | TextureFormat::Rgb10a2Uint => TextureSampleType::Uint,

        TextureFormat::R64Uint => TextureSampleType::Uint,

        TextureFormat::R8Sint
        | TextureFormat::Rg8Sint
        | TextureFormat::Rgba8Sint
        | TextureFormat::R16Sint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rgba16Sint
        | TextureFormat::R32Sint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rgba32Sint => TextureSampleType::Sint,

        TextureFormat::Depth16Unorm
        | TextureFormat::Depth24Plus
        | TextureFormat::Depth24PlusStencil8
        | TextureFormat::Depth32Float
        | TextureFormat::Depth32FloatStencil8 => TextureSampleType::Depth,

        TextureFormat::Stencil8 | TextureFormat::Nv12 | TextureFormat::P010 => return None,

        TextureFormat::R32Float | TextureFormat::Rg32Float | TextureFormat::Rgba32Float => {
            TextureSampleType::UnfilterableFloat
        }

        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::R16Float
        | TextureFormat::Rg16Float
        | TextureFormat::Rgba16Float => TextureSampleType::Float,
        TextureFormat::R16Unorm
        | TextureFormat::R16Snorm
        | TextureFormat::Rg16Unorm
        | TextureFormat::Rg16Snorm
        | TextureFormat::Rgba16Unorm
        | TextureFormat::Rgba16Snorm
        | TextureFormat::Rgb9e5Ufloat
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rg11b10Ufloat => TextureSampleType::Float,
        // Keep a newly added format fail-closed until its sample class has a
        // reviewed entry. Format probing walks this table on real adapters;
        // returning no sample contract is safe, while an `unreachable!` here
        // would turn a missing table arm into a device-dependent panic.
        _ => return None,
    })
}

/// The numeric class a fragment shader must write for one color format
/// (section 8.2).
///
/// A pure function of the format name, like [`format_aspects`]: the class follows
/// the format's components and not the device, so there is nothing here a backend
/// could answer differently. `None` is the depth and depth/stencil formats, which
/// are never color attachments — section 8.2 says so explicitly, and a class for
/// them would be an answer to a question the chapter refuses to ask.
pub(crate) fn color_output_type(format: TextureFormat) -> Option<ShaderNumericType> {
    // Block-compressed formats are color-aspect textures but never portable
    // render targets; do not manufacture a fragment-output contract for them.
    if is_compressed(format) {
        return None;
    }
    Some(match format {
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::R16Float
        | TextureFormat::Rg16Float
        | TextureFormat::Rgba16Float
        | TextureFormat::R32Float
        | TextureFormat::Rg32Float
        | TextureFormat::Rgba32Float => ShaderNumericType::Float32,
        TextureFormat::R16Unorm
        | TextureFormat::R16Snorm
        | TextureFormat::Rg16Unorm
        | TextureFormat::Rg16Snorm
        | TextureFormat::Rgba16Unorm
        | TextureFormat::Rgba16Snorm
        | TextureFormat::Rgb9e5Ufloat
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rg11b10Ufloat => ShaderNumericType::Float32,

        TextureFormat::R8Uint
        | TextureFormat::Rg8Uint
        | TextureFormat::Rgba8Uint
        | TextureFormat::R16Uint
        | TextureFormat::Rg16Uint
        | TextureFormat::Rgba16Uint
        | TextureFormat::R32Uint
        | TextureFormat::Rg32Uint
        | TextureFormat::Rgba32Uint => ShaderNumericType::Uint32,
        TextureFormat::Rgb10a2Uint | TextureFormat::R64Uint => ShaderNumericType::Uint32,

        TextureFormat::R8Sint
        | TextureFormat::Rg8Sint
        | TextureFormat::Rgba8Sint
        | TextureFormat::R16Sint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rgba16Sint
        | TextureFormat::R32Sint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rgba32Sint => ShaderNumericType::Sint32,

        TextureFormat::Depth16Unorm
        | TextureFormat::Depth24Plus
        | TextureFormat::Depth24PlusStencil8
        | TextureFormat::Depth32Float
        | TextureFormat::Depth32FloatStencil8 => return None,
        TextureFormat::Stencil8 | TextureFormat::Nv12 | TextureFormat::P010 => return None,
        // A future format without a reviewed color-output class is not a color
        // attachment candidate, never a backend panic.
        _ => return None,
    })
}

/// Whether this vocabulary member is block-compressed. This only decides
/// format-intrinsic validation; usable features are always reported per format.
pub(crate) fn is_compressed(format: TextureFormat) -> bool {
    block_extent(format) != (1, 1)
}

/// The key of a "can this texture be created" question.
///
/// Section 8.3 lists the four facts that change native legality — dimension,
/// format, usage, sample count — plus the two creation-time view intents, and
/// says explicitly *not* to put extent, mip level count, or array layer count in
/// the key. Those three are not part of the question because they are part of
/// the answer: [`TextureSupportLimits`] returns the maxima for the key, and the
/// descriptor's own values are then checked against them. Asking per-extent
/// would make the capability cache answer an unbounded number of questions.
///
/// The query is a value, not a handle: a caller builds it, asks the device, and
/// may keep it as a cache key.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureSupportQuery {
    dimension: TextureDimension,
    format: TextureFormat,
    usage: TextureUsage,
    sample_count: u32,

    /// Alternate view formats declared as permitted when the Texture is created.
    view_formats: Vec<TextureFormat>,

    /// View-compatibility intent that must be fixed when the Texture is created.
    ///
    /// For example, a Vulkan cube view requires the image to have cube-compatible
    /// semantics at creation time.
    view_compatibility: TextureViewCompatibility,
}

impl TextureSupportQuery {
    /// Opens a query with no alternate view format and no view intent.
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

    /// Declares one more alternate view format as permitted.
    ///
    /// Consuming, so a query is built and then used rather than mutated in
    /// place: the device caches support by key, and a key that could change
    /// after an answer was given would let the cache serve the wrong one.
    ///
    /// The list is *not* reordered or deduplicated here. Section 13.1 makes the
    /// descriptor's `view_formats` a canonical set, and section 13.3 requires
    /// the query to be built from that descriptor, so any list a caller adds by
    /// hand is their own key — not a value this type silently rewrites, which
    /// would make two spellings of one question compare unequal.
    pub fn with_view_format(mut self, format: TextureFormat) -> Self {
        self.view_formats.push(format);
        self
    }

    /// Declares a view-compatibility intent that must hold at creation time.
    ///
    /// Section 13.2 is why this is in the key at all: a Vulkan cube view needs
    /// the image created with cube-compatible semantics, so a query that asked
    /// only about format and usage would answer `Supported` for a texture no
    /// cube view can then be built from.
    pub fn with_view_compatibility(mut self, compatibility: TextureViewCompatibility) -> Self {
        self.view_compatibility = compatibility;
        self
    }

    /// The queried dimension.
    pub fn dimension(&self) -> TextureDimension {
        self.dimension
    }

    /// The queried format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// The queried usage mask.
    pub fn usage(&self) -> TextureUsage {
        self.usage
    }

    /// The queried sample count.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// The alternate view formats declared so far, in the order added.
    pub fn view_formats(&self) -> &[TextureFormat] {
        &self.view_formats
    }

    /// The declared view-compatibility intent.
    pub fn view_compatibility(&self) -> TextureViewCompatibility {
        self.view_compatibility
    }
}

/// The descriptor-specific maxima returned by a supported query.
///
/// These are the numbers section 8.3 keeps *out* of the query key: they are what
/// a `Supported` answer hands back, so that the same cached answer can serve
/// every extent, mip level count, and array layer count within it.
#[derive(Clone, Copy, Debug)]
pub struct TextureSupportLimits {
    max_extent: Extent3d,
    max_mip_levels: u32,
    max_array_layers: u32,
}

impl TextureSupportLimits {
    /// Records one descriptor's maxima.
    ///
    /// Crate-private: limits are a device answer, and a caller-built one would
    /// be a capability claim about hardware nobody asked.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn new(max_extent: Extent3d, max_mip_levels: u32, max_array_layers: u32) -> Self {
        Self {
            max_extent,
            max_mip_levels,
            max_array_layers,
        }
    }

    /// The largest extent this key permits.
    pub fn max_extent(&self) -> Extent3d {
        self.max_extent
    }

    /// The largest mip level count this key permits.
    pub fn max_mip_levels(&self) -> u32 {
        self.max_mip_levels
    }

    /// The largest array layer count this key permits.
    pub fn max_array_layers(&self) -> u32 {
        self.max_array_layers
    }
}

/// Whether a texture described by a [`TextureSupportQuery`] can be created.
///
/// `#[non_exhaustive]` because a future backend may need a third answer — a
/// "supported only through a fallback the caller must opt into" is the obvious
/// candidate — and callers must keep a wildcard arm rather than match exhaustively.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum TextureSupport {
    /// No texture with this key can be created on this device.
    Unsupported,

    /// The key is creatable, within the returned maxima.
    Supported(TextureSupportLimits),
}

impl TextureSupport {
    /// Whether the key is creatable.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// The maxima for the key, or `None` when it is unsupported.
    pub fn limits(&self) -> Option<&TextureSupportLimits> {
        match self {
            Self::Unsupported => None,
            Self::Supported(limits) => Some(limits),
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical capability encoding
// ---------------------------------------------------------------------------
//
// The rules of the encoding, and what it is for, are stated once in
// `api::capability::CapabilityFacts`. It lives here because every field read
// below is private to this module.

impl TextureFormat {
    /// Writes this format's canonical byte.
    ///
    /// A fieldless enum encodes as its discriminant; see
    /// [`crate::api::shader::ShaderStage::encode_into`] for why that dependency on
    /// declaration order is the intended one. This is the type on which that
    /// dependency is most visible — 90 variants — which is also why it is
    /// written as a cast rather than a 90-arm match that could be edited out of
    /// step with the declaration.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }

    /// Every format in section 8.1's frozen set, in declaration order.
    ///
    /// Crate-private, and a backend probing a format table is its caller. Written
    /// as an explicit list rather than as a range over the discriminants, because
    /// unlike [`Self::encode_into`] this one *is* a claim about which formats
    /// exist: a range would silently include a variant nobody meant to probe, and
    /// a list that falls behind the declaration is caught by the length assertion
    /// in this module's tests rather than by a capability answer that is quietly
    /// missing a format.
    ///
    /// Every variant appears, including the two depth formats that permit a
    /// driver to pick a bit layout — [`Self::Depth24Plus`] and
    /// [`Self::Depth24PlusStencil8`]. They are listed here because the *portable*
    /// set is what this enumerates; whether a given backend can name a native
    /// format for one is that backend's answer to give, and narrowing this list
    /// would move that decision into the format vocabulary.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        [
            Self::R8Unorm,
            Self::R8Snorm,
            Self::R8Uint,
            Self::R8Sint,
            Self::Rg8Unorm,
            Self::Rg8Snorm,
            Self::Rg8Uint,
            Self::Rg8Sint,
            Self::Rgba8Unorm,
            Self::Rgba8UnormSrgb,
            Self::Rgba8Snorm,
            Self::Rgba8Uint,
            Self::Rgba8Sint,
            Self::Bc1RgbaUnorm,
            Self::Bc1RgbaUnormSrgb,
            Self::Bc2RgbaUnorm,
            Self::Bc2RgbaUnormSrgb,
            Self::Bc3RgbaUnorm,
            Self::Bc3RgbaUnormSrgb,
            Self::Bc4RUnorm,
            Self::Bc4RSnorm,
            Self::Bc5RgUnorm,
            Self::Bc5RgSnorm,
            Self::Bc6hRgbUfloat,
            Self::Bc6hRgbFloat,
            Self::Bc7RgbaUnorm,
            Self::Bc7RgbaUnormSrgb,
            Self::Etc2Rgb8Unorm,
            Self::Etc2Rgb8UnormSrgb,
            Self::Etc2Rgb8A1Unorm,
            Self::Etc2Rgb8A1UnormSrgb,
            Self::Etc2Rgba8Unorm,
            Self::Etc2Rgba8UnormSrgb,
            Self::EacR11Unorm,
            Self::EacR11Snorm,
            Self::EacRg11Unorm,
            Self::EacRg11Snorm,
            Self::Astc4x4Unorm,
            Self::Astc4x4UnormSrgb,
            Self::Astc4x4Hdr,
            Self::Astc5x4Unorm,
            Self::Astc5x4UnormSrgb,
            Self::Astc5x4Hdr,
            Self::Astc5x5Unorm,
            Self::Astc5x5UnormSrgb,
            Self::Astc5x5Hdr,
            Self::Astc6x5Unorm,
            Self::Astc6x5UnormSrgb,
            Self::Astc6x5Hdr,
            Self::Astc6x6Unorm,
            Self::Astc6x6UnormSrgb,
            Self::Astc6x6Hdr,
            Self::Astc8x5Unorm,
            Self::Astc8x5UnormSrgb,
            Self::Astc8x5Hdr,
            Self::Astc8x6Unorm,
            Self::Astc8x6UnormSrgb,
            Self::Astc8x6Hdr,
            Self::Astc8x8Unorm,
            Self::Astc8x8UnormSrgb,
            Self::Astc8x8Hdr,
            Self::Astc10x5Unorm,
            Self::Astc10x5UnormSrgb,
            Self::Astc10x5Hdr,
            Self::Astc10x6Unorm,
            Self::Astc10x6UnormSrgb,
            Self::Astc10x6Hdr,
            Self::Astc10x8Unorm,
            Self::Astc10x8UnormSrgb,
            Self::Astc10x8Hdr,
            Self::Astc10x10Unorm,
            Self::Astc10x10UnormSrgb,
            Self::Astc10x10Hdr,
            Self::Astc12x10Unorm,
            Self::Astc12x10UnormSrgb,
            Self::Astc12x10Hdr,
            Self::Astc12x12Unorm,
            Self::Astc12x12UnormSrgb,
            Self::Astc12x12Hdr,
            Self::Bgra8Unorm,
            Self::Bgra8UnormSrgb,
            Self::R16Uint,
            Self::R16Sint,
            Self::R16Float,
            Self::R16Unorm,
            Self::R16Snorm,
            Self::Rg16Uint,
            Self::Rg16Sint,
            Self::Rg16Float,
            Self::Rg16Unorm,
            Self::Rg16Snorm,
            Self::Rgba16Uint,
            Self::Rgba16Sint,
            Self::Rgba16Float,
            Self::Rgba16Unorm,
            Self::Rgba16Snorm,
            Self::Rgb9e5Ufloat,
            Self::Rgb10a2Uint,
            Self::Rgb10a2Unorm,
            Self::Rg11b10Ufloat,
            Self::R32Uint,
            Self::R32Sint,
            Self::R32Float,
            Self::R64Uint,
            Self::Rg32Uint,
            Self::Rg32Sint,
            Self::Rg32Float,
            Self::Rgba32Uint,
            Self::Rgba32Sint,
            Self::Rgba32Float,
            Self::Depth16Unorm,
            Self::Depth24Plus,
            Self::Depth24PlusStencil8,
            Self::Depth32Float,
            Self::Depth32FloatStencil8,
            Self::Stencil8,
            Self::Nv12,
            Self::P010,
        ]
        .into_iter()
    }
}

impl FormatFacts {
    /// Writes these facts' canonical bytes.
    ///
    /// The inner format is written even though the map key is already a
    /// [`TextureFormat`]. A `FormatFacts` whose inner format disagrees with the
    /// key it was stored under is a provider bug, and dropping the field would
    /// make two *different* fact maps encode identically — which is the one thing
    /// an interning id must never do.
    ///
    /// The three storage answers go out as one bitmask rather than three bytes or
    /// a byte per [`StorageAccess`]: they are one probed record, and a bitmask is
    /// already canonical. `as u8` casts are deliberately avoided in favour of
    /// `u8::from`, which cannot read as a numeric conversion of a value whose
    /// numeric meaning would matter.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        self.format.encode_into(out);
        let access = u8::from(self.storage_access.read_only)
            | u8::from(self.storage_access.write_only) << 1
            | u8::from(self.storage_access.read_write) << 2;
        out.push(access);
        let attachments = u8::from(self.color_attachment)
            | u8::from(self.depth_attachment) << 1
            | u8::from(self.stencil_attachment) << 2
            | u8::from(self.blendable) << 3;
        out.push(attachments);
        out.push(u8::from(self.filterable) | (u8::from(self.storage_atomic) << 1));
    }
}

impl TextureSupport {
    /// Writes this answer as a tag, followed by the maxima when there are any.
    ///
    /// `Unsupported` carries no body, so it can never encode as `Supported` with
    /// zeroed maxima. The two say different things — one says the texture cannot
    /// exist, the other says it exists and may be no larger than zero on some
    /// axis — and the second is not a spelling of the first.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::Unsupported => out.push(0),
            Self::Supported(limits) => {
                out.push(1);
                out.extend_from_slice(&limits.max_extent.width.to_le_bytes());
                out.extend_from_slice(&limits.max_extent.height.to_le_bytes());
                out.extend_from_slice(&limits.max_extent.depth.to_le_bytes());
                out.extend_from_slice(&limits.max_mip_levels.to_le_bytes());
                out.extend_from_slice(&limits.max_array_layers.to_le_bytes());
            }
        }
    }
}
