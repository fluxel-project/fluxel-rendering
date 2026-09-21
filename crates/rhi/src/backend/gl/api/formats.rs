//! Exact format facts for one GL-family context.
//!
//! Facts are either unconditional guarantees of the selected API profile or
//! the result of a concrete operation probe.  A guarantee is not described as
//! a probe: providers may add operation-probed facts later without changing
//! the meaning of the baseline record.

use std::collections::BTreeMap;

use super::{GlExtensionSet, GlFamilyProfile, GlKnownExtension, GlLimits};

/// Formats with a current common-RHI meaning in the GL-family layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    /// Eight-bit normalized red, green, blue, and alpha channels.
    Rgba8Unorm,
    /// Eight-bit sRGB red, green, blue and normalized alpha channels.
    Rgba8Srgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8Srgb,
    R16Uint,
    R16Sint,
    R16Float,
    R16Unorm,
    R16Snorm,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rg16Unorm,
    Rg16Snorm,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Unorm,
    Rgba16Snorm,
    /// Sixteen-bit floating-point RGBA channels.
    Rgba16Float,
    /// Thirty-two-bit floating-point RGBA channels.
    Rgba32Float,
    Rgb9e5Ufloat,
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    R32Uint,
    R32Sint,
    R32Float,
    R64Uint,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba32Uint,
    Rgba32Sint,
    /// Sixteen-bit normalized depth.
    Depth16Unorm,
    Depth24Unorm,
    /// Twenty-four-bit depth plus eight-bit stencil.
    Depth24PlusStencil8,
    /// Thirty-two-bit floating-point depth.
    Depth32Float,
    Depth32FloatStencil8,
    Stencil8,
    /// BC1/DXT1 RGB blocks.
    Bc1RgbUnorm,
    /// BC1/DXT1 RGBA blocks.
    Bc1RgbaUnorm,
    /// BC1/DXT1 RGBA blocks decoded as sRGB.
    Bc1RgbaSrgb,
    /// BC1/DXT1 RGB blocks decoded as sRGB.
    Bc1RgbSrgb,
    /// BC2/DXT3 RGBA blocks.
    Bc2RgbaUnorm,
    /// BC2/DXT3 RGBA blocks decoded as sRGB.
    Bc2RgbaSrgb,
    /// BC3/DXT5 RGBA blocks.
    Bc3RgbaUnorm,
    /// BC3/DXT5 RGBA blocks decoded as sRGB.
    Bc3RgbaSrgb,
    /// BC4 one-channel unsigned-normalized blocks.
    Bc4RUnorm,
    /// BC4 one-channel signed-normalized blocks.
    Bc4RSnorm,
    /// BC5 two-channel unsigned-normalized blocks.
    Bc5RgUnorm,
    /// BC5 two-channel signed-normalized blocks.
    Bc5RgSnorm,
    /// BC6H unsigned-float RGB blocks.
    Bc6hRgbUfloat,
    /// BC6H signed-float RGB blocks.
    Bc6hRgbSfloat,
    /// BC7 RGBA blocks.
    Bc7RgbaUnorm,
    /// BC7 RGBA blocks decoded as sRGB.
    Bc7RgbaSrgb,
    /// Exact ASTC block dimensions and color space.
    Astc {
        block: GlAstcBlock,
        color_space: GlCompressedColorSpace,
    },
    /// ETC2 RGB blocks.
    Etc2Rgb8Unorm,
    /// ETC2 RGB blocks decoded as sRGB.
    Etc2Rgb8Srgb,
    /// ETC2 RGBA blocks.
    Etc2Rgba8Unorm,
    /// ETC2 RGBA blocks decoded as sRGB.
    Etc2Rgba8Srgb,
    /// ETC2 RGB with one-bit alpha blocks.
    Etc2Rgb8A1Unorm,
    /// ETC2 RGB with one-bit alpha blocks decoded as sRGB.
    Etc2Rgb8A1Srgb,
    /// EAC one-channel unsigned-normalized blocks.
    EacR11Unorm,
    /// EAC two-channel unsigned-normalized blocks.
    EacRg11Unorm,
    /// EAC one-channel signed-normalized blocks.
    EacR11Snorm,
    /// EAC two-channel signed-normalized blocks.
    EacRg11Snorm,
}

/// ASTC LDR block dimensions supported by the compressed-format contract.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlAstcBlock {
    /// 4 by 4 texels.
    B4x4,
    /// 5 by 4 texels.
    B5x4,
    /// 5 by 5 texels.
    B5x5,
    /// 6 by 5 texels.
    B6x5,
    /// 6 by 6 texels.
    B6x6,
    /// 8 by 5 texels.
    B8x5,
    /// 8 by 6 texels.
    B8x6,
    /// 8 by 8 texels.
    B8x8,
    /// 10 by 5 texels.
    B10x5,
    /// 10 by 6 texels.
    B10x6,
    /// 10 by 8 texels.
    B10x8,
    /// 10 by 10 texels.
    B10x10,
    /// 12 by 10 texels.
    B12x10,
    /// 12 by 12 texels.
    B12x12,
}

/// Transfer function encoded by a compressed texture format.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlCompressedColorSpace {
    /// Linear decoding.
    Linear,
    /// sRGB decoding.
    Srgb,
    /// ASTC HDR decode mode. It is not interchangeable with LDR-linear: the
    /// same block dimensions use a distinct capability extension and must
    /// remain distinguishable in the typed format key.
    Hdr,
}

/// Compression family. This is descriptive metadata, never an enablement bit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlCompressedTextureFamily {
    /// Block Compression / S3TC family.
    Bc,
    /// Adaptive Scalable Texture Compression.
    Astc,
    /// Ericsson Texture Compression 2.
    Etc2,
    /// Ericsson Alpha Compression.
    Eac,
}

/// Exact upload layout for a compressed texture format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlCompressedFormatInfo {
    /// Descriptive family; a caller must still select one exact `GlFormat`.
    pub family: GlCompressedTextureFamily,
    /// Encoded texels per block horizontally.
    pub block_width: u8,
    /// Encoded texels per block vertically.
    pub block_height: u8,
    /// Bytes in each encoded block.
    pub block_bytes: u8,
    /// Texture decode color space.
    pub color_space: GlCompressedColorSpace,
}

/// Failure to derive an exact compressed upload byte count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlCompressedSizeError {
    /// Compressed texture dimensions must both be nonzero.
    ZeroExtent,
    /// The encoded block count or byte count overflowed `u64`.
    Overflow,
}

impl GlCompressedFormatInfo {
    /// Returns the exact encoded byte count for one compressed mip extent.
    ///
    /// Uses ordinary ceil-divided block counts for each exact format.
    pub(crate) fn checked_encoded_size(
        self,
        width: u32,
        height: u32,
    ) -> Result<u64, GlCompressedSizeError> {
        if width == 0 || height == 0 {
            return Err(GlCompressedSizeError::ZeroExtent);
        }
        let blocks = |extent: u32, block: u8| -> Result<u64, GlCompressedSizeError> {
            u64::from(extent)
                .checked_add(u64::from(block) - 1)
                .ok_or(GlCompressedSizeError::Overflow)
                .map(|value| value / u64::from(block))
        };
        let blocks_wide = blocks(width, self.block_width)?;
        let blocks_high = blocks(height, self.block_height)?;
        blocks_wide
            .checked_mul(blocks_high)
            .and_then(|count| count.checked_mul(u64::from(self.block_bytes)))
            .ok_or(GlCompressedSizeError::Overflow)
    }
}

impl GlFormat {
    /// Returns exact compressed upload metadata, if this is a compressed format.
    pub(crate) fn compressed_info(self) -> Option<GlCompressedFormatInfo> {
        use GlCompressedColorSpace::{Linear, Srgb};
        use GlCompressedTextureFamily::{Astc, Bc, Eac, Etc2};
        let info =
            |family, block_width, block_height, block_bytes, color_space| GlCompressedFormatInfo {
                family,
                block_width,
                block_height,
                block_bytes,
                color_space,
            };
        match self {
            Self::Bc1RgbUnorm | Self::Bc1RgbaUnorm => Some(info(Bc, 4, 4, 8, Linear)),
            Self::Bc1RgbaSrgb | Self::Bc1RgbSrgb => Some(info(Bc, 4, 4, 8, Srgb)),
            Self::Bc2RgbaUnorm | Self::Bc3RgbaUnorm => Some(info(Bc, 4, 4, 16, Linear)),
            Self::Bc2RgbaSrgb | Self::Bc3RgbaSrgb => Some(info(Bc, 4, 4, 16, Srgb)),
            Self::Bc4RUnorm | Self::Bc4RSnorm => Some(info(Bc, 4, 4, 8, Linear)),
            Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbSfloat
            | Self::Bc7RgbaUnorm => Some(info(Bc, 4, 4, 16, Linear)),
            Self::Bc7RgbaSrgb => Some(info(Bc, 4, 4, 16, Srgb)),
            Self::Astc { block, color_space } => {
                let (block_width, block_height) = match block {
                    GlAstcBlock::B4x4 => (4, 4),
                    GlAstcBlock::B5x4 => (5, 4),
                    GlAstcBlock::B5x5 => (5, 5),
                    GlAstcBlock::B6x5 => (6, 5),
                    GlAstcBlock::B6x6 => (6, 6),
                    GlAstcBlock::B8x5 => (8, 5),
                    GlAstcBlock::B8x6 => (8, 6),
                    GlAstcBlock::B8x8 => (8, 8),
                    GlAstcBlock::B10x5 => (10, 5),
                    GlAstcBlock::B10x6 => (10, 6),
                    GlAstcBlock::B10x8 => (10, 8),
                    GlAstcBlock::B10x10 => (10, 10),
                    GlAstcBlock::B12x10 => (12, 10),
                    GlAstcBlock::B12x12 => (12, 12),
                };
                Some(info(Astc, block_width, block_height, 16, color_space))
            }
            Self::Etc2Rgb8Unorm => Some(info(Etc2, 4, 4, 8, Linear)),
            Self::Etc2Rgb8Srgb => Some(info(Etc2, 4, 4, 8, Srgb)),
            Self::Etc2Rgba8Unorm => Some(info(Etc2, 4, 4, 16, Linear)),
            Self::Etc2Rgba8Srgb => Some(info(Etc2, 4, 4, 16, Srgb)),
            Self::Etc2Rgb8A1Unorm => Some(info(Etc2, 4, 4, 8, Linear)),
            Self::Etc2Rgb8A1Srgb => Some(info(Etc2, 4, 4, 8, Srgb)),
            Self::EacR11Unorm | Self::EacR11Snorm => Some(info(Eac, 4, 4, 8, Linear)),
            Self::EacRg11Unorm | Self::EacRg11Snorm => Some(info(Eac, 4, 4, 16, Linear)),
            _ => None,
        }
    }

    /// Whether the selected profile guarantees this exact compressed format.
    ///
    /// GLES3 has ETC2/EAC in core. This deliberately answers per format, not
    /// per compression family, so a positive ETC2 fact cannot imply BC/ASTC.
    pub(crate) const fn is_core_compressed_for(self, profile: GlFamilyProfile) -> bool {
        matches!(
            (profile, self),
            (
                GlFamilyProfile::Embedded { major: 3, .. }
                    | GlFamilyProfile::Desktop {
                        major: 4,
                        minor: 3..
                    },
                Self::Etc2Rgb8Unorm
                    | Self::Etc2Rgb8Srgb
                    | Self::Etc2Rgba8Unorm
                    | Self::Etc2Rgba8Srgb
                    | Self::Etc2Rgb8A1Unorm
                    | Self::Etc2Rgb8A1Srgb
                    | Self::EacR11Unorm
                    | Self::EacRg11Unorm
                    | Self::EacR11Snorm
                    | Self::EacRg11Snorm
            )
        )
    }

    const fn is_non_compressed_core_baseline_for(self, profile: GlFamilyProfile) -> bool {
        // WebGL2 deliberately has a much smaller static internal-format
        // guarantee set than desktop GL or GLES.  These three are the exact
        // rows its browser discovery records without an extension or an FBO
        // probe; accepting the native table wholesale here would turn a
        // WebGL2 core guarantee into a capability lie.
        if matches!(profile, GlFamilyProfile::WebGl2) {
            return matches!(
                self,
                Self::Rgba8Unorm | Self::Rgba8Srgb | Self::Depth32Float
            );
        }
        match self {
            // Sized R/RG/RGBA integer, normalized, and float storage is core
            // in both desktop GL 4.x and GLES 3.x.  The table deliberately
            // does not turn this into a claim about filtering/renderability;
            // those remain exact operation facts.
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba32Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgb9e5Ufloat
            | Self::Rgb10a2Uint
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat
            | Self::Depth16Unorm
            | Self::Depth24Unorm
            | Self::Depth24PlusStencil8
            | Self::Depth32Float
            | Self::Stencil8 => true,
            // These are desktop-core spellings only at the GL 4.x floor. They
            // are intentionally not promoted to GLES3/WebGL2 guarantees.
            Self::R64Uint | Self::Depth32FloatStencil8 => {
                matches!(profile, GlFamilyProfile::Desktop { .. })
            }
            Self::Bgra8Unorm | Self::Bgra8Srgb => false,
            _ => false,
        }
    }

    const fn is_supplied_by_extension(self, extension: GlKnownExtension) -> bool {
        match extension {
            GlKnownExtension::CompressedTextureS3tc
            | GlKnownExtension::ExtTextureCompressionS3tc => matches!(
                self,
                Self::Bc1RgbUnorm | Self::Bc1RgbaUnorm | Self::Bc2RgbaUnorm | Self::Bc3RgbaUnorm
            ),
            GlKnownExtension::CompressedTextureS3tcSrgb => matches!(
                self,
                Self::Bc1RgbSrgb | Self::Bc1RgbaSrgb | Self::Bc2RgbaSrgb | Self::Bc3RgbaSrgb
            ),
            GlKnownExtension::CompressedTextureBptc
            | GlKnownExtension::ArbTextureCompressionBptc => matches!(
                self,
                Self::Bc6hRgbUfloat | Self::Bc6hRgbSfloat | Self::Bc7RgbaUnorm | Self::Bc7RgbaSrgb
            ),
            GlKnownExtension::CompressedTextureRgtc
            | GlKnownExtension::ExtTextureCompressionRgtc => matches!(
                self,
                Self::Bc4RUnorm | Self::Bc4RSnorm | Self::Bc5RgUnorm | Self::Bc5RgSnorm
            ),
            // WebGL's one ASTC extension carries an additional
            // `supportedProfiles` observation for HDR. Browser discovery adds
            // an HDR row only after that observation; native KHR LDR remains
            // restricted to LDR below.
            GlKnownExtension::CompressedTextureAstc => matches!(self, Self::Astc { .. }),
            GlKnownExtension::KhrTextureCompressionAstcLdr => matches!(
                self,
                Self::Astc {
                    color_space: GlCompressedColorSpace::Linear | GlCompressedColorSpace::Srgb,
                    ..
                }
            ),
            GlKnownExtension::KhrTextureCompressionAstcHdr => matches!(
                self,
                Self::Astc {
                    color_space: GlCompressedColorSpace::Hdr,
                    ..
                }
            ),
            GlKnownExtension::CompressedTextureEtc
            | GlKnownExtension::ArbEs3Compatibility
            | GlKnownExtension::OesCompressedEtc2Rgb8Texture => matches!(
                self,
                Self::Etc2Rgb8Unorm
                    | Self::Etc2Rgb8Srgb
                    | Self::Etc2Rgba8Unorm
                    | Self::Etc2Rgba8Srgb
                    | Self::Etc2Rgb8A1Unorm
                    | Self::Etc2Rgb8A1Srgb
                    | Self::EacR11Unorm
                    | Self::EacRg11Unorm
                    | Self::EacR11Snorm
                    | Self::EacRg11Snorm
            ),
            _ => false,
        }
    }
}

/// The resource class to which an exact format fact applies.
///
/// Multisample renderbuffers and multisample textures have different GL
/// contracts.  In particular, WebGL2 exposes the former but not the latter.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlFormatResourceKind {
    /// A texture, including a single-sample texture when `sample_count` is 1.
    Texture,
    /// A renderbuffer attachment.
    Renderbuffer,
}

/// Origin of an exact format fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlFormatEvidence {
    /// An unconditional guarantee of the selected GL4, GLES3, or WebGL2 API.
    CoreGuaranteed,
    /// An exact, callable compressed-texture extension on this same context.
    ExtensionAcquired(GlKnownExtension),
    /// A context-specific operation was executed and succeeded.
    OperationProbed,
}

/// Exact use facts for one `(format, resource kind, sample_count)` tuple.
///
/// A false field is a conservative unsupported fact, not an invitation for a
/// provider to emulate the operation. Storage facts stay split because
/// read/write support is not implied by each other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlFormatCapabilities {
    /// Queried format.
    pub format: GlFormat,
    /// Texture or renderbuffer contract to which this fact applies.
    pub resource_kind: GlFormatResourceKind,
    /// Exact sample count. Zero is invalid and rejected by the table.
    pub sample_count: u32,
    /// Whether this is a profile guarantee or concrete operation evidence.
    pub evidence: GlFormatEvidence,
    /// Sampling from this format/count is legal.
    pub sampled: bool,
    /// Linear filtering is legal for this format/count.
    pub filterable: bool,
    /// Framebuffer rendering is legal for this format/count.
    pub renderable: bool,
    /// Blending into a color attachment is legal for this format/count.
    pub blendable: bool,
    /// Storage image reads are legal for this format/count.
    pub storage_read: bool,
    /// Storage image writes are legal for this format/count.
    pub storage_write: bool,
    /// Copying from this format/count is legal.
    pub copy_source: bool,
    /// Copying to this format/count is legal.
    pub copy_destination: bool,
}

/// Immutable table keyed by exact format and sample count.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GlFormatTable {
    entries: BTreeMap<(GlFormat, GlFormatResourceKind, u32), GlFormatCapabilities>,
}

impl GlFormatTable {
    /// Inserts one exact observation; conflicting repeated observations are rejected.
    pub(crate) fn record(
        &mut self,
        capabilities: GlFormatCapabilities,
    ) -> Result<(), GlFormatTableError> {
        if capabilities.sample_count == 0 {
            return Err(GlFormatTableError::ZeroSampleCount {
                format: capabilities.format,
            });
        }
        if capabilities.format.compressed_info().is_some()
            && (capabilities.resource_kind != GlFormatResourceKind::Texture
                || capabilities.sample_count != 1
                || capabilities.renderable
                || capabilities.blendable
                || capabilities.storage_read
                || capabilities.storage_write)
        {
            return Err(GlFormatTableError::InvalidCompressedUsage {
                format: capabilities.format,
            });
        }
        if (capabilities.storage_read || capabilities.storage_write)
            && capabilities.evidence != GlFormatEvidence::OperationProbed
        {
            return Err(GlFormatTableError::StorageRequiresOperationProbe {
                format: capabilities.format,
                resource_kind: capabilities.resource_kind,
                sample_count: capabilities.sample_count,
            });
        }
        let key = (
            capabilities.format,
            capabilities.resource_kind,
            capabilities.sample_count,
        );
        if let Some(previous) = self.entries.get(&key) {
            if previous != &capabilities {
                return Err(GlFormatTableError::ConflictingObservation {
                    format: capabilities.format,
                    sample_count: capabilities.sample_count,
                });
            }
            return Ok(());
        }
        self.entries.insert(key, capabilities);
        Ok(())
    }

    /// Returns exact texture facts; texture is the common resource contract.
    pub(crate) fn get(&self, format: GlFormat, sample_count: u32) -> Option<GlFormatCapabilities> {
        self.get_for(GlFormatResourceKind::Texture, format, sample_count)
    }

    /// Returns exact facts for a resource kind, format, and sample count.
    pub(crate) fn get_for(
        &self,
        resource_kind: GlFormatResourceKind,
        format: GlFormat,
        sample_count: u32,
    ) -> Option<GlFormatCapabilities> {
        self.entries
            .get(&(format, resource_kind, sample_count))
            .copied()
    }

    /// Iterates in stable `(format, resource kind, sample count)` order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = GlFormatCapabilities> + '_ {
        self.entries.values().copied()
    }

    /// Returns whether at least one exact format/count pair proves read/write storage.
    pub(crate) fn has_storage_read_write(&self) -> bool {
        self.entries.values().any(|facts| {
            facts.resource_kind == GlFormatResourceKind::Texture
                && facts.evidence == GlFormatEvidence::OperationProbed
                && facts.storage_read
                && facts.storage_write
        })
    }

    /// Verifies evidence against the one profile and extension ledger that
    /// will own this immutable format table.
    pub(crate) fn validate_evidence(
        &self,
        profile: GlFamilyProfile,
        extensions: &GlExtensionSet,
    ) -> Result<(), GlFormatTableError> {
        for facts in self.entries.values() {
            match facts.evidence {
                GlFormatEvidence::CoreGuaranteed => {
                    if !(facts.format.is_non_compressed_core_baseline_for(profile)
                        || facts.format.is_core_compressed_for(profile))
                    {
                        return Err(GlFormatTableError::InvalidCoreGuarantee {
                            format: facts.format,
                            profile,
                        });
                    }
                }
                GlFormatEvidence::ExtensionAcquired(extension) => {
                    if !extension.is_legal_for(profile) {
                        return Err(GlFormatTableError::ExtensionIllegalForProfile {
                            format: facts.format,
                            extension,
                            profile,
                        });
                    }
                    if !extensions.is_acquired(extension) {
                        return Err(GlFormatTableError::ExtensionNotAcquired {
                            format: facts.format,
                            extension,
                        });
                    }
                    if !facts.format.is_supplied_by_extension(extension) {
                        return Err(GlFormatTableError::ExtensionDoesNotSupplyFormat {
                            format: facts.format,
                            extension,
                        });
                    }
                }
                GlFormatEvidence::OperationProbed => {}
            }
        }
        Ok(())
    }

    /// Validates mandatory common format records and every observed sample count.
    pub(crate) fn validate_for_limits(&self, limits: &GlLimits) -> Result<(), GlFormatTableError> {
        for format in [
            GlFormat::Rgba8Unorm,
            GlFormat::Rgba8Srgb,
            GlFormat::Depth32Float,
        ] {
            if !self
                .entries
                .contains_key(&(format, GlFormatResourceKind::Texture, 1))
            {
                return Err(GlFormatTableError::MissingRequiredSampleOne { format });
            }
        }
        for facts in self.entries.values() {
            if facts.sample_count == 1 {
                continue;
            }
            let maximum = match facts.resource_kind {
                GlFormatResourceKind::Renderbuffer => limits.max_samples,
                GlFormatResourceKind::Texture => match facts.format {
                    GlFormat::Depth16Unorm
                    | GlFormat::Depth24Unorm
                    | GlFormat::Depth24PlusStencil8
                    | GlFormat::Depth32Float
                    | GlFormat::Depth32FloatStencil8
                    | GlFormat::Stencil8 => limits.max_depth_texture_samples,
                    _ => limits.max_color_texture_samples,
                },
            };
            if facts.sample_count > maximum {
                return Err(GlFormatTableError::SampleCountExceedsLimit {
                    format: facts.format,
                    sample_count: facts.sample_count,
                    maximum,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GlAstcBlock, GlCompressedColorSpace, GlCompressedSizeError, GlFamilyProfile, GlFormat,
        GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable,
        GlFormatTableError,
    };
    use crate::backend::gl::api::GlLimits;

    fn baseline(table: &mut GlFormatTable) {
        for format in [
            GlFormat::Rgba8Unorm,
            GlFormat::Rgba8Srgb,
            GlFormat::Depth32Float,
        ] {
            table
                .record(GlFormatCapabilities {
                    format,
                    resource_kind: GlFormatResourceKind::Texture,
                    sample_count: 1,
                    evidence: GlFormatEvidence::CoreGuaranteed,
                    sampled: true,
                    filterable: format != GlFormat::Depth32Float,
                    renderable: true,
                    blendable: format != GlFormat::Depth32Float,
                    storage_read: false,
                    storage_write: false,
                    copy_source: format != GlFormat::Depth32Float,
                    copy_destination: format != GlFormat::Depth32Float,
                })
                .expect("baseline fact");
        }
    }

    #[test]
    fn static_guarantees_cannot_claim_storage_images() {
        let mut table = GlFormatTable::default();
        let error = table
            .record(GlFormatCapabilities {
                format: GlFormat::Rgba8Unorm,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: true,
                renderable: true,
                blendable: true,
                storage_read: true,
                storage_write: true,
                copy_source: true,
                copy_destination: true,
            })
            .expect_err("storage requires an operation probe");
        assert!(matches!(
            error,
            GlFormatTableError::StorageRequiresOperationProbe { .. }
        ));
    }

    #[test]
    fn renderbuffer_multisampling_does_not_prove_texture_multisampling() {
        let mut table = GlFormatTable::default();
        baseline(&mut table);
        table
            .record(GlFormatCapabilities {
                format: GlFormat::Rgba8Unorm,
                resource_kind: GlFormatResourceKind::Renderbuffer,
                sample_count: 4,
                evidence: GlFormatEvidence::OperationProbed,
                sampled: false,
                filterable: false,
                renderable: true,
                blendable: true,
                storage_read: false,
                storage_write: false,
                copy_source: false,
                copy_destination: false,
            })
            .expect("renderbuffer operation probe");
        let mut limits = GlLimits::unavailable();
        limits.max_samples = 4;
        assert!(table.validate_for_limits(&limits).is_ok());
        assert!(
            table
                .get_for(GlFormatResourceKind::Texture, GlFormat::Rgba8Unorm, 4)
                .is_none()
        );
    }

    #[test]
    fn compressed_formats_preserve_exact_block_layout_and_color_space() {
        let bc1 = GlFormat::Bc1RgbaSrgb
            .compressed_info()
            .expect("BC1 is compressed");
        assert_eq!(
            (bc1.block_width, bc1.block_height, bc1.block_bytes),
            (4, 4, 8)
        );
        assert_eq!(bc1.color_space, GlCompressedColorSpace::Srgb);
        assert_eq!(
            GlFormat::Bc1RgbSrgb
                .compressed_info()
                .expect("sRGB BC1 RGB")
                .color_space,
            GlCompressedColorSpace::Srgb
        );
        let astc = GlFormat::Astc {
            block: GlAstcBlock::B10x6,
            color_space: GlCompressedColorSpace::Linear,
        }
        .compressed_info()
        .expect("ASTC is compressed");
        assert_eq!(
            (astc.block_width, astc.block_height, astc.block_bytes),
            (10, 6, 16)
        );
    }

    #[test]
    fn compressed_family_metadata_cannot_enable_a_different_exact_format() {
        let bc1 = GlFormat::Bc1RgbaUnorm.compressed_info();
        let bc3 = GlFormat::Bc3RgbaUnorm.compressed_info();
        assert_ne!(
            bc1, bc3,
            "one BC family bit cannot replace exact format facts"
        );
        let mut table = GlFormatTable::default();
        let error = table
            .record(GlFormatCapabilities {
                format: GlFormat::Etc2Rgb8Unorm,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::OperationProbed,
                sampled: true,
                filterable: true,
                renderable: true,
                blendable: false,
                storage_read: false,
                storage_write: false,
                copy_source: false,
                copy_destination: false,
            })
            .expect_err("compressed formats cannot become render targets");
        assert!(matches!(
            error,
            GlFormatTableError::InvalidCompressedUsage { .. }
        ));
    }

    #[test]
    fn gles3_core_guarantee_is_exactly_etc2_and_eac() {
        let gles3 = GlFamilyProfile::Embedded { major: 3, minor: 0 };
        assert!(GlFormat::Etc2Rgba8Srgb.is_core_compressed_for(gles3));
        assert!(GlFormat::EacRg11Unorm.is_core_compressed_for(gles3));
        assert!(GlFormat::Etc2Rgb8A1Srgb.is_core_compressed_for(gles3));
        assert!(GlFormat::EacR11Snorm.is_core_compressed_for(gles3));
        assert!(
            !GlFormat::Astc {
                block: GlAstcBlock::B4x4,
                color_space: GlCompressedColorSpace::Linear,
            }
            .is_core_compressed_for(gles3)
        );
    }

    #[test]
    fn compressed_encoded_sizes_use_exact_blocks_and_reject_zero_extents() {
        let bc3 = GlFormat::Bc3RgbaUnorm.compressed_info().expect("BC3 info");
        assert_eq!(bc3.checked_encoded_size(5, 5), Ok(64));
        assert_eq!(
            bc3.checked_encoded_size(0, 8),
            Err(GlCompressedSizeError::ZeroExtent)
        );
    }
}

/// Why a format fact cannot enter the durable table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlFormatTableError {
    /// Sample zero is not an actual GL sample count.
    ZeroSampleCount {
        /// Format with the invalid observation.
        format: GlFormat,
    },
    /// A compressed texture claimed an operation that this core contract forbids.
    InvalidCompressedUsage {
        /// Compressed format with an invalid resource or capability fact.
        format: GlFormat,
    },
    /// A static core claim is not guaranteed by this selected profile.
    InvalidCoreGuarantee {
        /// Format with an invalid static guarantee.
        format: GlFormat,
        /// Profile that does not guarantee it.
        profile: GlFamilyProfile,
    },
    /// The claimed extension is unavailable for the snapshot's profile.
    ExtensionIllegalForProfile {
        /// Format supported by the invalid extension claim.
        format: GlFormat,
        /// Extension from the format evidence.
        extension: GlKnownExtension,
        /// Snapshot profile.
        profile: GlFamilyProfile,
    },
    /// The bound extension ledger did not acquire the claimed extension.
    ExtensionNotAcquired {
        /// Format requiring the extension.
        format: GlFormat,
        /// Extension missing from the acquired ledger.
        extension: GlKnownExtension,
    },
    /// An acquired extension was used to claim an unrelated exact format.
    ExtensionDoesNotSupplyFormat {
        /// Claimed format.
        format: GlFormat,
        /// Acquired extension that cannot supply this format.
        extension: GlKnownExtension,
    },
    /// Two probes disagreed about the same exact pair.
    ConflictingObservation {
        /// Format with conflicting observations.
        format: GlFormat,
        /// Sample count with conflicting observations.
        sample_count: u32,
    },
    /// Storage image access is valid only after a concrete operation probe.
    StorageRequiresOperationProbe {
        /// Format with storage access facts.
        format: GlFormat,
        /// Resource class with storage access facts.
        resource_kind: GlFormatResourceKind,
        /// Exact sample count with storage access facts.
        sample_count: u32,
    },
    /// A common RHI format has no required single-sample fact.
    MissingRequiredSampleOne {
        /// Missing format.
        format: GlFormat,
    },
    /// An observed format/count pair exceeds the corresponding queried limit.
    SampleCountExceedsLimit {
        /// Format with an invalid count.
        format: GlFormat,
        /// Observed sample count.
        sample_count: u32,
        /// Applicable queried maximum.
        maximum: u32,
    },
}
