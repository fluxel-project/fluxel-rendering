//! Route facts: whether a portable operation has a legal direct backend path
//! (specification sections 9.1 through 9.4).
//!
//! The chapter answers exactly one question, and section 9 writes it as a
//! warning because the question is easy to answer wrongly:
//!
//! ```text
//! asked:      does this operation have a legal direct RHI route?
//! NOT asked:  can the backend secretly replace it with a shader or a
//!             CPU round-trip?
//! ```
//!
//! # The fallback rule, which is a contract rather than a policy
//!
//! Section 9.4 freezes the answer, and it is the reason this module is not
//! optional machinery:
//!
//! ```text
//! RouteSupport::Unsupported  =>  the command returns Unsupported
//! ```
//!
//! A backend may not, without the caller knowing, lower a blit into a fullscreen
//! shader, a copy into a staging CPU round-trip, or a resolve into a compute
//! shader. If a caller wants a fallback, the caller queries the route and
//! selects another explicit graph pass or command route. The payoff is that
//! capture, statistics, and the performance model describe what was actually
//! executed instead of what was nominally requested — which is exactly what a
//! silent fallback would corrupt.
//!
//! # Why the key carries shape
//!
//! Section 9.1 puts texture dimension, format, and aspect in the key, and
//! sample counts for the two texture-to-texture routes, because otherwise the
//! gap it names opens: a route that answers `Supported` for a key that does not
//! include those facts will be asked to execute a descriptor it cannot, and the
//! failure lands on a driver instead of on portable validation.
//!
//! Buffer-to-texture and texture-to-buffer routes are single-sample only in P0;
//! that is enforced jointly by the texture query (a multisampled texture cannot
//! be a copy destination for these routes) and command validation.
//!
//! # Where the blit key's filter comes from
//!
//! Section 9.1's [`RouteQuery::Blit`] carries `filter: BlitFilter`, and that type
//! is owned by section 34.5 in the recording chapter
//! ([`crate::api::command::BlitFilter`]) rather than re-declared here: the filter
//! is what a *blit command* asks for, and this module only names it in a key. The
//! import is one-way — `route` reads `command`, never the reverse — so a caller
//! can build a blit key without a device and the two chapters cannot drift into
//! two spellings of `Linear`.

use crate::api::command::BlitFilter;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::resource::subresource::TextureAspect;
use crate::api::resource::texture::TextureDimension;

/// A portable copy/transfer operation whose legality is being asked about.
///
/// Every variant carries the shape facts that can change native legality, for
/// the reason given in the module documentation. The type is a value: a caller
/// builds a key, asks the device, and may keep it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RouteQuery {
    /// Buffer to buffer copy.
    BufferToBuffer,

    /// Buffer to texture copy.
    BufferToTexture {
        /// Dimensionality of the destination texture.
        dimension: TextureDimension,
        /// Format of the destination texture.
        format: TextureFormat,
        /// Aspect of the destination texture.
        aspect: TextureAspect,
    },

    /// Texture to buffer copy.
    TextureToBuffer {
        /// Dimensionality of the source texture.
        dimension: TextureDimension,
        /// Format of the source texture.
        format: TextureFormat,
        /// Aspect of the source texture.
        aspect: TextureAspect,
    },

    /// Texture to texture copy.
    TextureToTexture {
        /// Dimensionality of the source texture.
        src_dimension: TextureDimension,
        /// Format of the source texture.
        src_format: TextureFormat,
        /// Aspect of the source texture.
        src_aspect: TextureAspect,
        /// Sample count of the source texture.
        src_sample_count: u32,

        /// Dimensionality of the destination texture.
        dst_dimension: TextureDimension,
        /// Format of the destination texture.
        dst_format: TextureFormat,
        /// Aspect of the destination texture.
        dst_aspect: TextureAspect,
        /// Sample count of the destination texture.
        dst_sample_count: u32,
    },

    /// Multisampled to single-sampled resolve.
    Resolve {
        /// Format of both the source and the destination.
        format: TextureFormat,
        /// Sample count of the multisampled source.
        src_sample_count: u32,
    },

    /// Texture to texture blit, with the filter it asks for.
    ///
    /// The aspect is absent where the other texture routes carry one, because
    /// section 34.5 freezes P0 direct blit to `Color` on both sides: an aspect in
    /// the key could only ever take one value, and a key field that cannot vary
    /// suggests a question the route does not answer. Sample counts are absent for
    /// the same reason — P0 blit is single-sampled on both sides.
    Blit {
        /// Dimensionality of the source texture.
        src_dimension: TextureDimension,
        /// Format of the source texture.
        src_format: TextureFormat,

        /// Dimensionality of the destination texture.
        dst_dimension: TextureDimension,
        /// Format of the destination texture.
        dst_format: TextureFormat,

        /// The filter the blit asks for.
        ///
        /// Part of the key rather than a separate query because section 34.5 makes
        /// a filtered blit a different native operation: a route that supports a
        /// nearest blit is not thereby stating it supports a linear one, so the
        /// two must be different keys.
        filter: BlitFilter,
    },
}

/// The alignment a buffer-to-buffer copy must respect.
///
/// Returned by [`RouteSupport::capabilities`] for
/// [`RouteQuery::BufferToBuffer`]. The two numbers are the native copy's
/// requirements, not a portable convention: a backend that reports `Some` of
/// these is stating what its copy path accepts, and a backend that reports
/// `None` for the buffer-copy layout is stating that this route is not a
/// buffer copy at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferCopyLayoutLimits {
    offset_alignment: u64,
    size_alignment: u64,
}

impl BufferCopyLayoutLimits {
    /// Records one device's copy alignment.
    ///
    /// Crate-private: the numbers are a probed device answer, and a
    /// caller-built pair would describe hardware that was never asked.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn new(offset_alignment: u64, size_alignment: u64) -> Self {
        Self {
            offset_alignment,
            size_alignment,
        }
    }

    /// The byte alignment a copy's offset must satisfy.
    pub fn offset_alignment(&self) -> u64 {
        self.offset_alignment
    }

    /// The byte alignment a copy's size must satisfy.
    pub fn size_alignment(&self) -> u64 {
        self.size_alignment
    }

    /// Checks a copy's offset and size against this device's alignment.
    ///
    /// Section 12.4's "corresponding binding/copy alignment" rule, resolved for
    /// the one route that has alignment to resolve it against.
    pub(crate) fn validate(&self, offset: u64, size: u64) -> RhiResult<()> {
        if !is_aligned(offset, self.offset_alignment) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "copy offset {offset} is not aligned to {} bytes",
                    self.offset_alignment
                ),
            ));
        }
        if !is_aligned(size, self.size_alignment) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "copy size {size} is not aligned to {} bytes",
                    self.size_alignment
                ),
            ));
        }
        Ok(())
    }
}

/// The alignment a buffer-to-texture (or texture-to-buffer) copy must respect.
///
/// Opaque, and deliberately not a `HostTexelLayout` with more fields. Section
/// 14.5 draws the line between them: this describes the legality and alignment
/// of the *GPU-side* buffer-texture route, while a host layout describes the
/// *caller's* CPU bytes. An upload is free to repack one into the other, which
/// is why the two types are never interchangeable and why the texel row pitch a
/// caller must not be asked to meet is absent from the host type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelCopyLayoutLimits {
    buffer_offset_alignment: u64,
    bytes_per_row_alignment: u32,
}

impl TexelCopyLayoutLimits {
    /// Records one device's texel-copy alignment.
    ///
    /// Crate-private: the numbers are a probed device answer, and a
    /// caller-built pair would describe hardware that was never asked.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn new(buffer_offset_alignment: u64, bytes_per_row_alignment: u32) -> Self {
        Self {
            buffer_offset_alignment,
            bytes_per_row_alignment,
        }
    }

    /// The byte alignment the copy buffer's offset must satisfy.
    pub fn buffer_offset_alignment(&self) -> u64 {
        self.buffer_offset_alignment
    }

    /// The byte alignment the copy buffer's row pitch must satisfy.
    pub fn bytes_per_row_alignment(&self) -> u32 {
        self.bytes_per_row_alignment
    }

    /// Checks a GPU-side copy's offset and row pitch against this device's
    /// alignment.
    ///
    /// The counterpart of [`BufferCopyLayoutLimits::validate`] for the
    /// buffer-texture route. Section 9.2 notes that `rows_per_image` adds no
    /// alignment field of its own: its legality follows from the extent, the
    /// format's block geometry, and the route rules.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the buffer-texture copy route's validation calls this"
        )
    )]
    pub(crate) fn validate(&self, buffer_offset: u64, bytes_per_row: u32) -> RhiResult<()> {
        if !is_aligned(buffer_offset, self.buffer_offset_alignment) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "copy buffer offset {buffer_offset} is not aligned to {} bytes",
                    self.buffer_offset_alignment
                ),
            ));
        }
        if !is_aligned(
            u64::from(bytes_per_row),
            u64::from(self.bytes_per_row_alignment),
        ) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "copy row pitch {bytes_per_row} is not aligned to {} bytes",
                    self.bytes_per_row_alignment
                ),
            ));
        }
        Ok(())
    }
}

/// What a supported route can do.
///
/// Which layouts are present is itself information, exactly as section 9.3's
/// examples show: a buffer-to-buffer route reports a buffer copy layout, a
/// buffer-texture route reports a texel copy layout, and a route that reports
/// neither is one that has no copy alignment to state.
#[derive(Clone, Copy, Debug)]
pub struct RouteCapabilities {
    buffer_copy_layout: Option<BufferCopyLayoutLimits>,
    texel_copy_layout: Option<TexelCopyLayoutLimits>,
}

impl RouteCapabilities {
    /// Records the layouts a supported route can honour.
    ///
    /// Crate-private: these are probed device facts, and a caller-built answer
    /// would be a capability claim about hardware nobody asked.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn new(
        buffer_copy_layout: Option<BufferCopyLayoutLimits>,
        texel_copy_layout: Option<TexelCopyLayoutLimits>,
    ) -> Self {
        Self {
            buffer_copy_layout,
            texel_copy_layout,
        }
    }

    /// The buffer-copy alignment, when this route is a buffer copy.
    pub fn buffer_copy_layout(&self) -> Option<BufferCopyLayoutLimits> {
        self.buffer_copy_layout
    }

    /// The texel-copy alignment, when this route transfers texels.
    pub fn texel_copy_layout(&self) -> Option<TexelCopyLayoutLimits> {
        self.texel_copy_layout
    }
}

/// Whether a route described by a [`RouteQuery`] exists on this device.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum RouteSupport {
    /// No direct route exists, and section 9.4 forbids inventing one.
    Unsupported,

    /// A direct route exists, with the returned capabilities.
    Supported(RouteCapabilities),
}

impl RouteSupport {
    /// Whether the route exists.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// The route's capabilities, or `None` when the route does not exist.
    pub fn capabilities(&self) -> Option<&RouteCapabilities> {
        match self {
            Self::Unsupported => None,
            Self::Supported(capabilities) => Some(capabilities),
        }
    }
}

/// Whether a value satisfies an alignment.
///
/// A zero alignment is not a real device answer, so it is treated as "no
/// constraint" rather than as a `% 0` panic: a malformed capability must not be
/// able to turn a validation call into an abort, and a device that reports no
/// alignment is reporting that it has none to impose.
fn is_aligned(value: u64, alignment: u64) -> bool {
    alignment == 0 || value.is_multiple_of(alignment)
}

// ---------------------------------------------------------------------------
// Canonical capability encoding
// ---------------------------------------------------------------------------
//
// The rules of the encoding, and what it is for, are stated once in
// `api::capability::CapabilityFacts`. It lives here because every field read
// below is private to this module.

impl RouteQuery {
    /// Writes this query's canonical bytes: a tag, then the variant's fields in
    /// declaration order.
    ///
    /// Every field is written, including the ones a given route kind might seem to
    /// make redundant. Two keys that differ in any field are two different
    /// questions, and a route that a device supports for one of them is not
    /// thereby supported for the other — which is precisely what the id must be
    /// able to tell apart.
    ///
    /// No wildcard arm: a seventh route kind must state its encoding before this
    /// compiles, the same way `binding_class` has no wildcard so an unclassified
    /// kind cannot slip through.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::BufferToBuffer => out.push(0),
            Self::BufferToTexture {
                dimension,
                format,
                aspect,
            } => {
                out.push(1);
                dimension.encode_into(out);
                format.encode_into(out);
                aspect.encode_into(out);
            }
            Self::TextureToBuffer {
                dimension,
                format,
                aspect,
            } => {
                out.push(2);
                dimension.encode_into(out);
                format.encode_into(out);
                aspect.encode_into(out);
            }
            Self::TextureToTexture {
                src_dimension,
                src_format,
                src_aspect,
                src_sample_count,
                dst_dimension,
                dst_format,
                dst_aspect,
                dst_sample_count,
            } => {
                out.push(3);
                src_dimension.encode_into(out);
                src_format.encode_into(out);
                src_aspect.encode_into(out);
                out.extend_from_slice(&src_sample_count.to_le_bytes());
                dst_dimension.encode_into(out);
                dst_format.encode_into(out);
                dst_aspect.encode_into(out);
                out.extend_from_slice(&dst_sample_count.to_le_bytes());
            }
            Self::Resolve {
                format,
                src_sample_count,
            } => {
                out.push(4);
                format.encode_into(out);
                out.extend_from_slice(&src_sample_count.to_le_bytes());
            }
            Self::Blit {
                src_dimension,
                src_format,
                dst_dimension,
                dst_format,
                filter,
            } => {
                out.push(5);
                src_dimension.encode_into(out);
                src_format.encode_into(out);
                dst_dimension.encode_into(out);
                dst_format.encode_into(out);
                filter.encode_into(out);
            }
        }
    }
}

impl RouteSupport {
    /// Writes this answer as a tag, followed by the capabilities when there are
    /// any.
    ///
    /// Which layouts are present is part of what a supported route says — and
    /// their values are part of it too, since an alignment is a fact a caller
    /// obeys. A route that reports a texel-copy layout and one that reports none
    /// are different contracts, so an absent layout is tagged rather than encoded
    /// as a zero alignment, which is a value [`is_aligned`] deliberately reads as
    /// "no constraint" and would therefore have conflated the two.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::Unsupported => out.push(0),
            Self::Supported(capabilities) => {
                out.push(1);
                match capabilities.buffer_copy_layout {
                    None => out.push(0),
                    Some(layout) => {
                        out.push(1);
                        out.extend_from_slice(&layout.offset_alignment.to_le_bytes());
                        out.extend_from_slice(&layout.size_alignment.to_le_bytes());
                    }
                }
                match capabilities.texel_copy_layout {
                    None => out.push(0),
                    Some(layout) => {
                        out.push(1);
                        out.extend_from_slice(&layout.buffer_offset_alignment.to_le_bytes());
                        out.extend_from_slice(&layout.bytes_per_row_alignment.to_le_bytes());
                    }
                }
            }
        }
    }
}
