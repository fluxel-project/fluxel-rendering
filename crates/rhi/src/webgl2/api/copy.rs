//! Copy, upload, and readback contracts with reversible pixel-store state.

use super::{GlBufferRange, GlError, GlFamilyApi, GlTextureRegion};

/// The byte encoding supplied to or returned from a pixel transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlPixelFormat {
    Rgba8,
    Bgra8,
    Rgb8,
    Red8,
    Depth16,
    Depth24Stencil8,
    Depth32Float,
}

impl GlPixelFormat {
    pub(crate) const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgba8 | Self::Bgra8 | Self::Depth24Stencil8 | Self::Depth32Float => 4,
            Self::Rgb8 => 3,
            Self::Red8 => 1,
            Self::Depth16 => 2,
        }
    }
}

/// Whether a row stride which cannot be expressed by GL pixel-store is repacked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlRepackPolicy {
    Disallow,
    /// Allows an implementation-owned temporary staging allocation up to this exact bound.
    Bounded {
        max_bytes: u64,
    },
}

/// Explicit client-memory row layout. `bytes_per_row` includes caller-owned padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlPixelLayout {
    pub format: GlPixelFormat,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
    pub offset: u64,
    pub alignment: u8,
    pub repack: GlRepackPolicy,
}
impl GlPixelLayout {
    /// Whether GL's own pixel-store can express this layout's row stride.
    ///
    /// One question, asked by the validator, by both staging directions and by
    /// both pixel-store halves.  They have to answer it the same way, because
    /// the answer is what decides whether GL is describing the caller's buffer
    /// or a tightly packed staging one.
    pub(crate) fn stride_is_mappable(self) -> bool {
        self.bytes_per_row
            .is_multiple_of(self.format.bytes_per_pixel())
            && self.bytes_per_row.is_multiple_of(u32::from(self.alignment))
    }

    /// The `(ALIGNMENT, ROW_LENGTH)` pair GL is given for this layout.
    ///
    /// The mappable arm describes the caller's own buffer, which is the thing
    /// the payload is sliced from.  The other arm describes tightly staged
    /// rows: row length zero means "the sub-image's own width" and alignment
    /// one means nothing pads it.
    ///
    /// `PACK_*` and `UNPACK_*` take the same numbers for the same layout, so
    /// both directions read them from here instead of deciding again.  They had
    /// been decided twice, and the unpack half had drifted: it described the
    /// caller's pitch over a staging buffer that did not have that pitch, which
    /// is an out-of-bounds read on the native provider and a structured error
    /// on the browser one.
    pub(crate) fn staged_row_parameters(self) -> (u32, u32) {
        if self.stride_is_mappable() {
            (
                u32::from(self.alignment),
                self.bytes_per_row / self.format.bytes_per_pixel(),
            )
        } else {
            (1, 0)
        }
    }

    pub(crate) fn validate_for(self, region: GlTextureRegion) -> Result<(), GlCopyValidationError> {
        // First, because `required_bytes_unchecked` subtracts one from the
        // height and from the layer count: a zero extent would underflow there
        // -- panicking in debug, wrapping to `u32::MAX` in release -- and be
        // reported as a size problem rather than as the shape problem it is.
        // The guard has to live in this validator, because this is the one the
        // providers call; the `validate_for_basic` that carried it was
        // reachable from nowhere.
        if region.extent.width == 0
            || region.extent.height == 0
            || region.extent.depth_or_layers == 0
        {
            return Err(GlCopyValidationError::ZeroTransferExtent);
        }
        let packed = match region
            .extent
            .width
            .checked_mul(self.format.bytes_per_pixel())
        {
            Some(v) => v,
            None => return Err(GlCopyValidationError::LayoutOverflow),
        };
        if self.bytes_per_row < packed {
            return Err(GlCopyValidationError::RowTooShort);
        }
        if self.rows_per_image < region.extent.height {
            return Err(GlCopyValidationError::ImageTooShort);
        }
        if !matches!(self.alignment, 1 | 2 | 4 | 8) {
            return Err(GlCopyValidationError::InvalidPixelAlignment);
        }
        if !self.stride_is_mappable() {
            match self.repack {
                GlRepackPolicy::Disallow => return Err(GlCopyValidationError::UnmappableRowStride),
                GlRepackPolicy::Bounded { max_bytes }
                    if self.required_bytes_unchecked(region)? > max_bytes =>
                {
                    return Err(GlCopyValidationError::RepackBoundExceeded);
                }
                GlRepackPolicy::Bounded { .. } => {}
            }
        }
        Ok(())
    }
    pub(crate) fn required_bytes(
        self,
        region: GlTextureRegion,
    ) -> Result<u64, GlCopyValidationError> {
        self.validate_for(region)?;
        self.required_bytes_unchecked(region)
    }
    fn required_bytes_unchecked(
        self,
        region: GlTextureRegion,
    ) -> Result<u64, GlCopyValidationError> {
        let images = u64::from(region.extent.depth_or_layers - 1);
        let image_stride = u64::from(self.bytes_per_row)
            .checked_mul(u64::from(self.rows_per_image))
            .ok_or(GlCopyValidationError::LayoutOverflow)?;
        let final_rows = u64::from(region.extent.height - 1)
            .checked_mul(u64::from(self.bytes_per_row))
            .ok_or(GlCopyValidationError::LayoutOverflow)?;
        let final_width = u64::from(region.extent.width)
            .checked_mul(u64::from(self.format.bytes_per_pixel()))
            .ok_or(GlCopyValidationError::LayoutOverflow)?;
        self.offset
            .checked_add(
                images
                    .checked_mul(image_stride)
                    .ok_or(GlCopyValidationError::LayoutOverflow)?,
            )
            .and_then(|v| v.checked_add(final_rows))
            .and_then(|v| v.checked_add(final_width))
            .ok_or(GlCopyValidationError::LayoutOverflow)
    }
}

/// Complete GL pixel-store state that a scoped transfer must restore, even after driver error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlPixelStoreState {
    pub pack_alignment: u8,
    pub unpack_alignment: u8,
    pub pack_row_length: u32,
    pub unpack_row_length: u32,
    pub pack_skip_pixels: u32,
    pub pack_skip_rows: u32,
    pub pack_image_height: u32,
    pub pack_skip_images: u32,
    pub unpack_skip_pixels: u32,
    pub unpack_skip_rows: u32,
    pub unpack_image_height: u32,
    pub unpack_skip_images: u32,
}
impl GlPixelStoreState {
    pub(crate) const DEFAULT: Self = Self {
        pack_alignment: 4,
        unpack_alignment: 4,
        pack_row_length: 0,
        unpack_row_length: 0,
        pack_skip_pixels: 0,
        pack_skip_rows: 0,
        pack_image_height: 0,
        pack_skip_images: 0,
        unpack_skip_pixels: 0,
        unpack_skip_rows: 0,
        unpack_image_height: 0,
        unpack_skip_images: 0,
    };
    pub(crate) fn validate(self) -> Result<(), GlCopyValidationError> {
        if !matches!(self.pack_alignment, 1 | 2 | 4 | 8)
            || !matches!(self.unpack_alignment, 1 | 2 | 4 | 8)
        {
            return Err(GlCopyValidationError::InvalidPixelAlignment);
        }
        Ok(())
    }
    /// Validates image-height settings against a concrete transfer before mutation.
    pub(crate) fn validate_for_transfer(
        self,
        region: GlTextureRegion,
    ) -> Result<(), GlCopyValidationError> {
        self.validate()?;
        if (self.pack_image_height != 0 && self.pack_image_height < region.extent.height)
            || (self.unpack_image_height != 0 && self.unpack_image_height < region.extent.height)
        {
            return Err(GlCopyValidationError::ImageHeightTooShort);
        }
        Ok(())
    }
}

/// A successful readback always identifies the layout used for the returned bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlReadback {
    pub layout: GlPixelLayout,
    pub bytes: Vec<u8>,
}
impl GlReadback {
    /// Ensures a successful provider result is neither truncated nor silently overlong.
    pub(crate) fn validate_for(
        &self,
        source: GlTextureRegion,
    ) -> Result<(), GlCopyValidationError> {
        let required = self.layout.required_bytes(source)?;
        if u64::try_from(self.bytes.len()).ok() != Some(required) {
            return Err(GlCopyValidationError::ReadbackLengthMismatch);
        }
        Ok(())
    }
}

/// Copy validation failures which must occur before a GL command or pixel-store mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlCopyValidationError {
    RowTooShort,
    ImageTooShort,
    LayoutOverflow,
    InvalidPixelAlignment,
    UploadSourceTooShort,
    UnmappableRowStride,
    RepackBoundExceeded,
    ReadbackLengthMismatch,
    ZeroTransferExtent,
    ImageHeightTooShort,
}

/// Copy operations. Pixel-store changes are scoped: implementations snapshot `pixel_store`,
/// apply a transfer layout, and restore the exact snapshot before every return path.
///
/// Buffer binding points are deliberately *not* scoped. `COPY_READ_BUFFER` and
/// `COPY_WRITE_BUFFER` are private to the transfer implementation: a provider
/// binds the buffer it is about to use immediately before the command and
/// leaves the target bound on return. Nothing needs restoring because no verb
/// in this layer accepts a caller-supplied target, so neither the Layer 2 state
/// mirror nor a caller can hold an opinion about these targets — the mirror
/// covers only binding points some verb can name. Adding a restore here would
/// imply the opposite contract; adding a target-taking verb would break it.
/// `scripts/check_gl_architecture.py` confines the constants to provider
/// execution bodies so the two cannot drift apart.
pub(crate) trait GlCopyDomainApi: GlFamilyApi {
    fn copy_buffer_range(
        &mut self,
        source: GlBufferRange,
        destination: GlBufferRange,
    ) -> Result<(), GlError>;
    fn copy_texture_region(
        &mut self,
        source: GlTextureRegion,
        destination: GlTextureRegion,
    ) -> Result<(), GlError>;
    /// Uploads exactly `range.size` bytes at `range.offset` in one live buffer.
    fn upload_buffer(&mut self, destination: GlBufferRange, bytes: &[u8]) -> Result<(), GlError>;
    /// Reads back exactly `range.size` bytes from one live buffer.
    fn read_buffer(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError>;
    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError>;
    fn read_texture(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError>;
    fn pixel_store(&self) -> GlPixelStoreState;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readback_size_includes_last_image_only_once() {
        let context = super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        );
        let region = GlTextureRegion {
            subresource: super::super::GlTextureSubresource {
                texture: super::super::TextureId::new(context, 0, 0),
                aspect: super::super::GlTextureAspect::Color,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            origin: [0; 3],
            extent: super::super::GlExtent3d {
                width: 2,
                height: 2,
                depth_or_layers: 2,
            },
        };
        let layout = GlPixelLayout {
            format: GlPixelFormat::Rgba8,
            bytes_per_row: 8,
            rows_per_image: 2,
            offset: 0,
            alignment: 4,
            repack: GlRepackPolicy::Disallow,
        };
        assert_eq!(layout.required_bytes(region), Ok(32));
    }
    #[test]
    fn pixel_store_alignments_are_checked() {
        let mut state = GlPixelStoreState::DEFAULT;
        state.pack_alignment = 3;
        assert_eq!(
            state.validate(),
            Err(GlCopyValidationError::InvalidPixelAlignment)
        );
    }
    #[test]
    fn unmappable_rows_need_an_explicit_bounded_repack() {
        let region = test_region();
        let layout = GlPixelLayout {
            format: GlPixelFormat::Rgba8,
            bytes_per_row: 9,
            rows_per_image: 2,
            offset: 0,
            alignment: 4,
            repack: GlRepackPolicy::Disallow,
        };
        assert_eq!(
            layout.validate_for(region),
            Err(GlCopyValidationError::UnmappableRowStride)
        );
    }
    /// The layout the two directions disagreed about: `bytes_per_row` is a
    /// multiple of the pixel size but not of the declared alignment, so GL
    /// cannot express the stride and the payload is staged tightly.  The
    /// parameters have to describe the staging rather than the caller's buffer,
    /// because describing the caller's buffer walks GL a full `bytes_per_row`
    /// per row through a staging allocation that only holds `width * bpp`.
    #[test]
    fn an_unexpressible_stride_is_described_as_tight_not_as_the_callers() {
        let layout = GlPixelLayout {
            format: GlPixelFormat::Rgba8,
            bytes_per_row: 12,
            rows_per_image: 2,
            offset: 0,
            alignment: 8,
            repack: GlRepackPolicy::Bounded { max_bytes: 64 },
        };
        assert!(!layout.stride_is_mappable());
        assert_eq!(layout.staged_row_parameters(), (1, 0));
    }

    #[test]
    fn an_expressible_stride_is_described_as_the_callers_own_pitch() {
        let layout = GlPixelLayout {
            format: GlPixelFormat::Rgba8,
            bytes_per_row: 16,
            rows_per_image: 2,
            offset: 0,
            alignment: 4,
            repack: GlRepackPolicy::Disallow,
        };
        assert!(layout.stride_is_mappable());
        assert_eq!(layout.staged_row_parameters(), (4, 4));
    }

    /// The invariant the two tests above are instances of, so a later edit to
    /// either arm has to keep it: a stride GL cannot express never carries a
    /// row length, and an expressible one always does.
    #[test]
    fn the_row_length_is_present_exactly_when_the_stride_is_mappable() {
        for bytes_per_row in [3u32, 4, 5, 8, 12, 16, 17, 24] {
            for alignment in [1u8, 2, 4, 8] {
                let layout = GlPixelLayout {
                    format: GlPixelFormat::Rgba8,
                    bytes_per_row,
                    rows_per_image: 2,
                    offset: 0,
                    alignment,
                    repack: GlRepackPolicy::Bounded { max_bytes: 4_096 },
                };
                let (told_alignment, row_length) = layout.staged_row_parameters();
                assert_eq!(row_length != 0, layout.stride_is_mappable(), "{layout:?}");
                // Tight staging is alignment one, which is what makes a row
                // exactly `width * bpp` with nothing padding it.
                if row_length == 0 {
                    assert_eq!(told_alignment, 1, "{layout:?}");
                }
            }
        }
    }

    /// A zero extent used to reach `required_bytes_unchecked`, which subtracts
    /// one from the height and from the layer count: a panic in a debug build,
    /// a `u32::MAX` wrap reported as a size error in a release one.  The guard
    /// that existed for it was in a function nothing called.
    #[test]
    fn a_zero_extent_is_refused_as_a_shape_rather_than_underflowing() {
        let region = GlTextureRegion {
            subresource: test_region().subresource,
            origin: [0; 3],
            extent: super::super::GlExtent3d {
                width: 2,
                height: 0,
                depth_or_layers: 1,
            },
        };
        let layout = GlPixelLayout {
            format: GlPixelFormat::Rgba8,
            bytes_per_row: 8,
            rows_per_image: 0,
            offset: 0,
            alignment: 4,
            repack: GlRepackPolicy::Bounded { max_bytes: 4_096 },
        };
        assert_eq!(
            layout.validate_for(region),
            Err(GlCopyValidationError::ZeroTransferExtent)
        );
        // `required_bytes` validates first, so the subtraction it would
        // otherwise do on this height is never reached.
        assert_eq!(
            layout.required_bytes(region),
            Err(GlCopyValidationError::ZeroTransferExtent)
        );
    }

    #[test]
    fn readback_requires_the_exact_returned_length() {
        let region = test_region();
        let layout = GlPixelLayout {
            format: GlPixelFormat::Rgba8,
            bytes_per_row: 8,
            rows_per_image: 2,
            offset: 0,
            alignment: 4,
            repack: GlRepackPolicy::Disallow,
        };
        assert_eq!(
            GlReadback {
                layout,
                bytes: vec![0; 31]
            }
            .validate_for(region),
            Err(GlCopyValidationError::ReadbackLengthMismatch)
        );
    }
    fn test_region() -> GlTextureRegion {
        let context = super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        );
        GlTextureRegion {
            subresource: super::super::GlTextureSubresource {
                texture: super::super::TextureId::new(context, 0, 0),
                aspect: super::super::GlTextureAspect::Color,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            origin: [0; 3],
            extent: super::super::GlExtent3d {
                width: 2,
                height: 2,
                depth_or_layers: 1,
            },
        }
    }
}
