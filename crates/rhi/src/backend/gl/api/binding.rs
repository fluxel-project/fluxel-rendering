//! Per-unit texture, sampler, and indexed uniform-buffer binding vocabulary.
//!
//! These are the command words the Layer 2 state groups mirror. Rebinding the
//! identical slot with identical arguments is deliberately legal; redundancy
//! elimination belongs to the state layer, never to this contract.

use super::{BufferId, GlError, GlFamilyApi, SamplerId, TextureId};

/// A texture target that can hold one binding at one texture unit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlTextureTarget {
    D2,
    Cube,
    D3,
    D2Array,
}

/// Binding failures which must be rejected before any driver or browser call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlBindingValidationError {
    TextureUnitOutOfBounds,
    UniformIndexOutOfBounds,
    UniformOffsetMisaligned,
    UniformOffsetExceedsAllocation,
    UniformRangeOutOfBounds,
    /// An unbind carries no range; nonzero offset or size has no GL meaning.
    UniformUnbindCarriesRange,
}

impl GlBindingValidationError {
    /// The stable validation wording providers report without reformatting.
    pub(crate) const fn message(self) -> &'static str {
        match self {
            Self::TextureUnitOutOfBounds => "texture unit exceeds the discovered unit count",
            Self::UniformIndexOutOfBounds => "uniform binding index exceeds the discovered count",
            Self::UniformOffsetMisaligned => {
                "uniform binding offset misses the discovered alignment"
            }
            Self::UniformOffsetExceedsAllocation => {
                "uniform binding offset is past the buffer allocation"
            }
            Self::UniformRangeOutOfBounds => "uniform binding range leaves the buffer allocation",
            Self::UniformUnbindCarriesRange => "unbind must not carry an offset or size",
        }
    }
}

/// The discovered limits one binding validation needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBindingLimits {
    /// `MAX_COMBINED_TEXTURE_IMAGE_UNITS`.
    pub max_texture_units: u32,
    /// `MAX_UNIFORM_BUFFER_BINDINGS`.
    pub max_uniform_buffer_bindings: u32,
    /// `UNIFORM_BUFFER_OFFSET_ALIGNMENT`.
    pub uniform_buffer_offset_alignment: u64,
}

/// Validates that `unit` addresses an existing texture unit.
pub(crate) fn validate_texture_unit(
    unit: u32,
    limits: GlBindingLimits,
) -> Result<(), GlBindingValidationError> {
    if unit >= limits.max_texture_units {
        return Err(GlBindingValidationError::TextureUnitOutOfBounds);
    }
    Ok(())
}

/// Validates the indexed slot, the unbind shape, and the offset alignment.
///
/// `size == 0` means "to the end of the buffer", so the range itself is
/// checked against the recorded allocation by [`validate_uniform_range`].
pub(crate) fn validate_uniform_buffer_binding(
    index: u32,
    buffer: Option<BufferId>,
    offset: u32,
    size: u32,
    limits: GlBindingLimits,
) -> Result<(), GlBindingValidationError> {
    if index >= limits.max_uniform_buffer_bindings {
        return Err(GlBindingValidationError::UniformIndexOutOfBounds);
    }
    if buffer.is_none() {
        if offset != 0 || size != 0 {
            return Err(GlBindingValidationError::UniformUnbindCarriesRange);
        }
        return Ok(());
    }
    if limits.uniform_buffer_offset_alignment == 0
        || u64::from(offset) % limits.uniform_buffer_offset_alignment != 0
    {
        return Err(GlBindingValidationError::UniformOffsetMisaligned);
    }
    Ok(())
}

/// Validates a resolved uniform range against the recorded allocation size.
///
/// `size == 0` resolves to `allocation - offset`, which must keep at least one
/// byte: an offset at or past the allocation binds nothing in either form.
pub(crate) fn validate_uniform_range(
    offset: u32,
    size: u32,
    allocation: u64,
) -> Result<(), GlBindingValidationError> {
    if u64::from(offset) >= allocation {
        return Err(GlBindingValidationError::UniformOffsetExceedsAllocation);
    }
    let end = match size {
        0 => u64::from(offset),
        size => u64::from(offset)
            .checked_add(u64::from(size))
            .ok_or(GlBindingValidationError::UniformRangeOutOfBounds)?,
    };
    if end > allocation {
        return Err(GlBindingValidationError::UniformRangeOutOfBounds);
    }
    Ok(())
}

/// Per-unit texture/sampler and indexed uniform-buffer binding domain.
///
/// This is a common-profile domain: every GL 4.x, GLES 3.x, and WebGL2 context
/// owns texture units and indexed uniform bindings, so implementations which
/// execute the raster/copy domains also execute this one.
pub(crate) trait GlBindingApi: GlFamilyApi {
    /// Selects the texture unit subsequent per-unit work targets.
    fn active_texture(&mut self, unit: u32) -> Result<(), GlError>;
    /// Binds or unbinds one texture at one unit and target.
    fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) -> Result<(), GlError>;
    /// Binds or unbinds one sampler object at one unit.
    fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) -> Result<(), GlError>;
    /// Binds a byte range of one buffer at one indexed uniform binding point.
    ///
    /// `size == 0` binds through the end of the recorded allocation; `None`
    /// unbinds the slot and must not carry a range.
    fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    ) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{
        GlBindingLimits, GlBindingValidationError, validate_texture_unit,
        validate_uniform_buffer_binding, validate_uniform_range,
    };
    use crate::backend::gl::api::{BufferId, ContextEpoch, ContextStamp, DeviceIdentity};

    fn limits() -> GlBindingLimits {
        GlBindingLimits {
            max_texture_units: 16,
            max_uniform_buffer_bindings: 24,
            uniform_buffer_offset_alignment: 256,
        }
    }

    fn buffer() -> BufferId {
        BufferId::new(
            ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL),
            0,
            0,
        )
    }

    #[test]
    fn texture_units_are_bounded_by_the_discovered_count() {
        assert_eq!(validate_texture_unit(15, limits()), Ok(()));
        assert_eq!(
            validate_texture_unit(16, limits()),
            Err(GlBindingValidationError::TextureUnitOutOfBounds)
        );
    }

    #[test]
    fn uniform_slots_and_offsets_are_checked_before_ranges() {
        let bound = buffer();
        assert_eq!(
            validate_uniform_buffer_binding(24, Some(bound), 0, 16, limits()),
            Err(GlBindingValidationError::UniformIndexOutOfBounds)
        );
        assert_eq!(
            validate_uniform_buffer_binding(0, Some(bound), 3, 16, limits()),
            Err(GlBindingValidationError::UniformOffsetMisaligned)
        );
        assert_eq!(
            validate_uniform_buffer_binding(0, None, 0, 256, limits()),
            Err(GlBindingValidationError::UniformUnbindCarriesRange)
        );
        assert_eq!(
            validate_uniform_buffer_binding(0, None, 0, 0, limits()),
            Ok(())
        );
    }

    #[test]
    fn uniform_ranges_resolve_size_zero_to_the_allocation_end() {
        assert_eq!(validate_uniform_range(0, 0, 512), Ok(()));
        assert_eq!(validate_uniform_range(256, 256, 512), Ok(()));
        assert_eq!(
            validate_uniform_range(512, 0, 512),
            Err(GlBindingValidationError::UniformOffsetExceedsAllocation)
        );
        assert_eq!(
            validate_uniform_range(0, 513, 512),
            Err(GlBindingValidationError::UniformRangeOutOfBounds)
        );
    }
}
