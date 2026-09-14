//! Platform-neutral resource descriptions and validation.

use super::{BufferId, GlFormat, TextureId};

/// Buffer operations permitted for a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBufferUsage(u32);

impl GlBufferUsage {
    pub(crate) const COPY_SOURCE: Self = Self(1 << 0);
    pub(crate) const COPY_DESTINATION: Self = Self(1 << 1);
    pub(crate) const VERTEX: Self = Self(1 << 2);
    pub(crate) const INDEX: Self = Self(1 << 3);
    pub(crate) const UNIFORM: Self = Self(1 << 4);
    pub(crate) const STORAGE: Self = Self(1 << 5);
    pub(crate) const INDIRECT: Self = Self(1 << 6);
    pub(crate) const MAP_READ: Self = Self(1 << 7);
    pub(crate) const MAP_WRITE: Self = Self(1 << 8);
    pub(crate) const EMPTY: Self = Self(0);

    pub(crate) const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for GlBufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Immutable creation facts for a buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBufferDesc {
    pub size: u64,
    pub usage: GlBufferUsage,
}

impl GlBufferDesc {
    pub(crate) fn validate(self) -> Result<(), GlResourceValidationError> {
        if self.size == 0 {
            return Err(GlResourceValidationError::ZeroBufferSize);
        }
        if self.usage.is_empty() {
            return Err(GlResourceValidationError::EmptyBufferUsage);
        }
        Ok(())
    }
}

/// A byte range in one buffer.  `size` is deliberately never implicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBufferRange {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
}

impl GlBufferRange {
    pub(crate) fn validate_for(self, desc: GlBufferDesc) -> Result<(), GlResourceValidationError> {
        if self.size == 0 {
            return Err(GlResourceValidationError::ZeroRangeSize);
        }
        match self.offset.checked_add(self.size) {
            Some(end) if end <= desc.size => Ok(()),
            _ => Err(GlResourceValidationError::BufferRangeOutOfBounds),
        }
    }
}

/// Texture dimensionality accepted by the common GL-family layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlTextureDimension {
    D1,
    D2,
    D3,
    Cube,
    D2Array,
}

/// Texture operations permitted for a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureUsage(u32);
impl GlTextureUsage {
    pub(crate) const COPY_SOURCE: Self = Self(1 << 0);
    pub(crate) const COPY_DESTINATION: Self = Self(1 << 1);
    pub(crate) const SAMPLED: Self = Self(1 << 2);
    pub(crate) const RENDER_ATTACHMENT: Self = Self(1 << 3);
    pub(crate) const STORAGE_BINDING: Self = Self(1 << 4);
    pub(crate) const EMPTY: Self = Self(0);
    pub(crate) const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}
impl core::ops::BitOr for GlTextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Pixel extent.  The depth component is layers for arrays and depth for 3D textures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlExtent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_layers: u32,
}

/// Immutable creation facts for a texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureDesc {
    pub dimension: GlTextureDimension,
    pub extent: GlExtent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub format: GlFormat,
    pub usage: GlTextureUsage,
}

impl GlTextureDesc {
    pub(crate) fn validate(self) -> Result<(), GlResourceValidationError> {
        if self.extent.width == 0 || self.extent.height == 0 || self.extent.depth_or_layers == 0 {
            return Err(GlResourceValidationError::ZeroTextureExtent);
        }
        if self.mip_level_count == 0 {
            return Err(GlResourceValidationError::ZeroMipLevelCount);
        }
        if self.sample_count == 0 {
            return Err(GlResourceValidationError::ZeroSampleCount);
        }
        if self.usage.is_empty() {
            return Err(GlResourceValidationError::EmptyTextureUsage);
        }
        if self.sample_count > 1 && self.mip_level_count != 1 {
            return Err(GlResourceValidationError::MultisampleMipmapped);
        }
        match self.dimension {
            GlTextureDimension::D1
                if self.extent.height != 1 || self.extent.depth_or_layers != 1 =>
            {
                return Err(GlResourceValidationError::InvalidDimensionExtent);
            }
            GlTextureDimension::D2 if self.extent.depth_or_layers != 1 => {
                return Err(GlResourceValidationError::InvalidDimensionExtent);
            }
            GlTextureDimension::Cube
                if self.extent.width != self.extent.height || self.extent.depth_or_layers != 6 =>
            {
                return Err(GlResourceValidationError::InvalidCubeExtent);
            }
            _ => {}
        }
        if self.sample_count > 1
            && (self.dimension != GlTextureDimension::D2 || self.extent.depth_or_layers != 1)
        {
            return Err(GlResourceValidationError::MultisampleRequiresD2);
        }
        if self.mip_level_count > self.maximum_mip_level_count() {
            return Err(GlResourceValidationError::MipLevelCountOutOfBounds);
        }
        Ok(())
    }
    pub(crate) const fn maximum_mip_level_count(self) -> u32 {
        let mut largest = self.extent.width;
        if self.extent.height > largest {
            largest = self.extent.height;
        }
        if let GlTextureDimension::D3 = self.dimension {
            if self.extent.depth_or_layers > largest {
                largest = self.extent.depth_or_layers;
            }
        }
        32 - largest.leading_zeros()
    }
    pub(crate) const fn mip_extent(self, mip_level: u32) -> Option<GlExtent3d> {
        if mip_level >= self.mip_level_count {
            return None;
        }
        let width = mip_dimension(self.extent.width, mip_level);
        let height = mip_dimension(self.extent.height, mip_level);
        let depth_or_layers = match self.dimension {
            GlTextureDimension::D3 => mip_dimension(self.extent.depth_or_layers, mip_level),
            _ => self.extent.depth_or_layers,
        };
        Some(GlExtent3d {
            width,
            height,
            depth_or_layers,
        })
    }
}

/// Returns a nonzero mip dimension without relying on post-MSRT const methods.
const fn mip_dimension(value: u32, level: u32) -> u32 {
    if level >= u32::BITS {
        1
    } else {
        let shifted = value >> level;
        if shifted == 0 { 1 } else { shifted }
    }
}

/// A color, depth, or stencil plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlTextureAspect {
    All,
    Color,
    DepthOnly,
    StencilOnly,
}

/// One mip and contiguous layer range of a texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureSubresource {
    pub texture: TextureId,
    pub aspect: GlTextureAspect,
    pub mip_level: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl GlTextureSubresource {
    pub(crate) fn validate_for(self, desc: GlTextureDesc) -> Result<(), GlResourceValidationError> {
        if self.layer_count == 0 {
            return Err(GlResourceValidationError::ZeroLayerCount);
        }
        let Some(extent) = desc.mip_extent(self.mip_level) else {
            return Err(GlResourceValidationError::MipLevelOutOfBounds);
        };
        if desc.dimension == GlTextureDimension::D3
            && (self.base_layer != 0 || self.layer_count != 1)
        {
            return Err(GlResourceValidationError::ThreeDimensionalSubresourceLayers);
        }
        if self
            .base_layer
            .checked_add(self.layer_count)
            .is_none_or(|end| end > extent.depth_or_layers)
        {
            return Err(GlResourceValidationError::LayerRangeOutOfBounds);
        }
        Ok(())
    }
}

/// A rectangular region within one validated subresource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlTextureRegion {
    pub subresource: GlTextureSubresource,
    pub origin: [u32; 3],
    pub extent: GlExtent3d,
}
impl GlTextureRegion {
    pub(crate) fn validate_for(self, desc: GlTextureDesc) -> Result<(), GlResourceValidationError> {
        self.subresource.validate_for(desc)?;
        if self.extent.width == 0 || self.extent.height == 0 || self.extent.depth_or_layers == 0 {
            return Err(GlResourceValidationError::ZeroCopyExtent);
        }
        let Some(bound) = desc.mip_extent(self.subresource.mip_level) else {
            return Err(GlResourceValidationError::MipLevelOutOfBounds);
        };
        if self.origin[0]
            .checked_add(self.extent.width)
            .is_none_or(|end| end > bound.width)
            || self.origin[1]
                .checked_add(self.extent.height)
                .is_none_or(|end| end > bound.height)
        {
            return Err(GlResourceValidationError::TextureRegionOutOfBounds);
        }
        let maximum_depth = if desc.dimension == GlTextureDimension::D3 {
            bound.depth_or_layers
        } else {
            self.subresource.layer_count
        };
        if self.origin[2]
            .checked_add(self.extent.depth_or_layers)
            .is_none_or(|end| end > maximum_depth)
        {
            return Err(GlResourceValidationError::SubresourceRegionOutOfBounds);
        }
        Ok(())
    }
}

/// Validates an exact texture copy before a provider changes bindings or issues GL.
pub(crate) fn validate_texture_copy(
    source: GlTextureRegion,
    source_desc: GlTextureDesc,
    destination: GlTextureRegion,
    destination_desc: GlTextureDesc,
) -> Result<(), GlResourceValidationError> {
    source.validate_for(source_desc)?;
    destination.validate_for(destination_desc)?;
    if source_desc.format != destination_desc.format {
        return Err(GlResourceValidationError::IncompatibleCopyFormat);
    }
    if source_desc.sample_count != destination_desc.sample_count {
        return Err(GlResourceValidationError::IncompatibleCopySampleCount);
    }
    if source.subresource.aspect != destination.subresource.aspect {
        return Err(GlResourceValidationError::IncompatibleCopyAspect);
    }
    if source.extent != destination.extent {
        return Err(GlResourceValidationError::IncompatibleCopyExtent);
    }
    Ok(())
}

/// Validation failures which providers must report before issuing a GL call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlResourceValidationError {
    ZeroBufferSize,
    EmptyBufferUsage,
    ZeroRangeSize,
    BufferRangeOutOfBounds,
    ZeroTextureExtent,
    ZeroMipLevelCount,
    ZeroSampleCount,
    EmptyTextureUsage,
    MultisampleMipmapped,
    MultisampleRequiresD2,
    InvalidDimensionExtent,
    InvalidCubeExtent,
    MipLevelCountOutOfBounds,
    ZeroLayerCount,
    MipLevelOutOfBounds,
    LayerRangeOutOfBounds,
    ZeroCopyExtent,
    TextureRegionOutOfBounds,
    SubresourceRegionOutOfBounds,
    ThreeDimensionalSubresourceLayers,
    IncompatibleCopyFormat,
    IncompatibleCopySampleCount,
    IncompatibleCopyAspect,
    IncompatibleCopyExtent,
}

/// Resource allocation and destruction domain.
pub(crate) trait GlResourceApi: super::GlFamilyApi {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, super::GlError>;
    fn create_texture_resource(&mut self, desc: GlTextureDesc)
    -> Result<TextureId, super::GlError>;
    fn destroy_buffer_resource(&mut self, buffer: BufferId) -> Result<(), super::GlError>;
    fn destroy_texture_resource(&mut self, texture: TextureId) -> Result<(), super::GlError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_overflowing_buffer_ranges_before_use() {
        let d = GlBufferDesc {
            size: 8,
            usage: GlBufferUsage::COPY_SOURCE,
        };
        assert_eq!(
            GlBufferDesc {
                size: 0,
                usage: GlBufferUsage::EMPTY
            }
            .validate(),
            Err(GlResourceValidationError::ZeroBufferSize)
        );
        let id = BufferId::new(
            super::super::ContextStamp::new(
                super::super::DeviceIdentity::new(1).unwrap(),
                super::super::ContextEpoch::INITIAL,
            ),
            0,
            0,
        );
        assert_eq!(
            GlBufferRange {
                buffer: id,
                offset: u64::MAX,
                size: 1
            }
            .validate_for(d),
            Err(GlResourceValidationError::BufferRangeOutOfBounds)
        );
    }
    #[test]
    fn multisample_textures_cannot_have_mips() {
        let desc = GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 4,
                height: 4,
                depth_or_layers: 1,
            },
            mip_level_count: 2,
            sample_count: 4,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::SAMPLED,
        };
        assert_eq!(
            desc.validate(),
            Err(GlResourceValidationError::MultisampleMipmapped)
        );
    }
    #[test]
    fn mip_count_and_multisample_dimension_are_bounded() {
        let mut desc = GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 4,
                height: 4,
                depth_or_layers: 1,
            },
            mip_level_count: 4,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::SAMPLED,
        };
        assert_eq!(
            desc.validate(),
            Err(GlResourceValidationError::MipLevelCountOutOfBounds)
        );
        desc.mip_level_count = 1;
        desc.dimension = GlTextureDimension::D3;
        desc.sample_count = 4;
        assert_eq!(
            desc.validate(),
            Err(GlResourceValidationError::MultisampleRequiresD2)
        );
    }
}
