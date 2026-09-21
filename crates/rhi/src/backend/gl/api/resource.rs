//! Platform-neutral resource descriptions and validation.
//!
//! This module owns the descriptors for buffers, textures, texture views and
//! renderbuffers, the rules a descriptor must satisfy before any provider
//! allocates storage for it, and the range and subresource arithmetic those
//! rules are expressed in. It does not own allocation or object lifetime (that
//! is the provider) and it does not own the context facts a rule is checked
//! against: a provider supplies its discovered limits and format evidence, and
//! this module states the rule and the structured reason a violated one
//! produces.
//!
//! # Compressed mip chains
//!
//! A recorded contract decision, written here because the rule this module does
//! *not* enforce has to be a statement rather than silence.
//!
//! What is enforced: a compressed upload must define exactly one complete 2D
//! mip, and the byte count it supplies must equal that mip's exact encoded size
//! (`GlCompressedFormatInfo::checked_encoded_size`) rather than anything the
//! client asserts. Compressed storage stays undefined until something defines
//! it, so that rule is what makes a compressed upload mean anything; it lives
//! with the recorder's upload path and it is closed.
//!
//! What is deliberately not enforced: nothing requires every mip of a
//! `mip_level_count > 1` chain to be defined. A descriptor may therefore
//! allocate a whole chain while the caller fills in only some of its levels, and
//! no layer rejects that -- not `GlTextureDesc::validate`, which checks that the
//! requested count is legal for the extent and the format but never that the
//! chain is complete, and not any upload, which sees one subresource and cannot
//! see the chain it belongs to. A chain can stay permanently partial without the
//! descriptor ever being rejected.
//!
//! Why a partial chain is the caller's problem at this layer: completeness is a
//! distribution-of-uploads fact about an allocation's history, not a property of
//! any single command, so enforcing it needs per-texture defined-mip state.
//! Nothing needs that state yet -- reading a texture is a Layer 2/3 concern, and
//! this layer's contract is the command vocabulary plus per-command validation.
//! The tracking could not live in this module in any case: a descriptor here is
//! a value that is copied, compared and hashed, not a record of what has
//! happened to an allocation since it was created. Adding lifetime-scoped state
//! now would be an API built for a future need rather than for a current one.
//!
//! What would have to change to revisit it: a real consumer that needs a
//! complete chain before a read -- a Layer 2/3 path that samples a texture whose
//! levels the caller may only partly have uploaded. Chain completeness would
//! then be tracked where the lifetime already lives (the provider's object
//! store, keyed by texture identity and invalidated with the context epoch),
//! no descriptor field would change, and the rejection would belong at the first
//! command that reads the texture rather than at allocation, because allocation
//! is not the point at which a partial chain becomes wrong.

use super::{BufferId, GlFormat, RenderbufferId, TextureId};

/// Buffer operations permitted for a resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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

/// One binding role a buffer can be created for.
///
/// The roles are named after what the GPU does with the bytes. They exist so a
/// descriptor can be rejected *before* any driver object is created, and so the
/// rejection can say which two roles collided instead of reporting a generic
/// invalid descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlBufferRole {
    /// Attribute source.
    Vertex,
    /// Element source.
    Index,
    /// Uniform block source.
    Uniform,
    /// Shader storage block source, where the profile has one.
    Storage,
    /// Draw/dispatch parameter source, where the profile has one.
    Indirect,
    /// Source of a buffer-to-buffer or buffer-to-texture copy.
    CopySource,
    /// Destination of a buffer-to-buffer or texture-to-buffer copy.
    CopyDestination,
    /// CPU read mapping: a readback staging buffer.
    MapRead,
    /// CPU write mapping: an upload staging buffer.
    MapWrite,
}

impl GlBufferRole {
    /// Every role, in a fixed order so a rejection is deterministic.
    pub(crate) const ALL: [Self; 9] = [
        Self::MapRead,
        Self::MapWrite,
        Self::Vertex,
        Self::Index,
        Self::Uniform,
        Self::Storage,
        Self::Indirect,
        Self::CopySource,
        Self::CopyDestination,
    ];

    pub(crate) const fn usage(self) -> GlBufferUsage {
        match self {
            Self::Vertex => GlBufferUsage::VERTEX,
            Self::Index => GlBufferUsage::INDEX,
            Self::Uniform => GlBufferUsage::UNIFORM,
            Self::Storage => GlBufferUsage::STORAGE,
            Self::Indirect => GlBufferUsage::INDIRECT,
            Self::CopySource => GlBufferUsage::COPY_SOURCE,
            Self::CopyDestination => GlBufferUsage::COPY_DESTINATION,
            Self::MapRead => GlBufferUsage::MAP_READ,
            Self::MapWrite => GlBufferUsage::MAP_WRITE,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Index => "index",
            Self::Uniform => "uniform",
            Self::Storage => "storage",
            Self::Indirect => "indirect",
            Self::CopySource => "copy-source",
            Self::CopyDestination => "copy-destination",
            Self::MapRead => "map-read",
            Self::MapWrite => "map-write",
        }
    }

    /// Whether the role describes a CPU mapping rather than a GPU use.
    const fn is_mapping(self) -> bool {
        matches!(self, Self::MapRead | Self::MapWrite)
    }

    /// Whether the GPU reads the buffer through a binding point as a resource.
    const fn is_gpu_resource(self) -> bool {
        matches!(
            self,
            Self::Vertex | Self::Index | Self::Uniform | Self::Storage | Self::Indirect
        )
    }
}

/// Whether two roles cannot be declared on one buffer.
///
/// GL stores an untyped byte range, so one buffer legitimately serves several
/// consumption roles at once: vertex, index, uniform, copy-source, and
/// copy-destination may be combined freely, and storage and indirect join them
/// wherever the profile has those domains. Rejecting those pairings here would
/// invent a restriction neither the GL family nor the WebGPU-aligned model has;
/// they are validated where they are used instead, by requiring the descriptor
/// to declare the role being bound.
///
/// Mapping is the exception, and the only reason this table exists. A mapped
/// buffer is a staging buffer: its contents are well defined only while the CPU
/// holds the mapping, and the GL family gives no coherence between a mapping
/// and a concurrent GPU read of the same bytes. So a mapping role conflicts
/// with the other mapping role and with every role that lets the GPU consume
/// the buffer as a resource. Copy roles stay legal next to a mapping -- that is
/// exactly the upload and readback staging pattern.
const fn roles_conflict(first: GlBufferRole, second: GlBufferRole) -> bool {
    if first.is_mapping() {
        second.is_mapping() || second.is_gpu_resource()
    } else if second.is_mapping() {
        first.is_gpu_resource()
    } else {
        false
    }
}

impl GlBufferUsage {
    /// The first conflicting role pair this usage set declares, if any.
    ///
    /// Roles are examined in [`GlBufferRole::ALL`] order, so the same usage set
    /// always reports the same pair.
    pub(crate) const fn conflicting_roles(self) -> Option<(GlBufferRole, GlBufferRole)> {
        let mut i = 0;
        while i < GlBufferRole::ALL.len() {
            let first = GlBufferRole::ALL[i];
            if self.contains(first.usage()) {
                let mut j = i + 1;
                while j < GlBufferRole::ALL.len() {
                    let second = GlBufferRole::ALL[j];
                    if self.contains(second.usage()) && roles_conflict(first, second) {
                        return Some((first, second));
                    }
                    j += 1;
                }
            }
            i += 1;
        }
        None
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
        if let Some((first, second)) = self.usage.conflicting_roles() {
            return Err(GlResourceValidationError::ConflictingBufferRoles { first, second });
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

/// Immutable creation facts for a renderbuffer.
///
/// Renderbuffers are the GL-family's render-only two-dimensional attachment
/// storage; they are never sampled and never mipmapped, so the descriptor is
/// deliberately narrower than [`GlTextureDesc`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlRenderBufferDesc {
    pub format: GlFormat,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
}

impl GlRenderBufferDesc {
    pub(crate) fn validate(self) -> Result<(), GlResourceValidationError> {
        if self.width == 0 || self.height == 0 {
            return Err(GlResourceValidationError::ZeroRenderBufferExtent);
        }
        if self.samples == 0 {
            return Err(GlResourceValidationError::ZeroSampleCount);
        }
        Ok(())
    }
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
    if let Some(block) = source_desc.format.compressed_info() {
        let source_mip = source_desc
            .mip_extent(source.subresource.mip_level)
            .ok_or(GlResourceValidationError::MipLevelOutOfBounds)?;
        let destination_mip = destination_desc
            .mip_extent(destination.subresource.mip_level)
            .ok_or(GlResourceValidationError::MipLevelOutOfBounds)?;
        let aligned = |origin: u32, extent: u32, full: u32, block: u8| {
            origin % u32::from(block) == 0
                && (extent % u32::from(block) == 0 || origin.saturating_add(extent) == full)
        };
        if !aligned(
            source.origin[0],
            source.extent.width,
            source_mip.width,
            block.block_width,
        ) || !aligned(
            source.origin[1],
            source.extent.height,
            source_mip.height,
            block.block_height,
        ) || !aligned(
            destination.origin[0],
            destination.extent.width,
            destination_mip.width,
            block.block_width,
        ) || !aligned(
            destination.origin[1],
            destination.extent.height,
            destination_mip.height,
            block.block_height,
        ) {
            return Err(GlResourceValidationError::CompressedCopyBlockMisaligned);
        }
    }
    Ok(())
}

/// Validation failures which providers must report before issuing a GL call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlResourceValidationError {
    ZeroBufferSize,
    EmptyBufferUsage,
    /// Two usage roles that cannot share one buffer; see [`roles_conflict`].
    ConflictingBufferRoles {
        first: GlBufferRole,
        second: GlBufferRole,
    },
    ZeroRangeSize,
    BufferRangeOutOfBounds,
    ZeroTextureExtent,
    ZeroRenderBufferExtent,
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
    /// A compressed copy starts or ends inside a codec block. GL image copies
    /// transfer encoded blocks, so interior edges must be block-aligned.
    CompressedCopyBlockMisaligned,
}

impl GlResourceValidationError {
    /// A description for the provider's structured error, naming the actual
    /// cause.
    ///
    /// Providers report validation through `GlError::Validation { message }`.
    /// Deriving that message here keeps the offending roles visible to the
    /// caller instead of collapsing every descriptor failure into one generic
    /// string.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::ConflictingBufferRoles { first, second } => format!(
                "buffer usage combines conflicting roles `{}` and `{}`",
                first.name(),
                second.name()
            ),
            other => format!("{other:?}"),
        }
    }
}

/// Resource allocation and destruction domain.
pub(crate) trait GlResourceApi: super::GlFamilyApi {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, super::GlError>;
    fn create_texture_resource(&mut self, desc: GlTextureDesc)
    -> Result<TextureId, super::GlError>;
    fn create_render_buffer(
        &mut self,
        desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, super::GlError>;
    fn destroy_buffer_resource(&mut self, buffer: BufferId) -> Result<(), super::GlError>;
    fn destroy_texture_resource(&mut self, texture: TextureId) -> Result<(), super::GlError>;
    fn destroy_render_buffer(
        &mut self,
        render_buffer: RenderbufferId,
    ) -> Result<(), super::GlError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(usage: GlBufferUsage) -> GlBufferDesc {
        GlBufferDesc { size: 64, usage }
    }

    #[test]
    fn consumption_roles_combine_because_the_gl_family_stores_untyped_bytes() {
        // Vertex, index, uniform, copy-source, and copy-destination share one
        // buffer legally; the layer validates the declared role at each bind
        // site instead of forbidding the combination here.
        let usage = GlBufferUsage::VERTEX
            | GlBufferUsage::INDEX
            | GlBufferUsage::UNIFORM
            | GlBufferUsage::COPY_SOURCE
            | GlBufferUsage::COPY_DESTINATION;
        assert_eq!(buffer(usage).validate(), Ok(()));
        assert_eq!(usage.conflicting_roles(), None);
    }

    #[test]
    fn staging_roles_stay_legal_next_to_a_mapping() {
        // Upload staging and readback staging are the two patterns that need a
        // mapping beside a copy role.
        assert_eq!(
            buffer(GlBufferUsage::MAP_WRITE | GlBufferUsage::COPY_SOURCE).validate(),
            Ok(())
        );
        assert_eq!(
            buffer(GlBufferUsage::MAP_READ | GlBufferUsage::COPY_DESTINATION).validate(),
            Ok(())
        );
    }

    #[test]
    fn mapping_roles_are_rejected_beside_each_other_and_beside_gpu_use() {
        let read_write = GlBufferUsage::MAP_READ | GlBufferUsage::MAP_WRITE;
        assert_eq!(
            buffer(read_write).validate(),
            Err(GlResourceValidationError::ConflictingBufferRoles {
                first: GlBufferRole::MapRead,
                second: GlBufferRole::MapWrite,
            })
        );
        for role in [
            GlBufferRole::Vertex,
            GlBufferRole::Index,
            GlBufferRole::Uniform,
            GlBufferRole::Storage,
            GlBufferRole::Indirect,
        ] {
            for mapping in [GlBufferRole::MapRead, GlBufferRole::MapWrite] {
                let usage = mapping.usage() | role.usage();
                assert_eq!(
                    buffer(usage).validate(),
                    Err(GlResourceValidationError::ConflictingBufferRoles {
                        first: mapping,
                        second: role,
                    }),
                    "{mapping:?} beside {role:?} must be rejected"
                );
            }
        }
    }

    #[test]
    fn the_reported_conflict_pair_is_deterministic_and_named() {
        // `ALL` order decides which pair is reported when several conflicts
        // exist, so the same descriptor always produces the same error.
        let error =
            buffer(GlBufferUsage::VERTEX | GlBufferUsage::MAP_WRITE | GlBufferUsage::MAP_READ)
                .validate()
                .expect_err("three-way conflict is rejected");
        assert_eq!(
            error,
            GlResourceValidationError::ConflictingBufferRoles {
                first: GlBufferRole::MapRead,
                second: GlBufferRole::MapWrite,
            }
        );
        assert_eq!(
            error.message(),
            "buffer usage combines conflicting roles `map-read` and `map-write`"
        );
    }

    #[test]
    fn every_role_reports_a_distinct_usage_bit() {
        let mut seen = GlBufferUsage::EMPTY;
        for role in GlBufferRole::ALL {
            assert!(
                !seen.contains(role.usage()),
                "{role:?} reuses another role's usage bit"
            );
            seen = seen | role.usage();
        }
    }

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
    fn renderbuffer_descriptors_reject_zero_extent_and_samples() {
        let base = GlRenderBufferDesc {
            format: GlFormat::Rgba8Unorm,
            width: 4,
            height: 4,
            samples: 4,
        };
        assert_eq!(base.validate(), Ok(()));
        assert_eq!(
            GlRenderBufferDesc { width: 0, ..base }.validate(),
            Err(GlResourceValidationError::ZeroRenderBufferExtent)
        );
        assert_eq!(
            GlRenderBufferDesc { samples: 0, ..base }.validate(),
            Err(GlResourceValidationError::ZeroSampleCount)
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
