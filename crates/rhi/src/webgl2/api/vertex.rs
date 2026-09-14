//! Platform-neutral vertex-input and index-input vocabulary.

use super::{BufferId, GlError, GlFamilyApi, VertexArrayId};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlVertexFormat {
    Uint8x2,
    Uint8x4,
    Sint8x2,
    Sint8x4,
    Unorm8x2,
    Unorm8x4,
    Snorm8x2,
    Snorm8x4,
    Uint16x2,
    Uint16x4,
    Sint16x2,
    Sint16x4,
    Unorm16x2,
    Unorm16x4,
    Snorm16x2,
    Snorm16x4,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlVertexStepMode {
    Vertex,
    Instance,
}

/// One buffer slot in a VAO.  Stride zero is intentionally representable for
/// providers which validate it according to the selected GL profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlVertexBufferLayout {
    pub slot: u32,
    pub stride: u32,
    pub step_mode: GlVertexStepMode,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlVertexAttribute {
    pub location: u32,
    pub buffer_slot: u32,
    pub format: GlVertexFormat,
    pub offset: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlVertexLayout {
    pub buffers: Vec<GlVertexBufferLayout>,
    pub attributes: Vec<GlVertexAttribute>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlIndexFormat {
    Uint16,
    Uint32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlIndexBinding {
    pub buffer: BufferId,
    pub format: GlIndexFormat,
    pub offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlVertexBufferBinding {
    pub slot: u32,
    pub buffer: BufferId,
    pub offset: u64,
}

/// Allocation facts supplied by the resource domain before a binding is made.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlVertexBufferMetadata {
    pub buffer: BufferId,
    pub byte_length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlVertexValidationError {
    DuplicateBufferSlot,
    DuplicateAttributeLocation,
    AttributeBufferMissing,
    AttributeExceedsStride,
    BindingSlotMissing,
    DuplicateBindingSlot,
    ForeignBuffer,
    /// The caller supplied no allocation facts for a bound buffer at all.
    BindingMetadataMissing,
    BindingOffsetExceedsBuffer,
    IndexOffsetMisaligned,
    IndexOffsetExceedsBuffer,
}

impl GlVertexFormat {
    const fn byte_size(self) -> u32 {
        match self {
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Uint8x4 | Self::Sint8x4 | Self::Unorm8x4 | Self::Snorm8x4 => 4,
            Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Float16x2 => 4,
            Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4
            | Self::Float16x4 => 8,
            Self::Float32 | Self::Uint32 | Self::Sint32 => 4,
            Self::Float32x2 | Self::Uint32x2 | Self::Sint32x2 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
        }
    }
}

impl GlVertexLayout {
    pub(crate) fn validate(&self) -> Result<(), GlVertexValidationError> {
        let mut slots = BTreeSet::new();
        for buffer in &self.buffers {
            if !slots.insert(buffer.slot) {
                return Err(GlVertexValidationError::DuplicateBufferSlot);
            }
        }
        let mut locations = BTreeSet::new();
        for attribute in &self.attributes {
            if !locations.insert(attribute.location) {
                return Err(GlVertexValidationError::DuplicateAttributeLocation);
            }
            let Some(buffer) = self
                .buffers
                .iter()
                .find(|buffer| buffer.slot == attribute.buffer_slot)
            else {
                return Err(GlVertexValidationError::AttributeBufferMissing);
            };
            let size = attribute.format.byte_size();
            let stride = if buffer.stride == 0 {
                size
            } else {
                buffer.stride
            };
            if attribute
                .offset
                .checked_add(size)
                .is_none_or(|end| end > stride)
            {
                return Err(GlVertexValidationError::AttributeExceedsStride);
            }
        }
        Ok(())
    }
    pub(crate) fn validate_bindings(
        &self,
        bindings: &[GlVertexBufferBinding],
        index: Option<GlIndexBinding>,
        metadata: &[GlVertexBufferMetadata],
        current: super::ContextStamp,
    ) -> Result<(), GlVertexValidationError> {
        self.validate()?;
        let mut bound_slots = BTreeSet::new();
        for binding in bindings {
            if !bound_slots.insert(binding.slot) {
                return Err(GlVertexValidationError::DuplicateBindingSlot);
            }
            if !self
                .buffers
                .iter()
                .any(|layout| layout.slot == binding.slot)
            {
                return Err(GlVertexValidationError::BindingSlotMissing);
            }
            if binding.buffer.context != current {
                return Err(GlVertexValidationError::ForeignBuffer);
            }
            let Some(facts) = metadata.iter().find(|facts| facts.buffer == binding.buffer) else {
                return Err(GlVertexValidationError::BindingMetadataMissing);
            };
            if binding.offset > facts.byte_length {
                return Err(GlVertexValidationError::BindingOffsetExceedsBuffer);
            }
        }
        if self
            .buffers
            .iter()
            .any(|layout| !bound_slots.contains(&layout.slot))
        {
            return Err(GlVertexValidationError::BindingSlotMissing);
        }
        if let Some(index) = index {
            let alignment = match index.format {
                GlIndexFormat::Uint16 => 2,
                GlIndexFormat::Uint32 => 4,
            };
            if index.offset % alignment != 0 {
                return Err(GlVertexValidationError::IndexOffsetMisaligned);
            }
            if index.buffer.context != current {
                return Err(GlVertexValidationError::ForeignBuffer);
            }
            if metadata
                .iter()
                .find(|facts| facts.buffer == index.buffer)
                .is_none_or(|facts| index.offset > facts.byte_length)
            {
                return Err(GlVertexValidationError::IndexOffsetExceedsBuffer);
            }
        }
        Ok(())
    }
}

/// VAO creation and binding domain.  The higher layer supplies complete
/// structural bindings, preventing implicit global-VAO state from escaping.
pub(crate) trait GlVertexApi: GlFamilyApi {
    fn create_vertex_array(&mut self, layout: &GlVertexLayout) -> Result<VertexArrayId, GlError>;
    fn destroy_vertex_array(&mut self, vertex_array: VertexArrayId) -> Result<(), GlError>;
    fn bind_vertex_array(
        &mut self,
        vertex_array: VertexArrayId,
        buffers: &[GlVertexBufferBinding],
        index: Option<GlIndexBinding>,
    ) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attribute_is_hashable_structural_state() {
        let a = GlVertexAttribute {
            location: 2,
            buffer_slot: 1,
            format: GlVertexFormat::Float32x3,
            offset: 12,
        };
        assert_eq!(a, a);
    }
    #[test]
    fn rejects_duplicate_buffer_slots() {
        let layout = GlVertexLayout {
            buffers: vec![
                GlVertexBufferLayout {
                    slot: 0,
                    stride: 12,
                    step_mode: GlVertexStepMode::Vertex,
                },
                GlVertexBufferLayout {
                    slot: 0,
                    stride: 12,
                    step_mode: GlVertexStepMode::Vertex,
                },
            ],
            attributes: vec![],
        };
        assert_eq!(
            layout.validate(),
            Err(GlVertexValidationError::DuplicateBufferSlot)
        );
    }
    #[test]
    fn missing_binding_metadata_is_distinct_from_an_exceeded_offset() {
        let stamp = super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        );
        let buffer = BufferId::new(stamp, 0, 0);
        let layout = GlVertexLayout {
            buffers: vec![GlVertexBufferLayout {
                slot: 0,
                stride: 12,
                step_mode: GlVertexStepMode::Vertex,
            }],
            attributes: vec![],
        };
        let binding = GlVertexBufferBinding {
            slot: 0,
            buffer,
            offset: 0,
        };
        assert_eq!(
            layout.validate_bindings(&[binding], None, &[], stamp),
            Err(GlVertexValidationError::BindingMetadataMissing)
        );
        let beyond = GlVertexBufferBinding {
            slot: 0,
            buffer,
            offset: 8,
        };
        assert_eq!(
            layout.validate_bindings(
                &[beyond],
                None,
                &[GlVertexBufferMetadata {
                    buffer,
                    byte_length: 4,
                }],
                stamp,
            ),
            Err(GlVertexValidationError::BindingOffsetExceedsBuffer)
        );
    }
}
