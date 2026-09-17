//! The vertex-input vocabulary a raster pipeline is created from.
//!
//! This vocabulary is deliberately **closed**, for the same reason ADR-0006 and
//! ADR-0007 keep the raster recipes closed: a general vertex-format surface would
//! let a caller describe a stream no retained artifact declares, and nothing above
//! could say what such a pipeline means. The formats here are exactly the ones the
//! fixed artifacts bind, and a new one arrives only with the artifact that needs
//! it.
//!
//! The shape mirrors the GL family's `GlVertexBufferLayout` / `GlVertexAttribute`
//! pair on purpose, because lead 3F converges that family onto this contract: a
//! layout that is already stated this way in one implementation is the one the
//! other four can adopt without a translation step.

/// The element format of one vertex attribute.
///
/// Each variant is a tightly packed element of the named shape, which is what the
/// artifacts assume and what makes `stride` a caller-supplied number rather than a
/// derived one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VertexFormat {
    /// Two 32-bit floats: the UV stream.
    Float32x2,
    /// Three 32-bit floats: the position and normal streams.
    Float32x3,
    /// Four eight-bit unsigned normalized values: the vertex-colour stream.
    ///
    /// Normalized means the stored integers map to `[0, 1]` for unsigned values,
    /// which is why this format carries colour rather than a raw integer type.
    Unorm8x4,
}

impl VertexFormat {
    /// Returns the tightly packed size of one element in bytes.
    pub(crate) const fn size(self) -> u32 {
        match self {
            Self::Float32x2 => 8,
            Self::Float32x3 => 12,
            Self::Unorm8x4 => 4,
        }
    }
}

/// How a vertex stream advances between draws and instances.
///
/// `Instance` is present because the GL family already models it and the
/// capability ledger carries the rows that would gate it; no retained artifact
/// currently declares an instanced stream, so today every closed recipe uses
/// [`Self::Vertex`]. The variant exists so that adding such an artifact is a
/// recipe change rather than a vocabulary change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VertexStepMode {
    /// The stream advances once per vertex.
    Vertex,
    /// The stream advances once per instance.
    Instance,
}

/// One vertex buffer bound to one input slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct VertexBufferLayout {
    /// The input slot this buffer occupies.
    pub slot: u32,
    /// The byte distance between consecutive elements.
    pub stride: u32,
    /// Whether the stream advances per vertex or per instance.
    pub step_mode: VertexStepMode,
}

/// One attribute read out of a vertex buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct VertexAttribute {
    /// The shader input location this attribute feeds.
    pub location: u32,
    /// The input slot the attribute reads from.
    pub buffer_slot: u32,
    /// The element format at this location.
    pub format: VertexFormat,
    /// The byte offset of the attribute inside its element.
    pub offset: u32,
}

/// A complete vertex input description: buffers and the attributes that read them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VertexLayout {
    /// The buffers, in slot order.
    pub buffers: Vec<VertexBufferLayout>,
    /// The attributes, in location order.
    pub attributes: Vec<VertexAttribute>,
}

/// Why a vertex layout cannot describe a pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VertexLayoutError {
    /// An attribute reads from a buffer slot the layout does not declare.
    ///
    /// Also covers a layout with no attributes at all, which would describe a
    /// pipeline that reads no vertex data.
    UnknownBufferSlot {
        /// The attribute's location.
        location: u32,
        /// The slot it named.
        slot: u32,
    },
    /// Two attributes feed the same shader input location.
    DuplicateLocation {
        /// The repeated location.
        location: u32,
    },
    /// One attribute's bytes do not fit inside its element stride.
    ///
    /// An attribute that overruns its stride would read the next element's bytes
    /// on every vertex but the last, which is a silent wrong result rather than a
    /// driver error.
    AttributeExceedsStride {
        /// The attribute's location.
        location: u32,
        /// The attribute's end offset in bytes.
        end: u32,
        /// The buffer's stride.
        stride: u32,
    },
    /// A buffer declares no stride, so its elements have no extent.
    ZeroStride {
        /// The slot with the zero stride.
        slot: u32,
    },
}

impl VertexLayout {
    /// Rejects a layout that cannot describe one pipeline.
    ///
    /// This is the whole validation the closed artifacts need, and it is
    /// deliberately local: it checks the layout against itself and never against a
    /// device, because a limit or capability question belongs to the ledger.
    pub(crate) fn validate(&self) -> Result<(), VertexLayoutError> {
        for buffer in &self.buffers {
            if buffer.stride == 0 {
                return Err(VertexLayoutError::ZeroStride { slot: buffer.slot });
            }
        }
        let mut seen = Vec::with_capacity(self.attributes.len());
        for attribute in &self.attributes {
            if seen.contains(&attribute.location) {
                return Err(VertexLayoutError::DuplicateLocation {
                    location: attribute.location,
                });
            }
            seen.push(attribute.location);
            let Some(buffer) = self
                .buffers
                .iter()
                .find(|buffer| buffer.slot == attribute.buffer_slot)
            else {
                return Err(VertexLayoutError::UnknownBufferSlot {
                    location: attribute.location,
                    slot: attribute.buffer_slot,
                });
            };
            let end = attribute
                .offset
                .checked_add(attribute.format.size())
                .ok_or(VertexLayoutError::AttributeExceedsStride {
                    location: attribute.location,
                    end: u32::MAX,
                    stride: buffer.stride,
                })?;
            if end > buffer.stride {
                return Err(VertexLayoutError::AttributeExceedsStride {
                    location: attribute.location,
                    end,
                    stride: buffer.stride,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(stride: u32, attributes: Vec<VertexAttribute>) -> VertexLayout {
        VertexLayout {
            buffers: vec![VertexBufferLayout {
                slot: 0,
                stride,
                step_mode: VertexStepMode::Vertex,
            }],
            attributes,
        }
    }

    fn attribute(location: u32, format: VertexFormat, offset: u32) -> VertexAttribute {
        VertexAttribute {
            location,
            buffer_slot: 0,
            format,
            offset,
        }
    }

    #[test]
    fn a_position_and_colour_layout_is_accepted() {
        let vertices = layout(
            16,
            vec![
                attribute(0, VertexFormat::Float32x3, 0),
                attribute(1, VertexFormat::Unorm8x4, 12),
            ],
        );
        assert_eq!(vertices.validate(), Ok(()));
    }

    #[test]
    fn an_attribute_that_overruns_its_stride_is_rejected() {
        let vertices = layout(12, vec![attribute(0, VertexFormat::Float32x3, 4)]);
        assert_eq!(
            vertices.validate(),
            Err(VertexLayoutError::AttributeExceedsStride {
                location: 0,
                end: 16,
                stride: 12,
            })
        );
    }

    #[test]
    fn two_attributes_may_not_share_a_location() {
        let vertices = layout(
            16,
            vec![
                attribute(0, VertexFormat::Float32x3, 0),
                attribute(0, VertexFormat::Float32x2, 12),
            ],
        );
        assert_eq!(
            vertices.validate(),
            Err(VertexLayoutError::DuplicateLocation { location: 0 })
        );
    }

    #[test]
    fn an_attribute_must_name_a_declared_buffer() {
        let mut vertices = layout(16, vec![attribute(0, VertexFormat::Float32x3, 0)]);
        vertices.attributes[0].buffer_slot = 3;
        assert_eq!(
            vertices.validate(),
            Err(VertexLayoutError::UnknownBufferSlot { location: 0, slot: 3 })
        );
    }

    #[test]
    fn a_zero_stride_buffer_is_rejected() {
        let vertices = layout(0, vec![attribute(0, VertexFormat::Float32x3, 0)]);
        assert_eq!(
            vertices.validate(),
            Err(VertexLayoutError::ZeroStride { slot: 0 })
        );
    }

    #[test]
    fn element_sizes_are_the_tight_packing_the_artifacts_assume() {
        assert_eq!(VertexFormat::Float32x2.size(), 8);
        assert_eq!(VertexFormat::Float32x3.size(), 12);
        assert_eq!(VertexFormat::Unorm8x4.size(), 4);
    }
}
