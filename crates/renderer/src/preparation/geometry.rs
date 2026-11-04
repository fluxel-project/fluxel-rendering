//! Runtime geometry descriptions and logical Shader-input matching.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Range,
};

use fluxel_rhi::api::{
    BufferDesc, BufferUsage, BufferUses, DeviceId, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexStepMode,
};
use fluxel_shader::{ScalarType, VertexInputDeclaration, VertexType};

/// One runtime-owned geometry stream.
#[derive(Clone, Debug)]
pub struct GeometryStream {
    /// Bytes between consecutive elements in this stream.
    pub stride: u64,
    /// Whether elements advance per vertex or per instance.
    pub step_mode: VertexStepMode,
    /// Attributes packed into this stream.
    pub attributes: Vec<GeometryAttribute>,
}

/// One semantic attribute in a geometry stream.
#[derive(Clone, Debug)]
pub struct GeometryAttribute {
    /// Consumer-defined semantic matched to a Shader input declaration.
    pub semantic: String,
    /// Physical byte offset within each stream element.
    pub offset: u64,
    /// Physical format consumed by the RHI vertex layout.
    pub format: VertexFormat,
}

/// Runtime geometry inputs prepared against a selected Shader interface.
#[derive(Clone, Debug, Default)]
pub struct GeometryLayout {
    /// Streams owned by this geometry product.
    pub streams: Vec<GeometryStream>,
}

/// RHI layouts ready for pipeline creation after semantic resolution.
#[derive(Clone, Debug)]
pub struct PreparedGeometry {
    /// Concrete layouts in the same order as [`GeometryLayout::streams`].
    pub streams: Vec<VertexBufferLayout>,
}

/// Concrete facts recorded by the owner of one vertex buffer allocation.
///
/// The descriptor and incoming state are supplied by the allocation owner, not
/// reconstructed from a Graph declaration. They remain separate from the
/// logical [`GeometryLayout`] so a later Graph consumer can derive transitions
/// from the actual incoming state.
#[derive(Clone, Debug)]
pub struct GeometryBufferOwner {
    device: DeviceId,
    desc: BufferDesc,
    incoming: BufferUses,
}

impl GeometryBufferOwner {
    /// Captures facts recorded by the allocation owner at creation time.
    ///
    /// The caller must supply the descriptor and incoming state retained with
    /// the exact allocation. This constructor cannot itself prove that facts
    /// match a backend handle; recording code performs that later check before
    /// passing the handle to RHI.
    pub fn new(device: DeviceId, desc: BufferDesc, incoming: BufferUses) -> Self {
        Self {
            device,
            desc,
            incoming,
        }
    }

    /// Returns the device that owns this allocation.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Returns the immutable allocation descriptor supplied by the owner.
    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    /// Returns the allocation's concrete incoming synchronization state.
    pub fn incoming(&self) -> BufferUses {
        self.incoming
    }
}

/// One concrete vertex-buffer binding supplied for a prepared stream.
#[derive(Clone, Debug)]
pub struct GeometryStreamBinding {
    /// Owner-recorded facts for the allocation.
    pub owner: GeometryBufferOwner,
    /// Actual physical byte range bound to the vertex-input slot.
    pub range: Range<u64>,
}

/// One non-indexed draw range consumed by geometry preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryDraw {
    /// Vertex indices consumed by the draw.
    pub vertices: Range<u32>,
    /// Instance indices consumed by the draw.
    pub instances: Range<u32>,
}

/// A prepared layout paired with validated owner facts and concrete draw ranges.
///
/// This is preparation data only: it has no Graph, backend resource, or
/// submission authority. A later Graph consumer must consume `incoming` when
/// declaring the actual vertex reads.
#[derive(Debug)]
pub struct PreparedGeometryBindings<'a> {
    geometry: &'a PreparedGeometry,
    bindings: &'a [GeometryStreamBinding],
    draw: GeometryDraw,
    accessed: Vec<Option<Range<u64>>>,
}

impl<'a> PreparedGeometryBindings<'a> {
    /// Returns the resolved RHI vertex layouts.
    pub fn geometry(&self) -> &'a PreparedGeometry {
        self.geometry
    }

    /// Returns the validated owner facts and physical binding ranges.
    pub fn bindings(&self) -> &'a [GeometryStreamBinding] {
        self.bindings
    }

    /// Returns the validated non-indexed draw ranges.
    pub fn draw(&self) -> &GeometryDraw {
        &self.draw
    }
    /// Returns physical fetch bounds for each stream; empty draws have no accesses.
    /// These ranges keep the binding base separate from the first fetched vertex.
    pub fn accessed_ranges(&self) -> &[Option<Range<u64>>] {
        &self.accessed
    }
}

/// Geometry preparation refusal returned before backend work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GeometryError {
    /// No streams were supplied for a Shader that requires vertex inputs.
    MissingStreams,
    /// A stream has no usable element stride.
    ZeroStride {
        /// Stream index.
        stream: usize,
    },
    /// An attribute extends past the stride of its owning stream.
    AttributeOutOfBounds {
        /// Stream index.
        stream: usize,
        /// Attribute semantic.
        semantic: String,
    },
    /// The same semantic occurs in more than one geometry attribute.
    DuplicateSemantic {
        /// Repeated semantic.
        semantic: String,
    },
    /// Shader declarations reuse a semantic or a location.
    DuplicateRequirement {
        /// Repeated semantic.
        semantic: String,
        /// Repeated location.
        location: u32,
    },
    /// A Shader semantic has no physical geometry attribute.
    MissingAttribute {
        /// Required semantic.
        semantic: String,
        /// Required Shader location.
        location: u32,
    },
    /// A physical format cannot represent the logical Shader input type.
    TypeMismatch {
        /// Mismatched semantic.
        semantic: String,
        /// Mismatched Shader location.
        location: u32,
    },
    /// The selected RHI format is outside the initial Shader value vocabulary.
    UnsupportedFormat {
        /// Semantic using the unsupported physical format.
        semantic: String,
    },
    /// Prepared stream count differs from supplied concrete bindings.
    BindingCount {
        /// Number of layout streams.
        expected: usize,
        /// Number of supplied bindings.
        actual: usize,
    },
    /// A concrete owner belongs to a device other than the selected device.
    ForeignDevice {
        /// Stream index.
        stream: usize,
    },
    /// The owner allocation was not created for vertex input.
    MissingVertexUsage {
        /// Stream index.
        stream: usize,
    },
    /// A supplied physical binding range is inverted or lies outside its owner allocation.
    BindingRangeOutsideOwner {
        /// Stream index.
        stream: usize,
    },
    /// A vertex or instance range is inverted.
    InvalidDrawRange,
    /// A draw's physical fetch footprint exceeds its supplied binding range.
    DrawFootprintOutsideBinding {
        /// Stream index.
        stream: usize,
    },
    /// A publicly constructed prepared stream has an invalid stride or attribute.
    InvalidPreparedLayout {
        /// Stream index.
        stream: usize,
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "geometry preparation refused: {self:?}")
    }
}

impl std::error::Error for GeometryError {}

impl GeometryLayout {
    /// Resolves semantic requirements into RHI vertex-buffer layouts.
    pub fn prepare(
        &self,
        requirements: &[VertexInputDeclaration],
    ) -> Result<PreparedGeometry, GeometryError> {
        if !requirements.is_empty() && self.streams.is_empty() {
            return Err(GeometryError::MissingStreams);
        }
        let mut attributes = BTreeMap::new();
        let mut prepared_streams = Vec::with_capacity(self.streams.len());
        for (stream_index, stream) in self.streams.iter().enumerate() {
            if stream.stride == 0 {
                return Err(GeometryError::ZeroStride {
                    stream: stream_index,
                });
            }
            for attribute in &stream.attributes {
                let Some(size) = format_size(attribute.format) else {
                    return Err(GeometryError::UnsupportedFormat {
                        semantic: attribute.semantic.clone(),
                    });
                };
                if attribute
                    .offset
                    .checked_add(size)
                    .is_none_or(|end| end > stream.stride)
                {
                    return Err(GeometryError::AttributeOutOfBounds {
                        stream: stream_index,
                        semantic: attribute.semantic.clone(),
                    });
                }
                if attributes
                    .insert(attribute.semantic.as_str(), (stream_index, attribute))
                    .is_some()
                {
                    return Err(GeometryError::DuplicateSemantic {
                        semantic: attribute.semantic.clone(),
                    });
                }
            }
            prepared_streams.push(VertexBufferLayout {
                stride: stream.stride,
                step_mode: stream.step_mode,
                attributes: Vec::new(),
            });
        }
        let mut semantics = BTreeSet::new();
        let mut locations = BTreeSet::new();
        for requirement in requirements {
            if !semantics.insert(requirement.semantic.as_str())
                || !locations.insert(requirement.location)
            {
                return Err(GeometryError::DuplicateRequirement {
                    semantic: requirement.semantic.clone(),
                    location: requirement.location,
                });
            }
            let Some((stream_index, attribute)) = attributes.get(requirement.semantic.as_str())
            else {
                return Err(GeometryError::MissingAttribute {
                    semantic: requirement.semantic.clone(),
                    location: requirement.location,
                });
            };
            let Some(actual) = vertex_type(attribute.format) else {
                return Err(GeometryError::UnsupportedFormat {
                    semantic: requirement.semantic.clone(),
                });
            };
            if actual != requirement.ty {
                return Err(GeometryError::TypeMismatch {
                    semantic: requirement.semantic.clone(),
                    location: requirement.location,
                });
            }
            prepared_streams[*stream_index]
                .attributes
                .push(VertexAttribute {
                    location: requirement.location,
                    offset: attribute.offset,
                    format: attribute.format,
                });
        }
        for stream in &mut prepared_streams {
            stream
                .attributes
                .sort_by_key(|attribute| attribute.location);
        }
        Ok(PreparedGeometry {
            streams: prepared_streams,
        })
    }
}

impl PreparedGeometry {
    /// Pairs the resolved layouts with allocation-owner facts and draw ranges.
    ///
    /// Every range is checked against the allocation descriptor actually stored
    /// by the owner. A nonzero binding start and the final attribute footprint
    /// are included in the checked arithmetic; a Graph expectation is never
    /// used as an allocation fact.
    pub fn bind<'a>(
        &'a self,
        device: DeviceId,
        bindings: &'a [GeometryStreamBinding],
        draw: GeometryDraw,
    ) -> Result<PreparedGeometryBindings<'a>, GeometryError> {
        if self.streams.len() != bindings.len() {
            return Err(GeometryError::BindingCount {
                expected: self.streams.len(),
                actual: bindings.len(),
            });
        }
        if draw.vertices.start > draw.vertices.end || draw.instances.start > draw.instances.end {
            return Err(GeometryError::InvalidDrawRange);
        }
        let draws_nothing = draw.vertices.is_empty() || draw.instances.is_empty();
        let mut accessed = Vec::with_capacity(bindings.len());
        for (stream_index, (layout, binding)) in self.streams.iter().zip(bindings).enumerate() {
            if binding.owner.device != device {
                return Err(GeometryError::ForeignDevice {
                    stream: stream_index,
                });
            }
            if !binding.owner.desc.usage.contains(BufferUsage::VERTEX) {
                return Err(GeometryError::MissingVertexUsage {
                    stream: stream_index,
                });
            }
            if binding.range.start > binding.range.end
                || binding.range.end > binding.owner.desc.size
            {
                return Err(GeometryError::BindingRangeOutsideOwner {
                    stream: stream_index,
                });
            }
            let attribute_end = prepared_attribute_end(layout, stream_index)?;
            if draws_nothing {
                accessed.push(None);
                continue;
            }
            let Some(attribute_end) = attribute_end else {
                accessed.push(None);
                continue;
            };
            let elements = match layout.step_mode {
                VertexStepMode::Vertex => &draw.vertices,
                VertexStepMode::Instance => &draw.instances,
            };
            let final_element = u64::from(elements.end - 1);
            let required_end = binding
                .range
                .start
                .checked_add(
                    final_element
                        .checked_mul(layout.stride)
                        .and_then(|offset| offset.checked_add(attribute_end))
                        .ok_or(GeometryError::DrawFootprintOutsideBinding {
                            stream: stream_index,
                        })?,
                )
                .ok_or(GeometryError::DrawFootprintOutsideBinding {
                    stream: stream_index,
                })?;
            if required_end > binding.range.end {
                return Err(GeometryError::DrawFootprintOutsideBinding {
                    stream: stream_index,
                });
            }
            let minimum_offset = layout
                .attributes
                .iter()
                .map(|attribute| attribute.offset)
                .min()
                .unwrap_or(0);
            let required_start = u64::from(elements.start)
                .checked_mul(layout.stride)
                .and_then(|offset| offset.checked_add(minimum_offset))
                .and_then(|offset| binding.range.start.checked_add(offset))
                .ok_or(GeometryError::DrawFootprintOutsideBinding {
                    stream: stream_index,
                })?;
            accessed.push(Some(required_start..required_end));
        }
        Ok(PreparedGeometryBindings {
            geometry: self,
            bindings,
            draw,
            accessed,
        })
    }
}

fn prepared_attribute_end(
    layout: &VertexBufferLayout,
    stream: usize,
) -> Result<Option<u64>, GeometryError> {
    if layout.stride == 0 {
        return Err(GeometryError::InvalidPreparedLayout { stream });
    }
    let mut end = None;
    let mut locations = BTreeSet::new();
    for attribute in &layout.attributes {
        let Some(size) = format_size(attribute.format) else {
            return Err(GeometryError::InvalidPreparedLayout { stream });
        };
        let Some(attribute_end) = attribute.offset.checked_add(size) else {
            return Err(GeometryError::InvalidPreparedLayout { stream });
        };
        if attribute_end > layout.stride || !locations.insert(attribute.location) {
            return Err(GeometryError::InvalidPreparedLayout { stream });
        }
        end = Some(end.map_or(attribute_end, |current: u64| current.max(attribute_end)));
    }
    Ok(end)
}

fn format_size(format: VertexFormat) -> Option<u64> {
    let (bytes, components): (u64, u8) = match format {
        VertexFormat::Float32(n) | VertexFormat::Uint32(n) | VertexFormat::Sint32(n) => (4, n),
        VertexFormat::Float16(n)
        | VertexFormat::Uint16(n)
        | VertexFormat::Sint16(n)
        | VertexFormat::Unorm16(n)
        | VertexFormat::Snorm16(n) => (2, n),
        VertexFormat::Uint8(n)
        | VertexFormat::Sint8(n)
        | VertexFormat::Unorm8(n)
        | VertexFormat::Snorm8(n) => (1, n),
        VertexFormat::Float64(n) | VertexFormat::Uint64(n) => (8, n),
    };
    (1..=4)
        .contains(&components)
        .then_some(bytes * u64::from(components))
}

fn vertex_type(format: VertexFormat) -> Option<VertexType> {
    let (scalar, components) = match format {
        VertexFormat::Float16(n) => (ScalarType::F16, n),
        VertexFormat::Float32(n) => (ScalarType::F32, n),
        VertexFormat::Sint32(n) => (ScalarType::I32, n),
        VertexFormat::Uint32(n) => (ScalarType::U32, n),
        _ => return None,
    };
    match components {
        1 => Some(VertexType::Scalar(scalar)),
        2 => Some(VertexType::Vec2(scalar)),
        3 => Some(VertexType::Vec3(scalar)),
        4 => Some(VertexType::Vec4(scalar)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../tests/preparation/geometry.rs"]
mod tests;
