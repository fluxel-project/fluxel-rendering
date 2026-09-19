//! Section 24: vertex input.
//!
//! The vertex formats, the per-attribute layouts, and the two validators:
//! 24.1's self-consistency of a `VertexInputState`, and 24.2's agreement between
//! that state and the vertex entry point's location interface.
//!
//! Not owned here: the pipeline that consumes a `VertexInputState` (section 27,
//! `raster.rs`). These rules do not depend on a pipeline, so they are decided
//! here, and a caller can be told about its vertex layout before describing one.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::requirements::LimitKey;
use crate::api::shader::{ShaderInterface, ShaderLocation, ShaderNumericType};

// ---------------------------------------------------------------------------
// Section 24 - Vertex input
// ---------------------------------------------------------------------------

/// The layout of one vertex attribute as the vertex fetch stage reads it.
///
/// P0 freezes 32-bit float/integer formats plus the two byte-normalized forms
/// that a vertex fetch cannot express in any other way. Section 24.1 adds future
/// formats — `Snorm8`, `Uint8`/`Sint8`, `Unorm16`/`Snorm16`, `Uint16`/`Sint16`,
/// `Float16` — as variants of *this* enum, and explicitly not as a new
/// `VertexFormat` trait, because a trait would turn a capability question into a
/// type-system question a caller cannot ask at run time.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexFormat {
    /// One 32-bit float.
    Float32,
    /// Two 32-bit floats.
    Float32x2,
    /// Three 32-bit floats.
    Float32x3,
    /// Four 32-bit floats.
    Float32x4,

    /// One 32-bit unsigned integer.
    Uint32,
    /// Two 32-bit unsigned integers.
    Uint32x2,
    /// Three 32-bit unsigned integers.
    Uint32x3,
    /// Four 32-bit unsigned integers.
    Uint32x4,

    /// One 32-bit signed integer.
    Sint32,
    /// Two 32-bit signed integers.
    Sint32x2,
    /// Three 32-bit signed integers.
    Sint32x3,
    /// Four 32-bit signed integers.
    Sint32x4,

    /// Two unsigned 8-bit normalized values.
    Unorm8x2,
    /// Four unsigned 8-bit normalized values.
    Unorm8x4,
}

impl VertexFormat {
    /// The bytes one element of this format occupies.
    ///
    /// The number an attribute's `offset + byte_size <= stride` check needs, and
    /// the reason the rule can be written as one comparison rather than as a table
    /// per format.
    pub fn byte_size(self) -> u32 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 4,
            Self::Float32x2 | Self::Uint32x2 | Self::Sint32x2 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
            Self::Unorm8x2 => 2,
            Self::Unorm8x4 => 4,
        }
    }

    /// Portable numeric type sent to shader location after vertex fetch.
    ///
    /// The two `Unorm8` formats report [`ShaderNumericType::Float32`] because
    /// vertex fetch normalizes them into a float before the shader sees them; a
    /// shader input declared as an integer at a location fed by `Unorm8x4` is a
    /// mismatch even though the storage is an integer.
    pub fn shader_numeric_type(self) -> ShaderNumericType {
        match self {
            Self::Float32 | Self::Float32x2 | Self::Float32x3 | Self::Float32x4 => {
                ShaderNumericType::Float32
            }
            Self::Uint32 | Self::Uint32x2 | Self::Uint32x3 | Self::Uint32x4 => {
                ShaderNumericType::Uint32
            }
            Self::Sint32 | Self::Sint32x2 | Self::Sint32x3 | Self::Sint32x4 => {
                ShaderNumericType::Sint32
            }
            Self::Unorm8x2 | Self::Unorm8x4 => ShaderNumericType::Float32,
        }
    }

    /// How many components one element carries, `1..=4`.
    pub fn components(self) -> u8 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 1,
            Self::Float32x2 | Self::Uint32x2 | Self::Sint32x2 | Self::Unorm8x2 => 2,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 3,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 | Self::Unorm8x4 => 4,
        }
    }
}

/// Whether a vertex buffer advances per vertex or per instance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexStepMode {
    /// One element per vertex.
    Vertex,
    /// One element per instance.
    Instance,
}

/// One vertex attribute within a buffer layout.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct VertexAttribute {
    /// The shader location this attribute feeds.
    pub location: ShaderLocation,
    /// How the bytes are read.
    pub format: VertexFormat,
    /// Byte offset of the attribute within one element.
    pub offset: u64,
}

impl VertexAttribute {
    /// Declares one attribute. Checks nothing: the rules live in
    /// `validate_vertex_input_state`, which needs the stride.
    pub fn new(location: ShaderLocation, format: VertexFormat, offset: u64) -> Self {
        Self {
            location,
            format,
            offset,
        }
    }
}

/// One vertex buffer binding and the attributes read from it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct VertexBufferLayout {
    /// Byte distance between consecutive elements.
    pub stride: u64,
    /// Whether the buffer advances per vertex or per instance.
    pub step_mode: VertexStepMode,
    /// The attributes read from this buffer.
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    /// Describes one vertex buffer with no attributes yet.
    pub fn new(stride: u64, step_mode: VertexStepMode) -> Self {
        Self {
            stride,
            step_mode,
            attributes: Vec::new(),
        }
    }

    /// Adds one attribute.
    pub fn with_attribute(mut self, attribute: VertexAttribute) -> Self {
        self.attributes.push(attribute);
        self
    }
}

/// The complete vertex input state of a raster pipeline.
///
/// May contain attributes the shader does not consume, which section 24.2 allows
/// "provided the backend contract accepts them" — that condition is a backend
/// question, and the portable rule this module enforces is the opposite one: every
/// vertex shader input must have a matching attribute.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct VertexInputState {
    /// The vertex buffer bindings, in binding order.
    pub buffers: Vec<VertexBufferLayout>,
}

impl VertexInputState {
    /// An empty vertex input state, which is what a pipeline with no vertex
    /// buffers needs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one vertex buffer binding.
    pub fn with_buffer(mut self, layout: VertexBufferLayout) -> Self {
        self.buffers.push(layout);
        self
    }
}

/// Checks a vertex input state against the device's vertex limits.
///
/// Section 24.2's list, in its own order:
///
/// ```text
/// buffer count <= MaxVertexBuffers
/// attribute count <= MaxVertexAttributes
/// stride <= MaxVertexBufferArrayStride
/// ShaderLocation unique
/// attribute.offset + VertexFormat.byte_size <= stride
/// ```
///
/// The location-uniqueness rule spans every buffer, not each buffer separately: a
/// `ShaderLocation` is one location in the pipeline's vertex input, and two
/// attributes claiming it means the fetch stage has no way to say which one the
/// shader reads.
pub(crate) fn validate_vertex_input_state(
    state: &VertexInputState,
    limit: impl Fn(LimitKey) -> Option<u64>,
) -> RhiResult<()> {
    if let Some(max) = limit(LimitKey::MaxVertexBuffers) {
        if state.buffers.len() as u64 > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a vertex input state declares {} buffers, over the device maximum of {max}",
                    state.buffers.len()
                ),
            ));
        }
    }

    let mut attribute_count = 0u64;
    let mut locations: Vec<u32> = Vec::new();
    for buffer in &state.buffers {
        if let Some(max) = limit(LimitKey::MaxVertexBufferArrayStride) {
            if buffer.stride > max {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a vertex buffer stride of {} is over the device maximum of {max}",
                        buffer.stride
                    ),
                ));
            }
        }
        for attribute in &buffer.attributes {
            attribute_count += 1;
            locations.push(attribute.location.get());
            let end = attribute
                .offset
                .checked_add(attribute.format.byte_size() as u64)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        format!(
                            "vertex attribute at location {} has an offset that overflows u64",
                            attribute.location.get()
                        ),
                    )
                })?;
            if end > buffer.stride {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "vertex attribute at location {} ends at byte {end}, past the buffer \
                         stride {}",
                        attribute.location.get(),
                        buffer.stride
                    ),
                ));
            }
        }
    }

    if let Some(max) = limit(LimitKey::MaxVertexAttributes) {
        if attribute_count > max {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a vertex input state declares {attribute_count} attributes, over the device \
                     maximum of {max}"
                ),
            ));
        }
    }

    locations.sort_unstable();
    if let Some(duplicate) = locations.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("two vertex attributes claim location {}", duplicate[0]),
        ));
    }

    Ok(())
}

/// Checks a vertex input state against the vertex entry point's inputs.
///
/// Section 24.2's second list:
///
/// ```text
/// each Vertex Shader location input must have a matching attribute
/// numeric_type and components must be compatible
/// ```
///
/// "Compatible" is equality here, and it is not a narrowing: the shader input
/// declares what the shader reads, the attribute format declares what the fetch
/// stage produces ([`VertexFormat::shader_numeric_type`] and
/// [`VertexFormat::components`]), and a backend cannot convert between them
/// without a rule that no other backend reproduces.
///
/// Attributes the shader does not consume are not examined.
pub(crate) fn validate_vertex_input_against_interface(
    state: &VertexInputState,
    vertex: &ShaderInterface,
) -> RhiResult<()> {
    for input in vertex.inputs() {
        let attribute = state
            .buffers
            .iter()
            .flat_map(|buffer| buffer.attributes.iter())
            .find(|attribute| attribute.location == input.location);
        let Some(attribute) = attribute else {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "the vertex shader reads location {}, which no vertex attribute provides",
                    input.location.get()
                ),
            ));
        };

        if attribute.format.shader_numeric_type() != input.numeric_type
            || attribute.format.components() != input.components
        {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "vertex location {} is read as {:?}x{} but {:?} provides {:?}x{}",
                    input.location.get(),
                    input.numeric_type,
                    input.components,
                    attribute.format,
                    attribute.format.shader_numeric_type(),
                    attribute.format.components()
                ),
            ));
        }
    }

    Ok(())
}
