//! Match reflected Shader inputs to concrete vertex layouts before RHI creation.

use fluxel_rhi::api::{VertexBufferLayout, VertexFormat};
use fluxel_shader::{ReflectedVertexInput, ScalarType, VertexType};
use std::{error::Error, fmt};

/// A consumer-side rejection that occurs before RHI receives shader bytes.
#[derive(Clone, Debug)]
pub(super) enum VertexLayoutPreflightError {
    MissingLocation {
        location: u32,
        shader: VertexType,
    },
    DuplicateLocation {
        location: u32,
    },
    TypeMismatch {
        location: u32,
        shader: VertexType,
        layout: VertexFormat,
    },
    UnsupportedLayoutFormat {
        location: u32,
        layout: VertexFormat,
    },
}

impl fmt::Display for VertexLayoutPreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingLocation { location, shader } => {
                write!(
                    formatter,
                    "vertex layout misses shader location {location} ({shader:?})"
                )
            }
            Self::DuplicateLocation { location } => {
                write!(formatter, "vertex layout duplicates location {location}")
            }
            Self::TypeMismatch {
                location,
                shader,
                layout,
            } => write!(
                formatter,
                "vertex layout location {location} has {layout:?}, shader requires {shader:?}"
            ),
            Self::UnsupportedLayoutFormat { location, layout } => write!(
                formatter,
                "vertex layout location {location} has unsupported format {layout:?}"
            ),
        }
    }
}

impl Error for VertexLayoutPreflightError {}

pub(super) fn preflight_vertex_layout(
    inputs: &[ReflectedVertexInput],
    layouts: &[VertexBufferLayout],
) -> Result<(), VertexLayoutPreflightError> {
    for input in inputs {
        let mut matching = layouts
            .iter()
            .flat_map(|layout| layout.attributes.iter())
            .filter(|attribute| attribute.location == input.location);
        let Some(attribute) = matching.next() else {
            return Err(VertexLayoutPreflightError::MissingLocation {
                location: input.location,
                shader: input.ty,
            });
        };
        if matching.next().is_some() {
            return Err(VertexLayoutPreflightError::DuplicateLocation {
                location: input.location,
            });
        }
        let Some(actual) = vertex_type(attribute.format) else {
            return Err(VertexLayoutPreflightError::UnsupportedLayoutFormat {
                location: input.location,
                layout: attribute.format,
            });
        };
        if actual != input.ty {
            return Err(VertexLayoutPreflightError::TypeMismatch {
                location: input.location,
                shader: input.ty,
                layout: attribute.format,
            });
        }
    }
    Ok(())
}

fn vertex_type(format: VertexFormat) -> Option<VertexType> {
    let vector = |count, scalar| match count {
        1 => Some(VertexType::Scalar(scalar)),
        2 => Some(VertexType::Vec2(scalar)),
        3 => Some(VertexType::Vec3(scalar)),
        4 => Some(VertexType::Vec4(scalar)),
        _ => None,
    };
    match format {
        VertexFormat::Float32(count) => vector(count, ScalarType::F32),
        VertexFormat::Float16(count) => vector(count, ScalarType::F16),
        VertexFormat::Sint32(count) => vector(count, ScalarType::I32),
        VertexFormat::Uint32(count) => vector(count, ScalarType::U32),
        _ => None,
    }
}
