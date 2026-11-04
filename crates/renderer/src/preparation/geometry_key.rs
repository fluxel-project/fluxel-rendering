//! Structural cache keys for concrete RHI vertex-buffer layouts.
//!
//! RHI layout types intentionally do not expose equality or hashing as part of
//! their public contract. Pipeline caches need an exact local representation
//! instead of formatting the layout with `Debug`.

use fluxel_rhi::api::{VertexBufferLayout, VertexFormat, VertexStepMode};

/// A complete structural identity for one RHI vertex-buffer layout.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct VertexLayoutKey {
    stride: u64,
    step_mode: VertexStepModeKey,
    attributes: Vec<VertexAttributeKey>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum VertexStepModeKey {
    Vertex,
    Instance,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct VertexAttributeKey {
    location: u32,
    offset: u64,
    format: VertexFormatKey,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum VertexFormatKey {
    Float32(u8),
    Float16(u8),
    Uint32(u8),
    Sint32(u8),
    Uint16(u8),
    Sint16(u8),
    Uint8(u8),
    Sint8(u8),
    Unorm8(u8),
    Snorm8(u8),
    Unorm16(u8),
    Snorm16(u8),
    Float64(u8),
    Uint64(u8),
}

/// Produces a cache key that preserves every RHI-visible layout field.
pub(crate) fn vertex_layout_key(layout: &VertexBufferLayout) -> VertexLayoutKey {
    VertexLayoutKey {
        stride: layout.stride,
        step_mode: match layout.step_mode {
            VertexStepMode::Vertex => VertexStepModeKey::Vertex,
            VertexStepMode::Instance => VertexStepModeKey::Instance,
        },
        attributes: layout
            .attributes
            .iter()
            .map(|attribute| VertexAttributeKey {
                location: attribute.location,
                offset: attribute.offset,
                format: match attribute.format {
                    VertexFormat::Float32(components) => VertexFormatKey::Float32(components),
                    VertexFormat::Float16(components) => VertexFormatKey::Float16(components),
                    VertexFormat::Uint32(components) => VertexFormatKey::Uint32(components),
                    VertexFormat::Sint32(components) => VertexFormatKey::Sint32(components),
                    VertexFormat::Uint16(components) => VertexFormatKey::Uint16(components),
                    VertexFormat::Sint16(components) => VertexFormatKey::Sint16(components),
                    VertexFormat::Uint8(components) => VertexFormatKey::Uint8(components),
                    VertexFormat::Sint8(components) => VertexFormatKey::Sint8(components),
                    VertexFormat::Unorm8(components) => VertexFormatKey::Unorm8(components),
                    VertexFormat::Snorm8(components) => VertexFormatKey::Snorm8(components),
                    VertexFormat::Unorm16(components) => VertexFormatKey::Unorm16(components),
                    VertexFormat::Snorm16(components) => VertexFormatKey::Snorm16(components),
                    VertexFormat::Float64(components) => VertexFormatKey::Float64(components),
                    VertexFormat::Uint64(components) => VertexFormatKey::Uint64(components),
                },
            })
            .collect(),
    }
}
