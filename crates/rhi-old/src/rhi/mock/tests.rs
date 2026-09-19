//! The capability facts the mock's route table declares.
//!
//! These are not recorder contract tests. They pin the *input* to a rule the
//! recorder does not yet enforce: a texel copy's `buffer_offset` and
//! `bytes_per_row` must satisfy the layout limits of the route that serves the
//! copy (design sections 34.2 and 9.2/9.3). The limits are a device fact, so a
//! validator may only read them from the capability layer; until the mock
//! declares them there is no fact for any validator to read, and a route query
//! would answer `Unsupported` for copies every target backend can perform.

use super::*;

use crate::rhi::format::{RouteQuery, RouteSupport};
use crate::rhi::resource::{TextureAspect, TextureDimension};

/// The row pitch and buffer offset every target backend requires of a
/// command-buffer texel copy, and therefore the values the mock declares.
const COMMAND_COPY_ALIGNMENT: (u64, u32) = (256, 256);

#[test]
fn the_mock_serves_a_texel_copy_route_with_the_alignment_a_command_copy_has() {
    let mock = Mock::new();
    let capabilities = mock.device().capabilities();

    for query in [
        RouteQuery::BufferToTexture {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            aspect: TextureAspect::Color,
        },
        RouteQuery::TextureToBuffer {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            aspect: TextureAspect::Color,
        },
    ] {
        let support = capabilities.route(&query);
        assert!(
            support.is_supported(),
            "the mock must serve the copy route it hands out textures for: {query:?}"
        );
        let limits = support
            .capabilities()
            .and_then(|facts| facts.texel_copy_layout())
            .expect("a texel copy route must state its layout limits");
        assert_eq!(
            u64::from(limits.bytes_per_row_alignment()),
            COMMAND_COPY_ALIGNMENT.0,
            "a command-buffer texel copy is required to align its row pitch"
        );
        assert_eq!(
            limits.buffer_offset_alignment(),
            u64::from(COMMAND_COPY_ALIGNMENT.1),
            "a command-buffer texel copy is required to align its buffer offset"
        );
    }
}

#[test]
fn the_mock_serves_a_buffer_copy_route() {
    let mock = Mock::new();
    let support = mock
        .device()
        .capabilities()
        .route(&RouteQuery::BufferToBuffer);

    let limits = support
        .capabilities()
        .and_then(|facts| facts.buffer_copy_layout())
        .expect("a buffer copy route must state its layout limits");
    assert_eq!(limits.offset_alignment(), 4);
    assert_eq!(limits.size_alignment(), 4);
}

#[test]
fn a_route_the_mock_cannot_serve_is_reported_unsupported() {
    // The negative case is what makes the two tests above evidence rather than a
    // constant `Supported`: the table answers per query, so a shape the mock does
    // not declare is distinguishable from one it does.
    let mock = Mock::new();
    let support = mock.device().capabilities().route(&RouteQuery::BufferToTexture {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        aspect: TextureAspect::Color,
    });
    assert_eq!(support, RouteSupport::Unsupported);
}
