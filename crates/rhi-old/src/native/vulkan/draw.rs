//! Step 12's pure half: the portable raster state and draw ranges lowered onto the
//! commands the draw path records.
//!
//! It lowers the dynamic viewport and scissor, the index-buffer format, and the
//! vertex and index ranges `vkCmdDraw` / `vkCmdDrawIndexed` take. The owning half
//! is [`Encoder`](super::command::Encoder), which spells no viewport, scissor,
//! index type or count itself: every one of them comes from here, so there is one
//! place to be wrong instead of two.
//!
//! # The viewport keeps the borrowed path's Y flip, and that is a preserved semantic
//!
//! Step 6 emits the retained WGSL with `ADJUST_COORDINATE_SPACE` **clear**, because
//! the borrowed `wgpu-hal` Vulkan path being replaced does not set it. The emitted
//! `BuiltIn::Position` is therefore wgpu's Y-up clip space, while `Vulkan` maps a
//! positive viewport height to the *bottom* of a top-left-origin framebuffer. The
//! borrowed path reconciles the two by flipping the viewport rather than the shader
//! -- `y = y + height`, `height = -height` -- and this lowering writes the same two
//! fields for the same reason. A lowering that passed the viewport through unflipped
//! would render every retained recipe upside down while compiling cleanly and
//! passing every unit test, which is exactly the shape plan section 30.2 records
//! for the shader options.
//!
//! A negative viewport height is legal only with `VK_KHR_maintenance1` (or a Vulkan
//! 1.1 device, which this backend deliberately does not request), so
//! [`device`](super::device) verifies and enables that extension before a device
//! exists. The flip and the extension are one decision, not two.
//!
//! # The checks are the driver's own, repeated at the boundary
//!
//! The shared layer (`execution::raster`) already refuses a viewport or scissor that
//! leaves the pass extent. This module repeats the *driver-level* shape checks --
//! finite positive extents, an ordered `[0, 1]` depth range, a scissor offset that
//! fits the signed coordinate `VkRect2D` carries -- so a value that reached here
//! through a later call path still refuses before the driver instead of becoming a
//! validation error naming a handle. An inverted vertex or index range is refused
//! for the same reason: the count computed from it would wrap around.
//!
//! # Why the index type is exhaustive and the other mappings are not
//!
//! [`IndexFormat`] is this workspace's own closed enum, so a third format must be
//! taught to [`index_type`] at compile time rather than falling through to a
//! wildcard. That is the opposite shape from a mapping over a `#[non_exhaustive]`
//! portable enum, which returns `Option` rather than inventing a value.

use std::ops::Range;

use ash::vk;
use fluxel_rendergraph::{IndexFormat, ScissorRect, Viewport};

/// Why a raster state or range was refused.
///
/// The three variants are different sentences because they need different fixes:
/// a viewport whose numbers the driver rejects, a scissor whose shape it rejects,
/// and a range whose end precedes its start. None of them is a driver result: each
/// is decided here, before a command is recorded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrawError {
    /// The viewport's extent, coordinates or depth range is not one `Vulkan`
    /// accepts.
    Viewport,
    /// The scissor's extent is zero, or one of its offsets cannot be named as the
    /// signed coordinate `VkRect2D` carries.
    Scissor,
    /// The vertex or index range is inverted, so the count computed from it would
    /// wrap around.
    Range,
}

/// Lowers one portable viewport onto the `VkViewport` the driver is handed.
///
/// The Y flip is described in the module docs; the checks are the ones `Vulkan`
/// places on a viewport, so a value the driver would reject is refused by name
/// before `vkCmdSetViewport` is reached.
pub(crate) fn viewport(viewport: Viewport) -> Result<vk::Viewport, DrawError> {
    if !viewport.x.is_finite()
        || !viewport.y.is_finite()
        || !viewport.width.is_finite()
        || !viewport.height.is_finite()
        || !viewport.min_depth.is_finite()
        || !viewport.max_depth.is_finite()
        || viewport.width <= 0.0
        || viewport.height <= 0.0
        || !(0.0..=1.0).contains(&viewport.min_depth)
        || !(0.0..=1.0).contains(&viewport.max_depth)
        || viewport.min_depth > viewport.max_depth
    {
        return Err(DrawError::Viewport);
    }
    Ok(vk::Viewport {
        x: viewport.x,
        // The flip, written as the borrowed path writes it: the top edge becomes
        // the far edge and the height becomes negative, so wgpu's Y-up clip space
        // lands upright in a top-left-origin framebuffer.
        y: viewport.y + viewport.height,
        width: viewport.width,
        height: -viewport.height,
        min_depth: viewport.min_depth,
        max_depth: viewport.max_depth,
    })
}

/// Lowers one portable scissor rectangle onto the `VkRect2D` the driver is handed.
///
/// A zero extent is refused because the shared layer refuses it -- a scissor that
/// clips everything is a caller mistake rather than a request to draw nothing --
/// and an offset above `i32::MAX` is refused because `VkRect2D`'s offset is signed
/// and the `as i32` cast the borrowed path uses would wrap it into a negative
/// coordinate.
pub(crate) fn scissor(scissor: ScissorRect) -> Result<vk::Rect2D, DrawError> {
    if scissor.width == 0
        || scissor.height == 0
        || scissor.x > i32::MAX as u32
        || scissor.y > i32::MAX as u32
    {
        return Err(DrawError::Scissor);
    }
    Ok(vk::Rect2D {
        offset: vk::Offset2D {
            x: scissor.x as i32,
            y: scissor.y as i32,
        },
        extent: vk::Extent2D {
            width: scissor.width,
            height: scissor.height,
        },
    })
}

/// The `Vulkan` index type one portable format names.
///
/// Exhaustive rather than `Option`-returning, for the reason the module docs give:
/// [`IndexFormat`] is closed, so a format added later must be taught here at
/// compile time.
pub(crate) const fn index_type(format: IndexFormat) -> vk::IndexType {
    match format {
        IndexFormat::Uint16 => vk::IndexType::UINT16,
        IndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

/// The `(first, count)` pair one half-open range expresses.
///
/// `vkCmdDraw` and `vkCmdDrawIndexed` take a count and a first element rather than
/// a range, so the two are computed here. An inverted range is a value returned
/// before the driver, because `end - start` on it would wrap in release builds --
/// the same `u64::MAX` trap step 8's copy lowering records, in a `u32` range.
pub(crate) fn range(range: Range<u32>) -> Result<(u32, u32), DrawError> {
    let count = range.end.checked_sub(range.start).ok_or(DrawError::Range)?;
    Ok((range.start, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole_attachment() -> Viewport {
        Viewport {
            x: 0.0,
            y: 0.0,
            width: 16.0,
            height: 8.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    fn whole_scissor() -> ScissorRect {
        ScissorRect {
            x: 0,
            y: 0,
            width: 16,
            height: 8,
        }
    }

    #[test]
    fn a_viewport_is_flipped_so_wgpu_clip_space_lands_upright() {
        // The preserved semantic: the borrowed path writes `y + height` and a
        // negative height, and the two fields are the whole rule. `vk::Viewport`
        // derives no `PartialEq`, so the fields are asserted one by one.
        let lowered = viewport(whole_attachment()).expect("a whole-attachment viewport");
        assert_eq!(lowered.x, 0.0);
        assert_eq!(lowered.y, 8.0, "the top edge becomes the far edge");
        assert_eq!(lowered.width, 16.0);
        assert_eq!(lowered.height, -8.0, "the height is negative, which is the flip");
        assert_eq!(lowered.min_depth, 0.0);
        assert_eq!(lowered.max_depth, 1.0);
    }

    #[test]
    fn the_flip_follows_the_offset_and_keeps_the_depth_range() {
        let lowered = viewport(Viewport {
            x: 2.0,
            y: 3.0,
            width: 10.0,
            height: 4.0,
            min_depth: 0.25,
            max_depth: 0.75,
        })
        .expect("a viewport inside an extent");
        assert_eq!(lowered.x, 2.0);
        assert_eq!(lowered.y, 7.0);
        assert_eq!(lowered.height, -4.0);
        assert_eq!(lowered.min_depth, 0.25);
        assert_eq!(lowered.max_depth, 0.75);
    }

    #[test]
    fn a_viewport_the_driver_would_reject_is_refused_by_name() {
        // Every value here is one `Vulkan` refuses: a non-finite coordinate, a
        // zero extent, a depth outside `[0, 1]` and a reversed depth range. None of
        // them reaches the driver.
        let bad = [
            Viewport {
                x: f32::NAN,
                ..whole_attachment()
            },
            Viewport {
                y: f32::INFINITY,
                ..whole_attachment()
            },
            Viewport {
                width: 0.0,
                ..whole_attachment()
            },
            Viewport {
                height: 0.0,
                ..whole_attachment()
            },
            Viewport {
                min_depth: -0.1,
                ..whole_attachment()
            },
            Viewport {
                max_depth: 1.1,
                ..whole_attachment()
            },
            Viewport {
                min_depth: 0.75,
                max_depth: 0.25,
                ..whole_attachment()
            },
            Viewport {
                width: f32::INFINITY,
                ..whole_attachment()
            },
        ];
        for viewport_value in bad {
            // `vk::Viewport` derives no `PartialEq`, so the refusal is compared as
            // the error rather than as a whole `Result` -- the same shape sections
            // 31.4 and 32.3 record for `ImageSubresourceRange` and `BufferCopy`.
            assert_eq!(
                viewport(viewport_value).err(),
                Some(DrawError::Viewport),
                "{viewport_value:?} is not a viewport the driver accepts"
            );
        }
    }

    #[test]
    fn a_scissor_lowers_field_for_field() {
        let lowered = scissor(ScissorRect {
            x: 2,
            y: 3,
            width: 10,
            height: 4,
        })
        .expect("a scissor with a positive extent");
        assert_eq!(lowered.offset.x, 2);
        assert_eq!(lowered.offset.y, 3);
        assert_eq!(lowered.extent.width, 10);
        assert_eq!(lowered.extent.height, 4);
        // The whole-attachment case is the one the retained recipes use.
        assert_eq!(
            scissor(whole_scissor()).expect("the whole attachment"),
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: 16,
                    height: 8,
                },
            }
        );
    }

    #[test]
    fn a_scissor_the_driver_cannot_name_is_refused_by_name() {
        // A zero extent is the shared layer's refusal; an offset above `i32::MAX`
        // is one `VkRect2D` cannot carry as a signed coordinate, and the borrowed
        // path's `as i32` would have wrapped it negative.
        for bad in [
            ScissorRect {
                width: 0,
                ..whole_scissor()
            },
            ScissorRect {
                height: 0,
                ..whole_scissor()
            },
            ScissorRect {
                x: i32::MAX as u32 + 1,
                ..whole_scissor()
            },
            ScissorRect {
                y: u32::MAX,
                ..whole_scissor()
            },
        ] {
            assert_eq!(
                scissor(bad),
                Err(DrawError::Scissor),
                "{bad:?} is not a scissor the driver accepts"
            );
        }
    }

    #[test]
    fn the_index_types_are_distinct_and_match_the_portable_formats() {
        let formats = [IndexFormat::Uint16, IndexFormat::Uint32];
        let mapped: Vec<vk::IndexType> = formats.iter().copied().map(index_type).collect();
        assert_eq!(mapped[0], vk::IndexType::UINT16);
        assert_eq!(mapped[1], vk::IndexType::UINT32);
        assert_ne!(mapped[0], mapped[1]);
    }

    #[test]
    fn a_range_is_the_count_and_the_first_element() {
        assert_eq!(range(0..3), Ok((0, 3)));
        assert_eq!(range(4..7), Ok((4, 3)));
        // An empty range is legal: the driver records a no-op draw rather than
        // refusing the count, so this is a value rather than a refusal.
        assert_eq!(range(5..5), Ok((5, 0)));
    }

    #[test]
    fn an_inverted_range_is_refused_before_it_can_wrap() {
        // The ends are locals rather than literals because a literal reversed range is
        // what `clippy::reversed_empty_ranges` forbids; nothing iterates these, they
        // are only lowered, and the refusal is the subject.
        let (start, end) = (3u32, 2u32);
        assert_eq!(range(start..end), Err(DrawError::Range));
        let (start, end) = (u32::MAX, 0u32);
        assert_eq!(range(start..end), Err(DrawError::Range));
    }
}
