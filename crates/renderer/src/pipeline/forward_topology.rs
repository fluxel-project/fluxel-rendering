//! The renderer-owned Forward products selectable by a frame policy.

/// The fixed set of GPU products that the first Forward consumer can build.
///
/// This remains smaller than a Graph API: a pipeline selects an existing
/// product while the renderer owns declarations, resources, recording,
/// submission, and completion ownership.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FrameTopology {
    /// Upload prepared opaque draws, rasterize them, then read the target.
    OpaqueForward,
    /// One Forward pass with opaque/mask draws followed by the renderer-owned
    /// globally ordered alpha-blend range.
    SortedForward,
    /// Clear and read the target without resolving scene assets or recording draws.
    ClearOnly,
    /// Produce albedo/normal/depth, shade them, then read the offscreen target.
    Deferred,
}
