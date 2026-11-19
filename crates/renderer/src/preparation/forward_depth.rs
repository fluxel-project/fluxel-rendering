//! Immutable sampled depth-texture authoring data for Forward preparation.

use fluxel_assets::{AssetKind, AssetSnapshot};

use super::{ForwardTextureBinding, ForwardTextureSource};

/// Marker for immutable `Depth32Float` texture assets consumed by Forward.
pub struct DepthTextureAsset;

impl AssetKind for DepthTextureAsset {}

/// CPU payload for one tightly packed sampled `Depth32Float` texture.
///
/// `depth32f` stores one row-major depth value per texel. Values must be
/// finite and lie in the RHI depth interval `[0, 1]`.
#[derive(Clone, Debug)]
pub struct DepthTextureData {
    /// Nonzero texture width and height.
    pub extent: [u32; 2],
    /// Exactly one finite `[0, 1]` depth value per texel in row-major order.
    pub depth32f: Vec<f32>,
}

impl DepthTextureData {
    pub(super) fn is_valid(&self) -> bool {
        let [width, height] = self.extent;
        let Some(length) = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|texels| usize::try_from(texels).ok())
        else {
            return false;
        };
        width != 0
            && height != 0
            && self.depth32f.len() == length
            && self
                .depth32f
                .iter()
                .all(|depth| depth.is_finite() && (0.0..=1.0).contains(depth))
    }
}

impl ForwardTextureBinding {
    /// Binds one immutable sampled `Depth32Float` asset to `resource`.
    #[must_use]
    pub fn depth_asset(
        resource: impl Into<String>,
        asset: AssetSnapshot<DepthTextureAsset, DepthTextureData>,
    ) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::DepthAsset(asset),
        }
    }
}
