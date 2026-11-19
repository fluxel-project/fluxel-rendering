//! Immutable layered sampled depth-texture authoring data for Forward.

use fluxel_assets::{AssetKind, AssetSnapshot};

use super::{ForwardTextureBinding, ForwardTextureSource};

/// Marker for immutable layered `Depth32Float` assets consumed by Forward.
pub struct DepthTextureArrayAsset;

impl AssetKind for DepthTextureArrayAsset {}

/// One tightly packed, one-mip layered `Depth32Float` payload.
///
/// `depth32f` is layer-major, with each physical layer row-major. Every
/// value must be finite and in the RHI depth interval `[0, 1]`.
#[derive(Clone, Debug)]
pub struct DepthTextureArrayData {
    /// Nonzero width and height shared by every layer.
    pub extent: [u32; 2],
    /// Nonzero number of complete depth layers.
    pub layers: u32,
    /// Exactly one finite `[0, 1]` depth value per texel, layer-major.
    pub depth32f: Vec<f32>,
}

impl DepthTextureArrayData {
    pub(super) fn is_valid(&self) -> bool {
        let [width, height] = self.extent;
        let Some(length) = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|texels| texels.checked_mul(u64::from(self.layers)))
            .and_then(|texels| usize::try_from(texels).ok())
        else {
            return false;
        };
        width != 0
            && height != 0
            && self.layers != 0
            && self.depth32f.len() == length
            && self
                .depth32f
                .iter()
                .all(|depth| depth.is_finite() && (0.0..=1.0).contains(depth))
    }
}

impl ForwardTextureBinding {
    /// Binds one immutable layered sampled `Depth32Float` asset to `resource`.
    #[must_use]
    pub fn depth_array_asset(
        resource: impl Into<String>,
        asset: AssetSnapshot<DepthTextureArrayAsset, DepthTextureArrayData>,
    ) -> Self {
        Self {
            resource: resource.into(),
            source: ForwardTextureSource::DepthArrayAsset(asset),
        }
    }
}
