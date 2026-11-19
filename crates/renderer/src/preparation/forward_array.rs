//! Immutable layered color-texture authoring data for Forward preparation.

use fluxel_assets::AssetKind;

use super::TextureEncoding;

/// Marker for immutable layered RGBA8 texture assets consumed by Forward.
pub struct ColorTextureArrayAsset;

impl AssetKind for ColorTextureArrayAsset {}

/// CPU payload for one layered RGBA8 color texture.
///
/// `rgba8` is mip-major: every mip contains all complete row-major RGBA8
/// layers consecutively before the next mip. Mip extents halve each dimension
/// with a minimum of one texel. A cube consumer requires a square, six-layer
/// payload and interprets those physical layers as `+X, -X, +Y, -Y, +Z, -Z`.
#[derive(Clone, Debug)]
pub struct ColorTextureArrayData {
    /// Nonzero texture width and height shared by every layer.
    pub extent: [u32; 2],
    /// Nonzero number of complete RGBA8 image layers.
    pub layers: u32,
    /// Transfer encoding of the supplied RGBA8 payload.
    pub encoding: TextureEncoding,
    /// Number of complete authored mip payloads stored in [`Self::rgba8`].
    pub mip_levels: u32,
    /// Exactly four bytes per texel, ordered mip-major and then layer-major.
    pub rgba8: Vec<u8>,
}

impl ColorTextureArrayData {
    /// Returns every validated authored mip as its extent and consecutive
    /// layer-major RGBA8 payload.
    ///
    /// `None` means the base extent, layer count, mip count, or packed byte
    /// length is invalid. This is the sole parser for the authored array-mip
    /// layout; consumers must not infer offsets independently.
    pub(crate) fn mip_payloads(&self) -> Option<impl Iterator<Item = ([u32; 2], &[u8])>> {
        let [width, height] = self.extent;
        if width == 0 || height == 0 || self.layers == 0 || self.mip_levels == 0 {
            return None;
        }
        let maximum_mips = 1 + width.max(height).ilog2();
        if self.mip_levels > maximum_mips {
            return None;
        }

        let extents = (0..self.mip_levels).map(|mip| self.extent.map(|size| (size >> mip).max(1)));
        let payload_size = |[width, height]: [u32; 2]| {
            let bytes = u64::from(width)
                .checked_mul(u64::from(height))?
                .checked_mul(u64::from(self.layers))?
                .checked_mul(4)?;
            usize::try_from(bytes).ok()
        };
        let total = extents.clone().try_fold(0usize, |total, extent| {
            total.checked_add(payload_size(extent)?)
        })?;
        if total != self.rgba8.len() {
            return None;
        }
        let mut remaining = self.rgba8.as_slice();
        Some(extents.map(move |extent| {
            let (payload, rest) =
                remaining.split_at(payload_size(extent).expect("validated mip size"));
            remaining = rest;
            (extent, payload)
        }))
    }
}
