//! Immutable sampled 3D color-texture authoring data for Forward preparation.

use fluxel_assets::AssetKind;

use super::TextureEncoding;

/// Marker for immutable RGBA8 volume texture assets consumed by Forward.
pub struct ColorTextureVolumeAsset;

impl AssetKind for ColorTextureVolumeAsset {}

/// CPU payload for one sampled RGBA8 volume texture.
///
/// `rgba8` is mip-major. Each mip contains its complete Z slices in ascending
/// order; every Z slice is row-major RGBA8 texels. Each XYZ extent halves with
/// a minimum of one texel at the next mip.
#[derive(Clone, Debug)]
pub struct ColorTextureVolumeData {
    /// Nonzero base width, height, and depth.
    pub extent: [u32; 3],
    /// Number of complete authored mip payloads stored in [`Self::rgba8`].
    pub mip_levels: u32,
    /// Transfer encoding of the supplied RGBA8 payload.
    pub encoding: TextureEncoding,
    /// Exactly four bytes per texel, ordered mip-major then Z-slice row-major.
    pub rgba8: Vec<u8>,
}

impl ColorTextureVolumeData {
    /// Returns every validated authored mip as its extent and complete packed
    /// Z-slice payload.
    ///
    /// `None` means an extent, mip count, or packed payload length is invalid.
    /// This is the only parser for the volume mip layout, so staging consumers
    /// do not independently infer mip offsets or 3D physical footprints.
    pub(crate) fn mip_payloads(&self) -> Option<impl Iterator<Item = ([u32; 3], &[u8])>> {
        let [width, height, depth] = self.extent;
        if width == 0 || height == 0 || depth == 0 || self.mip_levels == 0 {
            return None;
        }
        let maximum_mips = 1 + width.max(height).max(depth).ilog2();
        if self.mip_levels > maximum_mips {
            return None;
        }

        let extents = (0..self.mip_levels).map(|mip| self.extent.map(|size| (size >> mip).max(1)));
        let payload_size = |[width, height, depth]: [u32; 3]| {
            let bytes = u64::from(width)
                .checked_mul(u64::from(height))?
                .checked_mul(u64::from(depth))?
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
                remaining.split_at(payload_size(extent).expect("validated volume mip size"));
            remaining = rest;
            (extent, payload)
        }))
    }
}
