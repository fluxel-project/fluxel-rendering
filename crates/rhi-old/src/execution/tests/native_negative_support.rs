//! Raster-negative fixture helpers.

use crate::*;
use fluxel_rendergraph::*;
pub(super) fn raster_negative_ops() -> AttachmentOps<[f32; 4]> {
    AttachmentOps {
        // No clear is necessary for a test that never draws or submits.
        load: LoadOp::Load,
        store: StoreOp::Store,
        write_coverage: WriteCoverage::Full,
    }
}

pub(super) fn raster_negative_color(texture: &Texture) -> RasterColorAttachment<'_, Texture> {
    RasterColorAttachment {
        index: 0,
        texture,
        range: TextureRange::Whole,
        operations: raster_negative_ops(),
    }
}
