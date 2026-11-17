//! Shared viewport/scissor setup for Deferred offscreen recording.

use fluxel_rhi::{
    api::{RasterPass, Rect, Result, Viewport},
    backend::RasterBackend,
};

pub(crate) fn render_area(extent: [u32; 2]) -> Rect {
    Rect {
        x: 0,
        y: 0,
        width: extent[0],
        height: extent[1],
    }
}

pub(crate) fn configure_raster<B: RasterBackend>(
    raster: &mut RasterPass<'_, B>,
    extent: [u32; 2],
) -> Result<()> {
    raster.viewport(Viewport {
        x: 0.0,
        y: 0.0,
        width: extent[0] as f32,
        height: extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(render_area(extent))
}
