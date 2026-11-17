//! Synthetic sRGB input and an oracle that decodes before bilinear filtering.

use fluxel_renderer::TextureEncoding;

use crate::{Probe, TextureInput, EXTENT};

pub(super) const SYNTHETIC_RGBA8: [u8; 16] = [
    0, 32, 128, 16, // black endpoint with non-binary companion channels and alpha
    255, 200, 64, 240, // white endpoint exercises decode-before-filter
    64, 96, 160, 80, 192, 144, 16, 160,
];

pub(super) fn texture() -> TextureInput {
    TextureInput {
        extent: [2, 2],
        rgba8: SYNTHETIC_RGBA8.to_vec(),
        sha256: crate::sha256(&SYNTHETIC_RGBA8),
        encoding: TextureEncoding::Srgb,
        source: "synthetic-2x2-srgb-filter",
    }
}

pub(super) fn probes(texture: &TextureInput) -> Vec<Probe> {
    // The synthetic quad shifts U so (319, 0) maps to U=0.5.  Its physical
    // V is 0.5/360 and clamps to row zero, yielding an actual black/white
    // half-texel sample whose red result is linear 0.5.
    [(319, 0), (160, 90), (480, 270), (320, 180)]
        .into_iter()
        .map(|(x, y)| Probe {
            x,
            y,
            expected: sample_at_pixel(texture, x, y),
        })
        .collect()
}

pub(super) fn sample_at_pixel(texture: &TextureInput, x: u32, y: u32) -> [u8; 4] {
    sample(
        texture,
        // Synthetic geometry begins at U=0.5/width, so raster pixel x uses
        // (x+1)/width rather than the legacy KTX quad's (x+.5)/width.
        (x as f64 + 1.0) / f64::from(EXTENT[0]),
        (y as f64 + 0.5) / f64::from(EXTENT[1]),
    )
}

pub(super) fn sample(texture: &TextureInput, u: f64, v: f64) -> [u8; 4] {
    let coordinate =
        |value: f64, size: u32| (value * f64::from(size) - 0.5).clamp(0.0, f64::from(size - 1));
    let (x, y) = (
        coordinate(u, texture.extent[0]),
        coordinate(v, texture.extent[1]),
    );
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = (
        (x0 + 1).min(texture.extent[0] - 1),
        (y0 + 1).min(texture.extent[1] - 1),
    );
    let (tx, ty) = (x - f64::from(x0), y - f64::from(y0));
    [0, 1, 2, 3].map(|channel| {
        if channel == 3 {
            return bilinear(
                texel(texture, x0, y0)[channel] as f64,
                texel(texture, x1, y0)[channel] as f64,
                texel(texture, x0, y1)[channel] as f64,
                texel(texture, x1, y1)[channel] as f64,
                tx,
                ty,
            )
            .round() as u8;
        }
        let decode = |value: u8| match texture.encoding {
            TextureEncoding::Linear => f64::from(value) / 255.0,
            TextureEncoding::Srgb => srgb_to_linear(f64::from(value) / 255.0),
        };
        (bilinear(
            decode(texel(texture, x0, y0)[channel]),
            decode(texel(texture, x1, y0)[channel]),
            decode(texel(texture, x0, y1)[channel]),
            decode(texel(texture, x1, y1)[channel]),
            tx,
            ty,
        ) * 255.0)
            .round() as u8
    })
}

fn texel(texture: &TextureInput, x: u32, y: u32) -> &[u8] {
    let at = ((y * texture.extent[0] + x) * 4) as usize;
    &texture.rgba8[at..at + 4]
}

fn bilinear(a: f64, b: f64, c: f64, d: f64, tx: f64, ty: f64) -> f64 {
    let top = a + (b - a) * tx;
    let bottom = c + (d - c) * tx;
    top + (bottom - top) * ty
}

pub(super) fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
#[path = "tests/srgb_filter.rs"]
mod tests;
