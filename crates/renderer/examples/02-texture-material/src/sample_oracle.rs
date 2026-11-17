//! Independent CPU texture sampling and retained readback expectations.

use std::{error::Error, fs, path::Path};

use crate::{encoding_name, is_srgb, srgb_filter, TextureInput, EXTENT};

#[derive(Clone, Copy)]
pub(super) struct Probe {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) expected: [u8; 4],
}

pub(super) fn expected_probes(texture: &TextureInput, direct_lighting: f64) -> Vec<Probe> {
    if is_srgb(texture) {
        return srgb_filter::probes(texture);
    }
    [
        (0, 0),
        (EXTENT[0] - 1, 0),
        (0, EXTENT[1] - 1),
        (EXTENT[0] - 1, EXTENT[1] - 1),
        (EXTENT[0] / 4, EXTENT[1] / 3),
        (EXTENT[0] * 3 / 4, EXTENT[1] * 2 / 3),
    ]
    .into_iter()
    .map(|(x, y)| Probe {
        x,
        y,
        expected: {
            let sampled = linear_clamp_sample(
                texture,
                (x as f64 + 0.5) / f64::from(EXTENT[0]),
                (y as f64 + 0.5) / f64::from(EXTENT[1]),
            );
            [
                (f64::from(sampled[0]) * direct_lighting).round() as u8,
                (f64::from(sampled[1]) * direct_lighting).round() as u8,
                (f64::from(sampled[2]) * direct_lighting).round() as u8,
                sampled[3],
            ]
        },
    })
    .collect()
}

pub(super) fn write_expected_probes(
    output: &Path,
    route: &str,
    texture: &TextureInput,
    probes: &[Probe],
    tolerance: u8,
) -> Result<(), Box<dyn Error>> {
    let probes = probes
        .iter()
        .map(|probe| {
            format!(
                "{{\"x\":{},\"y\":{},\"expectedRgba\":{:?}}}",
                probe.x, probe.y, probe.expected
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    fs::write(output.join(format!("{route}-expected-probes.json")), format!(
        "{{\"schemaVersion\":1,\"input\":{:?},\"encoding\":\"{}\",\"geometry\":\"expanded RHI 02 quad\",\"uvAtReadbackPixel\":{:?},\"sampler\":\"clamp-to-edge linear, LOD 0\",\"oracle\":\"{}\",\"tolerance\":{tolerance},\"probes\":[{probes}]}}\n",
        texture.source, encoding_name(texture.encoding),
        if is_srgb(texture) { "u=(x+1)/640; v=(y+.5)/360; pixel(319,0) has U=0.5 and V=0.5/360, clamped to texture row zero" } else { "u=(x+.5)/640; v=(y+.5)/360" },
        if is_srgb(texture) { "piecewise sRGB decode per texel before clamp-linear filtering; alpha remains linear" } else { "linear RGBA8 clamp-linear filtering" },
    ))?;
    Ok(())
}

pub(super) fn verify_probes(
    actual: &[u8],
    probes: &[Probe],
    tolerance: u8,
) -> Result<(), Box<dyn Error>> {
    for probe in probes {
        let at = ((probe.y * EXTENT[0] + probe.x) * 4) as usize;
        let found: [u8; 4] = actual[at..at + 4].try_into()?;
        if found
            .iter()
            .zip(probe.expected)
            .any(|(found, expected)| found.abs_diff(expected) > tolerance)
        {
            return Err(format!("Renderer sample at ({}, {}) is {:?}; independent texture oracle expected {:?} within {}", probe.x, probe.y, found, probe.expected, tolerance).into());
        }
    }
    Ok(())
}

fn linear_clamp_sample(texture: &TextureInput, u: f64, v: f64) -> [u8; 4] {
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
    let texel = |x: u32, y: u32| {
        let at = ((y * texture.extent[0] + x) * 4) as usize;
        &texture.rgba8[at..at + 4]
    };
    let (tx, ty) = (x - f64::from(x0), y - f64::from(y0));
    [0, 1, 2, 3].map(|channel| {
        let top = f64::from(texel(x0, y0)[channel])
            + (f64::from(texel(x1, y0)[channel]) - f64::from(texel(x0, y0)[channel])) * tx;
        let bottom = f64::from(texel(x0, y1)[channel])
            + (f64::from(texel(x1, y1)[channel]) - f64::from(texel(x0, y1)[channel])) * tx;
        (top + (bottom - top) * ty).round() as u8
    })
}
