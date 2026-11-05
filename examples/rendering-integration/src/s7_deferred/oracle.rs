//! Analytic S7 Deferred oracle; it derives triangle planes, not shader text.

use super::fixtures::{ALBEDO, EXTENT, flat_positions, tilt_positions};
use std::error::Error;

/// Two consumers share the same analytic raster plane but not final shading.
#[derive(Clone, Copy, Debug)]
pub(super) enum Route {
    Deferred,
    Forward,
}

pub(super) fn image(tilted: bool, route: Route) -> (Vec<u8>, Vec<u8>) {
    let vertices = if tilted {
        tilt_positions()
    } else {
        flat_positions()
    };
    image_for_positions(vertices, route)
}

/// Evaluates an independently supplied position plane; it never supplies GPU
/// vertex data and is only the readback oracle for the external deformer.
pub(super) fn image_for_positions(vertices: [[f32; 3]; 3], route: Route) -> (Vec<u8>, Vec<u8>) {
    image_for_color(vertices, route, ALBEDO)
}

/// Evaluates a supplied material output color over an analytic triangle plane.
/// This remains independent from composed WGSL and GPU uploads.
pub(super) fn image_for_color(
    vertices: [[f32; 3]; 3],
    route: Route,
    albedo: [u8; 4],
) -> (Vec<u8>, Vec<u8>) {
    let normal = geometric_normal(vertices);
    let encoded = normal.map(|value| quantize(value * 0.5 + 0.5));
    let decoded = encoded.map(|value| (f32::from(value) / 255.0) * 2.0 - 1.0);
    let mut rgba = vec![0; (EXTENT[0] * EXTENT[1] * 4) as usize];
    let mut mask = vec![0; (EXTENT[0] * EXTENT[1]) as usize];
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let point = ndc(x, y);
            let index = (y * EXTENT[0] + x) as usize;
            let offset = index * 4;
            let weights = barycentric(vertices, point);
            let edges = edge_distances(vertices, point);
            mask[index] = u8::from(edges.iter().all(|distance| *distance > 0.035));
            if let Some([a, b, c]) =
                weights.filter(|weights| weights.iter().all(|weight| *weight >= 0.0))
            {
                let depth = a * vertices[0][2] + b * vertices[1][2] + c * vertices[2][2];
                let factor = match route {
                    Route::Deferred => 0.2 + 0.6 * decoded[2].abs() + 0.2 * (1.0 - depth),
                    Route::Forward => 1.0,
                };
                rgba[offset..offset + 4].copy_from_slice(&[
                    quantize(f32::from(albedo[0]) / 255.0 * factor),
                    quantize(f32::from(albedo[1]) / 255.0 * factor),
                    quantize(f32::from(albedo[2]) / 255.0 * factor),
                    albedo[3],
                ]);
            }
        }
    }
    (rgba, mask)
}

pub(super) fn verify(
    actual: &[u8],
    expected: &[u8],
    mask: &[u8],
    name: &str,
    route: Route,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() || mask.len() * 4 != expected.len() {
        return Err("S7 Deferred oracle shape mismatch".into());
    }
    let mut interior = false;
    let mut exterior = false;
    for (index, stable) in mask.iter().enumerate() {
        if *stable == 0 {
            continue;
        }
        let expected_pixel = &expected[index * 4..index * 4 + 4];
        interior |= expected_pixel != &[0; 4];
        exterior |= expected_pixel == &[0; 4];
        let actual_pixel = actual
            .get(index * 4..index * 4 + 4)
            .ok_or("S7 Deferred actual image was truncated")?;
        let background = expected_pixel == &[0; 4];
        let rgb_matches = match route {
            // A Deferred pixel is stored once in the RGBA8 albedo G-buffer
            // and again in the RGBA8 lighting target. The ideal oracle makes
            // one final quantization, so independent UNORM rounding can move
            // any foreground RGB component by one code value. Background and
            // alpha remain exact; the bound also continues to monitor tilt.
            Route::Deferred if !background => actual_pixel[..3]
                .iter()
                .zip(&expected_pixel[..3])
                .all(|(actual, expected)| (*actual).abs_diff(*expected) <= 1),
            _ => actual_pixel[..3] == expected_pixel[..3],
        };
        if !rgb_matches || actual_pixel[3] != expected_pixel[3] {
            return Err(format!(
                "S7 {name} {route:?} differs at {},{}: actual {:?}, expected {:?}",
                index % EXTENT[0] as usize,
                index / EXTENT[0] as usize,
                actual_pixel,
                expected_pixel,
            )
            .into());
        }
    }
    if !interior || !exterior {
        return Err("S7 stable mask lacks interior/exterior coverage".into());
    }
    Ok(())
}

pub(super) fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() != (EXTENT[0] * EXTENT[1] * 4) as usize {
        return Err("S7 malformed RGBA image".into());
    }
    let mut output = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        output.extend_from_slice(&pixel[..3]);
    }
    Ok(output)
}

pub(super) fn plane(tilted: bool) -> [[f32; 3]; 3] {
    if tilted {
        tilt_positions()
    } else {
        flat_positions()
    }
}

fn ndc(x: u32, y: u32) -> [f32; 2] {
    [
        2.0 * (x as f32 + 0.5) / EXTENT[0] as f32 - 1.0,
        1.0 - 2.0 * (y as f32 + 0.5) / EXTENT[1] as f32,
    ]
}
fn barycentric(v: [[f32; 3]; 3], point: [f32; 2]) -> Option<[f32; 3]> {
    let area = edge(v[0], v[1], [v[2][0], v[2][1]]);
    (area != 0.0).then(|| {
        [
            edge(v[1], v[2], point) / area,
            edge(v[2], v[0], point) / area,
            edge(v[0], v[1], point) / area,
        ]
    })
}
fn edge(a: [f32; 3], b: [f32; 3], point: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
}
fn edge_distances(v: [[f32; 3]; 3], point: [f32; 2]) -> [f32; 3] {
    std::array::from_fn(|index| {
        let a = v[index];
        let b = v[(index + 1) % 3];
        edge(a, b, point).abs() / ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
    })
}
fn geometric_normal(v: [[f32; 3]; 3]) -> [f32; 3] {
    let a = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
    let b = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let length = (cross.into_iter().map(|value| value * value).sum::<f32>()).sqrt();
    cross.map(|value| value / length)
}
fn quantize(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
#[path = "../tests/s7_deferred_oracle.rs"]
mod tests;
