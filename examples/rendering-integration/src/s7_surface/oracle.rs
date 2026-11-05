//! Independent flat-triangle Deferred Color oracle for the Surface companion.

use std::error::Error;

use super::fixture::{COMPANION_EXTENT, POSITIONS};

pub(super) fn image() -> (Vec<u8>, Vec<u8>) {
    let mut image = vec![0; (COMPANION_EXTENT[0] * COMPANION_EXTENT[1] * 4) as usize];
    let mut mask = vec![0; (COMPANION_EXTENT[0] * COMPANION_EXTENT[1]) as usize];
    for y in 0..COMPANION_EXTENT[1] {
        for x in 0..COMPANION_EXTENT[0] {
            let point = [
                2.0 * (x as f32 + 0.5) / COMPANION_EXTENT[0] as f32 - 1.0,
                1.0 - 2.0 * (y as f32 + 0.5) / COMPANION_EXTENT[1] as f32,
            ];
            let index = (y * COMPANION_EXTENT[0] + x) as usize;
            let edges = edge_distances(point);
            mask[index] = u8::from(edges.iter().all(|value| *value > 0.035));
            if inside(point) {
                let offset = index * 4;
                image[offset..offset + 4].copy_from_slice(&[130, 72, 29, 255]);
            }
        }
    }
    (image, mask)
}

pub(super) fn verify(
    actual: &[u8],
    expected: &[u8],
    mask: &[u8],
    name: &str,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() || expected.len() != mask.len() * 4 {
        return Err("S7 Deferred Surface oracle dimensions differ".into());
    }
    let mut interior = false;
    let mut exterior = false;
    for (index, stable) in mask.iter().enumerate() {
        if *stable == 0 {
            continue;
        }
        let wanted = &expected[index * 4..index * 4 + 4];
        let observed = &actual[index * 4..index * 4 + 4];
        interior |= wanted != &[0; 4];
        exterior |= wanted == &[0; 4];
        let rgb = if wanted == &[0; 4] {
            observed[..3] == wanted[..3]
        } else {
            observed[..3]
                .iter()
                .zip(&wanted[..3])
                .all(|(actual, expected)| actual.abs_diff(*expected) <= 1)
        };
        if !rgb || observed[3] != wanted[3] {
            return Err(format!(
                "S7 Deferred Surface {name} differs at pixel {index}: {observed:?} vs {wanted:?}"
            )
            .into());
        }
    }
    if !interior || !exterior {
        return Err("S7 Deferred Surface oracle mask lacks coverage".into());
    }
    Ok(())
}

pub(super) fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() != (COMPANION_EXTENT[0] * COMPANION_EXTENT[1] * 4) as usize {
        return Err("S7 Deferred Surface malformed RGBA".into());
    }
    let mut bytes =
        format!("P6\n{} {}\n255\n", COMPANION_EXTENT[0], COMPANION_EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        bytes.extend_from_slice(&pixel[..3]);
    }
    Ok(bytes)
}

fn inside(point: [f32; 2]) -> bool {
    let area = edge(
        POSITIONS[0],
        POSITIONS[1],
        [POSITIONS[2][0], POSITIONS[2][1]],
    );
    [
        edge(POSITIONS[1], POSITIONS[2], point),
        edge(POSITIONS[2], POSITIONS[0], point),
        edge(POSITIONS[0], POSITIONS[1], point),
    ]
    .iter()
    .all(|value| *value / area >= 0.0)
}

fn edge_distances(point: [f32; 2]) -> [f32; 3] {
    std::array::from_fn(|index| {
        let a = POSITIONS[index];
        let b = POSITIONS[(index + 1) % 3];
        edge(a, b, point).abs() / ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
    })
}

fn edge(a: [f32; 3], b: [f32; 3], point: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
}
