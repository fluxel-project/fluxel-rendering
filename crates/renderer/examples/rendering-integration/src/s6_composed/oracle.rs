use std::error::Error;

pub(super) const EXTENT: [u32; 2] = [96, 96];
const CLEAR: [u8; 4] = [0, 0, 0, 0];

pub(super) fn image(mesh: Option<[u8; 4]>, quad: Option<[u8; 4]>, quad_after: bool) -> Vec<u8> {
    image_with_quad_half(mesh, quad, quad_after, 0.55)
}
pub(super) fn image_with_quad_half(
    mesh: Option<[u8; 4]>,
    quad: Option<[u8; 4]>,
    quad_after: bool,
    half: f32,
) -> Vec<u8> {
    let mut result = vec![0; (EXTENT[0] * EXTENT[1] * 4) as usize];
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let point = point(x, y);
            let color = match (
                mesh.filter(|_| triangle(point)),
                quad.filter(|_| quad_shape(point, half)),
                quad_after,
            ) {
                (Some(_), Some(q), true) => q,
                (Some(m), Some(_), false) => m,
                (Some(m), None, _) => m,
                (None, Some(q), _) => q,
                (None, None, _) => CLEAR,
            };
            let offset = ((y * EXTENT[0] + x) * 4) as usize;
            result[offset..offset + 4].copy_from_slice(&color);
        }
    }
    result
}

pub(super) fn mask() -> Vec<u8> {
    (0..EXTENT[1])
        .flat_map(|y| {
            (0..EXTENT[0]).map(move |x| {
                let p = point(x, y);
                u8::from(
                    (p[0].abs() - 0.55).abs() > 0.04
                        && (p[1].abs() - 0.55).abs() > 0.04
                        && (p[0].abs() - 0.275).abs() > 0.04
                        && (p[1].abs() - 0.275).abs() > 0.04
                        && triangle_margin(p) > 0.04,
                )
            })
        })
        .collect()
}

pub(super) fn verify(
    actual: &[u8],
    expected: &[u8],
    mask: &[u8],
    name: &str,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() || mask.len() * 4 != expected.len() {
        return Err("S6 oracle shape mismatch".into());
    }
    for (index, stable) in mask.iter().enumerate() {
        if *stable != 0 && actual[index * 4..index * 4 + 4] != expected[index * 4..index * 4 + 4] {
            return Err(
                format!("S6 {name} stable pixel {index} differs from independent oracle").into(),
            );
        }
    }
    Ok(())
}

pub(super) fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() % 4 != 0 {
        return Err("malformed S6 rgba image".into());
    }
    let mut out = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        out.extend_from_slice(&pixel[..3]);
    }
    Ok(out)
}

fn point(x: u32, y: u32) -> [f32; 2] {
    [
        ((x as f32 + 0.5) / EXTENT[0] as f32) * 2.0 - 1.0,
        1.0 - ((y as f32 + 0.5) / EXTENT[1] as f32) * 2.0,
    ]
}
fn quad_shape(p: [f32; 2], half: f32) -> bool {
    p[0].abs() <= half && p[1].abs() <= half
}
fn triangle_margin([x, y]: [f32; 2]) -> f32 {
    [
        0.90 * (y + 0.40),
        -0.85 * (x - 0.45) - 0.45 * (y + 0.40),
        0.85 * x - 0.45 * (y - 0.45),
    ]
    .into_iter()
    .map(f32::abs)
    .fold(f32::INFINITY, f32::min)
        / 0.90
}
fn triangle([x, y]: [f32; 2]) -> bool {
    [
        0.90 * (y + 0.40),
        -0.85 * (x - 0.45) - 0.45 * (y + 0.40),
        0.85 * x - 0.45 * (y - 0.45),
    ]
    .into_iter()
    .all(|edge| edge >= 0.0)
}
