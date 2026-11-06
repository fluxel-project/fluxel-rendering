//! Independent f64 oracle for the bounded metallic-roughness PBR consumer.

use std::error::Error;

use super::fixtures::EXTENT;

/// Whether the oracle models direct Forward shading or Deferred G-buffer reads.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Route {
    Forward,
    Deferred,
}

/// Immutable PBR inputs saved before a native submission.
#[derive(Clone, Copy)]
pub(crate) struct Input {
    pub(crate) positions: [[f64; 3]; 3],
    pub(crate) normals: [[f64; 3]; 3],
    pub(crate) transform: [[f64; 4]; 4],
    pub(crate) color: [u8; 4],
    pub(crate) roughness: f64,
}

pub(crate) fn image(input: Input, route: Route) -> (Vec<u8>, Vec<u8>) {
    image_with_world_normal(
        input,
        route,
        normalize(normal_from_local(input.transform, input.normals[0])),
    )
}

pub(super) fn image_with_world_normal(
    input: Input,
    route: Route,
    world_normal: [f64; 3],
) -> (Vec<u8>, Vec<u8>) {
    let vertices = input
        .positions
        .map(|position| point(input.transform, position));
    let normal = normalize(world_normal);
    let (base, metal, roughness, normal) = match route {
        Route::Forward => (
            rgb(input.color),
            f64::from(input.color[3]) / 255.0,
            input.roughness,
            normal,
        ),
        Route::Deferred => {
            let albedo = [
                quantize(rgb(input.color)[0]),
                quantize(rgb(input.color)[1]),
                quantize(rgb(input.color)[2]),
                input.color[3],
            ];
            let encoded = normal.map(|value| quantize(value * 0.5 + 0.5));
            let decoded = normalize(encoded.map(|value| f64::from(value) / 255.0 * 2.0 - 1.0));
            (
                [
                    f64::from(albedo[0]) / 255.0,
                    f64::from(albedo[1]) / 255.0,
                    f64::from(albedo[2]) / 255.0,
                ],
                f64::from(albedo[3]) / 255.0,
                f64::from(quantize(input.roughness)) / 255.0,
                decoded,
            )
        }
    };
    let shaded = shade(base, metal, roughness, normal);
    let pixel = [
        quantize(shaded[0]),
        quantize(shaded[1]),
        quantize(shaded[2]),
        255,
    ];
    let mut output = vec![0; (EXTENT[0] * EXTENT[1] * 4) as usize];
    let mut mask = vec![0; (EXTENT[0] * EXTENT[1]) as usize];
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let index = (y * EXTENT[0] + x) as usize;
            let p = ndc(x, y);
            let weights = barycentric(vertices, p);
            let edges = edge_distances(vertices, p);
            mask[index] = u8::from(edges.iter().all(|distance| *distance > 0.04));
            if weights.is_some_and(|w| w.iter().all(|value| *value >= 0.0)) {
                output[index * 4..index * 4 + 4].copy_from_slice(&pixel);
            }
        }
    }
    (output, mask)
}

pub(crate) fn verify(
    actual: &[u8],
    expected: &[u8],
    mask: &[u8],
    name: &str,
    route: Route,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() || expected.len() != mask.len() * 4 {
        return Err("S7 PBR oracle shape mismatch".into());
    }
    // G-buffer UNORM conversion is modeled above. One code value is reserved
    // only for backend floating-point/raster rounding, never PBR modelling.
    let tolerance = match route {
        Route::Forward | Route::Deferred => 1,
    };
    let mut foreground = false;
    let mut background = false;
    for (index, stable) in mask.iter().enumerate() {
        if *stable == 0 {
            continue;
        }
        let actual = &actual[index * 4..index * 4 + 4];
        let expected = &expected[index * 4..index * 4 + 4];
        let empty = expected == [0; 4];
        foreground |= !empty;
        background |= empty;
        let rgb = if empty {
            actual[..3] == expected[..3]
        } else {
            actual[..3]
                .iter()
                .zip(&expected[..3])
                .all(|(a, b)| a.abs_diff(*b) <= tolerance)
        };
        if !rgb || actual[3] != expected[3] {
            return Err(format!(
                "S7 PBR {name} {route:?} differs at {},{}: actual {:?}, expected {:?}, rgbTolerance {tolerance}",
                index % EXTENT[0] as usize,
                index / EXTENT[0] as usize,
                actual,
                expected,
            )
            .into());
        }
    }
    if !foreground || !background {
        return Err("S7 PBR stable mask lacks foreground/background coverage".into());
    }
    Ok(())
}

pub(crate) fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() != (EXTENT[0] * EXTENT[1] * 4) as usize {
        return Err("S7 PBR malformed RGBA image".into());
    }
    let mut result = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        result.extend_from_slice(&pixel[..3]);
    }
    Ok(result)
}

fn shade(base: [f64; 3], metal: f64, roughness: f64, normal: [f64; 3]) -> [f64; 3] {
    let n = normalize(normal);
    let l = normalize([0.6, 0.0, 0.8]);
    let v = [0.0, 0.0, 1.0];
    let no_l = dot(n, l).max(0.0);
    let no_v = dot(n, v).max(0.0);
    if no_l == 0.0 || no_v == 0.0 {
        return [0.0; 3];
    }
    let h = normalize(add(l, v));
    let no_h = dot(n, h).max(0.0);
    let vo_h = dot(v, h).max(0.0);
    let roughness = roughness.clamp(0.04, 1.0);
    let alpha = roughness * roughness;
    let alpha_squared = alpha * alpha;
    let denominator = std::f64::consts::PI * (no_h * no_h * (alpha_squared - 1.0) + 1.0).powi(2);
    let d = alpha_squared / denominator;
    let smith = 0.5
        / (no_l * (no_v * no_v * (1.0 - alpha_squared) + alpha_squared).sqrt()
            + no_v * (no_l * no_l * (1.0 - alpha_squared) + alpha_squared).sqrt());
    std::array::from_fn(|index| {
        let f0 = 0.04 * (1.0 - metal.clamp(0.0, 1.0)) + base[index] * metal.clamp(0.0, 1.0);
        let fresnel = f0 + (1.0 - f0) * (1.0 - vo_h).powi(5);
        let diffuse =
            (1.0 - fresnel) * (1.0 - metal.clamp(0.0, 1.0)) * base[index] / std::f64::consts::PI;
        (diffuse + d * smith * fresnel) * no_l * 3.0
    })
}

fn point(transform: [[f64; 4]; 4], position: [f64; 3]) -> [f64; 3] {
    let value = mul4(transform, [position[0], position[1], position[2], 1.0]);
    [
        value[0] / value[3],
        value[1] / value[3],
        value[2] / value[3],
    ]
}

fn normal_from_local(transform: [[f64; 4]; 4], normal: [f64; 3]) -> [f64; 3] {
    let c0 = [transform[0][0], transform[0][1], transform[0][2]];
    let c1 = [transform[1][0], transform[1][1], transform[1][2]];
    let c2 = [transform[2][0], transform[2][1], transform[2][2]];
    let a = cross(c1, c2);
    let b = cross(c2, c0);
    let c = cross(c0, c1);
    let determinant = dot(c0, a);
    [
        (a[0] * normal[0] + b[0] * normal[1] + c[0] * normal[2]) / determinant,
        (a[1] * normal[0] + b[1] * normal[1] + c[1] * normal[2]) / determinant,
        (a[2] * normal[0] + b[2] * normal[1] + c[2] * normal[2]) / determinant,
    ]
}

fn ndc(x: u32, y: u32) -> [f64; 2] {
    [
        2.0 * (f64::from(x) + 0.5) / f64::from(EXTENT[0]) - 1.0,
        1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(EXTENT[1]),
    ]
}
fn barycentric(v: [[f64; 3]; 3], p: [f64; 2]) -> Option<[f64; 3]> {
    let area = edge(v[0], v[1], [v[2][0], v[2][1]]);
    (area != 0.0).then(|| {
        [
            edge(v[1], v[2], p) / area,
            edge(v[2], v[0], p) / area,
            edge(v[0], v[1], p) / area,
        ]
    })
}
fn edge(a: [f64; 3], b: [f64; 3], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}
fn edge_distances(v: [[f64; 3]; 3], p: [f64; 2]) -> [f64; 3] {
    std::array::from_fn(|i| {
        let a = v[i];
        let b = v[(i + 1) % 3];
        edge(a, b, p).abs() / ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
    })
}
fn rgb(color: [u8; 4]) -> [f64; 3] {
    [
        f64::from(color[0]) / 255.0,
        f64::from(color[1]) / 255.0,
        f64::from(color[2]) / 255.0,
    ]
}
fn quantize(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let length = dot(a, a).sqrt();
    [a[0] / length, a[1] / length, a[2] / length]
}
fn mul4(m: [[f64; 4]; 4], value: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|row| (0..4).map(|column| m[column][row] * value[column]).sum())
}

#[cfg(test)]
#[path = "../tests/s7_pbr_oracle.rs"]
mod tests;
