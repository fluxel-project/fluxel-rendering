//! Independent clipped visibility and perspective interpolation for oracle probes.

use crate::scene::{self, Input};

#[derive(Clone, Copy)]
pub(super) struct Fragment {
    pub(super) pixel: [u32; 2],
    pub(super) triangle: usize,
    pub(super) uv: [f32; 2],
    pub(super) margin: f32,
    pub(super) depth: f32,
    pub(super) normal_z: f32,
}
pub(super) struct Raster {
    pub(super) pixels: Vec<Option<Fragment>>,
}
impl Raster {
    pub(super) fn from_input(input: &Input) -> Result<Self, String> {
        let mut pixels = vec![None; (scene::EXTENT[0] * scene::EXTENT[1]) as usize];
        let bytes = input
            .geometry
            .streams
            .first()
            .ok_or("tunnel geometry has no stream")?;
        if bytes.len() % 96 != 0 {
            return Err("tunnel geometry is not expanded 32-byte triangles".into());
        }
        for (id, bytes) in bytes.chunks_exact(96).enumerate() {
            triangle(
                &mut pixels,
                id,
                [
                    vertex(&bytes[0..32], input.vp),
                    vertex(&bytes[32..64], input.vp),
                    vertex(&bytes[64..96], input.vp),
                ],
            )?;
        }
        Ok(Self { pixels })
    }
    pub(super) fn at(&self, x: u32, y: u32) -> Option<Fragment> {
        (x < scene::EXTENT[0] && y < scene::EXTENT[1])
            .then(|| self.pixels[(y * scene::EXTENT[0] + x) as usize])
            .flatten()
    }
}
#[derive(Clone, Copy)]
struct Vertex {
    clip: [f32; 4],
    uv: [f32; 2],
    local: [f32; 3],
}
fn vertex(bytes: &[u8], vp: [[f32; 4]; 4]) -> Vertex {
    let f = |at| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let point = [f(0), f(4), f(8), 1.];
    Vertex {
        clip: core::array::from_fn(|row| (0..4).map(|col| vp[col][row] * point[col]).sum()),
        uv: [f(24) * 2., f(28)],
        local: [f(0), f(4), f(8)],
    }
}
fn triangle(
    pixels: &mut [Option<Fragment>],
    id: usize,
    vertices: [Vertex; 3],
) -> Result<(), String> {
    let a: [f32; 3] = core::array::from_fn(|i| vertices[1].local[i] - vertices[0].local[i]);
    let b: [f32; 3] = core::array::from_fn(|i| vertices[2].local[i] - vertices[0].local[i]);
    let n = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let length = n.iter().map(|v| v * v).sum::<f32>().sqrt();
    let normal_z = if length > 0.0 { n[2] / length } else { 0.0 };
    let clipped = clip(vertices);
    for index in 1..clipped.len().saturating_sub(1) {
        raster_triangle(
            pixels,
            id,
            [clipped[0], clipped[index], clipped[index + 1]],
            normal_z,
        );
    }
    Ok(())
}

fn distance(vertex: Vertex, plane: usize) -> f32 {
    let [x, y, z, w] = vertex.clip;
    [x + w, w - x, y + w, w - y, z, w - z][plane]
}

/// Clip all visibility contributors; only complete original triangles supply
/// oracle probes. Ignoring a clipped foreground face would expose hidden probes.
fn clip(vertices: [Vertex; 3]) -> Vec<Vertex> {
    let mut polygon = vertices.to_vec();
    for plane in 0..6 {
        let input = std::mem::take(&mut polygon);
        let Some(mut previous) = input.last().copied() else {
            break;
        };
        for current in input {
            let a = distance(previous, plane);
            let b = distance(current, plane);
            if (a >= 0.) != (b >= 0.) {
                let t = a / (a - b);
                polygon.push(Vertex {
                    clip: core::array::from_fn(|i| {
                        previous.clip[i] + (current.clip[i] - previous.clip[i]) * t
                    }),
                    uv: core::array::from_fn(|i| {
                        previous.uv[i] + (current.uv[i] - previous.uv[i]) * t
                    }),
                    local: core::array::from_fn(|i| {
                        previous.local[i] + (current.local[i] - previous.local[i]) * t
                    }),
                });
            }
            if b >= 0. {
                polygon.push(current);
            }
            previous = current;
        }
    }
    polygon
}

fn raster_triangle(
    pixels: &mut [Option<Fragment>],
    id: usize,
    vertices: [Vertex; 3],
    normal_z: f32,
) {
    if vertices.iter().any(|v| v.clip[3] <= 1e-8) {
        return;
    }
    let screen = vertices.map(|v| {
        [
            (v.clip[0] / v.clip[3] * 0.5 + 0.5) * scene::EXTENT[0] as f32,
            (0.5 - v.clip[1] / v.clip[3] * 0.5) * scene::EXTENT[1] as f32,
            v.clip[2] / v.clip[3],
            1. / v.clip[3],
        ]
    });
    let area = edge(screen[0], screen[1], screen[2][0], screen[2][1]);
    if area.abs() < 1e-8 {
        return;
    }
    let min_x = screen
        .iter()
        .map(|s| s[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.) as u32;
    let max_x = screen
        .iter()
        .map(|s| s[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(scene::EXTENT[0] as f32) as u32;
    let min_y = screen
        .iter()
        .map(|s| s[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.) as u32;
    let max_y = screen
        .iter()
        .map(|s| s[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(scene::EXTENT[1] as f32) as u32;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let bary = [
                edge(screen[1], screen[2], point[0], point[1]) / area,
                edge(screen[2], screen[0], point[0], point[1]) / area,
                edge(screen[0], screen[1], point[0], point[1]) / area,
            ];
            if bary.iter().any(|weight| *weight < 0.) {
                continue;
            }
            let depth = (0..3).map(|i| bary[i] * screen[i][2]).sum::<f32>();
            let at = (y * scene::EXTENT[0] + x) as usize;
            if pixels[at].is_some_and(|old| old.depth < depth) {
                continue;
            }
            let denominator = (0..3).map(|i| bary[i] * screen[i][3]).sum::<f32>();
            let uv = core::array::from_fn(|axis| {
                (0..3)
                    .map(|i| bary[i] * vertices[i].uv[axis] * screen[i][3])
                    .sum::<f32>()
                    / denominator
            });
            pixels[at] = Some(Fragment {
                pixel: [x, y],
                triangle: id,
                uv,
                margin: bary.iter().copied().fold(1., f32::min),
                depth,
                normal_z,
            });
        }
    }
}
fn edge(a: [f32; 4], b: [f32; 4], x: f32, y: f32) -> f32 {
    (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
}

#[cfg(test)]
#[path = "../tests/raster.rs"]
mod tests;
