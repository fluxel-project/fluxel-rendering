//! Independent CPU depth oracle for the 04 acceptance cases.
//!
//! It deliberately models only projected triangle coverage and `LessEqual`
//! replacement. It does not inspect Renderer preparation, graph declarations,
//! or readback code.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub depth: f64,
    pub rgb: [u8; 3],
}

/// Applies the RHI 04 depth contract at one already-covered sample.
pub fn less_equal(current: Option<Sample>, incoming: Sample) -> Sample {
    match current {
        Some(previous) if previous.depth < incoming.depth => previous,
        _ => incoming,
    }
}

/// A predeclared two-colour overlap control. `draw_order` must not affect the
/// visible nearer sample, so this rejects CPU draw sorting that bypasses depth.
pub fn overlap_expected(draw_order: [Sample; 2]) -> [u8; 3] {
    less_equal(Some(draw_order[0]), draw_order[1]).rgb
}

/// Selects conservative integer probes well within the 512² output. The grid
/// oracle stores these before GPU work and evaluates projected triangles in its
/// runtime module; these coordinates avoid primitive edges and the D3D/WGL
/// rasterisation tie rules.
pub const GRID_PROBES: [[u32; 2]; 5] = [[256, 256], [192, 256], [320, 256], [256, 192], [256, 320]];

/// CPU RGBA/depth image generated without Renderer or RHI output.
pub struct RasterImage {
    pub rgba: Vec<u8>,
    pub owners: Vec<u32>,
    next_triangle: u32,
    depth: Vec<f64>,
    extent: [u32; 2],
}
impl RasterImage {
    pub fn clear(extent: [u32; 2]) -> Self {
        let pixels = (extent[0] * extent[1]) as usize;
        Self {
            rgba: vec![0; pixels * 4],
            owners: vec![0; pixels],
            next_triangle: 0,
            depth: vec![1.0; pixels],
            extent,
        }
    }
    /// Perspective-correct RGB and `LessEqual` depth for one fully visible triangle.
    pub fn triangle(&mut self, clip: [[f64; 4]; 3], rgb: [[f64; 3]; 3]) {
        self.next_triangle += 1;
        if clip
            .iter()
            .any(|p| !p.iter().all(|v| v.is_finite()) || p[3] <= 0.0)
        {
            return;
        }
        let s = clip.map(|p| {
            [
                (p[0] / p[3] * 0.5 + 0.5) * self.extent[0] as f64,
                (0.5 - p[1] / p[3] * 0.5) * self.extent[1] as f64,
                p[2] / p[3],
                1.0 / p[3],
            ]
        });
        let area = edge(s[0], s[1], s[2][0], s[2][1]);
        if area == 0.0 {
            return;
        }
        let min_x = s
            .iter()
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as u32;
        let max_x = s
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min(self.extent[0] as f64) as u32;
        let min_y = s
            .iter()
            .map(|p| p[1])
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as u32;
        let max_y = s
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min(self.extent[1] as f64) as u32;
        for y in min_y..max_y {
            for x in min_x..max_x {
                let p = [x as f64 + 0.5, y as f64 + 0.5];
                let w = [
                    edge(s[1], s[2], p[0], p[1]) / area,
                    edge(s[2], s[0], p[0], p[1]) / area,
                    edge(s[0], s[1], p[0], p[1]) / area,
                ];
                if w.iter().any(|v| *v < 0.0) {
                    continue;
                }
                let denominator: f64 = (0..3).map(|i| w[i] * s[i][3]).sum();
                // NDC depth is linear in screen barycentrics. Only varying
                // attributes (RGB here) use reciprocal-w interpolation.
                let depth: f64 = (0..3).map(|i| w[i] * s[i][2]).sum();
                if !(0.0..=1.0).contains(&depth) {
                    continue;
                }
                let at = (y * self.extent[0] + x) as usize;
                if depth > self.depth[at] {
                    continue;
                }
                self.depth[at] = depth;
                self.owners[at] = self.next_triangle;
                for c in 0..3 {
                    let value: f64 =
                        (0..3).map(|i| w[i] * rgb[i][c] * s[i][3]).sum::<f64>() / denominator;
                    self.rgba[at * 4 + c] = (value * 255.).round().clamp(0., 255.) as u8;
                }
                self.rgba[at * 4 + 3] = 255;
            }
        }
    }
}

/// 125-cube reference raster from source positions/colors and saved matrices.
pub fn grid_reference(
    extent: [u32; 2],
    camera: [[f32; 4]; 4],
    models: &[[[f32; 4]; 4]],
) -> RasterImage {
    use crate::scene::{CUBE_INDICES, CUBE_VERTICES, multiply};
    let mut image = RasterImage::clear(extent);
    for model in models {
        let matrix = multiply(camera, *model).map(|c| c.map(f64::from));
        for triangle in CUBE_INDICES.chunks_exact(3) {
            let vertices = core::array::from_fn(|slot| CUBE_VERTICES[triangle[slot] as usize]);
            let clip =
                vertices.map(|v| point(matrix, [v[0] as f64, v[1] as f64, v[2] as f64, 1.0]));
            let colors = vertices.map(|v| [v[3] as f64, v[4] as f64, v[5] as f64]);
            image.triangle(clip, colors);
        }
    }
    image
}
fn point(matrix: [[f64; 4]; 4], p: [f64; 4]) -> [f64; 4] {
    core::array::from_fn(|r| (0..4).map(|c| matrix[c][r] * p[c]).sum())
}
fn edge(a: [f64; 4], b: [f64; 4], x: f64, y: f64) -> f64 {
    (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
}
