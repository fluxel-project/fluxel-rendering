//! Independent CPU oracle for the captured dragon's observable receiver pixels.
//!
//! It rasterizes only decoded vertex positions and UVs. It deliberately does
//! not share Material, Shader, Renderer, or Graph code with the workload.

use std::error::Error;

use fluxel_renderer::{GeometryData, RenderTransform};

use crate::inputs::EXTENT;

/// Transparent black used by both offscreen Forward targets before raster work.
pub const CLEAR_RGBA: [u8; 4] = [0, 0, 0, 0];
/// The producer color recorded for the primary capture attempt.
pub const PRIMARY_ALBEDO: [u8; 4] = [224, 92, 38, 255];
/// A deliberately distant producer-only control color.
pub const CONTROL_ALBEDO: [u8; 4] = [31, 179, 211, 255];

/// A final-target pixel and the capture texels which make it observable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Probe {
    /// Final receiver pixel, with top-left row zero.
    pub final_pixel: [u32; 2],
    /// Nearest capture texel selected from interpolated receiver UV.
    pub capture_texel: [u32; 2],
}

/// Independent expectations for both producer-color attempts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaptureOracle {
    /// Interior samples whose four-neighbour capture footprint is producer color.
    pub color_probes: Vec<Probe>,
    /// A receiver sample whose capture footprint is verified transparent clear.
    pub background_probe: Probe,
}

/// Computes observable color and clear probes from actual decoded geometry.
pub fn build(dragon: &GeometryData, plane: &GeometryData) -> Result<CaptureOracle, Box<dyn Error>> {
    let capture = coverage(dragon, EXTENT, 0.55, false)?;
    let receiver = uv_coverage(plane, EXTENT, 0.2, true)?;
    let mut color = Vec::new();
    let mut background = None;
    for (index, uv) in receiver.into_iter().enumerate() {
        let Some(uv) = uv else { continue };
        let texel = texture_texel(uv, EXTENT);
        let probe = Probe {
            final_pixel: [
                (index % EXTENT[0] as usize) as u32,
                (index / EXTENT[0] as usize) as u32,
            ],
            capture_texel: texel,
        };
        if footprint(&capture, texel, true) && separated_from(&color, probe) && color.len() < 4 {
            color.push(probe);
        }
        if background.is_none() && footprint(&capture, texel, false) {
            background = Some(probe);
        }
    }
    if color.len() != 4 || background.is_none() {
        return Err(format!(
            "capture framing produced {} separated interior probes and clearProbe={}",
            color.len(),
            background.is_some(),
        )
        .into());
    }
    Ok(CaptureOracle {
        color_probes: color,
        background_probe: background.expect("checked"),
    })
}

/// Resolves the post-transform clip-depth range of every decoded vertex.
pub fn clip_depth_range(
    geometry: &GeometryData,
    transform: RenderTransform,
) -> Result<[f32; 2], Box<dyn Error>> {
    let columns = transform.columns();
    let mut range = [f32::INFINITY, f32::NEG_INFINITY];
    for vertex in vertices(geometry)? {
        let [x, y, z] = vertex.position;
        let depth = columns[0][2] * x + columns[1][2] * y + columns[2][2] * z + columns[3][2];
        let w = columns[0][3] * x + columns[1][3] * y + columns[2][3] * z + columns[3][3];
        if !depth.is_finite() || !w.is_finite() || w <= 0.0 {
            return Err("capture framing produced an invalid clip depth".into());
        }
        let normalized = depth / w;
        range[0] = range[0].min(normalized);
        range[1] = range[1].max(normalized);
    }
    Ok(range)
}

fn separated_from(selected: &[Probe], candidate: Probe) -> bool {
    selected.iter().all(|probe| {
        let dx = probe.final_pixel[0].abs_diff(candidate.final_pixel[0]);
        let dy = probe.final_pixel[1].abs_diff(candidate.final_pixel[1]);
        dx.max(dy) >= 16
    })
}

/// Verifies exact primary/control values and transparent-background behaviour.
///
/// Passing this rejects a clear-only path, an ignored capture binding, and a
/// texture alias that does not observe the producer-color control input.
pub fn verify_pair(
    primary: &[u8],
    control: &[u8],
    oracle: &CaptureOracle,
) -> Result<(), Box<dyn Error>> {
    validate_image(primary)?;
    validate_image(control)?;
    for probe in &oracle.color_probes {
        if sample(primary, probe.final_pixel) != PRIMARY_ALBEDO {
            return Err(format!(
                "primary capture probe {:?} did not retain producer albedo",
                probe.final_pixel
            )
            .into());
        }
        if sample(control, probe.final_pixel) != CONTROL_ALBEDO {
            return Err(format!(
                "control capture probe {:?} did not retain producer albedo",
                probe.final_pixel
            )
            .into());
        }
    }
    for image in [primary, control] {
        if sample(image, oracle.background_probe.final_pixel) != CLEAR_RGBA {
            return Err("receiver background did not preserve capture clear color".into());
        }
    }
    Ok(())
}

fn validate_image(image: &[u8]) -> Result<(), Box<dyn Error>> {
    if image.len() == EXTENT[0] as usize * EXTENT[1] as usize * 4 {
        Ok(())
    } else {
        Err("readback extent differs from the fixed oracle extent".into())
    }
}

fn sample(image: &[u8], pixel: [u32; 2]) -> [u8; 4] {
    let at = ((pixel[1] * EXTENT[0] + pixel[0]) * 4) as usize;
    image[at..at + 4].try_into().expect("validated image")
}

fn coverage(
    geometry: &GeometryData,
    extent: [u32; 2],
    fit: f32,
    xz_to_xy: bool,
) -> Result<Vec<bool>, Box<dyn Error>> {
    let vertices = vertices(geometry)?;
    let mut covered = vec![false; extent[0] as usize * extent[1] as usize];
    for triangle in vertices.chunks_exact(3) {
        let points = projected_points(triangle, fit, xz_to_xy);
        raster_triangle(&mut covered, extent, points, None);
    }
    Ok(covered)
}

fn uv_coverage(
    geometry: &GeometryData,
    extent: [u32; 2],
    fit: f32,
    xz_to_xy: bool,
) -> Result<Vec<Option<[f32; 2]>>, Box<dyn Error>> {
    let vertices = vertices(geometry)?;
    let mut covered = vec![None; extent[0] as usize * extent[1] as usize];
    for triangle in vertices.chunks_exact(3) {
        let points = projected_points(triangle, fit, xz_to_xy);
        raster_triangle(
            &mut covered,
            extent,
            points,
            Some([triangle[0].uv, triangle[1].uv, triangle[2].uv]),
        );
    }
    Ok(covered)
}

fn projected_points(triangle: &[Vertex], fit: f32, xz_to_xy: bool) -> [[f32; 2]; 3] {
    [
        projected_point(triangle[0], fit, xz_to_xy),
        projected_point(triangle[1], fit, xz_to_xy),
        projected_point(triangle[2], fit, xz_to_xy),
    ]
}

fn projected_point(vertex: Vertex, fit: f32, xz_to_xy: bool) -> [f32; 2] {
    [
        vertex.position[0] * fit,
        if xz_to_xy {
            vertex.position[2] * fit
        } else {
            vertex.position[1] * fit
        },
    ]
}

trait CoverageCell: Copy {
    fn write(target: &mut Self, uv: Option<[f32; 2]>);
}
impl CoverageCell for bool {
    fn write(target: &mut Self, _: Option<[f32; 2]>) {
        *target = true;
    }
}
impl CoverageCell for Option<[f32; 2]> {
    fn write(target: &mut Self, uv: Option<[f32; 2]>) {
        *target = uv;
    }
}

fn raster_triangle<C: CoverageCell>(
    output: &mut [C],
    extent: [u32; 2],
    points: [[f32; 2]; 3],
    uvs: Option<[[f32; 2]; 3]>,
) {
    let pixel = |point: [f32; 2]| {
        [
            (point[0] + 1.0) * extent[0] as f32 / 2.0,
            (1.0 - point[1]) * extent[1] as f32 / 2.0,
        ]
    };
    let points = points.map(pixel);
    let area = edge(points[0], points[1], points[2]);
    if !area.is_finite() || area.abs() < 0.000_01 {
        return;
    }
    let min_x = points
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_x = points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(extent[0] as f32) as u32;
    let min_y = points
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_y = points
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(extent[1] as f32) as u32;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let weights = [
                edge(points[1], points[2], point) / area,
                edge(points[2], points[0], point) / area,
                edge(points[0], points[1], point) / area,
            ];
            if weights.iter().all(|weight| *weight >= -0.000_01) {
                let uv = uvs.map(|uvs| {
                    [
                        weights[0] * uvs[0][0] + weights[1] * uvs[1][0] + weights[2] * uvs[2][0],
                        weights[0] * uvs[0][1] + weights[1] * uvs[1][1] + weights[2] * uvs[2][1],
                    ]
                });
                C::write(&mut output[(y * extent[0] + x) as usize], uv);
            }
        }
    }
}

fn edge(a: [f32; 2], b: [f32; 2], point: [f32; 2]) -> f32 {
    (point[0] - a[0]) * (b[1] - a[1]) - (point[1] - a[1]) * (b[0] - a[0])
}

fn footprint(covered: &[bool], texel: [u32; 2], expected: bool) -> bool {
    texel[0] > 0
        && texel[1] > 0
        && texel[0] + 1 < EXTENT[0]
        && texel[1] + 1 < EXTENT[1]
        && [-1_i32, 0, 1].into_iter().all(|dy| {
            [-1_i32, 0, 1].into_iter().all(|dx| {
                covered[((texel[1] as i32 + dy) as u32 * EXTENT[0] + (texel[0] as i32 + dx) as u32)
                    as usize]
                    == expected
            })
        })
}

fn texture_texel(uv: [f32; 2], extent: [u32; 2]) -> [u32; 2] {
    [
        (uv[0].clamp(0.0, 0.999_999) * extent[0] as f32) as u32,
        (uv[1].clamp(0.0, 0.999_999) * extent[1] as f32) as u32,
    ]
}

#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

fn vertices(geometry: &GeometryData) -> Result<Vec<Vertex>, Box<dyn Error>> {
    if geometry.streams.len() != 1
        || geometry.layout.streams.len() != 1
        || geometry.layout.streams[0].stride != 32
    {
        return Err("oracle requires the fixed Renderer 07 position/normal/UV stream".into());
    }
    let source = &geometry.streams[0];
    let range = geometry.vertices.clone();
    let mut vertices = Vec::with_capacity((range.end - range.start) as usize);
    for index in range {
        let at = index as usize * 32;
        let read = |offset| -> Result<f32, Box<dyn Error>> {
            Ok(f32::from_le_bytes(
                source
                    .get(at + offset..at + offset + 4)
                    .ok_or("oracle vertex stream is truncated")?
                    .try_into()?,
            ))
        };
        vertices.push(Vertex {
            position: [read(0)?, read(4)?, read(8)?],
            uv: [read(24)?, read(28)?],
        });
    }
    if vertices.len() < 3 || vertices.len() % 3 != 0 {
        return Err("oracle requires non-indexed triangles".into());
    }
    Ok(vertices)
}
