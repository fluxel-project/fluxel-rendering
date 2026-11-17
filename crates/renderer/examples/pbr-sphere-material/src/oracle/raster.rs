use crate::scene::{self, Input};
#[derive(Clone, Copy)]
pub struct Fragment {
    pub pixel: [u32; 2],
    pub sphere: usize,
    pub depth: f32,
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
    pub edge_margin: f32,
    pub quality: f32,
}
#[derive(Clone, Copy)]
struct V {
    p: [f32; 3],
    n: [f32; 3],
    uv: [f32; 2],
    t: [f32; 4],
    clip: [f32; 4],
}
pub struct Raster {
    pixels: Vec<Option<Fragment>>,
}
impl Raster {
    pub fn from_input(i: &Input) -> Result<Self, String> {
        let mut p = vec![None; (scene::EXTENT[0] * scene::EXTENT[1]) as usize];
        for source in &i.object_sources {
            let model = source.world_from_local;
            // This consumer exercises distinct translations. Refuse a changed
            // transform family instead of silently ignoring its normal matrix.
            if model[0] != [1., 0., 0., 0.]
                || model[1] != [0., 1., 0., 0.]
                || model[2] != [0., 0., 1., 0.]
                || model[3][3] != 1.
            {
                return Err("sphere reference currently admits translation models only".into());
            }
            let geometry = i
                .assets
                .geometry
                .get(source.geometry_index)
                .ok_or("missing source geometry")?
                .value();
            let bytes = geometry.streams.first().ok_or("missing source stream")?;
            let start = geometry.vertices.start as usize * 48;
            let end = geometry.vertices.end as usize * 48;
            let bytes = bytes
                .get(start..end)
                .ok_or("source vertex range exceeds actual stream")?;
            if bytes.len() % 144 != 0 {
                return Err("source geometry is not complete 48-byte triangles".into());
            }
            for b in bytes.chunks_exact(144) {
                triangle(
                    &mut p,
                    core::array::from_fn(|n| vertex(&b[n * 48..n * 48 + 48], i.vp, model)),
                )?;
            }
        }
        Ok(Self { pixels: p })
    }
    pub fn candidates(&self, s: usize) -> Vec<Fragment> {
        self.pixels
            .iter()
            .flatten()
            .filter(|f| f.sphere == s && f.edge_margin > 0.10)
            .copied()
            .collect()
    }
    pub fn at(&self, pixel: [u32; 2]) -> Option<Fragment> {
        self.pixels[(pixel[1] * scene::EXTENT[0] + pixel[0]) as usize]
    }
}
fn vertex(b: &[u8], m: [[f32; 4]; 4], model: [[f32; 4]; 4]) -> V {
    let f = |n| f32::from_le_bytes(b[n..n + 4].try_into().unwrap());
    let local = [f(0), f(4), f(8), 1.];
    let world = mul(model, local);
    let p = [world[0], world[1], world[2]];
    V {
        p,
        n: [f(12), f(16), f(20)],
        uv: [f(24), f(28)],
        t: [f(32), f(36), f(40), f(44)],
        clip: mul(m, [p[0], p[1], p[2], 1.]),
    }
}
fn triangle(dst: &mut [Option<Fragment>], v: [V; 3]) -> Result<(), String> {
    if v.iter().any(|x| x.clip[3] <= 0.) {
        return Ok(());
    }
    let e = scene::EXTENT;
    let s = v.map(|x| {
        [
            (x.clip[0] / x.clip[3] * 0.5 + 0.5) * e[0] as f32,
            (0.5 - x.clip[1] / x.clip[3] * 0.5) * e[1] as f32,
            x.clip[2] / x.clip[3],
            1. / x.clip[3],
        ]
    });
    let area = edge(s[0], s[1], s[2][0], s[2][1]);
    if area.abs() < 1e-8 {
        return Ok(());
    }
    let minx = s
        .iter()
        .map(|x| x[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.) as u32;
    let maxx = s
        .iter()
        .map(|x| x[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(e[0] as f32) as u32;
    let miny = s
        .iter()
        .map(|x| x[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.) as u32;
    let maxy = s
        .iter()
        .map(|x| x[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(e[1] as f32) as u32;
    for y in miny..maxy {
        for x in minx..maxx {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let w = [
                edge(s[1], s[2], p[0], p[1]) / area,
                edge(s[2], s[0], p[0], p[1]) / area,
                edge(s[0], s[1], p[0], p[1]) / area,
            ];
            if w.iter().any(|z| *z < 0.) {
                continue;
            }
            let depth = (0..3).map(|n| w[n] * s[n][2]).sum::<f32>();
            if !(0.0..=1.0).contains(&depth) {
                continue;
            }
            let at = (y * e[0] + x) as usize;
            if dst[at].is_some_and(|old| old.depth < depth) {
                continue;
            }
            let d = (0..3).map(|n| w[n] * s[n][3]).sum::<f32>();
            let f3 = |z: [[f32; 3]; 3]| {
                core::array::from_fn(|c| (0..3).map(|n| w[n] * z[n][c] * s[n][3]).sum::<f32>() / d)
            };
            let pos = f3(v.map(|z| z.p));
            let normal = f3(v.map(|z| z.n));
            let uv = core::array::from_fn(|c| {
                (0..3).map(|n| w[n] * v[n].uv[c] * s[n][3]).sum::<f32>() / d
            });
            let tangent = core::array::from_fn(|c| {
                (0..3).map(|n| w[n] * v[n].t[c] * s[n][3]).sum::<f32>() / d
            });
            let margin = w.iter().copied().fold(1., f32::min);
            dst[at] = Some(Fragment {
                pixel: [x, y],
                sphere: ((uv[0] * 6.).floor() as usize).min(5),
                depth,
                pos,
                normal,
                tangent,
                uv,
                edge_margin: margin,
                quality: margin * 10. + normal[2].abs(),
            });
        }
    }
    Ok(())
}
fn mul(m: [[f32; 4]; 4], p: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|r| (0..4).map(|c| m[c][r] * p[c]).sum())
}
fn edge(a: [f32; 4], b: [f32; 4], x: f32, y: f32) -> f32 {
    (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
}
