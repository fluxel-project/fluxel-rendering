//! Independent source-byte sampling and planar geometry/PBR reference.
use crate::scene::{EXTENT, Input, TINTS};
use fluxel_renderer::TextureEncoding;

pub const TOLERANCE: u8 = 3;
#[derive(Clone, Debug)]
pub struct Probe {
    pub x: u32,
    pub y: u32,
    pub object: usize,
    pub uv: [f64; 2],
    pub forward: [u8; 4],
    pub deferred: [u8; 4],
}

pub fn reference(input: &Input) -> Result<Vec<Probe>, String> {
    // This consumer deliberately has one fixed plane mesh with positive diagonal
    // models. Reject changes instead of silently applying the planar oracle.
    let bytes = &input.geometry_bytes;
    if bytes.len() != 6 * 48 {
        return Err("oracle expects six interleaved plane vertices".into());
    }
    for vertex in bytes.chunks_exact(48) {
        let v: Vec<_> = vertex
            .chunks_exact(4)
            .map(|x| f64::from(f32::from_le_bytes(x.try_into().unwrap())))
            .collect();
        if v[2] != 0.
            || v[3..6] != [0., 0., 1.]
            || v[8..12] != [1., 0., 0., 1.]
            || v[6] != (v[0] + 1.) / 2.
            || v[7] != (1. - v[1]) / 2.
        {
            return Err("oracle plane/normal/tangent/UV contract changed".into());
        }
    }
    if input
        .assets
        .textures
        .iter()
        .any(|x| x.value().extent != [2, 2])
        || input.diffuse.value().extent != [1, 1]
    {
        return Err("oracle requires 2x2 material images and constant 1x1 irradiance".into());
    }
    // Selection uses CPU input cases only, never a GPU result. The same
    // coordinates serve every control so source changes remain comparable.
    let controls = crate::scene::Case::ALL
        .into_iter()
        .map(crate::scene::input)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let mut probes = Vec::new();
    for object in 0..2 {
        let m = input.models[object];
        if m[0][0] <= 0.
            || m[1][1] <= 0.
            || m[2] != [0., 0., 1., 0.]
            || m[0][1..] != [0., 0., 0.]
            || m[1][0] != 0.
            || m[1][2..] != [0., 0.]
            || m[3][3] != 1.
        {
            return Err("oracle supports positive diagonal plane models only".into());
        }
        for v in [0.3, 0.7] {
            for u in [0.2, 0.5, 0.8] {
                let (x, y, uv) = choose_stable_probe(&controls, object, m, [u, v], &probes)?;
                let forward = shade(input, object, uv, false);
                let deferred = shade(input, object, uv, true);
                if forward[..3].iter().chain(&deferred[..3]).any(|&x| x >= 250) {
                    return Err("oracle probe saturated".into());
                }
                probes.push(Probe {
                    x,
                    y,
                    object,
                    uv,
                    forward,
                    deferred,
                });
            }
        }
    }
    for (x, y) in [
        (0, 0),
        (EXTENT[0] - 1, 0),
        (0, EXTENT[1] - 1),
        (EXTENT[0] - 1, EXTENT[1] - 1),
    ] {
        probes.push(Probe {
            x,
            y,
            object: 2,
            uv: [0.; 2],
            forward: [0; 4],
            deferred: [0; 4],
        });
    }
    Ok(probes)
}

fn projected_probe(m: [[f32; 4]; 4], target: [f64; 2]) -> (u32, u32, [f64; 2]) {
    let world = [
        f64::from(m[0][0]) * (target[0] * 2. - 1.) + f64::from(m[3][0]),
        f64::from(m[1][1]) * (1. - target[1] * 2.) + f64::from(m[3][1]),
    ];
    let x = ((world[0] + 1.) * 0.5 * f64::from(EXTENT[0])).floor() as u32;
    let y = ((1. - world[1]) * 0.5 * f64::from(EXTENT[1])).floor() as u32;
    let ndc = [
        (f64::from(x) + 0.5) * 2. / f64::from(EXTENT[0]) - 1.,
        1. - (f64::from(y) + 0.5) * 2. / f64::from(EXTENT[1]),
    ];
    (
        x,
        y,
        [
            (1. + (ndc[0] - f64::from(m[3][0])) / f64::from(m[0][0])) / 2.,
            (1. - (ndc[1] - f64::from(m[3][1])) / f64::from(m[1][1])) / 2.,
        ],
    )
}

fn choose_stable_probe(
    controls: &[Input],
    object: usize,
    model: [[f32; 4]; 4],
    target: [f64; 2],
    occupied: &[Probe],
) -> Result<(u32, u32, [f64; 2]), String> {
    let mut candidates = Vec::new();
    for du in -10..=10 {
        for dv in -6..=6 {
            let uv = [
                target[0] + f64::from(du) * 0.025,
                target[1] + f64::from(dv) * 0.025,
            ];
            // Center-column probes must mix both axes, not only clamp samples.
            if !(uv[0] > 0.05 && uv[0] < 0.95 && uv[1] > 0.05 && uv[1] < 0.95)
                || (target[0] == 0.5
                    && !(uv[0] > 0.26 && uv[0] < 0.74 && uv[1] > 0.26 && uv[1] < 0.74))
            {
                continue;
            }
            candidates.push((du * du + dv * dv, projected_probe(model, uv)));
        }
    }
    candidates.sort_by_key(|(distance, (x, y, _))| (*distance, *x, *y));
    candidates
        .into_iter()
        .map(|(_, p)| p)
        .find(|(x, y, uv)| {
            occupied.iter().filter(|p| p.object == object).all(|p| {
                let dx = i64::from(p.x) - i64::from(*x);
                let dy = i64::from(p.y) - i64::from(*y);
                dx * dx + dy * dy >= 16 * 16
            }) && controls.iter().all(|input| {
                let stable = stable_sampler_arithmetic(input, object, *uv);
                stable
                    && [false, true]
                        .into_iter()
                        .all(|d| shade(input, object, *uv, d)[..3].iter().all(|&x| x < 250))
            })
        })
        .ok_or_else(|| format!("no independently stable probe near object {object} UV {target:?}"))
}

pub fn filter_precision_error(input: &Input, object: usize, uv: [f64; 2]) -> u8 {
    let weights = uv.map(|x| (x * 2. - 0.5) * 256.);
    let mut error = 0;
    for x in [weights[0].floor(), weights[0].ceil()] {
        for y in [weights[1].floor(), weights[1].ceil()] {
            let rounded_uv = [(x / 256. + 0.5) / 2., (y / 256. + 0.5) / 2.];
            for deferred in [false, true] {
                let a = shade(input, object, uv, deferred);
                let b = shade(input, object, rounded_uv, deferred);
                error = error.max(
                    a.into_iter()
                        .zip(b)
                        .take(3)
                        .map(|(a, b)| a.abs_diff(b))
                        .max()
                        .unwrap(),
                );
            }
        }
    }
    error
}

/// CPU-only probe admission for finite filter coordinates and UNORM8
/// filtering arithmetic. One code remains for final float/color conversion.
pub fn stable_sampler_arithmetic(input: &Input, object: usize, uv: [f64; 2]) -> bool {
    let reference = [
        shade(input, object, uv, false),
        shade(input, object, uv, true),
    ];
    let weights = uv.map(|x| (x * 2. - 0.5) * 256.);
    let mut coordinates = vec![uv];
    for x in [weights[0].floor(), weights[0].ceil()] {
        for y in [weights[1].floor(), weights[1].ceil()] {
            coordinates.push([(x / 256. + 0.5) / 2., (y / 256. + 0.5) / 2.]);
        }
    }
    let ends = |x: f64| [(x * 255.).floor() / 255., (x * 255.).ceil() / 255.];
    for coordinate in coordinates {
        let actual = sampled(input, object, coordinate);
        for x in ends(actual[1][0]) {
            for y in ends(actual[1][1]) {
                for z in ends(actual[1][2]) {
                    for rough in ends(actual[2][1]) {
                        for metal in ends(actual[2][2]) {
                            let mut bounded = actual;
                            bounded[1][..3].copy_from_slice(&[x, y, z]);
                            bounded[2][1] = rough;
                            bounded[2][2] = metal;
                            for (route, deferred) in [false, true].into_iter().enumerate() {
                                let candidate = shade_samples(
                                    TINTS[object],
                                    &input.diffuse.value().rgba8,
                                    bounded,
                                    deferred,
                                );
                                if candidate[..3]
                                    .iter()
                                    .zip(reference[route])
                                    .any(|(&a, b)| a.abs_diff(b) > TOLERANCE - 1)
                                {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

fn sampled(input: &Input, object: usize, uv: [f64; 2]) -> [[f64; 4]; 4] {
    [0, 1, 2, 3].map(|role| {
        let data = input.assets.textures[input.sources[object][role]].value();
        sample_image(&data.rgba8, data.extent, data.encoding, uv)
    })
}
pub fn shade(input: &Input, object: usize, uv: [f64; 2], deferred: bool) -> [u8; 4] {
    shade_samples(
        TINTS[object],
        &input.diffuse.value().rgba8,
        sampled(input, object, uv),
        deferred,
    )
}
/// Shades four decoded material samples without looking at renderer state.
pub fn shade_samples(
    tint: [f32; 4],
    diffuse: &[u8],
    samples: [[f64; 4]; 4],
    deferred: bool,
) -> [u8; 4] {
    let mut base = samples[0];
    for c in 0..3 {
        base[c] = (base[c] * f64::from(tint[c])).clamp(0., 1.);
    }
    let encoded = samples[1];
    let mut normal = normalize([
        encoded[0] * 2. - 1.,
        encoded[1] * 2. - 1.,
        encoded[2] * 2. - 1.,
    ]);
    let orm = samples[2];
    let (mut metallic, mut roughness, mut ao) = (
        orm[2].clamp(0., 1.),
        orm[1].clamp(0.04, 1.),
        orm[0].clamp(0., 1.),
    );
    let mut emission = samples[3].map(|x| x * 0.2);
    if deferred {
        base = base.map(unorm);
        metallic = unorm(metallic);
        roughness = unorm(roughness);
        normal = normalize(normal.map(|x| unorm(x * 0.5 + 0.5) * 2. - 1.));
        ao = half_round(ao);
        emission = emission.map(half_round);
    }
    let color = [base[0], base[1], base[2]];
    let direct = direct_ggx(color, metallic, roughness, normal);
    if diffuse.len() < 3 {
        return [0; 4];
    }
    let e = diffuse;
    let rgb = [0, 1, 2].map(|c| {
        direct[c] + f64::from(e[c]) / 255. * color[c] * (1. - metallic) * ao + emission[c]
    });
    [encode(rgb[0]), encode(rgb[1]), encode(rgb[2]), 255]
}

pub fn sample_image(
    bytes: &[u8],
    extent: [u32; 2],
    encoding: TextureEncoding,
    uv: [f64; 2],
) -> [f64; 4] {
    let coord = |v: f64, n: u32| (v * f64::from(n) - 0.5).clamp(0., f64::from(n - 1));
    let (x, y) = (coord(uv[0], extent[0]), coord(uv[1], extent[1]));
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(extent[0] - 1), (y0 + 1).min(extent[1] - 1));
    let at = |x: u32, y: u32, c: usize| {
        let v = f64::from(bytes[((y * extent[0] + x) * 4) as usize + c]) / 255.;
        if c < 3 && encoding == TextureEncoding::Srgb {
            decode(v)
        } else {
            v
        }
    };
    [0, 1, 2, 3].map(|c| {
        let top = at(x0, y0, c) + (at(x1, y0, c) - at(x0, y0, c)) * (x - f64::from(x0));
        let bottom = at(x0, y1, c) + (at(x1, y1, c) - at(x0, y1, c)) * (x - f64::from(x0));
        top + (bottom - top) * (y - f64::from(y0))
    })
}
pub fn decode(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
fn unorm(v: f64) -> f64 {
    v.clamp(0., 1.).mul_add(255., 0.).round() / 255.
}
fn encode(v: f64) -> u8 {
    (v.clamp(0., 1.) * 255.).round() as u8
}
fn half_round(v: f64) -> f64 {
    if v == 0. {
        return 0.;
    }
    let step = if v.abs() < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(v.abs().log2().floor() as i32 - 10)
    };
    (v / step).round_ties_even() * step
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = dot(v, v).sqrt();
    v.map(|x| x / len)
}
fn direct_ggx(base: [f64; 3], metal: f64, rough: f64, n: [f64; 3]) -> [f64; 3] {
    let l = [0.6, 0., 0.8];
    let v = [0., 0., 1.];
    let h = normalize([0.6, 0., 1.8]);
    let nl = dot(n, l).clamp(0., 1.);
    let nv = dot(n, v).clamp(0., 1.);
    if nl == 0. || nv == 0. {
        return [0.; 3];
    }
    let nh = dot(n, h).clamp(0., 1.);
    let vh = dot(v, h).clamp(0., 1.);
    let alpha2 = rough.powi(4);
    let term = nh * nh * (alpha2 - 1.) + 1.;
    let distribution = alpha2 / (std::f64::consts::PI * term * term);
    let visibility = 0.5
        / (nl * (nv * nv * (1. - alpha2) + alpha2).sqrt()
            + nv * (nl * nl * (1. - alpha2) + alpha2).sqrt());
    [0, 1, 2].map(|c| {
        let f0 = 0.04 + (base[c] - 0.04) * metal;
        let fresnel = f0 + (1. - f0) * (1. - vh).powi(5);
        let diffuse = (1. - fresnel) * (1. - metal) * base[c] / std::f64::consts::PI;
        (diffuse + distribution * visibility * fresnel) * nl * 3.
    })
}
#[cfg(test)]
#[path = "tests/oracle.rs"]
mod tests;
