use super::{raster::Fragment, sampling};
use crate::scene::Input;
pub fn forward(i: &Input, f: &Fragment) -> Result<[f32; 3], String> {
    shade(i, f, false, &i.orm_bytes, &i.radiance_mip_bytes)
}
pub fn deferred(i: &Input, f: &Fragment) -> Result<[f32; 3], String> {
    shade(i, f, true, &i.orm_bytes, &i.radiance_mip_bytes)
}
pub fn encode(rgb: [f32; 3]) -> [u8; 4] {
    [0, 1, 2]
        .map(|c| (rgb[c].clamp(0., 1.) * 255.).round() as u8)
        .into_iter()
        .chain([255])
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
fn shade(
    i: &Input,
    f: &Fragment,
    deferred: bool,
    orm_bytes: &[u8],
    rads: &[Vec<u8>],
) -> Result<[f32; 3], String> {
    let orm = sampling::rgba8(orm_bytes, [6, 1], f.uv);
    let ao = orm[0].clamp(0., 1.);
    let rough = orm[1].clamp(0.04, 1.);
    let metal = orm[2].clamp(0., 1.);
    // The authored `base()` callable returns `tint` directly. `base_color`
    // records the same RGB source fact for evidence; it is not another factor.
    let base = [i.tint[0], i.tint[1], i.tint[2]].map(|x| x.clamp(0., 1.));
    let n = world_normal(f.normal, f.tangent);
    let (pos, n, metal, rough, ao, base) = if deferred {
        // base/metal and normal/roughness are UNORM G-buffer attachments;
        // effects is RGBA16Float, with AO in its alpha component.
        let base = base.map(q);
        let metal = q(metal);
        let rough = q(rough);
        let ao = binary16_round(ao);
        let n = n.map(|x| q(x * 0.5 + 0.5) * 2. - 1.);
        (f.pos, n, metal, rough, ao, base)
    } else {
        (f.pos, n, metal, rough, ao, base)
    };
    let n = unit(n);
    let view = unit(sub(i.eye, pos));
    let direct = ggx(base, metal, rough, n, view);
    let ndv = dot(unit(n), view).clamp(0., 1.);
    let reflection = reflection(n, view);
    let uv = latlong(reflection);
    if rads.len() != 3 {
        return Err("radiance mip chain is not three saved images".into());
    }
    let mips: [(&[u8], [u32; 2]); 3] = [(&rads[0], [4, 2]), (&rads[1], [2, 1]), (&rads[2], [1, 1])];
    let env = sampling::mips(&mips, uv, (2. * rough).clamp(0., 2.));
    let lut = sampling::rgba16(&i.brdf_lut_bytes, [2, 2], [ndv, rough], false);
    let f0 = mix([0.04; 3], base, metal);
    let refl = f0.map(|x| ((1. - x) * lut[0] + x * lut[1]).clamp(0., 1.));
    let basef = 0.04 + (0.96 * metal);
    let energy = [1. - ((1. - basef) * lut[0] + basef * lut[1]).clamp(0., 1.); 3];
    let irradiance = sampling::rgba8_repeat_u(&i.diffuse_bytes, [2, 2], latlong(n));
    Ok(add(
        add(direct, mul([env[0], env[1], env[2]], refl)),
        scale(
            mul(
                mul([irradiance[0], irradiance[1], irradiance[2]], base),
                energy,
            ),
            ao * (1. - metal),
        ),
    ))
}
/// WGSL `reflect(-view, normal)`: `2*N*dot(N,V)-V`.
pub(super) fn reflection(normal: [f32; 3], view: [f32; 3]) -> [f32; 3] {
    sub(scale(normal, 2. * dot(normal, view)), view)
}
pub fn controls_reject(i: &Input) -> Result<(), String> {
    let probes = super::reference(i)?;
    let f = super::raster::Raster::from_input(i)?;
    let samples: Vec<_> = probes.iter().filter_map(|p| f.at(p.pixel)).collect();
    if samples.len() != probes.len() {
        return Err("a saved probe is not raster-covered".into());
    }
    if !samples.iter().any(|s| {
        let mut bad = *s;
        bad.pos = [0., 0., 0.];
        !close(
            encode(forward(i, &bad).expect("input")),
            encode(forward(i, s).expect("input")),
        )
    }) {
        return Err("ignored world position/fixed view control did not exceed tolerance".into());
    }
    let mut swapped = i.orm_bytes.clone();
    for p in swapped.chunks_exact_mut(4) {
        p.swap(1, 2)
    }
    let swapped_reject = samples.iter().try_fold(false, |rejected, s| {
        Ok::<_, String>(
            rejected
                || !close(
                    encode(shade(i, s, false, &swapped, &i.radiance_mip_bytes)?),
                    encode(forward(i, s)?),
                ),
        )
    })?;
    if !swapped_reject {
        return Err("ORM G/B swap did not exceed tolerance".into());
    }
    let mut no = i.radiance_mip_bytes.clone();
    no.iter_mut().for_each(|b| b.fill(0));
    let no_ibl_reject = samples.iter().try_fold(false, |rejected, s| {
        Ok::<_, String>(
            rejected
                || !close(
                    encode(shade(i, s, false, &i.orm_bytes, &no)?),
                    encode(forward(i, s)?),
                ),
        )
    })?;
    if !no_ibl_reject {
        return Err("missing IBL did not exceed tolerance".into());
    }
    if probes
        .iter()
        .map(|p| p.sphere_index)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != 6
    {
        return Err("six-sphere coverage missing".into());
    }
    Ok(())
}
fn close(a: [u8; 4], b: [u8; 4]) -> bool {
    a[..3]
        .iter()
        .zip(b[..3].iter())
        .all(|(x, y)| x.abs_diff(*y) <= super::TOLERANCE)
}
fn world_normal(n: [f32; 3], t: [f32; 4]) -> [f32; 3] {
    let n = unit(n);
    let raw = [t[0], t[1], t[2]];
    let ortho = sub(raw, scale(n, dot(n, raw)));
    let axis = if n[0].abs() > 0.9 {
        [0., 1., 0.]
    } else {
        [1., 0., 0.]
    };
    let tangent = if dot(ortho, ortho) > 1e-12 {
        unit(ortho)
    } else {
        unit(cross(n, axis))
    };
    let bitangent = scale(cross(n, tangent), if t[3] < 0. { -1. } else { 1. });
    unit(add(add(scale(tangent, 0.), scale(bitangent, 0.)), n))
}
fn ggx(base: [f32; 3], metal: f32, rough: f32, n: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    let n = unit(n);
    let l = [0.6, 0., 0.8];
    let h = unit(add(l, v));
    let (nl, nv) = (dot(n, l).clamp(0., 1.), dot(n, v).clamp(0., 1.));
    if nl <= 0. || nv <= 0. {
        return [0.; 3];
    }
    let a = rough.clamp(0.04, 1.).powi(4);
    let nh = dot(n, h).clamp(0., 1.);
    let vh = dot(v, h).clamp(0., 1.);
    let d = a / (core::f32::consts::PI * (nh * nh * (a - 1.) + 1.).powi(2));
    let vis = 0.5 / (nl * (nv * nv * (1. - a) + a).sqrt() + nv * (nl * nl * (1. - a) + a).sqrt());
    let f0 = mix([0.04; 3], base, metal);
    let fr = f0.map(|x| x + (1. - x) * (1. - vh).powi(5));
    add(
        mul(scale(fr.map(|x| 1. - x), 1. - metal), base).map(|x| x / core::f32::consts::PI),
        scale(fr, d * vis),
    )
    .map(|x| x * nl * 3.)
}
fn q(x: f32) -> f32 {
    (x.clamp(0., 1.) * 255.).round() / 255.
}
/// Independent round-trip through the finite positive binary16 values used by
/// the effects attachment. AO is bounded to `[0, 1]`, so this compact encoder
/// covers every normal and subnormal case relevant to the contract.
fn binary16_round(value: f32) -> f32 {
    if value <= 0. {
        return 0.;
    }
    if value >= 1. {
        return 1.;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    if exponent < -14 {
        let scaled = value * 16_777_216.;
        let mut rounded = scaled.floor() as u16;
        let remainder = scaled - scaled.floor();
        if remainder > 0.5 || (remainder == 0.5 && rounded & 1 != 0) {
            rounded += 1;
        }
        // Rounding the largest subnormal interval may produce the smallest
        // normal value (0x0400); clamping to 0x03ff would lose that carry.
        return sampling::half(rounded.min(0x0400));
    }
    let mut half_exponent = (exponent + 15) as u16;
    let fraction_bits = bits & 0x7f_ffff;
    let mut fraction = fraction_bits >> 13;
    let remainder = fraction_bits & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && fraction & 1 != 0) {
        fraction += 1;
    }
    if fraction == 0x400 {
        half_exponent += 1;
        fraction = 0;
    }
    sampling::half((half_exponent << 10) | fraction as u16)
}
fn latlong(d: [f32; 3]) -> [f32; 2] {
    let d = unit(d);
    [
        (d[2].atan2(d[0]) / core::f32::consts::TAU + 0.5).rem_euclid(1.),
        d[1].clamp(-1., 1.).acos() / core::f32::consts::PI,
    ]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn unit(x: [f32; 3]) -> [f32; 3] {
    let l = dot(x, x).sqrt();
    if l == 0. {
        [0., 0., -1.]
    } else {
        x.map(|v| v / l)
    }
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|i| a[i] - b[i])
}
fn scale(a: [f32; 3], x: f32) -> [f32; 3] {
    a.map(|v| v * x)
}
fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|i| a[i] * b[i])
}
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    core::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
#[path = "../tests/oracle_half.rs"]
mod half_tests;

#[cfg(test)]
#[path = "../tests/oracle_ao.rs"]
mod ao_tests;
