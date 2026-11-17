//! Independent numeric split-sum controls; this module never calls renderer PBR helpers.

/// One authored material/camera control.
#[derive(Clone, Copy, Debug)]
pub struct Control {
    pub name: &'static str,
    pub metallic: f32,
    pub roughness: f32,
    pub camera: [f32; 3],
    pub lowest_mip_only: bool,
    pub lut_variant: bool,
    pub diffuse: bool,
}

/// Computes the centre-plane result from the exact bytes retained by the
/// consumer, without invoking renderer code.
pub fn expected_from_input(control: Control, input: &crate::scene::Input) -> [f32; 3] {
    expected_from_input_with_lod(control, input, 2.0 * control.roughness.clamp(0.0, 1.0))
}

/// Testable wrong-LOD branch over the same saved payload and material input.
pub fn expected_from_input_with_lod(
    control: Control,
    input: &crate::scene::Input,
    lod: f32,
) -> [f32; 3] {
    let base = [
        input.texture_bytes[0],
        input.texture_bytes[1],
        input.texture_bytes[2],
    ]
    .map(|value| f32::from(value) / 255.0);
    let normal = [0.0, 0.0, 1.0];
    let view = normalize(control.camera);
    let ndotv = dot(normal, view).clamp(0.0, 1.0);
    let f0 = mix3([0.04; 3], base, control.metallic.clamp(0.0, 1.0));
    let reflection = [-view[0], -view[1], view[2]];
    let uv = [
        (reflection[2].atan2(reflection[0]) / std::f32::consts::TAU + 0.5).rem_euclid(1.0),
        reflection[1].clamp(-1.0, 1.0).acos() / std::f32::consts::PI,
    ];
    let mips: Vec<_> = input
        .environment
        .value()
        .radiance_mips
        .iter()
        .map(|mip| (mip.rgba16f_le.as_slice(), mip.extent))
        .collect();
    let environment = sample_mips_rgba16f(&mips, uv, lod);
    let lut = sample_rgba16f(
        &input.environment.value().brdf_lut.rgba16f_le,
        input.environment.value().brdf_lut.extent,
        [ndotv, control.roughness],
        false,
    );
    let reflectance = f0.map(|value| ((1.0 - value) * lut[0] + value * lut[1]).clamp(0.0, 1.0));
    let base_f0 = 0.04 + 0.96 * control.metallic.clamp(0.0, 1.0);
    let energy = 1.0 - ((1.0 - base_f0) * lut[0] + base_f0 * lut[1]).clamp(0.0, 1.0);
    let diffuse = input.diffuse.as_ref().map_or([0.0; 3], |image| {
        let irradiance = sample_rgba8(&image.value().rgba8, image.value().extent, [0.75, 0.5]);
        irradiance
            .into_iter()
            .zip(base)
            .map(|(light, color)| light * color * (1.0 - control.metallic) * energy)
            .collect::<Vec<_>>()
            .try_into()
            .expect("three")
    });
    add(
        add(
            mul(
                [environment[0], environment[1], environment[2]],
                reflectance,
            ),
            direct_ggx(base, control.metallic, control.roughness, normal, view),
        ),
        diffuse,
    )
}

/// The output target is UNORM: rounding happens only at this final boundary.
pub fn expected_rgba8(control: Control, input: &crate::scene::Input) -> [u8; 4] {
    encode_rgba8(expected_from_input(control, input))
}

pub fn encode_rgba8(rgb: [f32; 3]) -> [u8; 4] {
    [
        (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        255,
    ]
}

pub fn verify_rgb(actual: [u8; 4], expected: [u8; 4], tolerance: u8) -> Result<(), String> {
    if actual[..3]
        .iter()
        .zip(expected[..3].iter())
        .any(|(actual, expected)| actual.abs_diff(*expected) > tolerance)
    {
        return Err(format!(
            "found {actual:?}, expected {expected:?}, tolerance {tolerance}"
        ));
    }
    Ok(())
}

pub fn sample_rgba8(bytes: &[u8], extent: [u32; 2], uv: [f32; 2]) -> [f32; 3] {
    let x = (uv[0].clamp(0.0, 1.0) * extent[0] as f32 - 0.5).clamp(0.0, extent[0] as f32 - 1.0);
    let y = (uv[1].clamp(0.0, 1.0) * extent[1] as f32 - 0.5).clamp(0.0, extent[1] as f32 - 1.0);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(extent[0] - 1), (y0 + 1).min(extent[1] - 1));
    let texel = |x: u32, y: u32| {
        let at = ((y * extent[0] + x) * 4) as usize;
        [
            bytes[at] as f32 / 255.0,
            bytes[at + 1] as f32 / 255.0,
            bytes[at + 2] as f32 / 255.0,
        ]
    };
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let (a, b, c, d) = (texel(x0, y0), texel(x1, y0), texel(x0, y1), texel(x1, y1));
    [0, 1, 2].map(|channel| {
        let top = a[channel] + (b[channel] - a[channel]) * tx;
        let bottom = c[channel] + (d[channel] - c[channel]) * tx;
        top + (bottom - top) * ty
    })
}

/// Decodes an IEEE-754 binary16 component without relying on a product upload
/// helper. The controls deliberately use exact powers of two, so an input-byte
/// mismatch cannot be hidden by a floating point encoder choice.
pub fn half(value: u16) -> f32 {
    let sign = if value & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((value >> 10) & 0x1f) as i32;
    let fraction = (value & 0x03ff) as f32;
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        31 => f32::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Independent RGBA16F sampler matching the declared lat-long address facts:
/// repeat in U, clamp in V, and bilinear filtering within a saved mip.
pub fn sample_rgba16f(bytes: &[u8], extent: [u32; 2], uv: [f32; 2], repeat_u: bool) -> [f32; 4] {
    let u = if repeat_u {
        uv[0].rem_euclid(1.0)
    } else {
        uv[0].clamp(0.0, 1.0)
    };
    let v = uv[1].clamp(0.0, 1.0);
    let x = u * extent[0] as f32 - 0.5;
    let y = v * extent[1] as f32 - 0.5;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = x - x.floor();
    let ty = y - y.floor();
    let at = |x: i32, y: i32| {
        let x = if repeat_u {
            x.rem_euclid(extent[0] as i32)
        } else {
            x.clamp(0, extent[0] as i32 - 1)
        };
        let y = y.clamp(0, extent[1] as i32 - 1);
        let p = (y as usize * extent[0] as usize + x as usize) * 8;
        [0, 1, 2, 3].map(|c| half(u16::from_le_bytes([bytes[p + c * 2], bytes[p + c * 2 + 1]])))
    };
    let a = at(x0, y0);
    let b = at(x0 + 1, y0);
    let c = at(x0, y0 + 1);
    let d = at(x0 + 1, y0 + 1);
    a.into_iter()
        .zip(b)
        .zip(c)
        .zip(d)
        .map(|(((a, b), c), d)| (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty)
        .collect::<Vec<_>>()
        .try_into()
        .expect("four")
}

/// Independent fractional-LOD trilinear sampler over the persisted mip chain.
pub fn sample_mips_rgba16f(mips: &[(&[u8], [u32; 2])], uv: [f32; 2], lod: f32) -> [f32; 4] {
    let base = lod.floor().clamp(0.0, (mips.len() - 1) as f32) as usize;
    let next = (base + 1).min(mips.len() - 1);
    let a = sample_rgba16f(mips[base].0, mips[base].1, uv, true);
    let b = sample_rgba16f(mips[next].0, mips[next].1, uv, true);
    mix4(a, b, lod.fract())
}
fn mix4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| x + (y - x) * t)
        .collect::<Vec<_>>()
        .try_into()
        .expect("four")
}

fn direct_ggx(base: [f32; 3], metallic: f32, roughness: f32, n: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    let l = normalize([0.6, 0.0, 0.8]);
    let h = normalize(add(l, v));
    let (nl, nv) = (dot(n, l).max(0.0), dot(n, v).max(0.0));
    if nl == 0.0 || nv == 0.0 {
        return [0.0; 3];
    }
    let alpha2 = roughness.clamp(0.04, 1.0).powi(4);
    let nh = dot(n, h).clamp(0.0, 1.0);
    let vh = dot(v, h).clamp(0.0, 1.0);
    let distribution = alpha2 / (std::f32::consts::PI * (nh * nh * (alpha2 - 1.0) + 1.0).powi(2));
    let visibility = 0.5
        / (nl * (nv * nv * (1.0 - alpha2) + alpha2).sqrt()
            + nv * (nl * nl * (1.0 - alpha2) + alpha2).sqrt());
    let f0 = mix3([0.04; 3], base, metallic.clamp(0.0, 1.0));
    let fresnel = f0.map(|f| f + (1.0 - f) * (1.0 - vh).powi(5));
    let diffuse = mul(mul(fresnel.map(|f| 1.0 - f), [1.0 - metallic; 3]), base)
        .map(|x| x / std::f32::consts::PI);
    add(diffuse, fresnel.map(|x| x * distribution * visibility)).map(|x| x * nl * 3.0)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt();
    v.map(|x| x / l)
}
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| x + (y - x) * t)
        .collect::<Vec<_>>()
        .try_into()
        .expect("three")
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| x + y)
        .collect::<Vec<_>>()
        .try_into()
        .expect("three")
}
fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| x * y)
        .collect::<Vec<_>>()
        .try_into()
        .expect("three")
}

#[cfg(test)]
#[path = "tests/oracle.rs"]
mod tests;
