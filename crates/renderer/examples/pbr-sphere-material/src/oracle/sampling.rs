pub fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1. } else { -1. };
    let e = ((bits >> 10) & 31) as i32;
    let f = (bits & 1023) as f32;
    match e {
        0 => sign * f * 2f32.powi(-24),
        31 => f32::NAN,
        _ => sign * (1. + f / 1024.) * 2f32.powi(e - 15),
    }
}
pub fn rgba8(b: &[u8], e: [u32; 2], uv: [f32; 2]) -> [f32; 4] {
    sample(e, uv, false, |x, y| {
        let i = ((y * e[0] + x) * 4) as usize;
        [0, 1, 2, 3].map(|c| b[i + c] as f32 / 255.)
    })
}
pub fn rgba8_repeat_u(b: &[u8], e: [u32; 2], uv: [f32; 2]) -> [f32; 4] {
    sample(e, uv, true, |x, y| {
        let i = ((y * e[0] + x) * 4) as usize;
        [0, 1, 2, 3].map(|c| b[i + c] as f32 / 255.)
    })
}
pub fn rgba16(b: &[u8], e: [u32; 2], uv: [f32; 2], repeat: bool) -> [f32; 4] {
    sample(e, uv, repeat, |x, y| {
        let i = ((y * e[0] + x) * 8) as usize;
        [0, 1, 2, 3].map(|c| half(u16::from_le_bytes([b[i + c * 2], b[i + c * 2 + 1]])))
    })
}
fn sample<F: Fn(u32, u32) -> [f32; 4]>(
    e: [u32; 2],
    uv: [f32; 2],
    repeat: bool,
    get: F,
) -> [f32; 4] {
    let u = if repeat {
        uv[0].rem_euclid(1.)
    } else {
        uv[0].clamp(0., 1.)
    };
    let v = uv[1].clamp(0., 1.);
    let x = u * e[0] as f32 - 0.5;
    let y = v * e[1] as f32 - 0.5;
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let at = |x: i32, y: i32| {
        get(
            if repeat {
                x.rem_euclid(e[0] as i32) as u32
            } else {
                x.clamp(0, e[0] as i32 - 1) as u32
            },
            y.clamp(0, e[1] as i32 - 1) as u32,
        )
    };
    let (a, b, c, d) = (
        at(x0, y0),
        at(x0 + 1, y0),
        at(x0, y0 + 1),
        at(x0 + 1, y0 + 1),
    );
    let (tx, ty) = (x - x.floor(), y - y.floor());
    [0, 1, 2, 3].map(|i| {
        let top = a[i] + (b[i] - a[i]) * tx;
        let bottom = c[i] + (d[i] - c[i]) * tx;
        top + (bottom - top) * ty
    })
}
pub fn mips(m: &[(&[u8], [u32; 2])], uv: [f32; 2], lod: f32) -> [f32; 4] {
    let low = lod.floor().clamp(0., (m.len() - 1) as f32) as usize;
    let high = (low + 1).min(m.len() - 1);
    let (a, b) = (
        rgba16(m[low].0, m[low].1, uv, true),
        rgba16(m[high].0, m[high].1, uv, true),
    );
    core::array::from_fn(|i| a[i] + (b[i] - a[i]) * lod.fract())
}
