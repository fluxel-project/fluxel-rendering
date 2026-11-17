//! Independent aniso-1 reference over actual tunnel triangles and base bytes.
//!
//! Visibility includes homogeneous clipping. Probe derivative quads must stay
//! inside one visible original primitive; fine/coarse LOD precision variants
//! are screened on CPU before any GPU execution.

use crate::scene::{self, Input};

mod raster;
use raster::Raster;

pub const TOLERANCE: u8 = 4;

#[derive(Clone, Debug)]
pub struct Probe {
    pub pixel: [u32; 2],
    pub uv: [f32; 2],
    pub lod: f32,
    pub expected: [u8; 4],
    pub deferred_expected: [u8; 4],
    pub margin: f32,
    pub rejected_controls: [bool; 4],
}

pub fn reference(input: &Input) -> Result<Vec<Probe>, String> {
    let raster = Raster::from_input(input)?;
    let chain = mips(&input.image, input.extent)?;
    let mut candidates = Vec::new();
    for fragment in raster.pixels.iter().flatten().copied() {
        if fragment.margin < 0.12 {
            continue;
        }
        let x = fragment.pixel[0] & !1;
        let y = fragment.pixel[1] & !1;
        let quad = [
            raster.at(x, y),
            raster.at(x + 1, y),
            raster.at(x, y + 1),
            raster.at(x + 1, y + 1),
        ];
        let [Some(a), Some(b), Some(c), Some(d)] = quad else {
            continue;
        };
        if [a, b, c, d].iter().any(|v| v.triangle != fragment.triangle) {
            continue;
        }
        let ddx = if fragment.pixel[1] & 1 == 0 {
            sub(b.uv, a.uv)
        } else {
            sub(d.uv, c.uv)
        };
        let ddy = if fragment.pixel[0] & 1 == 0 {
            sub(c.uv, a.uv)
        } else {
            sub(d.uv, b.uv)
        };
        let scale = |d: [f32; 2]| [d[0] * input.extent[0] as f32, d[1] * input.extent[1] as f32];
        let (_, ideal) = footprint_lods(scale(ddx), scale(ddy));
        if !ideal.is_finite() {
            continue;
        }
        let lod = ideal.clamp(0., (chain.len() - 1) as f32);
        let mut expected = encode(trilinear(
            &chain,
            input.extent,
            fragment.uv,
            lod,
            false,
            false,
        ));
        // RHI-09 stores specular information in source alpha. This sampler
        // consumer is opaque; it does not reinterpret that channel as opacity.
        expected[3] = 255;
        // The legacy Deferred colored-albedo consumer has a documented
        // diagnostic lighting pass. Model its actual packed normal/depth rule,
        // separately from the direct-color Forward result.
        let packed = ((fragment.normal_z * 0.5 + 0.5) * 255.0 + 0.5).floor() / 255.0;
        let decoded_abs = (packed * 2.0 - 1.0).abs();
        let light = 0.2 + 0.6 * decoded_abs + 0.2 * (1.0 - fragment.depth);
        let deferred_expected = core::array::from_fn(|c| {
            if c == 3 {
                255
            } else {
                (expected[c] as f32 * light + 0.5).floor().clamp(0., 255.) as u8
            }
        });
        let mut models = Vec::new();
        for x in [sub(b.uv, a.uv), sub(d.uv, c.uv)] {
            for y in [sub(c.uv, a.uv), sub(d.uv, b.uv)] {
                let (axis, ellipse) = footprint_lods(scale(x), scale(y));
                for estimate in [axis, ellipse] {
                    for delta in [-1.0 / 32.0, 0.0, 1.0 / 32.0] {
                        models.push((estimate + delta).clamp(0., (chain.len() - 1) as f32));
                    }
                }
            }
        }
        // Source-only uncertainty controls cover the axis-max approximation,
        // the orthogonal footprint and every coarse/fine 2x2 derivative pair.
        // Tolerance remains fixed; no observed GPU value selects a probe.
        if models.iter().any(|value| {
            let alternate = encode(trilinear(
                &chain,
                input.extent,
                fragment.uv,
                *value,
                false,
                false,
            ));
            alternate[..3]
                .iter()
                .zip(expected[..3].iter())
                .any(|(a, b)| a.abs_diff(*b) > 1)
        }) {
            continue;
        }
        let shade = |color: [u8; 4]| {
            core::array::from_fn(|c| {
                if c == 3 {
                    255
                } else {
                    (color[c] as f32 * light + 0.5).floor().clamp(0., 255.) as u8
                }
            })
        };
        let mut rejected_controls = [true; 4];
        for model in &models {
            let bad = [
                encode(trilinear(
                    &chain,
                    input.extent,
                    fragment.uv,
                    0.,
                    false,
                    false,
                )),
                encode(nearest_mip(&chain, input.extent, fragment.uv, *model)),
                encode(trilinear(
                    &chain,
                    input.extent,
                    fragment.uv,
                    *model,
                    true,
                    false,
                )),
                encode(trilinear(
                    &chain[..1],
                    input.extent,
                    fragment.uv,
                    0.,
                    false,
                    false,
                )),
            ];
            for (index, bad) in bad.into_iter().enumerate() {
                rejected_controls[index] &=
                    differs(expected, bad) && differs(deferred_expected, shade(bad));
            }
        }
        candidates.push(Probe {
            pixel: fragment.pixel,
            uv: fragment.uv,
            lod,
            expected,
            deferred_expected,
            margin: fragment.margin,
            rejected_controls,
        });
    }
    candidates.sort_by(|a, b| b.margin.total_cmp(&a.margin));
    let mut selected: Vec<Probe> = Vec::new();
    // Different probes prove different functions. Requiring every near-view
    // probe to reject LOD0 would remove magnification coverage by construction.
    for extreme in [false, true] {
        let candidate = candidates.iter().min_by(|a, b| {
            if extreme {
                b.lod.total_cmp(&a.lod)
            } else {
                a.lod.total_cmp(&b.lod)
            }
        });
        if let Some(candidate) = candidate {
            if separated(&selected, candidate) {
                selected.push(candidate.clone());
            }
        }
    }
    for control in 0..4 {
        if selected.iter().any(|p| p.rejected_controls[control]) {
            continue;
        }
        let candidate = candidates
            .iter()
            .find(|p| p.rejected_controls[control] && separated(&selected, p))
            .ok_or_else(|| {
                format!("no stable original-tunnel probe rejects negative control {control}")
            })?;
        selected.push(candidate.clone());
    }
    for candidate in &candidates {
        if selected.len() >= 8 {
            break;
        }
        if separated(&selected, candidate) {
            selected.push(candidate.clone());
        }
    }
    if selected.len() < 4 {
        return Err(format!(
            "tunnel has fewer than four stable perspective sampler probes (candidates={})",
            candidates.len()
        ));
    }
    Ok(selected)
}

fn separated(probes: &[Probe], candidate: &Probe) -> bool {
    probes.iter().all(|p| {
        p.pixel[0]
            .abs_diff(candidate.pixel[0])
            .max(p.pixel[1].abs_diff(candidate.pixel[1]))
            >= 16
    })
}

pub fn verify(raw: &[u8], probes: &[Probe], deferred: bool) -> Result<(), String> {
    if raw.len() != (scene::EXTENT[0] * scene::EXTENT[1] * 4) as usize {
        return Err("unexpected tunnel readback shape".into());
    }
    for probe in probes {
        let at = ((probe.pixel[1] * scene::EXTENT[0] + probe.pixel[0]) * 4) as usize;
        let actual: [u8; 4] = raw[at..at + 4].try_into().unwrap();
        let expected = if deferred {
            probe.deferred_expected
        } else {
            probe.expected
        };
        if actual[..3]
            .iter()
            .zip(expected[..3].iter())
            .any(|(a, e)| a.abs_diff(*e) > TOLERANCE)
            || actual[3] != expected[3]
        {
            return Err(format!(
                "tunnel probe {:?}: got {actual:?}, expected {:?}, tolerance {TOLERANCE}",
                probe.pixel, expected
            ));
        }
    }
    Ok(())
}

pub fn write(output: &std::path::Path, probes: &[Probe]) -> Result<(), Box<dyn std::error::Error>> {
    let values = probes.iter().map(|p| format!("{{\"pixel\":{:?},\"uv\":{:?},\"lod\":{:.6},\"expected\":{:?},\"deferredExpected\":{:?},\"rejectedControls\":{:?},\"edgeMargin\":{:.6}}}", p.pixel, p.uv, p.lod, p.expected, p.deferred_expected, p.rejected_controls, p.margin)).collect::<Vec<_>>().join(",");
    std::fs::write(
        output.join("oracle.json"),
        format!(
            "{{\"schemaVersion\":1,\"scope\":\"clipped tunnel visibility and same-original-triangle 2x2 perspective derivative quads\",\"anisotropy\":1,\"tolerance\":{TOLERANCE},\"probes\":[{values}]}}\n"
        ),
    )?;
    Ok(())
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
/// Direct3D 11.3 section 7.18.11: orthogonalize the texture Jacobian.
/// The larger singular value is the ideal isotropic footprint; raw column
/// lengths are retained only as a source-only approximation control.
fn footprint_lods(dx: [f32; 2], dy: [f32; 2]) -> (f32, f32) {
    let a = dx[0] * dx[0] + dx[1] * dx[1];
    let b = dy[0] * dy[0] + dy[1] * dy[1];
    let dot = dx[0] * dy[0] + dx[1] * dy[1];
    let largest = ((a + b) + ((a - b) * (a - b) + 4.0 * dot * dot).sqrt()) * 0.5;
    (0.5 * a.max(b).log2(), 0.5 * largest.log2())
}

fn mips(base: &[u8], extent: [u32; 2]) -> Result<Vec<Vec<u8>>, String> {
    if base.len() != (extent[0] * extent[1] * 4) as usize {
        return Err("base source bytes do not match KTX extent".into());
    }
    let mut result = vec![base.to_vec()];
    let mut width = extent[0];
    let mut height = extent[1];
    while width > 1 || height > 1 {
        let next_w = (width / 2).max(1);
        let next_h = (height / 2).max(1);
        let prior = result.last().unwrap();
        let mut next = vec![0; (next_w * next_h * 4) as usize];
        for y in 0..next_h {
            for x in 0..next_w {
                for channel in 0..4 {
                    let sum: u16 = [0, 1]
                        .into_iter()
                        .flat_map(|dy| {
                            [0, 1].into_iter().map(move |dx| {
                                prior[(((y * 2 + dy).min(height - 1) * width
                                    + (x * 2 + dx).min(width - 1))
                                    * 4
                                    + channel) as usize] as u16
                            })
                        })
                        .sum();
                    next[((y * next_w + x) * 4 + channel) as usize] = ((sum + 2) / 4) as u8;
                }
            }
        }
        result.push(next);
        width = next_w;
        height = next_h;
    }
    Ok(result)
}
fn trilinear(
    chain: &[Vec<u8>],
    base: [u32; 2],
    uv: [f32; 2],
    lod: f32,
    clamp: bool,
    nearest: bool,
) -> [f32; 4] {
    let low = lod.floor().clamp(0., (chain.len() - 1) as f32) as usize;
    let high = (low + 1).min(chain.len() - 1);
    let a = bilinear(
        &chain[low],
        [(base[0] >> low).max(1), (base[1] >> low).max(1)],
        uv,
        clamp,
        nearest,
    );
    let b = bilinear(
        &chain[high],
        [(base[0] >> high).max(1), (base[1] >> high).max(1)],
        uv,
        clamp,
        nearest,
    );
    mix(a, b, lod.fract())
}
fn nearest_mip(chain: &[Vec<u8>], base: [u32; 2], uv: [f32; 2], lod: f32) -> [f32; 4] {
    let level = lod.round().clamp(0., (chain.len() - 1) as f32) as usize;
    bilinear(
        &chain[level],
        [(base[0] >> level).max(1), (base[1] >> level).max(1)],
        uv,
        false,
        false,
    )
}
fn bilinear(bytes: &[u8], extent: [u32; 2], uv: [f32; 2], clamp: bool, nearest: bool) -> [f32; 4] {
    let p = [
        uv[0] * extent[0] as f32 - 0.5,
        uv[1] * extent[1] as f32 - 0.5,
    ];
    if nearest {
        return texel(
            bytes,
            extent,
            p[0].round() as i32,
            p[1].round() as i32,
            clamp,
        );
    }
    let lo = [p[0].floor() as i32, p[1].floor() as i32];
    mix(
        mix(
            texel(bytes, extent, lo[0], lo[1], clamp),
            texel(bytes, extent, lo[0] + 1, lo[1], clamp),
            p[0] - p[0].floor(),
        ),
        mix(
            texel(bytes, extent, lo[0], lo[1] + 1, clamp),
            texel(bytes, extent, lo[0] + 1, lo[1] + 1, clamp),
            p[0] - p[0].floor(),
        ),
        p[1] - p[1].floor(),
    )
}
fn texel(bytes: &[u8], extent: [u32; 2], x: i32, y: i32, clamp: bool) -> [f32; 4] {
    let at = |coordinate: i32, length: u32| {
        if clamp {
            coordinate.clamp(0, length as i32 - 1) as u32
        } else {
            let period = length as i32 * 2;
            let folded = coordinate.rem_euclid(period);
            if folded < length as i32 {
                folded as u32
            } else {
                (period - 1 - folded) as u32
            }
        }
    };
    let i = ((at(y, extent[1]) * extent[0] + at(x, extent[0])) * 4) as usize;
    core::array::from_fn(|c| bytes[i + c] as f32 / 255.)
}
fn mix(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    core::array::from_fn(|c| a[c] + (b[c] - a[c]) * t)
}
fn encode(sample: [f32; 4]) -> [u8; 4] {
    sample.map(|value| (value.clamp(0., 1.) * 255. + 0.5).floor() as u8)
}
fn differs(a: [u8; 4], b: [u8; 4]) -> bool {
    a[..3]
        .iter()
        .zip(b[..3].iter())
        .any(|(a, b)| a.abs_diff(*b) > TOLERANCE)
}

#[cfg(test)]
#[path = "tests/oracle.rs"]
mod tests;
