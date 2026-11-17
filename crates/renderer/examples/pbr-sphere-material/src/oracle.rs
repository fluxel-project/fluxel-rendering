//! CPU-only raster and lighting reference for the six-sphere acceptance image.
mod raster;
mod sampling;
mod shading;

use crate::scene::{self, Input};
use raster::Raster;

pub const TOLERANCE: u8 = 5;
#[derive(Clone, Debug)]
pub struct Probe {
    pub pixel: [u32; 2],
    pub sphere_index: usize,
    pub expected_forward: [u8; 4],
    pub expected_deferred: [u8; 4],
    pub edge_margin: f32,
    pub depth: f32,
}
impl Probe {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"pixel\":[{},{}],\"sphereIndex\":{},\"forward\":{:?},\"deferred\":{:?},\"edgeMargin\":{:.5},\"depth\":{:.6}}}",
            self.pixel[0],
            self.pixel[1],
            self.sphere_index,
            self.expected_forward,
            self.expected_deferred,
            self.edge_margin,
            self.depth
        )
    }
}

pub fn reference(input: &Input) -> Result<Vec<Probe>, String> {
    let raster = Raster::from_input(input)?;
    let mut result: Vec<Probe> = Vec::with_capacity(12);
    for sphere in 0..6 {
        let mut candidates = raster.candidates(sphere);
        candidates.sort_by(|a, b| b.quality.total_cmp(&a.quality));
        for fragment in candidates {
            if result.iter().any(|probe| {
                probe.sphere_index == sphere
                    && probe.pixel[0]
                        .abs_diff(fragment.pixel[0])
                        .max(probe.pixel[1].abs_diff(fragment.pixel[1]))
                        < 8
            }) {
                continue;
            }
            let expected_forward = shading::encode(shading::forward(input, &fragment)?);
            let expected_deferred = shading::encode(shading::deferred(input, &fragment)?);
            if expected_forward[..3].iter().all(|x| *x < 250)
                && expected_deferred[..3].iter().all(|x| *x < 250)
            {
                result.push(Probe {
                    pixel: fragment.pixel,
                    sphere_index: sphere,
                    expected_forward,
                    expected_deferred,
                    edge_margin: fragment.edge_margin,
                    depth: fragment.depth,
                });
                if result.iter().filter(|p| p.sphere_index == sphere).count() == 2 {
                    break;
                }
            }
        }
        if result.iter().filter(|p| p.sphere_index == sphere).count() < 2 {
            return Err(format!(
                "sphere {sphere} lacks two safe non-saturated interior probes"
            ));
        }
    }
    Ok(result)
}
pub fn verify(raw: &[u8], probes: &[Probe], deferred: bool) -> Result<(), String> {
    let required = scene::EXTENT[0] as usize * scene::EXTENT[1] as usize * 4;
    if raw.len() != required {
        return Err(format!(
            "readback has {} bytes, expected {required}",
            raw.len()
        ));
    }
    if probes
        .iter()
        .map(|p| p.sphere_index)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != 6
    {
        return Err("probe set does not cover all six spheres".into());
    }
    for probe in probes {
        let i = ((probe.pixel[1] * scene::EXTENT[0] + probe.pixel[0]) * 4) as usize;
        let actual: [u8; 4] = raw[i..i + 4].try_into().expect("four bytes");
        let expected = if deferred {
            probe.expected_deferred
        } else {
            probe.expected_forward
        };
        if actual[..3]
            .iter()
            .zip(expected[..3].iter())
            .any(|(a, e)| a.abs_diff(*e) > TOLERANCE)
            || actual[3] != expected[3]
        {
            return Err(format!(
                "{} sphere {} at {:?}: got {actual:?}, expected {expected:?}, tolerance {TOLERANCE}",
                if deferred { "deferred" } else { "forward" },
                probe.sphere_index,
                probe.pixel
            ));
        }
    }
    Ok(())
}
pub fn controls_reject(input: &Input) -> Result<(), String> {
    shading::controls_reject(input)
}
#[cfg(test)]
#[path = "tests/oracle.rs"]
mod tests;
