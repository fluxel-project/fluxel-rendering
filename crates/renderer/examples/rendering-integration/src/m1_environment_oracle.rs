//! Independent numeric oracle for the environment integration workload.

use super::m1_environment_irradiance::{Case, Environment, BASE, NEW_ENV, OLD_ENV};

pub(crate) fn forward_expected(case: Case) -> [u8; 4] {
    shade(case.normal, BASE, case.metallic, 0.5, case.environment)
}

/// Deferred stores its material state in UNORM G-buffer attachments before
/// lighting. This deliberately models that route's own numeric boundary.
pub(crate) fn deferred_expected(case: Case) -> [u8; 4] {
    let base = BASE.map(unorm_roundtrip);
    let metallic = unorm_roundtrip(case.metallic);
    let roughness = unorm_roundtrip(0.5);
    let normal = normalize(
        case.normal
            .map(|value| unorm_roundtrip(value * 0.5 + 0.5) * 2.0 - 1.0),
    );
    shade(normal, base, metallic, roughness, case.environment)
}

fn shade(
    normal: [f64; 3],
    base: [f64; 3],
    metallic: f64,
    roughness: f64,
    environment_kind: Environment,
) -> [u8; 4] {
    let direct = direct_light(base, metallic, roughness, normal);
    let environment = match environment_kind {
        Environment::None => [0.0; 3],
        Environment::Old => sample_latlong(OLD_ENV, normal),
        Environment::Replacement => sample_latlong(NEW_ENV, normal),
    };
    [0, 1, 2]
        .map(|channel| {
            ((direct[channel] + environment[channel] * base[channel] * (1.0 - metallic))
                .clamp(0.0, 1.0)
                * 255.0)
                .round() as u8
        })
        .map_alpha(255)
}

fn direct_light(base: [f64; 3], metallic: f64, roughness: f64, normal: [f64; 3]) -> [f64; 3] {
    let light = [0.6, 0.0, 0.8];
    let view = [0.0, 0.0, 1.0];
    let half = normalize([0.6, 0.0, 1.8]);
    let no_l = dot(normal, light).clamp(0.0, 1.0);
    let no_v = dot(normal, view).clamp(0.0, 1.0);
    if no_l <= 0.0 || no_v <= 0.0 {
        return [0.0; 3];
    }
    let no_h = dot(normal, half).clamp(0.0, 1.0);
    let vo_h = dot(view, half).clamp(0.0, 1.0);
    let alpha_squared = roughness.powi(4);
    let term = no_h * no_h * (alpha_squared - 1.0) + 1.0;
    let distribution = alpha_squared / (std::f64::consts::PI * term * term);
    let visibility = 0.5
        / (no_l * (no_v * no_v * (1.0 - alpha_squared) + alpha_squared).sqrt()
            + no_v * (no_l * no_l * (1.0 - alpha_squared) + alpha_squared).sqrt());
    [0, 1, 2].map(|channel| {
        let f0 = 0.04 + (base[channel] - 0.04) * metallic;
        let fresnel = f0 + (1.0 - f0) * (1.0 - vo_h).powi(5);
        (((1.0 - fresnel) * (1.0 - metallic) * base[channel] / std::f64::consts::PI)
            + distribution * visibility * fresnel)
            * no_l
            * 3.0
    })
}

fn sample_latlong(bytes: [u8; 32], normal: [f64; 3]) -> [f64; 3] {
    let u = (normal[2].atan2(normal[0]) / std::f64::consts::TAU + 0.5).rem_euclid(1.0);
    let v = normal[1].clamp(-1.0, 1.0).acos() / std::f64::consts::PI;
    let x = u * 4.0 - 0.5;
    let y = (v * 2.0 - 0.5).clamp(0.0, 1.0);
    let x0 = x.floor() as i32;
    let y0 = y.floor() as usize;
    let fx = x - f64::from(x0);
    let fy = y - y0 as f64;
    [0, 1, 2].map(|channel| {
        let texel = |xx: i32, yy: usize| {
            f64::from(bytes[((yy * 4 + xx.rem_euclid(4) as usize) * 4) + channel]) / 255.0
        };
        let top = texel(x0, y0) + (texel(x0 + 1, y0) - texel(x0, y0)) * fx;
        let bottom = texel(x0, (y0 + 1).min(1))
            + (texel(x0 + 1, (y0 + 1).min(1)) - texel(x0, (y0 + 1).min(1))) * fx;
        top + (bottom - top) * fy
    })
}

fn unorm_roundtrip(value: f64) -> f64 {
    (value.clamp(0.0, 1.0) * 255.0).round() / 255.0
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn normalize(value: [f64; 3]) -> [f64; 3] {
    let length = dot(value, value).sqrt();
    value.map(|component| component / length)
}

trait Alpha {
    fn map_alpha(self, value: u8) -> [u8; 4];
}

impl Alpha for [u8; 3] {
    fn map_alpha(self, value: u8) -> [u8; 4] {
        [self[0], self[1], self[2], value]
    }
}
