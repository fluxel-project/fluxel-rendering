use super::*;
use crate::scene::{self, Case};

#[test]
fn srgb_midpoint_decodes_texels_before_filter_and_preserves_alpha() {
    let sample = sample_image(
        &[0, 0, 0, 16, 255, 255, 255, 240],
        [2, 1],
        TextureEncoding::Srgb,
        [0.5, 0.5],
    );
    assert_eq!(sample[..3], [0.5; 3]);
    assert!((sample[3] - 128. / 255.).abs() < 1e-12);
    assert!((decode(0.5) - 0.21404114048223255).abs() < 1e-12);
}

#[test]
fn binary16_rounding_has_an_independent_hand_computed_tie() {
    assert_eq!(half_round(1. + 1. / 2048.), 1.);
    assert_eq!(half_round(1. + 3. / 2048.), 1. + 2. / 1024.);
    assert_eq!(half_round(2f64.powi(-25)), 0.);
}

#[test]
fn selected_runtime_probes_observe_every_role_and_orm_channel() {
    let baseline = scene::input(Case::Baseline).unwrap();
    let expected = reference(&baseline).unwrap();
    assert_eq!(expected.len(), 16);
    for (i, a) in expected.iter().filter(|p| p.object < 2).enumerate() {
        assert!(stable_sampler_arithmetic(&baseline, a.object, a.uv));
        if i % 3 == 1 {
            assert!(
                a.uv.into_iter().all(|v| v > 0.25 && v < 0.75),
                "middle probes mix both axes"
            );
        }
        for b in expected.iter().skip(i + 1).filter(|p| p.object == a.object) {
            let dx = i64::from(a.x) - i64::from(b.x);
            let dy = i64::from(a.y) - i64::from(b.y);
            assert!(dx * dx + dy * dy >= 16 * 16);
        }
    }
    for case in Case::ALL.into_iter().skip(1) {
        let changed = scene::input(case).unwrap();
        let probes = reference(&changed).unwrap();
        for (a, b) in expected.iter().zip(&probes) {
            assert_eq!(
                (a.object, a.x, a.y, a.uv),
                (b.object, b.x, b.y, b.uv),
                "controls use identical independent probe coordinates"
            );
        }
        for deferred in [false, true] {
            let error = expected
                .iter()
                .zip(&probes)
                .flat_map(|(old, new)| {
                    let (a, b) = if deferred {
                        (old.deferred, new.deferred)
                    } else {
                        (old.forward, new.forward)
                    };
                    a.into_iter().zip(b).take(3).map(|(x, y)| x.abs_diff(y))
                })
                .max()
                .unwrap();
            assert!(
                error > TOLERANCE,
                "case {case:?} does not reject baseline reuse for deferred={deferred}"
            );
        }
    }
}

#[test]
fn wrong_resource_mapping_and_missing_normal_unpack_exceed_acceptance_tolerance() {
    let mut input = scene::input(Case::Baseline).unwrap();
    let probes = reference(&input).unwrap();
    input.sources[0].swap(0, 3);
    let wrong = reference(&input).unwrap();
    assert!(probes.iter().zip(wrong).any(|(a, b)| {
        a.forward[..3]
            .iter()
            .zip(b.forward)
            .any(|(&x, y)| x.abs_diff(y) > TOLERANCE)
    }));
    let unpacked = normalize([
        160. / 255. * 2. - 1.,
        112. / 255. * 2. - 1.,
        250. / 255. * 2. - 1.,
    ]);
    let encoded = normalize([160. / 255., 112. / 255., 250. / 255.]);
    let base = [0.2, 0.3, 0.4];
    let expected = direct_ggx(base, 0.2, 0.5, unpacked).map(encode);
    let wrong = direct_ggx(base, 0.2, 0.5, encoded).map(encode);
    assert!(
        expected
            .into_iter()
            .zip(wrong)
            .any(|(a, b)| a.abs_diff(b) > TOLERANCE)
    );
}

#[test]
fn chosen_probes_are_stable_at_eight_fractional_filter_coordinate_bits() {
    for case in Case::ALL {
        let input = scene::input(case).unwrap();
        for probe in reference(&input)
            .unwrap()
            .into_iter()
            .filter(|p| p.object < 2)
        {
            let weights = probe.uv.map(|u| (u * 2. - 0.5) * 256.);
            for x in [weights[0].floor(), weights[0].ceil()] {
                for y in [weights[1].floor(), weights[1].ceil()] {
                    let uv = [(x / 256. + 0.5) / 2., (y / 256. + 0.5) / 2.];
                    for deferred in [false, true] {
                        let ideal = if deferred {
                            probe.deferred
                        } else {
                            probe.forward
                        };
                        let rounded = shade(&input, probe.object, uv, deferred);
                        let error = ideal
                            .into_iter()
                            .zip(rounded)
                            .take(3)
                            .map(|(a, b)| a.abs_diff(b))
                            .max()
                            .unwrap();
                        assert!(
                            error <= TOLERANCE - 1,
                            "unstable {case:?} ({},{}) deferred={deferred}: ideal={ideal:?} rounded={rounded:?}",
                            probe.x,
                            probe.y
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn near_endpoint_filter_rounding_can_amplify_one_packed_normal_code() {
    let input = scene::input(Case::Baseline).unwrap();
    let uv = [0.7495659788334629, 0.2989430203453788];
    assert_eq!(shade(&input, 1, uv, true), [94, 177, 133, 255]);
    assert_eq!(shade(&input, 1, [0.75, uv[1]], true), [98, 184, 138, 255]);
    assert!(filter_precision_error(&input, 1, uv) > TOLERANCE);
}
