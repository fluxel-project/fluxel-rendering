use crate::{oracle, scene};

#[test]
fn multi_reference_uses_actual_local_streams_and_per_record_translations() {
    let single = scene::input(false, scene::EYE).unwrap();
    let multi = scene::multi_input(false, scene::EYE).unwrap();
    let probes = oracle::reference(&multi).unwrap();
    let combined = super::raster::Raster::from_input(&single).unwrap();
    let separate = super::raster::Raster::from_input(&multi).unwrap();
    assert_eq!(probes.len(), 12);
    for probe in probes {
        let expected = combined.at(probe.pixel).unwrap();
        let actual = separate.at(probe.pixel).unwrap();
        assert_eq!(actual.sphere, expected.sphere);
        for (a, b) in actual.pos.into_iter().zip(expected.pos) {
            assert!(
                (a - b).abs() < 1e-5,
                "local+model reconstruction at {:?}",
                probe.pixel
            );
        }
    }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn hand_checked_base_and_reflection_follow_authored_shader() {
    let input = scene::input(false, scene::EYE).unwrap();
    // `base()` returns tint, rather than multiplying tint by a second base.
    assert_eq!(input.tint[..3], input.base_color);

    let forward = super::shading::reflection([0., 0., 1.], [0., 0., 1.]);
    assert_eq!(forward, [0., 0., 1.]);
    // Hand calculation: 2 * Z * dot(Z, [3/5,0,4/5]) - [3/5,0,4/5].
    let oblique = super::shading::reflection([0., 0., 1.], [0.6, 0., 0.8]);
    assert!(close(oblique[0], -0.6) && close(oblique[1], 0.) && close(oblique[2], 0.8));
}

#[test]
fn hand_checked_latlong_sampling_repeats_u() {
    let texels = [10, 0, 0, 255, 110, 0, 0, 255];
    let left = super::sampling::rgba8_repeat_u(&texels, [2, 1], [-0.25, 0.5]);
    let right = super::sampling::rgba8_repeat_u(&texels, [2, 1], [0.75, 0.5]);
    assert!(close(left[0], right[0]));
    assert!(close(left[0], 110. / 255.));
}
#[test]
fn independent_controls_reject_missing_semantics() {
    let x = scene::input(false, [0., 0.35, 7.5]).unwrap();
    oracle::controls_reject(&x).unwrap();
    assert_eq!(oracle::reference(&x).unwrap().len(), 12);
}
