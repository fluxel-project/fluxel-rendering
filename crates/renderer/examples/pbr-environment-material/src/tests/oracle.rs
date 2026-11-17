use super::{
    Control, encode_rgba8, expected_from_input, expected_from_input_with_lod, expected_rgba8, half,
    sample_mips_rgba16f, sample_rgba16f, verify_rgb,
};
use crate::scene;

#[test]
fn metallic_roughness_camera_mip_and_lut_controls_are_distinct() {
    let base = Control {
        name: "base",
        metallic: 0.0,
        roughness: 0.1,
        camera: [0., 0., 2.],
        lowest_mip_only: false,
        lut_variant: false,
        diffuse: false,
    };
    let variations = [
        Control {
            metallic: 1.0,
            ..base
        },
        Control {
            roughness: 0.9,
            ..base
        },
        Control {
            camera: [1., 0., 1.],
            ..base
        },
        Control {
            lowest_mip_only: true,
            roughness: 0.9,
            ..base
        },
        Control {
            lut_variant: true,
            ..base
        },
        Control {
            diffuse: true,
            ..base
        },
    ];
    let reference = expected_from_input(base, &scene::input(base).unwrap());
    for control in variations {
        assert_ne!(
            expected_from_input(control, &scene::input(control).unwrap()),
            reference,
            "{control:?}"
        );
    }
}

#[test]
fn binary16_control_bytes_have_declared_exact_values() {
    assert_eq!(half(0x4000), 2.0);
    assert_eq!(half(0x3c00), 1.0);
    assert_eq!(half(0x3800), 0.5);
    assert_eq!(half(0x3400), 0.25);
}

#[test]
fn sampler_repeats_u_clamps_v_and_interpolates_fractional_lod() {
    let texel = |r: u16| {
        [
            r.to_le_bytes(),
            0x3c00u16.to_le_bytes(),
            0x3800u16.to_le_bytes(),
            0x3c00u16.to_le_bytes(),
        ]
        .concat()
    };
    let mip0 = [texel(0x3c00), texel(0x4000)].concat();
    let mip1 = texel(0x3800);
    assert_eq!(sample_rgba16f(&mip0, [2, 1], [-0.25, 9.], true)[0], 2.0);
    assert_eq!(sample_rgba16f(&mip0, [2, 1], [1.25, -9.], true)[0], 1.0);
    assert_eq!(
        sample_mips_rgba16f(&[(&mip0, [2, 1]), (&mip1, [1, 1])], [0.25, 0.5], 0.5)[0],
        0.75
    );
}

#[test]
fn saved_payload_oracle_rejects_no_ibl_wrong_lod_and_ignored_camera_with_runtime_tolerance() {
    let control = Control {
        name: "payload",
        metallic: 1.0,
        roughness: 0.9,
        camera: [1., 0., 1.],
        lowest_mip_only: false,
        lut_variant: false,
        diffuse: false,
    };
    let input = scene::input(control).unwrap();
    let correct = expected_rgba8(control, &input);
    let no_ibl = encode_rgba8(super::direct_ggx(
        [179. / 255., 102. / 255., 51. / 255.],
        control.metallic,
        control.roughness,
        [0., 0., 1.],
        super::normalize(control.camera),
    ));
    assert!(
        verify_rgb(no_ibl, correct, 5).is_err(),
        "removing saved radiance/LUT must exceed runtime tolerance"
    );
    let wrong_lod = encode_rgba8(expected_from_input_with_lod(control, &input, 0.0));
    assert!(
        verify_rgb(wrong_lod, correct, 5).is_err(),
        "incorrect payload LOD must exceed runtime tolerance"
    );
    let camera_control = Control {
        roughness: 0.1,
        ..control
    };
    let camera_input = scene::input(camera_control).unwrap();
    let ignored_camera = Control {
        camera: [0., 0., 1.],
        ..camera_control
    };
    assert!(
        verify_rgb(
            expected_rgba8(ignored_camera, &camera_input),
            expected_rgba8(camera_control, &camera_input),
            5
        )
        .is_err(),
        "ignoring camera reflection must exceed runtime tolerance"
    );
}
