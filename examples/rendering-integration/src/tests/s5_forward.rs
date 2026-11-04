use super::*;

#[test]
fn frozen_expected_image_passes_the_complete_stable_pixel_oracle() {
    let expected = expected_image();
    let mask = stable_mask();
    verify_stable_image(&expected, &expected, &mask)
        .expect("the independent expected image must satisfy its own stable mask");
}

#[test]
fn stable_interior_and_background_readback_tampering_are_refused() {
    let expected = expected_image();
    let mask = stable_mask();
    let mut interior = expected.clone();
    let interior_pixel = mask
        .iter()
        .enumerate()
        .find(|(pixel, stable)| {
            **stable != 0 && &expected[*pixel * 4..*pixel * 4 + 4] != CLEAR.as_slice()
        })
        .map(|(pixel, _)| pixel)
        .expect("fixture must contain a stable triangle-interior pixel");
    interior[interior_pixel * 4] ^= 0xff;
    assert!(verify_stable_image(&interior, &expected, &mask).is_err());

    let mut background = expected.clone();
    let background_pixel = mask
        .iter()
        .enumerate()
        .find(|(pixel, stable)| {
            **stable != 0 && &expected[*pixel * 4..*pixel * 4 + 4] == CLEAR.as_slice()
        })
        .map(|(pixel, _)| pixel)
        .expect("fixture must contain a stable background pixel");
    background[background_pixel * 4 + 2] ^= 0xff;
    assert!(verify_stable_image(&background, &expected, &mask).is_err());
}

#[test]
fn input_mutation_is_rejected_against_the_frozen_visual_oracle() {
    let mut geometry = [
        interleaved_triangle(),
        separate_triangle(),
        mixed_triangle(),
        position_triangle(),
    ];
    verify_fixture_inputs(&geometry).expect("fixture geometry must match its frozen oracle");
    geometry[0].streams[0][0] ^= 0x01;
    assert!(verify_fixture_inputs(&geometry).is_err());
}
