use super::*;

fn stable_pixel(expected: &[u8], mask: &[u8], foreground: bool) -> usize {
    mask.iter()
        .enumerate()
        .find(|(index, stable)| {
            **stable != 0 && (expected[index * 4..index * 4 + 4] != [0; 4]) == foreground
        })
        .map(|(index, _)| index)
        .expect("fixture must include stable requested pixel")
}

#[test]
fn deferred_ideal_oracle_passes() {
    let (expected, mask) = image(false, Route::Deferred);
    verify(&expected, &expected, &mask, "flat", Route::Deferred).unwrap();
}

#[test]
fn deferred_foreground_allows_one_rgb_code_value() {
    let (expected, mask) = image(false, Route::Deferred);
    let mut actual = expected.clone();
    let pixel = stable_pixel(&expected, &mask, true);
    actual[pixel * 4] = actual[pixel * 4].saturating_sub(1);
    verify(&actual, &expected, &mask, "flat", Route::Deferred).unwrap();
}

#[test]
fn deferred_foreground_rejects_two_rgb_code_values() {
    let (expected, mask) = image(false, Route::Deferred);
    let mut actual = expected.clone();
    let pixel = stable_pixel(&expected, &mask, true);
    actual[pixel * 4] = actual[pixel * 4].saturating_sub(2);
    assert!(verify(&actual, &expected, &mask, "flat", Route::Deferred).is_err());
}

#[test]
fn deferred_background_and_alpha_remain_exact() {
    let (expected, mask) = image(false, Route::Deferred);
    let background = stable_pixel(&expected, &mask, false);
    let foreground = stable_pixel(&expected, &mask, true);
    let mut background_changed = expected.clone();
    background_changed[background * 4] = 1;
    assert!(
        verify(
            &background_changed,
            &expected,
            &mask,
            "flat",
            Route::Deferred
        )
        .is_err()
    );
    let mut alpha_changed = expected.clone();
    alpha_changed[foreground * 4 + 3] = 254;
    assert!(verify(&alpha_changed, &expected, &mask, "flat", Route::Deferred).is_err());
}

#[test]
fn route_and_plane_mismatches_reject() {
    let (deferred, mask) = image(false, Route::Deferred);
    let (forward, _) = image(false, Route::Forward);
    assert!(verify(&deferred, &forward, &mask, "flat", Route::Forward).is_err());
    let (tilt, _) = image(true, Route::Deferred);
    assert!(verify(&tilt, &deferred, &mask, "flat", Route::Deferred).is_err());
}

#[test]
fn deformation_parameter_replacement_is_not_oracle_equivalent() {
    let source = flat_positions();
    let first = bend_positions(source, [0.45, -0.15, 0.12, 0.0]);
    let second = bend_positions(source, [-0.30, 0.15, -0.10, 0.05]);
    let (first_expected, _) = image_for_positions(first, Route::Deferred);
    let (second_expected, second_mask) = image_for_positions(second, Route::Deferred);
    assert_ne!(first_expected, second_expected);
    assert!(
        verify(
            &first_expected,
            &second_expected,
            &second_mask,
            "bend parameter replacement",
            Route::Deferred,
        )
        .is_err()
    );
}

#[test]
fn code_node_channel_variants_are_not_oracle_equivalent() {
    let first = [16, 80, 144, 255];
    let second = [32, 40, 144, 255];
    let (first_expected, _) = image_for_color(flat_positions(), Route::Deferred, first);
    let (second_expected, second_mask) = image_for_color(flat_positions(), Route::Deferred, second);
    assert_ne!(first_expected, second_expected);
    assert!(
        verify(
            &first_expected,
            &second_expected,
            &second_mask,
            "code node static variant",
            Route::Deferred,
        )
        .is_err()
    );
}

fn bend_positions(vertices: [[f32; 3]; 3], parameters: [f32; 4]) -> [[f32; 3]; 3] {
    vertices.map(|[x, y, z]| {
        [
            x + parameters[0] * y * y + parameters[1],
            y,
            z + parameters[2] * x + parameters[3],
        ]
    })
}
