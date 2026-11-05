use super::*;

fn input(color: [u8; 4], roughness: f64, transform: [[f64; 4]; 4]) -> Input {
    Input {
        positions: [[-0.4, -0.4, 0.5], [0.2, -0.4, 0.05], [-0.4, 0.35, 0.5]],
        normals: [[0.6, 0.0, 0.8]; 3],
        transform,
        color,
        roughness,
    }
}

fn identity() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[test]
fn pbr_oracle_accepts_its_ideal_forward_and_deferred_images() {
    let source = input([180, 100, 40, 128], 0.45, identity());
    for route in [Route::Forward, Route::Deferred] {
        let (expected, mask) = image(source, route);
        verify(&expected, &expected, &mask, "ideal", route).unwrap();
    }
}

#[test]
fn pbr_metal_roughness_and_normal_transform_change_the_oracle() {
    let dielectric = image(input([180, 100, 40, 0], 0.20, identity()), Route::Forward).0;
    let metallic = image(input([180, 100, 40, 255], 0.20, identity()), Route::Forward).0;
    let rough = image(input([180, 100, 40, 255], 0.85, identity()), Route::Forward).0;
    let transformed = image(
        input(
            [180, 100, 40, 255],
            0.85,
            [
                [1.0392304845, 0.0, -0.6, 0.0],
                [0.0, 0.7, 0.0, 0.0],
                [0.25, 0.0, 0.4330127019, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
        Route::Forward,
    )
    .0;
    assert_ne!(dielectric, metallic);
    assert_ne!(metallic, rough);
    assert_ne!(rough, transformed);
}

#[test]
fn inverse_transpose_normal_rejects_the_ordinary_model_matrix() {
    let transform = [
        [1.0392304845, 0.0, -0.6, 0.0],
        [0.0, 0.7, 0.0, 0.0],
        [0.25, 0.0, 0.4330127019, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let source = input([180, 100, 40, 255], 0.85, transform);
    let (expected, mask) = image(source, Route::Forward);
    let local = source.normals[0];
    let ordinary = normalize([
        transform[0][0] * local[0] + transform[1][0] * local[1] + transform[2][0] * local[2],
        transform[0][1] * local[0] + transform[1][1] * local[1] + transform[2][1] * local[2],
        transform[0][2] * local[0] + transform[1][2] * local[1] + transform[2][2] * local[2],
    ]);
    let (wrong, _) = image_with_world_normal(source, Route::Forward, ordinary);
    assert!(verify(&wrong, &expected, &mask, "ordinary matrix", Route::Forward).is_err());
}

#[test]
fn pbr_oracle_rejects_a_tampered_stable_readback() {
    let (expected, mask) = image(
        input([180, 100, 40, 128], 0.45, identity()),
        Route::Deferred,
    );
    let pixel = mask
        .iter()
        .enumerate()
        .find(|(index, stable)| **stable != 0 && expected[index * 4..index * 4 + 4] != [0; 4])
        .expect("stable PBR foreground")
        .0;
    let mut actual = expected.clone();
    actual[pixel * 4] = actual[pixel * 4].saturating_add(4);
    assert!(verify(&actual, &expected, &mask, "tampered", Route::Deferred).is_err());
}

#[test]
fn pbr_oracle_rejects_a_malformed_readback_shape() {
    let (expected, mask) = image(
        input([180, 100, 40, 128], 0.45, identity()),
        Route::Deferred,
    );
    assert!(
        verify(
            &expected[..expected.len() - 1],
            &expected,
            &mask,
            "truncated",
            Route::Deferred
        )
        .is_err()
    );
}
