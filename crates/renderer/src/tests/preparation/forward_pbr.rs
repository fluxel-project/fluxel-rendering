//! PBR Forward normal-transform tests.

use super::forward_pbr::normal_matrix;

#[test]
fn affine_inverse_transpose_uses_world_from_local_not_clip_space() {
    let world = crate::RenderTransform::from_columns([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [7.0, 8.0, 9.0, 1.0],
    ])
    .unwrap();
    let normal = normal_matrix(world).expect("finite invertible affine transform");
    assert_eq!(normal[0][0], 0.5);
    assert_eq!(normal[1][1], 1.0 / 3.0);
    assert_eq!(normal[2][2], 0.25);
    assert_eq!(normal[3], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn normal_matrix_refuses_non_affine_and_singular_world_transforms() {
    let projective = crate::RenderTransform::from_columns([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    .unwrap();
    let singular = crate::RenderTransform::from_columns([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    .unwrap();
    assert!(normal_matrix(projective).is_none());
    assert!(normal_matrix(singular).is_none());
}

#[test]
fn pbr_rotated_nonuniform_normal_remains_orthogonal_to_transformed_tangent() {
    let world = crate::RenderTransform::from_columns([
        [0.0, 2.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [5.0, 6.0, 7.0, 1.0],
    ])
    .unwrap();
    let normal = normal_matrix(world).unwrap();
    // Local tangent (2,3,0) and normal (3,-2,0) are perpendicular.
    // Rz(90) * scale(2,3,4) sends the tangent to (-9,4,0).
    let transformed_normal: [f32; 3] =
        std::array::from_fn(|row| normal[0][row] * 3.0 - normal[1][row] * 2.0);
    assert!((transformed_normal[0] - 2.0 / 3.0).abs() < 1e-6);
    assert!((transformed_normal[1] - 1.5).abs() < 1e-6);
    assert!((-9.0 * transformed_normal[0] + 4.0 * transformed_normal[1]).abs() < 1e-6);
}

#[test]
fn pbr_representable_normal_matrix_survives_large_finite_model_scale() {
    let world = crate::RenderTransform::from_columns([
        [1e20, 0.0, 0.0, 0.0],
        [0.0, 1e20, 0.0, 0.0],
        [0.0, 0.0, 1e20, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    .unwrap();
    let normal = normal_matrix(world).unwrap();
    assert!((normal[0][0] / 1e-20 - 1.0).abs() < 1e-6);
}
