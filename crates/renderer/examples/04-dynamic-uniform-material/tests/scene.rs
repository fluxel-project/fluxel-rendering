use renderer_04_dynamic_uniform_material::scene::{
    CUBE_COUNT, CUBE_INDICES, CUBE_VERTICES, camera, encode_matrix, model,
};

#[test]
fn source_cube_expands_to_thirty_six_nonindexed_vertices() {
    let expanded: Vec<_> = CUBE_INDICES
        .into_iter()
        .map(|index| CUBE_VERTICES[index as usize])
        .collect();
    assert_eq!(expanded.len(), 36);
    assert_eq!(expanded[0], CUBE_VERTICES[0]);
    assert_eq!(expanded[35], CUBE_VERTICES[3]);
}

#[test]
fn every_cube_has_a_stable_finite_64_byte_model_matrix() {
    for index in 0..CUBE_COUNT {
        let matrix = model(index, 0.75);
        assert!(matrix.into_iter().flatten().all(f32::is_finite));
        assert_eq!(encode_matrix(matrix).len(), 64);
    }
}

#[test]
fn time_changes_rotation_but_not_lattice_translation() {
    let first = model(42, 0.0);
    let second = model(42, 1.0);
    assert_ne!(first, second);
    assert_eq!(first[3], second[3]);
    assert!(camera([256, 256]).into_iter().flatten().all(f32::is_finite));
}
