use crate::material;

#[test]
fn permuted_material_keeps_names_and_moves_only_dense_group_zero_bindings() {
    let standard = format!("{:#?}", material(false));
    let permuted = format!("{:#?}", material(true));
    for name in ["material_tint", "material_texture", "material_sampler"] {
        assert!(standard.contains(name) && permuted.contains(name));
    }
    assert!(
        standard.contains("binding: 0")
            && standard.contains("binding: 1")
            && standard.contains("binding: 2")
    );
    let tint = permuted.find("name: \"material_tint\"").unwrap();
    let texture = permuted.find("name: \"material_texture\"").unwrap();
    let sampler = permuted.find("name: \"material_sampler\"").unwrap();
    assert!(permuted[tint..texture].contains("binding: 2"));
    assert!(permuted[texture..sampler].contains("binding: 0"));
    assert!(permuted[sampler..].contains("binding: 1"));
}
