use crate::{oracle::Control, scene};

#[test]
fn binary16_chain_is_complete_and_control_data_is_valid() {
    let input = scene::input(Control {
        name: "test",
        metallic: 1.0,
        roughness: 0.8,
        camera: [0., 0., 1.],
        lowest_mip_only: false,
        lut_variant: false,
        diffuse: false,
    })
    .unwrap();
    assert!(input.environment.value().is_valid());
    assert_eq!(input.geometry_bytes.len(), 144);
}
