use super::*;
use fluxel_material::DynamicValue;

fn pbr_input(world_from_local: [[f32; 4]; 4]) -> ForwardUniformInput {
    ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        mesh_visibility: None,
        normal_from_local: Some(crate::RenderTransform::identity().columns()),
        world_from_local: Some(world_from_local),
        clip_from_world: None,
        material_uniforms: vec![DynamicValue::Vec4([1.0; 4])],
        shading_camera: None,
        directional_light: crate::DirectionalLight::default(),
    }
}

#[test]
fn pbr_uniform_preserves_f64_negative_world_orientation_in_pbr_payload() {
    // This determinant is below the smallest f32 subnormal, but its inverse
    // remains representable.  The shader must consume the CPU-derived sign.
    let world = [
        [-1.0e-30, 0.0, 0.0, 0.0],
        [0.0, 1.0e-20, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let packed = pack_forward_uniforms(&[pbr_input(world)], 256, 512).unwrap();
    assert_eq!(packed.transforms, [0..PBR_TRANSFORM_BYTES]);
    let offset = PBR_WORLD_HANDEDNESS_OFFSET;
    assert_eq!(
        f32::from_le_bytes(packed.bytes[offset..offset + 4].try_into().unwrap()),
        -1.0
    );
}

#[test]
fn pbr_uniform_refuses_a_missing_world_matrix() {
    let mut input = pbr_input(crate::RenderTransform::identity().columns());
    input.world_from_local = None;
    assert!(matches!(
        pack_forward_uniforms(&[input], 256, 512),
        Err(Error::InvalidRange)
    ));
}
