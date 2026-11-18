use super::*;

#[test]
fn transform_and_tint_have_distinct_aligned_physical_ranges() {
    let input = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        normal_from_local: None,
        world_from_local: None,
        clip_from_world: None,
        tint: [0.25, 0.5, 0.75, 1.0],
        shading_camera: None,
    };
    let packed = pack_forward_uniforms(&[input], 256, 512).unwrap();
    assert_eq!(packed.transforms.len(), 1);
    assert_eq!(packed.transforms[0], 0..64);
    assert_eq!(packed.tints.len(), 1);
    assert_eq!(packed.tints[0], 256..272);
    assert_eq!(packed.bytes.len(), 512);
    assert_eq!(&packed.bytes[256..260], &0.25f32.to_le_bytes());
    assert!(packed.bytes[64..256].iter().all(|byte| *byte == 0));
}

#[test]
fn empty_input_and_capacity_failures_remain_bounded() {
    assert_eq!(pack_forward_uniforms(&[], 256, 16).unwrap().bytes.len(), 16);
    assert!(matches!(
        pack_forward_uniforms(&[], 0, 16),
        Err(Error::InvalidRange)
    ));
    let input = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        normal_from_local: None,
        world_from_local: None,
        clip_from_world: None,
        tint: [1.0; 4],
        shading_camera: None,
    };
    assert!(matches!(
        pack_forward_uniforms(&[input], 256, 511),
        Err(Error::InvalidRange)
    ));
}

#[test]
fn mixed_color_and_normal_payloads_keep_exact_transform_ranges() {
    let color = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        normal_from_local: None,
        world_from_local: None,
        clip_from_world: None,
        tint: [0.25, 0.5, 0.75, 1.0],
        shading_camera: None,
    };
    let normal = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        normal_from_local: Some([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        world_from_local: Some(crate::RenderTransform::identity().columns()),
        clip_from_world: None,
        tint: [1.0; 4],
        shading_camera: None,
    };
    let packed = pack_forward_uniforms(&[color, normal], 256, 1024).unwrap();
    // Color consumes clip only. The normal surface also consumes normal and
    // world matrices: the complete legacy PBR payload must be declared before
    // its tint binding.
    assert_eq!(packed.transforms, [0..64, 512..512 + PBR_TRANSFORM_BYTES]);
    assert_eq!(packed.tints, [256..272, 768..784]);
    assert_eq!(packed.bytes.len(), 1024);
    assert_eq!(&packed.bytes[576..580], &2.0f32.to_le_bytes());
    let world: Vec<_> = packed.bytes[640..704]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(
        world,
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
        ]
    );
    assert!(packed.bytes[720..768].iter().all(|byte| *byte == 0));
    assert!(packed.bytes[64..256].iter().all(|byte| *byte == 0));
    assert!(matches!(
        pack_forward_uniforms(
            &[ForwardUniformInput {
                transform: crate::RenderTransform::identity().columns(),
                normal_from_local: Some(crate::RenderTransform::identity().columns()),
                world_from_local: Some(crate::RenderTransform::identity().columns()),
                clip_from_world: None,
                tint: [1.0; 4],
                shading_camera: None,
            }],
            256,
            511,
        ),
        Err(Error::InvalidRange)
    ));
}

#[test]
fn non_finite_normal_matrix_is_rejected_before_byte_allocation() {
    let input = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        normal_from_local: Some([[f32::NAN, 0.0, 0.0, 0.0], [0.0; 4], [0.0; 4], [0.0; 4]]),
        world_from_local: Some(crate::RenderTransform::identity().columns()),
        clip_from_world: None,
        tint: [1.0; 4],
        shading_camera: None,
    };
    assert!(matches!(
        pack_forward_uniforms(&[input], 256, 1024),
        Err(Error::InvalidRange)
    ));
}

#[test]
fn pbr_transform_matrices_must_be_paired() {
    let matrix = crate::RenderTransform::identity().columns();
    for (normal_from_local, world_from_local, clip_from_world) in [
        (Some(matrix), None, None),
        (None, Some(matrix), None),
        (None, None, Some(matrix)),
    ] {
        assert!(matches!(
            pack_forward_uniforms(
                &[ForwardUniformInput {
                    transform: matrix,
                    normal_from_local,
                    world_from_local,
                    clip_from_world,
                    tint: [1.0; 4],
                    shading_camera: None,
                }],
                256,
                1024
            ),
            Err(Error::InvalidRange)
        ));
    }
}
