use super::*;

#[test]
fn transform_and_tint_have_distinct_aligned_physical_ranges() {
    let input = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        normal_from_local: None,
        tint: [0.25, 0.5, 0.75, 1.0],
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
        tint: [1.0; 4],
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
        tint: [0.25, 0.5, 0.75, 1.0],
    };
    let normal = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        normal_from_local: Some([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        tint: [1.0; 4],
    };
    let packed = pack_forward_uniforms(&[color, normal], 256, 1024).unwrap();
    assert_eq!(packed.transforms, [0..64, 512..640]);
    assert_eq!(packed.tints, [256..272, 768..784]);
    assert_eq!(packed.bytes.len(), 1024);
    assert_eq!(&packed.bytes[576..580], &2.0f32.to_le_bytes());
    assert!(packed.bytes[64..256].iter().all(|byte| *byte == 0));
    assert!(matches!(
        pack_forward_uniforms(
            &[ForwardUniformInput {
                transform: crate::RenderTransform::identity().columns(),
                normal_from_local: Some(crate::RenderTransform::identity().columns()),
                tint: [1.0; 4],
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
        tint: [1.0; 4],
    };
    assert!(matches!(
        pack_forward_uniforms(&[input], 256, 1024),
        Err(Error::InvalidRange)
    ));
}
