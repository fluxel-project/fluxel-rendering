use super::*;

#[test]
fn transform_and_tint_have_distinct_aligned_physical_ranges() {
    let input = ForwardUniformInput {
        transform: crate::RenderTransform::identity().columns(),
        tint: [0.25, 0.5, 0.75, 1.0],
    };
    let packed = pack_forward_uniforms(&[input], 256, 512).unwrap();
    assert_eq!(packed.transforms, [0..64]);
    assert_eq!(packed.tints, [256..272]);
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
        tint: [1.0; 4],
    };
    assert!(matches!(
        pack_forward_uniforms(&[input], 256, 511),
        Err(Error::InvalidRange)
    ));
}
