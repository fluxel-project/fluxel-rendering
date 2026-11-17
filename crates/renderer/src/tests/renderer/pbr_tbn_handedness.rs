use super::templates::{FORWARD_UV0_NORMAL, GBUFFER_UV0_NORMAL};

#[test]
fn normal_map_templates_consume_cpu_world_sign_and_never_zero_interpolated_handedness() {
    for source in [FORWARD_UV0_NORMAL, GBUFFER_UV0_NORMAL] {
        assert!(source.contains("pbr_transforms.normal_from_local[3].x"));
        assert!(source.contains("select(1.0, -1.0, tangent.w < 0.0)"));
        assert!(
            !source.contains("determinant(linear)"),
            "fragment code must not recompute a potentially underflowed f32 determinant"
        );
    }
}
