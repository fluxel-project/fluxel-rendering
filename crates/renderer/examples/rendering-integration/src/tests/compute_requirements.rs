use super::*;
use fluxel_rhi::api::DeviceLimits;

#[test]
fn fixed_compute_checks_each_required_capability_before_pipeline_creation() {
    let baseline = DeviceCapabilities {
        compute: true,
        storage_buffers: true,
        limits: DeviceLimits {
            max_buffer_bytes: 384,
            max_bind_groups: 1,
            max_bindings_per_group: 1,
            max_compute_workgroups: [2, 1, 1],
            max_compute_workgroup_size: [64, 1, 1],
            ..DeviceLimits::default()
        },
        ..DeviceCapabilities::default()
    };
    assert!(
        requirements(&baseline, true)
            .iter()
            .all(|(_, passed)| *passed)
    );
    assert!(!requirements(&baseline, false)[0].1);
    for case in 0..11 {
        let mut caps = baseline.clone();
        match case {
            0 => caps.compute = false,
            1 => caps.storage_buffers = false,
            2 => caps.limits.max_buffer_bytes = 383,
            3 => caps.limits.max_bind_groups = 0,
            4 => caps.limits.max_bindings_per_group = 0,
            5 => caps.limits.max_compute_workgroups[0] = 1,
            6..=7 => caps.limits.max_compute_workgroups[case - 5] = 0,
            8 => caps.limits.max_compute_workgroup_size[0] = 63,
            9..=10 => caps.limits.max_compute_workgroup_size[case - 8] = 0,
            _ => unreachable!(),
        }
        let checks = requirements(&caps, true);
        let refused: Vec<_> = checks.iter().filter(|(_, passed)| !*passed).collect();
        assert_eq!(refused.len(), 1, "case {case}: {checks:?}");
        assert!(!checks[case].1, "case {case}: {checks:?}");
    }
}
