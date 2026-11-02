use super::*;

#[test]
fn dispatch_is_ceiling_division_for_fixed_workgroups() {
    assert_eq!(dispatch_groups(1).unwrap(), 1);
    assert_eq!(dispatch_groups(64).unwrap(), 1);
    assert_eq!(dispatch_groups(65).unwrap(), 2);
    assert_eq!(dispatch_groups(u32::MAX).unwrap(), 67_108_864);
}
