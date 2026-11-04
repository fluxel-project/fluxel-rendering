use super::{Workload, parse_arguments};
use std::{ffi::OsString, path::PathBuf};

fn arguments(values: &[&str]) -> impl Iterator<Item = OsString> {
    values
        .iter()
        .map(OsString::from)
        .collect::<Vec<_>>()
        .into_iter()
}

#[test]
fn native_selectors_preserve_each_named_workload() {
    for (flag, expected) in [
        ("--s3-only", Workload::S3Material),
        ("--s2-cache-only", Workload::S2Cache),
        ("--s5-only", Workload::S5Forward),
        ("--s5-surface-only", Workload::S5Surface),
        ("--s5-device-only", Workload::S5DeviceIsolation),
        ("--s6-only", Workload::S6Composed),
        ("--s6-surface-only", Workload::S6Surface),
    ] {
        let (path, selected) = parse_arguments(arguments(&[flag, "--output", "attempt"])).unwrap();
        assert_eq!(path, PathBuf::from("attempt"));
        assert_eq!(selected, expected);
    }
    assert_eq!(
        parse_arguments(arguments(&[])).unwrap().1,
        Workload::Foundation
    );
}

#[test]
fn native_selectors_reject_mixed_workloads_and_invalid_arguments() {
    for args in [
        vec!["--s6-only", "--s6-surface-only"],
        vec!["--s5-only", "--s6-only"],
        vec!["--output"],
        vec!["--unknown"],
    ] {
        assert!(parse_arguments(arguments(&args)).is_err(), "{args:?}");
    }
    assert_eq!(
        parse_arguments(arguments(&["--s6-only", "--s6-only"]))
            .unwrap()
            .1,
        Workload::S6Composed,
    );
}
