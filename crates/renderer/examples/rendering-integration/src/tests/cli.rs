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
        ("--s7-deferred-only", Workload::S7Deferred),
        ("--s7-deferred-surface-only", Workload::S7DeferredSurface),
        ("--s7-device-isolation-only", Workload::S7DeviceIsolation),
        ("--s8-lifecycle-only", Workload::S8Lifecycle),
        ("--demo-forward", Workload::DemoForward),
        ("--demo-deferred", Workload::DemoDeferred),
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
        vec!["--s7-deferred-only", "--s6-only"],
        vec!["--s7-deferred-only", "--s7-deferred-surface-only"],
        vec!["--s7-deferred-only", "--s7-device-isolation-only"],
        vec!["--s8-material-only", "--s7-deferred-only"],
        vec!["--demo-forward", "--demo-deferred"],
        vec!["--demo-forward", "--s8-material-only"],
        vec!["--demo-deferred", "--s7-deferred-surface-only"],
        vec!["--demo-forward", "--s8-smoke"],
        vec!["--s8-smoke"],
        vec!["--s6-only", "--s8-scenario", "warm"],
        vec!["--s8-material-only", "--s8-scenario", "invalid"],
        vec!["--s8-material-only", "--s8-scenario"],
        vec!["--s8-lifecycle-only", "--s8-smoke"],
        vec!["--s8-lifecycle-only", "--s8-material-only"],
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

#[test]
fn s8_scenario_and_smoke_are_options_of_the_material_workload() {
    use crate::s8_material::{Config, Scenario};
    let (_, selected) = parse_arguments(arguments(&[
        "--s8-scenario",
        "replacement-10pct",
        "--s8-material-only",
        "--s8-smoke",
    ]))
    .unwrap();
    assert_eq!(
        selected,
        Workload::S8Material(Config::new(Scenario::Replacement10Pct, true))
    );
    assert_eq!(
        parse_arguments(arguments(&["--s8-material-only"]))
            .unwrap()
            .1,
        Workload::S8Material(Config::new(Scenario::Cold, false)),
    );
}
