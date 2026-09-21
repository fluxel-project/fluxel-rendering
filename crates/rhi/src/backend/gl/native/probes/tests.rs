use super::{ProbeAnswer, ProbeReport, record_extension_probes};
use crate::backend::gl::api::{
    CapabilityEvidence, CoreOrExtension, ExtensionProvenance, GlExtensionSet, GlFamilyProfile,
    GlKnownExtension, GlVersion,
};

const DESKTOP_42: GlFamilyProfile = GlFamilyProfile::Desktop { major: 4, minor: 2 };

/// A report in which only the compute probe answered; every other field
/// stays at its fail-closed default.
fn compute_only(answer: ProbeAnswer) -> ProbeReport {
    ProbeReport {
        compute: answer,
        ..ProbeReport::default()
    }
}

/// A ledger that reported and acquired the desktop compute extension.
fn acquired_compute() -> GlExtensionSet {
    let mut extensions = GlExtensionSet::default();
    extensions.report_raw(GlKnownExtension::ArbComputeShader.raw_name());
    assert!(extensions.acquire(GlKnownExtension::ArbComputeShader));
    extensions
}

/// The audit finding this producer closes, end to end on this side of the
/// boundary: a desktop context below the row's core version resolved no
/// evidence at all while `Probed` was unreachable, because the row asks for
/// a passing probe on top of an acquired extension. Acquisition alone must
/// still resolve nothing, and the probe answer is what flips it.
#[test]
fn a_passed_probe_is_what_lets_the_extension_route_resolve() {
    // Mirrors the shipped desktop compute row exactly.
    let compute_row = CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 3)),
        embedded_core: Some(GlVersion::new(3, 1)),
        extension: Some(GlKnownExtension::ArbComputeShader),
        extension_requires_probe: true,
    };
    let mut extensions = acquired_compute();
    assert_eq!(compute_row.resolve(DESKTOP_42, &extensions), None);
    record_extension_probes(
        DESKTOP_42,
        &compute_only(ProbeAnswer::Passed),
        &mut extensions,
    );
    assert_eq!(
        compute_row.resolve(DESKTOP_42, &extensions),
        Some(CapabilityEvidence::Extension(
            GlKnownExtension::ArbComputeShader
        ))
    );
    // A profile whose core already supplies the row still resolves through
    // core, which is why promotion cannot change a core-enabled context.
    assert_eq!(
        compute_row.resolve(GlFamilyProfile::Desktop { major: 4, minor: 6 }, &extensions),
        Some(CapabilityEvidence::Core(GlFamilyProfile::Desktop {
            major: 4,
            minor: 6
        }))
    );
}

/// The whole point of the producer: a probe that ran and passed is the only
/// path from acquisition to `Probed`.
#[test]
fn a_passed_probe_promotes_an_acquired_extension_to_probed() {
    let mut extensions = acquired_compute();
    record_extension_probes(
        DESKTOP_42,
        &compute_only(ProbeAnswer::Passed),
        &mut extensions,
    );
    assert!(extensions.is_probed(GlKnownExtension::ArbComputeShader));
    // Promotion must not lose the callable interface it built on:
    // `is_acquired` is what every limit and format reader consults.
    assert!(extensions.is_acquired(GlKnownExtension::ArbComputeShader));
}

/// `Probed` claims a driver executed an operation, so neither a probe that
/// could not run nor one the driver refused may reach it. The entry keeps
/// its acquisition instead of being demoted by a claim it never proved.
#[test]
fn a_probe_that_did_not_pass_leaves_the_ledger_acquired() {
    for answer in [ProbeAnswer::Unavailable, ProbeAnswer::Failed] {
        let mut extensions = acquired_compute();
        record_extension_probes(DESKTOP_42, &compute_only(answer), &mut extensions);
        assert_eq!(
            extensions.provenance(GlKnownExtension::ArbComputeShader),
            Some(ExtensionProvenance::Acquired),
            "{answer:?}"
        );
        assert!(!extensions.is_probed(GlKnownExtension::ArbComputeShader));
    }
}

/// A name that only appears in the extension string is not a route, so a
/// passing probe cannot promote it: the probe proves the context can run
/// the operation, not that Fluxel ever acquired the extension's interface.
#[test]
fn a_merely_reported_name_is_never_promoted_by_a_probe() {
    let mut extensions = GlExtensionSet::default();
    extensions.report_raw(GlKnownExtension::ArbComputeShader.raw_name());
    record_extension_probes(
        DESKTOP_42,
        &compute_only(ProbeAnswer::Passed),
        &mut extensions,
    );
    assert_eq!(
        extensions.provenance(GlKnownExtension::ArbComputeShader),
        Some(ExtensionProvenance::Reported)
    );
}

/// A refused acquisition stays refused. Nothing here calls `fail()`, so
/// this is the acquisition gate doing the work rather than a rewrite, which
/// is what keeps a probe from reviving an extension the loader rejected.
#[test]
fn a_failed_acquisition_is_not_revived_by_a_probe() {
    let mut extensions = GlExtensionSet::default();
    extensions.report_raw(GlKnownExtension::ArbComputeShader.raw_name());
    assert!(extensions.fail(GlKnownExtension::ArbComputeShader));
    record_extension_probes(
        DESKTOP_42,
        &compute_only(ProbeAnswer::Passed),
        &mut extensions,
    );
    assert_eq!(
        extensions.provenance(GlKnownExtension::ArbComputeShader),
        Some(ExtensionProvenance::Failed)
    );
}

/// The ledger entry is per-family evidence, so a route the family cannot
/// report never advances even on a context that happens to run the probe.
#[test]
fn a_route_illegal_for_the_profile_is_never_promoted() {
    let mut extensions = acquired_compute();
    record_extension_probes(
        GlFamilyProfile::WebGl2,
        &compute_only(ProbeAnswer::Passed),
        &mut extensions,
    );
    assert!(!extensions.is_probed(GlKnownExtension::ArbComputeShader));
}

/// Each pairing is keyed to its own field: a passing storage probe must not
/// promote the compute extension it shares a report with.
#[test]
fn a_probe_only_promotes_the_extension_it_belongs_to() {
    let mut extensions = GlExtensionSet::default();
    for extension in [
        GlKnownExtension::ArbComputeShader,
        GlKnownExtension::ArbShaderStorageBufferObject,
        GlKnownExtension::ArbShaderImageLoadStore,
    ] {
        extensions.report_raw(extension.raw_name());
        assert!(extensions.acquire(extension));
    }
    let report = ProbeReport {
        storage_buffer: ProbeAnswer::Passed,
        ..ProbeReport::default()
    };
    record_extension_probes(DESKTOP_42, &report, &mut extensions);
    assert!(extensions.is_probed(GlKnownExtension::ArbShaderStorageBufferObject));
    assert!(!extensions.is_probed(GlKnownExtension::ArbComputeShader));
    assert!(!extensions.is_probed(GlKnownExtension::ArbShaderImageLoadStore));
}
