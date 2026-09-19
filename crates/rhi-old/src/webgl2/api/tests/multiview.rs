//! Contract tests for the multiview row, which needs both a queried view
//! count and a proven attach path before the capability enables.

use super::*;

/// The multiview row is gated by its operation probe specifically: the
/// extension is acquired and the context answers a two-view limit, and the
/// capability still does not enable. An extension route that still owes a probe
/// contributes no evidence at all, so the fact cannot even be half-satisfied,
/// and no view count beyond the single view every pass already uses is
/// reported.
#[test]
fn multiview_stays_closed_while_its_attach_path_is_unproved() {
    let mut extensions = ledger(&["OVR_multiview2"]);
    assert!(extensions.acquire(GlKnownExtension::OvrMultiview2));
    let mut builder = builder(GlFamilyProfile::WebGl2, extensions, limits());
    builder.resolve(
        GlCapability::Multiview,
        multiview(),
        GlOperationProbe::NotRun,
    );
    let snapshot = builder.build();
    assert_eq!(
        snapshot
            .capabilities()
            .fact(GlCapability::Multiview)
            .expect("fact")
            .evidence,
        None
    );
    assert!(!snapshot.capabilities().supports(GlCapability::Multiview));
    // The queried number is a real observation and stays in the limits, while
    // the pass-facing view count falls back to the WebGPU default of one.
    assert_eq!(snapshot.limits().max_multiview_view_count, 2);
    assert_eq!(snapshot.max_multiview_view_count(), 1);
}

/// The other two halves of the multiview fact are already satisfied in this
/// fixture, so the probe is the one remaining gate; a successful probe is what
/// releases the queried view count.
#[test]
fn a_probed_attach_path_reports_the_queried_view_count() {
    let mut extensions = ledger(&["OVR_multiview2"]);
    assert!(extensions.acquire(GlKnownExtension::OvrMultiview2));
    assert!(extensions.probe(GlKnownExtension::OvrMultiview2));
    let mut builder = builder(GlFamilyProfile::WebGl2, extensions, limits());
    builder.resolve(
        GlCapability::Multiview,
        multiview(),
        GlOperationProbe::Passed,
    );
    let snapshot = builder.build();
    assert!(snapshot.capabilities().supports(GlCapability::Multiview));
    assert_eq!(snapshot.max_multiview_view_count(), 2);
}

/// The desktop core profile has no multiview route in this contract, so no
/// ledger can open the row there: the evidence stays absent even for a ledger
/// that claims a successful probe.
#[test]
fn multiview_has_no_route_on_the_desktop_core_profile() {
    let mut extensions = ledger(&["OVR_multiview2", "GL_OVR_multiview2"]);
    assert!(extensions.acquire(GlKnownExtension::OvrMultiview2));
    assert!(extensions.probe(GlKnownExtension::OvrMultiview2));
    let mut builder = builder(
        GlFamilyProfile::Desktop { major: 4, minor: 3 },
        extensions,
        desktop_limits(),
    );
    builder.resolve(
        GlCapability::Multiview,
        multiview(),
        GlOperationProbe::Passed,
    );
    let snapshot = builder.build();
    assert_eq!(
        snapshot
            .capabilities()
            .fact(GlCapability::Multiview)
            .expect("fact")
            .evidence,
        None
    );
    assert!(!snapshot.capabilities().supports(GlCapability::Multiview));
    assert_eq!(snapshot.max_multiview_view_count(), 1);
}

/// The shipped native row hands every native profile `NotRun`, so a native
/// context with the extension acquired keeps the capability closed and reports
/// the single-view count whatever its limits say.
#[test]
fn every_native_profile_stays_closed_on_the_shipped_unprobed_row() {
    for profile in [
        GlFamilyProfile::Desktop { major: 4, minor: 3 },
        GlFamilyProfile::Embedded { major: 3, minor: 1 },
    ] {
        let mut extensions = ledger(&["OVR_multiview2", "GL_OVR_multiview2"]);
        assert!(extensions.acquire(GlKnownExtension::OvrMultiview2));
        let mut builder = builder(profile, extensions, desktop_limits());
        builder.resolve(
            GlCapability::Multiview,
            multiview(),
            GlOperationProbe::NotRun,
        );
        let snapshot = builder.build();
        assert!(
            !snapshot.capabilities().supports(GlCapability::Multiview),
            "{profile:?}"
        );
        assert_eq!(snapshot.max_multiview_view_count(), 1, "{profile:?}");
    }
}
