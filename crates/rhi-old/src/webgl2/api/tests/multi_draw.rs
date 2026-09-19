//! Contract tests for the batch row, whose only route is an acquired
//! command set and which no native profile has in this contract.

use super::*;

/// A reported batch extension is not an acquired one: the domain enables only
/// from a command set whose entry points were all proved callable, and every
/// other context keeps the deterministic single-draw route.
#[test]
fn batch_domain_enables_only_from_an_acquired_command_set() {
    let mut reported = builder(
        GlFamilyProfile::WebGl2,
        ledger(&["WEBGL_multi_draw"]),
        limits(),
    );
    reported.resolve(
        GlCapability::MultiDraw,
        batch(),
        GlOperationProbe::NotRequired,
    );
    let snapshot = reported.build();
    assert_eq!(
        snapshot
            .capabilities()
            .fact(GlCapability::MultiDraw)
            .expect("fact")
            .evidence,
        None
    );
    assert!(!snapshot.capabilities().supports(GlCapability::MultiDraw));

    let mut extensions = ledger(&["WEBGL_multi_draw"]);
    assert!(extensions.acquire(GlKnownExtension::WebglMultiDraw));
    let mut acquired = builder(GlFamilyProfile::WebGl2, extensions, limits());
    acquired.resolve(
        GlCapability::MultiDraw,
        batch(),
        GlOperationProbe::NotRequired,
    );
    let snapshot = acquired.build();
    assert_eq!(
        snapshot
            .capabilities()
            .fact(GlCapability::MultiDraw)
            .expect("fact")
            .evidence,
        Some(CapabilityEvidence::Extension(
            GlKnownExtension::WebglMultiDraw
        ))
    );
    assert!(snapshot.capabilities().supports(GlCapability::MultiDraw));
}

/// No native profile has a batch route in this contract, so the row stays
/// closed for both families even when a ledger claims the extension.
#[test]
fn batch_domain_stays_closed_on_every_profile_without_a_route() {
    for profile in [
        GlFamilyProfile::Desktop { major: 4, minor: 3 },
        GlFamilyProfile::Embedded { major: 3, minor: 1 },
    ] {
        let mut extensions = ledger(&["WEBGL_multi_draw"]);
        assert!(extensions.acquire(GlKnownExtension::WebglMultiDraw));
        let mut builder = builder(profile, extensions, desktop_limits());
        builder.resolve(
            GlCapability::MultiDraw,
            batch(),
            GlOperationProbe::NotRequired,
        );
        let snapshot = builder.build();
        assert_eq!(
            snapshot
                .capabilities()
                .fact(GlCapability::MultiDraw)
                .expect("fact")
                .evidence,
            None,
            "{profile:?}"
        );
        assert!(!snapshot.capabilities().supports(GlCapability::MultiDraw));
    }
}
