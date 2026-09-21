//! Explicit environment/capability classification for common test runners.

/// The only outcomes a hardware conformance runner may report.
///
/// `Unsupported` means the exact portable capability/format/route was not
/// advertised and the workload was not recorded. `Skipped` means a required
/// environment fixture (loader, adapter, host target) did not exist. Neither
/// is success. An advertised route that records, submits, or reads back
/// incorrectly is `Failure`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CaseOutcome {
    /// An advertised portable workload completed and met its CPU assertion.
    Pass,
    /// The exact advertised capability precondition was absent.
    Unsupported(String),
    /// The runner could not construct a required real-hardware fixture.
    Skipped(String),
    /// An advertised workload violated its contract.
    Failure(String),
}

impl CaseOutcome {
    /// Builds the fail-closed result for a precise capability gate.
    pub(crate) fn capability(published: bool, requirement: impl Into<String>) -> Self {
        if published {
            Self::Pass
        } else {
            Self::Unsupported(requirement.into())
        }
    }

    /// Turns a non-pass outcome into a focused test failure.
    pub(crate) fn require_pass(self, case: &str) {
        match self {
            Self::Pass => {}
            Self::Unsupported(reason) => panic!("{case}: capability was not published: {reason}"),
            Self::Skipped(reason) => panic!("{case}: required fixture was unavailable: {reason}"),
            Self::Failure(reason) => panic!("{case}: conformance failure: {reason}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CaseOutcome;

    #[test]
    fn absent_capability_is_not_misreported_as_a_pass() {
        assert_eq!(
            CaseOutcome::capability(false, "RGBA8 copy-src route"),
            CaseOutcome::Unsupported("RGBA8 copy-src route".into())
        );
    }

    #[test]
    fn published_capability_starts_a_real_case() {
        assert_eq!(CaseOutcome::capability(true, "unused"), CaseOutcome::Pass);
    }

    #[test]
    #[should_panic(expected = "required fixture was unavailable")]
    fn skipped_fixture_never_becomes_a_pass() {
        CaseOutcome::Skipped("no native loader".into()).require_pass("fixture");
    }

    #[test]
    #[should_panic(expected = "conformance failure")]
    fn advertised_route_failure_never_becomes_a_pass() {
        CaseOutcome::Failure("bad readback".into()).require_pass("route");
    }
}
