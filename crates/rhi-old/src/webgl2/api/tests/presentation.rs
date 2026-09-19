//! Contract tests for what a presentation acquire answers per lifecycle.
//!
//! Both providers used to answer `Suspended` for any lifecycle that was not
//! `Active`, which reads as "try again later" and so told a caller holding a
//! lost or disposed context to retry something that is never coming back. The
//! decision is one function now, and these are its readings.

use super::*;

/// Every lifecycle state, so the table below is exhaustive by construction: a
/// variant added to the enum stops this test compiling rather than silently
/// escaping it.
const EVERY_LIFECYCLE: [GlContextLifecycle; 7] = [
    GlContextLifecycle::Inactive,
    GlContextLifecycle::Active,
    GlContextLifecycle::Suspended,
    GlContextLifecycle::Lost,
    GlContextLifecycle::Restoring,
    GlContextLifecycle::Poisoned,
    GlContextLifecycle::Disposed,
];

#[test]
fn only_active_proceeds_and_only_suspended_says_retry_later() {
    const OP: &str = "acquire-surface-image";
    for lifecycle in EVERY_LIFECYCLE {
        let observed = acquire_position(OP, lifecycle);
        match lifecycle {
            GlContextLifecycle::Active => {
                assert_eq!(observed, Ok(None), "an active context is asked for a lease");
            }
            GlContextLifecycle::Suspended => {
                assert_eq!(
                    observed,
                    Ok(Some(GlSurfaceAcquire::Suspended)),
                    "a suspended drawable is the one retryable answer"
                );
            }
            // Everything else is terminal or transitional, and none of them is
            // a state a caller may be told to wait out.
            terminal => {
                assert!(
                    observed.is_err(),
                    "{terminal:?} must not be answered with a lease or a retry, got {observed:?}"
                );
                assert_ne!(
                    observed,
                    Ok(Some(GlSurfaceAcquire::Suspended)),
                    "{terminal:?} must not be reported as merely suspended"
                );
            }
        }
    }
}

#[test]
fn the_acquire_reports_what_assert_ready_would_report_for_the_same_state() {
    // The invariant that keeps the two from drifting: an acquire is a verb, so
    // it names a lost context the way every other verb names one. Without this
    // the providers could answer `Suspended` here and `ContextLost` there, which
    // is exactly the state this pair was in.
    const OP: &str = "acquire-surface-image";
    for lifecycle in EVERY_LIFECYCLE {
        if lifecycle == GlContextLifecycle::Active {
            continue;
        }
        let expected = Err(lifecycle.refusal(OP));
        if lifecycle == GlContextLifecycle::Suspended {
            assert_ne!(
                acquire_position(OP, lifecycle),
                expected,
                "suspension is the one state whose acquire answer is an outcome, not an error"
            );
            continue;
        }
        assert_eq!(
            acquire_position(OP, lifecycle),
            expected,
            "{lifecycle:?} must be named identically by the acquire and by the preflight"
        );
    }
}

#[test]
fn each_terminal_state_keeps_its_own_variant_and_its_own_operation() {
    // A single `Err` assertion would pass on any error at all, including one
    // that named the wrong state or dropped the operation the caller needs to
    // find the call site.
    const OP: &str = "acquire-surface-image";
    assert_eq!(
        acquire_position(OP, GlContextLifecycle::Lost),
        Err(GlError::ContextLost { operation: OP })
    );
    assert_eq!(
        acquire_position(OP, GlContextLifecycle::Poisoned),
        Err(GlError::Poisoned { operation: OP })
    );
    assert_eq!(
        acquire_position(OP, GlContextLifecycle::Disposed),
        Err(GlError::Disposed { operation: OP })
    );
    // The two states with no dedicated variant still report themselves: a
    // caller learns which lifecycle blocked it, not only that one did.
    for lifecycle in [GlContextLifecycle::Inactive, GlContextLifecycle::Restoring] {
        assert_eq!(
            acquire_position(OP, lifecycle),
            Err(GlError::InvalidLifecycle {
                operation: OP,
                lifecycle,
            })
        );
    }
}
