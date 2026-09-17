//! Contract tests for the profile floor that decides whether a discovered
//! context is the profile it says it is.

use super::*;

/// A context that requires only 16-byte uniform offsets is a desktop context.
///
/// This is the regression guard for an inverted comparison rather than a
/// hypothetical: the floor used to demand `uniform_buffer_offset_alignment >=
/// 256`, and an alignment is a modulus a buffer offset has to land on, so a
/// driver requiring *less* alignment is the *more* capable one -- 16 admits
/// every 256-byte offset this crate binds, while 512 would admit fewer. The row
/// therefore rejected exactly the contexts that were better than the floor, and
/// it rejected the first real desktop GL context this repository ever opened
/// (AMD Radeon 780M, driver 32.0.21028.2002, which answers 16).
///
/// No test caught it because every fixture answered exactly 256, which is the
/// single value an inverted `>= 256` accepts. These two cases are the pair that
/// makes the assertions non-vacuous: 16 must be accepted, and a real capacity
/// floor must still be enforced, so "the validator stopped working" cannot pass
/// as "the defect was fixed".
#[test]
fn desktop_profile_accepts_a_context_that_requires_only_sixteen_byte_uniform_offsets() {
    let mut limits = desktop_limits();
    limits.uniform_buffer_offset_alignment = 16;
    limits.storage_buffer_offset_alignment = 16;
    assert_eq!(
        limits.validate_profile_minimums(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        Ok(())
    );
}

/// A context below a real capacity floor is still refused, by name.
///
/// The failure records which limit and which numbers, because the floor's only
/// consumer is a diagnosis: a context refused for its texture size has to be
/// distinguishable from one refused for its attachment count.
#[test]
fn desktop_profile_still_refuses_a_context_below_a_capacity_floor() {
    let mut limits = desktop_limits();
    // One below the desktop floor, and the first row the comparison reaches.
    limits.max_texture_size = 8_192;
    assert_eq!(
        limits.validate_profile_minimums(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        Err(GlLimitViolation {
            name: "max_texture_size",
            actual: 8_192,
            required: 16_384,
        })
    );
}

/// The floor is per profile, so the same context can be embedded and not desktop.
///
/// The embedded baseline fixture answers the embedded numbers and not the
/// desktop ones, which is what keeps the two profiles from sharing a single
/// accidental acceptance.
#[test]
fn the_same_limits_are_judged_by_the_profile_they_are_asked_about() {
    assert_eq!(
        limits().validate_profile_minimums(GlFamilyProfile::WebGl2),
        Ok(())
    );
    assert!(matches!(
        limits().validate_profile_minimums(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        Err(GlLimitViolation { .. })
    ));
}
