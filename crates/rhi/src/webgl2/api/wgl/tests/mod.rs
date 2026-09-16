//! Pure-logic tests for the WGL platform provider.
//!
//! Everything here is a test of a pure function or of this module's own
//! bookkeeping: the request attributes built from the required version, the
//! agreement between that constant and the marker discovery records, the
//! platform identity composition, the drawable-extent bound, and the
//! currentness arbiter's own records.
//!
//! None of it executes WGL. No test in this file opens a context, swaps a
//! buffer, or observes a driver, so nothing here is evidence that a real WGL
//! path works on hardware; that evidence belongs to the release-gate matrix.

use super::{
    CurrentBindingArbiter, RECORDED_DESKTOP_FLOOR, REQUIRED_DESKTOP_CONTEXT, check_drawable_extent,
    core_context_attributes, driver_identity_from, meets_required_desktop_context,
    recorded_desktop_floor, usable_wgl_proc, verify_recorded_desktop_floor, within_viewport_limit,
};
use crate::webgl2::api::{GlContextFlags, GlVersion};
use std::collections::BTreeSet;

#[test]
fn requests_exact_desktop_core_43_context() {
    let attributes = core_context_attributes();
    assert_eq!(attributes[1], i32::from(REQUIRED_DESKTOP_CONTEXT.major));
    assert_eq!(attributes[3], i32::from(REQUIRED_DESKTOP_CONTEXT.minor));
    assert_eq!(attributes.last(), Some(&0));
}

/// Pure logic: the request above, the check on the actual version and the
/// recorded marker must all name the same version.
#[test]
fn one_constant_governs_request_check_and_recorded_marker() {
    assert_eq!(REQUIRED_DESKTOP_CONTEXT, GlVersion::new(4, 3));
    assert_eq!(recorded_desktop_floor(), "gl.desktop-context-floor=4.3");
    assert_eq!(
        recorded_desktop_floor(),
        format!(
            "{RECORDED_DESKTOP_FLOOR}{}.{}",
            REQUIRED_DESKTOP_CONTEXT.major, REQUIRED_DESKTOP_CONTEXT.minor
        )
    );
}

/// Pure logic: the cross-check accepts the recorded floor and rejects a
/// snapshot recorded against any other one, which is what keeps this module's
/// copy from drifting away from `native::discovery`'s unnoticed.
#[test]
fn recorded_floor_cross_check_rejects_a_different_marker() {
    let agreeing = GlContextFlags {
        other: BTreeSet::from([recorded_desktop_floor()]),
        ..GlContextFlags::default()
    };
    assert!(verify_recorded_desktop_floor(&agreeing).is_ok());

    let absent = GlContextFlags::default();
    assert!(verify_recorded_desktop_floor(&absent).is_err());

    let diverging = GlContextFlags {
        other: BTreeSet::from([format!("{RECORDED_DESKTOP_FLOOR}6.0")]),
        ..GlContextFlags::default()
    };
    assert!(verify_recorded_desktop_floor(&diverging).is_err());
}

/// Pure logic: the version check reads the constant, not a literal.
#[test]
fn actual_profile_check_requires_the_constant_version() {
    assert!(meets_required_desktop_context("4.3.0 Vendor"));
    assert!(meets_required_desktop_context("4.6 Core Profile"));
    assert!(!meets_required_desktop_context("4.2.0 Vendor"));
    // A GLES/WebGL version string is not a desktop core version even when its
    // numeric prefix looks higher.
    assert!(!meets_required_desktop_context("OpenGL ES 3.2"));
    assert!(!meets_required_desktop_context(""));
}

/// Pure logic: the drawable bound is a comparison against the queried maximum,
/// with the two documented non-rejections.
#[test]
fn viewport_limit_bounds_recorded_drawable_extents() {
    assert!(within_viewport_limit([16384, 16384], [1920, 1080]));
    assert!(within_viewport_limit([16384, 16384], [16384, 16384]));
    assert!(!within_viewport_limit([16384, 16384], [16385, 1080]));
    assert!(!within_viewport_limit([16384, 16384], [1920, 16385]));
    // A minimized window is suspension, not an invalid drawable.
    assert!(within_viewport_limit([16384, 16384], [0, 0]));
    // An unqueried limit cannot be measured against, so it never rejects.
    assert!(within_viewport_limit([0, 0], [65535, 65535]));

    assert!(check_drawable_extent([16384, 16384], [1024, 768]).is_ok());
    assert!(check_drawable_extent([4096, 4096], [8192, 768]).is_err());
}

/// Pure logic: the platform identity is never the version string, and a blank
/// answer stays blank so discovery records unavailability honestly.
#[test]
fn platform_driver_identity_is_the_vendor_renderer_pair() {
    assert_eq!(
        driver_identity_from("NVIDIA Corporation", "GeForce RTX 4070/PCIe/SSE2"),
        "NVIDIA Corporation GeForce RTX 4070/PCIe/SSE2"
    );
    assert_eq!(driver_identity_from("", "Mesa Intel"), "Mesa Intel");
    assert_eq!(driver_identity_from("  Mesa  ", "  "), "Mesa");
    assert_eq!(driver_identity_from("", ""), "");
    assert_eq!(driver_identity_from("   ", "\t"), "");
}

#[test]
fn rejects_wgl_documented_invalid_proc_sentinels() {
    for value in [0_isize, 1, 2, 3, -1] {
        assert!(!usable_wgl_proc(value as *const core::ffi::c_void));
    }
    assert!(usable_wgl_proc(4_isize as *const core::ffi::c_void));
}

#[test]
fn arbiter_records_each_context_switch() {
    let arbiter = CurrentBindingArbiter::default();
    arbiter.record(11_usize as *const core::ffi::c_void);
    assert_eq!(arbiter.current(), Some(11));
    arbiter.record(12_usize as *const core::ffi::c_void);
    assert_eq!(arbiter.current(), Some(12));
    arbiter.clear_if(11_usize as *const core::ffi::c_void);
    assert_eq!(arbiter.current(), Some(12));
    arbiter.clear_if(12_usize as *const core::ffi::c_void);
    assert_eq!(arbiter.current(), None);
}
