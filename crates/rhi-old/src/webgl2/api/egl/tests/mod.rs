//! Pure-logic tests for the EGL platform provider.
//!
//! Everything here is a test of a pure function or of this module's own
//! bookkeeping: the exact context request attributes per GLES version, the
//! extension-token match, pbuffer attribute checking, the reported-version
//! parser, the sibling-currentness comparison, the platform identity
//! composition, and the drawable-extent bound.
//!
//! None of it executes EGL. No test in this file loads the EGL library, opens a
//! display, or creates a context, so nothing here is evidence that a real EGL
//! path works on hardware; that evidence belongs to the release-gate matrix.

use super::{
    EglGlesVersion, EglPbufferSize, check_drawable_extent, compose_egl_driver_identity,
    current_binding_is_self, exact_context_attributes, has_extension, native_window_pair,
    parse_gles_version, pbuffer_attributes, within_viewport_limit,
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle, XlibDisplayHandle, XlibWindowHandle};

#[test]
fn exact_gles_attributes_preserve_minor_version() {
    assert_eq!(
        exact_context_attributes(EglGlesVersion::V3_0),
        [0x3098, 3, 0x3038]
    );
    assert_eq!(
        exact_context_attributes(EglGlesVersion::V3_1),
        [0x3098, 3, 0x30FB, 1, 0x3038]
    );
    assert_eq!(
        exact_context_attributes(EglGlesVersion::V3_2),
        [0x3098, 3, 0x30FB, 2, 0x3038]
    );
}

#[test]
fn extension_match_is_token_not_substring() {
    assert!(has_extension(
        "EGL_KHR_create_context EGL_EXT_x",
        "EGL_KHR_create_context"
    ));
    assert!(!has_extension(
        "EGL_KHR_create_context_extra",
        "EGL_KHR_create_context"
    ));
}

#[test]
fn pbuffer_dimensions_are_checked_before_egl() {
    assert_eq!(
        pbuffer_attributes(EglPbufferSize {
            width: 3,
            height: 5
        })
        .unwrap(),
        [0x3057, 3, 0x3056, 5, 0x3038]
    );
    assert!(
        pbuffer_attributes(EglPbufferSize {
            width: 0,
            height: 5
        })
        .is_err()
    );
}

/// Pure logic: the platform identity names the EGL layer and stays empty when
/// the display answered nothing.
#[test]
fn platform_identity_is_the_egl_vendor_and_version_pair() {
    assert_eq!(
        compose_egl_driver_identity("Mesa Project", "1.5"),
        "Mesa Project EGL 1.5"
    );
    assert_eq!(compose_egl_driver_identity("", "1.5"), "1.5");
    assert_eq!(compose_egl_driver_identity("  NVIDIA  ", "\t"), "NVIDIA");
    assert_eq!(compose_egl_driver_identity("", ""), "");
    assert_eq!(compose_egl_driver_identity("  ", "  "), "");
}

/// Pure logic: an observed surface extent is bounded by the recorded maximum
/// viewport dimensions, with the two documented non-rejections.
#[test]
fn viewport_limit_bounds_observed_surface_extents() {
    assert!(within_viewport_limit([16384, 16384], [1920, 1080]));
    assert!(within_viewport_limit([16384, 16384], [16384, 16384]));
    assert!(!within_viewport_limit([16384, 16384], [1920, 16385]));
    // A minimized surface is suspension, not an unaddressable drawable.
    assert!(within_viewport_limit([16384, 16384], [0, 1080]));
    // An unqueried limit cannot be measured against, so it never rejects.
    assert!(within_viewport_limit([0, 0], [65535, 65535]));

    assert!(check_drawable_extent([16384, 16384], [1024, 768]).is_ok());
    assert!(check_drawable_extent([4096, 4096], [4096, 8192]).is_err());
}

#[test]
fn profile_verification_parses_the_complete_reported_version() {
    assert_eq!(
        parse_gles_version("OpenGL ES 3.2 Mesa 25"),
        Some(EglGlesVersion::V3_2)
    );
    assert_eq!(
        parse_gles_version("OpenGL ES 3.1"),
        Some(EglGlesVersion::V3_1)
    );
    assert_eq!(parse_gles_version("OpenGL ES 3.20"), None);
    assert_eq!(parse_gles_version("4.6"), None);
}

#[test]
fn a_context_never_detaches_a_sibling_current_binding() {
    assert!(current_binding_is_self(
        Some("ours"),
        Some("display"),
        "ours",
        "display"
    ));
    assert!(!current_binding_is_self(
        Some("sibling"),
        Some("display"),
        "ours",
        "display"
    ));
    assert!(!current_binding_is_self(
        Some("ours"),
        Some("other-display"),
        "ours",
        "display"
    ));
    assert!(!current_binding_is_self::<&str, &str>(
        None, None, "ours", "display"
    ));
}

#[test]
fn incomplete_xlib_handles_fail_before_the_unsafe_egl_boundary() {
    let display = RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0));
    let valid_window = RawWindowHandle::Xlib(XlibWindowHandle::new(1));
    assert!(native_window_pair(display, valid_window).is_err());

    let display = RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0));
    let empty_window = RawWindowHandle::Xlib(XlibWindowHandle::new(0));
    assert!(native_window_pair(display, empty_window).is_err());
}
