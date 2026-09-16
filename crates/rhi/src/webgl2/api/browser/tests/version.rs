//! The `WebGL 2.x` family-string gate.
//!
//! These two assertions moved out of `discovery.rs`: as plain `#[test]` items
//! there they were compiled but never executed by the wasm runner, so they
//! proved nothing about the strings a real browser reports.

use wasm_bindgen_test::*;

use super::*;

#[wasm_bindgen_test]
fn accepts_exact_webgl2_family_strings() {
    assert!(discovery::require_webgl2_version("WebGL 2.0 (OpenGL ES 3.0 Chromium)").is_ok());
    assert!(discovery::require_webgl2_version("WebGL 2").is_ok());
}

#[wasm_bindgen_test]
fn rejects_lookalike_major_versions() {
    assert!(discovery::require_webgl2_version("WebGL 20").is_err());
    assert!(discovery::require_webgl2_version("WebGL 2foo").is_err());
    assert!(discovery::require_webgl2_version("WebGL 1.0").is_err());
    assert!(discovery::require_webgl2_version("OpenGL ES 3.0").is_err());
    assert!(discovery::require_webgl2_version("WebGLX 2.0").is_err());
}
