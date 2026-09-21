//! OpenGL-family lowering for desktop GL 4.x, GLES 3.x and WebGL2.
//!
//! The three profiles share an explicit desired/applied state machine, but not
//! a provider or a capability ceiling.  Capability publication is derived from
//! the actual context version, extensions and loaded entry points.  A desktop
//! feature therefore never leaks into GLES or WebGL2 merely because the Rust
//! lowering knows how to express it.
//!
//! Minimum accepted contexts are desktop GL 4.0, GLES 3.0 and WebGL2.  Newer
//! core versions are used when present, and an extension route is admitted only
//! when every native function required by that route was loaded as well.

pub(crate) mod api;
pub(crate) mod capabilities;
pub(crate) mod facts;
pub(crate) mod features;
pub(crate) mod profile;
pub(crate) mod state;
pub(crate) mod translate;
pub(crate) mod translate_raster;

#[cfg(all(
    not(target_arch = "wasm32"),
    any(feature = "native-gl-wgl", feature = "native-gles-egl")
))]
pub(crate) mod native;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(crate) mod browser;

pub(crate) mod platform;
