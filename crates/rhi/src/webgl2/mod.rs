//! Private GL-family implementation layers.
//!
//! The retained 0.14 browser adapter remains under `experimental::webgl2`
//! until this API, its state owner, and the common compatibility adapter pass
//! their migration gates.

#[allow(
    dead_code,
    reason = "Layer 1 declarations intentionally land before their Layer 2/3 consumers"
)]
pub(super) mod api;

#[allow(
    dead_code,
    reason = "Layer 2 declarations intentionally land before the state domains that consume them"
)]
pub(super) mod state;

#[allow(
    dead_code,
    reason = "Layer 3 declarations intentionally land before the adapter that consumes them"
)]
pub(super) mod compat;

/// Real-context observation for the out-of-workspace desktop GL hardware gate.
///
/// It sits above the layers rather than inside one because it consumes Layer 1
/// today and will consume the adapter beside it: putting it under `api` would
/// make a lower layer depend on a higher one, which is the direction the whole
/// three-layer split exists to forbid.
///
/// `test-support` is part of the gate because the entry it holds has exactly one
/// possible caller -- the doc-hidden fixture surface -- and a build that cannot
/// reach that surface would otherwise compile a module nothing in the crate
/// names.
#[cfg(all(
    target_os = "windows",
    feature = "native-gl-wgl",
    feature = "test-support"
))]
pub(crate) mod conformance;
