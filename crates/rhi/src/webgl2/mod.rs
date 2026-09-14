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
