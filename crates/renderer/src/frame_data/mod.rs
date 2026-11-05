//! Per-frame transform, uniform, and similar transient input uploads.
//!
//! Frame data has a separate lifecycle from persistent asset residency; both
//! may share low-level transfers and RHI completion tracking.
pub(crate) mod deferred_pool;
pub(crate) mod forward;
pub(crate) mod pool;
