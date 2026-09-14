//! Native GL/GLES discovery over a context made current by the Host.
//!
//! This module never creates a window, display, surface, or context.  Its one
//! unsafe boundary is the `glow` adapter: callers must keep the supplied
//! context current on its owning thread for the entire call.  The resulting
//! snapshot is data only and remains bound to the caller supplied stamp.

mod discovery;
mod provider;
#[cfg(test)]
mod tests;

pub(crate) use discovery::{NativeDiscoveryError, discover_current_glow, parse_native_profile};
pub(crate) use provider::NativeGlProvider;
