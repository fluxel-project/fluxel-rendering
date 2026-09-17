//! Native GL/GLES discovery over a context made current by the Host.
//!
//! This module never creates a window, display, surface, or context.  Its one
//! unsafe boundary is the `glow` adapter: callers must keep the supplied
//! context current on its owning thread for the entire call.  The resulting
//! snapshot is data only and remains bound to the caller supplied stamp.

mod discovery;
mod probes;
mod provider;
mod surface;
mod surface_facts;
#[cfg(test)]
mod tests;

mod exec_compute;
mod exec_copy;
mod exec_framebuffer;
mod exec_raster;
mod exec_shader;
mod exec_sync;
mod exec_vertex;

pub(crate) use discovery::{
    DRIVER_IDENTITY_UNAVAILABLE, NativeDiscoveryError, REQUIRED_DESKTOP_VERSION,
    desktop_context_floor_marker, discover_current_glow, discover_current_glow_identified,
    discover_current_glow_with_loader, parse_native_profile,
};
pub(crate) use provider::NativeGlProvider;
