//! Browser-owned WebGL2 discovery and execution.
//!
//! The Host/JS bridge owns canvas creation, DOM events, RAF, and context-loss
//! listeners. RHI creates and owns the WebGL2 context associated with that
//! Host-provided canvas, then gathers immutable evidence for its `ContextStamp`.

#![cfg(target_arch = "wasm32")]

mod discovery;
mod exec_copy;
mod exec_framebuffer;
mod exec_multidraw;
mod exec_present;
mod exec_raster;
mod exec_shader;
mod exec_sync;
mod exec_vertex;
mod format_map;
mod objects;
mod provider;

pub(crate) use discovery::WebGl2BrowserDiscovery;
