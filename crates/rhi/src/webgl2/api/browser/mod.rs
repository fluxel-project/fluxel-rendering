//! Browser-owned WebGL2 discovery and execution.
//!
//! The Host/JS bridge owns canvas creation, DOM events, RAF, and context-loss
//! listeners. RHI creates and owns the WebGL2 context associated with that
//! Host-provided canvas, then gathers immutable evidence for its `ContextStamp`.

#![cfg(target_arch = "wasm32")]

mod discovery;
mod provider;

pub(crate) use discovery::WebGl2BrowserDiscovery;
