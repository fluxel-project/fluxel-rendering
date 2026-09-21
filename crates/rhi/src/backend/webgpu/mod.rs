//! Browser WebGPU lowering.
//!
//! WebGPU objects are JavaScript values, hence affine to the browser owner
//! thread.  This module deliberately exports only opaque numeric registrations
//! to the rest of the backend: a portable RHI handle, and even a backend trait
//! object, never contains a `JsValue` or a browser lease.

mod binding;
mod capabilities;
mod command;
mod js;
mod pipeline;
mod presentation;
mod provider;
mod registry;
mod resource;
mod shader;
mod translate;

// Kept adjacent to the registration layer so its construction is the only
// place a successfully settled browser device becomes a portable Device.
// The implementation lands with resource/command lowering.
mod device;

// These tests run in a headed browser against the actual adapter, never a JS
// mock or a software renderer.
#[cfg(all(test, target_arch = "wasm32"))]
mod tests;

pub(crate) use provider::WebGpuProvider;
