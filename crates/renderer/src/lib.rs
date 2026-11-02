//! Scene data, preparation, frame policy, and the private RenderGraph.
//!
//! This crate is being rebuilt from the current workspace architecture. The
//! initial Copy, offscreen and Surface Raster, and supported Compute fixtures
//! exercise asset retention and RHI completion. Complete rendering and extension contracts
//! remain under development.

#![deny(missing_docs)]

mod feature;
mod frame_data;
mod graph;
mod pipeline;
mod preparation;
mod render_scene;
mod renderer;
mod residency;

pub use renderer::{Renderer, RendererError, ShutdownReport, Submission};
