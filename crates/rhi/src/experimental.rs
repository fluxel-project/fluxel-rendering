//! Experimental APIs that intentionally do not belong to the stable RHI facade.
//!
//! The fixed raster artifacts here support the current renderer validation
//! slice. They are closed recipes, not a general graphics-pipeline API.

/// Closed renderer-shaped raster recipes and their native execution objects.
pub mod fixed_artifacts {
    pub use crate::execution::{RasterBackend, RasterBindings, RasterObjectProvider};
    pub use crate::resource::{
        RasterArtifactIdentity, RasterBindingVisibility, RasterCreateError,
        RasterFixedLightDirection, RasterKernel, RasterLightingModel, RasterLightingSpace,
        RasterNormalBindings, RasterNormalBindingsLease, RasterNormalInterpolation,
        RasterNormalNormalization, RasterPipeline, RasterPipelineLease, RasterSamplerAddressMode,
        RasterSamplerBindingType, RasterSamplerFilter, RasterTextureBindings,
        RasterTextureBindingsLease, RasterTextureSampleType, RasterUniformBindings,
        RasterUniformBindingsLease, RasterUvLinearClampTextureBindings,
        RasterUvLinearClampTextureBindingsLease, RasterUvTextureBindings,
        RasterUvTextureBindingsLease, RasterVertexColorBindings, RasterVertexColorBindingsLease,
        RasterVertexLayout,
    };
}

// Browser-independent lifecycle reducer. Keeping this tiny state machine
// portable lets host tests cover the ownership rules without pretending a host
// build has a WebGPU implementation.
#[cfg(any(test, all(target_arch = "wasm32", feature = "webgpu")))]
mod webgpu_state;

// Browser-independent construction ownership for partially created resource
// sets. Host tests inject creation/write failures to prove that a failed
// recipe destroys exactly what it created and never half-updates a registry.
#[cfg(any(test, all(target_arch = "wasm32", feature = "webgpu")))]
mod resource_candidate;

/// Closed browser WebGL2 execution for the retained Stage 1 unlit scene.
///
/// This module exists only in the browser build.  It deliberately exposes no
/// WebGL objects: the binding crate owns the JavaScript canvas value and RHI
/// owns the context, program, buffers, fences, and generations behind this
/// small session façade.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub mod webgl2;

/// Closed browser WebGPU execution for the retained fixed unlit scene.
///
/// This is deliberately a wasm-private RHI seam: all browser GPU objects and
/// Promise continuations stay behind [`WebGpuSession`](webgpu::WebGpuSession).
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod webgpu;
