//! Closed fixed-raster execution profile.
//!
//! This module assembles raster capability discovery, command recording, and
//! RenderGraph object lowering for the fixed Raster→Compute→Copy slice. It
//! owns the raster-specific binding and pass invariants, while native command
//! emission remains in `crate::imp` and portable plan validation remains in
//! RenderGraph.

use super::*;

pub(in crate::execution) mod capabilities;
pub(in crate::execution) mod operations;
mod provider;
mod recording;

pub use provider::RasterObjectProvider;

/// Binding objects accepted by the raster profile's two fixed compute recipes.
///
/// Keeping this enum private to the profile avoids promoting either fixed
/// recipe into a general-purpose bind-group API.
#[derive(Clone)]
pub enum RasterBindings {
    /// One fixed in-place RW-storage-buffer binding.
    Compute(ComputeBindings),
    /// The closed sampled-Rgba8-to-RW-storage-buffer X01 binding.
    TexturePack(TexturePackBindings),
    /// The closed camera/material raster uniform binding.
    RasterUniform(RasterUniformBindings),
    /// The closed camera/material uniform plus whole sampled texture binding.
    RasterTexture(RasterTextureBindings),
    /// The closed UV-textured raster binding with two expected vertex streams.
    RasterUvTexture(RasterUvTextureBindings),
    /// The closed UV-textured raster binding with RHI-owned linear clamp sampler.
    RasterUvLinearClampTexture(RasterUvLinearClampTextureBindings),
    /// The closed UV-textured sRGB base-color binding with RHI-owned linear
    /// clamp sampler.
    RasterUvLinearClampSrgbTexture(RasterUvLinearClampTextureBindings),
    /// Closed position-and-normal fixed Lambert binding.
    RasterNormal(RasterNormalBindings),
    /// Closed position-and-RGBA8 vertex-color binding.
    RasterVertexColor(RasterVertexColorBindings),
}

/// A serial DX12/Vulkan backend for the fixed Raster→Compute→Copy slice.
pub struct RasterBackend {
    device: Device,
    capabilities: DeviceCapabilities,
    retired: Vec<Retired>,
}
