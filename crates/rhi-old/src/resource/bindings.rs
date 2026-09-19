//! Closed fixed-raster binding handles and lifetime state.
use super::*;
pub(in crate::resource) struct RasterPipelineShared {
    pub(in crate::resource) _native: crate::imp::NativeRasterPipeline,
    pub(in crate::resource) kernel: RasterKernel,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// An opaque, device-affine pipeline for one fixed raster artifact.
#[derive(Clone)]
pub struct RasterPipeline(pub(in crate::resource) Arc<RasterPipelineShared>);

/// A cloneable strong lease for a raster pipeline and its native artifacts.
#[derive(Clone)]
pub struct RasterPipelineLease(pub(in crate::resource) Arc<RasterPipelineShared>);

pub(in crate::resource) struct RasterUniformBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeRasterUniformBindings,
    pub(in crate::resource) pipeline: RasterPipeline,
    pub(in crate::resource) _buffer: BufferLease,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed group-0/binding-0 80-byte frame uniform binding for the camera and
/// material raster artifact only.
#[derive(Clone)]
pub struct RasterUniformBindings(pub(in crate::resource) Arc<RasterUniformBindingsShared>);

/// A cloneable strong lease retaining the uniform bind group, pipeline, and buffer.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "the lease is retained through ResourceLease for terminal native completion"
)]
pub struct RasterUniformBindingsLease(pub(in crate::resource) Arc<RasterUniformBindingsShared>);

impl fmt::Debug for RasterUniformBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterUniformBindingsLease(..)")
    }
}

pub(in crate::resource) struct RasterTextureBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeRasterTextureBindings,
    pub(in crate::resource) pipeline: RasterPipeline,
    pub(in crate::resource) _uniform: BufferLease,
    pub(in crate::resource) _texture: TextureLease,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed group-0 bindings: one 80-byte frame uniform and one whole sampled RGBA8 texture.
#[derive(Clone)]
pub struct RasterTextureBindings(pub(in crate::resource) Arc<RasterTextureBindingsShared>);

/// Strong lease retaining the textured raster binding and both resources.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterTextureBindingsLease(pub(in crate::resource) Arc<RasterTextureBindingsShared>);

impl fmt::Debug for RasterTextureBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterTextureBindingsLease(..)")
    }
}

pub(in crate::resource) struct RasterUvTextureBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeRasterTextureBindings,
    pub(in crate::resource) pipeline: RasterPipeline,
    pub(in crate::resource) _uniform: BufferLease,
    pub(in crate::resource) _texture: TextureLease,
    pub(in crate::resource) _positions: BufferLease,
    pub(in crate::resource) _texture_coordinates: BufferLease,
    pub(in crate::resource) position_identity: PhysicalResourceIdentity,
    pub(in crate::resource) texture_coordinate_identity: PhysicalResourceIdentity,
    pub(in crate::resource) vertex_count: u32,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed explicit-UV raster binding. It retains every resource and records
/// the two physical vertex-role identities that native recording must match.
#[derive(Clone)]
pub struct RasterUvTextureBindings(pub(in crate::resource) Arc<RasterUvTextureBindingsShared>);

/// Strong lease retaining the explicit-UV binding and all its resources.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterUvTextureBindingsLease(pub(in crate::resource) Arc<RasterUvTextureBindingsShared>);

impl fmt::Debug for RasterUvTextureBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterUvTextureBindingsLease(..)")
    }
}

pub(in crate::resource) struct RasterUvLinearClampTextureBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeRasterTextureBindings,
    pub(in crate::resource) pipeline: RasterPipeline,
    pub(in crate::resource) _uniform: BufferLease,
    pub(in crate::resource) _texture: TextureLease,
    pub(in crate::resource) _positions: BufferLease,
    pub(in crate::resource) _texture_coordinates: BufferLease,
    pub(in crate::resource) position_identity: PhysicalResourceIdentity,
    pub(in crate::resource) texture_coordinate_identity: PhysicalResourceIdentity,
    pub(in crate::resource) vertex_count: u32,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

pub(in crate::resource) struct RasterNormalBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeRasterUniformBindings,
    pub(in crate::resource) pipeline: RasterPipeline,
    pub(in crate::resource) _uniform: BufferLease,
    pub(in crate::resource) _positions: BufferLease,
    pub(in crate::resource) _normals: BufferLease,
    pub(in crate::resource) position_identity: PhysicalResourceIdentity,
    pub(in crate::resource) normal_identity: PhysicalResourceIdentity,
    pub(in crate::resource) vertex_count: u32,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed camera/material plus position-and-normal binding for the fixed
/// object-space Lambert artifact. Both vertex stream identities are retained
/// so recording cannot exchange their roles.
#[derive(Clone)]
pub struct RasterNormalBindings(pub(in crate::resource) Arc<RasterNormalBindingsShared>);

/// Strong lease retaining the normal-Lambert binding and every referenced
/// resource through terminal completion.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterNormalBindingsLease(pub(in crate::resource) Arc<RasterNormalBindingsShared>);

impl fmt::Debug for RasterNormalBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterNormalBindingsLease(..)")
    }
}

pub(in crate::resource) struct RasterVertexColorBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeRasterUniformBindings,
    pub(in crate::resource) pipeline: RasterPipeline,
    pub(in crate::resource) _uniform: BufferLease,
    pub(in crate::resource) _positions: BufferLease,
    pub(in crate::resource) _colors: BufferLease,
    pub(in crate::resource) position_identity: PhysicalResourceIdentity,
    pub(in crate::resource) color_identity: PhysicalResourceIdentity,
    pub(in crate::resource) vertex_count: u32,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed camera/material plus position-and-RGBA8-color binding.
#[derive(Clone)]
pub struct RasterVertexColorBindings(pub(in crate::resource) Arc<RasterVertexColorBindingsShared>);

/// Strong lease retaining the vertex-color binding and all streams through completion.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterVertexColorBindingsLease(
    pub(in crate::resource) Arc<RasterVertexColorBindingsShared>,
);

impl fmt::Debug for RasterVertexColorBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterVertexColorBindingsLease(..)")
    }
}

/// Closed explicit-UV binding with an internally owned filtering linear-clamp
/// sampler. The sampler is not a graph resource or a configurable public API.
#[derive(Clone)]
pub struct RasterUvLinearClampTextureBindings(
    pub(in crate::resource) Arc<RasterUvLinearClampTextureBindingsShared>,
);

/// Strong lease retaining the closed linear-clamp binding, its private sampler,
/// pipeline, and every referenced resource through terminal completion.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterUvLinearClampTextureBindingsLease(
    pub(in crate::resource) Arc<RasterUvLinearClampTextureBindingsShared>,
);

impl fmt::Debug for RasterUvLinearClampTextureBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterUvLinearClampTextureBindingsLease(..)")
    }
}

pub(in crate::resource) struct TexturePackBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeTexturePackBindings,
    pub(in crate::resource) pipeline: ComputePipeline,
    pub(in crate::resource) _texture: TextureLease,
    pub(in crate::resource) _buffer: BufferLease,
    pub(in crate::resource) offset: u64,
    pub(in crate::resource) size: u64,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed X01 bindings: one complete sampled `Rgba8Unorm` texture and one RW
/// storage buffer receiving row-major packed pixels.
#[derive(Clone)]
pub struct TexturePackBindings(pub(in crate::resource) Arc<TexturePackBindingsShared>);

/// A cloneable strong lease for an X01 binding object and all its inputs.
#[derive(Clone)]
pub struct TexturePackBindingsLease(pub(in crate::resource) Arc<TexturePackBindingsShared>);
