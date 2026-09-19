//! Owns the safe recording state carried from encoder creation through submission.
//!
//! Compute, raster, and copy scopes are mutually exclusive. Selected pipelines,
//! bindings, buffers, and their leases remain retained until the command buffer is
//! submitted or rejected; raster epochs invalidate recipe-specific readiness after
//! pass or pipeline changes. No native representation escapes this module.

use super::*;

/// Opaque recording encoder for one copy-only graph submission.
pub struct CopyEncoder {
    pub(in crate::execution) native: crate::imp::CopyEncoder,
    pub(in crate::execution) device: fluxel_rendergraph::DeviceIdentity,
    pub(in crate::execution) leases: Vec<ResourceLease>,
    pub(in crate::execution) active_compute: Option<ComputePipeline>,
    pub(in crate::execution) bound_compute_pipeline: Option<ComputePipeline>,
    pub(in crate::execution) active_raster: Option<RasterPipeline>,
    pub(in crate::execution) bound_raster_uniform: Option<RasterUniformBindings>,
    pub(in crate::execution) bound_raster_texture: Option<RasterTextureBindings>,
    pub(in crate::execution) bound_raster_uv_texture: Option<RasterUvBindings>,
    pub(in crate::execution) bound_raster_normal: Option<RasterNormalBindings>,
    pub(in crate::execution) bound_raster_vertex_color: Option<RasterVertexColorBindings>,
    pub(in crate::execution) raster_uv_epoch: u64,
    pub(in crate::execution) raster_uv_binding_epoch: Option<u64>,
    pub(in crate::execution) raster_uv_vertex_slots: [Option<(Buffer, u64)>; 2],
    pub(in crate::execution) raster_uv_index_ready: bool,
    pub(in crate::execution) raster_normal_epoch: u64,
    pub(in crate::execution) raster_normal_binding_epoch: Option<u64>,
    pub(in crate::execution) raster_normal_vertex_slots: [Option<(Buffer, u64)>; 2],
    pub(in crate::execution) raster_normal_index_ready: bool,
    pub(in crate::execution) raster_vertex_color_epoch: u64,
    pub(in crate::execution) raster_vertex_color_binding_epoch: Option<u64>,
    pub(in crate::execution) raster_vertex_color_slots: [Option<(Buffer, u64)>; 2],
    pub(in crate::execution) raster_vertex_color_index_ready: bool,
    pub(in crate::execution) vertex_buffer: Option<(Buffer, u64)>,
    pub(in crate::execution) index_buffer: Option<(Buffer, u64, IndexFormat)>,
    pub(in crate::execution) raster_extent: Option<(u32, u32)>,
    pub(in crate::execution) compute_open: bool,
    pub(in crate::execution) copy_open: bool,
}

/// The two closed explicit-UV recipes share vertex/index readiness, but retain
/// distinct binding objects so a sampler can never be confused with the
/// integer-load recipe.
#[derive(Clone)]
pub(in crate::execution) enum RasterUvBindings {
    Texture(RasterUvTextureBindings),
    LinearClamp(RasterUvLinearClampTextureBindings),
    LinearClampSrgb(RasterUvLinearClampTextureBindings),
}

impl RasterUvBindings {
    pub(in crate::execution) fn position_identity(
        &self,
    ) -> fluxel_rendergraph::PhysicalResourceIdentity {
        match self {
            Self::Texture(value) => value.position_identity(),
            Self::LinearClamp(value) => value.position_identity(),
            Self::LinearClampSrgb(value) => value.position_identity(),
        }
    }

    pub(in crate::execution) fn texture_coordinate_identity(
        &self,
    ) -> fluxel_rendergraph::PhysicalResourceIdentity {
        match self {
            Self::Texture(value) => value.texture_coordinate_identity(),
            Self::LinearClamp(value) => value.texture_coordinate_identity(),
            Self::LinearClampSrgb(value) => value.texture_coordinate_identity(),
        }
    }

    pub(in crate::execution) fn vertex_count(&self) -> u32 {
        match self {
            Self::Texture(value) => value.vertex_count(),
            Self::LinearClamp(value) => value.vertex_count(),
            Self::LinearClampSrgb(value) => value.vertex_count(),
        }
    }

    pub(in crate::execution) fn native(&self) -> &crate::imp::NativeRasterTextureBindings {
        match self {
            Self::Texture(value) => value.native(),
            Self::LinearClamp(value) => value.native(),
            Self::LinearClampSrgb(value) => value.native(),
        }
    }
}
/// Opaque finished command buffer for one copy-only graph submission.
pub struct CopyCommandBuffer {
    pub(in crate::execution) native: crate::imp::CopyCommandBuffer,
    pub(in crate::execution) device: fluxel_rendergraph::DeviceIdentity,
    pub(in crate::execution) leases: Vec<ResourceLease>,
}
/// Opaque cloneable completion for an accepted native submission.
#[derive(Clone)]
pub struct NativeCompletion(pub(crate) crate::imp::NativeCompletion);

/// Uninhabited raster-pipeline placeholder for the copy-only backend.
pub enum UnsupportedRasterPipeline {}
/// Uninhabited compute-pipeline placeholder for the copy-only backend.
pub enum UnsupportedComputePipeline {}
/// Uninhabited binding placeholder for the copy-only backend.
pub enum UnsupportedBindings {}

pub(in crate::execution) struct Retired {
    pub(in crate::execution) completion: NativeCompletion,
    pub(in crate::execution) leases: Vec<ResourceLease>,
}

/// A serial DX12/Vulkan backend that executes only RenderGraph Copy plans.
pub struct CopyBackend {
    pub(in crate::execution) device: Device,
    pub(in crate::execution) capabilities: DeviceCapabilities,
    pub(in crate::execution) retired: Vec<Retired>,
}

impl CopyBackend {
    /// Creates a copy-only backend over an already opened native device.
    pub fn new(device: Device) -> Self {
        Self {
            capabilities: Self::portable_capabilities(),
            device,
            retired: Vec::new(),
        }
    }

    /// Returns the normalized capability profile shared by both native backends.
    pub fn portable_capabilities() -> DeviceCapabilities {
        copy_capabilities()
    }

    /// Waits outside graph execution for an accepted submission.
    pub fn wait(&self, completion: &NativeCompletion, timeout: Duration) -> Result<(), WaitError> {
        match crate::imp::wait_completion(&completion.0, timeout)
            .map_err(|error| WaitError::Backend(NativeExecutionError::Completion(error)))?
        {
            CompletionStatus::Complete => Ok(()),
            CompletionStatus::Pending => Err(WaitError::Timeout),
            CompletionStatus::Failed(reason) => Err(WaitError::Failed(reason)),
            _ => Err(WaitError::Backend(NativeExecutionError::Completion(
                "unknown completion status".into(),
            ))),
        }
    }
}
