//! Test-only native readback observation.

use super::*;

/// Reads one completed compute export through the test-only staging path.
///
/// The only accepted state input is the graph's recorded outgoing state.  This
/// deliberately prevents a fixture from repairing an incorrect graph export by
/// guessing `CopySource` or `Undefined` before the helper transition.
#[cfg(test)]
#[allow(
    dead_code,
    reason = "K01/K02 hardware fixtures consume this crate-private wrapper"
)]
pub(crate) fn readback_exported_buffer_for_test(
    device: &Device,
    exported: &fluxel_rendergraph::ExportedBuffer<ComputeBackend>,
) -> Result<Vec<u8>, NativeExecutionError> {
    if exported.physical.device_identity() != device.identity() {
        return Err(NativeExecutionError::ForeignResource);
    }
    match &exported.lease {
        ResourceLease::Buffer(lease)
            if lease.device_identity() == device.identity()
                && lease.identity() == exported.physical.identity() => {}
        _ => return Err(NativeExecutionError::ForeignResource),
    }
    crate::imp::readback_buffer_for_test(
        &device.inner,
        exported.physical.native(),
        exported.lease.clone(),
        exported.outgoing_state,
        exported.descriptor.size,
    )
    .map_err(NativeExecutionError::Recording)
}

/// Reads a completed RasterBackend buffer export through the test-only staging
/// path.  The graph-reported outgoing state is deliberately the helper's only
/// incoming-state input.
#[cfg(test)]
#[allow(
    dead_code,
    reason = "R01/R02/X01 hardware fixtures consume this crate-private wrapper"
)]
pub(crate) fn readback_raster_exported_buffer_for_test(
    device: &Device,
    exported: &fluxel_rendergraph::ExportedBuffer<RasterBackend>,
) -> Result<Vec<u8>, NativeExecutionError> {
    if exported.physical.device_identity() != device.identity() {
        return Err(NativeExecutionError::ForeignResource);
    }
    if !matches!(&exported.lease, ResourceLease::Buffer(_)) {
        return Err(NativeExecutionError::ForeignResource);
    }
    crate::imp::readback_buffer_for_test(
        &device.inner,
        exported.physical.native(),
        exported.lease.clone(),
        exported.outgoing_state,
        exported.descriptor.size,
    )
    .map_err(NativeExecutionError::Recording)
}

/// Exact byte result of a test-support raster texture readback.
///
/// The staging row pitch is exposed so fixtures can separately assert both
/// tight pixels and padding. This is observation-only and cannot create or
/// modify an RHI texture.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
#[allow(
    dead_code,
    reason = "no-backend unit-test builds retain this sibling-fixture result type"
)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RasterTextureReadback {
    /// Tightly packed RGBA8 pixels in row-major order.
    pub tight: Vec<u8>,
    /// Native staging rows including their required padding bytes.
    pub padded: Vec<u8>,
    /// Native staging pitch, in bytes.
    pub bytes_per_row: u32,
}

/// Reads a completed RasterBackend texture export after consuming its actual
/// graph-reported outgoing state; the helper never guesses or repairs it.
///
/// This is deliberately restricted to an `ExportedTexture<RasterBackend>`:
/// callers cannot use it to impose a guessed state on an ordinary texture.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
#[allow(
    dead_code,
    reason = "no-backend unit-test builds retain this sibling-fixture entry point"
)]
pub fn readback_exported_raster_texture_for_test(
    device: &Device,
    exported: &fluxel_rendergraph::ExportedTexture<RasterBackend>,
) -> Result<RasterTextureReadback, NativeExecutionError> {
    if exported.physical.device_identity() != device.identity() {
        return Err(NativeExecutionError::ForeignResource);
    }
    match &exported.lease {
        ResourceLease::Texture(lease)
            if lease.device_identity() == device.identity()
                && lease.identity() == exported.physical.identity() => {}
        _ => return Err(NativeExecutionError::ForeignResource),
    }
    let result = crate::imp::readback_texture_for_test(
        &device.inner,
        exported.physical.native(),
        exported.lease.clone(),
        exported.descriptor,
        exported.outgoing_state,
    )
    .map_err(NativeExecutionError::Recording)?;
    Ok(RasterTextureReadback {
        tight: result.tight,
        padded: result.padded,
        bytes_per_row: result.bytes_per_row,
    })
}

#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
pub(crate) use readback_exported_raster_texture_for_test as readback_raster_exported_texture_for_test;
