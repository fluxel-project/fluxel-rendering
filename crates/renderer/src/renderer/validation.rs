use super::RendererError;
use fluxel_rhi::{
    api::{Device, DeviceId, Error},
    backend::Backend,
};
const VERTEX_STRIDE: u64 = 24;
const PIXEL_BYTES: u32 = 4;
const COMPUTE_ELEMENT_BYTES: usize = 4;
const COMPUTE_WORKGROUP_SIZE: u32 = 64;
const COPY_ROW_ALIGNMENT: u64 = 256;

pub(super) fn validate_copy_requirements<B: Backend>(
    device: &Device<B>,
    upload_bytes: u64,
    target_bytes: u64,
) -> Result<(), RendererError> {
    require_buffers(device, &[upload_bytes, target_bytes])
}

pub(super) fn validate_compute_requirements<B: Backend>(
    device: &Device<B>,
    bytes: u64,
    groups: u32,
) -> Result<(), RendererError> {
    let caps = device.capabilities();
    if !caps.storage_buffers {
        return Err(RendererError::Rhi(Error::Unsupported("storage buffers")));
    }
    require_buffers(device, &[bytes, bytes])?;
    if caps.limits.max_bind_groups < 1 || caps.limits.max_bindings_per_group < 1 {
        return Err(RendererError::Rhi(Error::InvalidRange));
    }
    let limits = caps.limits;
    if groups == 0
        || groups > limits.max_compute_workgroups[0]
        || limits.max_compute_workgroups[1] < 1
        || limits.max_compute_workgroups[2] < 1
        || COMPUTE_WORKGROUP_SIZE > limits.max_compute_workgroup_size[0]
        || limits.max_compute_workgroup_size[1] < 1
        || limits.max_compute_workgroup_size[2] < 1
    {
        return Err(RendererError::InvalidComputeInput);
    }
    Ok(())
}

pub(super) fn validate_raster_requirements<B: Backend>(
    device: &Device<B>,
    vertex_bytes: u64,
    extent: [u32; 2],
    readback: bool,
    bind_group_entries: u32,
) -> Result<(), RendererError> {
    let readback_bytes = if readback {
        aligned_row_pitch(extent[0])?
            .checked_mul(u64::from(extent[1]))
            .ok_or(RendererError::Rhi(Error::InvalidRange))?
    } else {
        0
    };
    let sizes = [vertex_bytes, vertex_bytes, readback_bytes];
    require_buffers(device, if readback { &sizes } else { &sizes[..2] })?;
    validate_raster_features(device, extent, readback, 2, bind_group_entries)
}

fn validate_raster_features<B: Backend>(
    device: &Device<B>,
    extent: [u32; 2],
    owned_texture: bool,
    vertex_attributes: u32,
    bind_group_entries: u32,
) -> Result<(), RendererError> {
    let limits = device.capabilities().limits;
    if extent.contains(&0)
        || (owned_texture
            && (extent[0] > limits.max_texture_dimension_2d
                || extent[1] > limits.max_texture_dimension_2d
                || limits.max_texture_array_layers < 1))
        || limits.max_color_attachments < 1
        || (vertex_attributes > 0
            && (limits.max_vertex_buffers < 1 || limits.max_vertex_attributes < vertex_attributes))
        || (bind_group_entries > 0
            && (limits.max_bind_groups < 1 || limits.max_bindings_per_group < bind_group_entries))
    {
        return Err(RendererError::Rhi(Error::InvalidRange));
    }
    Ok(())
}

pub(super) fn validate_read_set_requirements<B: Backend>(
    device: &Device<B>,
    uniform_end: u64,
    uniform_offset: u64,
    upload_bytes: u64,
    readback_bytes: u64,
) -> Result<(), RendererError> {
    let alignment = device.capabilities().limits.uniform_offset_alignment;
    if alignment == 0 || uniform_offset % alignment != 0 {
        return Err(RendererError::Rhi(Error::InvalidRange));
    }
    require_buffers(device, &[upload_bytes, uniform_end, readback_bytes])?;
    validate_raster_features(device, [8, 8], true, 0, 2)
}

fn require_buffers<B: Backend>(device: &Device<B>, sizes: &[u64]) -> Result<(), RendererError> {
    if sizes
        .iter()
        .any(|size| *size == 0 || *size > device.capabilities().limits.max_buffer_bytes)
    {
        Err(RendererError::Rhi(Error::InvalidRange))
    } else {
        Ok(())
    }
}

fn aligned_row_pitch(width: u32) -> Result<u64, RendererError> {
    let bytes = u64::from(width)
        .checked_mul(u64::from(PIXEL_BYTES))
        .ok_or(RendererError::Rhi(Error::InvalidRange))?;
    bytes
        .checked_add(COPY_ROW_ALIGNMENT - 1)
        .map(|value| value / COPY_ROW_ALIGNMENT * COPY_ROW_ALIGNMENT)
        .ok_or(RendererError::Rhi(Error::InvalidRange))
}

pub(super) fn validate_compute_input<B: Backend>(
    bytes: &[u8],
    pipeline: DeviceId,
    layout: DeviceId,
    device: &Device<B>,
) -> Result<u64, RendererError> {
    if pipeline != device.id() || layout != device.id() {
        return Err(RendererError::Rhi(Error::WrongDevice));
    }
    if device.compute().is_none() {
        return Err(RendererError::UnsupportedCompute);
    }
    let groups = compute_dispatch_groups(validate_compute_bytes(bytes)?)?;
    let len = u64::try_from(bytes.len()).map_err(|_| RendererError::InvalidComputeInput)?;
    validate_compute_requirements(device, len, groups)?;
    Ok(len)
}
pub(super) fn validate_compute_bytes(bytes: &[u8]) -> Result<u32, RendererError> {
    if bytes.is_empty() || bytes.len() % COMPUTE_ELEMENT_BYTES != 0 {
        return Err(RendererError::InvalidComputeInput);
    }
    u32::try_from(bytes.len() / COMPUTE_ELEMENT_BYTES)
        .map_err(|_| RendererError::InvalidComputeInput)
}
pub(super) fn compute_dispatch_groups(count: u32) -> Result<u32, RendererError> {
    count
        .checked_add(COMPUTE_WORKGROUP_SIZE - 1)
        .map(|v| v / COMPUTE_WORKGROUP_SIZE)
        .ok_or(RendererError::InvalidComputeInput)
}
pub(super) fn validate_raster_input(
    bytes: &[u8],
    extent: [u32; 2],
    pipeline: DeviceId,
    device: DeviceId,
) -> Result<u64, RendererError> {
    if pipeline != device {
        return Err(RendererError::Rhi(Error::WrongDevice));
    }
    validate_raster_bytes(bytes, extent)?;
    u64::try_from(bytes.len()).map_err(|_| RendererError::InvalidRasterInput)
}
pub(super) fn validate_raster_bytes(bytes: &[u8], extent: [u32; 2]) -> Result<(), RendererError> {
    if extent.contains(&0) || bytes.is_empty() || bytes.len() % VERTEX_STRIDE as usize != 0 {
        return Err(RendererError::InvalidRasterInput);
    }
    let count = bytes.len() / VERTEX_STRIDE as usize;
    if count % 3 != 0 || count > u32::MAX as usize {
        return Err(RendererError::InvalidRasterInput);
    }
    for scalar in bytes.chunks_exact(4) {
        if !f32::from_le_bytes(
            scalar
                .try_into()
                .map_err(|_| RendererError::InvalidRasterInput)?,
        )
        .is_finite()
        {
            return Err(RendererError::InvalidRasterInput);
        }
    }
    Ok(())
}
pub(super) fn strip_raster_padding(
    bytes: Vec<u8>,
    extent: [u32; 2],
    row_pitch: u32,
) -> Result<Vec<u8>, RendererError> {
    let row = extent[0]
        .checked_mul(PIXEL_BYTES)
        .ok_or(RendererError::InvalidRasterInput)?;
    if row_pitch < row {
        return Err(RendererError::InvalidRasterInput);
    }
    let required = usize::try_from(
        u64::from(row_pitch)
            .checked_mul(u64::from(extent[1]))
            .ok_or(RendererError::InvalidRasterInput)?,
    )
    .map_err(|_| RendererError::InvalidRasterInput)?;
    if bytes.len() < required {
        return Err(RendererError::InvalidRasterInput);
    }
    let mut tight = Vec::with_capacity(
        usize::try_from(
            u64::from(row)
                .checked_mul(u64::from(extent[1]))
                .ok_or(RendererError::InvalidRasterInput)?,
        )
        .map_err(|_| RendererError::InvalidRasterInput)?,
    );
    for source in bytes[..required].chunks_exact(row_pitch as usize) {
        tight.extend_from_slice(&source[..row as usize]);
    }
    Ok(tight)
}
