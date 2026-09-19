//! Shared native-boundary validation helpers.

use super::*;

pub(in crate::execution) fn validate_buffer_copy(
    source: BufferDesc,
    destination: BufferDesc,
    region: BufferCopyRegion,
) -> Result<(), NativeExecutionError> {
    if region.size == 0 {
        return Err(NativeExecutionError::InvalidTransfer(
            "buffer copy size is zero",
        ));
    }
    if !region.source_offset.is_multiple_of(4)
        || !region.destination_offset.is_multiple_of(4)
        || !region.size.is_multiple_of(4)
    {
        return Err(NativeExecutionError::InvalidTransfer(
            "buffer copy offsets and size must be 4-byte aligned",
        ));
    }
    let source_end = region.source_offset.checked_add(region.size).ok_or(
        NativeExecutionError::InvalidTransfer("source range overflow"),
    )?;
    let destination_end = region.destination_offset.checked_add(region.size).ok_or(
        NativeExecutionError::InvalidTransfer("destination range overflow"),
    )?;
    if source_end > source.size || destination_end > destination.size {
        return Err(NativeExecutionError::InvalidTransfer(
            "buffer copy is out of bounds",
        ));
    }
    Ok(())
}

pub(in crate::execution) fn validate_texture_copy(
    source: TextureDesc,
    destination: TextureDesc,
    region: TextureCopyRegion,
) -> Result<(), NativeExecutionError> {
    if source.format != destination.format {
        return Err(NativeExecutionError::InvalidTransfer(
            "texture formats differ",
        ));
    }
    if region.extent.contains(&0) {
        return Err(NativeExecutionError::InvalidTransfer(
            "texture copy extent is zero",
        ));
    }
    if region.source_mip_level >= source.mip_levels
        || region.destination_mip_level >= destination.mip_levels
    {
        return Err(NativeExecutionError::InvalidTransfer(
            "texture mip is out of bounds",
        ));
    }
    let mip_extent = |desc: TextureDesc, mip: u32| {
        [
            (desc.extent.width >> mip).max(1),
            (desc.extent.height >> mip).max(1),
            (desc.extent.depth >> mip).max(1),
        ]
    };
    for (origin, extent, limit) in [
        (
            region.source_origin,
            region.extent,
            mip_extent(source, region.source_mip_level),
        ),
        (
            region.destination_origin,
            region.extent,
            mip_extent(destination, region.destination_mip_level),
        ),
    ] {
        for axis in 0..3 {
            if origin[axis]
                .checked_add(extent[axis])
                .is_none_or(|end| end > limit[axis])
            {
                return Err(NativeExecutionError::InvalidTransfer(
                    "texture copy is out of bounds",
                ));
            }
        }
    }
    Ok(())
}

impl CopyBackend {
    pub(in crate::execution) fn check_encoder(
        &self,
        encoder: &CopyEncoder,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            encoder.device,
            self.device.identity(),
            NativeExecutionError::ForeignEncoder,
        )
    }

    pub(in crate::execution) fn check_buffer(
        &self,
        buffer: &Buffer,
    ) -> Result<(), NativeExecutionError> {
        (buffer.device_identity() == self.device.identity())
            .then_some(())
            .ok_or(NativeExecutionError::ForeignResource)
    }
    pub(in crate::execution) fn check_texture(
        &self,
        texture: &Texture,
    ) -> Result<(), NativeExecutionError> {
        (texture.device_identity() == self.device.identity())
            .then_some(())
            .ok_or(NativeExecutionError::ForeignResource)
    }
}

pub(in crate::execution) fn require_device(
    actual: fluxel_rendergraph::DeviceIdentity,
    expected: fluxel_rendergraph::DeviceIdentity,
    mismatch: NativeExecutionError,
) -> Result<(), NativeExecutionError> {
    (actual == expected).then_some(()).ok_or(mismatch)
}
