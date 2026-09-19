//! Production immutable uploads and their structured failure stages.
//!
//! Staging/mapping mechanics and test readback observers live in private
//! children. Production uploads retain staging through submission and never
//! infer completion from a test observer.

use super::*;

pub(crate) fn upload_immutable_buffer(
    owner: &Arc<OpenedDevice>,
    target: &OwnedBuffer,
    target_lease: ResourceLease,
    bytes: &[u8],
) -> Result<NativeCompletion, (BufferUploadStage, String)> {
    let staging = create_staging_buffer(
        owner,
        bytes.len() as u64,
        wgt::BufferUses::MAP_WRITE | wgt::BufferUses::COPY_SRC,
    )
    .map_err(|reason| (BufferUploadStage::Staging, reason))?;
    with_mapped_write(&staging, bytes).map_err(|reason| (BufferUploadStage::Staging, reason))?;
    let mut encoder =
        begin_copy_encoder(owner).map_err(|reason| (BufferUploadStage::Recording, reason))?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopySource,
    )
    .map_err(|reason| (BufferUploadStage::Recording, reason))?;
    transition_buffer(
        &mut encoder,
        target,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )
    .map_err(|reason| (BufferUploadStage::Recording, reason))?;
    copy_buffer(
        &mut encoder,
        &staging,
        target,
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size: bytes.len() as u64,
        },
    )
    .map_err(|reason| (BufferUploadStage::Recording, reason))?;
    let command =
        finish_copy_encoder(encoder).map_err(|reason| (BufferUploadStage::Recording, reason))?;
    submit_copy_with_staging(command, vec![target_lease], vec![staging])
        .map_err(|reason| (BufferUploadStage::SubmitRejected, reason))
}

pub(crate) fn upload_immutable_texture(
    owner: &Arc<OpenedDevice>,
    target: &OwnedTexture,
    target_lease: ResourceLease,
    descriptor: TextureDesc,
    tight: &[u8],
) -> Result<NativeCompletion, (TextureUploadStage, String)> {
    let row_bytes = descriptor
        .extent
        .width
        .checked_mul(4)
        .ok_or_else(|| (TextureUploadStage::Staging, "row size overflow".into()))?;
    let pitch = row_bytes
        .checked_add(255)
        .and_then(|value| (value / 256).checked_mul(256))
        .ok_or_else(|| (TextureUploadStage::Staging, "row pitch overflow".into()))?;
    let staging_size = u64::from(pitch)
        .checked_mul(u64::from(descriptor.extent.height))
        .ok_or_else(|| (TextureUploadStage::Staging, "staging size overflow".into()))?;
    let tight_size = u64::from(row_bytes)
        .checked_mul(u64::from(descriptor.extent.height))
        .ok_or_else(|| (TextureUploadStage::Staging, "tight size overflow".into()))?;
    if u64::try_from(tight.len()).ok() != Some(tight_size) {
        return Err((
            TextureUploadStage::Staging,
            "tight texture upload length mismatch".into(),
        ));
    }
    let staging_len = usize::try_from(staging_size).map_err(|_| {
        (
            TextureUploadStage::Staging,
            "staging size exceeds usize".into(),
        )
    })?;
    let pitch_len = usize::try_from(pitch).map_err(|_| {
        (
            TextureUploadStage::Staging,
            "row pitch exceeds usize".into(),
        )
    })?;
    let row_len = usize::try_from(row_bytes)
        .map_err(|_| (TextureUploadStage::Staging, "row size exceeds usize".into()))?;
    let mut padded = vec![0xEE; staging_len];
    for (destination, source) in padded
        .chunks_exact_mut(pitch_len)
        .zip(tight.chunks_exact(row_len))
    {
        destination[..row_len].copy_from_slice(source);
    }
    let staging = create_staging_buffer(
        owner,
        staging_size,
        wgt::BufferUses::MAP_WRITE | wgt::BufferUses::COPY_SRC,
    )
    .map_err(|reason| (TextureUploadStage::Staging, reason))?;
    with_mapped_write(&staging, &padded).map_err(|reason| (TextureUploadStage::Staging, reason))?;
    let mut encoder =
        begin_copy_encoder(owner).map_err(|reason| (TextureUploadStage::Recording, reason))?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopySource,
    )
    .map_err(|reason| (TextureUploadStage::Recording, reason))?;
    transition_texture(
        &mut encoder,
        target,
        descriptor,
        TextureRange::Whole,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )
    .map_err(|reason| (TextureUploadStage::Recording, reason))?;
    copy_buffer_to_texture(
        &mut encoder,
        &staging,
        target,
        pitch,
        descriptor.extent.width,
        descriptor.extent.height,
    )
    .map_err(|reason| (TextureUploadStage::Recording, reason))?;
    let command =
        finish_copy_encoder(encoder).map_err(|reason| (TextureUploadStage::Recording, reason))?;
    submit_copy_with_staging(command, vec![target_lease], vec![staging])
        .map_err(|reason| (TextureUploadStage::SubmitRejected, reason))
}
