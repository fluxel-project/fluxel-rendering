//! Test-only upload and readback observers.
//!
//! These helpers may wait for completion and quarantine staging allocations when
//! completion is unknowable; production upload never uses their observation path.

use super::*;

#[cfg(any(test, feature = "test-support"))]
pub(crate) struct TextureReadback {
    pub(crate) tight: Vec<u8>,
    pub(crate) padded: Vec<u8>,
    pub(crate) bytes_per_row: u32,
}

#[cfg(test)]
pub(crate) fn upload_buffer_for_test(
    owner: &Arc<OpenedDevice>,
    target: &OwnedBuffer,
    target_lease: ResourceLease,
    bytes: &[u8],
) -> Result<ResourceAccessState, String> {
    let completion = upload_immutable_buffer(owner, target, target_lease, bytes)
        .map_err(|(_, reason)| reason)?;
    let status = wait_completion(&completion, core::time::Duration::from_secs(10))?;
    if status != CompletionStatus::Complete {
        return Err(format!("upload did not complete: {status:?}"));
    }
    drop(completion);
    Ok(ResourceAccessState::CopyDestination)
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn readback_buffer_for_test(
    owner: &Arc<OpenedDevice>,
    source: &OwnedBuffer,
    source_lease: ResourceLease,
    incoming_state: ResourceAccessState,
    size: u64,
) -> Result<Vec<u8>, String> {
    let staging = create_staging_buffer(
        owner,
        size,
        wgt::BufferUses::MAP_READ | wgt::BufferUses::COPY_DST,
    )?;
    let mut encoder = begin_copy_encoder(owner)?;
    transition_buffer(
        &mut encoder,
        source,
        incoming_state,
        ResourceAccessState::CopySource,
    )?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )?;
    copy_buffer(
        &mut encoder,
        source,
        &staging,
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size,
        },
    )?;
    // The test observer is not allowed to leave the physical resource in a
    // state different from the `UploadedBuffer` fact it borrowed. Restore the
    // exact incoming state in the same ordered submission after the copy.
    transition_buffer(
        &mut encoder,
        source,
        ResourceAccessState::CopySource,
        incoming_state,
    )?;
    let completion = submit_copy(finish_copy_encoder(encoder)?, vec![source_lease])?;
    let status = match wait_completion(&completion, core::time::Duration::from_secs(10)) {
        Ok(status) => status,
        Err(error) => {
            // The observer cannot prove whether the accepted copy still uses
            // staging. Quarantine it for process lifetime rather than violate
            // the HAL resource-lifetime contract.
            std::mem::forget(staging);
            let _quarantined_completion = std::mem::ManuallyDrop::new(completion);
            return Err(error);
        }
    };
    if status != CompletionStatus::Complete {
        // Pending and failed/accepted-unknown are both unproven retirement
        // states. This test-only path safely quarantines staging.
        std::mem::forget(staging);
        let _quarantined_completion = std::mem::ManuallyDrop::new(completion);
        return Err(format!("readback did not complete: {status:?}"));
    }
    drop(completion);
    let bytes = with_mapped_read(&staging, size)?;
    drop(staging);
    Ok(bytes)
}

#[cfg(any(test, feature = "test-support"))]
#[cfg(test)]
pub(crate) fn upload_texture_for_test(
    owner: &Arc<OpenedDevice>,
    target: &OwnedTexture,
    target_lease: ResourceLease,
    descriptor: TextureDesc,
    tight: &[u8],
) -> Result<ResourceAccessState, String> {
    let completion = upload_immutable_texture(owner, target, target_lease, descriptor, tight)
        .map_err(|(_, reason)| reason)?;
    let status = wait_completion(&completion, core::time::Duration::from_secs(10))?;
    if status != CompletionStatus::Complete {
        return Err(format!("texture upload did not complete: {status:?}"));
    }
    Ok(ResourceAccessState::CopyDestination)
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn readback_texture_for_test(
    owner: &Arc<OpenedDevice>,
    source: &OwnedTexture,
    source_lease: ResourceLease,
    descriptor: TextureDesc,
    incoming_state: ResourceAccessState,
) -> Result<TextureReadback, String> {
    let row_bytes = descriptor
        .extent
        .width
        .checked_mul(4)
        .ok_or("row size overflow")?;
    let pitch = row_bytes
        .checked_add(255)
        .and_then(|value| (value / 256).checked_mul(256))
        .ok_or("row pitch overflow")?;
    let staging_size = u64::from(pitch)
        .checked_mul(u64::from(descriptor.extent.height))
        .ok_or("staging size overflow")?;
    let staging = create_staging_buffer(
        owner,
        staging_size,
        wgt::BufferUses::MAP_READ | wgt::BufferUses::COPY_DST,
    )?;
    let mut encoder = begin_copy_encoder(owner)?;
    transition_texture(
        &mut encoder,
        source,
        descriptor,
        TextureRange::Whole,
        incoming_state,
        ResourceAccessState::CopySource,
    )?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )?;
    clear_buffer(&mut encoder, &staging, staging_size)?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::CopyDestination,
        ResourceAccessState::CopyDestination,
    )?;
    copy_texture_to_buffer(
        &mut encoder,
        source,
        &staging,
        pitch,
        descriptor.extent.width,
        descriptor.extent.height,
    )?;
    // Preserve the graph export's reported state after observation so every
    // surviving alias remains truthful for a later renderer use.
    transition_texture(
        &mut encoder,
        source,
        descriptor,
        TextureRange::Whole,
        ResourceAccessState::CopySource,
        incoming_state,
    )?;
    let completion = submit_copy(finish_copy_encoder(encoder)?, vec![source_lease])?;
    let status = match wait_completion(&completion, core::time::Duration::from_secs(10)) {
        Ok(status) => status,
        Err(error) => {
            std::mem::forget(staging);
            let _quarantined_completion = std::mem::ManuallyDrop::new(completion);
            return Err(error);
        }
    };
    if status != CompletionStatus::Complete {
        std::mem::forget(staging);
        let _quarantined_completion = std::mem::ManuallyDrop::new(completion);
        return Err(format!("texture readback did not complete: {status:?}"));
    }
    drop(completion);
    let padded = with_mapped_read(&staging, staging_size)?;
    let tight_size = u64::from(row_bytes)
        .checked_mul(u64::from(descriptor.extent.height))
        .ok_or("tight size overflow")?;
    let tight_capacity = usize::try_from(tight_size).map_err(|_| "tight size exceeds usize")?;
    let pitch_len = usize::try_from(pitch).map_err(|_| "row pitch exceeds usize")?;
    let row_len = usize::try_from(row_bytes).map_err(|_| "row size exceeds usize")?;
    let mut tight = Vec::with_capacity(tight_capacity);
    for row in padded.chunks_exact(pitch_len) {
        tight.extend_from_slice(&row[..row_len]);
    }
    drop(staging);
    Ok(TextureReadback {
        tight,
        padded,
        bytes_per_row: pitch,
    })
}
