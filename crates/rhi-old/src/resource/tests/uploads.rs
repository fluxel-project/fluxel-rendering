//! Resource uploads contract tests.

use super::*;

pub(super) fn immutable_upload_requires_exact_device_only_copy_destination_bytes() {
    let descriptor = BufferDescriptor {
        buffer: BufferDesc { size: 8 },
        usage: BufferUsage::empty().with(BufferUsageKind::CopyDestination),
        memory: MemoryPolicy::DeviceOnly,
    };
    assert_eq!(
        validate_immutable_upload_descriptor(descriptor, &[]),
        Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::EmptyData
        ))
    );
    assert_eq!(
        validate_immutable_upload_descriptor(descriptor, &[0; 6]),
        Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::DataLengthNotCopyAligned
        ))
    );
    assert_eq!(
        validate_immutable_upload_descriptor(descriptor, &[0; 4]),
        Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::DescriptorSizeMismatch
        ))
    );
    assert_eq!(
        validate_immutable_upload_descriptor(
            BufferDescriptor {
                usage: BufferUsage::empty(),
                ..descriptor
            },
            &[0; 8]
        ),
        Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::CopyDestinationUsageRequired
        ))
    );
    assert_eq!(
        validate_immutable_upload_descriptor(descriptor, &[0; 8]),
        Ok(())
    );
}

#[cfg(all(windows, feature = "dx12"))]
pub(super) fn immutable_upload_dx12_failure_contract() {
    run_immutable_upload_failure_contract(crate::Backend::Dx12);
}

#[cfg(all(windows, feature = "vulkan"))]
pub(super) fn immutable_upload_vulkan_failure_contract() {
    run_immutable_upload_failure_contract(crate::Backend::Vulkan);
}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
fn run_immutable_upload_failure_contract(backend: crate::Backend) {
    use core::time::Duration;

    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let descriptor = BufferDescriptor {
        buffer: BufferDesc { size: 8 },
        usage: BufferUsage::from_kinds([BufferUsageKind::CopyDestination]),
        memory: MemoryPolicy::DeviceOnly,
    };

    crate::imp::inject_submit_rejected_once();
    assert!(matches!(
        device.upload_immutable_buffer(descriptor, &[0; 8]),
        Err(BufferUploadError::Native {
            stage: BufferUploadStage::SubmitRejected,
            ..
        })
    ));

    crate::imp::inject_submit_accepted_unknown_once();
    let accepted_unknown = device.upload_immutable_buffer(descriptor, &[1; 8]).unwrap();
    assert_eq!(
        accepted_unknown.status().unwrap(),
        CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
    );
    let incomplete = accepted_unknown.finalize().unwrap_err();
    assert_eq!(
        incomplete.status(),
        CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
    );
    drop(incomplete);

    let pending = device.upload_immutable_buffer(descriptor, &[2; 8]).unwrap();
    crate::imp::inject_completion_pending_once();
    assert_eq!(pending.status().unwrap(), CompletionStatus::Pending);
    assert_eq!(
        pending.wait(Duration::from_secs(10)).unwrap(),
        CompletionStatus::Complete
    );
    assert!(pending.finalize().is_ok());
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(diagnostics.is_empty(), "{backend:?}: {diagnostics:?}");
    eprintln!(
        "artifact case=U01-failure backend={backend:?} submit_rejected=true accepted_unknown=DeviceLost observation=Pending completion=Complete diagnostics={diagnostics:?}"
    );
}

#[cfg(all(windows, feature = "dx12"))]
pub(super) fn immutable_texture_upload_dx12_failure_contract() {
    run_immutable_texture_upload_failure_contract(crate::Backend::Dx12);
}

#[cfg(all(windows, feature = "vulkan"))]
pub(super) fn immutable_texture_upload_vulkan_failure_contract() {
    run_immutable_texture_upload_failure_contract(crate::Backend::Vulkan);
}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
fn run_immutable_texture_upload_failure_contract(backend: crate::Backend) {
    use core::time::Duration;

    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let descriptor = TextureDescriptor {
        texture: TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 3,
                height: 2,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
        usage: TextureUsage::from_kinds([
            TextureUsageKind::CopyDestination,
            TextureUsageKind::Sampled,
            TextureUsageKind::CopySource,
        ]),
        memory: MemoryPolicy::DeviceOnly,
    };

    crate::imp::inject_submit_rejected_once();
    assert!(matches!(
        device.upload_immutable_texture(descriptor, &[0; 24]),
        Err(TextureUploadError::Native {
            stage: TextureUploadStage::SubmitRejected,
            ..
        })
    ));

    crate::imp::inject_submit_accepted_unknown_once();
    let unknown = device
        .upload_immutable_texture(descriptor, &[1; 24])
        .unwrap();
    assert_eq!(
        unknown.status().unwrap(),
        CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
    );
    let incomplete = unknown.finalize().unwrap_err();
    assert_eq!(
        incomplete.status(),
        CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
    );
    // An unproven accepted submission never publishes a texture. Dropping
    // this retained operation takes the same quarantine path as a caller
    // that abandons it before observing completion.
    drop(incomplete);

    let pending = device
        .upload_immutable_texture(descriptor, &[2; 24])
        .unwrap();
    crate::imp::inject_completion_pending_once();
    assert_eq!(pending.status().unwrap(), CompletionStatus::Pending);
    assert_eq!(
        pending.wait(Duration::from_secs(10)).unwrap(),
        CompletionStatus::Complete
    );
    let uploaded = pending.finalize().expect("complete upload is published");
    assert_eq!(
        uploaded.outgoing_state(),
        ResourceAccessState::CopyDestination
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(diagnostics.is_empty(), "{backend:?}: {diagnostics:?}");
    eprintln!(
        "artifact case=D2-failure backend={backend:?} submit_rejected=true accepted_unknown=DeviceLost observation=Pending completion=Complete diagnostics={diagnostics:?}"
    );
}
