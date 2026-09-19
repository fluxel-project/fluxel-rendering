//! Compute rejection witness.

use crate::*;
use fluxel_rendergraph::*;
pub(super) fn run_compute_negative_paths(backend_kind: crate::Backend) {
    let device = Device::open(
        backend_kind,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let usage = BufferUsage::from_kinds([
        fluxel_rendergraph::BufferUsageKind::StorageRead,
        fluxel_rendergraph::BufferUsageKind::StorageWrite,
    ]);
    let buffer = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 64 },
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let pipeline_a = device
        .create_compute_pipeline(crate::ComputeKernel::WrappingAdd)
        .unwrap();
    let pipeline_b = device
        .create_compute_pipeline(crate::ComputeKernel::WrappingMultiply)
        .unwrap();
    let binding_a = device
        .create_compute_bindings(&pipeline_a, &buffer, 0, 64)
        .unwrap();
    let binding_b = device
        .create_compute_bindings(&pipeline_b, &buffer, 0, 64)
        .unwrap();
    let mut provider = ComputeObjectProvider::new(&device);
    let pipeline_id = ComputePipelineId::new(901);
    let binding_id = BindingSetId::new(902);
    provider
        .register_pipeline(pipeline_id, pipeline_a.clone())
        .unwrap();
    provider.register_bindings(binding_id, pipeline_id).unwrap();
    assert!(
        provider
            .compute_pipeline(ComputePipelineId::new(999))
            .is_err()
    );
    assert!(provider.bindings(BindingSetId::new(999), &[], &[]).is_err());
    let wrong_semantic = [ResolvedBindingResource::Buffer {
        physical: &buffer,
        range: BufferRange::Whole,
        semantic: fluxel_rendergraph::BindingResourceSemantic::BufferRead(
            fluxel_rendergraph::BufferReadUse::Storage,
        ),
    }];
    assert!(provider.bindings(binding_id, &wrong_semantic, &[]).is_err());
    let misaligned = [ResolvedBindingResource::Buffer {
        physical: &buffer,
        range: BufferRange::Bytes { offset: 2, size: 4 },
        semantic: fluxel_rendergraph::BindingResourceSemantic::BufferReadWrite(
            BufferReadWriteUse::Storage,
        ),
    }];
    assert!(provider.bindings(binding_id, &misaligned, &[]).is_err());
    let foreign_device = Device::open(backend_kind, crate::DeviceOptions::default()).unwrap();
    let foreign_pipeline = foreign_device
        .create_compute_pipeline(crate::ComputeKernel::WrappingAdd)
        .unwrap();
    let mut backend = ComputeBackend::new(device.clone());
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    assert_eq!(
        backend.end_copy(&mut encoder),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    backend.begin_copy(&mut encoder, "copy-state").unwrap();
    assert_eq!(
        backend.begin_copy(&mut encoder, "nested-copy"),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    backend.end_copy(&mut encoder).unwrap();
    assert_eq!(
        backend.end_copy(&mut encoder),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    assert_eq!(
        backend.set_compute_pipeline(&mut encoder, &pipeline_a),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &binding_a),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert_eq!(
        backend.dispatch(&mut encoder, [1, 1, 1]),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    backend.begin_compute(&mut encoder, "negative").unwrap();
    assert_eq!(
        backend.transition_buffer(
            &mut encoder,
            &buffer,
            BufferRange::Whole,
            ResourceAccessState::ShaderStorageReadWrite,
            ResourceAccessState::CopySource,
        ),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    assert_eq!(
        backend.begin_copy(&mut encoder, "nested-copy"),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    assert_eq!(
        backend.copy_buffer(
            &mut encoder,
            &buffer,
            &buffer,
            BufferCopyRegion {
                source_offset: 0,
                destination_offset: 0,
                size: 4,
            },
        ),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    assert_eq!(
        backend.dispatch(&mut encoder, [1, 1, 1]),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert_eq!(
        backend.set_compute_pipeline(&mut encoder, &foreign_pipeline),
        Err(NativeExecutionError::ForeignResource)
    );
    backend
        .set_compute_pipeline(&mut encoder, &pipeline_a)
        .unwrap();
    assert_eq!(
        backend.dispatch(&mut encoder, [1, 1, 1]),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &binding_b),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    backend.set_bindings(&mut encoder, &binding_a).unwrap();
    let maximum = backend
        .capabilities()
        .limits
        .max_compute_workgroups_per_dimension;
    assert_eq!(
        backend.dispatch(&mut encoder, [maximum[0].saturating_add(1), 1, 1]),
        Err(NativeExecutionError::InvalidDispatch)
    );
    // No finish/submit: every negative assertion is pre-submission.
    drop(encoder);
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend.begin_copy(&mut encoder, "unfinished-copy").unwrap();
    assert!(matches!(
        backend.finish_encoder(encoder),
        Err(NativeExecutionError::CopyStateMismatch)
    ));
    assert!(crate::imp::validation_diagnostics(&device.inner).is_empty());
}
