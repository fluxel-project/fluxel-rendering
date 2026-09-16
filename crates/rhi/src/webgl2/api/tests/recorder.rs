//! Contract tests for the recorder oracle itself: object identity, epoch
//! restoration and the ordering of a rejection against its side effect.

use super::*;

#[test]
fn mock_restoration_reuses_raw_slot_only_in_a_new_epoch() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let old = api
        .create_buffer_resource(GlBufferDesc {
            size: 4,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("old buffer");
    api.context_lost().expect("loss");
    let new_stamp = api.context_restored().expect("restore");
    let new = api
        .create_buffer_resource(GlBufferDesc {
            size: 4,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("new buffer");
    assert_eq!(old.slot, new.slot);
    assert_ne!(old.context, new.context);
    assert_eq!(api.discovery().context_stamp(), new_stamp);
}

#[test]
fn mock_rejects_foreign_and_stale_objects_and_keeps_exact_trace() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let old = api
        .create_buffer_resource(GlBufferDesc {
            size: 4,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("buffer");
    let foreign = BufferId::new(
        ContextStamp::new(
            DeviceIdentity::new(8).expect("device"),
            ContextEpoch::INITIAL,
        ),
        old.slot,
        old.generation,
    );
    assert!(matches!(
        api.destroy_buffer_resource(foreign),
        Err(GlError::WrongContext { .. })
    ));
    api.context_lost().expect("loss");
    api.context_restored().expect("restore");
    assert!(matches!(
        api.destroy_buffer_resource(old),
        Err(GlError::StaleObject { .. })
    ));
    assert_eq!(
        api.calls(),
        &[
            MockCall::CreateBuffer(old),
            MockCall::Error(GlError::WrongContext {
                operation: "destroy-buffer",
                object: foreign.context,
                current: old.context,
            }),
            MockCall::ContextLost,
            MockCall::ContextRestored(api.context_stamp()),
            MockCall::Error(GlError::StaleObject {
                operation: "destroy-buffer",
                object: old.context,
                current: api.context_stamp(),
            }),
        ]
    );
}

#[test]
fn mock_injected_error_precedes_driver_side_effect() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let error = GlError::Driver {
        operation: "mock",
        message: "injected".into(),
    };
    api.fail_next(error.clone());
    assert_eq!(
        api.create_texture_resource(GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 1,
                height: 1,
                depth_or_layers: 1
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::SAMPLED
        }),
        Err(error.clone())
    );
    assert_eq!(api.calls(), &[MockCall::Error(error)]);
}

#[test]
fn webgl_mock_never_gains_compute_or_storage_and_desktop_wrapper_requires_evidence() {
    fn requires_compute_and_storage<T: GlComputeDispatchApi + GlStorageBufferApi>(_: &mut T) {}

    let webgl = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    assert!(MockComputeStorageApi::new(webgl).is_err());

    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(false),
    )
    .expect("desktop discovery");
    builder.resolve(GlCapability::Compute, compute(), GlOperationProbe::Passed);
    builder.resolve(
        GlCapability::StorageBuffer,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 3)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderStorageBufferObject),
            extension_requires_probe: true,
        },
        GlOperationProbe::Passed,
    );
    let mut compute = MockComputeStorageApi::new(MockGlFamilyApi::from_discovery(builder.build()))
        .expect("proven desktop compute/storage");
    requires_compute_and_storage(&mut compute);
}

#[test]
fn mock_rejects_command_on_non_owner_thread_before_recording() {
    let api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let result = std::thread::spawn(move || api.assert_owner_thread("thread-test"))
        .join()
        .expect("thread joins");
    assert!(matches!(
        result,
        Err(GlError::WrongThread {
            operation: "thread-test",
            ..
        })
    ));
}
