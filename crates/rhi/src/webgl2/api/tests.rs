use super::*;

fn limits() -> GlLimits {
    GlLimits {
        max_texture_size: 2_048,
        max_3d_texture_size: 256,
        max_array_texture_layers: 256,
        max_cube_map_texture_size: 2_048,
        max_renderbuffer_size: 4_096,
        max_color_attachments: 4,
        max_draw_buffers: 4,
        max_vertex_attributes: 16,
        max_viewport_dimensions: [2_048; 2],
        max_viewports: 1,
        max_vertex_texture_image_units: 16,
        max_fragment_texture_image_units: 16,
        max_combined_texture_image_units: 16,
        max_uniform_buffer_bindings: 24,
        max_uniform_block_size: 16_384,
        uniform_buffer_offset_alignment: 256,
        max_vertex_uniform_blocks: 12,
        max_fragment_uniform_blocks: 12,
        max_compute_uniform_blocks: 12,
        max_combined_uniform_blocks: 24,
        max_storage_buffer_bindings: 8,
        max_storage_block_size: 1 << 27,
        storage_buffer_offset_alignment: 256,
        max_vertex_storage_blocks: 4,
        max_fragment_storage_blocks: 4,
        max_compute_storage_blocks: 4,
        max_combined_storage_blocks: 8,
        max_image_units: 4,
        max_combined_image_units: 4,
        max_samples: 4,
        max_color_texture_samples: 4,
        max_depth_texture_samples: 4,
        max_integer_samples: 4,
        max_compute_work_group_count: [65_535; 3],
        max_compute_work_group_size: [1_024, 1_024, 64],
        max_compute_work_group_invocations: 1_024,
        max_multi_draw_indirect_count: Some(1),
        query_counter_bits: 32,
        max_texture_anisotropy: GlFiniteF32::new(16.0),
    }
}
fn desktop_limits() -> GlLimits {
    let mut limits = limits();
    limits.max_texture_size = 16_384;
    limits.max_3d_texture_size = 2_048;
    limits.max_array_texture_layers = 2_048;
    limits.max_cube_map_texture_size = 16_384;
    limits.max_renderbuffer_size = 16_384;
    limits.max_color_attachments = 8;
    limits.max_draw_buffers = 8;
    limits.max_viewport_dimensions = [16_384; 2];
    limits.max_uniform_buffer_bindings = 36;
    limits
}
fn context(profile: GlFamilyProfile) -> GlContextInfo {
    GlContextInfo::new(
        profile,
        "version",
        "glsl",
        "vendor",
        "renderer",
        "driver",
        GlContextFlags::default(),
    )
}
fn formats(storage: bool) -> GlFormatTable {
    let mut t = GlFormatTable::default();
    for format in [
        GlFormat::Rgba8Unorm,
        GlFormat::Rgba8Srgb,
        GlFormat::Depth32Float,
    ] {
        t.record(GlFormatCapabilities {
            format,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: if storage && format == GlFormat::Rgba8Unorm {
                GlFormatEvidence::OperationProbed
            } else {
                GlFormatEvidence::CoreGuaranteed
            },
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: true,
            storage_read: storage && format == GlFormat::Rgba8Unorm,
            storage_write: storage && format == GlFormat::Rgba8Unorm,
            copy_source: true,
            copy_destination: true,
        })
        .expect("unique fact");
    }
    t
}
fn stamp(epoch: ContextEpoch) -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(7).expect("identity"), epoch)
}
fn compute() -> CoreOrExtension {
    CoreOrExtension {
        desktop_core: Some(GlVersion::new(4, 3)),
        embedded_core: Some(GlVersion::new(3, 1)),
        extension: Some(GlKnownExtension::ArbComputeShader),
        extension_requires_probe: true,
    }
}

#[test]
fn builder_binds_core_evidence_to_its_own_context() {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(false),
    )
    .expect("desktop facts");
    builder.resolve(GlCapability::Compute, compute(), GlOperationProbe::Passed);
    assert!(
        builder
            .build()
            .capabilities()
            .supports(GlCapability::Compute)
    );
}
#[test]
fn webgl_cannot_receive_desktop_core_evidence() {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(false),
    )
    .expect("web facts");
    builder.resolve(GlCapability::Compute, compute(), GlOperationProbe::Passed);
    let snapshot = builder.build();
    assert_eq!(
        snapshot
            .capabilities()
            .fact(GlCapability::Compute)
            .expect("fact")
            .evidence,
        None
    );
    assert!(!snapshot.capabilities().supports(GlCapability::Compute));
}
#[test]
fn snapshot_carries_epoch_and_old_epoch_is_not_equal() {
    let current = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        formats(false),
    )
    .expect("facts")
    .build();
    let next = ContextEpoch::INITIAL.checked_next().expect("next");
    let restored = GlDiscoveryBuilder::new(
        stamp(next),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        formats(false),
    )
    .expect("facts")
    .build();
    assert_ne!(current.context_stamp(), restored.context_stamp());
}
#[test]
fn storage_image_requires_image_units_and_exact_read_write_format() {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(false),
    )
    .expect("facts");
    builder.resolve(
        GlCapability::StorageImage,
        CoreOrExtension {
            desktop_core: Some(GlVersion::new(4, 2)),
            embedded_core: Some(GlVersion::new(3, 1)),
            extension: Some(GlKnownExtension::ArbShaderImageLoadStore),
            extension_requires_probe: true,
        },
        GlOperationProbe::Passed,
    );
    assert!(
        !builder
            .build()
            .capabilities()
            .supports(GlCapability::StorageImage)
    );
}
#[test]
fn required_formats_and_sample_limits_are_enforced() {
    let error = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        GlFormatTable::default(),
    )
    .expect_err("empty format table");
    assert!(matches!(
        error,
        GlDiscoveryError::InvalidFormats(GlFormatTableError::MissingRequiredSampleOne { .. })
    ));
    let mut over = formats(false);
    over.record(GlFormatCapabilities {
        format: GlFormat::Rgba8Unorm,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 8,
        evidence: GlFormatEvidence::OperationProbed,
        sampled: true,
        filterable: true,
        renderable: true,
        blendable: true,
        storage_read: false,
        storage_write: false,
        copy_source: true,
        copy_destination: true,
    })
    .expect("new count");
    assert!(matches!(
        GlDiscoveryBuilder::new(
            stamp(ContextEpoch::INITIAL),
            context(GlFamilyProfile::WebGl2),
            GlExtensionSet::default(),
            limits(),
            over
        ),
        Err(GlDiscoveryError::InvalidFormats(
            GlFormatTableError::SampleCountExceedsLimit { .. }
        ))
    ));
}
#[test]
fn gles3_floor_is_accepted_and_non_gles3_is_rejected() {
    assert!(
        GlDiscoveryBuilder::new(
            stamp(ContextEpoch::INITIAL),
            context(GlFamilyProfile::Embedded { major: 3, minor: 0 }),
            GlExtensionSet::default(),
            limits(),
            formats(false)
        )
        .is_ok()
    );
    assert!(matches!(
        GlDiscoveryBuilder::new(
            stamp(ContextEpoch::INITIAL),
            context(GlFamilyProfile::Embedded { major: 4, minor: 0 }),
            GlExtensionSet::default(),
            limits(),
            formats(false)
        ),
        Err(GlDiscoveryError::InvalidProfile(_))
    ));
}
#[test]
fn finite_anisotropy_preserves_exact_bits() {
    let value = GlFiniteF32::new(3.5).expect("finite");
    assert_eq!(value.bits(), 3.5_f32.to_bits());
    assert_eq!(value.get(), 3.5);
    assert_eq!(GlFiniteF32::new(f32::NAN), None);
}

fn acquired_compressed_fact(format: GlFormat, extension: GlKnownExtension) -> GlFormatCapabilities {
    GlFormatCapabilities {
        format,
        resource_kind: GlFormatResourceKind::Texture,
        sample_count: 1,
        evidence: GlFormatEvidence::ExtensionAcquired(extension),
        sampled: true,
        filterable: true,
        renderable: false,
        blendable: false,
        storage_read: false,
        storage_write: false,
        copy_source: false,
        copy_destination: false,
    }
}

#[test]
fn extension_format_evidence_is_bound_to_this_profile_and_acquired_ledger() {
    let extension = GlKnownExtension::CompressedTextureS3tc;
    let mut reported_only = GlExtensionSet::default();
    reported_only.report_raw("WEBGL_compressed_texture_s3tc");
    let mut facts = formats(false);
    facts
        .record(acquired_compressed_fact(GlFormat::Bc1RgbaUnorm, extension))
        .expect("exact compressed fact");
    assert!(matches!(
        GlDiscoveryBuilder::new(
            stamp(ContextEpoch::INITIAL),
            context(GlFamilyProfile::WebGl2),
            reported_only,
            limits(),
            facts
        ),
        Err(GlDiscoveryError::InvalidFormats(
            GlFormatTableError::ExtensionNotAcquired { .. }
        ))
    ));

    let mut acquired = GlExtensionSet::default();
    acquired.report_raw("WEBGL_compressed_texture_s3tc");
    assert!(acquired.acquire(extension));
    let mut exact = formats(false);
    exact
        .record(acquired_compressed_fact(GlFormat::Bc1RgbaUnorm, extension))
        .expect("exact compressed fact");
    assert!(
        GlDiscoveryBuilder::new(
            stamp(ContextEpoch::INITIAL),
            context(GlFamilyProfile::WebGl2),
            acquired,
            limits(),
            exact
        )
        .is_ok()
    );

    let mut foreign = GlExtensionSet::default();
    foreign.report_raw("WEBGL_multi_draw");
    assert!(foreign.acquire(GlKnownExtension::WebglMultiDraw));
    let mut foreign_facts = formats(false);
    foreign_facts
        .record(acquired_compressed_fact(
            GlFormat::Bc1RgbaUnorm,
            GlKnownExtension::WebglMultiDraw,
        ))
        .expect("fact is structurally valid before profile binding");
    assert!(matches!(
        GlDiscoveryBuilder::new(
            stamp(ContextEpoch::INITIAL),
            context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
            foreign,
            desktop_limits(),
            foreign_facts
        ),
        Err(GlDiscoveryError::InvalidFormats(
            GlFormatTableError::ExtensionIllegalForProfile { .. }
        ))
    ));
}

fn snapshot(profile: GlFamilyProfile) -> GlDiscoverySnapshot {
    let limits = match profile {
        GlFamilyProfile::Desktop { .. } => desktop_limits(),
        _ => limits(),
    };
    GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(profile),
        GlExtensionSet::default(),
        limits,
        formats(false),
    )
    .expect("test discovery")
    .build()
}

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

fn mock_texture_desc() -> GlTextureDesc {
    texture_desc(1)
}

fn texture_desc(sample_count: u32) -> GlTextureDesc {
    GlTextureDesc {
        dimension: GlTextureDimension::D2,
        extent: GlExtent3d {
            width: 1,
            height: 1,
            depth_or_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        format: GlFormat::Rgba8Unorm,
        usage: GlTextureUsage::RENDER_ATTACHMENT,
    }
}

fn texture_view(texture: TextureId, sample_count: u32) -> GlTextureView {
    GlTextureView {
        target: GlAttachmentTarget::Texture(texture),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        width: 1,
        height: 1,
        sample_count,
    }
}

/// One extra exact fact beyond the single-sample baseline table.
fn snapshot_with_fact(
    resource_kind: GlFormatResourceKind,
    sample_count: u32,
) -> GlDiscoverySnapshot {
    let texture_facts = resource_kind == GlFormatResourceKind::Texture;
    let mut formats = formats(false);
    formats
        .record(GlFormatCapabilities {
            format: GlFormat::Rgba8Unorm,
            resource_kind,
            sample_count,
            evidence: GlFormatEvidence::CoreGuaranteed,
            sampled: texture_facts,
            filterable: texture_facts,
            renderable: true,
            blendable: texture_facts,
            storage_read: false,
            storage_write: false,
            copy_source: texture_facts,
            copy_destination: texture_facts,
        })
        .expect("extra exact fact");
    GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::WebGl2),
        GlExtensionSet::default(),
        limits(),
        formats,
    )
    .expect("test discovery")
    .build()
}

fn sampler_desc() -> GlSamplerDesc {
    GlSamplerDesc {
        address_mode_u: GlAddressMode::ClampToEdge,
        address_mode_v: GlAddressMode::ClampToEdge,
        address_mode_w: GlAddressMode::ClampToEdge,
        mag_filter: GlFilterMode::Linear,
        min_filter: GlFilterMode::Linear,
        mipmap_filter: GlMipmapFilterMode::Linear,
        lod_min_bits: 0.0f32.to_bits(),
        lod_max_bits: 1.0f32.to_bits(),
        compare: None,
        max_anisotropy_bits: None,
    }
}

#[test]
fn mock_binding_vocabulary_records_each_word_and_enforces_unit_limits() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = api
        .create_texture_resource(texture_desc(1))
        .expect("texture");
    let sampler = api.create_sampler(sampler_desc()).expect("sampler");
    let buffer = api
        .create_buffer_resource(GlBufferDesc {
            size: 512,
            usage: GlBufferUsage::UNIFORM,
        })
        .expect("uniform buffer");

    api.active_texture(3).expect("active texture");
    api.bind_texture(3, GlTextureTarget::D2, Some(texture))
        .expect("bind texture");
    api.bind_sampler(3, Some(sampler)).expect("bind sampler");
    api.bind_uniform_buffer(4, Some(buffer), 256, 256)
        .expect("bind uniform buffer");
    api.bind_texture(3, GlTextureTarget::D2, None)
        .expect("unbind texture");
    assert!(api.calls().ends_with(&[
        MockCall::ActiveTexture(3),
        MockCall::BindTexture {
            unit: 3,
            target: GlTextureTarget::D2,
            texture: Some(texture),
        },
        MockCall::BindSampler {
            unit: 3,
            sampler: Some(sampler),
        },
        MockCall::BindUniformBuffer {
            index: 4,
            buffer: Some(buffer),
            offset: 256,
            size: 256,
        },
        MockCall::BindTexture {
            unit: 3,
            target: GlTextureTarget::D2,
            texture: None,
        },
    ]));

    // Identical repeated bindings are legal; deduplication is Layer 2 work.
    api.bind_sampler(3, Some(sampler))
        .expect("repeated binding");
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::BindSampler {
            unit: 3,
            sampler: Some(_),
        })
    ));

    let trace_len = api.calls().len();
    assert!(matches!(
        api.active_texture(16),
        Err(GlError::Validation { .. })
    ));
    assert!(matches!(
        api.bind_texture(16, GlTextureTarget::D2, Some(texture)),
        Err(GlError::Validation { .. })
    ));
    assert!(matches!(
        api.bind_sampler(16, None),
        Err(GlError::Validation { .. })
    ));
    assert!(
        !api.calls()[trace_len..].iter().any(|call| matches!(
            call,
            MockCall::ActiveTexture(_)
                | MockCall::BindTexture { .. }
                | MockCall::BindSampler { .. }
        )),
        "failed binding words must not be recorded as executed calls"
    );
}

#[test]
fn mock_uniform_bindings_reject_bad_ranges_and_roles_before_recording() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let buffer = api
        .create_buffer_resource(GlBufferDesc {
            size: 512,
            usage: GlBufferUsage::UNIFORM,
        })
        .expect("uniform buffer");
    let copy_only = api
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("copy buffer");

    // Every failure below must leave the trace without a uniform binding word.
    let trace_len = api.calls().len();
    assert!(api.bind_uniform_buffer(0, Some(buffer), 3, 16).is_err());
    assert!(api.bind_uniform_buffer(0, Some(buffer), 256, 512).is_err());
    assert!(api.bind_uniform_buffer(0, Some(buffer), 1024, 0).is_err());
    assert!(api.bind_uniform_buffer(0, None, 0, 256).is_err());
    assert!(api.bind_uniform_buffer(24, Some(buffer), 0, 16).is_err());
    assert!(api.bind_uniform_buffer(0, Some(copy_only), 0, 16).is_err());
    assert!(
        !api.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::BindUniformBuffer { .. }))
    );

    api.bind_uniform_buffer(0, Some(buffer), 0, 0)
        .expect("size zero binds through the allocation end");
    api.bind_uniform_buffer(1, Some(buffer), 256, 0)
        .expect("size zero binds the tail range");
    api.bind_uniform_buffer(2, None, 0, 0)
        .expect("plain unbind");
    assert!(api.calls().ends_with(&[
        MockCall::BindUniformBuffer {
            index: 0,
            buffer: Some(buffer),
            offset: 0,
            size: 0,
        },
        MockCall::BindUniformBuffer {
            index: 1,
            buffer: Some(buffer),
            offset: 256,
            size: 0,
        },
        MockCall::BindUniformBuffer {
            index: 2,
            buffer: None,
            offset: 0,
            size: 0,
        },
    ]));
}

#[test]
fn mock_buffer_upload_and_readback_check_bounds_and_exact_lengths() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let buffer = api
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE | GlBufferUsage::COPY_DESTINATION,
        })
        .expect("buffer");
    let range = |offset, size| GlBufferRange {
        buffer,
        offset,
        size,
    };

    api.upload_buffer(range(0, 16), &[7; 16])
        .expect("subrange upload");
    let bytes = api.read_buffer(range(16, 16)).expect("readback");
    assert_eq!(bytes.len(), 16);
    assert!(api.calls().ends_with(&[
        MockCall::UploadBuffer {
            buffer,
            offset: 0,
            size: 16,
        },
        MockCall::ReadBuffer {
            buffer,
            offset: 16,
            size: 16,
        },
    ]));

    assert!(api.upload_buffer(range(0, 0), &[]).is_err());
    assert!(api.upload_buffer(range(16, 16), &[0; 15]).is_err());
    assert!(api.upload_buffer(range(48, 17), &[0; 17]).is_err());
    assert!(api.read_buffer(range(48, 17)).is_err());
    assert!(api.read_buffer(range(0, 0)).is_err());
}

#[test]
fn mock_blit_is_the_resolve_word_and_rejects_incompatible_requests() {
    let mut api =
        MockGlFamilyApi::from_discovery(snapshot_with_fact(GlFormatResourceKind::Texture, 4));
    let multisample_texture = api
        .create_texture_resource(texture_desc(4))
        .expect("multisample texture");
    let multisample_framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![texture_view(multisample_texture, 4)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("multisample framebuffer");
    let texture = api
        .create_texture_resource(texture_desc(1))
        .expect("texture");
    let framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![texture_view(texture, 1)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("framebuffer");
    let region = GlBlitRegion {
        src_offset: [0; 2],
        src_extent: [1, 1],
        dst_offset: [0; 2],
        dst_extent: [1, 1],
    };
    let color = GlBlitMask {
        color: true,
        depth: false,
        stencil: false,
    };

    // The one legal MSAA path: resolve with nearest filtering.
    assert!(
        api.blit_framebuffer(
            multisample_framebuffer,
            framebuffer,
            region,
            GlFilterMode::Linear,
            color
        )
        .is_err()
    );
    api.blit_framebuffer(
        multisample_framebuffer,
        framebuffer,
        region,
        GlFilterMode::Nearest,
        color,
    )
    .expect("resolve blit");
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::BlitFramebuffer { .. })
    ));

    assert!(
        api.blit_framebuffer(
            framebuffer,
            framebuffer,
            region,
            GlFilterMode::Nearest,
            color
        )
        .is_err(),
        "identical source and destination are rejected"
    );
    assert!(
        api.blit_framebuffer(
            framebuffer,
            multisample_framebuffer,
            region,
            GlFilterMode::Nearest,
            GlBlitMask {
                color: false,
                depth: false,
                stencil: false,
            },
        )
        .is_err(),
        "an empty mask selects no plane"
    );
    assert!(
        api.blit_framebuffer(
            framebuffer,
            multisample_framebuffer,
            GlBlitRegion {
                src_extent: [0, 1],
                ..region
            },
            GlFilterMode::Nearest,
            color,
        )
        .is_err()
    );
    let unknown = FramebufferId::new(api.context_stamp(), 9_999, 0);
    assert!(
        api.blit_framebuffer(unknown, framebuffer, region, GlFilterMode::Nearest, color)
            .is_err()
    );
}

#[test]
fn mock_renderbuffer_allocation_uses_renderbuffer_kind_facts() {
    let mut api =
        MockGlFamilyApi::from_discovery(snapshot_with_fact(GlFormatResourceKind::Renderbuffer, 4));
    let desc = GlRenderBufferDesc {
        format: GlFormat::Rgba8Unorm,
        width: 8,
        height: 8,
        samples: 4,
    };
    let renderbuffer = api.create_render_buffer(desc).expect("renderbuffer");
    api.destroy_render_buffer(renderbuffer)
        .expect("destroy renderbuffer");
    assert!(
        api.calls()
            .contains(&MockCall::CreateRenderBuffer(renderbuffer))
    );
    assert!(
        api.calls()
            .contains(&MockCall::DestroyRenderBuffer(renderbuffer))
    );

    assert!(
        api.create_render_buffer(GlRenderBufferDesc { samples: 8, ..desc })
            .is_err(),
        "sample count must stay within the discovered max_samples"
    );
    assert!(
        api.create_render_buffer(GlRenderBufferDesc { width: 0, ..desc })
            .is_err()
    );

    let mut without_facts = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let trace_len = without_facts.calls().len();
    assert!(
        without_facts.create_render_buffer(desc).is_err(),
        "a texture-only format table cannot authorize a renderbuffer"
    );
    assert!(
        !without_facts.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::CreateRenderBuffer(_)))
    );
}

#[test]
fn mock_framebuffer_draw_buffers_selection_is_validated_and_applied() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = api
        .create_texture_resource(texture_desc(1))
        .expect("texture");
    let view = texture_view(texture, 1);
    let descriptor = |color_count: usize, draw_buffers: Vec<u32>| GlFramebufferDescriptor {
        color_attachments: vec![view; color_count],
        depth_stencil_attachment: None,
        draw_buffers,
    };

    assert_eq!(
        api.create_framebuffer(&descriptor(1, vec![1])),
        Err(GlError::Validation {
            operation: "create-framebuffer",
            message: "invalid framebuffer descriptor".into(),
        })
    );
    assert!(api.create_framebuffer(&descriptor(1, vec![0, 0])).is_err());
    assert!(
        api.create_framebuffer(&descriptor(4, vec![0, 1, 2, 3, 0]))
            .is_err()
    );
    let mrt = api
        .create_framebuffer(&descriptor(2, vec![1, 0]))
        .expect("explicit MRT selection");
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::CreateFramebuffer(created)) if *created == mrt
    ));
    let stored = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![view],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("default selection stays legal");
    assert_ne!(stored, mrt);
}

fn compute_program(dialect: GlShaderDialect) -> GlProgramDescriptor {
    GlProgramDescriptor {
        kind: GlProgramKind::Compute {
            shader: GlShaderSource {
                stage: GlShaderStage::Compute,
                dialect,
                entry_point: "main".into(),
                source_hash: ShaderSourceHash([9; 32]),
                text: "void main() {}".into(),
                debug_name: None,
            },
        },
        layout: GlPipelineLayout { bindings: vec![] },
        debug_name: None,
    }
}

#[test]
fn mock_compute_program_requires_proved_capability_and_reflects_empty() {
    let mut web = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let trace_len = web.calls().len();
    assert!(matches!(
        web.create_program(&compute_program(GlShaderDialect::Embedded { version: 300 })),
        Err(GlError::Unsupported { .. })
    ));
    assert!(
        !web.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::CreateProgram(_)))
    );

    let mut desktop = MockGlFamilyApi::from_discovery(compute_storage_snapshot(false));
    let (program, reflection) = desktop
        .create_program(&compute_program(GlShaderDialect::Desktop { version: 430 }))
        .expect("proved compute capability accepts the compute kind");
    assert_eq!(
        reflection,
        GlProgramReflection {
            vertex_inputs: vec![],
            fragment_outputs: vec![],
            assignments: vec![],
        },
        "compute reflection stays empty until the reflection wave"
    );
    assert_eq!(
        desktop.calls().last(),
        Some(&MockCall::CreateProgram(program))
    );
}

#[test]
fn mock_records_render_pass_pipeline_and_draw_domains() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = api
        .create_texture_resource(mock_texture_desc())
        .expect("texture");
    let view = GlTextureView {
        target: GlAttachmentTarget::Texture(texture),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        width: 1,
        height: 1,
        sample_count: 1,
    };
    let framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![view],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("framebuffer");
    let program = api
        .create_program(&GlProgramDescriptor {
            kind: GlProgramKind::Raster {
                vertex: GlShaderSource {
                    stage: GlShaderStage::Vertex,
                    dialect: GlShaderDialect::Embedded { version: 300 },
                    entry_point: "main".into(),
                    source_hash: ShaderSourceHash([1; 32]),
                    text: "v".into(),
                    debug_name: None,
                },
                fragment: GlShaderSource {
                    stage: GlShaderStage::Fragment,
                    dialect: GlShaderDialect::Embedded { version: 300 },
                    entry_point: "main".into(),
                    source_hash: ShaderSourceHash([2; 32]),
                    text: "f".into(),
                    debug_name: None,
                },
            },
            layout: GlPipelineLayout { bindings: vec![] },
            debug_name: None,
        })
        .expect("program")
        .0;
    let vao = api
        .create_vertex_array(&GlVertexLayout {
            buffers: vec![],
            attributes: vec![],
        })
        .expect("vao");
    api.begin_render_pass(&GlRenderPassDescriptor {
        framebuffer,
        color_attachments: vec![GlColorAttachment {
            view,
            resolve_target: None,
            load: GlLoadOp::Clear,
            store: GlStoreOp::Store,
            clear: GlColorClearValue {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            },
        }],
        depth_stencil_attachment: None,
    })
    .expect("pass");
    api.set_raster_pipeline(&GlRasterPipeline {
        program,
        vertex_array: vao,
        state: GlRasterState {
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: None,
            color_targets: vec![],
            multisample: GlMultisampleState {
                sample_count: 1,
                alpha_to_coverage_enabled: false,
                sample_mask: u32::MAX,
            },
            viewport: GlViewport {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                min_depth: 0.0f32.to_bits(),
                max_depth: 1.0f32.to_bits(),
            },
            scissor: None,
            blend_constant: [0; 4],
        },
    })
    .expect("pipeline");
    api.draw_raster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex: 0,
        vertex_count: 3,
        instance_count: 1,
    }))
    .expect("draw");
    api.end_render_pass().expect("end pass");
    assert!(api.calls().ends_with(&[
        MockCall::BeginRenderPass(framebuffer),
        MockCall::SetRasterPipeline {
            program,
            vertex_array: vao
        },
        MockCall::DrawRaster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
            first_vertex: 0,
            vertex_count: 3,
            instance_count: 1
        })),
        MockCall::EndRenderPass,
    ]));
}

#[test]
fn mock_records_surface_lease_and_query_lifetimes() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let lease = match api.acquire_surface_image().expect("acquire") {
        GlSurfaceAcquire::Lease(lease) => lease,
        GlSurfaceAcquire::Suspended => panic!("active surface"),
    };
    api.present_surface(lease).expect("present");
    let query = api.create_query().expect("query");
    api.begin_occlusion_query(query).expect("begin");
    api.end_occlusion_query().expect("end");
    assert_eq!(
        api.query_result(query).expect("result"),
        GlQueryResult::Pending
    );
    api.destroy_query(query).expect("destroy");
    assert!(api.calls().ends_with(&[
        MockCall::AcquireSurface(lease),
        MockCall::PresentSurface(lease),
        MockCall::CreateQuery(query),
        MockCall::BeginOcclusion(query),
        MockCall::EndOcclusion,
        MockCall::QueryResult(query),
        MockCall::DestroyQuery(query),
    ]));
}

fn compute_storage_snapshot(storage_image: bool) -> GlDiscoverySnapshot {
    let mut builder = GlDiscoveryBuilder::new(
        stamp(ContextEpoch::INITIAL),
        context(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
        GlExtensionSet::default(),
        desktop_limits(),
        formats(storage_image),
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
    if storage_image {
        builder.resolve(
            GlCapability::StorageImage,
            CoreOrExtension {
                desktop_core: Some(GlVersion::new(4, 2)),
                embedded_core: Some(GlVersion::new(3, 1)),
                extension: Some(GlKnownExtension::ArbShaderImageLoadStore),
                extension_requires_probe: true,
            },
            GlOperationProbe::Passed,
        );
    }
    builder.build()
}

#[test]
fn mock_resource_and_copy_validation_do_not_emit_driver_commands() {
    let mut unsupported = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    assert!(
        unsupported
            .create_buffer_resource(GlBufferDesc {
                size: 64,
                usage: GlBufferUsage::STORAGE,
            })
            .is_err()
    );
    let first = unsupported
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("rejected allocation did not consume a slot");
    assert_eq!(first.slot, 0);

    let source = unsupported
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("source");
    let destination = unsupported
        .create_buffer_resource(GlBufferDesc {
            size: 64,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("destination without copy-destination usage");
    let trace_len = unsupported.calls().len();
    assert!(
        unsupported
            .copy_buffer_range(
                GlBufferRange {
                    buffer: source,
                    offset: 0,
                    size: 16,
                },
                GlBufferRange {
                    buffer: destination,
                    offset: 0,
                    size: 16,
                },
            )
            .is_err()
    );
    assert!(
        !unsupported.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::CopyBuffer { .. }))
    );
}

#[test]
fn mock_render_pass_must_match_the_stored_framebuffer_descriptor() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = api
        .create_texture_resource(mock_texture_desc())
        .expect("renderable texture");
    let view = GlTextureView {
        target: GlAttachmentTarget::Texture(texture),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        width: 1,
        height: 1,
        sample_count: 1,
    };
    let framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![view],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("framebuffer");
    let mut mismatched = view;
    mismatched.width = 2;
    let trace_len = api.calls().len();
    assert!(
        api.begin_render_pass(&GlRenderPassDescriptor {
            framebuffer,
            color_attachments: vec![GlColorAttachment {
                view: mismatched,
                resolve_target: None,
                load: GlLoadOp::Clear,
                store: GlStoreOp::Store,
                clear: GlColorClearValue {
                    red: 0,
                    green: 0,
                    blue: 0,
                    alpha: 0,
                },
            }],
            depth_stencil_attachment: None,
        })
        .is_err()
    );
    assert!(
        !api.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::BeginRenderPass(_)))
    );
    api.begin_render_pass(&GlRenderPassDescriptor {
        framebuffer,
        color_attachments: vec![GlColorAttachment {
            view,
            resolve_target: None,
            load: GlLoadOp::Clear,
            store: GlStoreOp::Store,
            clear: GlColorClearValue {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            },
        }],
        depth_stencil_attachment: None,
    })
    .expect("failed validation did not begin a pass");
}

#[test]
fn mock_storage_bindings_validate_usage_ranges_and_images_before_recording() {
    let mut inner = MockGlFamilyApi::from_discovery(compute_storage_snapshot(true));
    let non_storage = inner
        .create_buffer_resource(GlBufferDesc {
            size: 512,
            usage: GlBufferUsage::COPY_SOURCE,
        })
        .expect("buffer");
    let image = inner
        .create_texture_resource(GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 1,
                height: 1,
                depth_or_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::STORAGE_BINDING,
        })
        .expect("storage image");
    let mut api = MockComputeStorageApi::new(inner).expect("proved optional domains");
    let trace_len = api.calls().len();
    assert!(
        api.bind_storage_buffer(
            0,
            GlStorageBufferRange {
                buffer: non_storage,
                offset: 0,
                size: 256,
                usage: GlStorageBufferUsage::ReadOnly,
            },
        )
        .is_err()
    );
    assert!(
        api.bind_storage_image(
            0,
            GlStorageImageBinding {
                texture: image,
                level: 1,
                sample_count: 1,
                layered: false,
                layer: Some(0),
                format: GlFormat::Rgba8Unorm,
                access: GlStorageImageAccess::ReadWrite,
            },
        )
        .is_err()
    );
    assert!(!api.calls()[trace_len..].iter().any(|call| {
        matches!(
            call,
            MockCall::BindStorageBuffer { .. } | MockCall::BindStorageImage { .. }
        )
    }));
}
