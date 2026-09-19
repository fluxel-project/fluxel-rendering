//! Contract tests for discovery admission: profile floors, context
//! binding, epoch identity, format-table admission and exact limit values.

use super::*;

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
