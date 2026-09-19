//! Contract tests for compressed format evidence, which is only valid
//! while the ledger that acquired it is bound to the same profile.

use super::*;

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
