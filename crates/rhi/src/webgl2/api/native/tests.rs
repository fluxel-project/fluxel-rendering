use super::super::{
    ContextEpoch, ContextStamp, DeviceIdentity, GlCapability, GlFamilyProfile, GlFormat,
    GlFormatEvidence,
};
use super::discovery::{
    NativeDiscoveryError, NativeGlQuery, baseline_formats, discover_with_query, glow_const,
    parse_native_profile, required_string,
};

struct MissingStringQuery;
impl NativeGlQuery for MissingStringQuery {
    fn take_error(&self) -> bool {
        false
    }
    fn string(&self, _: u32) -> Option<String> {
        None
    }
    fn integer(&self, _: u32) -> Option<i64> {
        None
    }
    fn integer_pair(&self, _: u32) -> Option<[i64; 2]> {
        None
    }
    fn indexed_integer(&self, _: u32, _: u32) -> Option<i64> {
        None
    }
    fn float(&self, _: u32) -> Option<f32> {
        None
    }
    fn indexed_string(&self, _: u32, _: u32) -> Option<String> {
        None
    }
}

struct CompleteQuery {
    preexisting_error: bool,
}
impl NativeGlQuery for CompleteQuery {
    fn take_error(&self) -> bool {
        self.preexisting_error
    }
    fn string(&self, name: u32) -> Option<String> {
        match name {
            glow_const::VERSION => Some("4.6 test".into()),
            glow_const::SHADING_LANGUAGE_VERSION => Some("4.60 test".into()),
            glow_const::VENDOR => Some("test-vendor".into()),
            glow_const::RENDERER => Some("test-renderer".into()),
            _ => None,
        }
    }
    fn integer(&self, name: u32) -> Option<i64> {
        Some(if name == glow_const::NUM_EXTENSIONS {
            0
        } else {
            16_384
        })
    }
    fn integer_pair(&self, _: u32) -> Option<[i64; 2]> {
        Some([16_384; 2])
    }
    fn indexed_integer(&self, _: u32, _: u32) -> Option<i64> {
        Some(16_384)
    }
    fn float(&self, _: u32) -> Option<f32> {
        Some(16.0)
    }
    fn indexed_string(&self, _: u32, _: u32) -> Option<String> {
        None
    }
}

fn stamp() -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
}

#[test]
fn profile_parser_is_strict() {
    assert_eq!(
        parse_native_profile("4.6.0 AMD"),
        Some(GlFamilyProfile::Desktop { major: 4, minor: 6 })
    );
    assert_eq!(
        parse_native_profile("OpenGL ES 3.2 Mesa"),
        Some(GlFamilyProfile::Embedded { major: 3, minor: 2 })
    );
    assert_eq!(parse_native_profile("OpenGL ES 2.0"), None);
    assert_eq!(parse_native_profile("OpenGL 3.3"), None);
}

#[test]
fn mockable_required_queries_fail_closed() {
    assert_eq!(
        required_string(&MissingStringQuery, glow_const::VERSION, "GL_VERSION"),
        Err(NativeDiscoveryError::InvalidContextString("GL_VERSION"))
    );
}

#[test]
fn preexisting_error_is_not_cleared_and_reused_for_discovery() {
    assert_eq!(
        discover_with_query(
            &CompleteQuery {
                preexisting_error: true
            },
            stamp()
        ),
        Err(NativeDiscoveryError::PreExistingGlError)
    );
}

#[test]
fn not_run_disables_every_optional_command_capability() {
    let snapshot = discover_with_query(
        &CompleteQuery {
            preexisting_error: false,
        },
        stamp(),
    )
    .expect("complete mock discovery");
    for capability in [
        GlCapability::Compute,
        GlCapability::StorageBuffer,
        GlCapability::StorageImage,
        GlCapability::IndirectDraw,
        GlCapability::IndirectDispatch,
        GlCapability::MultiDrawIndirect,
        GlCapability::TimerQuery,
    ] {
        assert!(
            !snapshot.capabilities().supports(capability),
            "{capability:?}"
        );
    }
}

#[test]
fn static_native_baseline_does_not_claim_optional_format_support() {
    let formats = baseline_formats(GlFamilyProfile::Desktop { major: 4, minor: 6 })
        .expect("profile guarantees are well-formed");
    assert!(formats.get(GlFormat::Rgba16Float, 1).is_none());
    assert!(formats.get(GlFormat::Rgba32Float, 1).is_none());
    let depth = formats
        .get(GlFormat::Depth32Float, 1)
        .expect("required depth fact");
    assert_eq!(depth.evidence, GlFormatEvidence::CoreGuaranteed);
    assert!(depth.renderable);
    assert!(!depth.filterable && !depth.blendable);
    assert!(!depth.copy_source && !depth.copy_destination);
}

#[test]
fn gles3_records_only_exact_core_etc2_eac_facts() {
    let formats = baseline_formats(GlFamilyProfile::Embedded { major: 3, minor: 0 })
        .expect("GLES3 core compressed facts");
    assert_eq!(
        formats.get(GlFormat::Etc2Rgba8Unorm, 1).unwrap().evidence,
        GlFormatEvidence::CoreGuaranteed
    );
    assert_eq!(
        formats.get(GlFormat::EacRg11Snorm, 1).unwrap().evidence,
        GlFormatEvidence::CoreGuaranteed
    );
    assert!(formats.get(GlFormat::Bc1RgbUnorm, 1).is_none());
}
