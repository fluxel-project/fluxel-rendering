use super::super::{
    ContextEpoch, ContextStamp, DeviceIdentity, GlCapability, GlFamilyProfile, GlFormat,
    GlFormatEvidence,
};
use super::discovery::{
    NativeDiscoveryError, NativeGlQuery, discover_with_query, glow_const, parse_native_profile,
    required_string,
};
use super::probes::{NativeGlProbes, ProbeAnswer};

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

/// Probe answers a fixture wants; every unset probe stays `Unavailable`, which
/// is the fail-closed default of the trait itself.
#[derive(Clone, Copy, Default)]
struct ProbePlan {
    all_pass: bool,
    depth_texture: Option<ProbeAnswer>,
    depth_renderbuffer: Option<ProbeAnswer>,
    rgba16f: Option<ProbeAnswer>,
    rgba32f: Option<ProbeAnswer>,
    storage: Option<ProbeAnswer>,
    counter_bits: Option<u32>,
}

impl ProbePlan {
    fn answer(plan: Option<ProbeAnswer>, all_pass: bool) -> ProbeAnswer {
        match plan {
            Some(answer) => answer,
            None if all_pass => ProbeAnswer::Passed,
            None => ProbeAnswer::Unavailable,
        }
    }
}

impl NativeGlProbes for ProbePlan {
    fn query_counter_bits(&self) -> Option<u32> {
        self.counter_bits
            .or(if self.all_pass { Some(64) } else { None })
    }
    fn attachment_completes(
        &self,
        internal_format: u32,
        _upload_format: u32,
        _upload_type: u32,
        attachment: u32,
    ) -> ProbeAnswer {
        const RGBA16F: u32 = 0x881A;
        const RGBA32F: u32 = 0x8814;
        const DEPTH_ATTACHMENT: u32 = 0x8D00;
        match (internal_format, attachment) {
            (RGBA16F, _) => ProbePlan::answer(self.rgba16f, self.all_pass),
            (RGBA32F, _) => ProbePlan::answer(self.rgba32f, self.all_pass),
            (_, DEPTH_ATTACHMENT) => ProbePlan::answer(self.depth_texture, self.all_pass),
            _ => ProbeAnswer::Unavailable,
        }
    }
    fn renderbuffer_attachment_completes(
        &self,
        internal_format: u32,
        _samples: u32,
    ) -> ProbeAnswer {
        const DEPTH_COMPONENT32F: u32 = 0x8CAC;
        if internal_format == DEPTH_COMPONENT32F {
            ProbePlan::answer(self.depth_renderbuffer, self.all_pass)
        } else {
            ProbeAnswer::Unavailable
        }
    }
    fn binds_image_load_store(&self, _source: &'static str) -> ProbeAnswer {
        ProbePlan::answer(self.storage, self.all_pass)
    }
    fn links_trivial_program(&self, _stages: &[(u32, &'static str)]) -> ProbeAnswer {
        if self.all_pass {
            ProbeAnswer::Passed
        } else {
            ProbeAnswer::Unavailable
        }
    }
    fn dispatches_compute(&self, _source: &'static str) -> ProbeAnswer {
        if self.all_pass {
            ProbeAnswer::Passed
        } else {
            ProbeAnswer::Unavailable
        }
    }
    fn binds_shader_storage(&self, _source: &'static str) -> ProbeAnswer {
        if self.all_pass {
            ProbeAnswer::Passed
        } else {
            ProbeAnswer::Unavailable
        }
    }
    fn issues_indirect_draw(&self, _indexed: bool, _stages: &[(u32, &'static str)]) -> ProbeAnswer {
        if self.all_pass {
            ProbeAnswer::Passed
        } else {
            ProbeAnswer::Unavailable
        }
    }
    fn issues_indirect_dispatch(&self, _compute_source: &'static str) -> ProbeAnswer {
        if self.all_pass {
            ProbeAnswer::Passed
        } else {
            ProbeAnswer::Unavailable
        }
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

/// A combined query + probe fixture so `discover_with_query` sees both halves.
struct FailingProbesQuery {
    inner: CompleteQuery,
}
impl NativeGlQuery for FailingProbesQuery {
    fn take_error(&self) -> bool {
        self.inner.take_error()
    }
    fn string(&self, name: u32) -> Option<String> {
        self.inner.string(name)
    }
    fn integer(&self, name: u32) -> Option<i64> {
        self.inner.integer(name)
    }
    fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
        self.inner.integer_pair(name)
    }
    fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
        self.inner.indexed_integer(name, index)
    }
    fn float(&self, name: u32) -> Option<f32> {
        self.inner.float(name)
    }
    fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
        self.inner.indexed_string(name, index)
    }
}
impl NativeGlProbes for FailingProbesQuery {}

fn stamp() -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
}

fn discover_with(
    query: &CompleteQuery,
    probes: &ProbePlan,
) -> Result<super::super::GlDiscoverySnapshot, NativeDiscoveryError> {
    struct Combined<'a>(&'a CompleteQuery, &'a ProbePlan);
    impl NativeGlQuery for Combined<'_> {
        fn take_error(&self) -> bool {
            self.0.take_error()
        }
        fn string(&self, name: u32) -> Option<String> {
            self.0.string(name)
        }
        fn integer(&self, name: u32) -> Option<i64> {
            self.0.integer(name)
        }
        fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
            self.0.integer_pair(name)
        }
        fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
            self.0.indexed_integer(name, index)
        }
        fn float(&self, name: u32) -> Option<f32> {
            self.0.float(name)
        }
        fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
            self.0.indexed_string(name, index)
        }
    }
    impl NativeGlProbes for Combined<'_> {
        fn query_counter_bits(&self) -> Option<u32> {
            self.1.query_counter_bits()
        }
        fn attachment_completes(
            &self,
            internal_format: u32,
            upload_format: u32,
            upload_type: u32,
            attachment: u32,
        ) -> ProbeAnswer {
            self.1
                .attachment_completes(internal_format, upload_format, upload_type, attachment)
        }
        fn renderbuffer_attachment_completes(
            &self,
            internal_format: u32,
            samples: u32,
        ) -> ProbeAnswer {
            self.1
                .renderbuffer_attachment_completes(internal_format, samples)
        }
        fn binds_image_load_store(&self, source: &'static str) -> ProbeAnswer {
            self.1.binds_image_load_store(source)
        }
        fn links_trivial_program(&self, stages: &[(u32, &'static str)]) -> ProbeAnswer {
            self.1.links_trivial_program(stages)
        }
        fn dispatches_compute(&self, source: &'static str) -> ProbeAnswer {
            self.1.dispatches_compute(source)
        }
        fn binds_shader_storage(&self, source: &'static str) -> ProbeAnswer {
            self.1.binds_shader_storage(source)
        }
        fn issues_indirect_draw(
            &self,
            indexed: bool,
            stages: &[(u32, &'static str)],
        ) -> ProbeAnswer {
            self.1.issues_indirect_draw(indexed, stages)
        }
        fn issues_indirect_dispatch(&self, compute_source: &'static str) -> ProbeAnswer {
            self.1.issues_indirect_dispatch(compute_source)
        }
    }
    discover_with_query(&Combined(query, probes), stamp())
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
            &FailingProbesQuery {
                inner: CompleteQuery {
                    preexisting_error: true
                }
            },
            stamp()
        ),
        Err(NativeDiscoveryError::PreExistingGlError)
    );
}

#[test]
fn failed_or_missing_probes_disable_every_optional_command_capability() {
    let snapshot = discover_with(
        &CompleteQuery {
            preexisting_error: false,
        },
        &ProbePlan::default(),
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

/// A fixture whose probes all pass, which is the shape a real glow backend
/// reports on a full-capability GL 4.3+ context.
fn all_pass_plan() -> ProbePlan {
    ProbePlan {
        all_pass: true,
        depth_texture: Some(ProbeAnswer::Passed),
        depth_renderbuffer: Some(ProbeAnswer::Passed),
        rgba16f: Some(ProbeAnswer::Passed),
        rgba32f: Some(ProbeAnswer::Passed),
        storage: Some(ProbeAnswer::Passed),
        counter_bits: Some(64),
    }
}

#[test]
fn passed_probes_enable_core_proved_capabilities_on_desktop_46() {
    let snapshot = discover_with(
        &CompleteQuery {
            preexisting_error: false,
        },
        &all_pass_plan(),
    )
    .expect("complete mock discovery");
    for capability in [
        GlCapability::Compute,
        GlCapability::StorageBuffer,
        GlCapability::StorageImage,
        GlCapability::IndirectDraw,
        GlCapability::IndirectDispatch,
        GlCapability::TimerQuery,
    ] {
        assert!(
            snapshot.capabilities().supports(capability),
            "{capability:?} must enable on desktop 4.6 with a passed probe"
        );
    }
    // Multi-draw-indirect has no glow 0.18 entry point and no portable count
    // limit, so it stays disabled even on a passing probe set.
    assert!(
        !snapshot
            .capabilities()
            .supports(GlCapability::MultiDrawIndirect)
    );
    assert_eq!(snapshot.limits().query_counter_bits, 64);
    let storage = snapshot
        .formats()
        .get(GlFormat::Rgba8Unorm, 1)
        .expect("rgba8 fact");
    assert!(storage.storage_read && storage.storage_write);
}

#[test]
fn probe_facts_record_float_and_depth_renderability_honestly() {
    // Probes ran and answered: facts carry operation evidence.
    let snapshot = discover_with(
        &CompleteQuery {
            preexisting_error: false,
        },
        &all_pass_plan(),
    )
    .expect("complete mock discovery");
    for format in [GlFormat::Rgba16Float, GlFormat::Rgba32Float] {
        let facts = snapshot
            .formats()
            .get(format, 1)
            .expect("probed float fact");
        assert_eq!(facts.evidence, GlFormatEvidence::OperationProbed);
        assert!(facts.renderable);
        assert!(facts.filterable);
        assert!(facts.blendable, "{format:?} desktop blending is core legal");
    }
    let depth = snapshot
        .formats()
        .get(GlFormat::Depth32Float, 1)
        .expect("depth fact");
    assert_eq!(depth.evidence, GlFormatEvidence::OperationProbed);
    assert!(depth.renderable);

    // Probes answered negative: renderability records `false` with the same
    // operation evidence, never silently absent.
    let rejected = ProbePlan {
        depth_texture: Some(ProbeAnswer::Failed),
        rgba16f: Some(ProbeAnswer::Failed),
        rgba32f: Some(ProbeAnswer::Failed),
        ..ProbePlan::default()
    };
    let snapshot = discover_with(
        &CompleteQuery {
            preexisting_error: false,
        },
        &rejected,
    )
    .expect("discovery with negative probes");
    let depth = snapshot
        .formats()
        .get(GlFormat::Depth32Float, 1)
        .expect("depth fact");
    assert_eq!(depth.evidence, GlFormatEvidence::OperationProbed);
    assert!(!depth.renderable, "a negative probe must record false");
    for format in [GlFormat::Rgba16Float, GlFormat::Rgba32Float] {
        let facts = snapshot
            .formats()
            .get(format, 1)
            .expect("probed float fact");
        assert_eq!(facts.evidence, GlFormatEvidence::OperationProbed);
        assert!(!facts.renderable && !facts.blendable);
        assert!(!facts.copy_source && !facts.copy_destination);
    }

    // No probe backend: the conservative core record claims no renderability
    // instead of an optimistic guarantee.
    let snapshot = discover_with(
        &CompleteQuery {
            preexisting_error: false,
        },
        &ProbePlan::default(),
    )
    .expect("probe-less discovery");
    let depth = snapshot
        .formats()
        .get(GlFormat::Depth32Float, 1)
        .expect("required depth fact");
    assert!(
        !depth.renderable,
        "no float-renderable evidence means false"
    );
}

#[test]
fn gles3_records_only_exact_core_etc2_eac_facts() {
    struct EsQuery;
    impl NativeGlQuery for EsQuery {
        fn take_error(&self) -> bool {
            false
        }
        fn string(&self, name: u32) -> Option<String> {
            match name {
                glow_const::VERSION => Some("OpenGL ES 3.2 test".into()),
                glow_const::SHADING_LANGUAGE_VERSION => Some("OpenGL ES GLSL ES 3.20".into()),
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
    impl NativeGlProbes for EsQuery {}
    let snapshot = discover_with_query(&EsQuery, stamp()).expect("ES3 discovery");
    // ES3 core guarantees ETC2/EAC texture facts and RGBA8 renderbuffers.
    assert_eq!(
        snapshot
            .formats()
            .get(GlFormat::Etc2Rgba8Unorm, 1)
            .unwrap()
            .evidence,
        GlFormatEvidence::CoreGuaranteed
    );
    assert_eq!(
        snapshot
            .formats()
            .get(GlFormat::EacRg11Snorm, 1)
            .unwrap()
            .evidence,
        GlFormatEvidence::CoreGuaranteed
    );
    assert!(snapshot.formats().get(GlFormat::Bc1RgbUnorm, 1).is_none());
    // Without probe evidence the float formats stay entirely absent: no
    // record means no render/filter claim, which is the honest state.
    assert!(snapshot.formats().get(GlFormat::Rgba16Float, 1).is_none());
    assert!(snapshot.formats().get(GlFormat::Rgba32Float, 1).is_none());
}
