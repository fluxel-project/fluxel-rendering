//! Discovery tests against a mockable query: the fixtures every case shares, and
//! the cases that are about discovery itself -- the query contract, the probe
//! ledger, the profile parser, the driver identity and the desktop floor.
//!
//! Per-domain suites live in submodules and reach the fixtures through
//! `use super::*`: `samples` for the multisample texture facts and the
//! allocation gate that stands on them, `surface` for the drawable's recorded
//! format.

use super::super::{
    BufferId, ContextEpoch, ContextStamp, DeviceIdentity, GlBufferRange, GlCapability, GlError,
    GlExtent3d, GlFamilyProfile, GlFormat, GlFormatEvidence, GlFormatResourceKind, GlSurfaceFacts,
    GlTextureDesc, GlTextureDimension, GlTextureUsage, GlVersion,
};
use super::discovery::{
    DRIVER_IDENTITY_UNAVAILABLE, NativeDiscoveryError, NativeGlQuery, REQUIRED_DESKTOP_VERSION,
    TextureSampleClass, discover_with_query, discover_with_query_identified, glow_const,
    normalized_driver_identity, parse_native_profile, required_string,
    supports_multisample_texture_storage, texture_sample_ceiling, texture_sample_class,
};
use super::probes::{NativeGlProbes, ProbeAnswer};
use super::provider::{TextureStorageClass, texture_storage_class};
use super::surface::{SurfaceAcquireAttempt, surface_acquire_attempt};
use samples::SampleFactQuery;

/// The answer every fixture that does not model a drawable gives.
///
/// A fixture that never set out to describe a default framebuffer answers no
/// drawable-attachment query, and the observation reads a missing answer as a
/// refusal rather than as an absent buffer: the fixture did not say, and a
/// fixture not saying is not a fact about a drawable. `SurfaceFactQuery` is the
/// fixture that models one, and it is the only one whose surface record is
/// asserted.
pub(crate) fn no_drawable_attachment() -> Option<i64> {
    None
}

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
    fn drawable_attachment(&self, _: u32, _: u32) -> Option<i64> {
        no_drawable_attachment()
    }
}

/// Probe answers a fixture wants; every unset probe stays `Unavailable`, which
/// is the fail-closed default of the trait itself.
#[derive(Clone, Copy, Default)]
pub(crate) struct ProbePlan {
    all_pass: bool,
    depth_texture: Option<ProbeAnswer>,
    depth_renderbuffer: Option<ProbeAnswer>,
    rgba16f: Option<ProbeAnswer>,
    rgba32f: Option<ProbeAnswer>,
    storage: Option<ProbeAnswer>,
    counter_bits: Option<u32>,
}

impl ProbePlan {
    pub(crate) fn answer(plan: Option<ProbeAnswer>, all_pass: bool) -> ProbeAnswer {
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
    fn drawable_attachment(&self, _: u32, _: u32) -> Option<i64> {
        no_drawable_attachment()
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
    fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64> {
        self.inner.drawable_attachment(attachment, name)
    }
}
impl NativeGlProbes for FailingProbesQuery {}

pub(crate) fn stamp() -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
}

pub(crate) fn discover_with(
    query: &impl NativeGlQuery,
    probes: &ProbePlan,
) -> Result<super::super::GlDiscoverySnapshot, NativeDiscoveryError> {
    discover_with_query(&Combined(query, probes), stamp())
}

/// The two halves of a fixture, so one value answers both `NativeGlQuery` and
/// `NativeGlProbes` where a discovery entry point asks for a single fixture.
struct Combined<'a, Q>(&'a Q, &'a ProbePlan);
impl<Q: NativeGlQuery> NativeGlQuery for Combined<'_, Q> {
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
    fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64> {
        self.0.drawable_attachment(attachment, name)
    }
}
impl<Q: NativeGlQuery> NativeGlProbes for Combined<'_, Q> {
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
    fn renderbuffer_attachment_completes(&self, internal_format: u32, samples: u32) -> ProbeAnswer {
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
    fn issues_indirect_draw(&self, indexed: bool, stages: &[(u32, &'static str)]) -> ProbeAnswer {
        self.1.issues_indirect_draw(indexed, stages)
    }
    fn issues_indirect_dispatch(&self, compute_source: &'static str) -> ProbeAnswer {
        self.1.issues_indirect_dispatch(compute_source)
    }
}

/// The offset the indirect verbs hand GL is one convention, not two.
///
/// Both verbs describe the record relative to `range.size` and then have to
/// translate that into GL's absolute buffer offset, so identical inputs must
/// resolve to identical offsets on both paths. Pinning the pair is the whole
/// point of this test: they used to spell the sum out separately, and the
/// raster verb dropped `range.offset`, so an indirect draw read its count and
/// first-index from the start of the buffer while a dispatch read them from
/// the right place. Nothing reports that -- it is a silently wrong draw, which
/// is why it survived.
#[test]
fn both_indirect_verbs_resolve_the_same_absolute_offset() {
    let range = GlBufferRange {
        buffer: BufferId::new(stamp(), 1, 1),
        offset: 64,
        size: 48,
    };
    for command_offset in [0, 4, 16, 40] {
        let draw = super::exec_compute::draw_indirect_offset(range, command_offset)
            .expect("in-range draw record offset");
        let dispatch = super::exec_compute::dispatch_indirect_offset(range, command_offset)
            .expect("in-range dispatch record offset");
        assert_eq!(draw, dispatch, "command_offset {command_offset}");
        assert_eq!(
            draw as u64,
            range.offset + command_offset,
            "the resolved offset is the record's address in the buffer, not its \
             position inside the range"
        );
    }
}

/// An address neither verb can express fails closed on both, in the same shape
/// and with the same reason, instead of wrapping into a plausible offset that
/// still points at live bytes.
#[test]
fn both_indirect_verbs_reject_offsets_they_cannot_address() {
    let buffer = BufferId::new(stamp(), 1, 1);
    let overflowing = GlBufferRange {
        buffer,
        offset: u64::MAX - 8,
        size: 32,
    };
    let unaddressable = GlBufferRange {
        buffer,
        offset: u64::from(i32::MAX as u32) + 4,
        size: 32,
    };
    for (range, expected) in [
        (overflowing, "overflows the buffer address space"),
        (unaddressable, "exceeds GLintptr"),
    ] {
        // The operation name is the only thing allowed to differ between the
        // two refusals, so the comparison is on the reason itself.
        let reason = |error: GlError| match error {
            GlError::Validation { message, .. } => message,
            other => panic!("expected a structured validation error, got {other:?}"),
        };
        let draw = reason(super::exec_compute::draw_indirect_offset(range, 16).unwrap_err());
        let dispatch =
            reason(super::exec_compute::dispatch_indirect_offset(range, 16).unwrap_err());
        assert!(
            draw.contains(expected),
            "{draw:?} must say why the offset is unusable"
        );
        assert_eq!(draw, dispatch, "one convention, one refusal reason");
    }
}

/// The executor answers the presentation domain, and a drawable extent it has
/// not been told about suspends instead of leasing an invented size.
///
/// Native GL has no core query for the default framebuffer's extent, so the
/// extent reaches this executor only from the Host. The three states that must
/// not be confused here are a suspended drawable, an unreported extent, and a
/// reported zero-area extent: answering any of them with a lease would be a
/// wrong framebuffer size that no driver reports and no caller can see.
#[test]
fn a_drawable_extent_the_host_never_reported_leases_nothing() {
    // The trait is satisfied, which is what makes a generic presentation path
    // reach this executor at all; the audit's finding was the missing impl.
    fn requires_presentation<T: super::super::GlSurfacePresentationApi>() {}
    requires_presentation::<super::provider::NativeGlProvider<'static>>();

    let nonzero = super::super::GlSurfaceSize {
        width: 640,
        height: 480,
    };
    let zero = super::super::GlSurfaceSize {
        width: 0,
        height: 480,
    };
    assert_eq!(
        surface_acquire_attempt(false, Some(nonzero)),
        SurfaceAcquireAttempt::Lease(nonzero)
    );
    assert_eq!(
        surface_acquire_attempt(false, None),
        SurfaceAcquireAttempt::Suspended
    );
    assert_eq!(
        surface_acquire_attempt(false, Some(zero)),
        SurfaceAcquireAttempt::Suspended
    );
    // A suspended drawable stays suspended even when its extent is known: the
    // Host said it cannot present, and a lease is not the answer to that.
    assert_eq!(
        surface_acquire_attempt(true, Some(nonzero)),
        SurfaceAcquireAttempt::Suspended
    );
}

/// The driver identity is the platform's string or a recorded absence, never
/// the GL version repeated.
///
/// The GL version is already its own field, so a duplicate in the identity field
/// is indistinguishable from an observed driver string in every later report --
/// a conformance record that claims to know the driver while knowing only the
/// version, which is exactly the state audit P2-6 found.
#[test]
fn driver_identity_is_the_platform_string_or_a_recorded_absence() {
    assert_eq!(
        normalized_driver_identity("NVIDIA Corporation 537.13"),
        "NVIDIA Corporation 537.13"
    );
    for blank in ["", "   ", "\t\n"] {
        assert_eq!(
            normalized_driver_identity(blank),
            DRIVER_IDENTITY_UNAVAILABLE
        );
    }
    let observed = discover_with_query_identified(
        &Combined(&SampleFactQuery::new("4.6 test"), &all_pass_plan()),
        stamp(),
        "test-platform-driver 1.2",
    )
    .expect("complete mock discovery");
    assert_eq!(
        observed.context().driver_or_browser(),
        "test-platform-driver 1.2"
    );
    let unsupplied = discover_with_query(
        &Combined(&SampleFactQuery::new("4.6 test"), &all_pass_plan()),
        stamp(),
    )
    .expect("complete mock discovery");
    assert_eq!(
        unsupplied.context().driver_or_browser(),
        DRIVER_IDENTITY_UNAVAILABLE
    );
    // The version stays observed in its own field, so the absence of a driver
    // identity costs nothing that was actually read.
    assert_eq!(unsupplied.context().version(), "4.6 test");
    assert_ne!(
        unsupplied.context().driver_or_browser(),
        unsupplied.context().version()
    );
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

/// The desktop 4.3 floor is a requirement Fluxel places on the platform, not a
/// fact it read from the context, and the two are kept apart on purpose: the
/// parser accepts a 4.2 context so the family can describe it and let each
/// domain refuse it from recorded facts, and the floor rides the same record so
/// a report holding a 4.2 context can see what it was measured against. If the
/// parser quietly enforced the floor instead, the rejection would move into
/// discovery and every per-domain answer below it would become unreachable --
/// which is the reading audit P2-12 found missing. The marker is stamped only
/// where the requirement applies, so it never describes an embedded floor that
/// does not exist.
#[test]
fn the_desktop_context_floor_is_a_recorded_decision_not_a_parsed_rule() {
    assert_eq!(REQUIRED_DESKTOP_VERSION, GlVersion::new(4, 3));
    assert_eq!(
        parse_native_profile("4.2 test"),
        Some(GlFamilyProfile::Desktop { major: 4, minor: 2 })
    );
    // Below the floor the storage domain answers for itself rather than the
    // context being refused outright.
    assert!(!supports_multisample_texture_storage(
        GlFamilyProfile::Desktop { major: 4, minor: 2 }
    ));
    assert!(supports_multisample_texture_storage(
        GlFamilyProfile::Desktop { major: 4, minor: 3 }
    ));

    let desktop = discover_with(&SampleFactQuery::new("4.2 test"), &all_pass_plan())
        .expect("a below-floor desktop context is still discoverable");
    assert!(
        desktop
            .context()
            .flags()
            .other
            .contains("gl.desktop-context-floor=4.3")
    );
    let embedded = discover_with(
        &SampleFactQuery::new("OpenGL ES 3.2 test"),
        &all_pass_plan(),
    )
    .expect("complete mock ES discovery");
    assert!(
        !embedded
            .context()
            .flags()
            .other
            .iter()
            .any(|fact| fact.starts_with("gl.desktop-context-floor"))
    );
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
pub(crate) fn all_pass_plan() -> ProbePlan {
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
        fn drawable_attachment(&self, _: u32, _: u32) -> Option<i64> {
            no_drawable_attachment()
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

mod samples;
mod surface;
