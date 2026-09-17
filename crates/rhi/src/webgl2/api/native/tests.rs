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

/// A fixture parameterised by profile version and per-class sample ceilings.
///
/// The ceiling is the rule under test, so the fixture has to be able to report
/// small ones: a fixture answering one large value everywhere cannot tell a
/// record bounded by its class from an unbounded one, and cannot show that a
/// context below the multisample storage floor records nothing.
struct SampleFactQuery {
    version: &'static str,
    samples: i64,
    color_samples: i64,
    depth_samples: i64,
    integer_samples: i64,
}

impl SampleFactQuery {
    /// Ceilings that are deliberately not in agreement: the color ceiling sits
    /// above the multisample ceiling, the depth ceiling below it, and the
    /// integer ceiling beside all three, so a bound read from the wrong fact
    /// cannot pass unnoticed. The multisample ceiling cannot go below the
    /// profile floor of four, which is why it is the color ceiling that moves.
    fn new(version: &'static str) -> Self {
        Self {
            version,
            samples: 4,
            color_samples: 8,
            depth_samples: 2,
            integer_samples: 16,
        }
    }
}

impl NativeGlQuery for SampleFactQuery {
    fn take_error(&self) -> bool {
        false
    }
    fn string(&self, name: u32) -> Option<String> {
        match name {
            glow_const::VERSION => Some(self.version.into()),
            glow_const::SHADING_LANGUAGE_VERSION => Some("4.60 test".into()),
            glow_const::VENDOR => Some("test-vendor".into()),
            glow_const::RENDERER => Some("test-renderer".into()),
            _ => None,
        }
    }
    fn integer(&self, name: u32) -> Option<i64> {
        match name {
            glow_const::NUM_EXTENSIONS => Some(0),
            glow_const::MAX_SAMPLES => Some(self.samples),
            glow_const::MAX_COLOR_TEXTURE_SAMPLES => Some(self.color_samples),
            glow_const::MAX_DEPTH_TEXTURE_SAMPLES => Some(self.depth_samples),
            glow_const::MAX_INTEGER_SAMPLES => Some(self.integer_samples),
            _ => Some(16_384),
        }
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

/// One texture descriptor with the shape every native allocation requires.
fn texture_desc(format: GlFormat, sample_count: u32, usage: GlTextureUsage) -> GlTextureDesc {
    GlTextureDesc {
        dimension: GlTextureDimension::D2,
        extent: GlExtent3d {
            width: 4,
            height: 4,
            depth_or_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        format,
        usage,
    }
}

/// The recorded multisample texture facts stop at the ceiling of the format's
/// own class, and the record and the allocation gate read one class rule.
///
/// This is the property that makes the fact table usable as a gate: the
/// presence of a fact is what an allocation is allowed to stand on, so a record
/// that ran past its class ceiling would hand the provider a sample count the
/// driver then refuses, and a record that fell short of it would refuse
/// storage the context can allocate.
#[test]
fn multisample_texture_facts_stop_at_the_ceiling_of_their_class() {
    let snapshot = discover_with(&SampleFactQuery::new("4.6 test"), &all_pass_plan())
        .expect("complete mock discovery");
    let texture_fact = |format, sample_count| {
        snapshot
            .formats()
            .get_for(GlFormatResourceKind::Texture, format, sample_count)
    };
    for sample_count in [2, 4, 8] {
        let facts = texture_fact(GlFormat::Rgba8Unorm, sample_count).expect("color fact");
        assert_eq!(facts.evidence, GlFormatEvidence::CoreGuaranteed);
        // Multisample storage is read by a resolve, never by a sampler, so the
        // record must not claim the usage no resolve performs.
        assert!(facts.renderable && facts.blendable);
        assert!(!facts.sampled && !facts.filterable);
        assert!(!facts.copy_source && !facts.copy_destination);
    }
    for sample_count in [16, 32] {
        assert!(
            texture_fact(GlFormat::Rgba8Unorm, sample_count).is_none(),
            "color ceiling is 8 in this fixture"
        );
    }
    assert!(texture_fact(GlFormat::Rgba8Srgb, 2).is_some());
    assert!(texture_fact(GlFormat::Depth32Float, 2).is_some());
    assert!(
        texture_fact(GlFormat::Depth32Float, 4).is_none(),
        "the depth class has its own, lower ceiling"
    );
    // The record reads the ceiling of the format's own class, and the depth
    // class is depth-only: a single shared ceiling would have written a depth
    // fact at the color ceiling.
    assert_eq!(
        texture_sample_class(GlFormat::Depth32Float),
        TextureSampleClass::DepthStencil
    );
    assert_eq!(
        texture_sample_class(GlFormat::Rgba8Srgb),
        TextureSampleClass::Color
    );
    assert_eq!(
        texture_sample_ceiling(TextureSampleClass::Color, &snapshot.limits()),
        8
    );
    assert_eq!(
        texture_sample_ceiling(TextureSampleClass::DepthStencil, &snapshot.limits()),
        2
    );
    // An integer format would take the third ceiling; no `GlFormat` selects it
    // today, so this pins only that the arm reads its own recorded limit instead
    // of falling back to the color one.
    assert_eq!(
        texture_sample_ceiling(TextureSampleClass::Integer, &snapshot.limits()),
        16
    );
    // The renderbuffer record is bounded by the multisample ceiling instead, so
    // the two records cannot share one bound: this fixture has multisample 4 and
    // color 8, and the renderbuffer fact stops at 4.
    assert!(
        snapshot
            .formats()
            .get_for(GlFormatResourceKind::Renderbuffer, GlFormat::Rgba8Unorm, 4)
            .is_some()
    );
    assert!(
        snapshot
            .formats()
            .get_for(GlFormatResourceKind::Renderbuffer, GlFormat::Rgba8Unorm, 8)
            .is_none()
    );
}

/// A context below the multisample storage floor records no multisample texture
/// fact and no ceiling to read one against.
///
/// The floor is the recorded decision, and this is what it costs: on a context
/// whose dispatch table has no multisample storage entry point, the sample
/// ceilings stay at their one-sample fail-closed value rather than advertising
/// a multisample allocation that cannot be made.
#[test]
fn a_context_below_the_storage_floor_records_no_multisample_texture_fact() {
    for query in [
        SampleFactQuery::new("4.2 test"),
        SampleFactQuery::new("OpenGL ES 3.0 test"),
    ] {
        let snapshot = discover_with(&query, &all_pass_plan()).expect("complete mock discovery");
        assert_eq!(
            snapshot.limits().max_color_texture_samples,
            1,
            "{}",
            query.version
        );
        for sample_count in [2, 4, 8, 16] {
            assert!(
                snapshot
                    .formats()
                    .get_for(
                        GlFormatResourceKind::Texture,
                        GlFormat::Rgba8Unorm,
                        sample_count
                    )
                    .is_none(),
                "{} at {sample_count}",
                query.version
            );
        }
    }
}

/// The allocation gate accepts exactly the multisample storage the snapshot
/// recorded, and refuses every other request with a reason.
///
/// Both halves matter: an accepted case that never reaches the driver, and a
/// rejected case that never reaches the driver, are what keep a multisample
/// allocation from being attempted and then hoped for.
#[test]
fn multisample_texture_allocation_stands_on_recorded_facts_only() {
    let snapshot = discover_with(&SampleFactQuery::new("4.6 test"), &all_pass_plan())
        .expect("complete mock discovery");
    let classify = |format, sample_count, usage| {
        texture_storage_class(
            snapshot.context().profile(),
            &snapshot.limits(),
            snapshot.formats(),
            texture_desc(format, sample_count, usage),
        )
    };
    assert_eq!(
        classify(GlFormat::Rgba8Unorm, 4, GlTextureUsage::RENDER_ATTACHMENT),
        Ok(TextureStorageClass::Multisample)
    );
    assert_eq!(
        classify(GlFormat::Rgba8Unorm, 1, GlTextureUsage::RENDER_ATTACHMENT),
        Ok(TextureStorageClass::SingleSample)
    );
    for (format, sample_count, usage, reason) in [
        // Past the color class ceiling.
        (
            GlFormat::Rgba8Unorm,
            16,
            GlTextureUsage::RENDER_ATTACHMENT,
            "format has no discovery evidence at this sample count",
        ),
        // Past the depth class ceiling, which is lower in this fixture: the
        // depth fact stops at 2 even though the color ceiling is 8, so reading
        // one class's ceiling for the other would accept this request.
        (
            GlFormat::Depth32Float,
            4,
            GlTextureUsage::RENDER_ATTACHMENT,
            "format has no discovery evidence at this sample count",
        ),
        // Inside the color class ceiling but past the multisample ceiling. This
        // is the case that shows the two bounds are not the same one: the fact
        // exists, so only the second bound can refuse it.
        (
            GlFormat::Rgba8Unorm,
            8,
            GlTextureUsage::RENDER_ATTACHMENT,
            "sample count exceeds the recorded multisample ceiling",
        ),
        // No fact exists for this format at this count at all: float color is
        // recorded only from an attachment probe, and only at one sample.
        (
            GlFormat::Rgba16Float,
            4,
            GlTextureUsage::RENDER_ATTACHMENT,
            "format has no discovery evidence at this sample count",
        ),
        // Sampled and copy usage are refused because the multisample fact does
        // not claim them: a resolve is a framebuffer blit, not a read by a
        // sampler and not a texture copy. Both counts here are ones the two
        // bounds above accept, so the reason cannot be a count in disguise.
        (
            GlFormat::Rgba8Srgb,
            2,
            GlTextureUsage::SAMPLED,
            "format is not sampled at this sample count",
        ),
        (
            GlFormat::Rgba8Unorm,
            4,
            GlTextureUsage::COPY_SOURCE,
            "format is not a copy source at this sample count",
        ),
    ] {
        assert_eq!(
            classify(format, sample_count, usage),
            Err(reason),
            "{format:?} at {sample_count}"
        );
    }
    // Single-sample allocation is gated by the same rules, so a format with no
    // recorded fact at its own count is refused before any object exists.
    let probeless = discover_with(&SampleFactQuery::new("4.6 test"), &ProbePlan::default())
        .expect("complete mock discovery");
    assert_eq!(
        texture_storage_class(
            probeless.context().profile(),
            &probeless.limits(),
            probeless.formats(),
            texture_desc(GlFormat::Rgba16Float, 1, GlTextureUsage::RENDER_ATTACHMENT),
        ),
        Err("format has no discovery evidence at this sample count")
    );
    assert_eq!(
        texture_storage_class(
            probeless.context().profile(),
            &probeless.limits(),
            probeless.formats(),
            texture_desc(GlFormat::Rgba8Srgb, 1, GlTextureUsage::SAMPLED),
        ),
        Ok(TextureStorageClass::SingleSample)
    );
    // A profile below the floor refuses the allocation even when the fact table
    // was built elsewhere: the floor is a property of the context, not of the
    // facts a snapshot happens to carry.
    assert_eq!(
        texture_storage_class(
            GlFamilyProfile::Desktop { major: 4, minor: 2 },
            &snapshot.limits(),
            snapshot.formats(),
            texture_desc(GlFormat::Rgba8Unorm, 2, GlTextureUsage::RENDER_ATTACHMENT),
        ),
        Err("this context has no multisample texture storage")
    );
}

/// An attachment is made through the target the allocation was created with.
///
/// The descriptor names a view, not the storage behind it, so attaching a
/// multisample view through the single-sample target is an error the driver
/// reports only after the framebuffer is partly populated.
#[test]
fn a_multisample_allocation_is_attached_through_its_own_target() {
    assert_eq!(
        super::exec_framebuffer::attachment_target(1),
        glow::TEXTURE_2D
    );
    assert_eq!(
        super::exec_framebuffer::attachment_target(2),
        glow::TEXTURE_2D_MULTISAMPLE
    );
    assert_eq!(
        super::exec_framebuffer::attachment_target(8),
        glow::TEXTURE_2D_MULTISAMPLE
    );
}

/// The drawable fixture: every answer `SampleFactQuery` gives, with the default
/// framebuffer's queries overridden.
///
/// The surface record is the one place where a withheld answer and an observed
/// one have to be distinguishable, so this fixture can withhold the binding
/// query entirely and can report a bound application framebuffer, which is the
/// state in which the drawable queries answer about a different object.
struct SurfaceFactQuery {
    inner: SampleFactQuery,
    binding: Option<i64>,
    bits: Option<[i64; 8]>,
}

impl SurfaceFactQuery {
    /// A complete, drawable-bound context: every surface query answers.
    fn observed() -> Self {
        Self {
            inner: SampleFactQuery::new("4.6 test"),
            binding: Some(0),
            bits: Some([8, 8, 8, 8, 24, 8, 1, 4]),
        }
    }
}

impl NativeGlQuery for SurfaceFactQuery {
    fn take_error(&self) -> bool {
        self.inner.take_error()
    }
    fn string(&self, name: u32) -> Option<String> {
        self.inner.string(name)
    }
    fn integer(&self, name: u32) -> Option<i64> {
        let index = match name {
            glow_const::DRAW_FRAMEBUFFER_BINDING => return self.binding,
            glow_const::RED_BITS => 0,
            glow_const::GREEN_BITS => 1,
            glow_const::BLUE_BITS => 2,
            glow_const::ALPHA_BITS => 3,
            glow_const::DEPTH_BITS => 4,
            glow_const::STENCIL_BITS => 5,
            glow_const::SAMPLE_BUFFERS => 6,
            glow_const::SAMPLES => 7,
            _ => return self.inner.integer(name),
        };
        self.bits.map(|bits| bits[index])
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

/// The drawable's format is recorded from the drawable, or the record says why
/// it could not be.
///
/// The property that matters is that "no surface format was observed" can never
/// be read as an observed one: with an application framebuffer bound the same
/// queries answer about that framebuffer instead of the surface, and a driver
/// that refuses the query answers nothing at all. Both of those states must
/// leave a reason, not a plausible-looking format.
#[test]
fn surface_format_is_recorded_from_the_drawable_or_marked_unavailable() {
    let flags_of = |query: &SurfaceFactQuery| {
        discover_with(query, &all_pass_plan())
            .expect("complete mock discovery")
            .context()
            .flags()
            .other
            .clone()
    };
    let observed = flags_of(&SurfaceFactQuery::observed());
    for marker in [
        "gl.surface-color-bits=8,8,8,8",
        "gl.surface-depth-bits=24",
        "gl.surface-stencil-bits=8",
        "gl.surface-sample-buffers=1",
        "gl.surface-samples=4",
    ] {
        assert!(observed.contains(marker), "{marker} in {observed:?}");
    }
    // No accepted profile exposes the drawable's color encoding, so the record
    // says so rather than guessing between linear and sRGB.
    assert!(observed.contains("gl.surface-srgb=unavailable"));
    assert!(
        !observed
            .iter()
            .any(|marker| marker.contains("facts-unavailable")),
        "{observed:?}"
    );
    for (query, reason) in [
        (
            SurfaceFactQuery {
                binding: Some(2),
                ..SurfaceFactQuery::observed()
            },
            "gl.surface-facts-unavailable=draw-framebuffer-bound",
        ),
        (
            SurfaceFactQuery {
                binding: None,
                ..SurfaceFactQuery::observed()
            },
            "gl.surface-facts-unavailable=unqueried",
        ),
    ] {
        let flags = flags_of(&query);
        assert!(flags.contains(reason), "{reason} in {flags:?}");
        // A partially observed surface format would be worse than none: a
        // presenter cannot act on half a format, and a recorded value would
        // hide which half was missing.
        assert!(
            !flags
                .iter()
                .any(|marker| marker.starts_with("gl.surface-color-bits")),
            "{flags:?}"
        );
    }
    // A failed component names itself.  The record has to say which query failed:
    // a bare "query-failed" cannot be adjudicated by a reader outside the crate,
    // who has neither the context nor the driver that produced it.  This asserts
    // every one of the eight is named, so a component added to the observation
    // without a name is a failure and not a silently broader claim.
    let flags = flags_of(&SurfaceFactQuery {
        bits: None,
        ..SurfaceFactQuery::observed()
    });
    let mut named: Vec<&str> = flags
        .iter()
        .filter_map(|marker| marker.strip_prefix("gl.surface-facts-unavailable=query-failed:"))
        .collect();
    named.sort_unstable();
    assert_eq!(
        named,
        [
            "GL_ALPHA_BITS",
            "GL_BLUE_BITS",
            "GL_DEPTH_BITS",
            "GL_GREEN_BITS",
            "GL_RED_BITS",
            "GL_SAMPLES",
            "GL_SAMPLE_BUFFERS",
            "GL_STENCIL_BITS",
        ],
        "{flags:?}"
    );
    assert!(
        !flags.contains("gl.surface-facts-unavailable=query-failed"),
        "an unattributable failure was recorded beside the attributed ones: {flags:?}"
    );
    assert!(
        !flags
            .iter()
            .any(|marker| marker.starts_with("gl.surface-color-bits")),
        "{flags:?}"
    );
}

/// The typed surface facts answer the same question the recorded keys do.
///
/// One observation produces both renderings, and this asserts the property that
/// makes the duplicate worth having: the value and the keys agree about what was
/// observed, and a width the value cannot hold is a width the observation did not
/// produce. The last case is the one that separates "narrow the number and carry
/// on" from "the observation failed", and it has to be the second: a wrapped width
/// is a format claim, and the record would then say a format was observed that the
/// drawable never had.
#[test]
fn the_typed_surface_facts_agree_with_the_recorded_keys() {
    let facts_of = |query: &SurfaceFactQuery| {
        discover_with(query, &all_pass_plan())
            .expect("complete mock discovery")
            .surface_facts()
    };
    assert_eq!(
        facts_of(&SurfaceFactQuery::observed()),
        GlSurfaceFacts::Observed {
            color_bits: [8, 8, 8, 8],
        }
    );
    for (query, case) in [
        (
            SurfaceFactQuery {
                binding: Some(2),
                ..SurfaceFactQuery::observed()
            },
            "draw framebuffer bound",
        ),
        (
            SurfaceFactQuery {
                binding: None,
                ..SurfaceFactQuery::observed()
            },
            "binding unqueried",
        ),
        (
            SurfaceFactQuery {
                bits: None,
                ..SurfaceFactQuery::observed()
            },
            "components unqueried",
        ),
    ] {
        assert_eq!(
            facts_of(&query),
            GlSurfaceFacts::Unavailable,
            "{case} must claim no format"
        );
    }
    let negative = SurfaceFactQuery {
        bits: Some([8, 8, 8, -8, 24, 8, 1, 4]),
        ..SurfaceFactQuery::observed()
    };
    let snapshot = discover_with(&negative, &all_pass_plan()).expect("complete mock discovery");
    assert_eq!(snapshot.surface_facts(), GlSurfaceFacts::Unavailable);
    let flags = snapshot.context().flags().other.clone();
    assert!(
        flags.contains("gl.surface-facts-unavailable=query-failed:GL_ALPHA_BITS"),
        "{flags:?}"
    );
    assert!(
        !flags
            .iter()
            .any(|marker| marker.starts_with("gl.surface-") && !marker.contains("unavailable")),
        "{flags:?}"
    );
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
