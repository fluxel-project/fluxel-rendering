//! Multisample texture facts and the allocation gate that stands on them.
//!
//! The fixture here reports small per-class sample ceilings on purpose: the
//! ceiling is the rule under test, so a fixture answering one large value
//! everywhere could not tell a record bounded by its class from an unbounded
//! one, nor show that a context below the storage floor records nothing.

use super::*;

/// A fixture parameterised by profile version and per-class sample ceilings.
///
/// The ceiling is the rule under test, so the fixture has to be able to report
/// small ones: a fixture answering one large value everywhere cannot tell a
/// record bounded by its class from an unbounded one, and cannot show that a
/// context below the multisample storage floor records nothing.
pub(crate) struct SampleFactQuery {
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
    pub(crate) fn new(version: &'static str) -> Self {
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
    fn drawable_attachment(&self, _: u32, _: u32) -> Option<i64> {
        no_drawable_attachment()
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
        super::super::exec_framebuffer::attachment_target(1),
        glow::TEXTURE_2D
    );
    assert_eq!(
        super::super::exec_framebuffer::attachment_target(2),
        glow::TEXTURE_2D_MULTISAMPLE
    );
    assert_eq!(
        super::super::exec_framebuffer::attachment_target(8),
        glow::TEXTURE_2D_MULTISAMPLE
    );
}
