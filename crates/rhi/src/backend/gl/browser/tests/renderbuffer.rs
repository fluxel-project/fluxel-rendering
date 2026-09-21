//! Renderbuffer allocation evidence.
//!
//! The accepted case is asserted against the recorded fact table first and
//! against a real allocation second, so a failure says which half is missing;
//! every rejection this backend can produce is asserted here, including the
//! three that mean "this context has no proof" and must stay distinguishable
//! from the two that mean "this descriptor is bad".

use wasm_bindgen_test::*;

use super::*;

/// One exact renderbuffer fact, so a pure admission case can be built by hand.
fn renderbuffer_fact(
    format: GlFormat,
    sample_count: u32,
    renderable: bool,
) -> GlFormatCapabilities {
    GlFormatCapabilities {
        format,
        resource_kind: GlFormatResourceKind::Renderbuffer,
        sample_count,
        evidence: GlFormatEvidence::OperationProbed,
        sampled: false,
        filterable: false,
        renderable,
        blendable: false,
        storage_read: false,
        storage_write: false,
        copy_source: false,
        copy_destination: false,
    }
}

#[wasm_bindgen_test]
fn renderbuffer_allocates_from_a_recorded_fact_and_frees_exactly_once() {
    let mut provider = provider();
    let single = GlRenderBufferDesc {
        format: GlFormat::Rgba8Unorm,
        width: 4,
        height: 4,
        samples: 1,
    };
    // The accepted case is asserted against the recorded table first, so a
    // failure here says whether the fact or the allocation path is missing.
    assert!(
        provider
            .snapshot()
            .formats()
            .get_for(
                GlFormatResourceKind::Renderbuffer,
                single.format,
                single.samples
            )
            .is_some_and(|facts| facts.renderable),
        "context recorded no renderable RGBA8 renderbuffer fact"
    );
    let id = provider
        .create_render_buffer(single)
        .expect("allocate a single-sample renderbuffer");
    assert_eq!(id.context, provider.snapshot().context_stamp());
    let entry = provider
        .renderbuffers
        .get(&id.slot)
        .expect("renderbuffer record");
    assert_eq!(entry.generation, id.generation);
    assert_eq!(entry.desc, single);
    provider
        .destroy_render_buffer(id)
        .expect("destroy the renderbuffer");
    // A second destroy must fail rather than reach a browser call on a freed
    // object, which is why the record is the authority for liveness.
    assert!(is_validation(
        &provider.destroy_render_buffer(id).unwrap_err(),
        "destroy-render-buffer"
    ));

    // The multisampled path is exercised only when this context recorded a
    // count from the driver itself: a hard-coded count would be a guess, and a
    // guess is exactly what the fact table exists to remove.
    let multisample = provider
        .snapshot()
        .formats()
        .iter()
        .filter(|facts| {
            facts.resource_kind == GlFormatResourceKind::Renderbuffer
                && facts.format == GlFormat::Rgba8Unorm
                && facts.sample_count > 1
                && facts.renderable
        })
        .map(|facts| facts.sample_count)
        .max();
    if let Some(samples) = multisample {
        let desc = GlRenderBufferDesc { samples, ..single };
        let id = provider
            .create_render_buffer(desc)
            .expect("allocate a recorded multisample renderbuffer");
        provider.destroy_render_buffer(id).expect("destroy");
    }
}

#[wasm_bindgen_test]
fn renderbuffer_rejects_an_unrecorded_format_and_an_over_limit_extent() {
    let mut provider = provider();
    let limits = provider.snapshot().limits();
    // Depth16Unorm has no browser renderbuffer mapping in this backend, so the
    // provider must reject it without allocating rather than allocate and hope.
    let unmapped = GlRenderBufferDesc {
        format: GlFormat::Depth16Unorm,
        width: 4,
        height: 4,
        samples: 1,
    };
    let error = provider.create_render_buffer(unmapped).unwrap_err();
    assert!(
        is_unsupported(&error, "create-render-buffer"),
        "unrecorded renderbuffer format was not rejected as unsupported: {error:?}"
    );
    let too_wide = GlRenderBufferDesc {
        format: GlFormat::Rgba8Unorm,
        width: limits.max_renderbuffer_size + 1,
        height: 4,
        samples: 1,
    };
    let error = provider.create_render_buffer(too_wide).unwrap_err();
    assert!(
        is_validation(&error, "create-render-buffer"),
        "over-limit renderbuffer extent was not a validation failure: {error:?}"
    );
    // Neither rejection may leave a record behind.
    assert!(provider.renderbuffers.is_empty());
}

#[wasm_bindgen_test]
fn renderbuffer_admission_covers_every_rejection_and_the_accepted_case() {
    let provider = provider();
    let limits = provider.snapshot().limits();
    let recorded = provider.snapshot().formats().clone();
    let accepted = GlRenderBufferDesc {
        format: GlFormat::Rgba8Unorm,
        width: 4,
        height: 4,
        samples: 1,
    };
    assert_eq!(
        renderbuffer_facts::admit(&limits, &recorded, accepted),
        Ok(web_sys::WebGl2RenderingContext::RGBA8),
        "the recorded table must admit the case the provider itself recorded"
    );
    assert_eq!(
        renderbuffer_facts::admit(
            &limits,
            &recorded,
            GlRenderBufferDesc {
                width: limits.max_renderbuffer_size + 1,
                ..accepted
            }
        ),
        Err(RenderbufferRejection::ExtentExceedsLimit)
    );
    assert_eq!(
        renderbuffer_facts::admit(
            &limits,
            &recorded,
            GlRenderBufferDesc {
                height: limits.max_renderbuffer_size + 1,
                ..accepted
            }
        ),
        Err(RenderbufferRejection::ExtentExceedsLimit)
    );
    assert_eq!(
        renderbuffer_facts::admit(
            &limits,
            &recorded,
            GlRenderBufferDesc {
                samples: limits.max_samples + 1,
                ..accepted
            }
        ),
        Err(RenderbufferRejection::SampleCountExceedsLimit)
    );
    assert_eq!(
        renderbuffer_facts::admit(
            &limits,
            &recorded,
            GlRenderBufferDesc {
                format: GlFormat::Depth16Unorm,
                ..accepted
            }
        ),
        Err(RenderbufferRejection::NoFormatFact)
    );

    // A fact that exists and says "not renderable" is a different answer from a
    // fact that was never recorded, and the two must not collapse into one.
    let mut refused = GlFormatTable::default();
    refused
        .record(renderbuffer_fact(GlFormat::Rgba8Unorm, 1, false))
        .expect("record a refused fact");
    assert_eq!(
        renderbuffer_facts::admit(&limits, &refused, accepted),
        Err(RenderbufferRejection::NotRenderable)
    );

    // A renderable fact whose format this backend has no storage constant for
    // must still fail closed instead of allocating with a guessed constant.
    let mut unmapped = GlFormatTable::default();
    unmapped
        .record(renderbuffer_fact(GlFormat::Rgba16Float, 1, true))
        .expect("record an unmapped fact");
    assert_eq!(
        renderbuffer_facts::admit(
            &limits,
            &unmapped,
            GlRenderBufferDesc {
                format: GlFormat::Rgba16Float,
                ..accepted
            }
        ),
        Err(RenderbufferRejection::NoStorageMapping)
    );

    // The three evidence rejections are "this context has no proof", not caller
    // mistakes, so they stay unsupported; the two limit rejections are input
    // that violates a discovered ceiling and stay validation.
    for rejection in [
        RenderbufferRejection::NoFormatFact,
        RenderbufferRejection::NotRenderable,
        RenderbufferRejection::NoStorageMapping,
    ] {
        assert!(is_unsupported(
            &rejection.error("create-render-buffer"),
            "create-render-buffer"
        ));
    }
    for rejection in [
        RenderbufferRejection::ExtentExceedsLimit,
        RenderbufferRejection::SampleCountExceedsLimit,
    ] {
        assert!(is_validation(
            &rejection.error("create-render-buffer"),
            "create-render-buffer"
        ));
    }
}
