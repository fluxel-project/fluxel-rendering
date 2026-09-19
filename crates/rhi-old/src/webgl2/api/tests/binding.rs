//! Contract tests for the binding vocabulary: which word maps to which
//! slot family, and what unit and role limits are checked before recording.

use super::*;

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
