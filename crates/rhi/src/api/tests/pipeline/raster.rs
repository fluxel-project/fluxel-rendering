//! Section 27.3: raster pipeline validation, block by block.
//!
//! The ten validation blocks of section 27.3 in the order the specification lists
//! them, plus the portable-`Debug` check. `use super::*` brings in the fixtures and
//! the vocabulary the whole chapter's tests share; the banner below is the
//! original section banner.

use super::*;
// ---------------------------------------------------------------------------
// Section 27.3: raster pipeline validation.
// ---------------------------------------------------------------------------

#[test]
fn a_minimal_raster_pipeline_is_accepted() {
    let desc = raster_with(vertex_module(1, Vec::new()));
    assert!(check_raster(&desc, &permissive()).is_ok());
}

#[test]
fn a_raster_pipeline_refuses_a_module_from_another_device() {
    let vertex = module_on(
        other_device(),
        1,
        ShaderStage::Vertex,
        ShaderInterface::new().with_writes_position(true),
        ShaderRequirements::new(),
    );
    assert_kind(
        check_raster(&raster_with(vertex), &permissive()),
        RhiErrorKind::WrongDevice,
    );
}

#[test]
fn a_raster_pipeline_refuses_a_module_of_the_wrong_stage() {
    let wrong = module_on(
        device(),
        1,
        ShaderStage::Fragment,
        ShaderInterface::new(),
        ShaderRequirements::new(),
    );
    assert_kind(
        check_raster(&raster_with(wrong), &permissive()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn every_fragment_input_must_be_provided_by_a_vertex_output() {
    let provided = raster_with(vertex_module(1, vec![float32(0, 4)]))
        .with_fragment(fragment_module(2, vec![float32(0, 4)], Vec::new()));
    assert!(check_raster(&provided, &permissive()).is_ok());

    let missing = raster_with(vertex_module(1, vec![float32(0, 4)]))
        .with_fragment(fragment_module(2, vec![float32(1, 4)], Vec::new()));
    assert_kind(
        check_raster(&missing, &permissive()),
        RhiErrorKind::IncompatibleInterface,
    );

    let disagreeing = raster_with(vertex_module(1, vec![float32(0, 4)]))
        .with_fragment(fragment_module(2, vec![float32(0, 3)], Vec::new()));
    assert_kind(
        check_raster(&disagreeing, &permissive()),
        RhiErrorKind::IncompatibleInterface,
    );
}

#[test]
fn a_vertex_output_the_fragment_stage_does_not_read_is_legal() {
    let desc = raster_with(vertex_module(1, vec![float32(0, 4), float32(1, 2)]))
        .with_fragment(fragment_module(2, vec![float32(0, 4)], Vec::new()));
    assert!(check_raster(&desc, &permissive()).is_ok());
}

#[test]
fn a_color_target_output_must_match_the_numeric_type_of_the_format() {
    let matching = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(2, Vec::new(), vec![float32(0, 4)]))
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TARGET));
    assert!(check_raster(&matching, &permissive()).is_ok());

    let integer_output = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(
            2,
            Vec::new(),
            vec![location(0, ShaderNumericType::Uint32, 4)],
        ))
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TARGET));
    assert_kind(
        check_raster(&integer_output, &permissive()),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn a_color_target_without_a_fragment_output_must_write_nothing() {
    let writes_everything = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(2, Vec::new(), Vec::new()))
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TARGET));
    assert_kind(
        check_raster(&writes_everything, &permissive()),
        RhiErrorKind::InvalidUsage,
    );

    let writes_nothing = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(2, Vec::new(), Vec::new()))
        .with_color_target(
            ShaderLocation::new(0),
            ColorTargetState::new(TARGET).with_write_mask(ColorWriteMask::NONE),
        );
    assert!(check_raster(&writes_nothing, &permissive()).is_ok());
}

#[test]
fn a_pipeline_with_no_fragment_stage_may_not_have_a_color_target() {
    let desc = raster_with(vertex_module(1, Vec::new())).with_color_target(
        ShaderLocation::new(0),
        ColorTargetState::new(TARGET).with_write_mask(ColorWriteMask::NONE),
    );
    assert_kind(
        check_raster(&desc, &permissive()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn writing_the_fragment_depth_requires_a_depth_carrying_format() {
    let without =
        raster_with(vertex_module(1, Vec::new())).with_fragment(depth_writing_fragment(2));
    assert_kind(
        check_raster(&without, &permissive()),
        RhiErrorKind::InvalidUsage,
    );

    let with = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(depth_writing_fragment(2))
        .with_depth_stencil(DepthStencilState::new(TextureFormat::Depth32Float));
    assert!(check_raster(&with, &permissive()).is_ok());
}

#[test]
fn a_depth_or_stencil_state_must_match_the_format_aspects() {
    let depth_on_color = raster_with(vertex_module(1, Vec::new())).with_depth_stencil(
        DepthStencilState::new(TARGET).with_depth(DepthState::new(CompareFunction::Less)),
    );
    assert_kind(
        check_raster(&depth_on_color, &permissive()),
        RhiErrorKind::InvalidUsage,
    );

    let stencil_on_depth_only = raster_with(vertex_module(1, Vec::new())).with_depth_stencil(
        DepthStencilState::new(TextureFormat::Depth32Float).with_stencil(StencilState::new(
            StencilFaceState::new(CompareFunction::Always),
            StencilFaceState::new(CompareFunction::Always),
        )),
    );
    assert_kind(
        check_raster(&stencil_on_depth_only, &permissive()),
        RhiErrorKind::InvalidUsage,
    );

    let combined = raster_with(vertex_module(1, Vec::new())).with_depth_stencil(
        DepthStencilState::new(TextureFormat::Depth24PlusStencil8)
            .with_depth(DepthState::new(CompareFunction::Less).with_write_enabled(true))
            .with_stencil(StencilState::new(
                // Section 25.3 gives `StencilFaceState` only `new`, and its fields
                // are public: the operations are set by field, not by builder.
                StencilFaceState {
                    pass_op: StencilOperation::IncrementWrap,
                    ..StencilFaceState::new(CompareFunction::Always)
                },
                StencilFaceState::new(CompareFunction::Never),
            )),
    );
    assert!(check_raster(&combined, &permissive()).is_ok());
}

#[test]
fn a_target_the_device_cannot_create_is_unsupported() {
    let desc = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(2, Vec::new(), vec![float32(0, 4)]))
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TARGET));

    let facts = permissive().refuses_texture(TextureSupportQuery::new(
        TextureDimension::D2,
        TARGET,
        TextureUsage::COLOR_ATTACHMENT,
        1,
    ));
    assert_kind(check_raster(&desc, &facts), RhiErrorKind::Unsupported);

    let facts = permissive().not_a_color_attachment(TARGET);
    assert_kind(check_raster(&desc, &facts), RhiErrorKind::Unsupported);
}

#[test]
fn blending_requires_a_blendable_format() {
    let desc = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(2, Vec::new(), vec![float32(0, 4)]))
        .with_color_target(
            ShaderLocation::new(0),
            ColorTargetState::new(TARGET).with_blend(BlendState::new(
                BlendComponent::new(
                    BlendFactor::SrcAlpha,
                    BlendFactor::OneMinusSrcAlpha,
                    BlendOperation::Add,
                ),
                BlendComponent::new(
                    BlendFactor::One,
                    BlendFactor::OneMinusSrcAlpha,
                    BlendOperation::Add,
                ),
            )),
        );
    assert!(check_raster(&desc, &permissive()).is_ok());

    let facts = permissive().not_blendable(TARGET);
    assert_kind(check_raster(&desc, &facts), RhiErrorKind::Unsupported);
}

#[test]
fn the_color_attachment_count_and_bytes_are_bounded_by_the_device() {
    let desc = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(
            2,
            Vec::new(),
            vec![float32(0, 4), float32(1, 4)],
        ))
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TARGET))
        .with_color_target(ShaderLocation::new(1), ColorTargetState::new(TARGET));
    assert!(check_raster(&desc, &permissive()).is_ok());

    let facts = permissive().limit(LimitKey::MaxColorAttachments, 1);
    assert_kind(check_raster(&desc, &facts), RhiErrorKind::InvalidUsage);

    // The byte bound is the sum over the active targets' bytes per sample: two
    // `Rgba8Unorm` targets are 8 bytes, so 7 refuses and 8 fits.
    let facts = permissive().limit(LimitKey::MaxColorAttachmentBytesPerSample, 7);
    assert_kind(check_raster(&desc, &facts), RhiErrorKind::InvalidUsage);

    let facts = permissive().limit(LimitKey::MaxColorAttachmentBytesPerSample, 8);
    assert!(check_raster(&desc, &facts).is_ok());
}

#[test]
fn the_bind_groups_plus_vertex_buffers_limit_is_a_combined_bound() {
    let desc = raster_with(vertex_module(1, Vec::new())).with_vertex_input(
        VertexInputState::new().with_buffer(VertexBufferLayout::new(16, VertexStepMode::Vertex)),
    );
    let facts = permissive().limit(LimitKey::MaxBindGroupsPlusVertexBuffers, 0);
    assert_kind(check_raster(&desc, &facts), RhiErrorKind::InvalidUsage);

    let facts = permissive().limit(LimitKey::MaxBindGroupsPlusVertexBuffers, 1);
    assert!(check_raster(&desc, &facts).is_ok());
}

#[test]
fn a_strip_index_format_is_legal_only_for_a_strip_topology() {
    let stripped = raster_with(vertex_module(1, Vec::new())).with_primitive(
        PrimitiveState::new(PrimitiveTopology::TriangleStrip)
            .with_strip_index_format(IndexFormat::Uint16),
    );
    assert!(check_raster(&stripped, &permissive()).is_ok());

    let list = raster_with(vertex_module(1, Vec::new())).with_primitive(
        PrimitiveState::new(PrimitiveTopology::TriangleList)
            .with_strip_index_format(IndexFormat::Uint16),
    );
    assert_kind(
        check_raster(&list, &permissive()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn depth_bias_belongs_to_triangle_topologies_and_must_be_finite() {
    let triangles = raster_with(vertex_module(1, Vec::new())).with_primitive(
        PrimitiveState::new(PrimitiveTopology::TriangleList)
            .with_depth_bias(DepthBiasState::new(1, 1.0)),
    );
    assert!(check_raster(&triangles, &permissive()).is_ok());

    let lines = raster_with(vertex_module(1, Vec::new())).with_primitive(
        PrimitiveState::new(PrimitiveTopology::LineList)
            .with_depth_bias(DepthBiasState::new(1, 1.0)),
    );
    assert_kind(
        check_raster(&lines, &permissive()),
        RhiErrorKind::InvalidUsage,
    );

    let infinite = raster_with(vertex_module(1, Vec::new())).with_primitive(
        PrimitiveState::new(PrimitiveTopology::TriangleList)
            .with_depth_bias(DepthBiasState::new(0, f32::NAN)),
    );
    assert_kind(
        check_raster(&infinite, &permissive()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn alpha_to_coverage_requires_multisampling_a_float4_output_and_a_written_mask() {
    let base = || {
        raster_with(vertex_module(1, Vec::new()))
            .with_fragment(fragment_module(2, Vec::new(), vec![float32(0, 4)]))
            .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TARGET))
    };

    let single_sample =
        base().with_multisample(MultisampleState::new(1).with_alpha_to_coverage(true));
    assert_kind(
        check_raster(&single_sample, &permissive()),
        RhiErrorKind::InvalidUsage,
    );

    let multisampled =
        base().with_multisample(MultisampleState::new(4).with_alpha_to_coverage(true));
    assert!(check_raster(&multisampled, &permissive()).is_ok());

    let narrow_output = raster_with(vertex_module(1, Vec::new()))
        .with_fragment(fragment_module(2, Vec::new(), vec![float32(0, 3)]))
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(TARGET))
        .with_multisample(MultisampleState::new(4).with_alpha_to_coverage(true));
    assert_kind(
        check_raster(&narrow_output, &permissive()),
        RhiErrorKind::InvalidUsage,
    );

    let facts = permissive().without_alpha(TARGET);
    assert_kind(
        check_raster(&multisampled, &facts),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_raster_pipeline_debug_prints_portable_identity_only() {
    let pipeline = RasterPipeline::new(
        object(51),
        device(),
        raster_with(vertex_module(1, Vec::new())),
    );
    let text = format!("{pipeline:?}");
    assert!(text.contains("RasterPipeline"), "{text}");
    assert!(text.contains("id"), "{text}");
}
