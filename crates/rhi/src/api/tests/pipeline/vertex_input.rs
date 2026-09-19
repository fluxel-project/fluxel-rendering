//! Section 24: vertex input.
//!
//! Format facts, the aggregate buffer/attribute limits, stride containment,
//! location uniqueness, and both directions of the attribute-versus-vertex-shader
//! check. The last test is the section 19.6 interpolation vocabulary these tests
//! borrow. `use super::*` brings in the fixtures and the vocabulary the whole
//! chapter's tests share; the second banner is the original banner of the
//! borrowed vocabulary test.

use super::*;
// ---------------------------------------------------------------------------
// Section 24: vertex input.
// ---------------------------------------------------------------------------

#[test]
fn vertex_formats_report_their_size_width_and_numeric_type() {
    let table = [
        (VertexFormat::Float32, 4u32, 1u8, ShaderNumericType::Float32),
        (VertexFormat::Float32x2, 8, 2, ShaderNumericType::Float32),
        (VertexFormat::Float32x3, 12, 3, ShaderNumericType::Float32),
        (VertexFormat::Float32x4, 16, 4, ShaderNumericType::Float32),
        (VertexFormat::Uint32, 4, 1, ShaderNumericType::Uint32),
        (VertexFormat::Uint32x4, 16, 4, ShaderNumericType::Uint32),
        (VertexFormat::Sint32x2, 8, 2, ShaderNumericType::Sint32),
        (VertexFormat::Unorm8x2, 2, 2, ShaderNumericType::Float32),
        (VertexFormat::Unorm8x4, 4, 4, ShaderNumericType::Float32),
    ];
    for (format, byte_size, components, numeric_type) in table {
        assert_eq!(format.byte_size(), byte_size, "{format:?} byte size");
        assert_eq!(format.components(), components, "{format:?} components");
        assert_eq!(
            format.shader_numeric_type(),
            numeric_type,
            "{format:?} numeric type"
        );
    }
    // The two normalized formats are the interesting case: their storage is
    // integral and the shader sees a float.
    assert_eq!(
        VertexFormat::Unorm8x4.shader_numeric_type(),
        ShaderNumericType::Float32
    );
}

#[test]
fn a_vertex_buffer_stride_may_not_exceed_the_device_maximum() {
    let state =
        VertexInputState::new().with_buffer(VertexBufferLayout::new(512, VertexStepMode::Vertex));
    let facts = permissive().limit(LimitKey::MaxVertexBufferArrayStride, 256);
    assert_kind(
        validate_vertex_input_state(&state, |key| (facts.device().limit)(key)),
        RhiErrorKind::InvalidUsage,
    );

    // No key exposed (the default `Facts`) means the bound is not applicable.
    assert!(validate_vertex_input_state(&state, |_| None).is_ok());
    assert!(validate_vertex_input_state(&state, |_| Some(1024)).is_ok());
}

#[test]
fn a_vertex_input_state_may_not_declare_too_many_buffers_or_attributes() {
    let state = VertexInputState::new()
        .with_buffer(VertexBufferLayout::new(16, VertexStepMode::Vertex))
        .with_buffer(VertexBufferLayout::new(16, VertexStepMode::Instance));
    let facts = permissive().limit(LimitKey::MaxVertexBuffers, 1);
    assert_kind(
        validate_vertex_input_state(&state, |key| (facts.device().limit)(key)),
        RhiErrorKind::InvalidUsage,
    );

    let one_buffer = VertexInputState::new().with_buffer(
        VertexBufferLayout::new(16, VertexStepMode::Vertex)
            .with_attribute(VertexAttribute::new(
                ShaderLocation::new(0),
                VertexFormat::Float32x4,
                0,
            ))
            .with_attribute(VertexAttribute::new(
                ShaderLocation::new(1),
                VertexFormat::Float32x4,
                0,
            )),
    );
    let facts = permissive().limit(LimitKey::MaxVertexAttributes, 1);
    assert_kind(
        validate_vertex_input_state(&one_buffer, |key| (facts.device().limit)(key)),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_vertex_attribute_must_fit_inside_its_stride() {
    let state =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x3, 8),
            ),
        );
    // 8 + 12 = 20 > 16.
    assert_kind(
        validate_vertex_input_state(&state, |_| None),
        RhiErrorKind::InvalidUsage,
    );

    let fits =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x3, 4),
            ),
        );
    assert!(validate_vertex_input_state(&fits, |_| None).is_ok());
}

#[test]
fn two_vertex_attributes_may_not_claim_one_location() {
    let state = VertexInputState::new()
        .with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x4, 0),
            ),
        )
        .with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Instance).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x4, 0),
            ),
        );
    assert_kind(
        validate_vertex_input_state(&state, |_| None),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn every_vertex_shader_input_needs_a_matching_attribute() {
    let vertex = module_on(
        device(),
        1,
        ShaderStage::Vertex,
        ShaderInterface::new()
            .with_writes_position(true)
            .with_input(float32(0, 3)),
        ShaderRequirements::new(),
    );
    let artifact_interface = &vertex.artifact().interface;

    let missing = VertexInputState::new();
    assert_kind(
        validate_vertex_input_against_interface(&missing, artifact_interface),
        RhiErrorKind::IncompatibleInterface,
    );

    let wrong_numeric_type =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Uint32x4, 0),
            ),
        );
    assert_kind(
        validate_vertex_input_against_interface(&wrong_numeric_type, artifact_interface),
        RhiErrorKind::IncompatibleInterface,
    );

    let wrong_width =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x2, 0),
            ),
        );
    assert_kind(
        validate_vertex_input_against_interface(&wrong_width, artifact_interface),
        RhiErrorKind::IncompatibleInterface,
    );

    let matching =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x3, 0),
            ),
        );
    assert!(validate_vertex_input_against_interface(&matching, artifact_interface).is_ok());
}

#[test]
fn an_attribute_the_shader_does_not_read_is_legal() {
    // Section 24.2 permits extra attributes; only the shader's own inputs must be
    // covered.
    let vertex = vertex_module(1, Vec::new());
    let state =
        VertexInputState::new().with_buffer(
            VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
                VertexAttribute::new(ShaderLocation::new(7), VertexFormat::Float32x4, 0),
            ),
        );
    assert!(validate_vertex_input_against_interface(&state, &vertex.artifact().interface).is_ok());
}

// ---------------------------------------------------------------------------
// Section 19.6 vocabulary these tests borrow.
// ---------------------------------------------------------------------------

#[test]
fn an_interpolation_value_is_comparable_across_stages() {
    // The linkage rule compares `Option<ShaderInterpolation>` for equality, so the
    // type has to be comparable and each half of it has to participate.
    let flat_at_center = ShaderInterpolation {
        mode: InterpolationMode::Flat,
        sampling: InterpolationSampling::Center,
    };
    assert_eq!(flat_at_center, flat_at_center);
    assert_ne!(
        flat_at_center,
        ShaderInterpolation {
            mode: InterpolationMode::Flat,
            sampling: InterpolationSampling::Sample,
        },
        "the sampling half participates"
    );
    assert_ne!(
        flat_at_center,
        ShaderInterpolation {
            mode: InterpolationMode::Linear,
            sampling: InterpolationSampling::Center,
        },
        "the mode half participates"
    );
}
