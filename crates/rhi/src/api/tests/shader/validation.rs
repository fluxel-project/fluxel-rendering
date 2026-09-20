//! Sections 19.6-19.7: what an acceptable artifact is.
//!
//! The stage shape, canonical interface lists, binding capability, and requirement
//! collections. `use super::*` brings in
//! the fixtures; the banners below are the original section banners.

use super::*;

// ---------------------------------------------------------------------------
// Section 19.6 / 19.7: what an acceptable artifact is.
// ---------------------------------------------------------------------------

#[test]
fn a_canonical_vertex_artifact_is_accepted() {
    let artifact = artifact(ShaderStage::Vertex, vertex_interface());
    assert!(validate_shader_artifact(&artifact, permissive).is_ok());
}

#[test]
fn an_empty_entry_point_is_refused() {
    let mut artifact = artifact(ShaderStage::Vertex, vertex_interface());
    artifact.entry_point = String::new();
    assert_kind(
        validate_shader_artifact(&artifact, permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_vertex_entry_point_must_write_the_position_built_in() {
    // Section 19.6: a vertex stage that does not write the position can produce no
    // geometry at all, so the interface is refused rather than left to a backend.
    let artifact = artifact(ShaderStage::Vertex, ShaderInterface::new());
    assert_kind(
        validate_shader_artifact(&artifact, permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_compute_entry_point_may_declare_no_locations() {
    let artifact = artifact(ShaderStage::Compute, ShaderInterface::new());
    assert!(validate_shader_artifact(&artifact, permissive).is_ok());
}

#[test]
fn a_compute_entry_point_may_not_declare_stage_locations() {
    let interface = ShaderInterface::new().with_input(float32(0, 4));
    let artifact = artifact(ShaderStage::Compute, interface);
    assert_kind(
        validate_shader_artifact(&artifact, permissive),
        RhiErrorKind::InvalidUsage,
    );
}

// ---------------------------------------------------------------------------
// Section 19.6: canonical interface lists.
// ---------------------------------------------------------------------------

#[test]
fn a_resource_list_may_not_repeat_a_group_slot_pair() {
    let interface = vertex_interface()
        .with_resource(resource(0, 0))
        .with_resource(resource(0, 0));
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, interface), permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_resource_list_must_be_in_canonical_order() {
    // Section 19.6 orders resources lexicographically by `(group, slot)` and makes
    // a non-canonical list a rejection rather than something the RHI sorts: the
    // order feeds the section 19.8 canonical encoding.
    let interface = vertex_interface()
        .with_resource(resource(1, 0))
        .with_resource(resource(0, 0));
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, interface), permissive),
        RhiErrorKind::InvalidUsage,
    );
    assert!(
        validate_shader_artifact(
            &artifact(
                ShaderStage::Vertex,
                vertex_interface()
                    .with_resource(resource(0, 0))
                    .with_resource(resource(1, 0)),
            ),
            permissive,
        )
        .is_ok()
    );
}

#[test]
fn an_input_location_list_must_be_ascending_and_unique() {
    let descending = vertex_interface()
        .with_input(float32(1, 4))
        .with_input(float32(0, 4));
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, descending), permissive),
        RhiErrorKind::InvalidUsage,
    );

    let repeated = vertex_interface()
        .with_input(float32(0, 4))
        .with_input(float32(0, 4));
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, repeated), permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn an_output_location_list_must_be_ascending_and_unique() {
    let descending = vertex_interface()
        .with_output(float32(1, 4))
        .with_output(float32(0, 4));
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, descending), permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_location_width_must_be_between_one_and_four() {
    for components in [0u8, 5u8] {
        let interface = vertex_interface().with_output(float32(0, components));
        assert_kind(
            validate_shader_artifact(&artifact(ShaderStage::Vertex, interface), permissive),
            RhiErrorKind::InvalidUsage,
        );
    }
    // Four is the widest portable location and must be accepted.
    let interface = vertex_interface().with_output(float32(0, 4));
    assert!(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, interface), permissive).is_ok()
    );
}

#[test]
fn integer_inter_stage_io_must_be_flat() {
    // Section 19.6 makes this validation rather than guidance: there is no
    // interpolation between integers that every backend reproduces.
    let unmarked = vertex_interface().with_output(location(0, ShaderNumericType::Sint32, 1));
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, unmarked), permissive),
        RhiErrorKind::InvalidUsage,
    );

    let perspective = vertex_interface().with_output(ShaderLocationInterface {
        interpolation: Some(interpolation(InterpolationMode::Perspective)),
        ..location(0, ShaderNumericType::Uint32, 1)
    });
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, perspective), permissive),
        RhiErrorKind::InvalidUsage,
    );

    let flat = vertex_interface().with_output(ShaderLocationInterface {
        interpolation: Some(interpolation(InterpolationMode::Flat)),
        ..location(0, ShaderNumericType::Sint32, 2)
    });
    assert!(validate_shader_artifact(&artifact(ShaderStage::Vertex, flat), permissive).is_ok());

    // A float location needs no interpolation at all, because perspective-correct
    // interpolation is what a backend does by default.
    let float = vertex_interface().with_output(float32(0, 3));
    assert!(validate_shader_artifact(&artifact(ShaderStage::Vertex, float), permissive).is_ok());
}

// ---------------------------------------------------------------------------
// Section 19.7: binding capability and requirements.
// ---------------------------------------------------------------------------

#[test]
fn a_binding_the_device_cannot_express_refuses_the_artifact() {
    let interface = vertex_interface().with_resource(resource(0, 0));
    let artifact = artifact(ShaderStage::Vertex, interface);
    assert_kind(
        validate_shader_artifact(&artifact, refuses_bindings),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn a_buffer_binding_must_require_a_non_zero_size() {
    // Section 20.3 refuses a magic zero: zero is not a size, and a producer that
    // means "determined at bind time" must say something else.
    let interface = vertex_interface().with_resource(ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(0),
        kind: BindingKind::UniformBuffer { min_size: 0 },
        count: BindingCount::One,
    });
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, interface), permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_fixed_binding_count_of_one_is_refused_in_favour_of_one() {
    // Section 22.1's spelling rule, reached from the shader side because a shader
    // requirement and a layout entry are compared as values.
    let interface = vertex_interface().with_resource(ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(0),
        kind: BindingKind::StorageBuffer {
            access: BufferBindingAccess::ReadOnly,
            min_size: 4,
        },
        count: BindingCount::Fixed(1),
    });
    assert_kind(
        validate_shader_artifact(&artifact(ShaderStage::Vertex, interface), permissive),
        RhiErrorKind::InvalidUsage,
    );
}
