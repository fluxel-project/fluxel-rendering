//! Sections 19.6-19.7: what an acceptable artifact is.
//!
//! The stage shape, canonical interface lists, binding capability, and requirement
//! collections. `use super::*` brings in
//! the fixtures; the banners below are the original section banners.

use super::*;
use crate::api::shader::{PassthroughShaderProvenance, ShaderImmediateRequirement};

/// Trusted passthrough carries capture provenance, not an opaque escape hatch.
/// The unsafe marker is only meaningful when both fields identify who established
/// the exact-code/interface correspondence and how they did it.
#[test]
fn trusted_passthrough_rejects_incomplete_provenance() {
    let incomplete = unsafe {
        artifact(ShaderStage::Vertex, vertex_interface())
            .assume_trusted_passthrough(PassthroughShaderProvenance::new("", "verified reflection"))
    };
    assert_kind(
        validate_shader_artifact(&incomplete, permissive),
        RhiErrorKind::InvalidUsage,
    );

    let complete = unsafe {
        artifact(ShaderStage::Vertex, vertex_interface()).assume_trusted_passthrough(
            PassthroughShaderProvenance::new("fixture compiler", "exact-bytecode reflection"),
        )
    };
    assert!(validate_shader_artifact(&complete, permissive).is_ok());
    assert_eq!(
        complete
            .passthrough_provenance()
            .map(|value| value.producer()),
        Some("fixture compiler")
    );
}

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
fn a_compute_entry_point_with_a_non_zero_local_shape_may_declare_no_locations() {
    let artifact = artifact(ShaderStage::Compute, compute_interface());
    assert!(validate_shader_artifact(&artifact, permissive).is_ok());
}

#[test]
fn a_compute_entry_point_must_declare_its_local_shape() {
    let artifact = artifact(ShaderStage::Compute, ShaderInterface::new());
    assert_kind(
        validate_shader_artifact(&artifact, permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_compute_local_shape_requires_three_non_zero_axes() {
    for shape in [
        crate::api::shader::ComputeWorkgroupSize::new(0, 1, 1),
        crate::api::shader::ComputeWorkgroupSize::new(1, 0, 1),
        crate::api::shader::ComputeWorkgroupSize::new(1, 1, 0),
    ] {
        let artifact = artifact(
            ShaderStage::Compute,
            ShaderInterface::new().with_compute_workgroup_size(shape),
        );
        assert_kind(
            validate_shader_artifact(&artifact, permissive),
            RhiErrorKind::InvalidUsage,
        );
    }
}

#[test]
fn a_non_compute_entry_point_may_not_declare_a_compute_local_shape() {
    let artifact = artifact(
        ShaderStage::Vertex,
        vertex_interface()
            .with_compute_workgroup_size(crate::api::shader::ComputeWorkgroupSize::new(1, 1, 1)),
    );
    assert_kind(
        validate_shader_artifact(&artifact, permissive),
        RhiErrorKind::InvalidUsage,
    );
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
fn immediate_requirements_are_non_empty_ordered_and_non_overlapping() {
    let valid = vertex_interface()
        .with_immediate_requirement(ShaderImmediateRequirement::new(0, 4))
        .with_immediate_requirement(ShaderImmediateRequirement::new(4, 4));
    assert!(validate_shader_artifact(&artifact(ShaderStage::Vertex, valid), permissive).is_ok());

    for interface in [
        vertex_interface().with_immediate_requirement(ShaderImmediateRequirement::new(0, 0)),
        vertex_interface()
            .with_immediate_requirement(ShaderImmediateRequirement::new(4, 4))
            .with_immediate_requirement(ShaderImmediateRequirement::new(0, 4)),
        vertex_interface()
            .with_immediate_requirement(ShaderImmediateRequirement::new(0, 8))
            .with_immediate_requirement(ShaderImmediateRequirement::new(4, 4)),
    ] {
        assert_kind(
            validate_shader_artifact(&artifact(ShaderStage::Vertex, interface), permissive),
            RhiErrorKind::InvalidUsage,
        );
    }
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
