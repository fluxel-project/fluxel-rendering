//! Section 19.8: the canonical requirement collections.
//!
//! Features and limits, each unique and sorted. `use super::*`
//! brings in the fixtures.

use super::*;

// ---------------------------------------------------------------------------
// Section 19.8: canonical requirement collections.
// ---------------------------------------------------------------------------

#[test]
fn required_features_must_be_unique_and_sorted() {
    let repeated = artifact_with(
        ShaderStage::Vertex,
        vertex_interface(),
        crate::api::shader::ShaderRequirements::new()
            .require_feature(OptionalFeature::Compute)
            .require_feature(OptionalFeature::Compute),
    );
    assert_kind(
        validate_shader_artifact(&repeated, permissive),
        RhiErrorKind::InvalidUsage,
    );

    // `Compute` is the first variant and `BindingArrays` the third, so requiring
    // them in that order is canonical and the reverse is not.
    let sorted = artifact_with(
        ShaderStage::Vertex,
        vertex_interface(),
        crate::api::shader::ShaderRequirements::new()
            .require_feature(OptionalFeature::Compute)
            .require_feature(OptionalFeature::BindingArrays),
    );
    assert!(validate_shader_artifact(&sorted, permissive).is_ok());

    let unsorted = artifact_with(
        ShaderStage::Vertex,
        vertex_interface(),
        crate::api::shader::ShaderRequirements::new()
            .require_feature(OptionalFeature::BindingArrays)
            .require_feature(OptionalFeature::Compute),
    );
    assert_kind(
        validate_shader_artifact(&unsorted, permissive),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn limit_requirements_must_be_unique_and_sorted() {
    let repeated = artifact_with(
        ShaderStage::Vertex,
        vertex_interface(),
        crate::api::shader::ShaderRequirements::new()
            .require_limit(LimitRequirement::AtLeast {
                key: LimitKey::MaxBindGroups,
                value: 4,
            })
            .require_limit(LimitRequirement::AtLeast {
                key: LimitKey::MaxBindGroups,
                value: 4,
            }),
    );
    assert_kind(
        validate_shader_artifact(&repeated, permissive),
        RhiErrorKind::InvalidUsage,
    );
}
