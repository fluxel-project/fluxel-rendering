//! Section 19.8: the canonical requirement collections.
//!
//! Features, limits, and compiler options, each unique and sorted. `use super::*`
//! brings in the fixtures.

use super::*;

// ---------------------------------------------------------------------------
// Section 19.8: canonical requirement and provenance collections.
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

#[test]
fn a_compiler_option_list_must_be_sorted_and_unique() {
    // Section 19.8: the list feeds the canonical provenance encoding, so a
    // non-canonical list is a producer mistake rather than something the RHI
    // repairs.
    fn with_options(options: Vec<(String, String)>) -> ShaderArtifact {
        artifact(ShaderStage::Vertex, vertex_interface()).with_provenance(
            ShaderProvenance::PortableSource {
                language: PortableShaderLanguage::Wgsl,
                bytes: Arc::from(&b"@vertex fn main() {}"[..]),
                compiler_options: options,
            },
        )
    }

    let sorted = with_options(vec![
        ("fast_math".to_string(), "1".to_string()),
        ("strip_debug".to_string(), "false".to_string()),
    ]);
    assert!(validate_shader_artifact(&sorted, permissive).is_ok());

    let unsorted = with_options(vec![
        ("strip_debug".to_string(), "false".to_string()),
        ("fast_math".to_string(), "1".to_string()),
    ]);
    assert_kind(
        validate_shader_artifact(&unsorted, permissive),
        RhiErrorKind::InvalidUsage,
    );

    let repeated = with_options(vec![
        ("fast_math".to_string(), "1".to_string()),
        ("fast_math".to_string(), "1".to_string()),
    ]);
    assert_kind(
        validate_shader_artifact(&repeated, permissive),
        RhiErrorKind::InvalidUsage,
    );
}
