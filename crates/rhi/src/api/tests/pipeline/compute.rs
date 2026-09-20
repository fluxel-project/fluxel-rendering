//! Section 28: compute pipelines.
//!
//! The capability gate, stage and acceptance checks, and interface rules the
//! raster path shares. `use super::*` brings in the
//! fixtures and the vocabulary the whole chapter's tests share; the banner below
//! is the original section banner.

use super::*;
use crate::api::tests::mock::compute_pipeline_backend_for_test;
// ---------------------------------------------------------------------------
// Section 28: compute pipelines.
// ---------------------------------------------------------------------------

fn compute_descriptor() -> ComputePipelineDescriptor {
    ComputePipelineDescriptor::new(compute_module(1), no_bindings())
}

#[test]
fn a_compute_pipeline_requires_the_optional_feature() {
    let desc = compute_descriptor();
    assert!(check_compute(&desc, &permissive()).is_ok());

    let facts = permissive().without_feature(OptionalFeature::Compute);
    assert_kind(check_compute(&desc, &facts), RhiErrorKind::Unsupported);
}

#[test]
fn a_compute_pipeline_refuses_a_non_compute_entry_point() {
    let desc = ComputePipelineDescriptor::new(vertex_module(1, Vec::new()), no_bindings());
    assert_kind(
        check_compute(&desc, &permissive()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_compute_pipeline_refuses_a_shader_the_device_does_not_accept() {
    let facts = permissive().refuses_artifacts();
    assert_kind(
        check_compute(&compute_descriptor(), &facts),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn a_compute_pipeline_shares_the_interface_rules_with_raster() {
    // The same §23 interface and merge rules apply, so a compute shader requiring a
    // binding the interface does not declare is refused here too.
    let module = module_on(
        device(),
        1,
        ShaderStage::Compute,
        requirements(vec![storage_buffer(
            0,
            0,
            BufferBindingAccess::ReadWrite,
            64,
        )]),
        ShaderRequirements::new(),
    );
    let desc = ComputePipelineDescriptor::new(module, no_bindings());
    assert_kind(
        check_compute(&desc, &permissive()),
        RhiErrorKind::IncompatibleInterface,
    );

    let matching = ComputePipelineDescriptor::new(
        module_on(
            device(),
            2,
            ShaderStage::Compute,
            requirements(vec![storage_buffer(
                0,
                0,
                BufferBindingAccess::ReadWrite,
                64,
            )]),
            ShaderRequirements::new(),
        ),
        interface_of(vec![layout(vec![layout_slot(
            0,
            ShaderStages::COMPUTE,
            BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadWrite,
                min_size: 64,
            },
        )])]),
    );
    assert!(check_compute(&matching, &permissive()).is_ok());
}

#[test]
fn a_compute_pipeline_debug_prints_portable_identity_only() {
    let descriptor = compute_descriptor();
    let native = compute_pipeline_backend_for_test(descriptor.clone());
    let pipeline = ComputePipeline::new(object(52), device(), descriptor, native);
    let text = format!("{pipeline:?}");
    assert!(text.contains("ComputePipeline"), "{text}");
    assert!(text.contains("id"), "{text}");
    assert_eq!(pipeline.id(), object(52));
    assert_eq!(pipeline.interface().id(), object(40));
}
