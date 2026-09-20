//! Mesh and ray-tracing pipeline validation uses the same pre-native contract as
//! raster/compute: positive, rejection, and boundary cases live here rather than
//! relying on a driver to diagnose an invalid portable descriptor.

use super::*;
use crate::api::pipeline::{
    MeshPipelineDescriptor, RayTracingHitGroup, RayTracingPipelineDescriptor, RenderTargetSignature,
};

fn mesh(id: u64, interface: ShaderInterface) -> ShaderModule {
    module_on(
        device(),
        id,
        ShaderStage::Mesh,
        interface,
        ShaderRequirements::new(),
    )
}

#[test]
fn mesh_pipeline_validates_resources_linkage_and_targets_before_native_creation() {
    let descriptor = MeshPipelineDescriptor::new(
        mesh(901, ShaderInterface::new().with_output(float32(0, 4))),
        no_bindings(),
        RenderTargetSignature {
            color_formats: vec![Some(TextureFormat::Rgba8Unorm)],
            depth_stencil_format: None,
            sample_count: 1,
        },
    )
    .with_fragment(fragment_module(
        902,
        vec![float32(0, 4)],
        vec![float32(0, 4)],
    ));
    assert!(validate_mesh_pipeline_descriptor(&descriptor, permissive().device()).is_ok());

    let missing_producer = MeshPipelineDescriptor::new(
        mesh(903, ShaderInterface::new()),
        no_bindings(),
        RenderTargetSignature {
            color_formats: vec![Some(TextureFormat::Rgba8Unorm)],
            depth_stencil_format: None,
            sample_count: 1,
        },
    )
    .with_fragment(fragment_module(
        904,
        vec![float32(0, 4)],
        vec![float32(0, 4)],
    ));
    assert_kind(
        validate_mesh_pipeline_descriptor(&missing_producer, permissive().device()),
        RhiErrorKind::IncompatibleInterface,
    );

    let unavailable = MeshPipelineDescriptor::new(
        mesh(905, ShaderInterface::new()),
        no_bindings(),
        RenderTargetSignature {
            color_formats: Vec::new(),
            depth_stencil_format: None,
            sample_count: 1,
        },
    );
    assert_kind(
        validate_mesh_pipeline_descriptor(
            &unavailable,
            permissive()
                .without_feature(OptionalFeature::MeshShader)
                .device(),
        ),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn mesh_pipeline_rejects_fragmentless_target_and_attachment_limit_boundary() {
    let fragmentless = MeshPipelineDescriptor::new(
        mesh(906, ShaderInterface::new()),
        no_bindings(),
        RenderTargetSignature {
            color_formats: vec![Some(TextureFormat::Rgba8Unorm)],
            depth_stencil_format: None,
            sample_count: 1,
        },
    );
    assert_kind(
        validate_mesh_pipeline_descriptor(&fragmentless, permissive().device()),
        RhiErrorKind::InvalidUsage,
    );

    let descriptor = MeshPipelineDescriptor::new(
        mesh(907, ShaderInterface::new()),
        no_bindings(),
        RenderTargetSignature {
            color_formats: Vec::new(),
            depth_stencil_format: None,
            sample_count: 1,
        },
    );
    assert!(
        validate_mesh_pipeline_descriptor(
            &descriptor,
            permissive()
                .limit(LimitKey::MaxColorAttachments, 0)
                .device()
        )
        .is_ok()
    );
}

fn ray(stage: ShaderStage, id: u64) -> ShaderModule {
    module_on(
        device(),
        id,
        stage,
        ShaderInterface::new(),
        ShaderRequirements::new(),
    )
}

#[test]
fn ray_pipeline_validates_all_stage_resources_and_hit_group_shape() {
    let valid =
        RayTracingPipelineDescriptor::new(ray(ShaderStage::RayGeneration, 910), no_bindings())
            .with_miss(ray(ShaderStage::Miss, 911))
            .with_hit_group(RayTracingHitGroup {
                closest_hit: Some(ray(ShaderStage::ClosestHit, 912)),
                any_hit: None,
                intersection: None,
            });
    assert!(
        validate_ray_tracing_pipeline_descriptor(
            &valid,
            device(),
            permissive()
                .limit(LimitKey::MaxRayRecursionDepth, 1)
                .device()
        )
        .is_ok()
    );

    let empty_group =
        RayTracingPipelineDescriptor::new(ray(ShaderStage::RayGeneration, 913), no_bindings())
            .with_hit_group(RayTracingHitGroup {
                closest_hit: None,
                any_hit: None,
                intersection: None,
            });
    assert_kind(
        validate_ray_tracing_pipeline_descriptor(
            &empty_group,
            device(),
            permissive()
                .limit(LimitKey::MaxRayRecursionDepth, 1)
                .device(),
        ),
        RhiErrorKind::InvalidUsage,
    );

    let rejected_artifact =
        RayTracingPipelineDescriptor::new(ray(ShaderStage::RayGeneration, 914), no_bindings());
    assert_kind(
        validate_ray_tracing_pipeline_descriptor(
            &rejected_artifact,
            device(),
            permissive()
                .limit(LimitKey::MaxRayRecursionDepth, 1)
                .refuses_artifacts()
                .device(),
        ),
        RhiErrorKind::Unsupported,
    );
}
