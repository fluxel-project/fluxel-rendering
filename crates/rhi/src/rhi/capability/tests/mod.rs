//! Contract tests for the capability fact database.
//!
//! These pin the two properties the rest of the RHI depends on: a fact the
//! backend did not declare answers "no", and two databases with equal facts
//! intern to the same compatibility id no matter how they were built.

use std::sync::Arc;

use super::{CapabilityDataBuilder, EnabledCapabilities};
use crate::rhi::binding::{
    BindGroupIndex, BindingCount, BindingKind, BindingSlotId, BindingSupportQuery,
};
use crate::rhi::format::{
    BufferCopyLayoutLimits, LaneWorkDomains, RouteCapabilities, RouteQuery, RouteSupport,
    SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId, TextureSupportQuery,
    TexelCopyLayoutLimits,
};
use crate::rhi::platform::{BackendKind, LimitKey, OptionalFeature};
use crate::rhi::format::TextureFormat;
use crate::rhi::resource::{BufferUsage, Extent3d, TextureDimension, TextureUsage};
use crate::rhi::shader::{
    ArtifactAcceptance, ArtifactHash, ArtifactProducerId, ArtifactProducerVersion, ShaderAbiVersion,
    ShaderArtifact, ShaderCode, ShaderInterface, ShaderRequirements, ShaderResourceRequirement,
    ShaderStage, ShaderStages,
};
use crate::rhi::binding::BufferBindingAccess;

/// The base lane every device must expose.
fn base_submission() -> SubmissionCapabilities {
    let id = SubmissionLaneId::new(0);
    SubmissionCapabilities::new(
        vec![crate::rhi::format::SubmissionLaneInfo::new(
            id,
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
        )],
        Vec::new(),
    )
}

fn builder() -> CapabilityDataBuilder {
    let mut builder = CapabilityDataBuilder::new(
        BackendKind::Vulkan,
        ShaderAbiVersion::CURRENT,
        base_submission(),
    );
    builder
        .set_limit(LimitKey::MaxBufferSize, 1 << 30)
        .declare_format(TextureFormat::Rgba8Unorm)
        .declare_code_format(&ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())));
    builder
}

#[test]
fn undeclared_facts_answer_no() {
    let capabilities = EnabledCapabilities::new(builder().build().unwrap());

    assert!(
        capabilities
            .route(&RouteQuery::BufferToBuffer)
            .capabilities()
            .is_none(),
        "an unproved route must not be supported"
    );
    assert!(
        capabilities.format(TextureFormat::Rgba16Float).is_none(),
        "an undeclared format must be unavailable even when its facts exist"
    );
    assert!(
        !capabilities
            .binding_support(&BindingSupportQuery::new(
                ShaderStages::FRAGMENT,
                BindingKind::Sampler {
                    kind: crate::rhi::binding::SamplerKind::Filtering,
                },
            ))
            .is_supported(),
        "an undeclared binding shape must not be supported"
    );
    assert_eq!(
        capabilities.binding_limit(ShaderStage::Fragment, crate::rhi::binding::BindingLimitClass::Samplers),
        None
    );
    assert!(capabilities.limit(LimitKey::MaxTexture2dDimension).is_none());
    assert!(!capabilities.supports_feature(OptionalFeature::Compute));
}

#[test]
fn equal_facts_intern_to_one_identity_regardless_of_declaration_order() {
    let mut first = CapabilityDataBuilder::new(
        BackendKind::Vulkan,
        ShaderAbiVersion::CURRENT,
        base_submission(),
    );
    first
        .set_limit(LimitKey::MaxBufferSize, 1024)
        .set_limit(LimitKey::MaxTexture2dDimension, 8192)
        .declare_format(TextureFormat::Rgba8Unorm);
    let mut second = CapabilityDataBuilder::new(
        BackendKind::Vulkan,
        ShaderAbiVersion::CURRENT,
        base_submission(),
    );
    second
        .declare_format(TextureFormat::Rgba8Unorm)
        .set_limit(LimitKey::MaxTexture2dDimension, 8192)
        .set_limit(LimitKey::MaxBufferSize, 1024);

    let left = EnabledCapabilities::new(first.build().unwrap());
    let right = EnabledCapabilities::new(second.build().unwrap());

    assert_eq!(left.compatibility_id(), right.compatibility_id());
    assert_eq!(left.fingerprint(), right.fingerprint());
}

#[test]
fn different_facts_get_different_identities() {
    let mut other = CapabilityDataBuilder::new(
        BackendKind::Vulkan,
        ShaderAbiVersion::CURRENT,
        base_submission(),
    );
    other
        .set_limit(LimitKey::MaxBufferSize, 1024)
        .set_limit(LimitKey::MaxTexture2dDimension, 8192)
        .declare_format(TextureFormat::Rgba8Unorm)
        .declare_format(TextureFormat::Rgba16Float);

    let mut left = builder();
    left.set_limit(LimitKey::MaxTexture2dDimension, 8192);
    let left = EnabledCapabilities::new(left.build().unwrap());
    let right = EnabledCapabilities::new(other.build().unwrap());

    assert_ne!(left.compatibility_id(), right.compatibility_id());
    assert_ne!(left.fingerprint(), right.fingerprint());
}

#[test]
fn a_route_declaration_carries_its_limits() {
    let mut builder = builder();
    builder.declare_route(
        RouteQuery::BufferToBuffer,
        RouteCapabilities::buffer_copy(BufferCopyLayoutLimits::new(4, 4)),
    );
    builder.declare_route(
        RouteQuery::BufferToTexture {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            aspect: crate::rhi::resource::TextureAspect::Color,
        },
        RouteCapabilities::texel_copy(TexelCopyLayoutLimits::new(256, 256)),
    );
    let capabilities = EnabledCapabilities::new(builder.build().unwrap());

    let buffer_copy = capabilities.route(&RouteQuery::BufferToBuffer);
    assert!(matches!(buffer_copy, RouteSupport::Supported(_)));
    assert_eq!(
        buffer_copy
            .capabilities()
            .and_then(|capabilities| capabilities.buffer_copy_layout())
            .map(|limits| limits.size_alignment()),
        Some(4)
    );

    // The route key includes the destination shape, so a shape that was never
    // declared must not inherit the declaration of a different one.
    assert!(
        !capabilities
            .route(&RouteQuery::BufferToTexture {
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba8Unorm,
                aspect: crate::rhi::resource::TextureAspect::Color,
            })
            .is_supported()
    );
}

#[test]
fn texture_view_compatibility_is_a_declared_group() {
    let mut builder = builder();
    builder.declare_view_compatibility_group(&[
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
    ]);
    let capabilities = EnabledCapabilities::new(builder.build().unwrap());

    assert!(capabilities.texture_view_format_compatible(
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb
    ));
    assert!(capabilities.texture_view_format_compatible(
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Rgba8Unorm
    ));
    // Equal byte size is not view compatibility.
    assert!(!capabilities.texture_view_format_compatible(
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8Sint
    ));
    assert!(!capabilities.texture_view_format_compatible(
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8Unorm
    ));
}

#[test]
fn texture_support_is_answered_per_shape() {
    let mut builder = builder();
    builder.declare_texture_support(
        TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
            1,
        ),
        crate::rhi::format::TextureSupportLimits::new(Extent3d::d3(8192, 8192, 1), 1, 1),
    );
    let capabilities = EnabledCapabilities::new(builder.build().unwrap());

    assert!(
        capabilities
            .texture_support(&TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT,
                1,
            ))
            .is_supported()
    );
    assert!(
        !capabilities
            .texture_support(&TextureSupportQuery::new(
                TextureDimension::D3,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT,
                1,
            ))
            .is_supported(),
        "a 3D color attachment must not inherit the 2D declaration"
    );
    assert!(
        !capabilities
            .texture_support(&TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT,
                8,
            ))
            .is_supported(),
        "sample_count is part of the query key"
    );
}

#[test]
fn a_database_without_a_base_lane_is_rejected() {
    let submission = SubmissionCapabilities::new(
        vec![crate::rhi::format::SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::Compute,
            LaneWorkDomains::COMPUTE,
        )],
        Vec::new(),
    );
    let mut builder = CapabilityDataBuilder::new(
        BackendKind::Vulkan,
        ShaderAbiVersion::CURRENT,
        submission,
    );
    builder.enable_feature(OptionalFeature::Compute);

    assert!(
        builder.build().is_err(),
        "a lane set without RASTER and COPY must not become a device contract"
    );
}

fn artifact(code: ShaderCode, stage: ShaderStage, requirements: ShaderRequirements) -> ShaderArtifact {
    let interface = match stage {
        ShaderStage::Vertex => ShaderInterface::new().with_writes_position(true),
        _ => ShaderInterface::new(),
    };
    ShaderArtifact::new(
        stage,
        "main",
        code,
        ShaderAbiVersion::CURRENT,
        interface,
        requirements,
        ArtifactHash([0u8; 32]),
        ArtifactProducerId("test".to_owned()),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}

#[test]
fn shader_acceptance_refuses_before_module_creation() {
    let capabilities = EnabledCapabilities::new(builder().build().unwrap());

    // The database speaks SPIR-V.
    assert_eq!(
        capabilities.shader_acceptance(&artifact(
            ShaderCode::Wgsl(Arc::from("fn main() {}")),
            ShaderStage::Vertex,
            ShaderRequirements::new(),
        )),
        ArtifactAcceptance::UnsupportedCodeFormat
    );

    // A newer ABI is not interpretable by this device.
    let mut newer = artifact(
        ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())),
        ShaderStage::Vertex,
        ShaderRequirements::new(),
    );
    newer.abi_version = ShaderAbiVersion {
        major: ShaderAbiVersion::CURRENT.major + 1,
        minor: 0,
    };
    assert_eq!(
        capabilities.shader_acceptance(&newer),
        ArtifactAcceptance::UnsupportedAbi
    );

    // A required feature that was never enabled.
    assert_eq!(
        capabilities.shader_acceptance(&artifact(
            ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())),
            ShaderStage::Vertex,
            ShaderRequirements::new().require_feature(OptionalFeature::BindingArrays),
        )),
        ArtifactAcceptance::MissingFeature
    );

    // A limit the device never reported.
    assert_eq!(
        capabilities.shader_acceptance(&artifact(
            ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())),
            ShaderStage::Vertex,
            ShaderRequirements::new().require_limit(
                crate::rhi::platform::LimitRequirement::AtLeast {
                    key: LimitKey::MaxVertexBuffers,
                    value: 8,
                },
            ),
        )),
        ArtifactAcceptance::LimitExceeded
    );

    // The accepted path.
    assert_eq!(
        capabilities.shader_acceptance(&artifact(
            ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())),
            ShaderStage::Vertex,
            ShaderRequirements::new(),
        )),
        ArtifactAcceptance::Accepted
    );
}

#[test]
fn compute_requires_the_compute_feature() {
    let artifact = artifact(
        ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())),
        ShaderStage::Compute,
        ShaderRequirements::new().with_compute_workgroup(
            crate::rhi::shader::ComputeWorkgroupRequirements::new(8, 8, 1, 64, 0),
        ),
    );

    let without = EnabledCapabilities::new(builder().build().unwrap());
    assert_eq!(
        without.shader_acceptance(&artifact),
        ArtifactAcceptance::MissingFeature
    );

    // A compute-capable database needs a lane that accepts compute work; the
    // base lane above deliberately does not.
    let mut with_compute = CapabilityDataBuilder::new(
        BackendKind::Vulkan,
        ShaderAbiVersion::CURRENT,
        SubmissionCapabilities::new(
            vec![crate::rhi::format::SubmissionLaneInfo::new(
                SubmissionLaneId::new(0),
                SubmissionLaneClass::General,
                LaneWorkDomains::RASTER
                    .union(LaneWorkDomains::COPY)
                    .union(LaneWorkDomains::COMPUTE),
            )],
            Vec::new(),
        ),
    );
    with_compute
        .declare_code_format(&ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())))
        .declare_format(TextureFormat::Rgba8Unorm)
        .enable_feature(OptionalFeature::Compute)
        .set_limit(LimitKey::MaxComputeInvocationsPerWorkgroup, 256)
        .set_limit(LimitKey::MaxComputeWorkgroupSizeX, 64)
        .set_limit(LimitKey::MaxComputeWorkgroupSizeY, 64)
        .set_limit(LimitKey::MaxComputeWorkgroupSizeZ, 64)
        .set_limit(LimitKey::MaxComputeWorkgroupStorageSize, 16_384);
    let with_compute = EnabledCapabilities::new(with_compute.build().unwrap());
    assert_eq!(with_compute.shader_acceptance(&artifact), ArtifactAcceptance::Accepted);
}

#[test]
fn an_unservable_shader_binding_is_refused() {
    let interface = ShaderInterface::new()
        .with_writes_position(true)
        .with_resource(ShaderResourceRequirement {
            group: BindGroupIndex::new(0),
            slot: BindingSlotId::new(0),
            kind: BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadWrite,
                min_size: 4,
            },
            count: BindingCount::One,
        });
    let artifact = ShaderArtifact::new(
        ShaderStage::Vertex,
        "main",
        ShaderCode::SpirV(Arc::from([0u32; 4].as_slice())),
        ShaderAbiVersion::CURRENT,
        interface,
        ShaderRequirements::new(),
        ArtifactHash([0u8; 32]),
        ArtifactProducerId("test".to_owned()),
        ArtifactProducerVersion { major: 1, minor: 0 },
    );

    let capabilities = EnabledCapabilities::new(builder().build().unwrap());
    assert_eq!(
        capabilities.shader_acceptance(&artifact),
        ArtifactAcceptance::InterfaceUnsupported
    );

    let mut builder = builder();
    builder.declare_binding_support(BindingSupportQuery::new(
        ShaderStages::VERTEX,
        BindingKind::StorageBuffer {
            access: BufferBindingAccess::ReadWrite,
            min_size: 4,
        },
    ));
    let capabilities = EnabledCapabilities::new(builder.build().unwrap());
    assert_eq!(
        capabilities.shader_acceptance(&artifact),
        ArtifactAcceptance::Accepted
    );
}

#[test]
fn buffer_support_is_answered_per_usage_set() {
    let mut builder = builder();
    builder.declare_buffer_support(
        BufferUsage::UNIFORM,
        crate::rhi::format::BufferSupportLimits::new(65_536),
    );
    let capabilities = EnabledCapabilities::new(builder.build().unwrap());

    assert!(
        capabilities
            .buffer_support(&crate::rhi::format::BufferSupportQuery::new(
                BufferUsage::UNIFORM
            ))
            .is_supported()
    );
    assert!(
        !capabilities
            .buffer_support(&crate::rhi::format::BufferSupportQuery::new(
                BufferUsage::UNIFORM.union(BufferUsage::STORAGE)
            ))
            .is_supported()
    );
}
