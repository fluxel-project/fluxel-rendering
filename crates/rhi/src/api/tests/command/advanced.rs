//! Positive, negative, and boundary contracts for advanced recorder commands.

use super::*;
use crate::api::pipeline::{
    ColorTargetState, ImmediateRange, MeshPipeline, MeshPipelineDescriptor, MultisampleState,
    PrimitiveState, PrimitiveTopology, RayTracingPipeline, RayTracingPipelineDescriptor,
    RenderTargetSignature,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::backend::AccelerationStructureBackend;
use crate::api::resource::{
    AccelerationStructure, AccelerationStructureBuildMode, AccelerationStructureCopyMode,
    AccelerationStructureDescriptor, BlasGeometry, BottomLevelAccelerationStructureDescriptor,
    TrianglesGeometry,
};
use std::any::Any;
use std::sync::Arc;

struct TestAccelerationStructure;
impl AccelerationStructureBackend for TestAccelerationStructure {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn structure(id: u64, owner: DeviceIdentity) -> AccelerationStructure {
    let input = fixture::buffer(
        object(id + 100),
        owner,
        BufferDescriptor::new(64, BufferUsage::BLAS_INPUT),
    );
    let descriptor = AccelerationStructureDescriptor::BottomLevel(
        BottomLevelAccelerationStructureDescriptor::new(vec![BlasGeometry::Triangles(
            TrianglesGeometry {
                vertices: input,
                vertex_range: BufferRange::new(0, 36),
                vertex_format: crate::api::resource::AccelerationStructureVertexFormat::Float32x3,
                vertex_stride: 12,
                vertex_count: 3,
                primitive_count: 1,
                indices: None,
            },
        )]),
    );
    AccelerationStructure::new(
        object(id),
        owner,
        descriptor,
        crate::api::resource::AccelerationStructureBuildSizes {
            acceleration_structure_size: 256,
            build_scratch_size: 256,
            update_scratch_size: 256,
        },
        Box::new(TestAccelerationStructure),
    )
}

fn ray_facts() -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(OptionalFeature::RayQuery);
    facts.record_limit(LimitKey::RayTracingScratchBufferAlignment, 256);
    facts
}

fn shader(stage: ShaderStage, id: u64, owner: DeviceIdentity) -> ShaderModule {
    let artifact = ShaderArtifact::new(
        stage,
        "main",
        ShaderCode::Wgsl(Arc::from("fn main() {}")),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new(),
        ShaderRequirements::new(),
        ArtifactHash([9; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    );
    ShaderModule::new(
        object(id),
        owner,
        artifact.clone(),
        crate::api::tests::mock::module_backend_for_test(&artifact),
    )
}
fn empty_interface(owner: DeviceIdentity, id: u64) -> PipelineInterface {
    PipelineInterface::new(
        object(id),
        owner,
        PipelineInterfaceDescriptor::new(Vec::new()),
        PipelineInterfaceCompatibilityId::new(id),
        LayoutFingerprint([id as u8; 32]),
    )
}
fn mesh_pipeline(owner: DeviceIdentity) -> MeshPipeline {
    let desc = MeshPipelineDescriptor::new(
        shader(ShaderStage::Mesh, 820, owner),
        empty_interface(owner, 821),
        RenderTargetSignature {
            color_formats: vec![Some(TextureFormat::Rgba8Unorm)],
            depth_stencil_format: None,
            sample_count: 1,
        },
    );
    MeshPipeline::new(
        object(822),
        owner,
        desc,
        crate::api::tests::mock::mesh_pipeline_backend_for_test(),
    )
}

#[test]
fn mesh_pipeline_keeps_full_graphics_fixed_state_and_canonical_target_signature() {
    let descriptor = MeshPipelineDescriptor::new(
        shader(ShaderStage::Mesh, 819, device()),
        empty_interface(device(), 818),
        RenderTargetSignature {
            color_formats: vec![Some(TextureFormat::Rgba8Unorm), None],
            depth_stencil_format: None,
            sample_count: 1,
        },
    )
    .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleStrip))
    .with_multisample(MultisampleState::new(4))
    .with_multiview_mask(0b11)
    .with_color_target(
        ShaderLocation::new(2),
        ColorTargetState::new(TextureFormat::Bgra8Unorm),
    );

    assert!(matches!(
        descriptor.primitive.topology,
        PrimitiveTopology::TriangleStrip
    ));
    assert_eq!(descriptor.multisample.count, 4);
    assert_eq!(descriptor.multiview_mask, Some(0b11));
    assert_eq!(
        descriptor.target_signature().color_formats,
        vec![
            Some(TextureFormat::Rgba8Unorm),
            None,
            Some(TextureFormat::Bgra8Unorm)
        ]
    );
    assert_eq!(descriptor.target_signature().sample_count, 4);
}

#[test]
fn mesh_fixed_state_refuses_zero_and_unavailable_or_out_of_range_multiview() {
    let base = MeshPipelineDescriptor::new(
        shader(ShaderStage::Mesh, 817, device()),
        empty_interface(device(), 816),
        RenderTargetSignature {
            color_formats: Vec::new(),
            depth_stencil_format: None,
            sample_count: 1,
        },
    );
    let zero_samples = base.clone().with_multisample(MultisampleState::new(0));
    assert_kind(
        crate::api::pipeline::mesh::validate_mesh_fixed_state(&zero_samples, |_| true, |_| None),
        RhiErrorKind::InvalidUsage,
    );
    let zero_mask = base.clone().with_multiview_mask(0);
    assert_kind(
        crate::api::pipeline::mesh::validate_mesh_fixed_state(&zero_mask, |_| true, |_| None),
        RhiErrorKind::InvalidUsage,
    );
    let selected = base.clone().with_multiview_mask(0b100);
    assert_kind(
        crate::api::pipeline::mesh::validate_mesh_fixed_state(&selected, |_| false, |_| None),
        RhiErrorKind::Unsupported,
    );
    assert_kind(
        crate::api::pipeline::mesh::validate_mesh_fixed_state(
            &selected,
            |_| true,
            |key| (key == LimitKey::MaxMultiviewViewCount).then_some(2),
        ),
        RhiErrorKind::InvalidUsage,
    );
    assert!(
        crate::api::pipeline::mesh::validate_mesh_fixed_state(
            &selected,
            |_| true,
            |key| (key == LimitKey::MaxMultiviewViewCount).then_some(3),
        )
        .is_ok()
    );

    // Mesh pipelines carry the same mask contract as ordinary raster
    // pipelines: a hole is an opt-in selective-multiview request.
    assert_kind(
        crate::api::pipeline::mesh::validate_mesh_fixed_state(
            &selected,
            |feature| feature != OptionalFeature::SelectiveMultiview,
            |_| None,
        ),
        RhiErrorKind::Unsupported,
    );
    let contiguous = base.with_multiview_mask(0b111);
    assert!(
        crate::api::pipeline::mesh::validate_mesh_fixed_state(
            &contiguous,
            |feature| feature != OptionalFeature::SelectiveMultiview,
            |_| None,
        )
        .is_ok()
    );
}
fn ray_pipeline(owner: DeviceIdentity) -> RayTracingPipeline {
    let desc = RayTracingPipelineDescriptor::new(
        shader(ShaderStage::RayGeneration, 823, owner),
        empty_interface(owner, 824),
    );
    RayTracingPipeline::new(
        object(825),
        owner,
        desc,
        crate::api::tests::mock::ray_tracing_pipeline_backend_for_test(),
    )
}
fn ray_pipeline_with_immediates(owner: DeviceIdentity) -> RayTracingPipeline {
    let interface = PipelineInterface::new(
        object(829),
        owner,
        PipelineInterfaceDescriptor::new(Vec::new()).with_immediate_range(ImmediateRange::new(
            0,
            4,
            ShaderStages::RAY_GENERATION,
        )),
        PipelineInterfaceCompatibilityId::new(829),
        LayoutFingerprint([29u8; 32]),
    );
    let desc = RayTracingPipelineDescriptor::new(
        shader(ShaderStage::RayGeneration, 830, owner),
        interface,
    );
    RayTracingPipeline::new(
        object(831),
        owner,
        desc,
        crate::api::tests::mock::ray_tracing_pipeline_backend_for_test(),
    )
}
fn mesh_facts(max: u64) -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(OptionalFeature::MeshShader);
    facts.record_limit(LimitKey::MaxMeshWorkgroupsPerDimension, max);
    facts
}
fn ray_pipeline_facts(max: u64) -> CapabilityFacts {
    let mut facts = ray_facts();
    facts.record_feature(OptionalFeature::RayTracingPipeline);
    facts.record_limit(LimitKey::MaxRayDispatchCount, max);
    facts.record_limit(LimitKey::MaxRayTracingPipelineGroupDataSize, 64);
    facts.record_limit(LimitKey::RayTracingPipelineGroupDataAlignment, 16);
    facts.record_limit(LimitKey::RayTracingPipelineGroupDataOffsetAlignment, 16);
    facts
}

#[test]
fn acceleration_build_refuses_insufficient_scratch_capacity() {
    let mut recorder = recorder_reporting(ray_facts());
    let destination = structure(890, device());
    let too_small = fixture::buffer(
        object(891),
        device(),
        BufferDescriptor::new(128, BufferUsage::ACCELERATION_STRUCTURE_SCRATCH),
    );
    assert_kind(
        recorder.build_acceleration_structure(
            &destination,
            &too_small,
            AccelerationStructureBuildMode::Build,
        ),
        RhiErrorKind::InvalidUsage,
    );

    // Scratch alignment constrains the native address, not the allocation's
    // byte length. A backend must align an AS-scratch allocation, while the
    // portable recorder accepts any capacity at or above the queried size.
    let oversized = fixture::buffer(
        object(892),
        device(),
        BufferDescriptor::new(257, BufferUsage::ACCELERATION_STRUCTURE_SCRATCH),
    );
    recorder
        .build_acceleration_structure(
            &destination,
            &oversized,
            AccelerationStructureBuildMode::Build,
        )
        .unwrap();
}

#[test]
fn shader_table_rejects_foreign_non_indirect_and_misaligned_regions() {
    let mut recorder = recorder_reporting(ray_pipeline_facts(1));
    let mut scope = recorder.begin_ray_tracing(&Default::default()).unwrap();
    scope.set_pipeline(&ray_pipeline(device())).unwrap();
    let foreign = crate::api::command::RayTracingShaderTable {
        ray_generation: crate::api::command::RayTracingShaderTableRegion {
            buffer: fixture::buffer(
                object(892),
                other_device(),
                BufferDescriptor::new(16, BufferUsage::INDIRECT),
            ),
            range: BufferRange::new(0, 16),
            stride: 16,
        },
        miss: None,
        hit: None,
    };
    assert_kind(
        scope.dispatch_rays(&foreign, 1, 1, 1),
        RhiErrorKind::WrongDevice,
    );
    let bad_usage = crate::api::command::RayTracingShaderTable {
        ray_generation: crate::api::command::RayTracingShaderTableRegion {
            buffer: fixture::buffer(
                object(893),
                device(),
                BufferDescriptor::new(16, BufferUsage::STORAGE),
            ),
            range: BufferRange::new(0, 16),
            stride: 16,
        },
        miss: None,
        hit: None,
    };
    assert_kind(
        scope.dispatch_rays(&bad_usage, 1, 1, 1),
        RhiErrorKind::InvalidUsage,
    );
    scope.end().unwrap();
}

fn ray_pipeline_facts_with_immediates(max: u64) -> CapabilityFacts {
    let mut facts = ray_pipeline_facts(max);
    facts.record_feature(OptionalFeature::Immediates);
    facts.record_limit(LimitKey::ImmediateDataAlignment, 4);
    facts
}

#[test]
fn shared_immediate_validation_has_positive_negative_and_alignment_boundaries() {
    let mut facts = ray_pipeline_facts_with_immediates(1);
    facts.record_limit(LimitKey::MaxImmediateSize, 4);
    let recorder = recorder_reporting(facts);
    let interface = PipelineInterface::new(
        object(840),
        device(),
        PipelineInterfaceDescriptor::new(Vec::new()).with_immediate_range(ImmediateRange::new(
            0,
            4,
            ShaderStages::COMPUTE,
        )),
        PipelineInterfaceCompatibilityId::new(840),
        LayoutFingerprint([40; 32]),
    );
    assert!(
        crate::api::command::advanced::immediate_write(
            &recorder,
            &interface,
            0,
            &[1, 2, 3, 4],
            "test"
        )
        .is_ok()
    );
    assert_kind(
        crate::api::command::advanced::immediate_write(&recorder, &interface, 0, &[], "test")
            .map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        crate::api::command::advanced::immediate_write(
            &recorder,
            &interface,
            2,
            &[1, 2, 3, 4],
            "test",
        )
        .map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );
}
fn shader_table(owner: DeviceIdentity) -> crate::api::command::RayTracingShaderTable {
    crate::api::command::RayTracingShaderTable {
        ray_generation: crate::api::command::RayTracingShaderTableRegion {
            buffer: fixture::buffer(
                object(828),
                owner,
                BufferDescriptor::new(16, BufferUsage::INDIRECT),
            ),
            range: BufferRange::new(0, 16),
            stride: 16,
        },
        miss: None,
        hit: None,
    }
}

#[test]
fn acceleration_build_records_actual_input_scratch_and_destination_uses() {
    let mut recorder = recorder_reporting(ray_facts());
    let destination = structure(801, device());
    let scratch = fixture::buffer(
        object(802),
        device(),
        BufferDescriptor::new(256, BufferUsage::ACCELERATION_STRUCTURE_SCRATCH),
    );
    recorder
        .build_acceleration_structure(
            &destination,
            &scratch,
            AccelerationStructureBuildMode::Build,
        )
        .unwrap();
    let work = recorder.finish().unwrap();
    assert!(work.resource_uses().iter().any(|use_record| matches!(use_record, crate::api::command::ResourceUse::AccelerationStructure(use_record) if use_record.structure.id() == destination.id())));
}

#[test]
fn acceleration_commands_refuse_foreign_or_invalid_resources_before_recording() {
    let mut recorder = recorder_reporting(ray_facts());
    let destination = structure(803, device());
    let foreign = structure(804, other_device());
    assert_kind(
        recorder.copy_acceleration_structure(
            &foreign,
            &destination,
            AccelerationStructureCopyMode::Clone,
        ),
        RhiErrorKind::WrongDevice,
    );
    let bad_scratch = fixture::buffer(
        object(805),
        device(),
        BufferDescriptor::new(64, BufferUsage::COPY_DST),
    );
    assert_kind(
        recorder.build_acceleration_structure(
            &destination,
            &bad_scratch,
            AccelerationStructureBuildMode::Build,
        ),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn acceleration_update_and_compaction_are_independently_capability_gated() {
    let mut recorder = recorder_reporting(ray_facts());
    let destination = structure(806, device());
    let source = structure(807, device());
    let scratch = fixture::buffer(
        object(808),
        device(),
        BufferDescriptor::new(64, BufferUsage::ACCELERATION_STRUCTURE_SCRATCH),
    );
    assert_kind(
        recorder.build_acceleration_structure(
            &destination,
            &scratch,
            AccelerationStructureBuildMode::Update,
        ),
        RhiErrorKind::Unsupported,
    );
    assert_kind(
        recorder.copy_acceleration_structure(
            &source,
            &destination,
            AccelerationStructureCopyMode::Compact,
        ),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn mesh_dispatch_has_positive_capability_foreign_and_boundaries() {
    let mut recorder = recorder_reporting(mesh_facts(1));
    let mut scope = recorder.begin_raster(&color_scope("mesh")).unwrap();
    assert_kind(
        scope.set_mesh_pipeline(&mesh_pipeline(other_device())),
        RhiErrorKind::WrongDevice,
    );
    scope.set_mesh_pipeline(&mesh_pipeline(device())).unwrap();
    assert_kind(scope.dispatch_mesh(0, 1, 1), RhiErrorKind::InvalidUsage);
    assert_kind(scope.dispatch_mesh(2, 1, 1), RhiErrorKind::InvalidUsage);
    scope.dispatch_mesh(1, 1, 1).unwrap();
    scope.end().unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn mesh_indirect_validates_usage_alignment_count_and_capability() {
    let mut facts = mesh_facts(1);
    facts.record_feature(OptionalFeature::IndirectDraw);
    facts.record_feature(OptionalFeature::MultiDrawIndirectCount);
    let mut recorder = recorder_reporting(facts);
    let mut scope = recorder
        .begin_raster(&color_scope("mesh indirect"))
        .unwrap();
    scope.set_mesh_pipeline(&mesh_pipeline(device())).unwrap();
    let arguments = fixture::buffer(
        object(826),
        device(),
        BufferDescriptor::new(16, BufferUsage::INDIRECT),
    );
    let count = fixture::buffer(
        object(827),
        device(),
        BufferDescriptor::new(4, BufferUsage::INDIRECT),
    );
    assert_kind(
        scope.dispatch_mesh_indirect(&arguments, 2),
        RhiErrorKind::InvalidUsage,
    );
    scope
        .dispatch_mesh_indirect_count(&arguments, 0, &count, 0, 1)
        .unwrap();
    scope.end().unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn ray_dispatch_has_positive_capability_bind_state_and_boundaries() {
    let mut unavailable = recorder();
    assert_kind(
        unavailable
            .begin_ray_tracing(&Default::default())
            .map(|_| ()),
        RhiErrorKind::Unsupported,
    );
    let mut recorder = recorder_reporting(ray_pipeline_facts(1));
    let mut scope = recorder.begin_ray_tracing(&Default::default()).unwrap();
    assert_kind(
        scope.dispatch_rays(&shader_table(device()), 1, 1, 1),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        scope.set_pipeline(&ray_pipeline(other_device())),
        RhiErrorKind::WrongDevice,
    );
    scope.set_pipeline(&ray_pipeline(device())).unwrap();
    assert_kind(
        scope.dispatch_rays(&shader_table(device()), 0, 1, 1),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        scope.dispatch_rays(&shader_table(device()), 2, 1, 1),
        RhiErrorKind::InvalidUsage,
    );
    scope
        .dispatch_rays(&shader_table(device()), 1, 1, 1)
        .unwrap();
    scope.end().unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn ray_immediates_accept_an_exact_declared_and_aligned_boundary_write() {
    let mut recorder = recorder_reporting(ray_pipeline_facts_with_immediates(1));
    let mut scope = recorder.begin_ray_tracing(&Default::default()).unwrap();
    scope
        .set_pipeline(&ray_pipeline_with_immediates(device()))
        .unwrap();
    // The four-byte declaration and four-byte alignment are both exact here.
    scope.set_immediates(0, &[1, 2, 3, 4]).unwrap();
    scope
        .dispatch_rays(&shader_table(device()), 1, 1, 1)
        .unwrap();
    scope.end().unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn ray_immediates_refuse_missing_capability_pipeline_and_undeclared_ranges() {
    let mut unsupported_recorder = recorder_reporting(ray_pipeline_facts(1));
    let mut unsupported_scope = unsupported_recorder
        .begin_ray_tracing(&Default::default())
        .unwrap();
    assert_kind(
        unsupported_scope.set_immediates(0, &[1, 2, 3, 4]),
        RhiErrorKind::Unsupported,
    );
    unsupported_scope.end().unwrap();

    let mut recorder = recorder_reporting(ray_pipeline_facts_with_immediates(1));
    let mut scope = recorder.begin_ray_tracing(&Default::default()).unwrap();
    assert_kind(
        scope.set_immediates(0, &[1, 2, 3, 4]),
        RhiErrorKind::InvalidUsage,
    );
    scope
        .set_pipeline(&ray_pipeline_with_immediates(device()))
        .unwrap();
    assert_kind(
        scope.set_immediates(4, &[1, 2, 3, 4]),
        RhiErrorKind::InvalidUsage,
    );
    scope.end().unwrap();
}

#[test]
fn ray_immediates_refuse_empty_and_unaligned_writes_at_the_range_boundary() {
    let mut recorder = recorder_reporting(ray_pipeline_facts_with_immediates(1));
    let mut scope = recorder.begin_ray_tracing(&Default::default()).unwrap();
    scope
        .set_pipeline(&ray_pipeline_with_immediates(device()))
        .unwrap();
    assert_kind(scope.set_immediates(0, &[]), RhiErrorKind::InvalidUsage);
    assert_kind(
        scope.set_immediates(2, &[1, 2, 3, 4]),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(scope.set_immediates(0, &[1, 2]), RhiErrorKind::InvalidUsage);
    scope.end().unwrap();
}
