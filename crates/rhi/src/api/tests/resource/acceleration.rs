//! Acceleration-structure descriptor contract tests.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::acceleration::validate_descriptor;
use crate::api::resource::backend::AccelerationStructureBackend;
use crate::api::resource::{
    AabbGeometry, AccelerationStructure, AccelerationStructureBuildSizes,
    AccelerationStructureDescriptor, AccelerationStructureIndexFormat,
    AccelerationStructureVertexFormat, BlasGeometry, BottomLevelAccelerationStructureDescriptor,
    BufferDescriptor, BufferRange, BufferUsage, TlasInstance,
    TopLevelAccelerationStructureDescriptor, TrianglesGeometry,
};
use std::any::Any;

struct TestAccelerationStructure;
impl AccelerationStructureBackend for TestAccelerationStructure {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn blas(owner: DeviceIdentity) -> AccelerationStructure {
    let input = fixture::buffer(
        object(790),
        owner,
        BufferDescriptor::new(36, BufferUsage::BLAS_INPUT),
    );
    AccelerationStructure::new(
        object(791),
        owner,
        triangle(input, BufferRange::new(0, 36)),
        AccelerationStructureBuildSizes {
            acceleration_structure_size: 1,
            build_scratch_size: 1,
            update_scratch_size: 0,
        },
        Box::new(TestAccelerationStructure),
    )
}

fn triangle(vertices: Buffer, range: BufferRange) -> AccelerationStructureDescriptor {
    AccelerationStructureDescriptor::BottomLevel(BottomLevelAccelerationStructureDescriptor::new(
        vec![BlasGeometry::Triangles(TrianglesGeometry {
            vertices,
            vertex_range: range,
            vertex_format: AccelerationStructureVertexFormat::Float32x3,
            vertex_stride: 12,
            vertex_count: 3,
            primitive_count: 1,
            indices: None,
        })],
    ))
}

#[test]
fn blas_geometry_accepts_owned_blas_input() {
    let buffer = fixture::buffer(
        object(701),
        device(),
        BufferDescriptor::new(64, BufferUsage::BLAS_INPUT),
    );
    validate_descriptor(
        &triangle(buffer, BufferRange::new(0, 36)),
        device(),
        |_| None,
        |_| false,
    )
    .unwrap();
}

#[test]
fn blas_geometry_rejects_missing_usage_and_foreign_input_before_backend() {
    let wrong_usage = fixture::buffer(
        object(702),
        device(),
        BufferDescriptor::new(64, BufferUsage::VERTEX),
    );
    assert_kind(
        validate_descriptor(
            &triangle(wrong_usage, BufferRange::new(0, 36)),
            device(),
            |_| None,
            |_| false,
        ),
        RhiErrorKind::InvalidUsage,
    );
    let foreign = fixture::buffer(
        object(703),
        identity(2),
        BufferDescriptor::new(64, BufferUsage::BLAS_INPUT),
    );
    assert_kind(
        validate_descriptor(
            &triangle(foreign, BufferRange::new(0, 36)),
            device(),
            |_| None,
            |_| false,
        ),
        RhiErrorKind::WrongDevice,
    );
}

#[test]
fn blas_geometry_rejects_overflowing_range_and_invalid_stride() {
    let buffer = fixture::buffer(
        object(704),
        device(),
        BufferDescriptor::new(64, BufferUsage::BLAS_INPUT),
    );
    assert_kind(
        validate_descriptor(
            &triangle(buffer.clone(), BufferRange::new(u64::MAX, 4)),
            device(),
            |_| None,
            |_| false,
        ),
        RhiErrorKind::InvalidUsage,
    );
    let invalid = AccelerationStructureDescriptor::BottomLevel(
        BottomLevelAccelerationStructureDescriptor::new(vec![BlasGeometry::Triangles(
            TrianglesGeometry {
                vertices: buffer,
                vertex_range: BufferRange::new(0, 36),
                vertex_format: AccelerationStructureVertexFormat::Float32x3,
                vertex_stride: 10,
                vertex_count: 3,
                primitive_count: 1,
                indices: None,
            },
        )]),
    );
    assert_kind(
        validate_descriptor(&invalid, device(), |_| None, |_| false),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn blas_counts_and_ranges_accept_exact_end_and_reject_undercoverage_or_total_limit() {
    let vertices = fixture::buffer(
        object(705),
        device(),
        BufferDescriptor::new(72, BufferUsage::BLAS_INPUT),
    );
    let indices = fixture::buffer(
        object(706),
        device(),
        BufferDescriptor::new(12, BufferUsage::BLAS_INPUT),
    );
    let exact = AccelerationStructureDescriptor::BottomLevel(
        BottomLevelAccelerationStructureDescriptor::new(vec![BlasGeometry::Triangles(
            TrianglesGeometry {
                vertices: vertices.clone(),
                vertex_range: BufferRange::new(0, 72),
                vertex_format: AccelerationStructureVertexFormat::Float32x3,
                vertex_stride: 12,
                vertex_count: 6,
                primitive_count: 2,
                indices: Some((
                    indices.clone(),
                    BufferRange::new(0, 12),
                    AccelerationStructureIndexFormat::Uint16,
                )),
            },
        )]),
    );
    validate_descriptor(&exact, device(), |_| None, |_| false).unwrap();
    assert_kind(
        validate_descriptor(
            &exact,
            device(),
            |key| (key == LimitKey::MaxBlasPrimitiveCount).then_some(1),
            |_| false,
        ),
        RhiErrorKind::InvalidUsage,
    );
    let undercovered = AccelerationStructureDescriptor::BottomLevel(
        BottomLevelAccelerationStructureDescriptor::new(vec![BlasGeometry::Triangles(
            TrianglesGeometry {
                vertices,
                vertex_range: BufferRange::new(0, 60),
                vertex_format: AccelerationStructureVertexFormat::Float32x3,
                vertex_stride: 12,
                vertex_count: 6,
                primitive_count: 2,
                indices: Some((
                    indices,
                    BufferRange::new(0, 10),
                    AccelerationStructureIndexFormat::Uint16,
                )),
            },
        )]),
    );
    assert_kind(
        validate_descriptor(&undercovered, device(), |_| None, |_| false),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn aabb_and_extended_format_have_capability_and_coverage_boundaries() {
    let boxes = fixture::buffer(
        object(707),
        device(),
        BufferDescriptor::new(48, BufferUsage::BLAS_INPUT),
    );
    let aabb = AccelerationStructureDescriptor::BottomLevel(
        BottomLevelAccelerationStructureDescriptor::new(vec![BlasGeometry::Aabbs(AabbGeometry {
            boxes,
            range: BufferRange::new(0, 48),
            stride: 24,
            primitive_count: 2,
        })]),
    );
    validate_descriptor(&aabb, device(), |_| None, |_| false).unwrap();
    let vertices = fixture::buffer(
        object(708),
        device(),
        BufferDescriptor::new(24, BufferUsage::BLAS_INPUT),
    );
    let extended = AccelerationStructureDescriptor::BottomLevel(
        BottomLevelAccelerationStructureDescriptor::new(vec![BlasGeometry::Triangles(
            TrianglesGeometry {
                vertices,
                vertex_range: BufferRange::new(0, 24),
                vertex_format: AccelerationStructureVertexFormat::Float16x4,
                vertex_stride: 8,
                vertex_count: 3,
                primitive_count: 1,
                indices: None,
            },
        )]),
    );
    assert_kind(
        validate_descriptor(&extended, device(), |_| None, |_| false),
        RhiErrorKind::Unsupported,
    );
    validate_descriptor(
        &extended,
        device(),
        |_| None,
        |feature| feature == OptionalFeature::ExtendedAccelerationStructureVertexFormats,
    )
    .unwrap();
}

#[test]
fn tlas_rejects_non_finite_transforms_and_nonportable_shader_offsets() {
    let valid = TlasInstance {
        bottom_level: blas(device()),
        transform: [[1.0, 0.0, 0.0, 0.0]; 3],
        mask: 0xff,
        shader_record_offset: 0,
    };
    let mut bad_transform = valid.clone();
    bad_transform.transform[0][0] = f32::NAN;
    let descriptor = AccelerationStructureDescriptor::TopLevel(
        TopLevelAccelerationStructureDescriptor::new(vec![bad_transform]),
    );
    assert_kind(
        validate_descriptor(&descriptor, device(), |_| None, |_| false),
        RhiErrorKind::InvalidUsage,
    );
    let mut bad_offset = valid;
    bad_offset.shader_record_offset = 0x01_00_00_00;
    let descriptor = AccelerationStructureDescriptor::TopLevel(
        TopLevelAccelerationStructureDescriptor::new(vec![bad_offset]),
    );
    assert_kind(
        validate_descriptor(&descriptor, device(), |_| None, |_| false),
        RhiErrorKind::InvalidUsage,
    );
}
