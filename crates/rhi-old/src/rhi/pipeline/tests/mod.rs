//! Contract tests for the pipeline vocabulary.
//!
//! These pin the rules that make a pipeline descriptor mean one thing across
//! backends: strip state is decided at creation, trailing empty color locations
//! have one representation, and a zero-sized or mis-summed compute workgroup is
//! refused before any backend sees it.

use super::state::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWriteMask, DepthBiasState,
    DepthState, DepthStencilState, MultisampleState, PrimitiveState, PrimitiveTopology,
    RenderTargetSignature, StencilFaceState, StencilState, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexInputLimits, VertexInputState, VertexStepMode,
    validate_depth_stencil_state, validate_multisample_state, validate_primitive_state,
    validate_vertex_input,
};
use super::{ComputeLimits, IndexFormat, RasterLimits, validate_compute_workgroup};
use crate::rhi::format::TextureFormat;
use crate::rhi::platform::RhiErrorKind;
use crate::rhi::resource::CompareFunction;
use crate::rhi::shader::{ComputeWorkgroupRequirements, ShaderLocation};

fn vertex_limits() -> VertexInputLimits {
    VertexInputLimits {
        max_buffers: 8,
        max_attributes: 16,
        max_stride: 2048,
    }
}

fn workgroup_limits() -> ComputeLimits {
    ComputeLimits {
        max_size_x: 64,
        max_size_y: 64,
        max_size_z: 64,
        max_invocations: 256,
        max_workgroup_storage: 16_384,
    }
}

#[test]
fn strip_index_format_is_rejected_on_non_strip_topologies() {
    let list = PrimitiveState::new(PrimitiveTopology::TriangleList);
    assert!(validate_primitive_state(&list).is_ok());

    let bad = list.with_strip_index_format(IndexFormat::Uint16);
    assert_eq!(
        validate_primitive_state(&bad).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );

    let strip = PrimitiveState::new(PrimitiveTopology::TriangleStrip)
        .with_strip_index_format(IndexFormat::Uint32);
    assert!(validate_primitive_state(&strip).is_ok());
}

#[test]
fn depth_bias_is_triangles_only_and_needs_a_finite_slope() {
    let lines = PrimitiveState::new(PrimitiveTopology::LineList)
        .with_depth_bias(DepthBiasState::new(1, 1.0));
    assert_eq!(
        validate_primitive_state(&lines).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );

    let infinite = PrimitiveState::new(PrimitiveTopology::TriangleList)
        .with_depth_bias(DepthBiasState::new(1, f32::INFINITY));
    assert_eq!(
        validate_primitive_state(&infinite).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );

    let finite = PrimitiveState::new(PrimitiveTopology::TriangleList)
        .with_depth_bias(DepthBiasState::new(-4, 1.5));
    assert!(validate_primitive_state(&finite).is_ok());
}

#[test]
fn trailing_empty_color_locations_have_one_representation() {
    let padded =
        RenderTargetSignature::new(vec![Some(TextureFormat::Rgba8Unorm), None, None], None, 1);
    let trimmed = RenderTargetSignature::new(vec![Some(TextureFormat::Rgba8Unorm)], None, 1);
    assert_eq!(padded, trimmed);
    assert_eq!(padded.active_color_count(), 1);

    // A hole below an active location is not trailing and must survive.
    let holed = RenderTargetSignature::new(
        vec![
            Some(TextureFormat::Rgba8Unorm),
            None,
            Some(TextureFormat::Rgba8Unorm),
        ],
        None,
        1,
    );
    assert_eq!(holed.active_color_count(), 2);
    assert_eq!(holed.color_formats.len(), 3);
}

#[test]
fn an_attribute_past_the_stride_is_refused() {
    let layout = VertexBufferLayout::new(12, VertexStepMode::Vertex).with_attribute(
        VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x4, 0),
    );
    let state = VertexInputState::new().with_buffer(layout);
    assert_eq!(
        validate_vertex_input(&state, vertex_limits())
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage,
        "a 16-byte attribute does not fit a 12-byte stride"
    );

    let layout = VertexBufferLayout::new(16, VertexStepMode::Vertex).with_attribute(
        VertexAttribute::new(ShaderLocation::new(0), VertexFormat::Float32x4, 0),
    );
    assert!(
        validate_vertex_input(&VertexInputState::new().with_buffer(layout), vertex_limits()).is_ok()
    );
}

#[test]
fn a_stride_beyond_the_limit_is_unsupported_rather_than_invalid() {
    let layout = VertexBufferLayout::new(4096, VertexStepMode::Vertex);
    let state = VertexInputState::new().with_buffer(layout);
    assert_eq!(
        validate_vertex_input(&state, vertex_limits()).unwrap_err().kind(),
        RhiErrorKind::Unsupported
    );
}

#[test]
fn depth_and_stencil_must_agree_with_the_format_aspects() {
    let stencil_on_depth_only = DepthStencilState::new(TextureFormat::Depth32Float).with_stencil(
        StencilState::new(
            StencilFaceState::new(CompareFunction::Always),
            StencilFaceState::new(CompareFunction::Always),
        ),
    );
    assert_eq!(
        validate_depth_stencil_state(&stencil_on_depth_only)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );

    let depth_only = DepthStencilState::new(TextureFormat::Depth32Float)
        .with_depth(DepthState::new(CompareFunction::Less));
    assert!(validate_depth_stencil_state(&depth_only).is_ok());

    let neither = DepthStencilState::new(TextureFormat::Depth32Float);
    assert_eq!(
        validate_depth_stencil_state(&neither).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn alpha_to_coverage_needs_more_than_one_sample() {
    assert_eq!(
        validate_multisample_state(&MultisampleState::new(1).with_alpha_to_coverage(true))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    assert!(
        validate_multisample_state(&MultisampleState::new(4).with_alpha_to_coverage(true)).is_ok()
    );
    assert_eq!(
        validate_multisample_state(&MultisampleState::new(0))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn min_and_max_blending_require_unit_factors() {
    let unit = BlendState::new(
        BlendComponent::new(BlendFactor::One, BlendFactor::One, BlendOperation::Min),
        BlendComponent::new(BlendFactor::One, BlendFactor::One, BlendOperation::Min),
    );
    assert!(unit.color.is_well_formed());

    let non_unit = BlendState::new(
        BlendComponent::new(BlendFactor::SrcAlpha, BlendFactor::One, BlendOperation::Max),
        BlendComponent::new(BlendFactor::One, BlendFactor::One, BlendOperation::Add),
    );
    assert!(!non_unit.color.is_well_formed());
    assert!(non_unit.alpha.is_well_formed());
}

#[test]
fn write_masks_compose() {
    let rg = ColorWriteMask::RED.union(ColorWriteMask::GREEN);
    assert!(rg.contains(ColorWriteMask::RED));
    assert!(rg.contains(ColorWriteMask::GREEN));
    assert!(!rg.contains(ColorWriteMask::BLUE));
    assert!(ColorWriteMask::ALL.contains(rg));
    assert_eq!(ColorWriteMask::NONE.bits(), 0);
}

#[test]
fn compute_workgroup_requirements_are_checked_before_the_backend() {
    let limits = workgroup_limits();

    let good = ComputeWorkgroupRequirements::new(8, 8, 1, 64, 0);
    assert!(validate_compute_workgroup(&good, limits).is_ok());

    let zero = ComputeWorkgroupRequirements::new(0, 8, 1, 0, 0);
    assert_eq!(
        validate_compute_workgroup(&zero, limits).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );

    let mis_summed = ComputeWorkgroupRequirements::new(8, 8, 1, 63, 0);
    assert_eq!(
        validate_compute_workgroup(&mis_summed, limits)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );

    // Every dimension fits, but the product does not.
    let too_many = ComputeWorkgroupRequirements::new(32, 32, 1, 1024, 0);
    assert_eq!(
        validate_compute_workgroup(&too_many, limits).unwrap_err().kind(),
        RhiErrorKind::Unsupported
    );

    // A dimension that alone exceeds its limit.
    let too_wide = ComputeWorkgroupRequirements::new(128, 1, 1, 128, 0);
    assert_eq!(
        validate_compute_workgroup(&too_wide, limits).unwrap_err().kind(),
        RhiErrorKind::Unsupported
    );

    let too_much_storage = ComputeWorkgroupRequirements::new(8, 8, 1, 64, 32_768);
    assert_eq!(
        validate_compute_workgroup(&too_much_storage, limits)
            .unwrap_err()
            .kind(),
        RhiErrorKind::Unsupported
    );
}

#[test]
fn the_raster_limits_record_carries_the_vertex_limits() {
    let limits = RasterLimits {
        max_color_attachments: 8,
        max_inter_stage_variables: 16,
        max_bind_groups_plus_vertex_buffers: None,
        vertex: vertex_limits(),
    };
    assert_eq!(limits.max_color_attachments, 8);
    assert_eq!(limits.vertex.max_stride, 2048);
}
