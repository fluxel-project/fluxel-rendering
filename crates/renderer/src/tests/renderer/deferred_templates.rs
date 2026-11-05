use super::*;

use crate::{GeometryAttribute, GeometryLayout, GeometryStream};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

fn position_layout() -> GeometryLayout {
    GeometryLayout {
        streams: vec![GeometryStream {
            stride: 12,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![GeometryAttribute {
                semantic: "POSITION".into(),
                offset: 0,
                format: VertexFormat::Float32(3),
            }],
        }],
    }
}

#[test]
fn position_only_template_declares_mrt_geometry_and_derivative_normal() {
    let request = gbuffer_request(&position_layout()).expect("position-only layout is supported");
    let source = &request.source_units[0].source;
    assert!(source.contains("@location(0) albedo"));
    assert!(source.contains("@location(1) normal"));
    assert!(source.contains("cross(dpdx(input.local_position), dpdy(input.local_position))"));
    assert!(
        request
            .static_options
            .contains_key("deferred.vertex_inputs")
    );
}

#[test]
fn template_refuses_attributes_it_cannot_preserve() {
    let mut layout = position_layout();
    layout.streams[0].attributes.push(GeometryAttribute {
        semantic: "UV".into(),
        offset: 12,
        format: VertexFormat::Float32(2),
    });
    assert!(gbuffer_request(&layout).is_none());
}

#[test]
fn lighting_reads_all_gbuffer_products_and_clears_depth_to_black() {
    assert!(LIGHTING_TEMPLATE.contains("textureLoad(deferred_albedo"));
    assert!(LIGHTING_TEMPLATE.contains("textureLoad(deferred_normal"));
    assert!(LIGHTING_TEMPLATE.contains("textureLoad(deferred_depth"));
    assert!(LIGHTING_TEMPLATE.contains("if depth >= 1.0"));
    assert!(LIGHTING_TEMPLATE.contains("0.2 + 0.6 * abs(normal.z) + 0.2 * (1.0 - depth)"));
}
