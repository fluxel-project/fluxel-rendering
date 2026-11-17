//! Unlit ColorUv0 template admission tests.

use super::*;
use crate::{FeatureIssuer, GeometryAttribute, GeometryStream};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

fn layout(uv0: bool) -> GeometryLayout {
    let mut attributes = vec![GeometryAttribute {
        semantic: "POSITION".into(),
        offset: 0,
        format: VertexFormat::Float32(3),
    }];
    if uv0 {
        attributes.push(GeometryAttribute {
            semantic: "UV0".into(),
            offset: 12,
            format: VertexFormat::Float32(2),
        });
    }
    GeometryLayout {
        streams: vec![GeometryStream {
            stride: if uv0 { 20 } else { 12 },
            step_mode: VertexStepMode::Vertex,
            attributes,
        }],
    }
}

#[test]
fn color_uv0_template_requires_named_uv0_and_declares_it_at_location_one() {
    let source = ForwardDrawSource::Feature {
        feature: FeatureIssuer::new(0, 0).issue("color-uv0-test").unwrap(),
        local_id: 0,
    };
    let request = color_uv0_template(&layout(true), source)
        .expect("POSITION plus UV0 admits the ColorUv0 template")
        .request();
    assert!(
        request
            .vertex_inputs
            .iter()
            .any(|input| input.semantic == "UV0" && input.location == 1)
    );
    assert!(
        !request
            .vertex_inputs
            .iter()
            .any(|input| input.semantic == "UV")
    );
    assert!(matches!(
        color_uv0_template(&layout(false), source),
        Err(ForwardPreparationError::MissingUv0 { .. })
    ));
}
