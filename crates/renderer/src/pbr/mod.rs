//! Shared opaque metallic/roughness consumers for Forward and Deferred.
//!
//! This selected lighting policy uses a world-space directional light and a
//! constant orthographic view direction. It is not a perspective camera or an
//! environment-lighting implementation. Material owns surface evaluation;
//! Renderer owns normal transforms, lighting, and G-buffer encoding.

use fluxel_material::SurfaceOutput;
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ScalarType, ShaderCompositionRequest, ShaderStage,
    SourceUnit, VertexInputDeclaration, VertexType,
};

use crate::GeometryLayout;

mod templates;

pub(crate) fn is_pbr(output: &SurfaceOutput) -> bool {
    matches!(output, SurfaceOutput::MetallicRoughness { .. })
}

pub(crate) fn forward_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    geometry_request(layout, false)
}

pub(crate) fn gbuffer_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    geometry_request(layout, true)
}

fn geometry_request(layout: &GeometryLayout, deferred: bool) -> Option<ShaderCompositionRequest> {
    if layout.streams.is_empty()
        || layout.streams.iter().any(|stream| {
            stream.attributes.is_empty() || !matches!(stream.step_mode, VertexStepMode::Vertex)
        })
    {
        return None;
    }
    let attributes: Vec<_> = layout.streams.iter().flat_map(|s| &s.attributes).collect();
    if attributes.len() != 2
        || ["POSITION", "NORMAL"].iter().any(|semantic| {
            attributes
                .iter()
                .filter(|a| a.semantic == *semantic && matches!(a.format, VertexFormat::Float32(3)))
                .count()
                != 1
        })
    {
        return None;
    }
    let (identity, vertex, fragment, body) = if deferred {
        (
            "pbr-gbuffer",
            "deferred_gbuffer_vs",
            "deferred_gbuffer_fs",
            templates::GBUFFER,
        )
    } else {
        (
            "pbr-forward",
            "forward_vs",
            "forward_fs",
            templates::FORWARD,
        )
    };
    let mut units = vec![
        SourceUnit::new("pbr-normal", templates::NORMAL),
        SourceUnit::new("pbr-transforms", templates::TRANSFORMS),
        SourceUnit::new(identity, body),
    ];
    if !deferred {
        units.push(SourceUnit::new("pbr-direct-light", templates::DIRECT_LIGHT));
    }
    Some(
        ShaderCompositionRequest::new(units)
            .with_entry_point(EntryPoint::new(vertex, ShaderStage::Vertex))
            .with_entry_point(EntryPoint::new(fragment, ShaderStage::Fragment))
            .with_binding(
                BindingDeclaration::new(1, 0, BindingKind::Uniform)
                    .with_visibility([ShaderStage::Vertex]),
            )
            .with_vertex_input(VertexInputDeclaration::new(
                "POSITION",
                0,
                VertexType::Vec3(ScalarType::F32),
            ))
            .with_vertex_input(VertexInputDeclaration::new(
                "NORMAL",
                1,
                VertexType::Vec3(ScalarType::F32),
            ))
            .with_static_option("renderer.surface", "opaque-metallic-roughness")
            .with_static_option("renderer.light", "world-directional-orthographic-v1"),
    )
}

pub(crate) fn lighting_request() -> ShaderCompositionRequest {
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("pbr-normal", templates::NORMAL),
        SourceUnit::new("pbr-direct-light", templates::DIRECT_LIGHT),
        SourceUnit::new("pbr-deferred-lighting", templates::LIGHTING),
    ])
    .with_entry_point(EntryPoint::new("deferred_lighting_vs", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new(
        "deferred_lighting_fs",
        ShaderStage::Fragment,
    ))
    .with_static_option("renderer.light", "world-directional-orthographic-v1");
    for binding in 0..3 {
        request = request.with_binding(
            BindingDeclaration::new(0, binding, BindingKind::Texture)
                .with_visibility([ShaderStage::Fragment]),
        );
    }
    request
}
