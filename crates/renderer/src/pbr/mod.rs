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
    matches!(
        output,
        SurfaceOutput::MetallicRoughness { .. }
            | SurfaceOutput::MetallicRoughnessUv0 { .. }
            | SurfaceOutput::MetallicRoughnessUv0Normal { .. }
    )
}

pub(crate) fn forward_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    geometry_request(layout, false)
}

pub(crate) fn gbuffer_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    geometry_request(layout, true)
}

pub(crate) fn forward_uv0_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    geometry_uv0_request(layout, false)
}

pub(crate) fn gbuffer_uv0_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    geometry_uv0_request(layout, true)
}

pub(crate) fn forward_uv0_normal_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    geometry_uv0_normal_request(layout, false)
}

pub(crate) fn gbuffer_uv0_normal_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    geometry_uv0_normal_request(layout, true)
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

fn geometry_uv0_request(
    layout: &GeometryLayout,
    deferred: bool,
) -> Option<ShaderCompositionRequest> {
    let mut base_layout = layout.clone();
    for stream in &mut base_layout.streams {
        stream
            .attributes
            .retain(|attribute| attribute.semantic != "UV0");
    }
    // UV0 may be supplied by its own stream.  The legacy PBR request only
    // needs POSITION/NORMAL, so omit a stream made empty solely by removing
    // the UV0 attribute rather than treating that legal physical layout as an
    // invalid zero-attribute vertex stream.
    base_layout
        .streams
        .retain(|stream| !stream.attributes.is_empty());
    let mut request = geometry_request(&base_layout, deferred)?;
    let attributes: Vec<_> = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .collect();
    if attributes.len() != 3
        || attributes
            .iter()
            .filter(|attribute| {
                attribute.semantic == "UV0" && matches!(attribute.format, VertexFormat::Float32(2))
            })
            .count()
            != 1
    {
        return None;
    }
    request.source_units.retain(|unit| {
        unit.id.0
            != if deferred {
                "pbr-gbuffer"
            } else {
                "pbr-forward"
            }
    });
    request.source_units.push(SourceUnit::new(
        if deferred {
            "pbr-gbuffer-uv0"
        } else {
            "pbr-forward-uv0"
        },
        if deferred {
            templates::GBUFFER_UV0
        } else {
            templates::FORWARD_UV0
        },
    ));
    request.vertex_inputs.push(VertexInputDeclaration::new(
        "UV0",
        2,
        VertexType::Vec2(ScalarType::F32),
    ));
    request
        .static_options
        .insert("renderer.material_context".into(), "uv0".into());
    Some(request)
}

fn geometry_uv0_normal_request(
    layout: &GeometryLayout,
    deferred: bool,
) -> Option<ShaderCompositionRequest> {
    let mut base = layout.clone();
    for stream in &mut base.streams {
        stream
            .attributes
            .retain(|attribute| attribute.semantic != "UV0" && attribute.semantic != "TANGENT");
    }
    base.streams.retain(|stream| !stream.attributes.is_empty());
    let mut request = geometry_request(&base, deferred)?;
    let attributes: Vec<_> = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .collect();
    if attributes.len() != 4
        || attributes
            .iter()
            .filter(|a| a.semantic == "UV0" && matches!(a.format, VertexFormat::Float32(2)))
            .count()
            != 1
        || attributes
            .iter()
            .filter(|a| a.semantic == "TANGENT" && matches!(a.format, VertexFormat::Float32(4)))
            .count()
            != 1
    {
        return None;
    }
    request.source_units.retain(|unit| {
        unit.id.0
            != if deferred {
                "pbr-gbuffer"
            } else {
                "pbr-forward"
            }
    });
    request.source_units.push(SourceUnit::new(
        if deferred {
            "pbr-gbuffer-uv0-normal"
        } else {
            "pbr-forward-uv0-normal"
        },
        if deferred {
            templates::GBUFFER_UV0_NORMAL
        } else {
            templates::FORWARD_UV0_NORMAL
        },
    ));
    request.vertex_inputs.push(VertexInputDeclaration::new(
        "UV0",
        2,
        VertexType::Vec2(ScalarType::F32),
    ));
    request.vertex_inputs.push(VertexInputDeclaration::new(
        "TANGENT",
        3,
        VertexType::Vec4(ScalarType::F32),
    ));
    request.bindings[0] = BindingDeclaration::new(1, 0, BindingKind::Uniform)
        .with_visibility([ShaderStage::Vertex, ShaderStage::Fragment]);
    request
        .static_options
        .insert("renderer.material_context".into(), "uv0-normal".into());
    Some(request)
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

#[cfg(test)]
#[path = "../tests/renderer/pbr_tbn_handedness.rs"]
mod tbn_handedness_tests;
