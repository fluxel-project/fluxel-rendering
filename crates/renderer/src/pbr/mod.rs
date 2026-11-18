//! Shared opaque metallic/roughness consumers for Forward and Deferred.
//!
//! The selected lighting policy uses one view-owned directional light and
//! supports orthographic and perspective shading cameras. Material owns
//! surface evaluation. Renderer owns transforms, lighting and G-buffer encoding.

use fluxel_material::SurfaceOutput;
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ScalarType, ShaderCompositionRequest, ShaderStage,
    SourceUnit, VertexInputDeclaration, VertexType,
};

use crate::GeometryLayout;
use crate::preparation::forward_fragment::{
    ForwardFragmentInterface, with_ordinary_forward_fragment,
};

mod direct_light;
mod scene_refraction;
pub(crate) mod specular;
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

/// Builds the Forward PBR route with a scene-level diffuse irradiance input.
///
/// This is intentionally a diffuse-only IBL subset; it does not claim the
/// prefiltered radiance or BRDF LUT required for specular IBL.
pub(crate) fn forward_irradiance_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    let mut request = geometry_request(layout, false)?;
    request.source_units.push(SourceUnit::new(
        "pbr-diffuse-irradiance",
        templates::DIFFUSE_IRRADIANCE,
    ));
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-forward");
    request.source_units.push(SourceUnit::new(
        "pbr-forward-irradiance",
        templates::FORWARD_IRRADIANCE,
    ));
    request = request
        .with_binding(
            BindingDeclaration::new(2, 0, BindingKind::Texture)
                .with_visibility([ShaderStage::Fragment]),
        )
        .with_binding(
            BindingDeclaration::new(2, 1, BindingKind::Sampler)
                .with_visibility([ShaderStage::Fragment]),
        )
        .with_static_option("renderer.environment", "latlong-diffuse-irradiance-v1");
    Some(request)
}

pub(crate) fn forward_uv0_irradiance_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    irradiance_forward_request(
        geometry_uv0_request(layout, false)?,
        "pbr-forward-uv0",
        "pbr-forward-uv0-irradiance",
        templates::FORWARD_UV0_IRRADIANCE,
    )
}

pub(crate) fn forward_uv0_normal_irradiance_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    irradiance_forward_request(
        geometry_uv0_normal_request(layout, false)?,
        "pbr-forward-uv0-normal",
        "pbr-forward-uv0-normal-irradiance",
        templates::FORWARD_UV0_NORMAL_IRRADIANCE,
    )
}

fn irradiance_forward_request(
    mut request: ShaderCompositionRequest,
    previous_entry: &str,
    identity: &str,
    body: &str,
) -> Option<ShaderCompositionRequest> {
    request.source_units.push(SourceUnit::new(
        "pbr-diffuse-irradiance",
        templates::DIFFUSE_IRRADIANCE,
    ));
    request
        .source_units
        .retain(|unit| unit.id.0 != previous_entry);
    request.source_units.push(SourceUnit::new(identity, body));
    Some(
        request
            .with_binding(
                BindingDeclaration::new(2, 0, BindingKind::Texture)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(2, 1, BindingKind::Sampler)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_static_option("renderer.environment", "latlong-diffuse-irradiance-v1"),
    )
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

pub(crate) fn forward_uv0_normal_effect_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    let mut request = geometry_uv0_normal_request(layout, false)?;
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-forward-uv0-normal");
    request.source_units.push(SourceUnit::new(
        "pbr-forward-uv0-normal-effect",
        templates::FORWARD_UV0_NORMAL_EFFECT,
    ));
    request.static_options.insert(
        "renderer.surface_effects".into(),
        "emissive-ao-rgba16float-v1".into(),
    );
    Some(request)
}

pub(crate) fn forward_uv0_normal_effect_irradiance_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    let mut request = forward_uv0_normal_effect_request(layout)?;
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-forward-uv0-normal-effect");
    request.source_units.push(SourceUnit::new(
        "pbr-diffuse-irradiance",
        templates::DIFFUSE_IRRADIANCE,
    ));
    request.source_units.push(SourceUnit::new(
        "pbr-forward-uv0-normal-effect-irradiance",
        templates::FORWARD_UV0_NORMAL_EFFECT_IRRADIANCE,
    ));
    Some(
        request
            .with_binding(
                BindingDeclaration::new(2, 0, BindingKind::Texture)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(2, 1, BindingKind::Sampler)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_static_option("renderer.environment", "latlong-diffuse-irradiance-v1"),
    )
}

pub(crate) fn gbuffer_uv0_normal_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    geometry_uv0_normal_request(layout, true)
}

/// Deferred MRT request for a UV0 normal surface that has an AO or emissive
/// extension. The extra target is selected by the Renderer only for such a
/// material; the legacy request retains its two colour attachments.
pub(crate) fn gbuffer_uv0_normal_effect_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    let mut request = geometry_uv0_normal_request(layout, true)?;
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-gbuffer-uv0-normal");
    request.source_units.push(SourceUnit::new(
        "pbr-gbuffer-uv0-normal-effect",
        templates::GBUFFER_UV0_NORMAL_EFFECT,
    ));
    request.static_options.insert(
        "renderer.surface_effects".into(),
        "emissive-ao-rgba16float-v1".into(),
    );
    Some(request)
}

const VIEW_FACING: &str = "pbr-view-facing";

/// Specializes the PBR normal-facing predicate for a reflected camera.
pub(crate) fn with_mirrored_view(mut request: ShaderCompositionRequest) -> ShaderCompositionRequest {
    let unit = request
        .source_units
        .iter_mut()
        .find(|unit| unit.id.0 == VIEW_FACING)
        .expect("PBR geometry includes its view-facing predicate");
    *unit = SourceUnit::new(
        VIEW_FACING,
        "fn pbr_flip_back_normal(front_facing: bool) -> bool { return front_facing; }",
    );
    request
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
        SourceUnit::new(
            VIEW_FACING,
            "fn pbr_flip_back_normal(front_facing: bool) -> bool { return !front_facing; }",
        ),
        SourceUnit::new("pbr-surface-normal", templates::SURFACE_NORMAL),
        SourceUnit::new("pbr-transforms", templates::TRANSFORMS),
        SourceUnit::new(identity, body),
    ];
    if !deferred {
        units.push(SourceUnit::new("pbr-direct-light", direct_light::WGSL));
        units.push(SourceUnit::new("pbr-directional-light-binding", templates::FORWARD_LIGHT));
    }
    let transform_stages = if deferred {
        vec![ShaderStage::Vertex]
    } else {
        vec![ShaderStage::Vertex, ShaderStage::Fragment]
    };
    let request = ShaderCompositionRequest::new(units)
        .with_entry_point(EntryPoint::new(vertex, ShaderStage::Vertex))
        .with_entry_point(EntryPoint::new(fragment, ShaderStage::Fragment))
        .with_binding(
            BindingDeclaration::new(1, 0, BindingKind::Uniform).with_visibility(transform_stages),
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
        .with_static_option("renderer.light", "world-directional-uniform-v2");
    Some(if deferred {
        request
    } else {
        with_ordinary_forward_fragment(request, ForwardFragmentInterface::Pbr)
    })
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
    if !deferred {
        request = with_ordinary_forward_fragment(request, ForwardFragmentInterface::PbrUv0);
    }
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
    request.source_units.push(SourceUnit::new(
        "pbr-tangent-basis",
        templates::TANGENT_BASIS,
    ));
    request
        .static_options
        .insert("renderer.material_context".into(), "uv0-normal".into());
    if !deferred {
        request = with_ordinary_forward_fragment(request, ForwardFragmentInterface::PbrUv0Normal);
    }
    Some(request)
}

pub(crate) fn basic_deferred_view_binding(effects: bool) -> u32 {
    if effects { 4 } else { 3 }
}

fn with_basic_deferred_view(
    mut request: ShaderCompositionRequest,
    effects: bool,
) -> ShaderCompositionRequest {
    let binding = basic_deferred_view_binding(effects);
    request.source_units.push(SourceUnit::new(
        "pbr-basic-light-view",
        templates::BASIC_LIGHT_VIEW.replace("@binding(4)", &format!("@binding({binding})")),
    ));
    request.with_binding(
        BindingDeclaration::new(0, binding, BindingKind::Uniform)
            .with_visibility([ShaderStage::Fragment]),
    )
}

pub(crate) fn lighting_request() -> ShaderCompositionRequest {
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("pbr-normal", templates::NORMAL),
        SourceUnit::new("pbr-direct-light", direct_light::WGSL),
        SourceUnit::new("pbr-diffuse-irradiance", templates::DIFFUSE_IRRADIANCE),
        SourceUnit::new("pbr-deferred-lighting", templates::LIGHTING),
    ])
    .with_entry_point(EntryPoint::new("deferred_lighting_vs", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new(
        "deferred_lighting_fs",
        ShaderStage::Fragment,
    ))
    .with_static_option("renderer.light", "world-directional-uniform-v2");
    for binding in 0..3 {
        request = request.with_binding(
            BindingDeclaration::new(0, binding, BindingKind::Texture)
                .with_visibility([ShaderStage::Fragment]),
        );
    }
    with_basic_deferred_view(request, false)
}

/// Builds the Deferred lighting route with the same diffuse irradiance input
/// used by the Forward consumer.
pub(crate) fn lighting_irradiance_request() -> ShaderCompositionRequest {
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("pbr-normal", templates::NORMAL),
        SourceUnit::new("pbr-direct-light", direct_light::WGSL),
        SourceUnit::new("pbr-diffuse-irradiance", templates::DIFFUSE_IRRADIANCE),
        SourceUnit::new(
            "pbr-deferred-lighting-irradiance",
            templates::LIGHTING_IRRADIANCE,
        ),
    ])
    .with_entry_point(EntryPoint::new("deferred_lighting_vs", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new(
        "deferred_lighting_fs",
        ShaderStage::Fragment,
    ))
    .with_static_option("renderer.light", "world-directional-uniform-v2")
    .with_static_option("renderer.environment", "latlong-diffuse-irradiance-v1");
    for binding in 0..3 {
        request = request.with_binding(
            BindingDeclaration::new(0, binding, BindingKind::Texture)
                .with_visibility([ShaderStage::Fragment]),
        );
    }
    let request = request
        .with_binding(
            BindingDeclaration::new(1, 0, BindingKind::Texture)
                .with_visibility([ShaderStage::Fragment]),
        )
        .with_binding(
            BindingDeclaration::new(1, 1, BindingKind::Sampler)
                .with_visibility([ShaderStage::Fragment]),
        );
    with_basic_deferred_view(request, false)
}

pub(crate) fn lighting_effect_request() -> ShaderCompositionRequest {
    lighting_effect_request_inner(false)
}

pub(crate) fn lighting_effect_irradiance_request() -> ShaderCompositionRequest {
    lighting_effect_request_inner(true)
}

fn lighting_effect_request_inner(environment: bool) -> ShaderCompositionRequest {
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("pbr-normal", templates::NORMAL),
        SourceUnit::new("pbr-direct-light", direct_light::WGSL),
        SourceUnit::new("pbr-diffuse-irradiance", templates::DIFFUSE_IRRADIANCE),
        SourceUnit::new(
            if environment {
                "pbr-deferred-lighting-effect-irradiance"
            } else {
                "pbr-deferred-lighting-effect"
            },
            if environment {
                templates::LIGHTING_EFFECT_IRRADIANCE
            } else {
                templates::LIGHTING_EFFECT
            },
        ),
    ])
    .with_entry_point(EntryPoint::new("deferred_lighting_vs", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new(
        "deferred_lighting_fs",
        ShaderStage::Fragment,
    ))
    .with_static_option("renderer.light", "world-directional-uniform-v2")
    .with_static_option("renderer.surface_effects", "emissive-ao-rgba16float-v1");
    for binding in 0..4 {
        request = request.with_binding(
            BindingDeclaration::new(0, binding, BindingKind::Texture)
                .with_visibility([ShaderStage::Fragment]),
        );
    }
    if environment {
        request = request
            .with_binding(
                BindingDeclaration::new(1, 0, BindingKind::Texture)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(1, 1, BindingKind::Sampler)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_static_option("renderer.environment", "latlong-diffuse-irradiance-v1");
    }
    with_basic_deferred_view(request, true)
}

#[cfg(test)]
#[path = "../tests/renderer/pbr_tbn_handedness.rs"]
mod tbn_handedness_tests;
