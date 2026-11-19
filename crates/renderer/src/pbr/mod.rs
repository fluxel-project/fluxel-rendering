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

mod deferred_lighting;
mod direct_light;
mod normal_basis;
pub(crate) use normal_basis::specialize_normal_basis;
mod request_options;
mod scene_refraction;
pub(crate) mod specular;
mod templates;
mod uv0_effects;

pub(crate) use deferred_lighting::{
    AMBIENT_OCCLUSION_BINDING, DIFFUSE_ROUGHNESS_BINDING, basic_deferred_view_binding,
    lighting_effect_irradiance_request, lighting_effect_request, lighting_irradiance_request,
    lighting_request, with_deferred_ambient_occlusion, with_deferred_diffuse_roughness,
};
use request_options::{
    VIEW_FACING, install_ambient_occlusion_evaluator, install_diffuse_roughness_evaluator,
    install_emissive_evaluator,
};
pub(crate) use request_options::{
    with_gbuffer_ambient_occlusion, with_gbuffer_diffuse_roughness, with_mirrored_view,
    without_emissive,
};

pub(crate) fn specialize_diffuse_roughness(request: &mut ShaderCompositionRequest, enabled: bool) {
    if !request
        .source_units
        .iter()
        .any(|unit| unit.id.0.starts_with("pbr-forward") || unit.id.0.starts_with("pbr-gbuffer"))
    {
        return;
    }
    let material_uses_uv0 = request
        .static_options
        .get("renderer.material_context")
        .is_some();
    install_diffuse_roughness_evaluator(request, enabled, material_uses_uv0);
}

pub(crate) fn specialize_ambient_occlusion(request: &mut ShaderCompositionRequest, enabled: bool) {
    if !request
        .source_units
        .iter()
        .any(|unit| unit.id.0.starts_with("pbr-forward") || unit.id.0.starts_with("pbr-gbuffer"))
    {
        return;
    }
    let material_uses_uv0 = request
        .static_options
        .contains_key("renderer.material_context");
    install_ambient_occlusion_evaluator(request, enabled, material_uses_uv0);
}

pub(crate) fn is_pbr(output: &SurfaceOutput) -> bool {
    matches!(
        output,
        SurfaceOutput::MetallicRoughness { .. }
            | SurfaceOutput::MetallicRoughnessUv0 { .. }
            | SurfaceOutput::MetallicRoughnessUv0Normal { .. }
            | SurfaceOutput::MetallicRoughnessUv0Graph { .. }
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

pub(crate) fn forward_uv0_effect_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    uv0_effects::forward_request(layout)
}

pub(crate) fn forward_uv0_effect_irradiance_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    uv0_effects::forward_irradiance_request(layout)
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
    install_emissive_evaluator(&mut request);
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

pub(crate) fn gbuffer_uv0_effect_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    uv0_effects::gbuffer_request(layout)
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
    install_emissive_evaluator(&mut request);
    Some(request)
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
        units.push(SourceUnit::new(
            "pbr-directional-light-binding",
            templates::FORWARD_LIGHT,
        ));
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
            .retain(|attribute| attribute.semantic != "UV0" && attribute.semantic != "UV1");
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
        .filter(|attribute| attribute.semantic != "UV1")
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
    request
        .source_units
        .push(SourceUnit::new("pbr-uv0-vertex", templates::UV0_VERTEX));
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
        stream.attributes.retain(|attribute| {
            attribute.semantic != "UV0"
                && attribute.semantic != "UV1"
                && attribute.semantic != "TANGENT"
        });
    }
    base.streams.retain(|stream| !stream.attributes.is_empty());
    let mut request = geometry_request(&base, deferred)?;
    let attributes: Vec<_> = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .filter(|attribute| attribute.semantic != "UV1")
        .collect();
    let tangent = attributes
        .iter()
        .find(|attribute| attribute.semantic == "TANGENT");
    if attributes.len() != 3 + usize::from(tangent.is_some())
        || attributes
            .iter()
            .filter(|a| a.semantic == "UV0" && matches!(a.format, VertexFormat::Float32(2)))
            .count()
            != 1
        || tangent.is_some_and(|attribute| !matches!(attribute.format, VertexFormat::Float32(4)))
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
    if tangent.is_some() {
        request.vertex_inputs.push(VertexInputDeclaration::new(
            "TANGENT",
            3,
            VertexType::Vec4(ScalarType::F32),
        ));
    } else {
        normal_basis::mark_derivative(&mut request);
    }
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

#[cfg(test)]
#[path = "../tests/renderer/pbr_tbn_handedness.rs"]
mod tbn_handedness_tests;
