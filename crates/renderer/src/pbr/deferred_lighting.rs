//! Deferred PBR lighting requests and view binding.

use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

use super::{direct_light, templates};

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
