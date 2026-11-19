//! Deferred PBR lighting requests and view binding.

use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

use super::{direct_light, templates};

pub(crate) const DIFFUSE_ROUGHNESS_BINDING: u32 = 12;
pub(crate) const AMBIENT_OCCLUSION_BINDING: u32 = 13;

pub(crate) fn with_deferred_ambient_occlusion(
    mut request: ShaderCompositionRequest,
) -> ShaderCompositionRequest {
    for unit in &mut request.source_units {
        if !unit.id.0.starts_with("pbr-deferred-lighting") && unit.id.0 != "pbr-basic-light-view" {
            continue;
        }
        unit.source = unit
            .source
            .replace(
                "fn pbr_directional_light()",
                &format!("@group(0) @binding({AMBIENT_OCCLUSION_BINDING}) var pbr_ambient_occlusion: texture_2d<f32>;\nfn pbr_directional_light()"),
            )
            .replace(
                "let normal = normal_roughness.xyz;",
                "let normal = normal_roughness.xyz;\n    let ambient_occlusion = textureLoad(pbr_ambient_occlusion, coordinate, 0);",
            )
            .replace(
                "let normal = nr.xyz;",
                "let normal = nr.xyz;\n    let ambient_occlusion = textureLoad(pbr_ambient_occlusion, coordinate, 0);",
            )
            .replace(
                "pbr_direct_light_view_with_diffuse_roughness(",
                "pbr_direct_light_view_with_diffuse_occlusion(",
            )
            .replace(
                "vec3<f32>(0.04), diffuse_roughness)",
                "vec3<f32>(0.04), diffuse_roughness, ambient_occlusion)",
            )
            .replace(
                "pbr_direct_light_view(",
                "pbr_direct_light_view_with_diffuse_occlusion(",
            )
            .replace(
                "vec3<f32>(0.04))",
                "vec3<f32>(0.04), 0.0, ambient_occlusion)",
            )
            .replace(
                "pbr_diffuse_irradiance(base_metal.rgb, base_metal.a, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04)) * effect.a",
                "pbr_diffuse_irradiance(base_metal.rgb, base_metal.a, normal, pbr_environment_irradiance, pbr_environment_sampler, vec3<f32>(1.0 - 0.04)) * ambient_occlusion.rgb",
            )
            .replace("* surface_effects.a", "* ambient_occlusion.rgb")
            .replace(
                " + terms.diffuse;",
                " + terms.diffuse * mix(vec3<f32>(1.0), ambient_occlusion.rgb, vec3<f32>(ambient_occlusion.a));",
            );
    }
    request.bindings.push(
        BindingDeclaration::new(0, AMBIENT_OCCLUSION_BINDING, BindingKind::Texture)
            .with_visibility([ShaderStage::Fragment]),
    );
    request.static_options.insert(
        "renderer.deferred_ambient_occlusion".into(),
        "rgba32float-v1".into(),
    );
    request
}

pub(crate) fn basic_deferred_view_binding(effects: bool) -> u32 {
    if effects { 4 } else { 3 }
}

/// Adds the R32Float diffuse-roughness G-buffer read at the stable binding
/// shared by all Deferred lighting variants.
pub(crate) fn with_deferred_diffuse_roughness(
    mut request: ShaderCompositionRequest,
) -> ShaderCompositionRequest {
    for unit in &mut request.source_units {
        if !unit.id.0.starts_with("pbr-deferred-lighting") && unit.id.0 != "pbr-basic-light-view" {
            continue;
        }
        unit.source = unit
            .source
            .replace(
                "fn pbr_directional_light()",
                &format!("@group(0) @binding({DIFFUSE_ROUGHNESS_BINDING}) var pbr_diffuse_roughness: texture_2d<f32>;\nfn pbr_directional_light()"),
            )
            .replace(
                "let normal = normal_roughness.xyz;",
                "let normal = normal_roughness.xyz;\n    let diffuse_roughness = textureLoad(pbr_diffuse_roughness, coordinate, 0).r;",
            )
            .replace(
                "let normal = nr.xyz;",
                "let normal = nr.xyz;\n    let diffuse_roughness = textureLoad(pbr_diffuse_roughness, coordinate, 0).r;",
            )
            .replace(
                "pbr_direct_light_view(",
                "pbr_direct_light_view_with_diffuse_roughness(",
            )
            .replace(
                "pbr_direct_light_components(",
                "pbr_direct_light_components_with_diffuse_roughness(",
            )
            .replace(
                "vec3<f32>(0.04))",
                "vec3<f32>(0.04), diffuse_roughness)",
            )
            .replace("normal, view, vec3(0.04));", "normal, view, vec3(0.04), diffuse_roughness);");
    }
    request.bindings.push(
        BindingDeclaration::new(0, DIFFUSE_ROUGHNESS_BINDING, BindingKind::Texture)
            .with_visibility([ShaderStage::Fragment]),
    );
    request.static_options.insert(
        "renderer.deferred_diffuse_roughness".into(),
        "r32float-v1".into(),
    );
    request
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
