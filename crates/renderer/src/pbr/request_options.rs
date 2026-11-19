//! Renderer-owned PBR request specializations.

use fluxel_shader::{ShaderCompositionRequest, SourceUnit};

pub(super) const VIEW_FACING: &str = "pbr-view-facing";
const EMISSIVE_EVALUATOR: &str = "pbr-emissive";
const DIFFUSE_ROUGHNESS_EVALUATOR: &str = "pbr-diffuse-roughness";
const AMBIENT_OCCLUSION_EVALUATOR: &str = "pbr-ambient-occlusion";

pub(super) fn install_ambient_occlusion_evaluator(
    request: &mut ShaderCompositionRequest,
    enabled: bool,
    material_uses_uv0: bool,
) {
    if !enabled
        && !request
            .static_options
            .contains_key("renderer.gbuffer_ambient_occlusion")
    {
        return;
    }
    let evaluator = if enabled {
        "evaluate_ambient_occlusion_uv0(/* fluxel:uv1-local-argument */uv, authored)"
    } else if material_uses_uv0 {
        "vec4<f32>(vec3<f32>(clamp(authored * evaluate_occlusion_uv0(/* fluxel:uv1-local-argument */uv), 0.0, 1.0)), 0.0)"
    } else {
        "vec4<f32>(vec3<f32>(authored), 0.0)"
    };
    let monochrome = if enabled && material_uses_uv0 {
        "evaluate_ambient_occlusion_monochrome(ao.rgb)"
    } else {
        "ao.r"
    };
    request.source_units.push(SourceUnit::new(
        AMBIENT_OCCLUSION_EVALUATOR,
        format!(
            "fn pbr_ambient_occlusion(/* fluxel:uv1-local-param */uv: vec2<f32>, authored: f32) -> vec4<f32> {{ return {evaluator}; }}\nfn pbr_ambient_occlusion_monochrome(ao: vec4<f32>) -> f32 {{ return {monochrome}; }}"
        ),
    ));
    if enabled && material_uses_uv0 {
        let ao = "pbr_ambient_occlusion(/* fluxel:uv1-argument */input.uv0, surface.occlusion)";
        for unit in &mut request.source_units {
            if !unit.id.0.starts_with("pbr-forward") {
                continue;
            }
            unit.source = unit
                .source
                .replace(
                    "let material = surface.base;",
                    &format!("let material = surface.base; let ambient_occlusion_color = {ao};"),
                )
                .replace(
                    " + terms.diffuse",
                    " + terms.diffuse * mix(vec3<f32>(1.0), ambient_occlusion_color.rgb, vec3<f32>(ambient_occlusion_color.a))",
                )
                .replace(
                    " + direct_diffuse;",
                    " + direct_diffuse * mix(vec3<f32>(1.0), ambient_occlusion_color.rgb, vec3<f32>(ambient_occlusion_color.a));",
                )
                .replace(
                    "pbr_direct_light_view_with_diffuse_roughness(",
                    "pbr_direct_light_view_with_diffuse_occlusion(",
                )
                .replace(
                    "vec3<f32>(0.04), diffuse_roughness)",
                    "vec3<f32>(0.04), diffuse_roughness, ambient_occlusion_color)",
                )
                .replace(
                    "pbr_direct_light_view(",
                    "pbr_direct_light_view_with_diffuse_occlusion(",
                )
                .replace(
                    "vec3<f32>(0.04))",
                    "vec3<f32>(0.04), 0.0, ambient_occlusion_color)",
                )
                .replace(
                    "let ambient_occlusion = surface.occlusion;",
                    "let ambient_occlusion = pbr_ambient_occlusion_monochrome(ambient_occlusion_color);",
                )
                .replace(
                    "let ambient_occlusion = clamp(surface.occlusion * evaluate_occlusion_uv0(/* fluxel:uv1-argument */input.uv0), 0.0, 1.0);",
                    "let ambient_occlusion = pbr_ambient_occlusion_monochrome(ambient_occlusion_color);",
                )
                .replace(
                    "* clamp(surface.occlusion * evaluate_occlusion_uv0(/* fluxel:uv1-argument */input.uv0), 0.0, 1.0)",
                    "* ambient_occlusion_color.rgb",
                )
                .replace(
                    "* clamp(surface.occlusion * evaluate_occlusion_uv0(/* fluxel:uv1-argument */ input.uv0), 0.0, 1.0)",
                    "* ambient_occlusion_color.rgb",
                )
                .replace("* ambient_occlusion;", "* ambient_occlusion_color.rgb;")
                .replace(
                    "* ambient_occlusion *",
                    "* ambient_occlusion_color.rgb *",
                );
        }
        request.static_options.insert(
            "renderer.ambient_occlusion_color".into(),
            "babylon-rgb-direct-weight-v1".into(),
        );
    }
}

/// Installs the material-owned diffuse-roughness call and routes the selected
/// surface entry through the EON-specific direct-light overload.
pub(super) fn install_diffuse_roughness_evaluator(
    request: &mut ShaderCompositionRequest,
    enabled: bool,
    material_uses_uv0: bool,
) {
    let evaluator = if !enabled {
        "0.0"
    } else if material_uses_uv0 {
        "evaluate_diffuse_roughness_uv0(/* fluxel:uv1-local-argument */uv)"
    } else {
        "evaluate_diffuse_roughness()"
    };
    request.source_units.push(SourceUnit::new(
        DIFFUSE_ROUGHNESS_EVALUATOR,
        format!(
            "fn pbr_diffuse_roughness(/* fluxel:uv1-local-param */uv: vec2<f32>) -> f32 {{ return {evaluator}; }}"
        ),
    ));
    if !enabled {
        return;
    }
    let roughness = if material_uses_uv0 {
        "pbr_diffuse_roughness(/* fluxel:uv1-argument */input.uv0)"
    } else {
        "pbr_diffuse_roughness(vec2<f32>(0.0))"
    };
    for unit in &mut request.source_units {
        if !unit.id.0.starts_with("pbr-forward") {
            continue;
        }
        let source = unit
            .source
            .replace(
                "front_facing: bool) -> vec4<f32> {",
                &format!(
                    "front_facing: bool) -> vec4<f32> {{ let diffuse_roughness = {roughness};"
                ),
            )
            .replace(
                "pbr_direct_light_view(",
                "pbr_direct_light_view_with_diffuse_roughness(",
            )
            .replace(
                "pbr_direct_light_components(",
                "pbr_direct_light_components_with_diffuse_roughness(",
            )
            .replace("vec3<f32>(0.04))", "vec3<f32>(0.04), diffuse_roughness)")
            .replace(
                "pbr_direct_light_components_with_diffuse_roughness(base, properties, normal, view, dielectric_f0);",
                "pbr_direct_light_components_with_diffuse_roughness(base, properties, normal, view, dielectric_f0, diffuse_roughness);",
            )
            .replace(
                "dielectric_f0, 0.0, energy);",
                "dielectric_f0, diffuse_roughness, energy);",
            );
        unit.source = source;
    }
    request.static_options.insert(
        "renderer.diffuse_roughness".into(),
        "babylon-eon-separate-v1".into(),
    );
}

/// Adds the roughness-only Deferred G-buffer attachment. The material
/// preparation step installs either the authored evaluator or a zero writer.
pub(crate) fn with_gbuffer_diffuse_roughness(
    mut request: ShaderCompositionRequest,
    effects: bool,
) -> ShaderCompositionRequest {
    let location = 2 + u32::from(effects);
    let roughness = if request
        .static_options
        .contains_key("renderer.material_context")
    {
        "pbr_diffuse_roughness(/* fluxel:uv1-argument */input.uv0)"
    } else {
        "pbr_diffuse_roughness(vec2<f32>(0.0))"
    };
    for unit in &mut request.source_units {
        if !unit.id.0.starts_with("pbr-gbuffer") {
            continue;
        }
        let source = unit.source.replace(
            "normal_roughness: vec4<f32>,",
            &format!(
                "normal_roughness: vec4<f32>,\n    @location({location}) diffuse_roughness: f32,"
            ),
        );
        unit.source = source
            .replace(
                "if material_alpha_rejected(",
                &format!("let diffuse_roughness = {roughness}; if material_alpha_rejected("),
            )
            .replace(
                "output.normal_roughness =",
                "output.diffuse_roughness = diffuse_roughness; output.normal_roughness =",
            )
            .replace(
                "o.normal_roughness =",
                "o.diffuse_roughness = diffuse_roughness; o.normal_roughness =",
            );
    }
    request.static_options.insert(
        "renderer.gbuffer_diffuse_roughness".into(),
        "r32float-v1".into(),
    );
    request
}

/// Adds the optional Babylon colored-AO target after effects and diffuse
/// roughness. Its RGB is AO and alpha is the direct-light blend weight.
pub(crate) fn with_gbuffer_ambient_occlusion(
    mut request: ShaderCompositionRequest,
    effects: bool,
    diffuse_roughness: bool,
) -> ShaderCompositionRequest {
    let location = 2 + u32::from(effects) + u32::from(diffuse_roughness);
    let uv = if request
        .static_options
        .contains_key("renderer.material_context")
    {
        "/* fluxel:uv1-argument */input.uv0"
    } else {
        "vec2<f32>(0.0)"
    };
    let authored = if request
        .static_options
        .contains_key("renderer.material_context")
    {
        "surface.occlusion"
    } else {
        "1.0"
    };
    for unit in &mut request.source_units {
        if !unit.id.0.starts_with("pbr-gbuffer") {
            continue;
        }
        unit.source = unit
            .source
            .replace(
                "normal_roughness: vec4<f32>,",
                &format!("normal_roughness: vec4<f32>,\n    @location({location}) ambient_occlusion: vec4<f32>,"),
            )
            .replace(
                "if material_alpha_rejected(",
                &format!("let ambient_occlusion = pbr_ambient_occlusion({uv}, {authored}); if material_alpha_rejected("),
            )
            .replace(
                "output.normal_roughness =",
                "output.ambient_occlusion = ambient_occlusion; output.normal_roughness =",
            )
            .replace(
                "o.normal_roughness =",
                "o.ambient_occlusion = ambient_occlusion; o.normal_roughness =",
            )
            .replace(
                "vec4<f32>(emissive, occlusion)",
                "vec4<f32>(emissive, pbr_ambient_occlusion_monochrome(ambient_occlusion))",
            )
            .replace(
                "vec4(emissive, occlusion)",
                "vec4(emissive, pbr_ambient_occlusion_monochrome(ambient_occlusion))",
            )
            .replace(
                "let occlusion = clamp(surface.occlusion * evaluate_occlusion_uv0(/* fluxel:uv1-argument */input.uv0), 0.0, 1.0);",
                "",
            )
            .replace(
                "let occlusion = clamp(surface.occlusion * evaluate_occlusion_uv0(/* fluxel:uv1-argument */ input.uv0), 0.0, 1.0);",
                "",
            );
    }
    request.static_options.insert(
        "renderer.gbuffer_ambient_occlusion".into(),
        "rgba32float-v1".into(),
    );
    request
}

pub(super) fn install_emissive_evaluator(request: &mut ShaderCompositionRequest) {
    request.source_units.push(SourceUnit::new(
        EMISSIVE_EVALUATOR,
        "fn pbr_emissive(/* fluxel:uv1-local-param */uv: vec2<f32>) -> vec3<f32> { return evaluate_emissive_uv0(/* fluxel:uv1-local-argument */uv); }",
    ));
}

/// Removes the material emissive evaluator from an effect-enabled PBR request.
///
/// The material declaration remains intact so that resource ownership stays
/// material-local; the selected fragment route cannot reach it. This mirrors
/// Babylon's global emissive admission without importing its coordinate rules.
pub(crate) fn without_emissive(mut request: ShaderCompositionRequest) -> ShaderCompositionRequest {
    let unit = request
        .source_units
        .iter_mut()
        .find(|unit| unit.id.0 == EMISSIVE_EVALUATOR)
        .expect("surface-effect PBR request includes its emissive evaluator");
    *unit = SourceUnit::new(
        EMISSIVE_EVALUATOR,
        "fn pbr_emissive(/* fluxel:uv1-local-param */_uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0); }",
    );
    request
        .static_options
        .insert("renderer.emissive".into(), "disabled".into());
    request
}

/// Specializes the PBR normal-facing predicate for a reflected camera.
pub(crate) fn with_mirrored_view(
    mut request: ShaderCompositionRequest,
) -> ShaderCompositionRequest {
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
