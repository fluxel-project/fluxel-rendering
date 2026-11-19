//! Renderer-owned PBR request specializations.

use fluxel_shader::{ShaderCompositionRequest, SourceUnit};

pub(super) const VIEW_FACING: &str = "pbr-view-facing";
const EMISSIVE_EVALUATOR: &str = "pbr-emissive";

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
