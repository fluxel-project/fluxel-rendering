//! Recomposition of retained Forward inputs for actual OIT material stages.

use super::*;
use crate::preparation::forward_fragment::{
    OitDepthPrecision, OitDrawStage, with_oit_forward_fragment,
};
use fluxel_material::SurfaceAlphaMode;

/// One stable OIT partition of already prepared Forward draws.
///
/// `draws` is opaque/mask prefix, peelable blend middle, then the explicitly
/// excluded blend suffix. The renderer still owns all phase declaration and
/// recording; this carries no Graph or backend authority.
pub(crate) struct OitDrawPartition {
    pub(crate) draws: Vec<ForwardDrawInput>,
    pub(crate) opaque_count: usize,
    pub(crate) peel_count: usize,
}

/// Stably partitions one received Forward draw order for an admitted OIT frame.
///
/// Each segment preserves its received order. The caller retains duplicate
/// validation and any ordinary fallback sorting; this function deliberately
/// does not reimplement that scheduler or turn unavailable OIT into a fallback.
pub(crate) fn partition_oit_draws(
    draws: Vec<ForwardDrawInput>,
    config: &crate::DualDepthPeelingConfig,
) -> OitDrawPartition {
    let mut opaque = Vec::with_capacity(draws.len());
    let mut peel = Vec::new();
    let mut excluded = Vec::new();
    for draw in draws {
        match draw.prepared().material().surface_policy().alpha_mode {
            SurfaceAlphaMode::Opaque | SurfaceAlphaMode::Mask => opaque.push(draw),
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
                if config.excludes_source(draw.source()) =>
            {
                excluded.push(draw);
            }
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend => peel.push(draw),
        }
    }
    let opaque_count = opaque.len();
    let peel_count = peel.len();
    opaque.extend(peel);
    opaque.extend(excluded);
    OitDrawPartition {
        draws: opaque,
        opaque_count,
        peel_count,
    }
}

/// Rebuilds the exact immutable material and geometry against an OIT fragment
/// entry. The caller supplies precision admitted for its actual targets; this
/// function only validates the complete logical program and material schema.
pub(crate) fn prepare_oit_material(
    input: &ForwardDrawInput,
    stage: OitDrawStage,
    precision: OitDepthPrecision,
    composer: &mut ShaderComposer,
) -> Result<PreparedMaterial, ForwardPreparationError> {
    let request = material_request(
        input.material().value(),
        &input.geometry().value().layout,
        input.view(),
        input.source(),
        input.prepared().uses_environment_irradiance(),
        input.prepared().uses_specular_ibl(),
    )?;
    let request = with_oit_forward_fragment(
        request,
        stage,
        precision,
        input
            .prepared()
            .material()
            .surface_policy()
            .two_sided_lighting,
    )
    .ok_or(ForwardPreparationError::MaterialBindingSchema {
        source: input.source(),
    })?;
    let prepared = prepare_material(
        input.material(),
        &input.geometry().value().layout,
        composer,
        request,
    )
    .map_err(ForwardPreparationError::Material)?;
    let schema = super::super::material_binding_schema(&prepared, input.source())?;
    if input.prepared().binding_schema() != Some(&schema)
        || prepared.material().key() != input.prepared().material().key()
        || prepared.uses_environment_irradiance() != input.prepared().uses_environment_irradiance()
        || prepared.uses_specular_ibl() != input.prepared().uses_specular_ibl()
        || prepared.uses_transmission() != input.prepared().uses_transmission()
    {
        return Err(ForwardPreparationError::MaterialBindingSchema {
            source: input.source(),
        });
    }
    Ok(prepared.with_binding_schema(schema))
}
