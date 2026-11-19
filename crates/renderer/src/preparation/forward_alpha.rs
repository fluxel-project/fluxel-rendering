//! Babylon PBR alpha admission from validated author and mesh facts.

use fluxel_material::{DynamicValue, MaterialDefinition, MaterialInstance, SurfaceAlphaMode};

use crate::MeshVisibility;

use super::{ForwardDrawSource, ForwardPreparationError};

pub(super) fn resolve(
    material: &MaterialDefinition,
    instance: Option<&MaterialInstance>,
    source: ForwardDrawSource,
) -> Result<Option<SurfaceAlphaMode>, ForwardPreparationError> {
    let Some(automatic) = material.pbr_automatic_alpha() else {
        return Ok(None);
    };
    let opacity = material.pbr_opacity_configuration();
    let alpha = if let Some(parameter) = opacity.and_then(|value| value.alpha_parameter.as_ref()) {
        match instance.and_then(|instance| instance.parameter(parameter)) {
            Some(DynamicValue::F32(value)) if value.is_finite() => *value,
            _ => {
                return Err(ForwardPreparationError::InvalidMaterialParameter {
                    source,
                    parameter: parameter.clone(),
                });
            }
        }
    } else {
        1.0
    };
    let opacity_texture = opacity.is_some_and(|value| value.texture.is_some());
    let mode = if alpha < 1.0
        || opacity_texture
        || (automatic.base_texture_has_alpha && automatic.use_alpha_from_base_texture)
    {
        SurfaceAlphaMode::Blend
    } else if automatic.base_texture_has_alpha {
        SurfaceAlphaMode::Mask
    } else {
        SurfaceAlphaMode::Opaque
    };
    Ok(Some(mode))
}

pub(super) fn with_visibility(
    mode: Option<SurfaceAlphaMode>,
    visibility: MeshVisibility,
) -> Option<SurfaceAlphaMode> {
    mode.map(|mode| {
        if visibility.get() < 1.0 {
            SurfaceAlphaMode::Blend
        } else {
            mode
        }
    })
}
