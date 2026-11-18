//! Shared surface request selection for ordinary and capture Forward draws.

use fluxel_material::{MaterialDefinition, SurfaceOutput};
use fluxel_shader::ShaderCompositionRequest;

use super::{
    ForwardDrawSource, ForwardPreparationError, GeometryLayout, RenderView, color_uv0_template,
    forward_template,
};

pub(super) fn material_request(
    material: &MaterialDefinition,
    geometry: &GeometryLayout,
    view: &RenderView,
    source: ForwardDrawSource,
    environment: bool,
    specular: bool,
    scene_color: Option<(&str, &str)>,
) -> Result<ShaderCompositionRequest, ForwardPreparationError> {
    let has_effects = material.requires_surface_effects();
    let transmission = material.transmission_extensions().is_some();
    let request = match material.output() {
        SurfaceOutput::Color => forward_template(geometry, source)?.request(),
        SurfaceOutput::ColorUv0 => color_uv0_template(geometry, source)?.request(),
        SurfaceOutput::MetallicRoughness { .. } => (if environment {
            crate::pbr::forward_irradiance_request(geometry)
        } else {
            crate::pbr::forward_request(geometry)
        })
        .ok_or(ForwardPreparationError::MissingNormal { source })?,
        SurfaceOutput::MetallicRoughnessUv0 { .. } => (if environment {
            crate::pbr::forward_uv0_irradiance_request(geometry)
        } else {
            crate::pbr::forward_uv0_request(geometry)
        })
        .ok_or(ForwardPreparationError::MissingUv0 { source })?,
        SurfaceOutput::MetallicRoughnessUv0Normal { .. } if has_effects => (if environment {
            crate::pbr::forward_uv0_normal_effect_irradiance_request(geometry)
        } else {
            crate::pbr::forward_uv0_normal_effect_request(geometry)
        })
        .ok_or(ForwardPreparationError::MissingNormal { source })?,
        SurfaceOutput::MetallicRoughnessUv0Normal { .. } => (if environment {
            crate::pbr::forward_uv0_normal_irradiance_request(geometry)
        } else {
            crate::pbr::forward_uv0_normal_request(geometry)
        })
        .ok_or(ForwardPreparationError::MissingNormal { source })?,
    };
    if transmission {
        if !specular {
            return Err(
                ForwardPreparationError::TransmissionRequiresSpecularEnvironment { source },
            );
        }
        if view.shading_camera.is_none() {
            return Err(ForwardPreparationError::InvalidView);
        }
        Ok(match scene_color {
            Some((texture, sampler)) => {
                crate::pbr::specular::forward_scene_transmission_request(request, texture, sampler)
            }
            None => crate::pbr::specular::forward_transmission_request(request),
        })
    } else if specular && crate::pbr::is_pbr(material.output()) {
        if view.shading_camera.is_none() {
            return Err(ForwardPreparationError::InvalidView);
        }
        Ok(crate::pbr::specular::forward_specular_request(request))
    } else {
        Ok(request)
    }
}
