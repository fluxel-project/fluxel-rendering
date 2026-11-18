//! CPU admission shared by single and multiple Deferred scene draws.

use super::*;
use crate::frame_data::forward::{ForwardUniformData, ForwardUniformInput, pack_forward_uniforms};
use crate::{ForwardDrawInput, PreparedMaterial, prepare_material};

pub(super) fn materials(
    inputs: &[ForwardDrawInput],
) -> Result<(Vec<PreparedMaterial>, bool, bool), RendererError> {
    let first = inputs
        .first()
        .ok_or(Error::Unsupported("Deferred requires visible geometry"))?;
    if inputs
        .iter()
        .any(|input| input.material().value().transmission_extensions().is_some())
    {
        return Err(Error::Unsupported(
            "Deferred transmission requires the transmission composition pass",
        )
        .into());
    }
    let pbr = crate::pbr::is_pbr(first.material().value().output());
    if inputs.len() > 1
        && inputs.iter().any(|input| {
            !matches!(
                input.material().value().output(),
                fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. }
            )
        })
    {
        return Err(
            Error::Unsupported("multiple Deferred draws require the UV0-normal MR family").into(),
        );
    }
    if inputs.iter().any(|input| {
        input.extent() != first.extent()
            || input.view().transform.columns() != first.view().transform.columns()
            || input.view().shading_camera != first.view().shading_camera
    }) {
        return Err(Error::InvalidRange.into());
    }
    let effects = inputs.iter().any(|input| {
        let extensions = input.material().value().surface_extensions();
        extensions.occlusion.is_some() || extensions.emissive.is_some()
    });
    let mut prepared = Vec::with_capacity(inputs.len());
    for input in inputs {
        let layout = &input.geometry().value().layout;
        let request = match input.material().value().output() {
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. } if effects => {
                crate::pbr::gbuffer_uv0_normal_effect_request(layout)
            }
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. } => {
                crate::pbr::gbuffer_uv0_normal_request(layout)
            }
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0 { .. } => {
                crate::pbr::gbuffer_uv0_request(layout)
            }
            fluxel_material::SurfaceOutput::MetallicRoughness { .. } => {
                crate::pbr::gbuffer_request(layout)
            }
            fluxel_material::SurfaceOutput::ColorUv0 => {
                super::templates::gbuffer_uv0_request(layout)
            }
            fluxel_material::SurfaceOutput::Color => super::templates::gbuffer_request(layout),
        }
        .ok_or(Error::Unsupported(
            "Deferred material cannot consume this geometry layout",
        ))?;
        let material = prepare_material(
            input.material(),
            layout,
            &mut fluxel_shader::ShaderComposer::new(),
            request,
        )
        .map_err(RendererError::DeferredMaterial)?;
        let schema = crate::preparation::material_binding_schema(&material, input.source())
            .map_err(RendererError::ForwardPreparation)?;
        prepared.push(material.with_binding_schema(schema));
    }
    Ok((prepared, pbr, effects))
}

pub(super) fn uniforms(
    inputs: &[ForwardDrawInput],
    pbr: bool,
    alignment: u64,
    max_bytes: u64,
) -> Result<ForwardUniformData, Error> {
    let mut values = Vec::with_capacity(inputs.len());
    for input in inputs {
        if input.transform_uniform_bytes()
            != if pbr {
                crate::frame_data::forward::PBR_TRANSFORM_BYTES
            } else {
                64
            }
        {
            return Err(Error::InvalidRange);
        }
        values.push(ForwardUniformInput {
            transform: input.clip_from_local(),
            normal_from_local: if pbr {
                Some(input.normal_from_local().ok_or(Error::InvalidRange)?)
            } else {
                None
            },
            world_from_local: pbr.then(|| input.transform().columns()),
            clip_from_world: pbr.then(|| input.view().transform.columns()),
            tint: input.tint(),
            shading_camera: input.view().shading_camera,
        });
    }
    pack_forward_uniforms(&values, alignment, max_bytes)
}
