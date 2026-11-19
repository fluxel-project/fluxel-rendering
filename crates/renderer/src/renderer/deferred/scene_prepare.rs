//! CPU admission shared by single and multiple Deferred scene draws.

use super::*;
use crate::frame_data::forward::{ForwardUniformData, ForwardUniformInput, pack_forward_uniforms};
use crate::{ForwardDrawInput, PreparedMaterial, prepare_material};

pub(super) fn materials(
    inputs: &[ForwardDrawInput],
) -> Result<(Vec<PreparedMaterial>, bool, bool, bool), RendererError> {
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
                    | fluxel_material::SurfaceOutput::MetallicRoughnessUv0Graph { .. }
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
            || input.view().directional_light != first.view().directional_light
            || input.view().mirrored != first.view().mirrored
            || input.view().emissive_enabled != first.view().emissive_enabled
    }) {
        return Err(Error::InvalidRange.into());
    }
    if inputs
        .iter()
        .any(|input| input.mesh_visibility().get() != 1.0)
    {
        return Err(Error::InvalidState.into());
    }
    let effects = inputs
        .iter()
        .any(|input| input.material().value().requires_surface_effects());
    let diffuse_roughness = inputs
        .iter()
        .any(|input| input.prepared().material().uses_diffuse_roughness());
    let mut prepared = Vec::with_capacity(inputs.len());
    for input in inputs {
        let layout = &input.geometry().value().layout;
        let mesh_layout = crate::preparation::mesh_vertex_layout(layout);
        let request = match input.material().value().output() {
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Graph { normal: None, .. }
                if effects =>
            {
                crate::pbr::gbuffer_uv0_effect_request(&mesh_layout)
            }
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. }
            | fluxel_material::SurfaceOutput::MetallicRoughnessUv0Graph {
                normal: Some(_), ..
            } if effects => crate::pbr::gbuffer_uv0_normal_effect_request(&mesh_layout),
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. }
            | fluxel_material::SurfaceOutput::MetallicRoughnessUv0Graph {
                normal: Some(_), ..
            } => crate::pbr::gbuffer_uv0_normal_request(&mesh_layout),
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0 { .. }
            | fluxel_material::SurfaceOutput::MetallicRoughnessUv0Graph { normal: None, .. } => {
                crate::pbr::gbuffer_uv0_request(&mesh_layout)
            }
            fluxel_material::SurfaceOutput::MetallicRoughness { .. } => {
                crate::pbr::gbuffer_request(&mesh_layout)
            }
            fluxel_material::SurfaceOutput::ColorUv0 => {
                super::templates::gbuffer_uv0_request(&mesh_layout)
            }
            fluxel_material::SurfaceOutput::Color => {
                super::templates::gbuffer_request(&mesh_layout)
            }
        }
        .ok_or(Error::Unsupported(
            "Deferred material cannot consume this geometry layout",
        ))?;
        let request = if pbr && input.view().mirrored {
            crate::pbr::with_mirrored_view(request)
        } else {
            request
        };
        let request = if effects && !input.view().emissive_enabled {
            crate::pbr::without_emissive(request)
        } else {
            request
        };
        let request = if diffuse_roughness {
            crate::pbr::with_gbuffer_diffuse_roughness(request, effects)
        } else {
            request
        };
        let request = if input.has_instance_transforms() {
            crate::preparation::with_instance_transform(request, input.source())
                .map_err(RendererError::ForwardPreparation)?
        } else {
            request
        };
        let material = prepare_material(
            input.material(),
            layout,
            &mut fluxel_shader::ShaderComposer::new(),
            request,
        )
        .map_err(RendererError::DeferredMaterial)?;
        let schema = crate::preparation::material_binding_schema(&material, input.source())
            .map_err(RendererError::ForwardPreparation)?;
        if input.prepared().binding_schema() != Some(&schema) {
            return Err(RendererError::ForwardPreparation(
                crate::ForwardPreparationError::MaterialBindingSchema {
                    source: input.source(),
                },
            ));
        }
        prepared.push(material.with_binding_schema(schema));
    }
    Ok((prepared, pbr, effects, diffuse_roughness))
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
                crate::frame_data::forward::UNLIT_FORWARD_TRANSFORM_BYTES
            }
        {
            return Err(Error::InvalidRange);
        }
        values.push(ForwardUniformInput {
            transform: input.clip_from_local(),
            mesh_visibility: None,
            normal_from_local: if pbr {
                Some(input.normal_from_local().ok_or(Error::InvalidRange)?)
            } else {
                None
            },
            world_from_local: pbr.then(|| input.transform().columns()),
            clip_from_world: None,
            material_uniforms: input.uniform_values().to_vec(),
            shading_camera: input.view().shading_camera,
            directional_light: input.view().directional_light,
        });
    }
    pack_forward_uniforms(&values, alignment, max_bytes)
}
