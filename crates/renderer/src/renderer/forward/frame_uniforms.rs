//! Packs the material and per-draw transform uniforms for Forward preparation.

use fluxel_rhi::api::{DeviceLimits, Result};

use crate::{
    ForwardDrawInput,
    frame_data::forward::{
        ForwardUniformData, ForwardUniformInput, pack_forward_uniforms,
        pack_forward_uniforms_with_particle,
    },
};

pub(super) fn pack_inputs(
    inputs: &[ForwardDrawInput],
    limits: DeviceLimits,
) -> Result<ForwardUniformData> {
    let uniform_inputs: Vec<_> = inputs
        .iter()
        .map(|input| ForwardUniformInput {
            transform: input.clip_from_local(),
            mesh_visibility: Some(input.mesh_visibility()),
            normal_from_local: input.normal_from_local(),
            world_from_local: input
                .normal_from_local()
                .map(|_| input.transform().columns()),
            clip_from_world: input
                .prepared()
                .uses_scene_refraction()
                .then(|| input.view().transform.columns()),
            material_uniforms: input.uniform_values().to_vec(),
            shading_camera: input.view().shading_camera,
            directional_light: input.view().directional_light,
        })
        .collect();
    if inputs.iter().any(|input| input.particle().is_some()) {
        let transforms: Vec<_> = inputs
            .iter()
            .map(|input| input.particle().map(|particle| particle.transform()))
            .collect();
        pack_forward_uniforms_with_particle(
            &uniform_inputs,
            &transforms,
            limits.uniform_offset_alignment,
            limits.max_buffer_bytes,
        )
    } else {
        pack_forward_uniforms(
            &uniform_inputs,
            limits.uniform_offset_alignment,
            limits.max_buffer_bytes,
        )
    }
}
