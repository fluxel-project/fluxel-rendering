//! Per-draw concrete G-buffer bindings over shared frame ownership.

use super::program::GbufferProgram;
use super::*;
use crate::frame_data::forward::ForwardUniformData;
use crate::residency::{DeferredDraw, ForwardResidentBuffers};
use crate::{ForwardDrawInput, PreparedMaterial};
use fluxel_rhi::api::{BindGroupDesc, BindGroupEntry, BindingResource, Buffer, Device};
use std::sync::Arc;

pub(super) fn draws<B: ResourceBackend>(
    device: &Device<B>,
    inputs: &[ForwardDrawInput],
    prepared: &[PreparedMaterial],
    programs: &[Arc<GbufferProgram<B>>],
    resident: &ForwardResidentBuffers<B>,
    uniform: &Buffer<B>,
    values: &ForwardUniformData,
    mappings: &[Vec<usize>],
    textures: &[Vec<usize>],
    capture_textures: &[Vec<bool>],
    pbr: bool,
) -> Result<Vec<DeferredDraw<B>>, Error> {
    let count = inputs.len();
    if [
        prepared.len(),
        programs.len(),
        mappings.len(),
        textures.len(),
        capture_textures.len(),
        values.transforms.len(),
        values.tints.len(),
    ]
    .iter()
    .any(|length| *length != count)
    {
        return Err(Error::InvalidState);
    }
    let transform_bytes = if pbr {
        crate::frame_data::forward::PBR_TRANSFORM_BYTES
    } else {
        64
    };
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let schema = prepared[index]
            .binding_schema()
            .ok_or(Error::InvalidState)?;
        if schema.textures.len() != textures[index].len()
            || capture_textures[index].len() != textures[index].len()
            || capture_textures[index]
                .iter()
                .enumerate()
                .any(|(slot, &capture)| {
                    capture != inputs[index].capture_texture_slot(slot).is_some()
                })
        {
            return Err(Error::InvalidRange);
        }
        let mut entries = vec![BindGroupEntry {
            binding: schema.tint,
            resource: BindingResource::Buffer {
                buffer: uniform,
                range: 0..16,
            },
        }];
        for (texture_index, binding) in schema.textures.iter().enumerate() {
            let slot = textures[index][texture_index];
            let texture = resident.textures.get(slot).ok_or(Error::InvalidRange)?;
            crate::renderer::forward::validate_material_texture_owner(
                resident,
                slot,
                device.id(),
                &inputs[index],
                texture_index,
                crate::renderer::forward::material_requires_ordinary_sampler(&prepared[index]),
            )?;
            entries.push(BindGroupEntry {
                binding: binding.binding,
                resource: BindingResource::TextureView(&texture.view),
            });
        }
        if let Some(binding) = schema.sampler {
            let expected = crate::renderer::forward::material_sampler_for_input(&inputs[index])?;
            let sampler = resident
                .samplers
                .iter()
                .find(|sampler| {
                    sampler.sampler.device() == device.id()
                        && crate::renderer::forward::same_material_sampler(&sampler.desc, &expected)
                })
                .ok_or(Error::InvalidState)?;
            crate::renderer::forward::validate_material_sampler(&sampler.desc)?;
            entries.push(BindGroupEntry {
                binding,
                resource: BindingResource::Sampler(&sampler.sampler),
            });
        }
        entries.sort_by_key(|entry| entry.binding);
        let material_group = device.create_bind_group(&BindGroupDesc {
            label: Some("Deferred G-buffer material"),
            layout: &programs[index].material_layout,
            entries: &entries,
        })?;
        let transform_group = device.create_bind_group(&BindGroupDesc {
            label: Some("Deferred G-buffer transform"),
            layout: &programs[index].transform_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: uniform,
                    range: 0..transform_bytes,
                },
            }],
        })?;
        let ranges = mappings[index]
            .iter()
            .map(|slot| {
                resident
                    .streams
                    .get(*slot)
                    .map(|stream| 0..stream.desc.size)
                    .ok_or(Error::InvalidRange)
            })
            .collect::<Result<Vec<_>, _>>()?;
        result.push(DeferredDraw {
            pipeline: programs[index].pipeline.clone(),
            material_group,
            transform_group,
            vertex_layouts: prepared[index].geometry().streams.clone(),
            vertex_ranges: ranges,
            vertex_streams: mappings[index].clone(),
            textures: textures[index].clone(),
            capture_textures: capture_textures[index].clone(),
            vertices: inputs[index].geometry().value().vertices.clone(),
            tint_range: values.tints[index].clone(),
            transform_range: values.transforms[index].clone(),
            transform_bytes,
        });
    }
    Ok(result)
}
