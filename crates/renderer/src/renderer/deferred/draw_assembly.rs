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
    textures: &[usize],
    pbr: bool,
) -> Result<Vec<DeferredDraw<B>>, Error> {
    let count = inputs.len();
    if [
        prepared.len(),
        programs.len(),
        mappings.len(),
        textures.len(),
        values.transforms.len(),
        values.tints.len(),
    ]
    .iter()
    .any(|length| *length != count)
    {
        return Err(Error::InvalidState);
    }
    let transform_bytes = if pbr { 192 } else { 64 };
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let texture = resident
            .textures
            .get(textures[index])
            .ok_or(Error::InvalidRange)?;
        let mut entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: uniform,
                    range: 0..16,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&texture.view),
            },
        ];
        crate::renderer::forward::validate_material_texture_owner(
            resident,
            textures[index],
            device.id(),
            &inputs[index],
            crate::renderer::forward::material_requires_ordinary_sampler(&prepared[index]),
        )?;
        if crate::renderer::forward::material_requires_ordinary_sampler(&prepared[index]) {
            let sampler = resident.samplers.first().ok_or(Error::InvalidState)?;
            crate::renderer::forward::validate_ordinary_sampler(&sampler.desc)?;
            entries.push(BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler.sampler),
            });
        }
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
            texture: textures[index],
            vertices: inputs[index].geometry().value().vertices.clone(),
            tint_range: values.tints[index].clone(),
            transform_range: values.transforms[index].clone(),
            transform_bytes,
        });
    }
    Ok(result)
}
