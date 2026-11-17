//! Deduplicated actual resident ownership for one Deferred frame.

use super::*;
use crate::residency::{
    DeferredPublication, ForwardResidentBuffers, ForwardResidentCache, ForwardResidentUpload,
    ForwardSampler, ForwardStreamInput, ForwardTexture, ForwardTextureInput, ForwardVertexStream,
    plan_forward_resident,
};
use crate::{ForwardDrawInput, PreparedMaterial};
use fluxel_rhi::api::Support;
use std::{collections::HashMap, sync::Arc};

pub(super) struct ResidentAssembly<B: ResourceBackend> {
    pub resident: Arc<ForwardResidentBuffers<B>>,
    pub upload: ForwardResidentUpload,
    pub publication: DeferredPublication,
    pub mappings: Vec<Vec<usize>>,
    pub textures: Vec<Vec<usize>>,
    pub buffer_allocations: u64,
    pub texture_allocations: u64,
}

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    pub(super) fn assemble_deferred_residents(
        &mut self,
        inputs: &[ForwardDrawInput],
        prepared: &[PreparedMaterial],
    ) -> Result<ResidentAssembly<B>, RendererError> {
        if inputs.len() != prepared.len() {
            return Err(Error::InvalidState.into());
        }
        let mut geometries = HashMap::new();
        let mut textures = HashMap::new();
        let mut final_streams: Vec<Option<ForwardVertexStream<B>>> = Vec::new();
        let mut final_textures: Vec<Option<ForwardTexture<B>>> = Vec::new();
        let mut fresh_stream_slots = Vec::new();
        let mut fresh_texture_slots = Vec::new();
        let mut stream_inputs = Vec::new();
        let mut texture_inputs = Vec::new();
        let mut mappings = Vec::new();
        let mut draw_textures = Vec::new();
        for (input, material) in inputs.iter().zip(prepared) {
            let geometry_key = ForwardResidentCache::<B>::geometry_key(input, self.device.id());
            let mapping = if let Some(mapping) = geometries.get(&geometry_key) {
                Vec::<usize>::clone(mapping)
            } else {
                let mut mapping = Vec::new();
                if let Some(cached) = self.deferred_residency.geometry(geometry_key) {
                    self.deferred_stats.geometry_hits =
                        self.deferred_stats.geometry_hits.saturating_add(1);
                    for stream in cached {
                        mapping.push(final_streams.len());
                        final_streams.push(Some(stream));
                    }
                } else {
                    for (bytes, layout) in input
                        .geometry()
                        .value()
                        .streams
                        .iter()
                        .zip(&material.geometry().streams)
                    {
                        let index = final_streams.len();
                        mapping.push(index);
                        fresh_stream_slots.push(index);
                        final_streams.push(None);
                        stream_inputs.push(ForwardStreamInput {
                            bytes: bytes.clone(),
                            layout: layout.clone(),
                        });
                    }
                }
                geometries.insert(geometry_key, mapping.clone());
                mapping
            };
            mappings.push(mapping);
            let mut draw_slots = Vec::with_capacity(
                material
                    .binding_schema()
                    .ok_or(Error::InvalidState)?
                    .textures
                    .len(),
            );
            for texture_index in 0..input.resolved_textures().len() {
                let texture_key =
                    ForwardResidentCache::<B>::texture_key(input, texture_index, self.device.id())
                        .ok_or(Error::InvalidState)?;
                let data = input
                    .asset_texture(texture_index)
                    .ok_or(Error::InvalidState)?
                    .value();
                if crate::renderer::forward::material_requires_ordinary_sampler(material) {
                    crate::renderer::forward::require_filterable_material_texture_with_mip_levels(
                        &self.device,
                        data.extent,
                        data.encoding,
                        data.mip_levels,
                    )
                    .map_err(|error| self.observe_rhi_error(error))?;
                }
                let slot = if let Some(slot) = textures.get(&texture_key) {
                    *slot
                } else {
                    let slot = final_textures.len();
                    if let Some(cached) = self.deferred_residency.texture(texture_key) {
                        self.deferred_stats.texture_hits =
                            self.deferred_stats.texture_hits.saturating_add(1);
                        final_textures.push(Some(cached));
                    } else {
                        let pitch = super::prepare::aligned_row(data.extent[0])?;
                        let size = u64::from(pitch)
                            .checked_mul(u64::from(data.extent[1]))
                            .ok_or(Error::InvalidRange)?;
                        if size > self.device.capabilities().limits.max_buffer_bytes {
                            return Err(Error::InvalidRange.into());
                        }
                        let mut bytes =
                            vec![0; usize::try_from(size).map_err(|_| Error::InvalidRange)?];
                        let tight = usize::try_from(data.extent[0])
                            .map_err(|_| Error::InvalidRange)?
                            .checked_mul(4)
                            .ok_or(Error::InvalidRange)?;
                        for (row, source) in data.rgba8.chunks_exact(tight).enumerate() {
                            let start = row * pitch as usize;
                            bytes[start..start + tight].copy_from_slice(source);
                        }
                        texture_inputs.push(ForwardTextureInput {
                            bytes,
                            extent: data.extent,
                            bytes_per_row: pitch,
                            encoding: data.encoding,
                            mip_levels: data.mip_levels,
                        });
                        fresh_texture_slots.push(slot);
                        final_textures.push(None);
                    }
                    textures.insert(texture_key, slot);
                    slot
                };
                draw_slots.push(slot);
            }
            draw_textures.push(draw_slots);
        }
        let plan = plan_forward_resident(stream_inputs, texture_inputs)?;
        for desc in plan
            .stream_descs
            .iter()
            .chain(std::iter::once(&plan.upload_desc))
        {
            if let Support::Unsupported(reason) = self.device.buffer_support(desc) {
                return Err(Error::Unsupported(reason).into());
            }
        }
        for desc in &plan.texture_descs {
            if let Support::Unsupported(reason) = self.device.texture_support(desc) {
                return Err(Error::Unsupported(reason).into());
            }
        }
        let buffer_allocations = plan.stream_descs.len() as u64 + 1;
        let texture_allocations = plan.texture_descs.len() as u64;
        let (resident, mut upload) = ForwardResidentBuffers::allocate_planned(&self.device, plan)
            .map_err(|error| self.observe_rhi_error(error))?;
        let mut resident = Arc::try_unwrap(resident).map_err(|_| Error::InvalidState)?;
        for (index, stream) in std::mem::take(&mut resident.streams)
            .into_iter()
            .enumerate()
        {
            final_streams[fresh_stream_slots[index]] = Some(stream);
        }
        for (index, texture) in std::mem::take(&mut resident.textures)
            .into_iter()
            .enumerate()
        {
            final_textures[fresh_texture_slots[index]] = Some(texture);
        }
        for copy in &mut upload.vertices {
            copy.stream = fresh_stream_slots[copy.stream];
        }
        for copy in &mut upload.textures {
            copy.texture = fresh_texture_slots[copy.texture];
        }
        resident.streams = final_streams
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidState)?;
        resident.textures = final_textures
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidState)?;
        let mut sampler_descs = Vec::new();
        for (input, material) in inputs.iter().zip(prepared) {
            if !crate::renderer::forward::material_requires_ordinary_sampler(material) {
                continue;
            }
            let desc = crate::renderer::forward::material_sampler_for_input(input)
                .map_err(|error| self.observe_rhi_error(error))?;
            if !sampler_descs
                .iter()
                .any(|present| crate::renderer::forward::same_material_sampler(present, &desc))
            {
                sampler_descs.push(desc);
            }
        }
        for desc in sampler_descs {
            let sampler = self
                .device
                .create_sampler(&desc)
                .map_err(|error| self.observe_rhi_error(error))?;
            resident.samplers.push(ForwardSampler { sampler, desc });
        }
        if upload.vertices.is_empty() && upload.textures.is_empty() {
            upload.bytes.clear();
        }
        Ok(ResidentAssembly {
            resident: Arc::new(resident),
            upload,
            publication: DeferredPublication {
                geometry: geometries.into_iter().collect(),
                textures: textures.into_iter().collect(),
            },
            mappings,
            textures: draw_textures,
            buffer_allocations,
            texture_allocations,
        })
    }
}
