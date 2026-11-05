//! Surface-only owner assembly for the Deferred path.

use std::sync::Arc;

use fluxel_rhi::api::{
    BindGroupDesc, BindGroupEntry, BindingResource, Error, Support, TextureFormat,
};

use crate::frame_data::forward::{ForwardUniformInput, pack_forward_uniforms};
use crate::residency::{
    DeferredDraw, DeferredSurfaceFrame, ForwardResidentBuffers, ForwardStreamInput,
    ForwardTextureInput, SurfaceForwardFrameBuffers, plan_deferred_gbuffer, plan_forward_resident,
    plan_surface_forward_frame,
};
use crate::{ForwardDrawInput, PreparedMaterial, prepare_material};

use super::prepare::aligned_row;
use super::templates::gbuffer_request;
use super::*;

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
    pub(super) async fn prepare_deferred_surface_frame(
        &mut self,
        input: &ForwardDrawInput,
        color_format: TextureFormat,
    ) -> std::result::Result<DeferredSurfaceFrame<B>, RendererError> {
        self.ensure_open()?;
        let pbr = crate::pbr::is_pbr(input.material().value().output());
        let request = if pbr {
            crate::pbr::gbuffer_request(&input.geometry().value().layout)
        } else {
            gbuffer_request(&input.geometry().value().layout)
        }
        .ok_or(Error::Unsupported(
            "Deferred material route cannot consume this geometry layout",
        ))?;
        let prepared = prepare_material(
            input.material(),
            &input.geometry().value().layout,
            &mut fluxel_shader::ShaderComposer::new(),
            request,
        )
        .map_err(RendererError::DeferredMaterial)?;
        self.prepare_deferred_surface_resources(input, &prepared, pbr, color_format)
            .await
    }

    async fn prepare_deferred_surface_resources(
        &mut self,
        input: &ForwardDrawInput,
        prepared: &PreparedMaterial,
        pbr: bool,
        color_format: TextureFormat,
    ) -> std::result::Result<DeferredSurfaceFrame<B>, RendererError> {
        let limits = self.device.capabilities().limits;
        let transform_bytes = if pbr { 128 } else { 64 };
        if input.transform_uniform_bytes() != transform_bytes {
            return Err(Error::InvalidRange.into());
        }
        let normal_from_local = if pbr {
            Some(input.normal_from_local().ok_or(Error::InvalidRange)?)
        } else {
            None
        };
        let uniforms = pack_forward_uniforms(
            &[ForwardUniformInput {
                transform: input.clip_from_local(),
                normal_from_local,
                tint: input.tint(),
            }],
            limits.uniform_offset_alignment,
            limits.max_buffer_bytes,
        )?;
        let uniform_len = u64::try_from(uniforms.bytes.len()).map_err(|_| Error::InvalidRange)?;
        let texture = input.texture().value();
        let texture_pitch = aligned_row(texture.extent[0])?;
        let texture_bytes = u64::from(texture_pitch)
            .checked_mul(u64::from(texture.extent[1]))
            .ok_or(Error::InvalidRange)?;
        if texture_bytes > limits.max_buffer_bytes {
            return Err(Error::InvalidRange.into());
        }
        let mut padded_texture =
            vec![0; usize::try_from(texture_bytes).map_err(|_| Error::InvalidRange)?];
        let tight = usize::try_from(texture.extent[0])
            .map_err(|_| Error::InvalidRange)?
            .checked_mul(4)
            .ok_or(Error::InvalidRange)?;
        for (row, source) in texture.rgba8.chunks_exact(tight).enumerate() {
            let start = row
                .checked_mul(usize::try_from(texture_pitch).map_err(|_| Error::InvalidRange)?)
                .ok_or(Error::InvalidRange)?;
            padded_texture[start..start + tight].copy_from_slice(source);
        }

        let geometry_key =
            crate::residency::ForwardResidentCache::<B>::geometry_key(input, self.device.id());
        let texture_key =
            crate::residency::ForwardResidentCache::<B>::texture_key(input, self.device.id());
        let cached_geometry = self.deferred_residency.geometry(geometry_key);
        let cached_texture = self.deferred_residency.texture(texture_key);
        if cached_geometry.is_some() {
            self.deferred_stats.geometry_hits = self.deferred_stats.geometry_hits.saturating_add(1);
        }
        if cached_texture.is_some() {
            self.deferred_stats.texture_hits = self.deferred_stats.texture_hits.saturating_add(1);
        }
        let resident_plan = plan_forward_resident(
            if cached_geometry.is_some() {
                Vec::new()
            } else {
                input
                    .geometry()
                    .value()
                    .streams
                    .iter()
                    .zip(&prepared.geometry().streams)
                    .map(|(bytes, layout)| ForwardStreamInput {
                        bytes: bytes.clone(),
                        layout: layout.clone(),
                    })
                    .collect()
            },
            if cached_texture.is_some() {
                Vec::new()
            } else {
                vec![ForwardTextureInput {
                    bytes: padded_texture,
                    extent: texture.extent,
                    bytes_per_row: texture_pitch,
                }]
            },
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        let frame_plan = plan_surface_forward_frame(
            uniform_len,
            uniform_len,
            0..uniform_len,
            limits.uniform_offset_alignment,
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        let gbuffer_plan = plan_deferred_gbuffer(&self.device, input.extent())
            .map_err(|error| self.observe_rhi_error(error))?;
        for desc in resident_plan
            .stream_descs
            .iter()
            .chain(std::iter::once(&resident_plan.upload_desc))
            .chain([&frame_plan.upload_desc, &frame_plan.uniform_desc])
        {
            if let Support::Unsupported(reason) = self.device.buffer_support(desc) {
                return Err(self.observe_rhi_error(Error::Unsupported(reason)));
            }
        }
        for desc in &resident_plan.texture_descs {
            if let Support::Unsupported(reason) = self.device.texture_support(desc) {
                return Err(self.observe_rhi_error(Error::Unsupported(reason)));
            }
        }
        let geometry_program = self.deferred_gbuffer_program(prepared, pbr).await?;
        let lighting_program = self.deferred_lighting_program(pbr, color_format).await?;
        let resident_buffers = resident_plan.stream_descs.len() as u64 + 1;
        let resident_textures = resident_plan.texture_descs.len() as u64;
        let (resident, mut resident_upload) =
            ForwardResidentBuffers::allocate_planned(&self.device, resident_plan)
                .map_err(|error| self.observe_rhi_error(error))?;
        let mut resident = Arc::try_unwrap(resident).map_err(|_| Error::InvalidState)?;
        if let Some(streams) = cached_geometry {
            resident.streams = streams;
        }
        if let Some(texture) = cached_texture {
            resident.textures = vec![texture];
        }
        if resident_upload.vertices.is_empty() && resident_upload.textures.is_empty() {
            resident_upload.bytes.clear();
        }
        let resident = Arc::new(resident);
        let (frame, uniform_upload) =
            SurfaceForwardFrameBuffers::allocate_planned(&self.device, frame_plan)
                .map_err(|error| self.observe_rhi_error(error))?;
        let gbuffer = gbuffer_plan
            .allocate(&self.device)
            .map_err(|error| self.observe_rhi_error(error))?;
        let material_group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Deferred surface G-buffer material"),
                layout: &geometry_program.material_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: &frame.uniform,
                            range: 0..16,
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&resident.textures[0].view),
                    },
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let transform_group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Deferred surface G-buffer transform"),
                layout: &geometry_program.transform_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &frame.uniform,
                        range: 0..transform_bytes,
                    },
                }],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let lighting_group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Deferred surface lighting G-buffer"),
                layout: &lighting_program.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&gbuffer[0].view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&gbuffer[1].view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&gbuffer[2].view),
                    },
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let draw = DeferredDraw {
            pipeline: geometry_program.pipeline.clone(),
            material_group,
            transform_group,
            vertex_layouts: prepared.geometry().streams.clone(),
            vertex_ranges: resident
                .streams
                .iter()
                .map(|stream| 0..stream.desc.size)
                .collect(),
            vertices: input.geometry().value().vertices.clone(),
            tint_range: uniforms.tints[0].clone(),
            transform_range: uniforms.transforms[0].clone(),
            transform_bytes,
        };
        self.deferred_stats.buffer_allocations = self
            .deferred_stats
            .buffer_allocations
            .saturating_add(2 + resident_buffers);
        self.deferred_stats.texture_allocations = self
            .deferred_stats
            .texture_allocations
            .saturating_add(3 + resident_textures);
        self.deferred_stats.resident_upload_bytes = self
            .deferred_stats
            .resident_upload_bytes
            .saturating_add(resident_upload.bytes.len() as u64);
        self.deferred_stats.frame_upload_bytes = self
            .deferred_stats
            .frame_upload_bytes
            .saturating_add(uniform_len);
        Ok(DeferredSurfaceFrame {
            geometry_key: Some(geometry_key),
            texture_key: Some(texture_key),
            deformation: None,
            resident,
            resident_upload,
            frame,
            uniform_upload,
            uniform_bytes: uniforms.bytes,
            gbuffer,
            draw,
            lighting_pipeline: lighting_program.pipeline.clone(),
            lighting_group,
            extent: input.extent(),
            color_format,
        })
    }
}
