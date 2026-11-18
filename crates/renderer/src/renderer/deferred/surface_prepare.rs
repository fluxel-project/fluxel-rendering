//! Surface-only owner assembly for the Deferred path.

use fluxel_rhi::api::{
    BindGroupDesc, BindGroupEntry, BindingResource, Error, Support, TextureFormat,
};

use crate::residency::{
    DeferredSurfaceFrame, SurfaceForwardFrameBuffers, allocate_deferred_scene_color_desc,
    deferred_scene_color_desc_with_format, plan_deferred_gbuffer_with_effects,
    plan_surface_forward_frame,
};
use crate::{ForwardDrawInput, PreparedMaterial};

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
        self.prepare_deferred_surface_frames(std::slice::from_ref(input), color_format)
            .await
    }

    pub(super) async fn prepare_deferred_surface_frames(
        &mut self,
        inputs: &[ForwardDrawInput],
        color_format: TextureFormat,
    ) -> std::result::Result<DeferredSurfaceFrame<B>, RendererError> {
        self.ensure_open()?;
        crate::renderer::require_opaque_or_mask(inputs)
            .map_err(|error| self.observe_rhi_error(error))?;
        let (prepared, pbr, effects) = super::scene_prepare::materials(inputs)?;
        self.prepare_deferred_surface_resources(inputs, &prepared, pbr, effects, color_format)
            .await
    }

    async fn prepare_deferred_surface_resources(
        &mut self,
        inputs: &[ForwardDrawInput],
        prepared: &[PreparedMaterial],
        pbr: bool,
        effects: bool,
        color_format: TextureFormat,
    ) -> std::result::Result<DeferredSurfaceFrame<B>, RendererError> {
        let input = inputs.first().ok_or(Error::InvalidState)?;
        let specular = if pbr {
            self.environment_specular
                .as_ref()
                .map(|snapshot| {
                    crate::renderer::lighting::allocate_specular(
                        &self.device,
                        snapshot,
                        input.view(),
                    )
                })
                .transpose()
                .map_err(|error| self.observe_rhi_error(error))?
        } else {
            None
        };
        let limits = self.device.capabilities().limits;
        let ordinary_sampler = prepared
            .iter()
            .any(crate::renderer::forward::material_requires_ordinary_sampler);
        let lighting_bindings = 3 + u32::from(effects) + u32::from(pbr && specular.is_none());
        if limits.max_bindings_per_group < lighting_bindings {
            return Err(self.observe_rhi_error(Error::Unsupported("Deferred lighting bindings")));
        }
        if ordinary_sampler && limits.max_bindings_per_group < 3 {
            return Err(
                self.observe_rhi_error(Error::Unsupported("three Deferred material bindings"))
            );
        }
        let mut uniforms = super::scene_prepare::uniforms(
            inputs,
            pbr,
            limits.uniform_offset_alignment,
            limits.max_buffer_bytes,
        )?;
        let lighting_view = if pbr && specular.is_none() {
            Some(
                crate::frame_data::specular_view::append_basic_deferred_view(
                    &mut uniforms.bytes,
                    input.view(),
                    limits.max_buffer_bytes,
                )?,
            )
        } else {
            None
        };
        let uniform_len = u64::try_from(uniforms.bytes.len()).map_err(|_| Error::InvalidRange)?;
        let frame_plan = plan_surface_forward_frame(
            uniform_len,
            uniform_len,
            0..uniform_len,
            limits.uniform_offset_alignment,
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        let gbuffer_plan =
            plan_deferred_gbuffer_with_effects(&self.device, input.extent(), pbr, effects)
                .map_err(|error| self.observe_rhi_error(error))?;
        for desc in [&frame_plan.upload_desc, &frame_plan.uniform_desc] {
            if let Support::Unsupported(reason) = self.device.buffer_support(desc) {
                return Err(self.observe_rhi_error(Error::Unsupported(reason)));
            }
        }
        self.retain_deferred_gbuffer_programs(inputs, prepared, pbr, effects);
        let mut geometry_programs = Vec::with_capacity(prepared.len());
        for (input, material) in inputs.iter().zip(prepared) {
            geometry_programs.push(
                self.deferred_gbuffer_program(
                    material,
                    pbr,
                    effects,
                    crate::renderer::raster_surface_state(input),
                )
                .await?,
            );
        }
        let environment = if pbr {
            self.environment_irradiance
                .as_ref()
                .map(|snapshot| {
                    crate::residency::EnvironmentFrameOwner::allocate(
                        &self.device,
                        snapshot.value(),
                    )
                })
                .transpose()
                .map_err(|error| self.observe_rhi_error(error))?
        } else {
            None
        };
        let scene_color_desc = deferred_scene_color_desc_with_format(
            input.extent(),
            self.deferred_scene_color_format(),
        );
        self.admit_deferred_scene_color(&scene_color_desc)?;
        let lighting_program = self
            .deferred_lighting_program(
                pbr,
                environment.is_some(),
                specular.is_some(),
                effects,
                scene_color_desc.format,
            )
            .await?;
        let composite_program = self
            .scene_color_composite_program(color_format, scene_color_desc.format)
            .await?;
        let super::resident_assembly::ResidentAssembly {
            resident,
            upload: resident_upload,
            publication,
            mappings,
            textures: draw_textures,
            capture_textures: draw_capture_textures,
            buffer_allocations: resident_buffers,
            texture_allocations: resident_textures,
        } = self.assemble_deferred_residents(inputs, prepared, None)?;

        let (frame, uniform_upload) =
            SurfaceForwardFrameBuffers::allocate_planned(&self.device, frame_plan)
                .map_err(|error| self.observe_rhi_error(error))?;
        let (gbuffer, effects_attachment) = gbuffer_plan
            .allocate_with_effects(&self.device)
            .map_err(|error| self.observe_rhi_error(error))?;
        let scene_color = allocate_deferred_scene_color_desc(&self.device, scene_color_desc)
            .map_err(|error| self.observe_rhi_error(error))?;
        let scene_color_sampler_desc = crate::renderer::forward::default_material_sampler();
        let scene_color_sampler = self
            .device
            .create_sampler(&scene_color_sampler_desc)
            .map_err(|error| self.observe_rhi_error(error))?;
        if !scene_color.filterable {
            return Err(
                self.observe_rhi_error(Error::Unsupported("filterable Deferred scene-color"))
            );
        }
        crate::renderer::forward::validate_ordinary_sampler(&scene_color_sampler_desc)
            .map_err(|error| self.observe_rhi_error(error))?;
        let draws = super::draw_assembly::draws(
            &self.device,
            inputs,
            prepared,
            &geometry_programs,
            &resident,
            &frame.uniform,
            &uniforms,
            &mappings,
            &draw_textures,
            &draw_capture_textures,
            pbr,
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        let mut lighting_entries = vec![
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
        ];
        if let Some(effects) = &effects_attachment {
            lighting_entries.push(BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&effects.view),
            });
        }
        if let Some(range) = &lighting_view {
            lighting_entries.push(BindGroupEntry {
                binding: crate::pbr::basic_deferred_view_binding(effects),
                resource: BindingResource::Buffer {
                    buffer: &frame.uniform,
                    range: range.clone(),
                },
            });
        }
        let lighting_group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Deferred surface lighting G-buffer"),
                layout: &lighting_program.layout,
                entries: &lighting_entries,
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        if let Some(environment) = environment.as_ref() {
            environment
                .validate_for_sampling(self.device.id())
                .map_err(|error| self.observe_rhi_error(error))?;
        }
        let lighting_environment_group = lighting_program
            .environment_layout
            .as_ref()
            .map(|layout| {
                crate::renderer::lighting::bind_group(
                    &self.device,
                    layout,
                    environment.as_ref(),
                    specular.as_ref(),
                    environment.is_some(),
                    specular.is_some(),
                )
            })
            .transpose()
            .map_err(|error| self.observe_rhi_error(error))?;
        let composite_group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Deferred surface scene-color composite"),
                layout: &composite_program.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&scene_color.view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&scene_color_sampler),
                    },
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))?;

        self.deferred_stats.buffer_allocations = self
            .deferred_stats
            .buffer_allocations
            .saturating_add(2 + resident_buffers);
        self.deferred_stats.texture_allocations = self
            .deferred_stats
            .texture_allocations
            .saturating_add(4 + resident_textures);
        self.deferred_stats.resident_upload_bytes = self
            .deferred_stats
            .resident_upload_bytes
            .saturating_add(resident_upload.bytes.len() as u64);
        self.deferred_stats.frame_upload_bytes = self
            .deferred_stats
            .frame_upload_bytes
            .saturating_add(uniform_len);
        Ok(DeferredSurfaceFrame {
            pbr,
            publication,
            deformation: None,
            resident,
            resident_upload,
            frame,
            uniform_upload,
            uniform_bytes: uniforms.bytes,
            environment,
            specular,
            gbuffer,
            effects: effects_attachment,
            scene_color,
            scene_color_sampler,
            scene_color_sampler_desc,
            draws,
            lighting_pipeline: lighting_program.pipeline.clone(),
            lighting_group,
            lighting_view,
            lighting_environment_group,
            composite_pipeline: composite_program.pipeline,
            composite_group,
            extent: input.extent(),
            color_format,
        })
    }
}
