//! Deferred offscreen target assembly over validated scene draw ownership.

use std::sync::Arc;

use fluxel_rhi::api::{
    BindGroupDesc, BindGroupEntry, BindingResource, Error, Support, TextureDesc, TextureUsage,
};

use crate::residency::{
    DeferredFrame, ForwardFrameBuffers, allocate_deferred_scene_color_desc,
    deferred_scene_color_desc_with_format, plan_deferred_gbuffer_with_effects,
    plan_deferred_msaa_with_effects, plan_forward_frame,
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
    pub(super) fn deferred_scene_color_format(&self) -> fluxel_rhi::api::TextureFormat {
        if self.output_processing().is_some() {
            fluxel_rhi::api::TextureFormat::Rgba16Float
        } else {
            fluxel_rhi::api::TextureFormat::Rgba8Unorm
        }
    }

    pub(super) fn admit_deferred_scene_color(
        &mut self,
        desc: &TextureDesc,
    ) -> std::result::Result<(), RendererError> {
        match self.device.texture_support(desc) {
            Support::Supported(facts) if facts.filterable => Ok(()),
            Support::Supported(_) => {
                Err(self.observe_rhi_error(Error::Unsupported("filterable Deferred scene-color")))
            }
            Support::Unsupported(reason) => Err(self.observe_rhi_error(Error::Unsupported(reason))),
        }
    }

    /// Rebuilds one exact Forward draw with its Deferred surface template.
    ///
    /// The caller has already run `prepare_forward` for scene/asset/schema
    /// validation. This method intentionally recompiles the exact material and
    /// geometry against the Deferred template, so its distinct MRT interface is
    /// reflected and checked before any Deferred allocation or recording.
    pub(super) async fn prepare_deferred_frame(
        &mut self,
        input: &ForwardDrawInput,
    ) -> std::result::Result<DeferredFrame<B>, RendererError> {
        self.prepare_deferred_frames(std::slice::from_ref(input))
            .await
    }

    pub(super) async fn prepare_deferred_frames(
        &mut self,
        inputs: &[ForwardDrawInput],
    ) -> std::result::Result<DeferredFrame<B>, RendererError> {
        self.prepare_deferred_frames_with_capture(inputs, None)
            .await
    }

    pub(super) async fn prepare_deferred_frames_with_capture(
        &mut self,
        inputs: &[ForwardDrawInput],
        capture: Option<&crate::renderer::forward::CaptureTextureOwner<B>>,
    ) -> std::result::Result<DeferredFrame<B>, RendererError> {
        self.prepare_deferred_frames_with_color_copy(inputs, capture, false)
            .await
    }

    pub(super) async fn prepare_deferred_frames_with_color_copy(
        &mut self,
        inputs: &[ForwardDrawInput],
        capture: Option<&crate::renderer::forward::CaptureTextureOwner<B>>,
        copy_source: bool,
    ) -> std::result::Result<DeferredFrame<B>, RendererError> {
        self.ensure_open()?;
        crate::renderer::require_opaque_or_mask(inputs)
            .map_err(|error| self.observe_rhi_error(error))?;
        let (prepared, pbr, effects) = super::scene_prepare::materials(inputs)?;
        let extent = inputs.first().ok_or(Error::InvalidState)?.extent();
        let mut scene_color =
            deferred_scene_color_desc_with_format(extent, self.deferred_scene_color_format());
        if copy_source {
            scene_color.usage |= TextureUsage::COPY_SRC;
            match self.device.texture_support(&scene_color) {
                Support::Supported(facts) if facts.filterable => {}
                Support::Supported(_) => {
                    return Err(Error::Unsupported("filterable Deferred capture color").into());
                }
                Support::Unsupported(reason) => return Err(Error::Unsupported(reason).into()),
            }
        }
        self.admit_deferred_scene_color(&scene_color)?;
        self.prepare_deferred_resources(inputs, &prepared, pbr, effects, capture, scene_color)
            .await
    }

    async fn prepare_deferred_resources(
        &mut self,
        inputs: &[ForwardDrawInput],
        prepared: &[PreparedMaterial],
        pbr: bool,
        effects: bool,
        capture: Option<&crate::renderer::forward::CaptureTextureOwner<B>>,
        scene_color_desc: TextureDesc,
    ) -> std::result::Result<DeferredFrame<B>, RendererError> {
        super::resident_assembly::validate_capture_inputs(
            inputs,
            prepared,
            capture,
            self.device.id(),
        )?;
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
        let lighting_bindings = 3 + u32::from(effects) + u32::from(pbr && specular.is_none());
        if limits.max_bindings_per_group < lighting_bindings {
            return Err(self.observe_rhi_error(Error::Unsupported("Deferred lighting bindings")));
        }
        for material in prepared {
            crate::renderer::forward::validate_material_binding_limit(
                material,
                limits.max_bindings_per_group,
            )
            .map_err(|error| self.observe_rhi_error(error))?;
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
        let row_pitch = aligned_row(input.extent()[0])?;
        let frame_plan = plan_forward_frame(
            input.extent(),
            uniform_len,
            uniform_len,
            0..uniform_len,
            row_pitch,
            limits.uniform_offset_alignment,
            false,
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        let gbuffer_plan =
            plan_deferred_gbuffer_with_effects(&self.device, input.extent(), pbr, effects)
                .map_err(|error| self.observe_rhi_error(error))?;
        let msaa_plan = plan_deferred_msaa_with_effects(
            &self.device,
            input.extent(),
            pbr,
            effects,
            self.deferred_sample_count(),
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        // All Forward-owner descriptors are known here, before the first of
        // those owners is allocated. The G-buffer plan already performed its
        // equivalent exact texture-support checks while it was built above.
        for desc in [
            &frame_plan.upload_desc,
            &frame_plan.uniform_desc,
            &frame_plan.readback_desc,
        ] {
            if let Support::Unsupported(reason) = self.device.buffer_support(desc) {
                return Err(self.observe_rhi_error(Error::Unsupported(reason)));
            }
        }
        if let Support::Unsupported(reason) = self.device.texture_support(&frame_plan.target_desc) {
            return Err(self.observe_rhi_error(Error::Unsupported(reason)));
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
            .scene_color_composite_program(
                fluxel_rhi::api::TextureFormat::Rgba8Unorm,
                scene_color_desc.format,
            )
            .await?;
        let frame_slot = self.free_slot().ok_or(RendererError::Backpressure)?;

        let super::resident_assembly::ResidentAssembly {
            resident,
            upload: resident_upload,
            publication,
            mappings,
            textures: draw_textures,
            capture_textures: draw_capture_textures,
            buffer_allocations: resident_buffers,
            texture_allocations: resident_textures,
        } = self.assemble_deferred_residents(inputs, prepared, capture)?;

        // Reuse only owners that were retained after both GPU completion and
        // successful CPU readback. The compatibility check reads the saved
        // descriptors, provenance, and post-use states; it does not trust the
        // new Graph declaration as an allocation fact.
        let pooled = self.deferred_frames[frame_slot].take().filter(|entry| {
            crate::frame_data::deferred_pool::compatible(
                entry,
                self.device.id(),
                input.extent(),
                uniform_len,
                row_pitch,
                pbr,
                effects,
            )
        });
        let reused_frame = pooled.is_some();
        let (frame, uniform_upload, gbuffer, effects_attachment) = if let Some(entry) = pooled {
            (
                entry.frame,
                frame_plan.uniform_upload.clone(),
                entry.gbuffer,
                entry.effects,
            )
        } else {
            let (frame, uniform_upload) =
                ForwardFrameBuffers::allocate_planned(&self.device, frame_plan)
                    .map_err(|error| self.observe_rhi_error(error))?;
            let (gbuffer, effects_attachment) = gbuffer_plan
                .allocate_with_effects(&self.device)
                .map_err(|error| self.observe_rhi_error(error))?;
            (frame, uniform_upload, gbuffer, effects_attachment)
        };
        let scene_color = allocate_deferred_scene_color_desc(&self.device, scene_color_desc)
            .map_err(|error| self.observe_rhi_error(error))?;
        let msaa = self.realize_deferred_msaa(msaa_plan, pbr, effects).await?;
        let scene_color_sampler_desc = crate::renderer::forward::material_sampler(1)
            .map_err(|error| self.observe_rhi_error(error))?;
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
            capture.map(|texture| texture.desc.mip_levels),
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
                label: Some("Deferred lighting G-buffer"),
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
                label: Some("Deferred scene-color composite"),
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

        if reused_frame {
            self.deferred_stats.frame_reuses = self.deferred_stats.frame_reuses.saturating_add(1);
        } else {
            self.deferred_stats.buffer_allocations =
                self.deferred_stats.buffer_allocations.saturating_add(3);
            self.deferred_stats.texture_allocations =
                self.deferred_stats.texture_allocations.saturating_add(5);
        }
        self.deferred_stats.buffer_allocations = self
            .deferred_stats
            .buffer_allocations
            .saturating_add(resident_buffers);
        self.deferred_stats.texture_allocations = self
            .deferred_stats
            .texture_allocations
            .saturating_add(resident_textures);
        self.deferred_stats.resident_upload_bytes = self
            .deferred_stats
            .resident_upload_bytes
            .saturating_add(resident_upload.bytes.len() as u64);
        self.deferred_stats.frame_upload_bytes = self
            .deferred_stats
            .frame_upload_bytes
            .saturating_add(uniform_len);
        Ok(DeferredFrame {
            pbr,
            publication,
            deformation: None,
            resident: Arc::clone(&resident),
            resident_upload,
            frame,
            uniform_upload,
            uniform_bytes: uniforms.bytes,
            environment,
            specular,
            gbuffer,
            effects: effects_attachment,
            msaa,
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
            readback_row_pitch: row_pitch,
        })
    }
}

pub(super) fn aligned_row(width: u32) -> std::result::Result<u32, Error> {
    let bytes = u64::from(width).checked_mul(4).ok_or(Error::InvalidRange)?;
    let aligned = bytes.checked_add(255).ok_or(Error::InvalidRange)? / 256 * 256;
    u32::try_from(aligned).map_err(|_| Error::InvalidRange)
}
