//! Deferred lighting shader and pipeline realization.

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
    pub(in crate::renderer::deferred) async fn deferred_lighting_program(
        &mut self,
        pbr: bool,
        use_environment: bool,
        use_specular: bool,
        effects: bool,
        diffuse_roughness: bool,
        ambient_occlusion: bool,
        color_format: TextureFormat,
    ) -> std::result::Result<Arc<LightingProgram<B>>, RendererError> {
        self.ensure_open()?;
        let request = if pbr {
            if effects && use_environment {
                crate::pbr::lighting_effect_irradiance_request()
            } else if effects {
                crate::pbr::lighting_effect_request()
            } else if use_environment {
                crate::pbr::lighting_irradiance_request()
            } else {
                crate::pbr::lighting_request()
            }
        } else {
            ShaderCompositionRequest::new(vec![SourceUnit::new(
                "deferred-lighting",
                LIGHTING_TEMPLATE,
            )])
            .with_entry_point(EntryPoint::new(
                "deferred_lighting_vs",
                LogicalShaderStage::Vertex,
            ))
            .with_entry_point(EntryPoint::new(
                "deferred_lighting_fs",
                LogicalShaderStage::Fragment,
            ))
            .with_binding(
                BindingDeclaration::new(0, 0, BindingKind::Texture)
                    .with_visibility([LogicalShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(0, 1, BindingKind::Texture)
                    .with_visibility([LogicalShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(0, 2, BindingKind::Texture)
                    .with_visibility([LogicalShaderStage::Fragment]),
            )
        };
        let request = if use_specular {
            crate::pbr::specular::deferred_specular_request(request)
        } else {
            request
        };
        let request = if diffuse_roughness {
            crate::pbr::with_deferred_diffuse_roughness(request)
        } else {
            request
        };
        let request = if ambient_occlusion {
            crate::pbr::with_deferred_ambient_occlusion(request)
        } else {
            request
        };
        let artifact = fluxel_shader::ShaderComposer::new()
            .compose(&request)
            .map_err(RendererError::DeferredShader)?;
        let key = program_cache::LightingProgramKey::for_artifact(
            self.device.id(),
            artifact.cache_key().clone(),
            pbr,
            effects,
            diffuse_roughness,
            ambient_occlusion,
            color_format,
        );
        if let Some(program) = self.deferred_programs.get_lighting(&key) {
            self.deferred_stats.lighting_program_hits += 1;
            return Ok(program);
        }
        let vertex = artifact
            .entry_point(LogicalShaderStage::Vertex)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let fragment = artifact
            .entry_point(LogicalShaderStage::Fragment)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let descriptor = ShaderDesc {
            label: Some("Fluxel Deferred lighting program".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(artifact.source().as_bytes()),
            entries: vec![
                ShaderEntryPoint {
                    name: vertex.clone(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: fragment.clone(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: artifact.requirements().names.iter().cloned().collect(),
        };
        self.admit_deferred_shader(
            &descriptor,
            artifact
                .requirements()
                .naga_capabilities
                .iter()
                .map(|capability| capability.name()),
        )?;
        let shader = self
            .device
            .shader(descriptor)
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        let layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some("Deferred lighting bindings".into()),
                entries: {
                    let mut entries = vec![
                        sampled_texture(0, TextureSampleType::Float),
                        sampled_texture(
                            1,
                            if pbr {
                                TextureSampleType::UnfilterableFloat
                            } else {
                                TextureSampleType::Float
                            },
                        ),
                        sampled_texture(2, TextureSampleType::Depth),
                    ];
                    if effects {
                        entries.push(sampled_texture(
                            3,
                            if ambient_occlusion {
                                TextureSampleType::UnfilterableFloat
                            } else {
                                TextureSampleType::Float
                            },
                        ));
                    }
                    if pbr && !use_specular {
                        entries.push(BindGroupLayoutEntry {
                            binding: crate::pbr::basic_deferred_view_binding(effects),
                            visibility: vec![ShaderStage::Fragment],
                            ty: BindingType::UniformBuffer {
                                dynamic_offset: false,
                                min_size:
                                    crate::frame_data::specular_view::BASIC_DEFERRED_VIEW_BYTES,
                            },
                            count: BindingCount::One,
                            partially_bound: false,
                            nonuniform_indexing: false,
                        });
                    }
                    if diffuse_roughness {
                        entries.push(sampled_texture(
                            crate::pbr::DIFFUSE_ROUGHNESS_BINDING,
                            TextureSampleType::UnfilterableFloat,
                        ));
                    }
                    if ambient_occlusion {
                        entries.push(sampled_texture(
                            crate::pbr::AMBIENT_OCCLUSION_BINDING,
                            TextureSampleType::UnfilterableFloat,
                        ));
                    }
                    entries
                },
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let environment_layout = if use_environment || use_specular {
            Some(
                self.device
                    .create_bind_group_layout(&BindGroupLayoutDesc {
                        label: Some("Deferred PBR environment bindings".into()),
                        entries: crate::renderer::lighting::layout_entries(
                            use_environment,
                            use_specular,
                        ),
                    })
                    .map_err(|error| self.observe_rhi_error(error))?,
            )
        } else {
            None
        };
        let groups: Vec<_> = std::iter::once(&layout)
            .chain(environment_layout.iter())
            .collect();
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Deferred lighting pipeline layout"),
                groups: &groups,
                push_constants: &[],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline = self
            .device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("Deferred lighting pipeline".into()),
                layout: pipeline_layout,
                vertex: PipelineStage {
                    module: shader.clone(),
                    entry: vertex,
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: shader,
                    entry: fragment,
                    constants: vec![],
                }),
                vertex_buffers: vec![],
                color_targets: vec![Some(color_target_with_format(color_format))],
                primitive: opaque_triangles(),
                depth_stencil: None,
                multisample: single_sample(),
                multiview: None,
                cache: None,
            })
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        // As with G-buffer programs, compilation failures never replace a
        // usable cache entry. Submitted frames retain their own Arc owner.
        let program = Arc::new(LightingProgram {
            pipeline,
            layout,
            environment_layout,
        });
        self.deferred_programs
            .publish_lighting(key, Arc::clone(&program));
        Ok(program)
    }
}
