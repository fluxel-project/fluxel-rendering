//! Device pipeline realization for prepared Forward materials.
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
    pub(in crate::renderer) async fn forward_pipeline(
        &mut self,
        prepared: &PreparedMaterial,
        color_format: TextureFormat,
    ) -> std::result::Result<Arc<ForwardPipelineObjects<B>>, RendererError> {
        self.ensure_open()?;
        let key = ForwardProgramKey::for_material(prepared, color_format);
        if let Some(objects) = self.forward_programs.get(&key) {
            return Ok(Arc::clone(objects));
        }
        let artifact = prepared.shader();
        let vertex = artifact
            .entry_point(fluxel_shader::ShaderStage::Vertex)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let fragment = artifact
            .entry_point(fluxel_shader::ShaderStage::Fragment)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let descriptor = ShaderDesc {
            label: Some("Fluxel Forward material program".into()),
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
        let demands: Vec<_> = artifact
            .requirements()
            .naga_capabilities
            .iter()
            .map(|capability| capability.name().to_owned())
            .collect();
        admission::admit(
            &demands,
            self.device.capabilities(),
            PipelineCapabilityFacts {
                multiview_views: None,
                push_constant_bytes: 0,
                multisample_count: 1,
                nonuniform: Default::default(),
            },
        )
        .map_err(RendererError::ShaderAdmission)?;
        // Route query and shader creation consume this same complete descriptor.
        if matches!(
            self.device.shader_route(&descriptor),
            ShaderRoute::Unsupported
        ) {
            return Err(Error::Unsupported("Forward shader route").into());
        }
        let shader = self
            .device
            .shader(descriptor)
            .await
            .map_err(|error| self.observe_forward_error("shader creation", error))?;
        let uniform_entry = |binding, size, visibility| BindGroupLayoutEntry {
            binding,
            visibility: vec![visibility],
            ty: BindingType::UniformBuffer {
                dynamic_offset: true,
                min_size: size,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        };
        let material_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some("Forward material bindings".into()),
                entries: vec![
                    uniform_entry(0, 16, ShaderStage::Fragment),
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: vec![ShaderStage::Fragment],
                        ty: BindingType::SampledTexture {
                            sample: TextureSampleType::Float,
                            dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: BindingCount::One,
                        partially_bound: false,
                        nonuniform_indexing: false,
                    },
                ],
            })
            .map_err(|error| self.observe_forward_error("material layout", error))?;
        let transform_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some("Forward transform bindings".into()),
                entries: vec![uniform_entry(0, 64, ShaderStage::Vertex)],
            })
            .map_err(|error| self.observe_forward_error("transform layout", error))?;
        let layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Forward pipeline layout"),
                groups: &[&material_layout, &transform_layout],
                push_constants: &[],
            })
            .map_err(|error| self.observe_forward_error("pipeline layout", error))?;
        let pipeline = self
            .device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("Forward opaque pipeline".into()),
                layout,
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
                vertex_buffers: prepared.geometry().streams.clone(),
                color_targets: vec![Some(ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: 0xf,
                })],
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: CullMode::None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: u64::from(u32::MAX),
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await
            .map_err(|error| self.observe_forward_error("raster pipeline", error))?;
        let objects = Arc::new(ForwardPipelineObjects {
            pipeline,
            material_layout,
            transform_layout,
        });
        self.forward_programs.insert(key, Arc::clone(&objects));
        Ok(objects)
    }
}
