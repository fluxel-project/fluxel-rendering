//! Scene preparation, residency, and frame assembly for Forward submission.
use super::assembly::{AssembledForward, AssembledForwardTarget};
use super::*;
use std::time::Instant;

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
    /// Builds one fully owned Forward frame before Graph declaration.
    ///
    /// This seam deliberately stops before recording.  Capture composition can
    /// therefore assemble two independently retained frames and create the
    /// receiver material bind group against the capture product, while the
    /// normal path below continues to execute exactly one assembled frame.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn assemble_prepared_forward(
        &mut self,
        retained_snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        prepared_inputs: Vec<crate::ForwardDrawInput>,
        extent: [u32; 2],
        topology: crate::FrameTopology,
        depth_mode: crate::ForwardDepthMode,
        assets: &crate::ForwardAssets,
        surface: Option<(SurfaceImage<B>, [u32; 2])>,
        color_format: TextureFormat,
        preparation_started: Instant,
        capture_texture: Option<&super::captured::CaptureTextureOwner<B>>,
        sampled_target: bool,
        environment_snapshot: Option<
            fluxel_assets::AssetSnapshot<
                crate::EnvironmentIrradianceAsset,
                crate::EnvironmentIrradianceData,
            >,
        >,
    ) -> std::result::Result<AssembledForward<B>, RendererError> {
        use crate::frame_data::forward::{ForwardUniformInput, pack_forward_uniforms};
        use crate::residency::{
            EnvironmentFrameOwner, ForwardBindGroup, ForwardDraw, ForwardFrame,
            ForwardFrameBuffers, ForwardResidentBuffers, ForwardSampler, ForwardStreamInput,
            SurfaceForwardFrame, SurfaceForwardFrameBuffers,
        };
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let depth = super::depth::allocate_depth(&self.device, depth_mode, topology, extent)?;
        let specular = if let Some(input) = prepared_inputs
            .iter()
            .find(|input| input.prepared().uses_specular_ibl())
        {
            let snapshot = self
                .environment_specular
                .as_ref()
                .ok_or(Error::InvalidState)?;
            Some(crate::renderer::lighting::allocate_specular(
                &self.device,
                snapshot,
                input.view(),
            )?)
        } else {
            None
        };
        let batched = crate::preparation::batch_forward_draws(prepared_inputs);
        let batch_stats = batched.stats();
        let (inputs, batches) = batched.into_draws_and_batches();
        let draw_batches: Vec<_> = batches.iter().map(|batch| batch.draws.clone()).collect();
        let draw_batch_ids: Vec<_> = batches
            .iter()
            .enumerate()
            .flat_map(|(batch, entry)| std::iter::repeat_n(batch, entry.draws.len()))
            .collect();
        let draw_compatibility: Vec<_> = batches
            .iter()
            .flat_map(|entry| std::iter::repeat_n(entry.key.clone(), entry.draws.len()))
            .collect();
        if draw_batch_ids.len() != inputs.len() || draw_compatibility.len() != inputs.len() {
            return Err(Error::InvalidState.into());
        }

        let active_programs: std::collections::HashSet<_> = inputs
            .iter()
            .map(|input| {
                ForwardProgramKey::for_material(
                    input.prepared(),
                    color_format,
                    input.transform_uniform_bytes(),
                    depth_mode,
                )
            })
            .collect();
        // Submitted frames retain their own executable handles. Cache residency
        // is bounded by this frame's concrete program/layout set.
        self.forward_programs
            .retain(|key, _| active_programs.contains(key));
        let uniform_inputs: Vec<_> = inputs
            .iter()
            .map(|input| ForwardUniformInput {
                transform: input.clip_from_local(),
                normal_from_local: input.normal_from_local(),
                world_from_local: input
                    .normal_from_local()
                    .map(|_| input.transform().columns()),
                tint: input.tint(),
            })
            .collect();
        let limits = self.device.capabilities().limits;
        if inputs
            .iter()
            .any(|input| material_requires_ordinary_sampler(input.prepared()))
            && limits.max_bindings_per_group < 3
        {
            return Err(self.observe_forward_error(
                "ordinary sampler binding admission",
                Error::Unsupported("three Forward material bindings"),
            ));
        }
        let uniforms = pack_forward_uniforms(
            &uniform_inputs,
            limits.uniform_offset_alignment,
            limits.max_buffer_bytes,
        )?;
        let uniform_len = u64::try_from(uniforms.bytes.len()).map_err(|_| Error::InvalidRange)?;
        let row_pitch = aligned_forward_row(extent[0])?;
        self.forward_residency.retain_assets(assets);
        let mut stream_inputs = Vec::new();
        let mut texture_inputs = Vec::new();
        let mut final_streams = Vec::<Option<crate::residency::ForwardVertexStream<B>>>::new();
        let mut final_textures = Vec::<Option<crate::residency::ForwardTexture<B>>>::new();
        let mut fresh_stream_slots = Vec::new();
        let mut fresh_texture_slots = Vec::new();
        let mut geometry_slots = std::collections::HashMap::new();
        let mut texture_slots = std::collections::HashMap::new();
        let mut capture_slots = std::collections::HashMap::new();
        let mut mappings = Vec::new();
        let mut draw_textures = Vec::new();
        let mut draw_capture_textures = Vec::new();
        for input in &inputs {
            let geometry_key =
                crate::residency::ForwardResidentCache::<B>::geometry_key(input, self.device.id());
            let mapping = if let Some(mapping) = geometry_slots.get(&geometry_key) {
                Vec::<usize>::clone(mapping)
            } else {
                let mut mapping = Vec::new();
                if let Some(cached) = self.forward_residency.geometry(geometry_key) {
                    self.forward_stats.geometry_hits =
                        self.forward_stats.geometry_hits.saturating_add(1);
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
                        .zip(&input.prepared().geometry().streams)
                    {
                        let slot = final_streams.len();
                        mapping.push(slot);
                        fresh_stream_slots.push(slot);
                        final_streams.push(None);
                        stream_inputs.push(ForwardStreamInput {
                            bytes: bytes.clone(),
                            layout: layout.clone(),
                        });
                    }
                }
                geometry_slots.insert(geometry_key, mapping.clone());
                mapping
            };
            mappings.push(mapping);
            let mut input_texture_slots = Vec::new();
            let mut input_capture_slots = Vec::new();
            for texture_index in 0..input.resolved_textures().len() {
                let texture = input.asset_texture(texture_index);
                if material_requires_ordinary_sampler(input.prepared()) {
                    match texture {
                        Some(texture) => require_filterable_material_texture(
                            &self.device,
                            texture.value().extent,
                            texture.value().encoding,
                        )
                        .map_err(|error| {
                            self.observe_forward_error("sampled material texture admission", error)
                        })?,
                        None => {
                            validate_capture_material_texture(
                                capture_texture,
                                input,
                                texture_index,
                                self.device.id(),
                            )?;
                        }
                    }
                }
                let (texture_slot, is_capture) = if let Some(texture) = texture {
                    let texture_key = crate::residency::ForwardResidentCache::<B>::texture_key(
                        input,
                        texture_index,
                        self.device.id(),
                    )
                    .ok_or(Error::InvalidState)?;
                    let slot = if let Some(slot) = texture_slots.get(&texture_key) {
                        *slot
                    } else {
                        let slot = final_textures.len();
                        if let Some(cached) = self.forward_residency.texture(texture_key) {
                            self.forward_stats.texture_hits =
                                self.forward_stats.texture_hits.saturating_add(1);
                            final_textures.push(Some(cached));
                        } else {
                            let data = texture.value();
                            texture_inputs
                                .push(super::texture_upload::stage(data, limits.max_buffer_bytes)?);
                            fresh_texture_slots.push(slot);
                            final_textures.push(None);
                        }
                        texture_slots.insert(texture_key, slot);
                        slot
                    };
                    (slot, false)
                } else {
                    let capture = capture_texture.ok_or(Error::InvalidState)?;
                    let capture_name = input
                        .capture_texture_slot(texture_index)
                        .ok_or(Error::InvalidState)?;
                    capture.validate_for_receiver(
                        self.device.id(),
                        capture_name,
                        material_requires_ordinary_sampler(input.prepared()),
                    )?;
                    let texture_slot = if let Some(slot) = capture_slots.get(capture_name) {
                        *slot
                    } else {
                        let slot = final_textures.len();
                        final_textures.push(Some(capture.resident()));
                        capture_slots.insert(capture_name.to_owned(), slot);
                        slot
                    };
                    (texture_slot, true)
                };
                input_texture_slots.push(texture_slot);
                input_capture_slots.push(is_capture);
            }
            draw_textures.push(input_texture_slots);
            draw_capture_textures.push(input_capture_slots);
        }
        let resident_plan = crate::residency::plan_forward_resident(stream_inputs, texture_inputs)
            .map_err(|error| self.observe_forward_error("resident planning", error))?;
        let resident_buffer_allocations = u64::try_from(resident_plan.stream_descs.len())
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let resident_buffer_allocation_bytes = resident_plan
            .stream_descs
            .iter()
            .fold(resident_plan.upload_desc.size, |total, desc| {
                total.saturating_add(desc.size)
            });
        let resident_texture_allocations =
            u64::try_from(resident_plan.texture_descs.len()).unwrap_or(u64::MAX);
        let (fresh, mut resident_upload) =
            ForwardResidentBuffers::allocate_planned(&self.device, resident_plan)
                .map_err(|error| self.observe_forward_error("resident allocation", error))?;
        let mut fresh = Arc::try_unwrap(fresh).map_err(|_| Error::InvalidState)?;
        // An ordinary sampler is a concrete device object rather than a Graph
        // resource. The resident owner below is retained by PendingWork, so a
        // native sampler cannot disappear while its bind group is in flight.
        if inputs
            .iter()
            .any(|input| material_requires_ordinary_sampler(input.prepared()))
        {
            let desc = default_material_sampler();
            let sampler = self
                .device
                .create_sampler(&desc)
                .map_err(|error| self.observe_forward_error("material sampler", error))?;
            fresh.samplers.push(ForwardSampler { sampler, desc });
        }
        for (index, stream) in std::mem::take(&mut fresh.streams).into_iter().enumerate() {
            final_streams[fresh_stream_slots[index]] = Some(stream);
        }
        for (index, texture) in std::mem::take(&mut fresh.textures).into_iter().enumerate() {
            final_textures[fresh_texture_slots[index]] = Some(texture);
        }
        for upload in &mut resident_upload.vertices {
            upload.stream = fresh_stream_slots[upload.stream];
        }
        for upload in &mut resident_upload.textures {
            upload.texture = fresh_texture_slots[upload.texture];
        }
        fresh.streams = final_streams
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidState)?;
        fresh.textures = final_textures
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidState)?;
        let resident = Arc::new(fresh);
        let publication = ForwardPublication {
            geometry: geometry_slots.into_iter().collect(),
            textures: texture_slots.into_iter().collect(),
        };
        let reusable_frame = if sampled_target {
            None
        } else {
            self.forward_frames[slot].take().filter(|buffers| {
                crate::frame_data::pool::compatible(
                    buffers,
                    self.device.id(),
                    extent,
                    uniform_len,
                    row_pitch,
                )
            })
        };
        let (
            mut offscreen_frame,
            mut surface_frame,
            uniform_upload,
            frame_buffer_allocations,
            frame_buffer_allocation_bytes,
            frame_texture_allocations,
        ) = if surface.is_some() {
            let frame_plan = crate::residency::plan_surface_forward_frame(
                uniform_len,
                uniform_len,
                0..uniform_len,
                limits.uniform_offset_alignment,
            )
            .map_err(|error| self.observe_forward_error("surface frame planning", error))?;
            let allocation_bytes = frame_plan
                .upload_desc
                .size
                .saturating_add(frame_plan.uniform_desc.size);
            let (buffers, upload) =
                SurfaceForwardFrameBuffers::allocate_planned(&self.device, frame_plan).map_err(
                    |error| self.observe_forward_error("surface frame allocation", error),
                )?;
            (None, Some(buffers), upload, 2, allocation_bytes, 0)
        } else if let Some(buffers) = reusable_frame {
            self.forward_stats.frame_reuses = self.forward_stats.frame_reuses.saturating_add(1);
            (
                Some(buffers),
                None,
                crate::residency::ForwardUniformUpload {
                    source: 0..uniform_len,
                    destination: 0..uniform_len,
                },
                0,
                0,
                0,
            )
        } else {
            let frame_plan = crate::residency::plan_forward_frame(
                extent,
                uniform_len,
                uniform_len,
                0..uniform_len,
                row_pitch,
                limits.uniform_offset_alignment,
                sampled_target,
            )
            .map_err(|error| self.observe_forward_error("frame planning", error))?;
            let allocation_bytes = frame_plan
                .upload_desc
                .size
                .saturating_add(frame_plan.uniform_desc.size)
                .saturating_add(frame_plan.readback_desc.size);
            let (buffers, upload) = ForwardFrameBuffers::allocate_planned(&self.device, frame_plan)
                .map_err(|error| self.observe_forward_error("frame allocation", error))?;
            (Some(buffers), None, upload, 3, allocation_bytes, 1)
        };
        match (&mut offscreen_frame, &mut surface_frame) {
            (Some(frame), None) => frame.depth = depth,
            (None, Some(frame)) => frame.depth = depth,
            _ => return Err(Error::InvalidState.into()),
        }
        let frame_uniform = match (&offscreen_frame, &surface_frame) {
            (Some(buffers), None) => &buffers.uniform,
            (None, Some(buffers)) => &buffers.uniform,
            _ => return Err(Error::InvalidState.into()),
        };
        let environment = environment_snapshot
            .as_ref()
            .map(|snapshot| EnvironmentFrameOwner::allocate(&self.device, snapshot.value()))
            .transpose()
            .map_err(|error| {
                self.observe_forward_error("environment irradiance allocation", error)
            })?;
        if let Some(owner) = environment.as_ref() {
            owner
                .validate_for_sampling(self.device.id())
                .map_err(|error| {
                    self.observe_forward_error("environment irradiance owner", error)
                })?;
        }
        let mut draws = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let schema = input
                .prepared()
                .binding_schema()
                .ok_or(Error::InvalidState)?;
            let objects = self
                .forward_pipeline(
                    input.prepared(),
                    color_format,
                    input.transform_uniform_bytes(),
                    depth_mode,
                )
                .await?;
            let mut material_entries = vec![BindGroupEntry {
                binding: schema.tint,
                resource: BindingResource::Buffer {
                    buffer: frame_uniform,
                    range: 0..16,
                },
            }];
            if schema.textures.len() != draw_textures[index].len() {
                return Err(Error::InvalidRange.into());
            }
            for (texture_index, texture) in schema.textures.iter().enumerate() {
                let slot = draw_textures[index][texture_index];
                validate_material_texture_owner(
                    &resident,
                    slot,
                    self.device.id(),
                    input,
                    texture_index,
                    material_requires_ordinary_sampler(input.prepared()),
                )
                .map_err(|error| self.observe_forward_error("sampled material owner", error))?;
                material_entries.push(BindGroupEntry {
                    binding: texture.binding,
                    resource: BindingResource::TextureView(&resident.textures[slot].view),
                });
            }
            if let Some(binding) = schema.sampler {
                let sampler = resident.samplers.first().ok_or(Error::InvalidState)?;
                validate_ordinary_sampler(&sampler.desc)?;
                material_entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::Sampler(&sampler.sampler),
                });
            }
            material_entries.sort_by_key(|entry| entry.binding);
            let material_group = self
                .device
                .create_bind_group(&BindGroupDesc {
                    label: Some("Forward material"),
                    layout: &objects.material_layout,
                    entries: &material_entries,
                })
                .map_err(|error| self.observe_forward_error("material bind group", error))?;
            let transform_group = self
                .device
                .create_bind_group(&BindGroupDesc {
                    label: Some("Forward transform"),
                    layout: &objects.transform_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: frame_uniform,
                            range: 0..input.transform_uniform_bytes(),
                        },
                    }],
                })
                .map_err(|error| self.observe_forward_error("transform bind group", error))?;
            let environment_group = objects
                .environment_layout
                .as_ref()
                .map(|layout| {
                    crate::renderer::lighting::bind_group(
                        &self.device,
                        layout,
                        environment.as_ref(),
                        specular.as_ref(),
                        input.prepared().uses_environment_irradiance(),
                        input.prepared().uses_specular_ibl(),
                    )
                })
                .transpose()
                .map_err(|error| self.observe_forward_error("environment bind group", error))?;
            let transform_range = uniforms.transforms[index].clone();
            let tint_range = uniforms.tints[index].clone();
            let vertex_ranges = mappings[index]
                .iter()
                .map(|&stream| 0..resident.streams[stream].desc.size)
                .collect();
            draws.push(ForwardDraw {
                batch: draw_batch_ids[index],
                compatibility: draw_compatibility[index].clone(),
                pipeline: objects.pipeline.clone(),
                textures: draw_textures[index].clone(),
                capture_textures: draw_capture_textures[index].clone(),
                vertex_streams: mappings[index].clone(),
                vertex_layouts: input.prepared().geometry().streams.clone(),
                vertex_ranges,
                vertices: input.geometry().value().vertices.clone(),
                instances: 0..1,
                bind_groups: {
                    let mut groups = vec![
                        ForwardBindGroup {
                            slot: 0,
                            group: material_group,
                            dynamic_offsets: vec![
                                u32::try_from(tint_range.start).map_err(|_| Error::InvalidRange)?,
                            ],
                            uniform_ranges: vec![tint_range],
                        },
                        ForwardBindGroup {
                            slot: 1,
                            group: transform_group,
                            dynamic_offsets: vec![
                                u32::try_from(transform_range.start)
                                    .map_err(|_| Error::InvalidRange)?,
                            ],
                            uniform_ranges: vec![transform_range],
                        },
                    ];
                    if let Some(group) = environment_group {
                        groups.push(ForwardBindGroup {
                            slot: 2,
                            group,
                            dynamic_offsets: Vec::new(),
                            uniform_ranges: Vec::new(),
                        });
                    }
                    groups
                },
            });
        }
        let retained = PendingInputs::Frame {
            _snapshot: retained_snapshot,
            _inputs: inputs,
        };
        let preparation_elapsed_ns = preparation_started.elapsed().as_nanos();
        let target = match (surface, offscreen_frame.take(), surface_frame.take()) {
            (Some((image, _)), None, Some(frame)) => {
                let frame = SurfaceForwardFrame {
                    resident,
                    resident_upload,
                    frame,
                    uniform_upload,
                    uniform_bytes: uniforms.bytes,
                    environment,
                    specular,
                    draws,
                    draw_batches,
                    extent,
                    color_format,
                };
                AssembledForwardTarget::Surface { frame, image }
            }
            (None, Some(frame), None) => {
                let frame = ForwardFrame {
                    resident,
                    resident_upload,
                    frame,
                    uniform_upload,
                    uniform_bytes: uniforms.bytes,
                    environment,
                    specular,
                    draws,
                    draw_batches,
                    extent,
                    readback_row_pitch: row_pitch,
                };
                AssembledForwardTarget::Offscreen { frame }
            }
            _ => return Err(Error::InvalidState.into()),
        };
        Ok(AssembledForward {
            retained,
            target,
            publication,
            topology,
            batch_stats,
            preparation_elapsed_ns,
            buffer_allocations: resident_buffer_allocations
                .saturating_add(frame_buffer_allocations),
            buffer_allocation_bytes: resident_buffer_allocation_bytes
                .saturating_add(frame_buffer_allocation_bytes),
            texture_allocations: resident_texture_allocations
                .saturating_add(frame_texture_allocations)
                .saturating_add(u64::from(depth_mode != crate::ForwardDepthMode::Disabled)),
        })
    }
}
