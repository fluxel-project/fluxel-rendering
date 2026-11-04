//! Scene preparation, residency, and frame assembly for Forward submission.
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
    #[allow(clippy::too_many_arguments)] // CPU products, target lease and declaration callback have independent ownership.
    pub(super) async fn submit_prepared_forward<F>(
        &mut self,
        retained_snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        prepared_inputs: Vec<crate::ForwardDrawInput>,
        extent: [u32; 2],
        topology: crate::FrameTopology,
        assets: &crate::ForwardAssets,
        surface: Option<(SurfaceImage<B>, [u32; 2])>,
        color_format: TextureFormat,
        preparation_started: Instant,
        contribute: F,
    ) -> std::result::Result<super::Submission, RendererError>
    where
        F: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        use crate::frame_data::forward::{ForwardUniformInput, pack_forward_uniforms};
        use crate::residency::{
            ForwardBindGroup, ForwardDraw, ForwardFrame, ForwardFrameBuffers,
            ForwardResidentBuffers, ForwardStreamInput, ForwardTextureInput, SurfaceForwardFrame,
            SurfaceForwardFrameBuffers,
        };
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
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
            .map(|input| ForwardProgramKey::for_material(input.prepared(), color_format))
            .collect();
        // Submitted frames retain their own executable handles. Cache residency
        // is bounded by this frame's concrete program/layout set.
        self.forward_programs
            .retain(|key, _| active_programs.contains(key));
        let uniform_inputs: Vec<_> = inputs
            .iter()
            .map(|input| ForwardUniformInput {
                transform: input.clip_from_local(),
                tint: input.tint(),
            })
            .collect();
        let limits = self.device.capabilities().limits;
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
        let mut mappings = Vec::new();
        let mut draw_textures = Vec::new();
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
                        mapping.push(final_streams.len());
                        fresh_stream_slots.push(final_streams.len());
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
            let texture_key =
                crate::residency::ForwardResidentCache::<B>::texture_key(input, self.device.id());
            let texture_slot = if let Some(slot) = texture_slots.get(&texture_key) {
                *slot
            } else {
                let slot = final_textures.len();
                if let Some(cached) = self.forward_residency.texture(texture_key) {
                    self.forward_stats.texture_hits =
                        self.forward_stats.texture_hits.saturating_add(1);
                    final_textures.push(Some(cached));
                } else {
                    let data = input.texture().value();
                    let pitch = aligned_forward_row(data.extent[0])?;
                    let size = u64::from(pitch)
                        .checked_mul(u64::from(data.extent[1]))
                        .ok_or(Error::InvalidRange)?;
                    if size > limits.max_buffer_bytes {
                        return Err(Error::InvalidRange.into());
                    }
                    let size = usize::try_from(size).map_err(|_| Error::InvalidRange)?;
                    let mut bytes = Vec::new();
                    bytes
                        .try_reserve_exact(size)
                        .map_err(|_| Error::OutOfMemory)?;
                    bytes.resize(size, 0);
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
                    });
                    fresh_texture_slots.push(slot);
                    final_textures.push(None);
                }
                texture_slots.insert(texture_key, slot);
                slot
            };
            draw_textures.push(texture_slot);
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
        } else if let Some(buffers) = self.forward_frames[slot].take().filter(|buffers| {
            crate::frame_data::pool::compatible(
                buffers,
                self.device.id(),
                extent,
                uniform_len,
                row_pitch,
            )
        }) {
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
        let frame_uniform = match (&offscreen_frame, &surface_frame) {
            (Some(buffers), None) => &buffers.uniform,
            (None, Some(buffers)) => &buffers.uniform,
            _ => return Err(Error::InvalidState.into()),
        };
        let mut draws = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let objects = self
                .forward_pipeline(input.prepared(), color_format)
                .await?;
            let material_group = self
                .device
                .create_bind_group(&BindGroupDesc {
                    label: Some("Forward material"),
                    layout: &objects.material_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer {
                                buffer: frame_uniform,
                                range: 0..16,
                            },
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(
                                &resident.textures[draw_textures[index]].view,
                            ),
                        },
                    ],
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
                            range: 0..64,
                        },
                    }],
                })
                .map_err(|error| self.observe_forward_error("transform bind group", error))?;
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
                texture: draw_textures[index],
                vertex_streams: mappings[index].clone(),
                vertex_layouts: input.prepared().geometry().streams.clone(),
                vertex_ranges,
                vertices: input.geometry().value().vertices.clone(),
                instances: 0..1,
                bind_groups: vec![
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
                ],
            });
        }
        let retained = (retained_snapshot, inputs);
        let preparation_elapsed_ns = preparation_started.elapsed().as_nanos();
        let result = match (surface, offscreen_frame.take(), surface_frame.take()) {
            (Some((image, _)), None, Some(frame)) => {
                let frame = SurfaceForwardFrame {
                    resident,
                    resident_upload,
                    frame,
                    uniform_upload,
                    uniform_bytes: uniforms.bytes,
                    draws,
                    draw_batches,
                    extent,
                    color_format,
                };
                self.execute_forward_surface(
                    retained,
                    frame,
                    publication,
                    topology,
                    batch_stats,
                    image,
                    contribute,
                )
                .await
            }
            (None, Some(frame), None) => {
                let frame = ForwardFrame {
                    resident,
                    resident_upload,
                    frame,
                    uniform_upload,
                    uniform_bytes: uniforms.bytes,
                    draws,
                    draw_batches,
                    extent,
                    readback_row_pitch: row_pitch,
                };
                self.execute_forward_with_contribution(
                    retained,
                    frame,
                    publication,
                    topology,
                    batch_stats,
                    contribute,
                )
                .await
            }
            _ => Err(Error::InvalidState.into()),
        };
        if result.is_ok() {
            self.forward_stats.preparation_elapsed_ns = self
                .forward_stats
                .preparation_elapsed_ns
                .saturating_add(preparation_elapsed_ns);
            self.forward_stats.accepted_buffer_allocations = self
                .forward_stats
                .accepted_buffer_allocations
                .saturating_add(resident_buffer_allocations)
                .saturating_add(frame_buffer_allocations);
            self.forward_stats.accepted_buffer_allocation_bytes = self
                .forward_stats
                .accepted_buffer_allocation_bytes
                .saturating_add(resident_buffer_allocation_bytes)
                .saturating_add(frame_buffer_allocation_bytes);
            self.forward_stats.accepted_texture_allocations = self
                .forward_stats
                .accepted_texture_allocations
                .saturating_add(resident_texture_allocations)
                .saturating_add(frame_texture_allocations);
        }
        result
    }
}
