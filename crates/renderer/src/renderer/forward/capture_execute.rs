//! Assembly and single-execution recording for Forward capture.

use super::assembly::{AssembledForward, AssembledForwardTarget};
use super::capture_receiver_contribution::receiver_contribution;
use super::*;
use std::{sync::Arc, time::Instant};

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
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) async fn submit_prepared_forward_capture<C>(
        &mut self,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        capture_inputs: Vec<crate::ForwardDrawInput>,
        receiver_input: crate::ForwardDrawInput,
        producer_slot: CaptureTextureSlot,
        extent: [u32; 2],
        capture_topology: crate::FrameTopology,
        receiver_topology: crate::FrameTopology,
        depth_mode: crate::ForwardDepthMode,
        assets: &crate::ForwardAssets,
        surface: Option<SurfaceImage<B>>,
        preparation_started: Instant,
        capture_environment: Option<
            fluxel_assets::AssetSnapshot<
                crate::EnvironmentIrradianceAsset,
                crate::EnvironmentIrradianceData,
            >,
        >,
        capture_contribute: C,
    ) -> std::result::Result<super::Submission, RendererError>
    where
        C: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let preserve_capture_background = receiver_input.prepared().uses_scene_refraction();
        let scene_color_format = if self.output_processing().is_some() {
            TextureFormat::Rgba16Float
        } else {
            TextureFormat::Rgba8Unorm
        };
        if preserve_capture_background
            && (receiver_topology != crate::FrameTopology::SortedForward
                || depth_mode != crate::ForwardDepthMode::Depth32LessEqualWrite)
        {
            return Err(Error::InvalidState.into());
        }
        // The receiver is assembled after its producer, but its target facts
        // are already known. Admit a transparent receiver before any producer
        // allocation so an unsupported final target cannot strand that work.
        super::target_admission::admit(
            &self.device,
            receiver_topology,
            extent,
            scene_color_format,
            false,
            surface.is_some() || scene_color_format == TextureFormat::Rgba16Float,
            preserve_capture_background,
        )
        .map_err(|error| self.observe_forward_error("capture receiver blend admission", error))?;
        let capture = self
            .assemble_prepared_forward(
                snapshot.clone(),
                capture_inputs,
                extent,
                capture_topology,
                depth_mode,
                assets,
                None,
                scene_color_format,
                preparation_started,
                None,
                true,
                capture_environment,
            )
            .await?;
        let AssembledForward {
            retained:
                PendingInputs::Frame {
                    _snapshot: capture_snapshot,
                    _inputs: capture_inputs,
                },
            target:
                AssembledForwardTarget::Offscreen {
                    frame: capture_frame,
                },
            publication: capture_publication,
            topology: _,
            batch_stats: _,
            preparation_elapsed_ns: _,
            allocations: capture_allocations,
        } = capture
        else {
            return Err(Error::InvalidState.into());
        };
        let mut capture_owner = super::captured::CaptureTextureOwner::from_capture_frame(
            &self.device,
            producer_slot,
            &capture_frame,
        )
        .map_err(|error| self.observe_forward_error("capture owner admission", error))?;
        if preserve_capture_background {
            capture_owner.refraction_depth = Some(
                capture_frame
                    .frame
                    .depth
                    .as_ref()
                    .ok_or(Error::InvalidState)?
                    .retain(),
            );
        }
        let receiver_environment = self
            .environment_irradiance
            .clone()
            .filter(|_| receiver_input.prepared().uses_environment_irradiance());
        let receiver_depth_mode = if receiver_topology == crate::FrameTopology::SortedForward {
            crate::ForwardDepthMode::Depth32LessEqualWrite
        } else {
            depth_mode
        };
        let receiver = self
            .assemble_prepared_forward(
                snapshot,
                vec![receiver_input],
                extent,
                receiver_topology,
                receiver_depth_mode,
                assets,
                None,
                scene_color_format,
                preparation_started,
                Some(&capture_owner),
                surface.is_some(),
                receiver_environment,
            )
            .await?;
        let AssembledForward {
            retained:
                PendingInputs::Frame {
                    _snapshot: _,
                    mut _inputs,
                },
            target:
                AssembledForwardTarget::Offscreen {
                    frame: receiver_frame,
                },
            publication: receiver_publication,
            topology: receiver_topology,
            batch_stats: _,
            preparation_elapsed_ns,
            allocations: receiver_allocations,
        } = receiver
        else {
            return Err(Error::InvalidState.into());
        };
        let resident_copy_bytes = [&capture_frame, &receiver_frame]
            .into_iter()
            .map(|frame| u64::try_from(frame.resident_upload.bytes.len()).unwrap_or(u64::MAX))
            .fold(0_u64, u64::saturating_add);
        let frame_upload_bytes = [&capture_frame, &receiver_frame]
            .into_iter()
            .map(|frame| u64::try_from(frame.uniform_bytes.len()).unwrap_or(u64::MAX))
            .fold(0_u64, u64::saturating_add);
        let draws = capture_frame
            .draws
            .len()
            .saturating_add(receiver_frame.draws.len());
        let batches = capture_frame
            .draw_batches
            .len()
            .saturating_add(receiver_frame.draw_batches.len());
        let receiver_input = _inputs.pop().ok_or(Error::InvalidState)?;
        if !_inputs.is_empty() {
            return Err(Error::InvalidState.into());
        }
        let composite = match surface.as_ref() {
            Some(image) => Some(self.surface_composite_owner(&receiver_frame, image).await?),
            None => None,
        };
        let offscreen_composite = if surface.is_none()
            && receiver_frame.frame.target_desc.format == TextureFormat::Rgba16Float
        {
            Some(
                self.color_composite_from_target(
                    &receiver_frame.frame.target,
                    &receiver_frame.frame.target_view,
                    &receiver_frame.frame.target_desc,
                    receiver_frame.frame.target_instance,
                    receiver_frame.extent,
                    TextureFormat::Rgba8Unorm,
                )
                .await?,
            )
        } else {
            None
        };
        let result = if let (Some(image), Some(composite)) = (surface, composite) {
            self.execute_forward_capture_surface(
                PendingInputs::CaptureForward {
                    _capture_snapshot: capture_snapshot,
                    _capture_inputs: capture_inputs,
                    _receiver: receiver_input,
                },
                capture_frame,
                receiver_frame,
                (capture_publication, receiver_publication),
                composite,
                image,
                capture_topology,
                receiver_topology,
                preserve_capture_background,
                capture_contribute,
            )
            .await
        } else {
            self.execute_forward_capture(
                PendingInputs::CaptureForward {
                    _capture_snapshot: capture_snapshot,
                    _capture_inputs: capture_inputs,
                    _receiver: receiver_input,
                },
                capture_frame,
                receiver_frame,
                (capture_publication, receiver_publication),
                capture_topology,
                receiver_topology,
                preserve_capture_background,
                offscreen_composite,
                capture_contribute,
            )
            .await
        };
        result.map(|submission| {
            self.forward_stats.resident_copy_bytes = self
                .forward_stats
                .resident_copy_bytes
                .saturating_add(resident_copy_bytes);
            self.forward_stats.frame_upload_bytes = self
                .forward_stats
                .frame_upload_bytes
                .saturating_add(frame_upload_bytes);
            self.forward_stats.opaque_candidates = self
                .forward_stats
                .opaque_candidates
                .saturating_add(u64::try_from(draws).unwrap_or(u64::MAX));
            self.forward_stats.opaque_draws = self
                .forward_stats
                .opaque_draws
                .saturating_add(u64::try_from(draws).unwrap_or(u64::MAX));
            self.forward_stats.opaque_batches = self
                .forward_stats
                .opaque_batches
                .saturating_add(u64::try_from(batches).unwrap_or(u64::MAX));
            self.forward_stats.pending_peak = self
                .forward_stats
                .pending_peak
                .max(u64::try_from(self.pending()).unwrap_or(u64::MAX));
            self.forward_stats.preparation_elapsed_ns = self
                .forward_stats
                .preparation_elapsed_ns
                .saturating_add(preparation_elapsed_ns);
            self.forward_stats.accepted_buffer_allocations = self
                .forward_stats
                .accepted_buffer_allocations
                .saturating_add(capture_allocations.buffer_count)
                .saturating_add(receiver_allocations.buffer_count);
            self.forward_stats.accepted_buffer_allocation_bytes = self
                .forward_stats
                .accepted_buffer_allocation_bytes
                .saturating_add(capture_allocations.buffer_bytes)
                .saturating_add(receiver_allocations.buffer_bytes);
            self.forward_stats.accepted_texture_allocations = self
                .forward_stats
                .accepted_texture_allocations
                .saturating_add(capture_allocations.texture_count)
                .saturating_add(receiver_allocations.texture_count);
            submission
        })
    }

    async fn execute_forward_capture<C>(
        &mut self,
        retained: PendingInputs,
        capture: crate::residency::ForwardFrame<B>,
        receiver: crate::residency::ForwardFrame<B>,
        publications: (ForwardPublication, ForwardPublication),
        capture_topology: crate::FrameTopology,
        receiver_topology: crate::FrameTopology,
        preserve_capture_background: bool,
        composite: Option<super::SurfaceCompositeOwner<B>>,
        capture_contribute: C,
    ) -> std::result::Result<super::Submission, RendererError>
    where
        C: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let extent = receiver.extent;
        let row_pitch = receiver.readback_row_pitch;
        let declaration = crate::graph::CaptureForwardPlan::declare_with(
            &self.device,
            &capture,
            &receiver,
            capture_topology,
            receiver_topology,
            preserve_capture_background,
            capture_contribute,
            |context| {
                receiver_contribution(context, receiver_topology, preserve_capture_background)
            },
        )
        .map_err(|error| self.forward_plan_error("capture graph declaration", error))?;
        let id = self.next_forward_plan_id()?;
        let plan = Arc::new(
            crate::graph::CaptureForwardPlan::compile_declared(&self.device, id, declaration)
                .map_err(|error| self.forward_plan_error("capture graph compilation", error))?,
        );
        plan.validate_frames(&capture, &receiver)
            .map_err(|error| self.observe_forward_error("capture frame validation", error))?;
        self.bindings_validated();
        self.write_forward_upload(&capture.resident.upload, &capture.resident_upload.bytes)
            .await?;
        self.write_forward_upload(&capture.frame.upload, &capture.uniform_bytes)
            .await?;
        for specular in [capture.specular.as_ref(), receiver.specular.as_ref()]
            .into_iter()
            .flatten()
        {
            self.write_forward_upload(&specular.upload, &specular.upload_bytes)
                .await?;
        }
        for environment in [capture.environment.as_ref(), receiver.environment.as_ref()]
            .into_iter()
            .flatten()
        {
            self.write_forward_upload(&environment.upload, &environment.upload_bytes)
                .await?;
        }
        self.write_forward_upload(&receiver.resident.upload, &receiver.resident_upload.bytes)
            .await?;
        self.write_forward_upload(&receiver.frame.upload, &receiver.uniform_bytes)
            .await?;
        let serial = self.next_serial()?;
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_forward_error("capture begin execution", error))?;
        let first = execution.commands(plan.capture_upload_step(), |encoder| {
            plan.record_capture_upload(encoder, &capture, &receiver)
        });
        let capture_mips = first.and_then(|()| {
            for (index, generation) in plan.capture_mip_generations().iter().enumerate() {
                execution.commands(plan.capture_mip_generation_step(index)?, |encoder| {
                    plan.record_capture_mip_generation(encoder, &capture, &receiver, generation)
                })?;
            }
            Ok(())
        });
        let second = capture_mips.and_then(|()| {
            execution.commands(plan.capture_raster_step(), |encoder| {
                plan.record_capture_raster(encoder, &capture, &receiver)
            })
        });
        let third = second.and_then(|()| {
            execution.commands(plan.receiver_upload_step(), |encoder| {
                plan.record_receiver_upload(encoder, &capture, &receiver)
            })
        });
        let receiver_mips = third.and_then(|()| {
            for (index, generation) in plan.receiver_mip_generations().iter().enumerate() {
                execution.commands(plan.receiver_mip_generation_step(index)?, |encoder| {
                    plan.record_receiver_mip_generation(encoder, &capture, &receiver, generation)
                })?;
            }
            Ok(())
        });
        let background = receiver_mips.and_then(|()| {
            if let Some(step) = plan.receiver_background_copy_step() {
                execution.commands(step, |encoder| {
                    plan.record_receiver_background_copy(encoder, &capture, &receiver)
                })
            } else {
                Ok(())
            }
        });
        let fourth = background.and_then(|()| {
            execution.commands(plan.receiver_raster_step(), |encoder| {
                plan.record_receiver_raster(encoder, &capture, &receiver)
            })
        });
        let terminal = fourth.and_then(|()| {
            if let Some(step) = plan.receiver_hdr_terminal_step() {
                execution.commands(step, |encoder| {
                    plan.record_hdr_terminal(
                        &self.device,
                        encoder,
                        &capture,
                        &receiver,
                        composite.as_ref().ok_or(Error::InvalidState)?,
                    )
                })
            } else if composite.is_some() {
                Err(Error::InvalidState)
            } else {
                Ok(())
            }
        });
        let mut ticket = None;
        let recorded = terminal.and_then(|()| {
            execution.commands(plan.receiver_readback_step(), |encoder| {
                ticket = Some(plan.record_receiver_readback(encoder, &capture, &receiver)?);
                Ok(())
            })
        });
        let outcome = unsafe { execution.finish() };
        self.settle_submission(
            slot,
            serial,
            retained,
            ticket,
            (outcome, recorded),
            super::PendingWork::CaptureForward {
                _capture: Box::new(capture),
                _receiver: Box::new(receiver),
                _composite: composite,
                capture_publication: publications.0,
                receiver_publication: publications.1,
                plan,
                extent,
                row_pitch,
            },
        )
    }
}

#[path = "capture_surface_execute.rs"]
mod surface_execute;
