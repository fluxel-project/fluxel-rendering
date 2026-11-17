//! Assembly and single-execution recording for Forward capture.

use super::assembly::{AssembledForward, AssembledForwardTarget};
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
    pub(super) async fn submit_prepared_forward_capture<C>(
        &mut self,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        capture_inputs: Vec<crate::ForwardDrawInput>,
        receiver_input: crate::ForwardDrawInput,
        producer_slot: CaptureTextureSlot,
        extent: [u32; 2],
        capture_topology: crate::FrameTopology,
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
        let capture = self
            .assemble_prepared_forward(
                snapshot.clone(),
                capture_inputs,
                extent,
                capture_topology,
                depth_mode,
                assets,
                None,
                TextureFormat::Rgba8Unorm,
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
            buffer_allocations: capture_buffer_allocations,
            buffer_allocation_bytes: capture_buffer_allocation_bytes,
            texture_allocations: capture_texture_allocations,
        } = capture
        else {
            return Err(Error::InvalidState.into());
        };
        let capture_owner = super::captured::CaptureTextureOwner::from_capture_frame(
            &self.device,
            producer_slot,
            &capture_frame,
        )
        .map_err(|error| self.observe_forward_error("capture owner admission", error))?;
        let receiver_environment = self
            .environment_irradiance
            .clone()
            .filter(|_| receiver_input.prepared().uses_environment_irradiance());
        let receiver = self
            .assemble_prepared_forward(
                snapshot,
                vec![receiver_input],
                extent,
                crate::FrameTopology::OpaqueForward,
                depth_mode,
                assets,
                None,
                TextureFormat::Rgba8Unorm,
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
            buffer_allocations,
            buffer_allocation_bytes,
            texture_allocations,
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
                .saturating_add(capture_buffer_allocations)
                .saturating_add(buffer_allocations);
            self.forward_stats.accepted_buffer_allocation_bytes = self
                .forward_stats
                .accepted_buffer_allocation_bytes
                .saturating_add(capture_buffer_allocation_bytes)
                .saturating_add(buffer_allocation_bytes);
            self.forward_stats.accepted_texture_allocations = self
                .forward_stats
                .accepted_texture_allocations
                .saturating_add(capture_texture_allocations)
                .saturating_add(texture_allocations);
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
            capture_contribute,
            |context| {
                context.opaque(|pass| {
                    pass.depends_on_uploads()?;
                    pass.read_vertex_streams()?;
                    pass.read_frame_uniforms()?;
                    pass.read_sampled_textures()?;
                    pass.write_selected_depth()?;
                    pass.write_main_color()
                })
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
        let second = first.and_then(|()| {
            execution.commands(plan.capture_raster_step(), |encoder| {
                plan.record_capture_raster(encoder, &capture, &receiver)
            })
        });
        let third = second.and_then(|()| {
            execution.commands(plan.receiver_upload_step(), |encoder| {
                plan.record_receiver_upload(encoder, &capture, &receiver)
            })
        });
        let fourth = third.and_then(|()| {
            execution.commands(plan.receiver_raster_step(), |encoder| {
                plan.record_receiver_raster(encoder, &capture, &receiver)
            })
        });
        let mut ticket = None;
        let recorded = fourth.and_then(|()| {
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
                capture_publication: publications.0,
                receiver_publication: publications.1,
                plan,
                extent,
                row_pitch,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_forward_capture_surface<C>(
        &mut self,
        retained: PendingInputs,
        capture: crate::residency::ForwardFrame<B>,
        receiver: crate::residency::ForwardFrame<B>,
        publications: (ForwardPublication, ForwardPublication),
        composite: super::SurfaceCompositeOwner<B>,
        image: SurfaceImage<B>,
        capture_topology: crate::FrameTopology,
        receiver_topology: crate::FrameTopology,
        capture_contribute: C,
    ) -> std::result::Result<super::Submission, RendererError>
    where
        C: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let declaration = crate::graph::CaptureForwardSurfacePlan::declare_with(
            &self.device,
            &capture,
            &receiver,
            &image,
            capture_topology,
            receiver_topology,
            capture_contribute,
            |context| {
                context.opaque(|pass| {
                    pass.depends_on_uploads()?;
                    pass.read_vertex_streams()?;
                    pass.read_frame_uniforms()?;
                    pass.read_sampled_textures()?;
                    pass.write_selected_depth()?;
                    pass.write_main_color()
                })
            },
        )
        .map_err(|error| self.forward_plan_error("capture surface graph declaration", error))?;
        let id = self.next_forward_plan_id()?;
        let plan = Arc::new(
            crate::graph::CaptureForwardSurfacePlan::compile_declared(
                &self.device,
                id,
                declaration,
            )
            .map_err(|error| self.forward_plan_error("capture surface graph compilation", error))?,
        );
        plan.validate_frames_with_image(&self.device, &capture, &receiver, &image, &composite)
            .map_err(|error| {
                self.observe_forward_error("capture surface frame validation", error)
            })?;
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
        .map_err(|error| self.observe_forward_error("capture surface begin execution", error))?;
        let mut returned_present = Some(PresentBinding::new(image));
        let present_image = returned_present
            .as_ref()
            .ok_or(Error::InvalidState)?
            .image();
        let a = execution.commands(plan.capture_upload_step(), |e| {
            plan.record_capture_upload(e, &capture, &receiver, present_image)
        });
        let b = a.and_then(|()| {
            execution.commands(plan.capture_raster_step(), |e| {
                plan.record_capture_raster(e, &capture, &receiver, present_image)
            })
        });
        let c = b.and_then(|()| {
            execution.commands(plan.receiver_upload_step(), |e| {
                plan.record_receiver_upload(e, &capture, &receiver, present_image)
            })
        });
        let d = c.and_then(|()| {
            execution.commands(plan.receiver_raster_step(), |e| {
                plan.record_receiver_raster(e, &capture, &receiver, present_image)
            })
        });
        let recorded = d
            .and_then(|()| {
                execution.commands(plan.composite_step(), |e| {
                    plan.record_surface_composite(e, &capture, &receiver, &composite, present_image)
                })
            })
            .and_then(|()| {
                execution
                    .present(
                        plan.present_step(),
                        returned_present.take().ok_or(Error::InvalidState)?,
                    )
                    .map_err(|failure| {
                        returned_present = failure.binding;
                        failure.error
                    })
            });
        let outcome = unsafe { execution.finish() };
        self.settle_submission(
            slot,
            serial,
            retained,
            None,
            (outcome, recorded),
            super::PendingWork::CaptureForwardSurface {
                _capture: Box::new(capture),
                _receiver: Box::new(receiver),
                capture_publication: publications.0,
                receiver_publication: publications.1,
                _composite: composite,
                _present: returned_present,
                plan,
            },
        )
    }
}
