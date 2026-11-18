//! One execution and covering completion for both capture OIT products.

use super::capture_oit_entry::CaptureOitSide;
use super::oit_execute::OitExecutionTarget;
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
    pub(in crate::renderer) async fn execute_capture_oit(
        &mut self,
        capture: CaptureOitSide<B>,
        receiver: CaptureOitSide<B>,
        target: OitExecutionTarget<B>,
        started: Instant,
    ) -> std::result::Result<Submission, RendererError> {
        self.execute_capture_oit_with_capture_contribution(
            capture,
            receiver,
            target,
            started,
            |context| {
                context.sorted(|pass| {
                    pass.depends_on_uploads()?;
                    pass.read_vertex_streams()?;
                    pass.read_frame_uniforms()?;
                    pass.read_sampled_textures()?;
                    pass.write_selected_depth()?;
                    pass.read_write_main_color()
                })
            },
            |context| context.all_reads(),
        )
        .await
    }

    pub(in crate::renderer) async fn execute_capture_oit_with_capture_contribution<N, C>(
        &mut self,
        capture: CaptureOitSide<B>,
        receiver: CaptureOitSide<B>,
        target: OitExecutionTarget<B>,
        started: Instant,
        capture_ordinary: N,
        capture_contribute: C,
    ) -> std::result::Result<Submission, RendererError>
    where
        N: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
        C: FnMut(
            &mut crate::graph::ForwardOitReadContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let scene_refraction = match (&capture.retained, &receiver.retained) {
            (PendingInputs::Frame { .. }, PendingInputs::Frame { _inputs, .. }) => {
                let first = _inputs.first().ok_or(Error::InvalidState)?;
                let scene_refraction = first.prepared().uses_scene_refraction();
                if _inputs.len() > 1
                    && (!scene_refraction
                        || _inputs
                            .iter()
                            .any(|input| !input.prepared().uses_scene_refraction()))
                {
                    return Err(Error::InvalidState.into());
                }
                scene_refraction
            }
            _ => return Err(Error::InvalidState.into()),
        };
        let capture_input = capture.graph_input();
        let receiver_input = receiver.graph_input();
        let (mut present, composite) = match target {
            OitExecutionTarget::Readback => (None, None),
            OitExecutionTarget::Surface { image, composite } => {
                (Some(PresentBinding::new(image)), Some(composite))
            }
        };
        let image = present.as_ref().map(PresentBinding::image);
        // Readback from a linear HDR scene product must go through the same
        // output terminal as the ordinary Forward capture path. The source is
        // the receiver's actual final product, which is the OIT output when
        // peeling was selected and otherwise the receiver color attachment.
        let offscreen_composite = if image.is_none()
            && receiver.frame.frame.target_desc.format == TextureFormat::Rgba16Float
        {
            match receiver.oit.as_ref() {
                Some(oit) => Some(
                    self.oit_offscreen_composite_owner(
                        &oit.output,
                        receiver.frame.extent,
                        TextureFormat::Rgba8Unorm,
                    )
                    .await?,
                ),
                None => Some(
                    self.color_composite_from_target(
                        &receiver.frame.frame.target,
                        &receiver.frame.frame.target_view,
                        &receiver.frame.frame.target_desc,
                        receiver.frame.frame.target_instance,
                        receiver.frame.extent,
                        TextureFormat::Rgba8Unorm,
                    )
                    .await?,
                ),
            }
        } else {
            None
        };
        let declaration = match image {
            Some(image) => crate::graph::CaptureOitPlan::declare_surface_with_capture_contribution(
                &self.device,
                &capture.frame,
                &receiver.frame,
                capture_input.as_ref(),
                receiver_input.as_ref(),
                scene_refraction,
                image,
                capture_ordinary,
                capture_contribute,
            ),
            None => crate::graph::CaptureOitPlan::declare_with_capture_contribution(
                &self.device,
                &capture.frame,
                &receiver.frame,
                capture_input.as_ref(),
                receiver_input.as_ref(),
                scene_refraction,
                capture_ordinary,
                capture_contribute,
            ),
        }
        .map_err(|error| self.forward_plan_error("capture OIT declaration", error))?;
        let id = self.next_forward_plan_id()?;
        let compile_started = Instant::now();
        let plan = Arc::new(
            crate::graph::CaptureOitPlan::compile_declared(&self.device, id, declaration)
                .map_err(|error| self.forward_plan_error("capture OIT compilation", error))?,
        );
        let compile_ns = compile_started.elapsed().as_nanos();
        self.forward_plan_compilations = self.forward_plan_compilations.saturating_add(1);
        match (image, composite.as_ref()) {
            (Some(image), Some(composite)) => plan.validate_frames_with_image(
                &self.device,
                &capture.frame,
                &receiver.frame,
                capture_input.as_ref(),
                receiver_input.as_ref(),
                image,
                composite,
            ),
            (None, None) => plan.validate_frames(
                &capture.frame,
                &receiver.frame,
                capture_input.as_ref(),
                receiver_input.as_ref(),
            ),
            _ => Err(Error::InvalidState),
        }
        .map_err(|error| self.observe_forward_error("capture OIT frame validation", error))?;
        self.bindings_validated();
        for frame in [&capture.frame, &receiver.frame] {
            self.write_forward_upload(&frame.resident.upload, &frame.resident_upload.bytes)
                .await?;
            self.write_forward_upload(&frame.frame.upload, &frame.uniform_bytes)
                .await?;
            if let Some(owner) = frame.specular.as_ref() {
                self.write_forward_upload(&owner.upload, &owner.upload_bytes)
                    .await?;
            }
            if let Some(owner) = frame.environment.as_ref() {
                self.write_forward_upload(&owner.upload, &owner.upload_bytes)
                    .await?;
            }
        }
        let capture_draws = capture.oit.as_ref().map(OitFrameResources::draw_commands);
        let receiver_draws = receiver.oit.as_ref().map(OitFrameResources::draw_commands);
        let capture_commands = capture
            .oit
            .as_ref()
            .zip(capture_draws.as_ref())
            .map(|(oit, draws)| oit.commands(&capture.frame, draws));
        let receiver_commands = receiver
            .oit
            .as_ref()
            .zip(receiver_draws.as_ref())
            .map(|(oit, draws)| oit.commands(&receiver.frame, draws));
        // Resolve every fallible step lookup before execution can submit work.
        let present_step = image.map(|_| plan.present_step()).transpose()?;
        let serial = self.next_serial()?;
        let preparation_ns = started.elapsed().as_nanos();
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_forward_error("capture OIT begin execution", error))?;
        let mut recorded = Ok(());
        let mut ticket = None;
        // This iterator is the actual compiled Graph order. It is also used
        // by completion validation; there is no parallel capture scheduler.
        for (step, pass) in plan.steps() {
            if recorded.is_ok() {
                recorded = execution.commands(step, |encoder| match (image, composite.as_ref()) {
                    (Some(image), Some(composite)) => plan.record_surface_pass(
                        &self.device,
                        encoder,
                        &capture.frame,
                        &receiver.frame,
                        capture_input.as_ref(),
                        receiver_input.as_ref(),
                        capture_commands.as_ref(),
                        receiver_commands.as_ref(),
                        image,
                        composite,
                        pass,
                    ),
                    (None, None) => {
                        if let Some(readback) = plan.record_pass(
                            &self.device,
                            encoder,
                            &capture.frame,
                            &receiver.frame,
                            capture_input.as_ref(),
                            receiver_input.as_ref(),
                            capture_commands.as_ref(),
                            receiver_commands.as_ref(),
                            offscreen_composite.as_ref(),
                            pass,
                        )? {
                            if ticket.is_some() {
                                return Err(Error::InvalidState);
                            }
                            ticket = Some(readback);
                        }
                        Ok(())
                    }
                    _ => Err(Error::InvalidState),
                });
            }
        }
        if let Some(step) = present_step {
            if recorded.is_ok() {
                recorded = match present.take() {
                    Some(binding) => execution.present(step, binding).map_err(|failure| {
                        present = failure.binding;
                        failure.error
                    }),
                    None => Err(Error::InvalidState),
                };
            }
        }
        let outcome = unsafe { execution.finish() };
        let extent = receiver.frame.extent;
        let row_pitch = receiver.frame.readback_row_pitch;
        let buffers = capture
            .allocations
            .buffer_count
            .saturating_add(receiver.allocations.buffer_count);
        let buffer_bytes = capture
            .allocations
            .buffer_bytes
            .saturating_add(receiver.allocations.buffer_bytes);
        let textures = capture
            .allocations
            .texture_count
            .saturating_add(receiver.allocations.texture_count)
            .saturating_add(8 * u64::from(capture.oit.is_some()))
            .saturating_add(8 * u64::from(receiver.oit.is_some()));
        let draws = capture
            .frame
            .draws
            .len()
            .saturating_add(receiver.frame.draws.len());
        let batches = capture
            .frame
            .draw_batches
            .len()
            .saturating_add(receiver.frame.draw_batches.len());
        let upload_bytes = [&capture.frame, &receiver.frame]
            .iter()
            .map(|frame| u64::try_from(frame.resident_upload.bytes.len()).unwrap_or(u64::MAX))
            .fold(0_u64, u64::saturating_add);
        let uniform_bytes = [&capture.frame, &receiver.frame]
            .iter()
            .map(|frame| u64::try_from(frame.uniform_bytes.len()).unwrap_or(u64::MAX))
            .fold(0_u64, u64::saturating_add);
        let PendingInputs::Frame {
            _snapshot,
            _inputs: capture_inputs,
        } = capture.retained
        else {
            unreachable!("capture assembly retains its frame inputs");
        };
        let PendingInputs::Frame {
            _inputs: receivers, ..
        } = receiver.retained
        else {
            unreachable!("receiver assembly retains its frame inputs");
        };
        let retained = PendingInputs::CaptureForward {
            _capture_snapshot: _snapshot,
            _capture_inputs: capture_inputs,
            _receivers: receivers,
        };
        let workload = match composite {
            Some(composite) => PendingWork::CaptureOitSurface {
                _capture: Box::new(capture.frame),
                _receiver: Box::new(receiver.frame),
                _capture_oit: capture.oit.map(Box::new),
                _receiver_oit: receiver.oit.map(Box::new),
                capture_publication: capture.publication,
                receiver_publication: receiver.publication,
                _composite: composite,
                _present: present,
                plan,
            },
            None => PendingWork::CaptureOit {
                _capture: Box::new(capture.frame),
                _receiver: Box::new(receiver.frame),
                _capture_oit: capture.oit.map(Box::new),
                _receiver_oit: receiver.oit.map(Box::new),
                capture_publication: capture.publication,
                receiver_publication: receiver.publication,
                _composite: offscreen_composite,
                plan,
                extent,
                row_pitch,
            },
        };
        let result = self
            .settle_submission(
                slot,
                serial,
                retained,
                ticket,
                (outcome, recorded),
                workload,
            )
            .map_err(|error| match error {
                RendererError::Rhi(error) => {
                    self.observe_forward_error("capture OIT execution", error)
                }
                error => error,
            });
        if result.is_ok() {
            let stats = &mut self.forward_stats;
            stats.accepted_buffer_allocations =
                stats.accepted_buffer_allocations.saturating_add(buffers);
            stats.accepted_buffer_allocation_bytes = stats
                .accepted_buffer_allocation_bytes
                .saturating_add(buffer_bytes);
            stats.accepted_texture_allocations =
                stats.accepted_texture_allocations.saturating_add(textures);
            stats.opaque_candidates = stats
                .opaque_candidates
                .saturating_add(u64::try_from(draws).unwrap_or(u64::MAX));
            stats.opaque_draws = stats
                .opaque_draws
                .saturating_add(u64::try_from(draws).unwrap_or(u64::MAX));
            stats.opaque_batches = stats
                .opaque_batches
                .saturating_add(u64::try_from(batches).unwrap_or(u64::MAX));
            stats.resident_copy_bytes = stats.resident_copy_bytes.saturating_add(upload_bytes);
            stats.frame_upload_bytes = stats.frame_upload_bytes.saturating_add(uniform_bytes);
            stats.graph_compile_elapsed_ns =
                stats.graph_compile_elapsed_ns.saturating_add(compile_ns);
            stats.preparation_elapsed_ns =
                stats.preparation_elapsed_ns.saturating_add(preparation_ns);
            self.forward_stats.pending_peak = self
                .forward_stats
                .pending_peak
                .max(u64::try_from(self.pending()).unwrap_or(u64::MAX));
        }
        result
    }
}
