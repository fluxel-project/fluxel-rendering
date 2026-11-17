//! One execution and completion owner for the complete OIT Forward Graph.

use super::*;
use crate::graph::ForwardOitInput;
use std::{num::NonZeroU32, time::Instant};

pub(in crate::renderer) enum OitExecutionTarget<B: ResourceBackend> {
    Readback,
    Surface {
        image: SurfaceImage<B>,
        composite: SurfaceCompositeOwner<B>,
    },
}

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend + RasterBackend,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_forward_oit<C>(
        &mut self,
        retained: PendingInputs,
        frame: crate::residency::ForwardFrame<B>,
        oit: OitFrameResources<B>,
        publication: ForwardPublication,
        opaque_count: usize,
        peel_count: usize,
        rounds: NonZeroU32,
        batch_stats: crate::preparation::ForwardBatchStats,
        target: OitExecutionTarget<B>,
        contribute: C,
    ) -> std::result::Result<Submission, RendererError>
    where
        C: FnMut(
            &mut crate::graph::ForwardOitReadContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let extent = frame.extent;
        let row_pitch = frame.readback_row_pitch;
        let uniform_len =
            u64::try_from(frame.uniform_bytes.len()).map_err(|_| Error::InvalidRange)?;
        let draw_count = u64::try_from(frame.draws.len()).map_err(|_| Error::InvalidRange)?;
        let resident_upload_bytes =
            frame
                .resident_upload
                .vertices
                .iter()
                .map(|upload| upload.destination.end - upload.destination.start)
                .chain(frame.resident_upload.textures.iter().map(|upload| {
                    u64::from(upload.bytes_per_row) * u64::from(upload.rows_per_image)
                }))
                .sum::<u64>();
        let input = ForwardOitInput {
            owners: &oit.owners,
            output: &oit.output,
            opaque_count,
            peel_count,
            rounds,
        };
        let (mut present, composite) = match target {
            OitExecutionTarget::Readback => (None, None),
            OitExecutionTarget::Surface { image, composite } => {
                (Some(PresentBinding::new(image)), Some(composite))
            }
        };
        let image = present.as_ref().map(PresentBinding::image);
        if let (Some(image), Some(composite)) = (image, composite.as_ref()) {
            composite.validate_for(&self.device, image)?;
            if composite.target_instance() != oit.output.instance {
                return Err(Error::InvalidState.into());
            }
        }
        let declaration = match image {
            Some(image) => crate::graph::ForwardPlan::declare_surface_oit_contribution(
                &self.device,
                &frame,
                &input,
                image,
                contribute,
            ),
            None => crate::graph::ForwardPlan::declare_with_oit_contribution(
                &self.device,
                &frame,
                &input,
                contribute,
            ),
        }
        .map_err(|error| self.forward_plan_error("OIT graph declaration", error))?;
        let id = self.next_forward_plan_id()?;
        let compilation_started = Instant::now();
        let plan = Arc::new(
            crate::graph::ForwardPlan::compile_declared_oit(&self.device, id, declaration)
                .map_err(|error| self.forward_plan_error("OIT graph compilation", error))?,
        );
        let compile_elapsed = compilation_started.elapsed().as_nanos();
        self.forward_plan_compilations = self.forward_plan_compilations.saturating_add(1);
        plan.validate_frame_with_oit(&frame, &input, image)
            .map_err(|error| self.observe_forward_error("OIT frame validation", error))?;
        self.bindings_validated();
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
        let draw_commands = oit.draw_commands();
        let commands = oit.commands(&frame, &draw_commands);
        let surface_steps = if image.is_some() {
            Some((plan.oit_surface_step()?, plan.oit_present_step()?))
        } else {
            None
        };
        let serial = self.next_serial()?;
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_forward_error("OIT begin execution", error))?;
        let mut operation = "OIT upload recording";
        let mut recorded = execution.commands(plan.upload_step(), |encoder| {
            plan.record_upload_oit(encoder, &frame, &input, image)
        });
        for (index, step) in plan.mip_steps().enumerate() {
            if recorded.is_ok() {
                operation = "OIT material mip recording";
                recorded = execution.commands(step, |encoder| {
                    plan.record_mip_oit(index, encoder, &frame, &input, image)
                });
            }
        }
        if recorded.is_ok() {
            operation = "OIT opaque recording";
            recorded = execution.commands(plan.raster_step(), |encoder| {
                plan.record_opaque_oit(encoder, &frame, &input, image)
            });
        }
        for (step, pass) in plan.oit_steps() {
            if recorded.is_ok() {
                operation = "OIT peel and composition recording";
                recorded = execution.commands(step, |encoder| {
                    plan.record_oit_pass(encoder, &frame, &input, &commands, pass, image)
                });
            }
        }
        if let Some(step) = plan.oit_fallback_step() {
            if recorded.is_ok() {
                operation = "OIT excluded transparency recording";
                recorded = execution.commands(step, |encoder| {
                    plan.record_fallback_oit(encoder, &frame, &input, &commands, image)
                });
            }
        }
        let mut ticket = None;
        if recorded.is_ok() {
            recorded = match (image, composite.as_ref(), surface_steps) {
                (Some(image), Some(composite), Some((surface_step, present_step))) => {
                    operation = "OIT Surface composite recording";
                    let composite_result = execution.commands(surface_step, |encoder| {
                        plan.record_oit_surface_composite(
                            &self.device,
                            encoder,
                            &frame,
                            &input,
                            composite,
                            image,
                        )
                    });
                    match composite_result {
                        Ok(()) => {
                            operation = "OIT presentation";
                            match present.take() {
                                Some(binding) => {
                                    execution.present(present_step, binding).map_err(|failure| {
                                        present = failure.binding;
                                        failure.error
                                    })
                                }
                                None => Err(Error::InvalidState),
                            }
                        }
                        Err(error) => Err(error),
                    }
                }
                (None, None, None) => {
                    operation = "OIT final readback recording";
                    execution.commands(plan.readback_step(), |encoder| {
                        ticket = Some(
                            plan.record_readback_oit(encoder, &frame, &input, &commands, None)?,
                        );
                        Ok(())
                    })
                }
                _ => Err(Error::InvalidState),
            };
        }
        let outcome = unsafe { execution.finish() };
        if recorded.is_ok() {
            operation = "OIT execution finalization";
        }
        let workload = match composite {
            Some(composite) => super::PendingWork::ForwardOitSurface {
                _frame: Box::new(frame),
                _oit: Box::new(oit),
                _composite: composite,
                _present: present,
                publication,
                plan,
            },
            None => super::PendingWork::Forward {
                _frame: Box::new(frame),
                _oit: Some(Box::new(oit)),
                publication,
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
                RendererError::Rhi(error) => self.observe_forward_error(operation, error),
                error => error,
            });
        if result.is_ok() {
            self.forward_stats.resident_copy_bytes = self
                .forward_stats
                .resident_copy_bytes
                .saturating_add(resident_upload_bytes);
            self.forward_stats.frame_upload_bytes = self
                .forward_stats
                .frame_upload_bytes
                .saturating_add(uniform_len);
            self.forward_stats.opaque_candidates = self
                .forward_stats
                .opaque_candidates
                .saturating_add(u64::try_from(batch_stats.candidates).unwrap_or(u64::MAX));
            self.forward_stats.opaque_batches = self
                .forward_stats
                .opaque_batches
                .saturating_add(u64::try_from(batch_stats.batches).unwrap_or(u64::MAX));
            self.forward_stats.opaque_draws =
                self.forward_stats.opaque_draws.saturating_add(draw_count);
            self.forward_stats.graph_compile_elapsed_ns = self
                .forward_stats
                .graph_compile_elapsed_ns
                .saturating_add(compile_elapsed);
            self.forward_stats.pending_peak = self
                .forward_stats
                .pending_peak
                .max(u64::try_from(self.pending()).unwrap_or(u64::MAX));
        }
        result
    }
}
