//! Single-Graph execution for a linear Forward scene color and terminal output.

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
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) async fn execute_forward_hdr_surface<F>(
        &mut self,
        retained: PendingInputs,
        frame: crate::residency::ForwardFrame<B>,
        publication: ForwardPublication,
        topology: crate::FrameTopology,
        batch_stats: crate::preparation::ForwardBatchStats,
        image: SurfaceImage<B>,
        contribute: F,
    ) -> std::result::Result<Submission, RendererError>
    where
        F: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let composite = self
            .surface_composite_from_target(
                &frame.frame.target,
                &frame.frame.target_view,
                &frame.frame.target_desc,
                frame.frame.target_instance,
                frame.extent,
                &image,
            )
            .await?;
        self.execute_forward_hdr(
            retained,
            frame,
            publication,
            topology,
            batch_stats,
            Some((image, composite)),
            contribute,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) async fn execute_forward_hdr<F>(
        &mut self,
        retained: PendingInputs,
        frame: crate::residency::ForwardFrame<B>,
        publication: ForwardPublication,
        topology: crate::FrameTopology,
        batch_stats: crate::preparation::ForwardBatchStats,
        surface: Option<(SurfaceImage<B>, SurfaceCompositeOwner<B>)>,
        contribute: F,
    ) -> std::result::Result<Submission, RendererError>
    where
        F: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let extent = frame.extent;
        let row_pitch = frame.readback_row_pitch;
        let image = surface.as_ref().map(|(image, _)| image);
        let declaration = crate::graph::ForwardPlan::declare_hdr_with(
            &self.device,
            &frame,
            topology,
            image,
            contribute,
        )
        .map_err(|error| self.forward_plan_error("HDR graph declaration", error))?;
        let compile_started = Instant::now();
        let id = self.next_forward_plan_id()?;
        let plan = Arc::new(
            crate::graph::ForwardPlan::compile_declared(&self.device, id, declaration)
                .map_err(|error| self.forward_plan_error("HDR graph compilation", error))?,
        );
        let compile_ns = compile_started.elapsed().as_nanos();
        self.forward_plan_compilations = self.forward_plan_compilations.saturating_add(1);
        plan.validate_hdr_frame(&frame, image)
            .map_err(|error| self.observe_forward_error("HDR frame validation", error))?;
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
        for (_, producer) in &frame.indirect {
            self.write_forward_upload(&producer.upload.buffer, &producer.upload_bytes)
                .await?;
        }
        for (_, step) in &frame.particles {
            self.write_forward_upload(&step.upload.buffer, &step.upload_bytes)
                .await?;
        }
        let indirect_steps = (0..frame.indirect.len())
            .map(|index| {
                Ok((
                    plan.indirect_upload_step(index)?,
                    plan.indirect_compute_step(index)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let particle_steps = (0..frame.particles.len())
            .map(|index| {
                Ok((
                    plan.particle_upload_step(index)?,
                    plan.particle_update_step(index)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let offscreen_composite = if surface.is_none() {
            Some(
                self.color_composite_from_target(
                    &frame.frame.target,
                    &frame.frame.target_view,
                    &frame.frame.target_desc,
                    frame.frame.target_instance,
                    frame.extent,
                    TextureFormat::Rgba8Unorm,
                )
                .await?,
            )
        } else {
            None
        };
        let (mut present, composite) = match surface {
            Some((image, composite)) => (Some(PresentBinding::new(image)), Some(composite)),
            None => (None, None),
        };
        let image = present.as_ref().map(PresentBinding::image);
        let serial = self.next_serial()?;
        // All Graph lookup failures must be discovered before execution owns
        // a submission or a Surface lease can be consumed by present.
        let hdr_composite_step = plan.hdr_composite_step()?;
        let hdr_readback_step = if image.is_none() {
            Some(plan.hdr_readback_step()?)
        } else {
            None
        };
        let hdr_present_step = if image.is_some() {
            Some(plan.hdr_present_step()?)
        } else {
            None
        };
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_forward_error("HDR begin execution", error))?;
        let mut recorded = execution.commands(plan.upload_step(), |encoder| {
            plan.record_hdr_upload(encoder, &frame, image)
        });
        for (index, step) in plan.mip_steps().enumerate() {
            if recorded.is_ok() {
                recorded = execution.commands(step, |encoder| {
                    plan.record_hdr_mip(index, encoder, &frame, image)
                });
            }
        }
        for (index, (upload_step, compute_step)) in indirect_steps.iter().copied().enumerate() {
            if recorded.is_ok() {
                recorded = execution.commands(upload_step, |encoder| {
                    plan.record_hdr_indirect_upload(index, encoder, &frame, image)
                });
            }
            if recorded.is_ok() {
                recorded = execution.commands(compute_step, |encoder| {
                    plan.record_hdr_indirect_compute(&self.device, index, encoder, &frame, image)
                });
            }
        }
        for (index, ((upload_step, update_step), (_, step))) in
            particle_steps.into_iter().zip(&frame.particles).enumerate()
        {
            if recorded.is_ok() {
                recorded = execution.commands(upload_step, |encoder| {
                    if step.source_incoming.is_none() {
                        step.mark_recording();
                    }
                    plan.record_hdr_particle_upload(index, encoder, &frame, image)
                });
            }
            if recorded.is_ok() {
                recorded = execution.commands(update_step, |encoder| {
                    step.mark_recording();
                    plan.record_hdr_particle_update(&self.device, index, encoder, &frame, image)
                });
            }
        }
        if recorded.is_ok() {
            recorded = execution.commands(plan.raster_step(), |encoder| {
                if frame.indirect.is_empty() && frame.particles.is_empty() {
                    plan.record_hdr_raster(encoder, &frame, image)
                } else {
                    plan.record_hdr_raster_indirect(&self.device, encoder, &frame, image)
                }
            });
        }
        let mut ticket = None;
        if recorded.is_ok() {
            recorded = match (image, composite.as_ref()) {
                (Some(image), Some(composite)) => {
                    let present_step = hdr_present_step.ok_or(Error::InvalidState)?;
                    execution
                        .commands(hdr_composite_step, |encoder| {
                            plan.record_hdr_surface_composite(
                                &self.device,
                                encoder,
                                &frame,
                                composite,
                                image,
                            )
                        })
                        .and_then(|()| match present.take() {
                            Some(binding) => {
                                execution.present(present_step, binding).map_err(|failure| {
                                    present = failure.binding;
                                    failure.error
                                })
                            }
                            None => Err(Error::InvalidState),
                        })
                }
                (None, None) => execution
                    .commands(hdr_composite_step, |encoder| {
                        plan.record_hdr_offscreen_composite(
                            &self.device,
                            encoder,
                            &frame,
                            offscreen_composite.as_ref().ok_or(Error::InvalidState)?,
                        )
                    })
                    .and_then(|()| {
                        execution.commands(
                            hdr_readback_step.ok_or(Error::InvalidState)?,
                            |encoder| {
                                ticket = Some(plan.record_hdr_readback(encoder, &frame)?);
                                Ok(())
                            },
                        )
                    }),
                _ => Err(Error::InvalidState),
            };
        }
        let outcome = unsafe { execution.finish() };
        let workload = match composite {
            Some(composite) => PendingWork::ForwardHdrSurface {
                _frame: Box::new(frame),
                _composite: composite,
                _present: present,
                publication,
                plan,
            },
            None => PendingWork::Forward {
                _frame: Box::new(frame),
                _oit: None,
                _composite: offscreen_composite,
                publication,
                plan,
                extent,
                row_pitch,
            },
        };
        let result = self.settle_submission(
            slot,
            serial,
            retained,
            ticket,
            (outcome, recorded),
            workload,
        );
        if result.is_ok() {
            self.forward_stats.graph_compile_elapsed_ns = self
                .forward_stats
                .graph_compile_elapsed_ns
                .saturating_add(compile_ns);
            self.forward_stats.opaque_candidates = self
                .forward_stats
                .opaque_candidates
                .saturating_add(u64::try_from(batch_stats.candidates).unwrap_or(u64::MAX));
        }
        result
    }
}
