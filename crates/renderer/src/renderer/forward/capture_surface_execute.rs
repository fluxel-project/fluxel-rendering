//! Surface execution for the combined Forward capture Graph.

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
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_forward_capture_surface<C>(
        &mut self,
        retained: PendingInputs,
        capture: crate::residency::ForwardFrame<B>,
        receiver: crate::residency::ForwardFrame<B>,
        publications: (ForwardPublication, ForwardPublication),
        composite: super::SurfaceCompositeOwner<B>,
        image: SurfaceImage<B>,
        capture_topology: crate::FrameTopology,
        receiver_topology: crate::FrameTopology,
        preserve_capture_background: bool,
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
            preserve_capture_background,
            capture_contribute,
            |context| {
                receiver_contribution(context, receiver_topology, preserve_capture_background)
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
        let capture_mips = a.and_then(|()| {
            for (index, generation) in plan.capture_mip_generations().iter().enumerate() {
                execution.commands(plan.capture_mip_generation_step(index)?, |encoder| {
                    plan.record_capture_mip_generation(
                        encoder,
                        &capture,
                        &receiver,
                        present_image,
                        generation,
                    )
                })?;
            }
            Ok(())
        });
        let b = capture_mips.and_then(|()| {
            execution.commands(plan.capture_raster_step(), |e| {
                plan.record_capture_raster(e, &capture, &receiver, present_image)
            })
        });
        let c = b.and_then(|()| {
            execution.commands(plan.receiver_upload_step(), |e| {
                plan.record_receiver_upload(e, &capture, &receiver, present_image)
            })
        });
        let receiver_mips = c.and_then(|()| {
            for (index, generation) in plan.receiver_mip_generations().iter().enumerate() {
                execution.commands(plan.receiver_mip_generation_step(index)?, |encoder| {
                    plan.record_receiver_mip_generation(
                        encoder,
                        &capture,
                        &receiver,
                        present_image,
                        generation,
                    )
                })?;
            }
            Ok(())
        });
        let background = receiver_mips.and_then(|()| {
            if let Some(step) = plan.receiver_background_copy_step() {
                execution.commands(step, |encoder| {
                    plan.record_receiver_background_copy(
                        encoder,
                        &capture,
                        &receiver,
                        present_image,
                    )
                })
            } else {
                Ok(())
            }
        });
        let d = background.and_then(|()| {
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
