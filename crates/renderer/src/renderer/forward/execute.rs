//! Validated Forward Graph selection, upload, recording, and submission.
use super::*;
use std::time::Instant;

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend + RasterBackend,
{
    #[allow(clippy::too_many_arguments)] // The acquired image and policy callback have independent ownership.
    pub(in crate::renderer) async fn execute_forward_surface<T, F>(
        &mut self,
        retained: T,
        frame: crate::residency::SurfaceForwardFrame<B>,
        publication: ForwardPublication,
        topology: crate::FrameTopology,
        batch_stats: crate::preparation::ForwardBatchStats,
        image: SurfaceImage<B>,
        contribute: F,
    ) -> std::result::Result<super::Submission, RendererError>
    where
        T: Send + Sync + 'static,
        F: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        if image.device() != self.device.id() || !image.is_current() {
            return Err(if image.device() != self.device.id() {
                Error::WrongDevice
            } else {
                Error::SurfaceOutdated
            }
            .into());
        }
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
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
        let declaration = crate::graph::SurfaceForwardPlan::declare_with(
            &self.device,
            &frame,
            topology,
            contribute,
        )
        .map_err(|error| self.forward_plan_error("surface graph declaration", error))?;
        let mut graph_compile_elapsed_ns = 0;
        let plan = if let Some(cached) = self.forward_surface_plans.iter().find(|candidate| {
            candidate.validate_declaration(&declaration).is_ok()
                && candidate.validate_frame(&frame).is_ok()
        }) {
            Arc::clone(cached)
        } else {
            let id = self.next_forward_plan_id()?;
            let compilation_started = Instant::now();
            let plan = Arc::new(
                crate::graph::SurfaceForwardPlan::compile_declared(&self.device, id, declaration)
                    .map_err(|error| self.forward_plan_error("surface graph compilation", error))?,
            );
            graph_compile_elapsed_ns = compilation_started.elapsed().as_nanos();
            self.forward_plan_compilations = self.forward_plan_compilations.saturating_add(1);
            if self.forward_surface_plans.len() == super::SLOT_COUNT + 1 {
                self.forward_surface_plans.remove(0);
            }
            self.forward_surface_plans.push(Arc::clone(&plan));
            plan
        };
        plan.validate_frame(&frame)
            .map_err(|error| self.observe_forward_error("surface frame validation", error))?;
        self.bindings_validated();
        self.write_forward_upload(&frame.resident.upload, &frame.resident_upload.bytes)
            .await?;
        self.write_forward_upload(&frame.frame.upload, &frame.uniform_bytes)
            .await?;
        let serial = self.next_serial()?;
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_forward_error("surface begin execution", error))?;
        let upload = execution.commands(plan.upload_step(), |encoder| {
            plan.record_upload(encoder, &frame)
        });
        let raster = match upload {
            Ok(()) => execution.commands(plan.raster_step(), |encoder| {
                plan.record_raster(encoder, &frame, &image)
            }),
            Err(error) => Err(error),
        };
        let recorded = match raster {
            Ok(()) => execution
                .present(plan.present_step(), PresentBinding::new(image))
                .map_err(|failure| failure.error),
            Err(error) => Err(error),
        };
        let outcome = unsafe { execution.finish() };
        let result = self.settle_submission(
            slot,
            serial,
            retained,
            None,
            (outcome, recorded),
            super::PendingWork::ForwardSurface {
                _frame: Box::new(frame),
                publication,
                plan,
            },
        );
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
                .saturating_add(graph_compile_elapsed_ns);
            self.forward_stats.pending_peak = self
                .forward_stats
                .pending_peak
                .max(u64::try_from(self.pending()).unwrap_or(u64::MAX));
        }
        result
    }

    pub(in crate::renderer) async fn execute_forward_with_contribution<T, F>(
        &mut self,
        retained: T,
        frame: crate::residency::ForwardFrame<B>,
        publication: ForwardPublication,
        topology: crate::FrameTopology,
        batch_stats: crate::preparation::ForwardBatchStats,
        contribute: F,
    ) -> std::result::Result<super::Submission, RendererError>
    where
        T: Send + Sync + 'static,
        F: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
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
        let declaration =
            crate::graph::ForwardPlan::declare_with(&self.device, &frame, topology, contribute)
                .map_err(|error| self.forward_plan_error("graph declaration", error))?;
        let mut graph_compile_elapsed_ns = 0;
        let plan = if let Some(cached) = self.forward_plans.iter().find(|candidate| {
            candidate.validate_declaration(&declaration).is_ok()
                && candidate.validate_frame(&frame).is_ok()
        }) {
            Arc::clone(cached)
        } else {
            let id = self.next_forward_plan_id()?;
            let compilation_started = Instant::now();
            let plan = Arc::new(
                crate::graph::ForwardPlan::compile_declared(&self.device, id, declaration)
                    .map_err(|error| self.forward_plan_error("graph compilation", error))?,
            );
            graph_compile_elapsed_ns = compilation_started.elapsed().as_nanos();
            self.forward_plan_compilations = self.forward_plan_compilations.saturating_add(1);
            // Bound the cache while pending submissions retain their own plans.
            if self.forward_plans.len() == super::SLOT_COUNT + 1 {
                self.forward_plans.remove(0);
            }
            self.forward_plans.push(Arc::clone(&plan));
            plan
        };
        plan.validate_frame(&frame)
            .map_err(|error| self.observe_forward_error("frame validation", error))?;
        self.bindings_validated();
        self.write_forward_upload(&frame.resident.upload, &frame.resident_upload.bytes)
            .await?;
        self.write_forward_upload(&frame.frame.upload, &frame.uniform_bytes)
            .await?;
        let serial = self.next_serial()?;
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_forward_error("begin execution", error))?;
        let upload = execution.commands(plan.upload_step(), |encoder| {
            plan.record_upload(encoder, &frame)
        });
        let mut failure_operation = "upload recording";
        let raster = match upload {
            Ok(()) => {
                failure_operation = "raster recording";
                execution.commands(plan.raster_step(), |encoder| {
                    plan.record_raster(encoder, &frame)
                })
            }
            Err(error) => Err(error),
        };
        let mut ticket = None;
        let recorded = match raster {
            Ok(()) => {
                failure_operation = "readback recording";
                execution.commands(plan.readback_step(), |encoder| {
                    ticket = Some(plan.record_readback(encoder, &frame)?);
                    Ok(())
                })
            }
            Err(error) => Err(error),
        };
        let outcome = unsafe { execution.finish() };
        if recorded.is_ok() {
            failure_operation = "execution finalization";
        }
        let result = self
            .settle_submission(
                slot,
                serial,
                retained,
                ticket,
                (outcome, recorded),
                super::PendingWork::Forward {
                    _frame: Box::new(frame),
                    publication,
                    plan,
                    extent,
                    row_pitch,
                },
            )
            .map_err(|error| match error {
                RendererError::Rhi(error) => self.observe_forward_error(failure_operation, error),
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
                .saturating_add(graph_compile_elapsed_ns);
            self.forward_stats.pending_peak = self
                .forward_stats
                .pending_peak
                .max(u64::try_from(self.pending()).unwrap_or(u64::MAX));
        }
        result
    }

    fn next_forward_plan_id(&self) -> std::result::Result<PlanId, RendererError> {
        let ordinal = crate::atomic::claim_next(&super::NEXT_PLAN_ORDINAL)
            .ok_or(RendererError::IdentityExhausted)?;
        Ok(PlanId {
            device: self.device.id(),
            ordinal,
        })
    }

    fn forward_plan_error(
        &mut self,
        operation: &'static str,
        error: crate::graph::ForwardPlanBuildError,
    ) -> RendererError {
        match error {
            crate::graph::ForwardPlanBuildError::Rhi(error) => {
                self.observe_forward_error(operation, error)
            }
            crate::graph::ForwardPlanBuildError::Pipeline(error) => {
                RendererError::ForwardPipeline(error)
            }
        }
    }

    async fn write_forward_upload(
        &mut self,
        buffer: &Buffer<B>,
        bytes: &[u8],
    ) -> std::result::Result<(), RendererError> {
        let len = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        // This is fresh unsubmitted staging owned by this preparation.
        let mut mapped = unsafe { self.device.map_buffer(buffer, 0..len, MapMode::Write).await }
            .map_err(|error| self.observe_forward_error("staging map", error))?;
        mapped
            .bytes_mut()
            .map_err(|error| self.observe_forward_error("mapped bytes", error))?
            .copy_from_slice(bytes);
        mapped
            .flush(0..len)
            .map_err(|error| self.observe_forward_error("staging flush", error))?;
        Ok(())
    }
}
