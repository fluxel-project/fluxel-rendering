//! Four-pass Deferred recording and shared completion-held pending ownership.

use super::*;
use crate::{
    graph::DeferredPlan,
    residency::{DeferredFrame, DeferredSurfaceFrame},
};
use fluxel_rhi::api::{Buffer, CompletionRequest, MapMode, PresentBinding, SurfaceImage};
use std::sync::{Arc, atomic::Ordering};

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend + RasterBackend,
{
    pub(super) async fn execute_deferred_probe<T: Send + Sync + 'static, F>(
        &mut self,
        retained: T,
        frame: DeferredFrame<B>,
        declare: F,
    ) -> Result<Submission, RendererError>
    where
        F: FnOnce(&mut crate::DeferredPhaseContext<'_, '_>) -> Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let mut contribution_error = None;
        // Contributions always author this frame's real Graph, including on a
        // cache hit. Reuse compares that declaration rather than replaying an
        // old callback or trusting preparation-side metadata.
        let declaration = DeferredPlan::declare_with(&self.device, &frame, |context| {
            declare(context).inspect_err(|error| contribution_error = Some(error.clone()))
        });
        let declaration = declaration.map_err(|error| match contribution_error {
            Some(error) => RendererError::DeferredPipeline(error),
            None => self.observe_rhi_error(error),
        })?;
        let plan = if let Some(cached) = self.deferred_plans.iter().find(|candidate| {
            candidate.validate_declaration(&declaration).is_ok()
                && candidate.validate_frame(&frame).is_ok()
        }) {
            self.deferred_stats.graph_hits = self.deferred_stats.graph_hits.saturating_add(1);
            Arc::clone(cached)
        } else {
            let ordinal = super::super::NEXT_PLAN_ORDINAL
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| RendererError::IdentityExhausted)?;
            let id = PlanId {
                device: self.device.id(),
                ordinal,
            };
            let plan = Arc::new(
                DeferredPlan::compile_declared(&self.device, id, declaration)
                    .map_err(|error| self.observe_rhi_error(error))?,
            );
            self.deferred_plan_compilations = self.deferred_plan_compilations.saturating_add(1);
            if self.deferred_plans.len() == super::super::SLOT_COUNT {
                self.deferred_plans.remove(0);
            }
            self.deferred_plans.push(Arc::clone(&plan));
            plan
        };
        plan.validate_frame(&frame)
            .map_err(|error| self.observe_rhi_error(error))?;
        self.bindings_validated();
        let (buffer_bindings, texture_bindings) = plan.binding_counts();
        let facts = DeferredProbeFacts {
            plan: plan.id(),
            passes: plan.pass_count(),
            buffer_bindings,
            texture_bindings,
            gbuffer_formats: std::array::from_fn(|index| frame.gbuffer[index].desc.format),
            gbuffer_extents: std::array::from_fn(|index| frame.gbuffer[index].desc.extent),
        };
        if !frame.resident_upload.bytes.is_empty() {
            self.write_deferred_upload(&frame.resident.upload, &frame.resident_upload.bytes)
                .await?;
        }
        self.write_deferred_upload(&frame.frame.upload, &frame.uniform_bytes)
            .await?;
        if let Some(producer) = &frame.deformation {
            self.write_deferred_upload(&producer.upload.buffer, &producer.upload_bytes)
                .await?;
        }
        let serial = self.next_serial()?;
        // SAFETY: all bindings were checked against the single compiled Graph;
        // frame owners remain held below through recording and final settlement.
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        let mut recorded = execution.commands(plan.upload_step(), |encoder| {
            plan.record_upload(encoder, &frame)
        });
        if recorded.is_ok() {
            if let Some(step) = plan.deformation_step() {
                recorded =
                    execution.commands(step, |encoder| plan.record_deformation(encoder, &frame));
            }
        }
        if recorded.is_ok() {
            recorded = execution.commands(plan.geometry_step(), |encoder| {
                plan.record_geometry(encoder, &frame)
            });
        }
        if recorded.is_ok() {
            recorded = execution.commands(plan.lighting_step(), |encoder| {
                plan.record_lighting(encoder, &frame)
            });
        }
        let mut ticket = None;
        if recorded.is_ok() {
            recorded = execution.commands(plan.readback_step(), |encoder| {
                ticket = Some(plan.record_readback(encoder, &frame)?);
                Ok(())
            });
        }
        // SAFETY: every attempted step is from this Graph plan. Partial effects
        // are settled by the shared owner path; no submitted resources drop early.
        let outcome = unsafe { execution.finish() };
        let extent = frame.extent;
        let row_pitch = frame.readback_row_pitch;
        let submission = self.settle_submission(
            slot,
            serial,
            retained,
            ticket,
            (outcome, recorded),
            super::super::PendingWork::Deferred {
                _frame: Box::new(frame),
                plan,
                extent,
                row_pitch,
            },
        )?;
        self.deferred_probe_facts = Some(facts);
        Ok(submission)
    }

    async fn write_deferred_upload(
        &mut self,
        buffer: &Buffer<B>,
        bytes: &[u8],
    ) -> Result<(), RendererError> {
        let len = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        // SAFETY: fresh staging has not been recorded or submitted; Graph
        // validation has established the complete source footprint.
        let mut mapped = unsafe { self.device.map_buffer(buffer, 0..len, MapMode::Write).await }
            .map_err(|error| self.observe_rhi_error(error))?;
        mapped
            .bytes_mut()
            .map_err(|error| self.observe_rhi_error(error))?
            .copy_from_slice(bytes);
        mapped
            .flush(0..len)
            .map_err(|error| self.observe_rhi_error(error))?;
        Ok(())
    }

    pub(super) async fn execute_deferred_surface<T: Send + Sync + 'static, F>(
        &mut self,
        retained: T,
        frame: DeferredSurfaceFrame<B>,
        image: SurfaceImage<B>,
        declare: F,
    ) -> Result<Submission, RendererError>
    where
        F: FnOnce(&mut crate::DeferredPhaseContext<'_, '_>) -> Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        if image.extent() != frame.extent || image.format() != frame.color_format {
            return Err(Error::InvalidRange.into());
        }
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let mut contribution_error = None;
        let declaration = DeferredPlan::declare_surface_with(&self.device, &frame, |context| {
            declare(context).inspect_err(|error| contribution_error = Some(error.clone()))
        });
        let declaration = declaration.map_err(|error| match contribution_error {
            Some(error) => RendererError::DeferredPipeline(error),
            None => self.observe_rhi_error(error),
        })?;
        let plan = if let Some(cached) = self.deferred_surface_plans.iter().find(|candidate| {
            candidate.validate_declaration(&declaration).is_ok()
                && candidate.validate_surface_frame(&frame).is_ok()
        }) {
            self.deferred_stats.graph_hits = self.deferred_stats.graph_hits.saturating_add(1);
            Arc::clone(cached)
        } else {
            let ordinal = super::super::NEXT_PLAN_ORDINAL
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| RendererError::IdentityExhausted)?;
            let id = PlanId {
                device: self.device.id(),
                ordinal,
            };
            let plan = Arc::new(
                DeferredPlan::compile_declared(&self.device, id, declaration)
                    .map_err(|error| self.observe_rhi_error(error))?,
            );
            self.deferred_plan_compilations = self.deferred_plan_compilations.saturating_add(1);
            if self.deferred_surface_plans.len() == super::super::SLOT_COUNT {
                self.deferred_surface_plans.remove(0);
            }
            self.deferred_surface_plans.push(Arc::clone(&plan));
            plan
        };
        plan.validate_surface_frame(&frame)
            .map_err(|error| self.observe_rhi_error(error))?;
        self.bindings_validated();
        let (buffer_bindings, texture_bindings) = plan.binding_counts();
        let facts = DeferredProbeFacts {
            plan: plan.id(),
            passes: plan.pass_count(),
            buffer_bindings,
            texture_bindings,
            gbuffer_formats: std::array::from_fn(|index| frame.gbuffer[index].desc.format),
            gbuffer_extents: std::array::from_fn(|index| frame.gbuffer[index].desc.extent),
        };
        if !frame.resident_upload.bytes.is_empty() {
            self.write_deferred_upload(&frame.resident.upload, &frame.resident_upload.bytes)
                .await?;
        }
        self.write_deferred_upload(&frame.frame.upload, &frame.uniform_bytes)
            .await?;
        if let Some(producer) = &frame.deformation {
            self.write_deferred_upload(&producer.upload.buffer, &producer.upload_bytes)
                .await?;
        }
        let serial = self.next_serial()?;
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        let mut recorded = execution.commands(plan.upload_step(), |encoder| {
            plan.surface_record_upload(encoder, &frame)
        });
        if recorded.is_ok() {
            if let Some(step) = plan.deformation_step() {
                recorded = execution.commands(step, |encoder| {
                    plan.surface_record_deformation(encoder, &frame)
                });
            }
        }
        if recorded.is_ok() {
            recorded = execution.commands(plan.geometry_step(), |encoder| {
                plan.surface_record_geometry(encoder, &frame)
            });
        }
        if recorded.is_ok() {
            recorded = execution.commands(plan.lighting_step(), |encoder| {
                plan.surface_record_lighting(encoder, &frame, &image)
            });
        }
        if recorded.is_ok() {
            recorded = execution
                .present(plan.present_step()?, PresentBinding::new(image))
                .map_err(|failure| failure.error);
        }
        let outcome = unsafe { execution.finish() };
        let submission = self.settle_submission(
            slot,
            serial,
            retained,
            None,
            (outcome, recorded),
            super::super::PendingWork::DeferredSurface {
                _frame: Box::new(frame),
                plan,
            },
        )?;
        self.deferred_probe_facts = Some(facts);
        Ok(submission)
    }
}
