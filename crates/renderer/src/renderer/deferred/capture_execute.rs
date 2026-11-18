//! One submission retaining both Deferred capture namespaces through completion.

use super::*;
use crate::residency::DeferredFrame;
use fluxel_rhi::api::{CompletionRequest, PresentBinding};
use std::sync::Arc;

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
    pub(super) async fn execute_deferred_capture<F, C>(
        &mut self,
        retained: PendingInputs,
        producer: DeferredFrame<B>,
        composition: Option<super::DeferredCaptureComposition<B>>,
        receiver: DeferredFrame<B>,
        surface: Option<SurfaceImage<B>>,
        contribute: F,
        overlay_contribute: C,
    ) -> Result<Submission, RendererError>
    where
        F: FnOnce(&mut crate::DeferredPhaseContext<'_, '_>) -> Result<(), crate::PipelineError>,
        C: FnMut(
            &mut crate::graph::ForwardOitReadContext<'_, '_>,
        ) -> Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let composite = match surface.as_ref() {
            Some(image) => Some(
                self.deferred_surface_composite_owner(
                    &receiver.scene_color,
                    receiver.extent,
                    image,
                )
                .await?,
            ),
            None => None,
        };
        // Keep the lease outside command closures. A recording failure must
        // not drop an image already referenced by earlier submitted scopes.
        let mut present = surface.map(PresentBinding::new);
        let image = present.as_ref().map(PresentBinding::image);
        let mut contribution_error = None;
        let mut overlay_error = None;
        let overlay = composition
            .as_ref()
            .map(|composition| composition.graph_input());
        let declaration = crate::graph::DeferredCapturePlan::declare(
            &self.device,
            &producer,
            overlay.as_ref(),
            &receiver,
            image,
            |context| {
                contribute(context).inspect_err(|error| {
                    contribution_error = Some(error.clone());
                })
            },
            overlay_contribute,
            &mut overlay_error,
        )
        .map_err(|error| match contribution_error.or(overlay_error) {
            Some(error) => RendererError::DeferredPipeline(error),
            None => self.observe_rhi_error(error),
        })?;
        let ordinal = crate::atomic::claim_next(&super::super::NEXT_PLAN_ORDINAL)
            .ok_or(RendererError::IdentityExhausted)?;
        let plan = Arc::new(
            crate::graph::DeferredCapturePlan::compile_declared(
                &self.device,
                PlanId {
                    device: self.device.id(),
                    ordinal,
                },
                declaration,
            )
            .map_err(|error| self.observe_rhi_error(error))?,
        );
        self.deferred_plan_compilations = self.deferred_plan_compilations.saturating_add(1);
        plan.validate_frames(
            &self.device,
            &producer,
            overlay.as_ref(),
            &receiver,
            image,
            composite.as_ref(),
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        self.bindings_validated();
        for frame in [&producer, &receiver] {
            for (upload, bytes) in [
                (
                    &frame.resident.upload,
                    frame.resident_upload.bytes.as_slice(),
                ),
                (&frame.frame.upload, frame.uniform_bytes.as_slice()),
            ] {
                if !bytes.is_empty() {
                    self.write_deferred_upload(upload, bytes).await?;
                }
            }
            if let Some(owner) = frame.environment.as_ref() {
                self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                    .await?;
            }
            if let Some(owner) = frame.specular.as_ref() {
                self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                    .await?;
            }
        }
        if let Some(composition) = composition.as_ref() {
            let frame = &composition.frame;
            for (upload, bytes) in [
                (
                    &frame.resident.upload,
                    frame.resident_upload.bytes.as_slice(),
                ),
                (&frame.frame.upload, frame.uniform_bytes.as_slice()),
            ] {
                if !bytes.is_empty() {
                    self.write_deferred_upload(upload, bytes).await?;
                }
            }
            if let Some(owner) = frame.environment.as_ref() {
                self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                    .await?;
            }
            if let Some(owner) = frame.specular.as_ref() {
                self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                    .await?;
            }
        }
        let draw_programs = composition
            .as_ref()
            .and_then(|composition| composition.oit.as_ref())
            .map(|oit| oit.draw_commands());
        let oit_commands = match (composition.as_ref(), draw_programs.as_deref()) {
            (Some(composition), Some(draws)) => composition.oit.as_ref().map(|oit| {
                oit.commands_for_opaque(&composition.frame, draws, &producer.scene_color)
            }),
            _ => None,
        };
        let present_step = image.map(|_| plan.present_step()).transpose()?;
        let serial = self.next_serial()?;
        // SAFETY: the one combined Graph has checked both actual namespaces,
        // and every owner is retained below regardless of recording success.
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        let mut recorded = Ok(());
        let mut ticket = None;
        for (step, pass) in plan.steps() {
            if recorded.is_ok() {
                recorded = execution.commands(step, |encoder| {
                    if let Some(readback) = plan.record_pass(
                        &self.device,
                        encoder,
                        &producer,
                        overlay.as_ref(),
                        oit_commands.as_ref(),
                        &receiver,
                        image,
                        composite.as_ref(),
                        pass,
                    )? {
                        if image.is_some() || ticket.is_some() {
                            return Err(Error::InvalidState);
                        }
                        ticket = Some(readback);
                    }
                    Ok(())
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
        // SAFETY: finish is unconditional after begin, including partial
        // recording and rejected presentation; settlement consumes its facts.
        let outcome = unsafe { execution.finish() };
        let extent = receiver.extent;
        let row_pitch = receiver.readback_row_pitch;
        let workload = match composite {
            Some(composite) => super::super::PendingWork::DeferredCaptureSurface {
                _producer: Box::new(producer),
                _composition: composition.map(Box::new),
                _receiver: Box::new(receiver),
                _composite: composite,
                _present: present,
                plan,
            },
            None => super::super::PendingWork::DeferredCapture {
                _producer: Box::new(producer),
                _composition: composition.map(Box::new),
                _receiver: Box::new(receiver),
                plan,
                extent,
                row_pitch,
            },
        };
        self.settle_submission(
            slot,
            serial,
            retained,
            ticket,
            (outcome, recorded),
            workload,
        )
    }
}
