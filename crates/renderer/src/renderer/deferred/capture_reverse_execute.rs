//! One submission for a transparent-only producer and Deferred capture receiver.

use super::*;
use crate::{renderer::forward::CaptureOitSide, residency::DeferredFrame};
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
    pub(super) async fn execute_forward_deferred_capture(
        &mut self,
        retained: PendingInputs,
        producer: CaptureOitSide<B>,
        receiver: DeferredFrame<B>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError> {
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
        let mut present = surface.map(PresentBinding::new);
        let image = present.as_ref().map(PresentBinding::image);
        let input = producer.graph_input();
        let declaration = crate::graph::ForwardDeferredCapturePlan::declare(
            &self.device,
            &producer.frame,
            input.as_ref(),
            &receiver,
            image,
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        let ordinal = crate::atomic::claim_next(&super::super::NEXT_PLAN_ORDINAL)
            .ok_or(RendererError::IdentityExhausted)?;
        let plan = Arc::new(
            crate::graph::ForwardDeferredCapturePlan::compile_declared(
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
            &producer.frame,
            input.as_ref(),
            &receiver,
            image,
            composite.as_ref(),
        )
        .map_err(|error| self.observe_rhi_error(error))?;
        self.bindings_validated();
        for (upload, bytes) in [
            (
                &producer.frame.resident.upload,
                producer.frame.resident_upload.bytes.as_slice(),
            ),
            (
                &producer.frame.frame.upload,
                producer.frame.uniform_bytes.as_slice(),
            ),
            (
                &receiver.resident.upload,
                receiver.resident_upload.bytes.as_slice(),
            ),
            (&receiver.frame.upload, receiver.uniform_bytes.as_slice()),
        ] {
            if !bytes.is_empty() {
                self.write_deferred_upload(upload, bytes).await?;
            }
        }
        for owner in [
            producer.frame.environment.as_ref(),
            receiver.environment.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                .await?;
        }
        for owner in [producer.frame.specular.as_ref(), receiver.specular.as_ref()]
            .into_iter()
            .flatten()
        {
            self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                .await?;
        }
        let draws = producer.oit.as_ref().map(|oit| oit.draw_commands());
        let commands = match (producer.oit.as_ref(), draws.as_deref()) {
            (Some(oit), Some(draws)) => Some(oit.commands(&producer.frame, draws)),
            _ => None,
        };
        let present_step = image.map(|_| plan.present_step()).transpose()?;
        let serial = self.next_serial()?;
        // SAFETY: all native owners satisfy this one compiled Graph and are
        // retained below through finish and its covering completion.
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        let mut ticket = None;
        let mut recorded = Ok(());
        for (step, pass) in plan.steps() {
            if recorded.is_ok() {
                recorded = execution.commands(step, |encoder| {
                    if let Some(readback) = plan.record_pass(
                        &self.device,
                        encoder,
                        &producer.frame,
                        input.as_ref(),
                        commands.as_ref(),
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
        // SAFETY: finish is unconditional once execution begins, including
        // partial command recording or rejected presentation handoff.
        let outcome = unsafe { execution.finish() };
        let extent = receiver.extent;
        let row_pitch = receiver.readback_row_pitch;
        let workload = match composite {
            Some(composite) => super::super::PendingWork::ForwardDeferredCaptureSurface {
                _producer: Box::new(producer),
                _receiver: Box::new(receiver),
                _composite: composite,
                _present: present,
                plan,
            },
            None => super::super::PendingWork::ForwardDeferredCapture {
                _producer: Box::new(producer),
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
