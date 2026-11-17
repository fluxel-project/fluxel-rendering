//! One execution owning Deferred lighting, peeling and the final output.

use super::super::forward::{ForwardPublication, OitFrameResources};
use super::*;
use crate::{
    graph::ForwardOitInput,
    residency::{DeferredFrame, ForwardFrame},
};
use fluxel_rhi::api::{CompletionRequest, PresentBinding};
use std::{num::NonZeroU32, sync::Arc};

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend + RasterBackend,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_deferred_oit<F>(
        &mut self,
        retained: PendingInputs,
        opaque: DeferredFrame<B>,
        overlay: ForwardFrame<B>,
        oit: OitFrameResources<B>,
        publication: ForwardPublication,
        composite: super::super::SurfaceCompositeOwner<B>,
        peel_count: usize,
        rounds: NonZeroU32,
        surface: Option<SurfaceImage<B>>,
        contribute: F,
    ) -> Result<Submission, RendererError>
    where
        F: FnOnce(&mut crate::DeferredPhaseContext<'_, '_>) -> Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let input = ForwardOitInput {
            owners: &oit.owners,
            output: &oit.output,
            opaque_count: 0,
            peel_count,
            rounds,
        };
        let mut present = surface.map(PresentBinding::new);
        let image = present.as_ref().map(PresentBinding::image);
        let mut contribution_error = None;
        let declare = |context: &mut crate::DeferredPhaseContext<'_, '_>| {
            contribute(context).inspect_err(|error| contribution_error = Some(error.clone()))
        };
        let declaration = match image {
            Some(image) => crate::graph::DeferredOitPlan::declare_surface(
                &self.device,
                &opaque,
                &overlay,
                &input,
                image,
                declare,
            ),
            None => crate::graph::DeferredOitPlan::declare(
                &self.device,
                &opaque,
                &overlay,
                &input,
                declare,
            ),
        }
        .map_err(|error| match contribution_error {
            Some(error) => RendererError::DeferredPipeline(error),
            None => self.observe_rhi_error(error),
        })?;
        let ordinal = crate::atomic::claim_next(&super::super::NEXT_PLAN_ORDINAL)
            .ok_or(RendererError::IdentityExhausted)?;
        let plan = Arc::new(
            crate::graph::DeferredOitPlan::compile_declared(
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
        match image {
            Some(image) => plan.validate_frames_with_image(
                &self.device,
                &opaque,
                &overlay,
                &input,
                image,
                &composite,
            ),
            None => plan.validate_frames(&self.device, &opaque, &overlay, &input, &composite),
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        self.bindings_validated();
        for (upload, bytes) in [
            (
                &opaque.resident.upload,
                opaque.resident_upload.bytes.as_slice(),
            ),
            (&opaque.frame.upload, opaque.uniform_bytes.as_slice()),
            (
                &overlay.resident.upload,
                overlay.resident_upload.bytes.as_slice(),
            ),
            (&overlay.frame.upload, overlay.uniform_bytes.as_slice()),
        ] {
            if !bytes.is_empty() {
                self.write_deferred_upload(upload, bytes).await?;
            }
        }
        for owner in [opaque.environment.as_ref(), overlay.environment.as_ref()]
            .into_iter()
            .flatten()
        {
            self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                .await?;
        }
        for owner in [opaque.specular.as_ref(), overlay.specular.as_ref()]
            .into_iter()
            .flatten()
        {
            self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                .await?;
        }
        let draws = oit.draw_commands();
        let commands = oit.commands_for_opaque(&overlay, &draws, &opaque.scene_color);
        let present_step = image.map(|_| plan.present_step()).transpose()?;
        let serial = self.next_serial()?;
        // SAFETY: all owners were checked against this one compiled Graph and
        // remain retained until the matching completion permits retirement.
        let mut execution = unsafe {
            self.device
                .begin_execution(plan.submission().clone(), CompletionRequest::End)
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        let mut recorded = Ok(());
        let mut ticket = None;
        for (step, pass) in plan.steps() {
            if recorded.is_ok() {
                recorded = execution.commands(step, |encoder| match image {
                    Some(image) => plan.record_surface_pass(
                        &self.device,
                        encoder,
                        &opaque,
                        &overlay,
                        &input,
                        &commands,
                        image,
                        &composite,
                        pass,
                    ),
                    None => {
                        if let Some(readback) = plan.record_pass(
                            &self.device,
                            encoder,
                            &opaque,
                            &overlay,
                            &input,
                            &commands,
                            &composite,
                            pass,
                        )? {
                            if ticket.is_some() {
                                return Err(Error::InvalidState);
                            }
                            ticket = Some(readback);
                        }
                        Ok(())
                    }
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
        // SAFETY: every attempted command/present belongs to this plan. Finish
        // is unconditional even when recording or the present handoff failed.
        let outcome = unsafe { execution.finish() };
        let extent = opaque.extent;
        let row_pitch = opaque.readback_row_pitch;
        let workload = if present_step.is_some() {
            super::super::PendingWork::DeferredOitSurface {
                _opaque: Box::new(opaque),
                _transparent: Box::new(overlay),
                _oit: Box::new(oit),
                _composite: composite,
                _present: present,
                publication,
                plan,
            }
        } else {
            super::super::PendingWork::DeferredOit {
                _opaque: Box::new(opaque),
                _transparent: Box::new(overlay),
                _oit: Box::new(oit),
                _composite: composite,
                publication,
                plan,
                extent,
                row_pitch,
            }
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
