//! One submission retaining the Deferred opaque and Forward transparent owners.

use super::super::forward::OitExecutionTarget;
use super::*;
use crate::residency::{DeferredFrame, ForwardFrame};
use fluxel_rhi::api::PresentBinding;
use std::sync::Arc;

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend + RasterBackend,
{
    pub(super) async fn execute_deferred_transparent<F>(
        &mut self,
        retained: PendingInputs,
        opaque: DeferredFrame<B>,
        transparent: ForwardFrame<B>,
        publication: super::super::forward::ForwardPublication,
        target: OitExecutionTarget<B>,
        contribute: F,
    ) -> Result<Submission, RendererError>
    where
        F: FnOnce(&mut crate::DeferredPhaseContext<'_, '_>) -> Result<(), crate::PipelineError>,
    {
        self.ensure_open()?;
        let slot = self.free_slot().ok_or(RendererError::Backpressure)?;
        let (mut present, composite) = match target {
            OitExecutionTarget::Readback => (None, None),
            OitExecutionTarget::Surface { image, composite } => {
                (Some(PresentBinding::new(image)), Some(composite))
            }
        };
        let image = present.as_ref().map(PresentBinding::image);
        let mut contribution_error = None;
        let declare = |context: &mut crate::DeferredPhaseContext<'_, '_>| {
            contribute(context).inspect_err(|error| contribution_error = Some(error.clone()))
        };
        let declaration = match image {
            Some(image) => crate::graph::DeferredTransparentPlan::declare_surface(
                &self.device,
                &opaque,
                &transparent,
                image,
                declare,
            ),
            None => crate::graph::DeferredTransparentPlan::declare(
                &self.device,
                &opaque,
                &transparent,
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
            crate::graph::DeferredTransparentPlan::compile_declared(
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
        match (image, composite.as_ref()) {
            (Some(image), Some(composite)) => plan.validate_frames_with_image(
                &self.device,
                &opaque,
                &transparent,
                image,
                composite,
            ),
            (None, None) => plan.validate_frames(&opaque, &transparent),
            _ => Err(Error::InvalidState),
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
                &transparent.resident.upload,
                transparent.resident_upload.bytes.as_slice(),
            ),
            (
                &transparent.frame.upload,
                transparent.uniform_bytes.as_slice(),
            ),
        ] {
            if !bytes.is_empty() {
                self.write_deferred_upload(upload, bytes).await?;
            }
        }
        for owner in [
            opaque.environment.as_ref(),
            transparent.environment.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                .await?;
        }
        for owner in [opaque.specular.as_ref(), transparent.specular.as_ref()]
            .into_iter()
            .flatten()
        {
            self.write_deferred_upload(&owner.upload, &owner.upload_bytes)
                .await?;
        }
        let present_step = image.map(|_| plan.present_step()).transpose()?;
        let serial = self.next_serial()?;
        // SAFETY: all native owners satisfy this one compiled Graph. Both frame
        // namespaces remain alive through finish and completion settlement.
        let mut execution = unsafe {
            self.device.begin_execution(
                plan.submission().clone(),
                fluxel_rhi::api::CompletionRequest::End,
            )
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        let mut recorded = Ok(());
        let mut ticket = None;
        for (step, pass) in plan.steps() {
            if recorded.is_ok() {
                recorded = execution.commands(step, |encoder| match (image, composite.as_ref()) {
                    (Some(image), Some(composite)) => plan.record_surface_pass(
                        &self.device,
                        encoder,
                        &opaque,
                        &transparent,
                        image,
                        composite,
                        pass,
                    ),
                    (None, None) => {
                        if let Some(readback) =
                            plan.record_pass(encoder, &opaque, &transparent, pass)?
                        {
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
        // SAFETY: partial recording is finalized before any owner can be
        // released. Failed completion waits retain the complete pending value.
        let outcome = unsafe { execution.finish() };
        let extent = opaque.extent;
        let row_pitch = opaque.readback_row_pitch;
        let workload = match composite {
            Some(composite) => super::super::PendingWork::DeferredTransparentSurface {
                _opaque: Box::new(opaque),
                _transparent: Box::new(transparent),
                publication,
                plan,
                _composite: composite,
                _present: present,
            },
            None => super::super::PendingWork::DeferredTransparent {
                _opaque: Box::new(opaque),
                _transparent: Box::new(transparent),
                publication,
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
