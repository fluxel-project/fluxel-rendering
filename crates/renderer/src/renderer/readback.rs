use super::{Outcome, Pending, PendingWork, Renderer, RendererError, Submission};
use fluxel_rhi::{
    api::{Error, ExecutionStatus, StepEffect},
    backend::{
        ExecutionBackend, ExecutionOutcome, MappingBackend, ReadbackBackend, ResourceBackend,
    },
};
use std::sync::Arc;

impl<B> Pending<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    fn completion(&self) -> Result<fluxel_rhi::api::CompletionPoint<B>, Error> {
        match &self.workload {
            PendingWork::Deferred { plan, .. } => plan.completion(&self.outcome),
            PendingWork::DeferredSurface { plan, .. } => plan.surface_completion(&self.outcome),
            PendingWork::Forward { plan, .. } => plan.completion(&self.outcome),
            PendingWork::ForwardSurface { plan, .. } => plan.completion(&self.outcome),
            PendingWork::Copy { plan, .. } => plan.completion(&self.outcome),
            PendingWork::Raster { plan, .. } => plan.completion(&self.outcome),
            PendingWork::SurfaceRaster { plan, .. } => plan.completion(&self.outcome),
            PendingWork::Compute { plan, .. } => plan.completion(&self.outcome),
            PendingWork::ReadSetRaster { plan, .. } => plan.completion(&self.outcome),
        }
    }
    fn tight_readback(&self, bytes: Vec<u8>) -> Result<Vec<u8>, RendererError> {
        match &self.workload {
            PendingWork::Deferred {
                extent, row_pitch, ..
            } => super::strip_raster_padding(bytes, *extent, *row_pitch),
            PendingWork::Forward {
                extent, row_pitch, ..
            } => super::strip_raster_padding(bytes, *extent, *row_pitch),
            PendingWork::Copy { .. } | PendingWork::Compute { .. } => Ok(bytes),
            PendingWork::Raster {
                extent, row_pitch, ..
            } => super::strip_raster_padding(bytes, *extent, *row_pitch),
            PendingWork::ReadSetRaster {
                extent, row_pitch, ..
            } => super::strip_raster_padding(bytes, *extent, *row_pitch),
            PendingWork::SurfaceRaster { .. }
            | PendingWork::ForwardSurface { .. }
            | PendingWork::DeferredSurface { .. } => Err(RendererError::RetainedExecution {
                status: self.outcome.facts().status,
                recording_error: self.recording_error.clone(),
                execution_error: self.execution_error.clone(),
            }),
        }
    }
}
impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Waits for exact readback completion, then maps and returns copied bytes.
    pub async fn readback(&mut self, submission: &Submission) -> Result<Vec<u8>, RendererError> {
        if !Arc::ptr_eq(&self.owner, &submission.owner) || submission.device != self.device.id() {
            return Err(RendererError::ForeignSubmission);
        }
        self.ensure_device_available()?;
        let completion = {
            let pending = self.pending_for(submission)?;
            if pending.ticket.is_none() {
                return Err(RendererError::RetainedExecution {
                    status: pending.outcome.facts().status,
                    recording_error: pending.recording_error.clone(),
                    execution_error: pending.execution_error.clone(),
                });
            }
            match pending.completion() {
                Ok(v) => v,
                Err(_) if pending.outcome.facts().status == ExecutionStatus::Accepted => {
                    return Err(RendererError::MissingCompletion);
                }
                Err(_) => {
                    return Err(RendererError::RetainedExecution {
                        status: pending.outcome.facts().status,
                        recording_error: pending.recording_error.clone(),
                        execution_error: pending.execution_error.clone(),
                    });
                }
            }
        };
        if let Err(error) = self.device.wait_completion(&completion).await {
            self.note_device_loss(&error);
            return Err(error.into());
        }
        let mut pending = self.slots[submission.slot]
            .take()
            .filter(|v| v.serial == submission.serial)
            .ok_or(RendererError::StaleSubmission)?;
        let status = pending.outcome.facts().status;
        // Terminal GPU completion is established even if the subsequent CPU
        // read/map fails. Only completed products enter the reusable cache.
        self.publish_forward_completion(&pending.workload);
        self.publish_deferred_completion(&pending.workload);
        let ticket = match pending.ticket.take() {
            Some(v) => v,
            None => {
                let recording_error = pending.recording_error.clone();
                let execution_error = pending.execution_error.clone();
                self.slots[submission.slot] = Some(pending);
                return Err(RendererError::RetainedExecution {
                    status,
                    recording_error,
                    execution_error,
                });
            }
        };
        let data = match unsafe { pending.device.read_after(ticket, &completion).await } {
            Ok(data) => data,
            Err(error) => {
                self.note_device_loss(&error);
                return Err(error.into());
            }
        };
        let result = pending.tight_readback(data.bytes);
        if result.is_ok() || !matches!(&pending.workload, PendingWork::Deferred { .. }) {
            self.recycle_completed_readback(pending.workload, submission.slot);
        }
        result
    }

    /// Waits for the exact completion evidence of a submitted surface frame.
    ///
    /// Surface frames have no readback ticket: their raster buffers and native
    /// presentation outcome retire only after this method returns successfully.
    pub async fn complete(&mut self, submission: &Submission) -> Result<(), RendererError> {
        if !Arc::ptr_eq(&self.owner, &submission.owner) || submission.device != self.device.id() {
            return Err(RendererError::ForeignSubmission);
        }
        self.ensure_device_available()?;
        let completion = {
            let pending = self.pending_for(submission)?;
            if !matches!(
                pending.workload,
                PendingWork::SurfaceRaster { .. }
                    | PendingWork::ForwardSurface { .. }
                    | PendingWork::DeferredSurface { .. }
            ) {
                return Err(RendererError::StaleSubmission);
            }
            match pending.completion() {
                Ok(value) => value,
                Err(_) if pending.outcome.facts().status == ExecutionStatus::Accepted => {
                    return Err(RendererError::MissingCompletion);
                }
                Err(_) => {
                    return Err(RendererError::RetainedExecution {
                        status: pending.outcome.facts().status,
                        recording_error: pending.recording_error.clone(),
                        execution_error: pending.execution_error.clone(),
                    });
                }
            }
        };
        if let Err(error) = self.device.wait_completion(&completion).await {
            self.note_device_loss(&error);
            return Err(error.into());
        }
        let pending = self.slots[submission.slot]
            .take()
            .filter(|value| value.serial == submission.serial)
            .ok_or(RendererError::StaleSubmission)?;
        if !matches!(
            pending.workload,
            PendingWork::SurfaceRaster { .. }
                | PendingWork::ForwardSurface { .. }
                | PendingWork::DeferredSurface { .. }
        ) {
            self.slots[submission.slot] = Some(pending);
            return Err(RendererError::StaleSubmission);
        }
        self.publish_forward_completion(&pending.workload);
        self.publish_deferred_completion(&pending.workload);
        let slot = submission.slot;
        self.recycle_forward_surface(pending.workload, slot);
        Ok(())
    }
}
pub(super) fn releases_without_effect<B: ExecutionBackend>(outcome: &Outcome<B>) -> bool {
    let facts = outcome.facts();
    let accepted = matches!(facts.accepted,Some(v)if v.is_empty());
    let affected = matches!(facts.affected,Some(v)if v.is_empty());
    let no_present = facts.presents.is_none_or(|v| v.iter().all(|p| !p.accepted));
    let clean = match facts.status {
        ExecutionStatus::Rejected => facts.failed_step.is_none(),
        ExecutionStatus::Failed => matches!(facts.failed_step,Some(v)if v.effect==StepEffect::None),
        _ => false,
    };
    clean && accepted && affected && no_present
}
