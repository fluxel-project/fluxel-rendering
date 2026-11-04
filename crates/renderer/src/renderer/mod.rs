//! Bounded ownership for the initial Copy, Raster, Compute, and Surface fixtures.
//!
//! This module is deliberately not a persistent asset cache, frame scheduler,
//! or general rendering API. It retains each submission until exact RHI
//! completion evidence permits retirement.

mod admission;
mod cache;
mod compute;
mod copy;
mod error;
mod forward;
mod raster;
mod read_set;
mod readback;
mod shutdown;
mod submission;
mod validation;

use std::{
    any::Any,
    collections::HashMap,
    mem,
    sync::{Arc, atomic::AtomicU64},
};

use fluxel_rhi::{
    api::{ComputePipeline, Device, Error, RasterPipeline, ReadbackTicket},
    backend::{
        ExecutionBackend, ExecutionMode, ExecutionOutcome, MappingBackend, ReadbackBackend,
        ResourceBackend,
    },
};

use crate::{
    graph::{
        ComputeFixtureBindings, ComputePlan, CopyFrame, CopyPlan, RasterPlan,
        ReadSetRasterBindings, ReadSetRasterFrame, ReadSetRasterPlan, SurfaceRasterPlan,
    },
    residency::{
        ComputeBuffers, RasterBuffers, ReadSetRasterBuffers, SurfaceRasterBuffers, UploadBuffers,
    },
};

pub use admission::AdmissionError as ShaderAdmissionError;
pub use error::RendererError;
pub use forward::ForwardCacheStats;
pub use shutdown::ShutdownReport;
pub use submission::Submission;

use readback::releases_without_effect;
use shutdown::Lifecycle;
#[cfg(test)]
use validation::validate_compute_requirements;
#[cfg(test)]
use validation::{compute_dispatch_groups, validate_compute_bytes, validate_raster_bytes};
use validation::{
    strip_raster_padding, validate_compute_input, validate_copy_requirements,
    validate_raster_input, validate_raster_requirements, validate_read_set_requirements,
};

const SLOT_COUNT: usize = 3;
pub(super) static NEXT_PLAN_ORDINAL: AtomicU64 = AtomicU64::new(1);

pub(super) type Outcome<B> = <<B as ExecutionBackend>::Mode as ExecutionMode<B>>::Outcome;

/// A three-slot owner for the initial fixture submissions.
///
/// Its plan cache contains immutable device execution topology. Copy frame
/// extents are revalidated separately; Raster and Compute still use fixed-shape
/// cache keys. It does not realize persistent logical assets.
pub struct Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    device: Device<B>,
    owner: Arc<()>,
    plans: HashMap<cache::PlanKey, cache::CachedPlan<B>>,
    forward_programs: HashMap<forward::ForwardProgramKey, Arc<forward::ForwardPipelineObjects<B>>>,
    forward_plans: Vec<Arc<crate::graph::ForwardPlan<B>>>,
    forward_surface_plans: Vec<Arc<crate::graph::SurfaceForwardPlan<B>>>,
    forward_plan_compilations: usize,
    forward_residency: crate::residency::ForwardResidentCache<B>,
    forward_stats: forward::ForwardCacheStats,
    forward_frames: [Option<crate::residency::ForwardFrameBuffers<B>>; SLOT_COUNT],
    slots: [Option<Pending<B>>; SLOT_COUNT],
    next_serial: u64,
    binding_refreshes: usize,
    lifecycle: Lifecycle,
}

pub(super) struct Pending<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    pub(super) serial: u64,
    pub(super) _snapshot: Box<dyn Any + Send + Sync>,
    pub(super) ticket: Option<ReadbackTicket<B>>,
    pub(super) outcome: Outcome<B>,
    pub(super) recording_error: Option<Error>,
    pub(super) execution_error: Option<Error>,
    pub(super) workload: PendingWork<B>,
    pub(super) device: Device<B>,
}

pub(super) enum PendingWork<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    Forward {
        _frame: Box<crate::residency::ForwardFrame<B>>,
        publication: forward::ForwardPublication,
        plan: Arc<crate::graph::ForwardPlan<B>>,
        extent: [u32; 2],
        row_pitch: u32,
    },
    ForwardSurface {
        _frame: Box<crate::residency::SurfaceForwardFrame<B>>,
        publication: forward::ForwardPublication,
        plan: Arc<crate::graph::SurfaceForwardPlan<B>>,
    },
    Copy {
        _buffers: UploadBuffers<B>,
        _frame: CopyFrame,
        plan: Arc<CopyPlan<B>>,
    },
    Raster {
        _buffers: Box<RasterBuffers<B>>,
        _pipeline: RasterPipeline<B>,
        plan: Arc<RasterPlan<B>>,
        extent: [u32; 2],
        row_pitch: u32,
    },
    SurfaceRaster {
        _buffers: SurfaceRasterBuffers<B>,
        _pipeline: RasterPipeline<B>,
        plan: Arc<SurfaceRasterPlan<B>>,
    },
    Compute {
        _buffers: ComputeBuffers<B>,
        _bindings: ComputeFixtureBindings<B>,
        _pipeline: ComputePipeline<B>,
        plan: Arc<ComputePlan<B>>,
    },
    ReadSetRaster {
        _frame: ReadSetRasterFrame,
        _buffers: Box<ReadSetRasterBuffers<B>>,
        _bindings: ReadSetRasterBindings<B>,
        _pipeline: RasterPipeline<B>,
        plan: Arc<ReadSetRasterPlan<B>>,
        extent: [u32; 2],
        row_pitch: u32,
    },
}

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Creates a renderer owner for an already-open RHI device.
    pub fn new(device: Device<B>) -> Result<Self, RendererError> {
        Ok(Self {
            device,
            owner: Arc::new(()),
            plans: HashMap::new(),
            forward_programs: HashMap::new(),
            forward_plans: Vec::new(),
            forward_surface_plans: Vec::new(),
            forward_plan_compilations: 0,
            forward_residency: Default::default(),
            forward_stats: Default::default(),
            forward_frames: std::array::from_fn(|_| None),
            slots: std::array::from_fn(|_| None),
            next_serial: 1,
            binding_refreshes: 0,
            lifecycle: Lifecycle::Active,
        })
    }

    /// Returns the number of submissions whose owners remain retained.
    pub fn pending(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// Returns the plan identity only when exactly one private topology was compiled.
    pub fn plan_id(&self) -> Option<fluxel_rhi::api::PlanId> {
        if self.plans.len() + self.forward_plan_compilations != 1 {
            None
        } else {
            self.plans
                .values()
                .next()
                .map(cache::CachedPlan::id)
                .or_else(|| self.forward_plans.first().map(|plan| plan.id()))
        }
    }

    /// Returns the number of cached fixture topologies.
    pub fn compilations(&self) -> usize {
        self.plans.len() + self.forward_plan_compilations
    }

    /// Returns the number of frame binding sets that passed resource preflight.
    pub fn binding_refreshes(&self) -> usize {
        self.binding_refreshes
    }

    /// Returns whether this renderer completed a successful shutdown.
    ///
    /// An unavailable device is not reported as shut down: its retained owners
    /// remain quarantined because a failed RHI wait is not retirement evidence.
    pub fn is_shutdown(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Closed)
    }

    pub(super) fn bindings_validated(&mut self) {
        self.binding_refreshes = self.binding_refreshes.saturating_add(1);
    }

    /// Rejects new fixture work after shutdown or a terminal device failure.
    pub(super) fn ensure_open(&self) -> Result<(), RendererError> {
        self.lifecycle.ensure_open()
    }

    pub(super) fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_none)
    }

    pub(super) fn next_serial(&mut self) -> Result<u64, RendererError> {
        let serial = self.next_serial;
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or(RendererError::IdentityExhausted)?;
        Ok(serial)
    }

    pub(super) fn pending_for(
        &self,
        submission: &Submission,
    ) -> Result<&Pending<B>, RendererError> {
        self.slots
            .get(submission.slot)
            .and_then(Option::as_ref)
            .filter(|pending| pending.serial == submission.serial)
            .ok_or(RendererError::StaleSubmission)
    }

    /// Converts an RHI error while making a terminal device failure sticky.
    pub(super) fn observe_rhi_error(&mut self, error: Error) -> RendererError {
        self.note_device_loss(&error);
        RendererError::Rhi(error)
    }

    pub(super) fn observe_renderer_error(&mut self, error: RendererError) -> RendererError {
        if let RendererError::Rhi(rhi_error) = &error {
            self.note_device_loss(rhi_error);
        }
        error
    }

    /// Records the ownership result after `finish` has supplied authoritative
    /// execution facts. All fixture paths use this to keep their result and
    /// retention semantics identical.
    pub(super) fn settle_submission<T: Send + Sync + 'static>(
        &mut self,
        slot: usize,
        serial: u64,
        snapshot: T,
        ticket: Option<ReadbackTicket<B>>,
        execution: (Outcome<B>, Result<(), Error>),
        workload: PendingWork<B>,
    ) -> Result<Submission, RendererError> {
        let (outcome, recorded) = execution;
        let recording_error = recorded.err();
        if let Some(error) = recording_error.as_ref() {
            self.note_device_loss(error);
        }
        let outcome_error = outcome.error().cloned();
        if let Some(error) = outcome_error.as_ref() {
            self.note_device_loss(error);
        }
        let status = outcome.facts().status;
        if releases_without_effect::<B>(&outcome) {
            return Err(self.observe_rhi_error(
                recording_error
                    .or(outcome_error)
                    .unwrap_or(Error::InvalidState),
            ));
        }

        // A loss reported by either recording or finish makes acceptance
        // indeterminate for this renderer instance. Preserve the original
        // outcome and all owners, but never hand the caller a success handle.
        let device_lost = matches!(recording_error.as_ref(), Some(Error::DeviceLost))
            || matches!(outcome_error.as_ref(), Some(Error::DeviceLost));
        if recording_error.is_some()
            || outcome_error.is_some()
            || status != fluxel_rhi::api::ExecutionStatus::Accepted
            || device_lost
        {
            self.slots[slot] = Some(Pending {
                serial,
                _snapshot: Box::new(snapshot),
                ticket,
                outcome,
                recording_error: recording_error.clone(),
                execution_error: outcome_error.clone(),
                workload,
                device: self.device.clone(),
            });
            return Err(RendererError::RetainedExecution {
                status,
                recording_error,
                execution_error: outcome_error,
            });
        }

        let submission = Submission {
            owner: Arc::clone(&self.owner),
            device: self.device.id(),
            slot,
            serial,
        };
        self.slots[slot] = Some(Pending {
            serial,
            _snapshot: Box::new(snapshot),
            ticket,
            outcome,
            recording_error: None,
            execution_error: None,
            workload,
            device: self.device.clone(),
        });
        Ok(submission)
    }
}

impl<B> Drop for Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    fn drop(&mut self) {
        for pending in self.slots.iter_mut().filter_map(Option::take) {
            if !releases_without_effect::<B>(&pending.outcome) {
                // A failed or absent shutdown supplies no retirement evidence.
                // Keep submitted or uncertain owners alive rather than freeing
                // native resources still referenced by GPU work.
                mem::forget(pending);
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/renderer/lifecycle.rs"]
mod lifecycle_tests;
#[cfg(test)]
#[path = "../tests/renderer/contracts.rs"]
mod tests;
